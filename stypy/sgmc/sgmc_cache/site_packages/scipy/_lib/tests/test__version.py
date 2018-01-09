
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from numpy.testing import assert_
4: from pytest import raises as assert_raises
5: from scipy._lib._version import NumpyVersion
6: 
7: 
8: def test_main_versions():
9:     assert_(NumpyVersion('1.8.0') == '1.8.0')
10:     for ver in ['1.9.0', '2.0.0', '1.8.1']:
11:         assert_(NumpyVersion('1.8.0') < ver)
12: 
13:     for ver in ['1.7.0', '1.7.1', '0.9.9']:
14:         assert_(NumpyVersion('1.8.0') > ver)
15: 
16: 
17: def test_version_1_point_10():
18:     # regression test for gh-2998.
19:     assert_(NumpyVersion('1.9.0') < '1.10.0')
20:     assert_(NumpyVersion('1.11.0') < '1.11.1')
21:     assert_(NumpyVersion('1.11.0') == '1.11.0')
22:     assert_(NumpyVersion('1.99.11') < '1.99.12')
23: 
24: 
25: def test_alpha_beta_rc():
26:     assert_(NumpyVersion('1.8.0rc1') == '1.8.0rc1')
27:     for ver in ['1.8.0', '1.8.0rc2']:
28:         assert_(NumpyVersion('1.8.0rc1') < ver)
29: 
30:     for ver in ['1.8.0a2', '1.8.0b3', '1.7.2rc4']:
31:         assert_(NumpyVersion('1.8.0rc1') > ver)
32: 
33:     assert_(NumpyVersion('1.8.0b1') > '1.8.0a2')
34: 
35: 
36: def test_dev_version():
37:     assert_(NumpyVersion('1.9.0.dev-Unknown') < '1.9.0')
38:     for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev-ffffffff']:
39:         assert_(NumpyVersion('1.9.0.dev-f16acvda') < ver)
40: 
41:     assert_(NumpyVersion('1.9.0.dev-f16acvda') == '1.9.0.dev-11111111')
42: 
43: 
44: def test_dev_a_b_rc_mixed():
45:     assert_(NumpyVersion('1.9.0a2.dev-f16acvda') == '1.9.0a2.dev-11111111')
46:     assert_(NumpyVersion('1.9.0a2.dev-6acvda54') < '1.9.0a2')
47: 
48: 
49: def test_dev0_version():
50:     assert_(NumpyVersion('1.9.0.dev0+Unknown') < '1.9.0')
51:     for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev0+ffffffff']:
52:         assert_(NumpyVersion('1.9.0.dev0+f16acvda') < ver)
53: 
54:     assert_(NumpyVersion('1.9.0.dev0+f16acvda') == '1.9.0.dev0+11111111')
55: 
56: 
57: def test_dev0_a_b_rc_mixed():
58:     assert_(NumpyVersion('1.9.0a2.dev0+f16acvda') == '1.9.0a2.dev0+11111111')
59:     assert_(NumpyVersion('1.9.0a2.dev0+6acvda54') < '1.9.0a2')
60: 
61: 
62: def test_raises():
63:     for ver in ['1.9', '1,9.0', '1.7.x']:
64:         assert_raises(ValueError, NumpyVersion, ver)
65: 
66: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.testing import assert_' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712830 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_712830) is not StypyTypeError):

    if (import_712830 != 'pyd_module'):
        __import__(import_712830)
        sys_modules_712831 = sys.modules[import_712830]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', sys_modules_712831.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_712831, sys_modules_712831.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_712830)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from pytest import assert_raises' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712832 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest')

if (type(import_712832) is not StypyTypeError):

    if (import_712832 != 'pyd_module'):
        __import__(import_712832)
        sys_modules_712833 = sys.modules[import_712832]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', sys_modules_712833.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_712833, sys_modules_712833.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', import_712832)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy._lib._version import NumpyVersion' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712834 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._version')

if (type(import_712834) is not StypyTypeError):

    if (import_712834 != 'pyd_module'):
        __import__(import_712834)
        sys_modules_712835 = sys.modules[import_712834]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._version', sys_modules_712835.module_type_store, module_type_store, ['NumpyVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_712835, sys_modules_712835.module_type_store, module_type_store)
    else:
        from scipy._lib._version import NumpyVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._version', None, module_type_store, ['NumpyVersion'], [NumpyVersion])

else:
    # Assigning a type to the variable 'scipy._lib._version' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._version', import_712834)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')


@norecursion
def test_main_versions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_main_versions'
    module_type_store = module_type_store.open_function_context('test_main_versions', 8, 0, False)
    
    # Passed parameters checking function
    test_main_versions.stypy_localization = localization
    test_main_versions.stypy_type_of_self = None
    test_main_versions.stypy_type_store = module_type_store
    test_main_versions.stypy_function_name = 'test_main_versions'
    test_main_versions.stypy_param_names_list = []
    test_main_versions.stypy_varargs_param_name = None
    test_main_versions.stypy_kwargs_param_name = None
    test_main_versions.stypy_call_defaults = defaults
    test_main_versions.stypy_call_varargs = varargs
    test_main_versions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_main_versions', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_main_versions', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_main_versions(...)' code ##################

    
    # Call to assert_(...): (line 9)
    # Processing the call arguments (line 9)
    
    
    # Call to NumpyVersion(...): (line 9)
    # Processing the call arguments (line 9)
    str_712838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 25), 'str', '1.8.0')
    # Processing the call keyword arguments (line 9)
    kwargs_712839 = {}
    # Getting the type of 'NumpyVersion' (line 9)
    NumpyVersion_712837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 9)
    NumpyVersion_call_result_712840 = invoke(stypy.reporting.localization.Localization(__file__, 9, 12), NumpyVersion_712837, *[str_712838], **kwargs_712839)
    
    str_712841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 37), 'str', '1.8.0')
    # Applying the binary operator '==' (line 9)
    result_eq_712842 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 12), '==', NumpyVersion_call_result_712840, str_712841)
    
    # Processing the call keyword arguments (line 9)
    kwargs_712843 = {}
    # Getting the type of 'assert_' (line 9)
    assert__712836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 9)
    assert__call_result_712844 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), assert__712836, *[result_eq_712842], **kwargs_712843)
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_712845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    str_712846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'str', '1.9.0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 15), list_712845, str_712846)
    # Adding element type (line 10)
    str_712847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 25), 'str', '2.0.0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 15), list_712845, str_712847)
    # Adding element type (line 10)
    str_712848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 34), 'str', '1.8.1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 15), list_712845, str_712848)
    
    # Testing the type of a for loop iterable (line 10)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 10, 4), list_712845)
    # Getting the type of the for loop variable (line 10)
    for_loop_var_712849 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 10, 4), list_712845)
    # Assigning a type to the variable 'ver' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'ver', for_loop_var_712849)
    # SSA begins for a for statement (line 10)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 11)
    # Processing the call arguments (line 11)
    
    
    # Call to NumpyVersion(...): (line 11)
    # Processing the call arguments (line 11)
    str_712852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 29), 'str', '1.8.0')
    # Processing the call keyword arguments (line 11)
    kwargs_712853 = {}
    # Getting the type of 'NumpyVersion' (line 11)
    NumpyVersion_712851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 11)
    NumpyVersion_call_result_712854 = invoke(stypy.reporting.localization.Localization(__file__, 11, 16), NumpyVersion_712851, *[str_712852], **kwargs_712853)
    
    # Getting the type of 'ver' (line 11)
    ver_712855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 40), 'ver', False)
    # Applying the binary operator '<' (line 11)
    result_lt_712856 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 16), '<', NumpyVersion_call_result_712854, ver_712855)
    
    # Processing the call keyword arguments (line 11)
    kwargs_712857 = {}
    # Getting the type of 'assert_' (line 11)
    assert__712850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 11)
    assert__call_result_712858 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), assert__712850, *[result_lt_712856], **kwargs_712857)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_712859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    str_712860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'str', '1.7.0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), list_712859, str_712860)
    # Adding element type (line 13)
    str_712861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'str', '1.7.1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), list_712859, str_712861)
    # Adding element type (line 13)
    str_712862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 34), 'str', '0.9.9')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), list_712859, str_712862)
    
    # Testing the type of a for loop iterable (line 13)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 13, 4), list_712859)
    # Getting the type of the for loop variable (line 13)
    for_loop_var_712863 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 13, 4), list_712859)
    # Assigning a type to the variable 'ver' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ver', for_loop_var_712863)
    # SSA begins for a for statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 14)
    # Processing the call arguments (line 14)
    
    
    # Call to NumpyVersion(...): (line 14)
    # Processing the call arguments (line 14)
    str_712866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'str', '1.8.0')
    # Processing the call keyword arguments (line 14)
    kwargs_712867 = {}
    # Getting the type of 'NumpyVersion' (line 14)
    NumpyVersion_712865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 14)
    NumpyVersion_call_result_712868 = invoke(stypy.reporting.localization.Localization(__file__, 14, 16), NumpyVersion_712865, *[str_712866], **kwargs_712867)
    
    # Getting the type of 'ver' (line 14)
    ver_712869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 40), 'ver', False)
    # Applying the binary operator '>' (line 14)
    result_gt_712870 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 16), '>', NumpyVersion_call_result_712868, ver_712869)
    
    # Processing the call keyword arguments (line 14)
    kwargs_712871 = {}
    # Getting the type of 'assert_' (line 14)
    assert__712864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 14)
    assert__call_result_712872 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), assert__712864, *[result_gt_712870], **kwargs_712871)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_main_versions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_main_versions' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_712873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712873)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_main_versions'
    return stypy_return_type_712873

# Assigning a type to the variable 'test_main_versions' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'test_main_versions', test_main_versions)

@norecursion
def test_version_1_point_10(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_version_1_point_10'
    module_type_store = module_type_store.open_function_context('test_version_1_point_10', 17, 0, False)
    
    # Passed parameters checking function
    test_version_1_point_10.stypy_localization = localization
    test_version_1_point_10.stypy_type_of_self = None
    test_version_1_point_10.stypy_type_store = module_type_store
    test_version_1_point_10.stypy_function_name = 'test_version_1_point_10'
    test_version_1_point_10.stypy_param_names_list = []
    test_version_1_point_10.stypy_varargs_param_name = None
    test_version_1_point_10.stypy_kwargs_param_name = None
    test_version_1_point_10.stypy_call_defaults = defaults
    test_version_1_point_10.stypy_call_varargs = varargs
    test_version_1_point_10.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_version_1_point_10', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_version_1_point_10', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_version_1_point_10(...)' code ##################

    
    # Call to assert_(...): (line 19)
    # Processing the call arguments (line 19)
    
    
    # Call to NumpyVersion(...): (line 19)
    # Processing the call arguments (line 19)
    str_712876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'str', '1.9.0')
    # Processing the call keyword arguments (line 19)
    kwargs_712877 = {}
    # Getting the type of 'NumpyVersion' (line 19)
    NumpyVersion_712875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 19)
    NumpyVersion_call_result_712878 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), NumpyVersion_712875, *[str_712876], **kwargs_712877)
    
    str_712879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 36), 'str', '1.10.0')
    # Applying the binary operator '<' (line 19)
    result_lt_712880 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 12), '<', NumpyVersion_call_result_712878, str_712879)
    
    # Processing the call keyword arguments (line 19)
    kwargs_712881 = {}
    # Getting the type of 'assert_' (line 19)
    assert__712874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 19)
    assert__call_result_712882 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), assert__712874, *[result_lt_712880], **kwargs_712881)
    
    
    # Call to assert_(...): (line 20)
    # Processing the call arguments (line 20)
    
    
    # Call to NumpyVersion(...): (line 20)
    # Processing the call arguments (line 20)
    str_712885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'str', '1.11.0')
    # Processing the call keyword arguments (line 20)
    kwargs_712886 = {}
    # Getting the type of 'NumpyVersion' (line 20)
    NumpyVersion_712884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 20)
    NumpyVersion_call_result_712887 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), NumpyVersion_712884, *[str_712885], **kwargs_712886)
    
    str_712888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 37), 'str', '1.11.1')
    # Applying the binary operator '<' (line 20)
    result_lt_712889 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 12), '<', NumpyVersion_call_result_712887, str_712888)
    
    # Processing the call keyword arguments (line 20)
    kwargs_712890 = {}
    # Getting the type of 'assert_' (line 20)
    assert__712883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 20)
    assert__call_result_712891 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), assert__712883, *[result_lt_712889], **kwargs_712890)
    
    
    # Call to assert_(...): (line 21)
    # Processing the call arguments (line 21)
    
    
    # Call to NumpyVersion(...): (line 21)
    # Processing the call arguments (line 21)
    str_712894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'str', '1.11.0')
    # Processing the call keyword arguments (line 21)
    kwargs_712895 = {}
    # Getting the type of 'NumpyVersion' (line 21)
    NumpyVersion_712893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 21)
    NumpyVersion_call_result_712896 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), NumpyVersion_712893, *[str_712894], **kwargs_712895)
    
    str_712897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 38), 'str', '1.11.0')
    # Applying the binary operator '==' (line 21)
    result_eq_712898 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 12), '==', NumpyVersion_call_result_712896, str_712897)
    
    # Processing the call keyword arguments (line 21)
    kwargs_712899 = {}
    # Getting the type of 'assert_' (line 21)
    assert__712892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 21)
    assert__call_result_712900 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), assert__712892, *[result_eq_712898], **kwargs_712899)
    
    
    # Call to assert_(...): (line 22)
    # Processing the call arguments (line 22)
    
    
    # Call to NumpyVersion(...): (line 22)
    # Processing the call arguments (line 22)
    str_712903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'str', '1.99.11')
    # Processing the call keyword arguments (line 22)
    kwargs_712904 = {}
    # Getting the type of 'NumpyVersion' (line 22)
    NumpyVersion_712902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 22)
    NumpyVersion_call_result_712905 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), NumpyVersion_712902, *[str_712903], **kwargs_712904)
    
    str_712906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 38), 'str', '1.99.12')
    # Applying the binary operator '<' (line 22)
    result_lt_712907 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), '<', NumpyVersion_call_result_712905, str_712906)
    
    # Processing the call keyword arguments (line 22)
    kwargs_712908 = {}
    # Getting the type of 'assert_' (line 22)
    assert__712901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 22)
    assert__call_result_712909 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), assert__712901, *[result_lt_712907], **kwargs_712908)
    
    
    # ################# End of 'test_version_1_point_10(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_version_1_point_10' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_712910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712910)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_version_1_point_10'
    return stypy_return_type_712910

# Assigning a type to the variable 'test_version_1_point_10' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'test_version_1_point_10', test_version_1_point_10)

@norecursion
def test_alpha_beta_rc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_alpha_beta_rc'
    module_type_store = module_type_store.open_function_context('test_alpha_beta_rc', 25, 0, False)
    
    # Passed parameters checking function
    test_alpha_beta_rc.stypy_localization = localization
    test_alpha_beta_rc.stypy_type_of_self = None
    test_alpha_beta_rc.stypy_type_store = module_type_store
    test_alpha_beta_rc.stypy_function_name = 'test_alpha_beta_rc'
    test_alpha_beta_rc.stypy_param_names_list = []
    test_alpha_beta_rc.stypy_varargs_param_name = None
    test_alpha_beta_rc.stypy_kwargs_param_name = None
    test_alpha_beta_rc.stypy_call_defaults = defaults
    test_alpha_beta_rc.stypy_call_varargs = varargs
    test_alpha_beta_rc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_alpha_beta_rc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_alpha_beta_rc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_alpha_beta_rc(...)' code ##################

    
    # Call to assert_(...): (line 26)
    # Processing the call arguments (line 26)
    
    
    # Call to NumpyVersion(...): (line 26)
    # Processing the call arguments (line 26)
    str_712913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'str', '1.8.0rc1')
    # Processing the call keyword arguments (line 26)
    kwargs_712914 = {}
    # Getting the type of 'NumpyVersion' (line 26)
    NumpyVersion_712912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 26)
    NumpyVersion_call_result_712915 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), NumpyVersion_712912, *[str_712913], **kwargs_712914)
    
    str_712916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 40), 'str', '1.8.0rc1')
    # Applying the binary operator '==' (line 26)
    result_eq_712917 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 12), '==', NumpyVersion_call_result_712915, str_712916)
    
    # Processing the call keyword arguments (line 26)
    kwargs_712918 = {}
    # Getting the type of 'assert_' (line 26)
    assert__712911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 26)
    assert__call_result_712919 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), assert__712911, *[result_eq_712917], **kwargs_712918)
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_712920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    str_712921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'str', '1.8.0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), list_712920, str_712921)
    # Adding element type (line 27)
    str_712922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'str', '1.8.0rc2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), list_712920, str_712922)
    
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 4), list_712920)
    # Getting the type of the for loop variable (line 27)
    for_loop_var_712923 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 4), list_712920)
    # Assigning a type to the variable 'ver' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ver', for_loop_var_712923)
    # SSA begins for a for statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 28)
    # Processing the call arguments (line 28)
    
    
    # Call to NumpyVersion(...): (line 28)
    # Processing the call arguments (line 28)
    str_712926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'str', '1.8.0rc1')
    # Processing the call keyword arguments (line 28)
    kwargs_712927 = {}
    # Getting the type of 'NumpyVersion' (line 28)
    NumpyVersion_712925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 28)
    NumpyVersion_call_result_712928 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), NumpyVersion_712925, *[str_712926], **kwargs_712927)
    
    # Getting the type of 'ver' (line 28)
    ver_712929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 43), 'ver', False)
    # Applying the binary operator '<' (line 28)
    result_lt_712930 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 16), '<', NumpyVersion_call_result_712928, ver_712929)
    
    # Processing the call keyword arguments (line 28)
    kwargs_712931 = {}
    # Getting the type of 'assert_' (line 28)
    assert__712924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 28)
    assert__call_result_712932 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assert__712924, *[result_lt_712930], **kwargs_712931)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_712933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    str_712934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'str', '1.8.0a2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 15), list_712933, str_712934)
    # Adding element type (line 30)
    str_712935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'str', '1.8.0b3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 15), list_712933, str_712935)
    # Adding element type (line 30)
    str_712936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'str', '1.7.2rc4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 15), list_712933, str_712936)
    
    # Testing the type of a for loop iterable (line 30)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 30, 4), list_712933)
    # Getting the type of the for loop variable (line 30)
    for_loop_var_712937 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 30, 4), list_712933)
    # Assigning a type to the variable 'ver' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'ver', for_loop_var_712937)
    # SSA begins for a for statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 31)
    # Processing the call arguments (line 31)
    
    
    # Call to NumpyVersion(...): (line 31)
    # Processing the call arguments (line 31)
    str_712940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'str', '1.8.0rc1')
    # Processing the call keyword arguments (line 31)
    kwargs_712941 = {}
    # Getting the type of 'NumpyVersion' (line 31)
    NumpyVersion_712939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 31)
    NumpyVersion_call_result_712942 = invoke(stypy.reporting.localization.Localization(__file__, 31, 16), NumpyVersion_712939, *[str_712940], **kwargs_712941)
    
    # Getting the type of 'ver' (line 31)
    ver_712943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 43), 'ver', False)
    # Applying the binary operator '>' (line 31)
    result_gt_712944 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 16), '>', NumpyVersion_call_result_712942, ver_712943)
    
    # Processing the call keyword arguments (line 31)
    kwargs_712945 = {}
    # Getting the type of 'assert_' (line 31)
    assert__712938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 31)
    assert__call_result_712946 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assert__712938, *[result_gt_712944], **kwargs_712945)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_(...): (line 33)
    # Processing the call arguments (line 33)
    
    
    # Call to NumpyVersion(...): (line 33)
    # Processing the call arguments (line 33)
    str_712949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'str', '1.8.0b1')
    # Processing the call keyword arguments (line 33)
    kwargs_712950 = {}
    # Getting the type of 'NumpyVersion' (line 33)
    NumpyVersion_712948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 33)
    NumpyVersion_call_result_712951 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), NumpyVersion_712948, *[str_712949], **kwargs_712950)
    
    str_712952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 38), 'str', '1.8.0a2')
    # Applying the binary operator '>' (line 33)
    result_gt_712953 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 12), '>', NumpyVersion_call_result_712951, str_712952)
    
    # Processing the call keyword arguments (line 33)
    kwargs_712954 = {}
    # Getting the type of 'assert_' (line 33)
    assert__712947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 33)
    assert__call_result_712955 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), assert__712947, *[result_gt_712953], **kwargs_712954)
    
    
    # ################# End of 'test_alpha_beta_rc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_alpha_beta_rc' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_712956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712956)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_alpha_beta_rc'
    return stypy_return_type_712956

# Assigning a type to the variable 'test_alpha_beta_rc' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'test_alpha_beta_rc', test_alpha_beta_rc)

@norecursion
def test_dev_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dev_version'
    module_type_store = module_type_store.open_function_context('test_dev_version', 36, 0, False)
    
    # Passed parameters checking function
    test_dev_version.stypy_localization = localization
    test_dev_version.stypy_type_of_self = None
    test_dev_version.stypy_type_store = module_type_store
    test_dev_version.stypy_function_name = 'test_dev_version'
    test_dev_version.stypy_param_names_list = []
    test_dev_version.stypy_varargs_param_name = None
    test_dev_version.stypy_kwargs_param_name = None
    test_dev_version.stypy_call_defaults = defaults
    test_dev_version.stypy_call_varargs = varargs
    test_dev_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dev_version', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dev_version', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dev_version(...)' code ##################

    
    # Call to assert_(...): (line 37)
    # Processing the call arguments (line 37)
    
    
    # Call to NumpyVersion(...): (line 37)
    # Processing the call arguments (line 37)
    str_712959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'str', '1.9.0.dev-Unknown')
    # Processing the call keyword arguments (line 37)
    kwargs_712960 = {}
    # Getting the type of 'NumpyVersion' (line 37)
    NumpyVersion_712958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 37)
    NumpyVersion_call_result_712961 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), NumpyVersion_712958, *[str_712959], **kwargs_712960)
    
    str_712962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 48), 'str', '1.9.0')
    # Applying the binary operator '<' (line 37)
    result_lt_712963 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 12), '<', NumpyVersion_call_result_712961, str_712962)
    
    # Processing the call keyword arguments (line 37)
    kwargs_712964 = {}
    # Getting the type of 'assert_' (line 37)
    assert__712957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 37)
    assert__call_result_712965 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), assert__712957, *[result_lt_712963], **kwargs_712964)
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_712966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    str_712967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'str', '1.9.0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), list_712966, str_712967)
    # Adding element type (line 38)
    str_712968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'str', '1.9.0a1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), list_712966, str_712968)
    # Adding element type (line 38)
    str_712969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'str', '1.9.0b2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), list_712966, str_712969)
    # Adding element type (line 38)
    str_712970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 47), 'str', '1.9.0b2.dev-ffffffff')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), list_712966, str_712970)
    
    # Testing the type of a for loop iterable (line 38)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 4), list_712966)
    # Getting the type of the for loop variable (line 38)
    for_loop_var_712971 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 4), list_712966)
    # Assigning a type to the variable 'ver' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'ver', for_loop_var_712971)
    # SSA begins for a for statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 39)
    # Processing the call arguments (line 39)
    
    
    # Call to NumpyVersion(...): (line 39)
    # Processing the call arguments (line 39)
    str_712974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 29), 'str', '1.9.0.dev-f16acvda')
    # Processing the call keyword arguments (line 39)
    kwargs_712975 = {}
    # Getting the type of 'NumpyVersion' (line 39)
    NumpyVersion_712973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 39)
    NumpyVersion_call_result_712976 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), NumpyVersion_712973, *[str_712974], **kwargs_712975)
    
    # Getting the type of 'ver' (line 39)
    ver_712977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 53), 'ver', False)
    # Applying the binary operator '<' (line 39)
    result_lt_712978 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 16), '<', NumpyVersion_call_result_712976, ver_712977)
    
    # Processing the call keyword arguments (line 39)
    kwargs_712979 = {}
    # Getting the type of 'assert_' (line 39)
    assert__712972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 39)
    assert__call_result_712980 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assert__712972, *[result_lt_712978], **kwargs_712979)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_(...): (line 41)
    # Processing the call arguments (line 41)
    
    
    # Call to NumpyVersion(...): (line 41)
    # Processing the call arguments (line 41)
    str_712983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'str', '1.9.0.dev-f16acvda')
    # Processing the call keyword arguments (line 41)
    kwargs_712984 = {}
    # Getting the type of 'NumpyVersion' (line 41)
    NumpyVersion_712982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 41)
    NumpyVersion_call_result_712985 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), NumpyVersion_712982, *[str_712983], **kwargs_712984)
    
    str_712986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 50), 'str', '1.9.0.dev-11111111')
    # Applying the binary operator '==' (line 41)
    result_eq_712987 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 12), '==', NumpyVersion_call_result_712985, str_712986)
    
    # Processing the call keyword arguments (line 41)
    kwargs_712988 = {}
    # Getting the type of 'assert_' (line 41)
    assert__712981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 41)
    assert__call_result_712989 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), assert__712981, *[result_eq_712987], **kwargs_712988)
    
    
    # ################# End of 'test_dev_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dev_version' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_712990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712990)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dev_version'
    return stypy_return_type_712990

# Assigning a type to the variable 'test_dev_version' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'test_dev_version', test_dev_version)

@norecursion
def test_dev_a_b_rc_mixed(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dev_a_b_rc_mixed'
    module_type_store = module_type_store.open_function_context('test_dev_a_b_rc_mixed', 44, 0, False)
    
    # Passed parameters checking function
    test_dev_a_b_rc_mixed.stypy_localization = localization
    test_dev_a_b_rc_mixed.stypy_type_of_self = None
    test_dev_a_b_rc_mixed.stypy_type_store = module_type_store
    test_dev_a_b_rc_mixed.stypy_function_name = 'test_dev_a_b_rc_mixed'
    test_dev_a_b_rc_mixed.stypy_param_names_list = []
    test_dev_a_b_rc_mixed.stypy_varargs_param_name = None
    test_dev_a_b_rc_mixed.stypy_kwargs_param_name = None
    test_dev_a_b_rc_mixed.stypy_call_defaults = defaults
    test_dev_a_b_rc_mixed.stypy_call_varargs = varargs
    test_dev_a_b_rc_mixed.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dev_a_b_rc_mixed', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dev_a_b_rc_mixed', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dev_a_b_rc_mixed(...)' code ##################

    
    # Call to assert_(...): (line 45)
    # Processing the call arguments (line 45)
    
    
    # Call to NumpyVersion(...): (line 45)
    # Processing the call arguments (line 45)
    str_712993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'str', '1.9.0a2.dev-f16acvda')
    # Processing the call keyword arguments (line 45)
    kwargs_712994 = {}
    # Getting the type of 'NumpyVersion' (line 45)
    NumpyVersion_712992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 45)
    NumpyVersion_call_result_712995 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), NumpyVersion_712992, *[str_712993], **kwargs_712994)
    
    str_712996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 52), 'str', '1.9.0a2.dev-11111111')
    # Applying the binary operator '==' (line 45)
    result_eq_712997 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 12), '==', NumpyVersion_call_result_712995, str_712996)
    
    # Processing the call keyword arguments (line 45)
    kwargs_712998 = {}
    # Getting the type of 'assert_' (line 45)
    assert__712991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 45)
    assert__call_result_712999 = invoke(stypy.reporting.localization.Localization(__file__, 45, 4), assert__712991, *[result_eq_712997], **kwargs_712998)
    
    
    # Call to assert_(...): (line 46)
    # Processing the call arguments (line 46)
    
    
    # Call to NumpyVersion(...): (line 46)
    # Processing the call arguments (line 46)
    str_713002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 25), 'str', '1.9.0a2.dev-6acvda54')
    # Processing the call keyword arguments (line 46)
    kwargs_713003 = {}
    # Getting the type of 'NumpyVersion' (line 46)
    NumpyVersion_713001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 46)
    NumpyVersion_call_result_713004 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), NumpyVersion_713001, *[str_713002], **kwargs_713003)
    
    str_713005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 51), 'str', '1.9.0a2')
    # Applying the binary operator '<' (line 46)
    result_lt_713006 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 12), '<', NumpyVersion_call_result_713004, str_713005)
    
    # Processing the call keyword arguments (line 46)
    kwargs_713007 = {}
    # Getting the type of 'assert_' (line 46)
    assert__713000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 46)
    assert__call_result_713008 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), assert__713000, *[result_lt_713006], **kwargs_713007)
    
    
    # ################# End of 'test_dev_a_b_rc_mixed(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dev_a_b_rc_mixed' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_713009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_713009)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dev_a_b_rc_mixed'
    return stypy_return_type_713009

# Assigning a type to the variable 'test_dev_a_b_rc_mixed' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'test_dev_a_b_rc_mixed', test_dev_a_b_rc_mixed)

@norecursion
def test_dev0_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dev0_version'
    module_type_store = module_type_store.open_function_context('test_dev0_version', 49, 0, False)
    
    # Passed parameters checking function
    test_dev0_version.stypy_localization = localization
    test_dev0_version.stypy_type_of_self = None
    test_dev0_version.stypy_type_store = module_type_store
    test_dev0_version.stypy_function_name = 'test_dev0_version'
    test_dev0_version.stypy_param_names_list = []
    test_dev0_version.stypy_varargs_param_name = None
    test_dev0_version.stypy_kwargs_param_name = None
    test_dev0_version.stypy_call_defaults = defaults
    test_dev0_version.stypy_call_varargs = varargs
    test_dev0_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dev0_version', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dev0_version', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dev0_version(...)' code ##################

    
    # Call to assert_(...): (line 50)
    # Processing the call arguments (line 50)
    
    
    # Call to NumpyVersion(...): (line 50)
    # Processing the call arguments (line 50)
    str_713012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 25), 'str', '1.9.0.dev0+Unknown')
    # Processing the call keyword arguments (line 50)
    kwargs_713013 = {}
    # Getting the type of 'NumpyVersion' (line 50)
    NumpyVersion_713011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 50)
    NumpyVersion_call_result_713014 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), NumpyVersion_713011, *[str_713012], **kwargs_713013)
    
    str_713015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 49), 'str', '1.9.0')
    # Applying the binary operator '<' (line 50)
    result_lt_713016 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 12), '<', NumpyVersion_call_result_713014, str_713015)
    
    # Processing the call keyword arguments (line 50)
    kwargs_713017 = {}
    # Getting the type of 'assert_' (line 50)
    assert__713010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 50)
    assert__call_result_713018 = invoke(stypy.reporting.localization.Localization(__file__, 50, 4), assert__713010, *[result_lt_713016], **kwargs_713017)
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 51)
    list_713019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 51)
    # Adding element type (line 51)
    str_713020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 16), 'str', '1.9.0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 15), list_713019, str_713020)
    # Adding element type (line 51)
    str_713021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'str', '1.9.0a1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 15), list_713019, str_713021)
    # Adding element type (line 51)
    str_713022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 36), 'str', '1.9.0b2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 15), list_713019, str_713022)
    # Adding element type (line 51)
    str_713023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 47), 'str', '1.9.0b2.dev0+ffffffff')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 15), list_713019, str_713023)
    
    # Testing the type of a for loop iterable (line 51)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 4), list_713019)
    # Getting the type of the for loop variable (line 51)
    for_loop_var_713024 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 4), list_713019)
    # Assigning a type to the variable 'ver' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'ver', for_loop_var_713024)
    # SSA begins for a for statement (line 51)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 52)
    # Processing the call arguments (line 52)
    
    
    # Call to NumpyVersion(...): (line 52)
    # Processing the call arguments (line 52)
    str_713027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'str', '1.9.0.dev0+f16acvda')
    # Processing the call keyword arguments (line 52)
    kwargs_713028 = {}
    # Getting the type of 'NumpyVersion' (line 52)
    NumpyVersion_713026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 52)
    NumpyVersion_call_result_713029 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), NumpyVersion_713026, *[str_713027], **kwargs_713028)
    
    # Getting the type of 'ver' (line 52)
    ver_713030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 54), 'ver', False)
    # Applying the binary operator '<' (line 52)
    result_lt_713031 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 16), '<', NumpyVersion_call_result_713029, ver_713030)
    
    # Processing the call keyword arguments (line 52)
    kwargs_713032 = {}
    # Getting the type of 'assert_' (line 52)
    assert__713025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 52)
    assert__call_result_713033 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assert__713025, *[result_lt_713031], **kwargs_713032)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_(...): (line 54)
    # Processing the call arguments (line 54)
    
    
    # Call to NumpyVersion(...): (line 54)
    # Processing the call arguments (line 54)
    str_713036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'str', '1.9.0.dev0+f16acvda')
    # Processing the call keyword arguments (line 54)
    kwargs_713037 = {}
    # Getting the type of 'NumpyVersion' (line 54)
    NumpyVersion_713035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 54)
    NumpyVersion_call_result_713038 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), NumpyVersion_713035, *[str_713036], **kwargs_713037)
    
    str_713039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 51), 'str', '1.9.0.dev0+11111111')
    # Applying the binary operator '==' (line 54)
    result_eq_713040 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 12), '==', NumpyVersion_call_result_713038, str_713039)
    
    # Processing the call keyword arguments (line 54)
    kwargs_713041 = {}
    # Getting the type of 'assert_' (line 54)
    assert__713034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 54)
    assert__call_result_713042 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), assert__713034, *[result_eq_713040], **kwargs_713041)
    
    
    # ################# End of 'test_dev0_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dev0_version' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_713043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_713043)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dev0_version'
    return stypy_return_type_713043

# Assigning a type to the variable 'test_dev0_version' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'test_dev0_version', test_dev0_version)

@norecursion
def test_dev0_a_b_rc_mixed(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dev0_a_b_rc_mixed'
    module_type_store = module_type_store.open_function_context('test_dev0_a_b_rc_mixed', 57, 0, False)
    
    # Passed parameters checking function
    test_dev0_a_b_rc_mixed.stypy_localization = localization
    test_dev0_a_b_rc_mixed.stypy_type_of_self = None
    test_dev0_a_b_rc_mixed.stypy_type_store = module_type_store
    test_dev0_a_b_rc_mixed.stypy_function_name = 'test_dev0_a_b_rc_mixed'
    test_dev0_a_b_rc_mixed.stypy_param_names_list = []
    test_dev0_a_b_rc_mixed.stypy_varargs_param_name = None
    test_dev0_a_b_rc_mixed.stypy_kwargs_param_name = None
    test_dev0_a_b_rc_mixed.stypy_call_defaults = defaults
    test_dev0_a_b_rc_mixed.stypy_call_varargs = varargs
    test_dev0_a_b_rc_mixed.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dev0_a_b_rc_mixed', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dev0_a_b_rc_mixed', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dev0_a_b_rc_mixed(...)' code ##################

    
    # Call to assert_(...): (line 58)
    # Processing the call arguments (line 58)
    
    
    # Call to NumpyVersion(...): (line 58)
    # Processing the call arguments (line 58)
    str_713046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'str', '1.9.0a2.dev0+f16acvda')
    # Processing the call keyword arguments (line 58)
    kwargs_713047 = {}
    # Getting the type of 'NumpyVersion' (line 58)
    NumpyVersion_713045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 58)
    NumpyVersion_call_result_713048 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), NumpyVersion_713045, *[str_713046], **kwargs_713047)
    
    str_713049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 53), 'str', '1.9.0a2.dev0+11111111')
    # Applying the binary operator '==' (line 58)
    result_eq_713050 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 12), '==', NumpyVersion_call_result_713048, str_713049)
    
    # Processing the call keyword arguments (line 58)
    kwargs_713051 = {}
    # Getting the type of 'assert_' (line 58)
    assert__713044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 58)
    assert__call_result_713052 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), assert__713044, *[result_eq_713050], **kwargs_713051)
    
    
    # Call to assert_(...): (line 59)
    # Processing the call arguments (line 59)
    
    
    # Call to NumpyVersion(...): (line 59)
    # Processing the call arguments (line 59)
    str_713055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 25), 'str', '1.9.0a2.dev0+6acvda54')
    # Processing the call keyword arguments (line 59)
    kwargs_713056 = {}
    # Getting the type of 'NumpyVersion' (line 59)
    NumpyVersion_713054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'NumpyVersion', False)
    # Calling NumpyVersion(args, kwargs) (line 59)
    NumpyVersion_call_result_713057 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), NumpyVersion_713054, *[str_713055], **kwargs_713056)
    
    str_713058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 52), 'str', '1.9.0a2')
    # Applying the binary operator '<' (line 59)
    result_lt_713059 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 12), '<', NumpyVersion_call_result_713057, str_713058)
    
    # Processing the call keyword arguments (line 59)
    kwargs_713060 = {}
    # Getting the type of 'assert_' (line 59)
    assert__713053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 59)
    assert__call_result_713061 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), assert__713053, *[result_lt_713059], **kwargs_713060)
    
    
    # ################# End of 'test_dev0_a_b_rc_mixed(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dev0_a_b_rc_mixed' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_713062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_713062)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dev0_a_b_rc_mixed'
    return stypy_return_type_713062

# Assigning a type to the variable 'test_dev0_a_b_rc_mixed' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'test_dev0_a_b_rc_mixed', test_dev0_a_b_rc_mixed)

@norecursion
def test_raises(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_raises'
    module_type_store = module_type_store.open_function_context('test_raises', 62, 0, False)
    
    # Passed parameters checking function
    test_raises.stypy_localization = localization
    test_raises.stypy_type_of_self = None
    test_raises.stypy_type_store = module_type_store
    test_raises.stypy_function_name = 'test_raises'
    test_raises.stypy_param_names_list = []
    test_raises.stypy_varargs_param_name = None
    test_raises.stypy_kwargs_param_name = None
    test_raises.stypy_call_defaults = defaults
    test_raises.stypy_call_varargs = varargs
    test_raises.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_raises', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_raises', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_raises(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 63)
    list_713063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 63)
    # Adding element type (line 63)
    str_713064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'str', '1.9')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 15), list_713063, str_713064)
    # Adding element type (line 63)
    str_713065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'str', '1,9.0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 15), list_713063, str_713065)
    # Adding element type (line 63)
    str_713066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 32), 'str', '1.7.x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 15), list_713063, str_713066)
    
    # Testing the type of a for loop iterable (line 63)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 4), list_713063)
    # Getting the type of the for loop variable (line 63)
    for_loop_var_713067 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 4), list_713063)
    # Assigning a type to the variable 'ver' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'ver', for_loop_var_713067)
    # SSA begins for a for statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_raises(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'ValueError' (line 64)
    ValueError_713069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'ValueError', False)
    # Getting the type of 'NumpyVersion' (line 64)
    NumpyVersion_713070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'NumpyVersion', False)
    # Getting the type of 'ver' (line 64)
    ver_713071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 48), 'ver', False)
    # Processing the call keyword arguments (line 64)
    kwargs_713072 = {}
    # Getting the type of 'assert_raises' (line 64)
    assert_raises_713068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 64)
    assert_raises_call_result_713073 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert_raises_713068, *[ValueError_713069, NumpyVersion_713070, ver_713071], **kwargs_713072)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_raises(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_raises' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_713074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_713074)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_raises'
    return stypy_return_type_713074

# Assigning a type to the variable 'test_raises' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'test_raises', test_raises)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
