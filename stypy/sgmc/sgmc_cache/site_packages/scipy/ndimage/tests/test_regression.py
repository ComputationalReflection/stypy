
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_array_almost_equal
5: 
6: import scipy.ndimage as ndimage
7: 
8: 
9: def test_byte_order_median():
10:     '''Regression test for #413: median_filter does not handle bytes orders.'''
11:     a = np.arange(9, dtype='<f4').reshape(3, 3)
12:     ref = ndimage.filters.median_filter(a,(3, 3))
13:     b = np.arange(9, dtype='>f4').reshape(3, 3)
14:     t = ndimage.filters.median_filter(b, (3, 3))
15:     assert_array_almost_equal(ref, t)
16: 
17: 
18: def test_zoom_output_shape():
19:     '''Ticket #643'''
20:     x = np.arange(12).reshape((3,4))
21:     ndimage.zoom(x, 2, output=np.zeros((6,8)))
22: 
23: 
24: def test_ticket_742():
25:     def SE(img, thresh=.7, size=4):
26:         mask = img > thresh
27:         rank = len(mask.shape)
28:         la, co = ndimage.label(mask,
29:                                ndimage.generate_binary_structure(rank, rank))
30:         slices = ndimage.find_objects(la)
31: 
32:     if np.dtype(np.intp) != np.dtype('i'):
33:         shape = (3,1240,1240)
34:         a = np.random.rand(np.product(shape)).reshape(shape)
35:         # shouldn't crash
36:         SE(a)
37: 
38: 
39: def test_gh_issue_3025():
40:     '''Github issue #3025 - improper merging of labels'''
41:     d = np.zeros((60,320))
42:     d[:,:257] = 1
43:     d[:,260:] = 1
44:     d[36,257] = 1
45:     d[35,258] = 1
46:     d[35,259] = 1
47:     assert ndimage.label(d, np.ones((3,3)))[1] == 1
48: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_162735 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_162735) is not StypyTypeError):

    if (import_162735 != 'pyd_module'):
        __import__(import_162735)
        sys_modules_162736 = sys.modules[import_162735]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_162736.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_162735)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_almost_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_162737 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_162737) is not StypyTypeError):

    if (import_162737 != 'pyd_module'):
        __import__(import_162737)
        sys_modules_162738 = sys.modules[import_162737]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_162738.module_type_store, module_type_store, ['assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_162738, sys_modules_162738.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal'], [assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_162737)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import scipy.ndimage' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_162739 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.ndimage')

if (type(import_162739) is not StypyTypeError):

    if (import_162739 != 'pyd_module'):
        __import__(import_162739)
        sys_modules_162740 = sys.modules[import_162739]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'ndimage', sys_modules_162740.module_type_store, module_type_store)
    else:
        import scipy.ndimage as ndimage

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'ndimage', scipy.ndimage, module_type_store)

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.ndimage', import_162739)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')


@norecursion
def test_byte_order_median(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_byte_order_median'
    module_type_store = module_type_store.open_function_context('test_byte_order_median', 9, 0, False)
    
    # Passed parameters checking function
    test_byte_order_median.stypy_localization = localization
    test_byte_order_median.stypy_type_of_self = None
    test_byte_order_median.stypy_type_store = module_type_store
    test_byte_order_median.stypy_function_name = 'test_byte_order_median'
    test_byte_order_median.stypy_param_names_list = []
    test_byte_order_median.stypy_varargs_param_name = None
    test_byte_order_median.stypy_kwargs_param_name = None
    test_byte_order_median.stypy_call_defaults = defaults
    test_byte_order_median.stypy_call_varargs = varargs
    test_byte_order_median.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_byte_order_median', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_byte_order_median', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_byte_order_median(...)' code ##################

    str_162741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'str', 'Regression test for #413: median_filter does not handle bytes orders.')
    
    # Assigning a Call to a Name (line 11):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to reshape(...): (line 11)
    # Processing the call arguments (line 11)
    int_162750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 42), 'int')
    int_162751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 45), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_162752 = {}
    
    # Call to arange(...): (line 11)
    # Processing the call arguments (line 11)
    int_162744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
    # Processing the call keyword arguments (line 11)
    str_162745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'str', '<f4')
    keyword_162746 = str_162745
    kwargs_162747 = {'dtype': keyword_162746}
    # Getting the type of 'np' (line 11)
    np_162742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 11)
    arange_162743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), np_162742, 'arange')
    # Calling arange(args, kwargs) (line 11)
    arange_call_result_162748 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), arange_162743, *[int_162744], **kwargs_162747)
    
    # Obtaining the member 'reshape' of a type (line 11)
    reshape_162749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), arange_call_result_162748, 'reshape')
    # Calling reshape(args, kwargs) (line 11)
    reshape_call_result_162753 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), reshape_162749, *[int_162750, int_162751], **kwargs_162752)
    
    # Assigning a type to the variable 'a' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', reshape_call_result_162753)
    
    # Assigning a Call to a Name (line 12):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to median_filter(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'a' (line 12)
    a_162757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 40), 'a', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 12)
    tuple_162758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 12)
    # Adding element type (line 12)
    int_162759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 43), tuple_162758, int_162759)
    # Adding element type (line 12)
    int_162760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 43), tuple_162758, int_162760)
    
    # Processing the call keyword arguments (line 12)
    kwargs_162761 = {}
    # Getting the type of 'ndimage' (line 12)
    ndimage_162754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'ndimage', False)
    # Obtaining the member 'filters' of a type (line 12)
    filters_162755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 10), ndimage_162754, 'filters')
    # Obtaining the member 'median_filter' of a type (line 12)
    median_filter_162756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 10), filters_162755, 'median_filter')
    # Calling median_filter(args, kwargs) (line 12)
    median_filter_call_result_162762 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), median_filter_162756, *[a_162757, tuple_162758], **kwargs_162761)
    
    # Assigning a type to the variable 'ref' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ref', median_filter_call_result_162762)
    
    # Assigning a Call to a Name (line 13):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to reshape(...): (line 13)
    # Processing the call arguments (line 13)
    int_162771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 42), 'int')
    int_162772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 45), 'int')
    # Processing the call keyword arguments (line 13)
    kwargs_162773 = {}
    
    # Call to arange(...): (line 13)
    # Processing the call arguments (line 13)
    int_162765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'int')
    # Processing the call keyword arguments (line 13)
    str_162766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 27), 'str', '>f4')
    keyword_162767 = str_162766
    kwargs_162768 = {'dtype': keyword_162767}
    # Getting the type of 'np' (line 13)
    np_162763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 13)
    arange_162764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), np_162763, 'arange')
    # Calling arange(args, kwargs) (line 13)
    arange_call_result_162769 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), arange_162764, *[int_162765], **kwargs_162768)
    
    # Obtaining the member 'reshape' of a type (line 13)
    reshape_162770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), arange_call_result_162769, 'reshape')
    # Calling reshape(args, kwargs) (line 13)
    reshape_call_result_162774 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), reshape_162770, *[int_162771, int_162772], **kwargs_162773)
    
    # Assigning a type to the variable 'b' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'b', reshape_call_result_162774)
    
    # Assigning a Call to a Name (line 14):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to median_filter(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'b' (line 14)
    b_162778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 38), 'b', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 14)
    tuple_162779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 14)
    # Adding element type (line 14)
    int_162780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 42), tuple_162779, int_162780)
    # Adding element type (line 14)
    int_162781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 42), tuple_162779, int_162781)
    
    # Processing the call keyword arguments (line 14)
    kwargs_162782 = {}
    # Getting the type of 'ndimage' (line 14)
    ndimage_162775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'ndimage', False)
    # Obtaining the member 'filters' of a type (line 14)
    filters_162776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), ndimage_162775, 'filters')
    # Obtaining the member 'median_filter' of a type (line 14)
    median_filter_162777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), filters_162776, 'median_filter')
    # Calling median_filter(args, kwargs) (line 14)
    median_filter_call_result_162783 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), median_filter_162777, *[b_162778, tuple_162779], **kwargs_162782)
    
    # Assigning a type to the variable 't' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 't', median_filter_call_result_162783)
    
    # Call to assert_array_almost_equal(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'ref' (line 15)
    ref_162785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 30), 'ref', False)
    # Getting the type of 't' (line 15)
    t_162786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 35), 't', False)
    # Processing the call keyword arguments (line 15)
    kwargs_162787 = {}
    # Getting the type of 'assert_array_almost_equal' (line 15)
    assert_array_almost_equal_162784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 15)
    assert_array_almost_equal_call_result_162788 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), assert_array_almost_equal_162784, *[ref_162785, t_162786], **kwargs_162787)
    
    
    # ################# End of 'test_byte_order_median(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_byte_order_median' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_162789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162789)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_byte_order_median'
    return stypy_return_type_162789

# Assigning a type to the variable 'test_byte_order_median' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_byte_order_median', test_byte_order_median)

@norecursion
def test_zoom_output_shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_zoom_output_shape'
    module_type_store = module_type_store.open_function_context('test_zoom_output_shape', 18, 0, False)
    
    # Passed parameters checking function
    test_zoom_output_shape.stypy_localization = localization
    test_zoom_output_shape.stypy_type_of_self = None
    test_zoom_output_shape.stypy_type_store = module_type_store
    test_zoom_output_shape.stypy_function_name = 'test_zoom_output_shape'
    test_zoom_output_shape.stypy_param_names_list = []
    test_zoom_output_shape.stypy_varargs_param_name = None
    test_zoom_output_shape.stypy_kwargs_param_name = None
    test_zoom_output_shape.stypy_call_defaults = defaults
    test_zoom_output_shape.stypy_call_varargs = varargs
    test_zoom_output_shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_zoom_output_shape', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_zoom_output_shape', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_zoom_output_shape(...)' code ##################

    str_162790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'str', 'Ticket #643')
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to reshape(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 20)
    tuple_162797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 20)
    # Adding element type (line 20)
    int_162798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 31), tuple_162797, int_162798)
    # Adding element type (line 20)
    int_162799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 31), tuple_162797, int_162799)
    
    # Processing the call keyword arguments (line 20)
    kwargs_162800 = {}
    
    # Call to arange(...): (line 20)
    # Processing the call arguments (line 20)
    int_162793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'int')
    # Processing the call keyword arguments (line 20)
    kwargs_162794 = {}
    # Getting the type of 'np' (line 20)
    np_162791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 20)
    arange_162792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), np_162791, 'arange')
    # Calling arange(args, kwargs) (line 20)
    arange_call_result_162795 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), arange_162792, *[int_162793], **kwargs_162794)
    
    # Obtaining the member 'reshape' of a type (line 20)
    reshape_162796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), arange_call_result_162795, 'reshape')
    # Calling reshape(args, kwargs) (line 20)
    reshape_call_result_162801 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), reshape_162796, *[tuple_162797], **kwargs_162800)
    
    # Assigning a type to the variable 'x' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'x', reshape_call_result_162801)
    
    # Call to zoom(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'x' (line 21)
    x_162804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'x', False)
    int_162805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'int')
    # Processing the call keyword arguments (line 21)
    
    # Call to zeros(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Obtaining an instance of the builtin type 'tuple' (line 21)
    tuple_162808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 21)
    # Adding element type (line 21)
    int_162809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 40), tuple_162808, int_162809)
    # Adding element type (line 21)
    int_162810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 40), tuple_162808, int_162810)
    
    # Processing the call keyword arguments (line 21)
    kwargs_162811 = {}
    # Getting the type of 'np' (line 21)
    np_162806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 30), 'np', False)
    # Obtaining the member 'zeros' of a type (line 21)
    zeros_162807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 30), np_162806, 'zeros')
    # Calling zeros(args, kwargs) (line 21)
    zeros_call_result_162812 = invoke(stypy.reporting.localization.Localization(__file__, 21, 30), zeros_162807, *[tuple_162808], **kwargs_162811)
    
    keyword_162813 = zeros_call_result_162812
    kwargs_162814 = {'output': keyword_162813}
    # Getting the type of 'ndimage' (line 21)
    ndimage_162802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ndimage', False)
    # Obtaining the member 'zoom' of a type (line 21)
    zoom_162803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), ndimage_162802, 'zoom')
    # Calling zoom(args, kwargs) (line 21)
    zoom_call_result_162815 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), zoom_162803, *[x_162804, int_162805], **kwargs_162814)
    
    
    # ################# End of 'test_zoom_output_shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_zoom_output_shape' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_162816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162816)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_zoom_output_shape'
    return stypy_return_type_162816

# Assigning a type to the variable 'test_zoom_output_shape' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'test_zoom_output_shape', test_zoom_output_shape)

@norecursion
def test_ticket_742(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ticket_742'
    module_type_store = module_type_store.open_function_context('test_ticket_742', 24, 0, False)
    
    # Passed parameters checking function
    test_ticket_742.stypy_localization = localization
    test_ticket_742.stypy_type_of_self = None
    test_ticket_742.stypy_type_store = module_type_store
    test_ticket_742.stypy_function_name = 'test_ticket_742'
    test_ticket_742.stypy_param_names_list = []
    test_ticket_742.stypy_varargs_param_name = None
    test_ticket_742.stypy_kwargs_param_name = None
    test_ticket_742.stypy_call_defaults = defaults
    test_ticket_742.stypy_call_varargs = varargs
    test_ticket_742.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ticket_742', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ticket_742', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ticket_742(...)' code ##################


    @norecursion
    def SE(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_162817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'float')
        int_162818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 32), 'int')
        defaults = [float_162817, int_162818]
        # Create a new context for function 'SE'
        module_type_store = module_type_store.open_function_context('SE', 25, 4, False)
        
        # Passed parameters checking function
        SE.stypy_localization = localization
        SE.stypy_type_of_self = None
        SE.stypy_type_store = module_type_store
        SE.stypy_function_name = 'SE'
        SE.stypy_param_names_list = ['img', 'thresh', 'size']
        SE.stypy_varargs_param_name = None
        SE.stypy_kwargs_param_name = None
        SE.stypy_call_defaults = defaults
        SE.stypy_call_varargs = varargs
        SE.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'SE', ['img', 'thresh', 'size'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'SE', localization, ['img', 'thresh', 'size'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'SE(...)' code ##################

        
        # Assigning a Compare to a Name (line 26):
        
        # Assigning a Compare to a Name (line 26):
        
        # Getting the type of 'img' (line 26)
        img_162819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'img')
        # Getting the type of 'thresh' (line 26)
        thresh_162820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'thresh')
        # Applying the binary operator '>' (line 26)
        result_gt_162821 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 15), '>', img_162819, thresh_162820)
        
        # Assigning a type to the variable 'mask' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'mask', result_gt_162821)
        
        # Assigning a Call to a Name (line 27):
        
        # Assigning a Call to a Name (line 27):
        
        # Call to len(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'mask' (line 27)
        mask_162823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'mask', False)
        # Obtaining the member 'shape' of a type (line 27)
        shape_162824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 19), mask_162823, 'shape')
        # Processing the call keyword arguments (line 27)
        kwargs_162825 = {}
        # Getting the type of 'len' (line 27)
        len_162822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'len', False)
        # Calling len(args, kwargs) (line 27)
        len_call_result_162826 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), len_162822, *[shape_162824], **kwargs_162825)
        
        # Assigning a type to the variable 'rank' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'rank', len_call_result_162826)
        
        # Assigning a Call to a Tuple (line 28):
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_162827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to label(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'mask' (line 28)
        mask_162830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'mask', False)
        
        # Call to generate_binary_structure(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'rank' (line 29)
        rank_162833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 65), 'rank', False)
        # Getting the type of 'rank' (line 29)
        rank_162834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 71), 'rank', False)
        # Processing the call keyword arguments (line 29)
        kwargs_162835 = {}
        # Getting the type of 'ndimage' (line 29)
        ndimage_162831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'ndimage', False)
        # Obtaining the member 'generate_binary_structure' of a type (line 29)
        generate_binary_structure_162832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 31), ndimage_162831, 'generate_binary_structure')
        # Calling generate_binary_structure(args, kwargs) (line 29)
        generate_binary_structure_call_result_162836 = invoke(stypy.reporting.localization.Localization(__file__, 29, 31), generate_binary_structure_162832, *[rank_162833, rank_162834], **kwargs_162835)
        
        # Processing the call keyword arguments (line 28)
        kwargs_162837 = {}
        # Getting the type of 'ndimage' (line 28)
        ndimage_162828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'ndimage', False)
        # Obtaining the member 'label' of a type (line 28)
        label_162829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 17), ndimage_162828, 'label')
        # Calling label(args, kwargs) (line 28)
        label_call_result_162838 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), label_162829, *[mask_162830, generate_binary_structure_call_result_162836], **kwargs_162837)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___162839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), label_call_result_162838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_162840 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___162839, int_162827)
        
        # Assigning a type to the variable 'tuple_var_assignment_162733' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_162733', subscript_call_result_162840)
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_162841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to label(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'mask' (line 28)
        mask_162844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'mask', False)
        
        # Call to generate_binary_structure(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'rank' (line 29)
        rank_162847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 65), 'rank', False)
        # Getting the type of 'rank' (line 29)
        rank_162848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 71), 'rank', False)
        # Processing the call keyword arguments (line 29)
        kwargs_162849 = {}
        # Getting the type of 'ndimage' (line 29)
        ndimage_162845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'ndimage', False)
        # Obtaining the member 'generate_binary_structure' of a type (line 29)
        generate_binary_structure_162846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 31), ndimage_162845, 'generate_binary_structure')
        # Calling generate_binary_structure(args, kwargs) (line 29)
        generate_binary_structure_call_result_162850 = invoke(stypy.reporting.localization.Localization(__file__, 29, 31), generate_binary_structure_162846, *[rank_162847, rank_162848], **kwargs_162849)
        
        # Processing the call keyword arguments (line 28)
        kwargs_162851 = {}
        # Getting the type of 'ndimage' (line 28)
        ndimage_162842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'ndimage', False)
        # Obtaining the member 'label' of a type (line 28)
        label_162843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 17), ndimage_162842, 'label')
        # Calling label(args, kwargs) (line 28)
        label_call_result_162852 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), label_162843, *[mask_162844, generate_binary_structure_call_result_162850], **kwargs_162851)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___162853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), label_call_result_162852, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_162854 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___162853, int_162841)
        
        # Assigning a type to the variable 'tuple_var_assignment_162734' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_162734', subscript_call_result_162854)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_162733' (line 28)
        tuple_var_assignment_162733_162855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_162733')
        # Assigning a type to the variable 'la' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'la', tuple_var_assignment_162733_162855)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_162734' (line 28)
        tuple_var_assignment_162734_162856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_162734')
        # Assigning a type to the variable 'co' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'co', tuple_var_assignment_162734_162856)
        
        # Assigning a Call to a Name (line 30):
        
        # Assigning a Call to a Name (line 30):
        
        # Call to find_objects(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'la' (line 30)
        la_162859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 38), 'la', False)
        # Processing the call keyword arguments (line 30)
        kwargs_162860 = {}
        # Getting the type of 'ndimage' (line 30)
        ndimage_162857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'ndimage', False)
        # Obtaining the member 'find_objects' of a type (line 30)
        find_objects_162858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 17), ndimage_162857, 'find_objects')
        # Calling find_objects(args, kwargs) (line 30)
        find_objects_call_result_162861 = invoke(stypy.reporting.localization.Localization(__file__, 30, 17), find_objects_162858, *[la_162859], **kwargs_162860)
        
        # Assigning a type to the variable 'slices' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'slices', find_objects_call_result_162861)
        
        # ################# End of 'SE(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'SE' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_162862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_162862)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'SE'
        return stypy_return_type_162862

    # Assigning a type to the variable 'SE' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'SE', SE)
    
    
    
    # Call to dtype(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'np' (line 32)
    np_162865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'np', False)
    # Obtaining the member 'intp' of a type (line 32)
    intp_162866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), np_162865, 'intp')
    # Processing the call keyword arguments (line 32)
    kwargs_162867 = {}
    # Getting the type of 'np' (line 32)
    np_162863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 7), 'np', False)
    # Obtaining the member 'dtype' of a type (line 32)
    dtype_162864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 7), np_162863, 'dtype')
    # Calling dtype(args, kwargs) (line 32)
    dtype_call_result_162868 = invoke(stypy.reporting.localization.Localization(__file__, 32, 7), dtype_162864, *[intp_162866], **kwargs_162867)
    
    
    # Call to dtype(...): (line 32)
    # Processing the call arguments (line 32)
    str_162871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 37), 'str', 'i')
    # Processing the call keyword arguments (line 32)
    kwargs_162872 = {}
    # Getting the type of 'np' (line 32)
    np_162869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 28), 'np', False)
    # Obtaining the member 'dtype' of a type (line 32)
    dtype_162870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 28), np_162869, 'dtype')
    # Calling dtype(args, kwargs) (line 32)
    dtype_call_result_162873 = invoke(stypy.reporting.localization.Localization(__file__, 32, 28), dtype_162870, *[str_162871], **kwargs_162872)
    
    # Applying the binary operator '!=' (line 32)
    result_ne_162874 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 7), '!=', dtype_call_result_162868, dtype_call_result_162873)
    
    # Testing the type of an if condition (line 32)
    if_condition_162875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 4), result_ne_162874)
    # Assigning a type to the variable 'if_condition_162875' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'if_condition_162875', if_condition_162875)
    # SSA begins for if statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 33):
    
    # Assigning a Tuple to a Name (line 33):
    
    # Obtaining an instance of the builtin type 'tuple' (line 33)
    tuple_162876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 33)
    # Adding element type (line 33)
    int_162877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 17), tuple_162876, int_162877)
    # Adding element type (line 33)
    int_162878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 17), tuple_162876, int_162878)
    # Adding element type (line 33)
    int_162879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 17), tuple_162876, int_162879)
    
    # Assigning a type to the variable 'shape' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'shape', tuple_162876)
    
    # Assigning a Call to a Name (line 34):
    
    # Assigning a Call to a Name (line 34):
    
    # Call to reshape(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'shape' (line 34)
    shape_162891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 54), 'shape', False)
    # Processing the call keyword arguments (line 34)
    kwargs_162892 = {}
    
    # Call to rand(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to product(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'shape' (line 34)
    shape_162885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 38), 'shape', False)
    # Processing the call keyword arguments (line 34)
    kwargs_162886 = {}
    # Getting the type of 'np' (line 34)
    np_162883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'np', False)
    # Obtaining the member 'product' of a type (line 34)
    product_162884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 27), np_162883, 'product')
    # Calling product(args, kwargs) (line 34)
    product_call_result_162887 = invoke(stypy.reporting.localization.Localization(__file__, 34, 27), product_162884, *[shape_162885], **kwargs_162886)
    
    # Processing the call keyword arguments (line 34)
    kwargs_162888 = {}
    # Getting the type of 'np' (line 34)
    np_162880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 34)
    random_162881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), np_162880, 'random')
    # Obtaining the member 'rand' of a type (line 34)
    rand_162882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), random_162881, 'rand')
    # Calling rand(args, kwargs) (line 34)
    rand_call_result_162889 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), rand_162882, *[product_call_result_162887], **kwargs_162888)
    
    # Obtaining the member 'reshape' of a type (line 34)
    reshape_162890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), rand_call_result_162889, 'reshape')
    # Calling reshape(args, kwargs) (line 34)
    reshape_call_result_162893 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), reshape_162890, *[shape_162891], **kwargs_162892)
    
    # Assigning a type to the variable 'a' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'a', reshape_call_result_162893)
    
    # Call to SE(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'a' (line 36)
    a_162895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'a', False)
    # Processing the call keyword arguments (line 36)
    kwargs_162896 = {}
    # Getting the type of 'SE' (line 36)
    SE_162894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'SE', False)
    # Calling SE(args, kwargs) (line 36)
    SE_call_result_162897 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), SE_162894, *[a_162895], **kwargs_162896)
    
    # SSA join for if statement (line 32)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_ticket_742(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ticket_742' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_162898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162898)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ticket_742'
    return stypy_return_type_162898

# Assigning a type to the variable 'test_ticket_742' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'test_ticket_742', test_ticket_742)

@norecursion
def test_gh_issue_3025(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gh_issue_3025'
    module_type_store = module_type_store.open_function_context('test_gh_issue_3025', 39, 0, False)
    
    # Passed parameters checking function
    test_gh_issue_3025.stypy_localization = localization
    test_gh_issue_3025.stypy_type_of_self = None
    test_gh_issue_3025.stypy_type_store = module_type_store
    test_gh_issue_3025.stypy_function_name = 'test_gh_issue_3025'
    test_gh_issue_3025.stypy_param_names_list = []
    test_gh_issue_3025.stypy_varargs_param_name = None
    test_gh_issue_3025.stypy_kwargs_param_name = None
    test_gh_issue_3025.stypy_call_defaults = defaults
    test_gh_issue_3025.stypy_call_varargs = varargs
    test_gh_issue_3025.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gh_issue_3025', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gh_issue_3025', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gh_issue_3025(...)' code ##################

    str_162899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'str', 'Github issue #3025 - improper merging of labels')
    
    # Assigning a Call to a Name (line 41):
    
    # Assigning a Call to a Name (line 41):
    
    # Call to zeros(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining an instance of the builtin type 'tuple' (line 41)
    tuple_162902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 41)
    # Adding element type (line 41)
    int_162903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 18), tuple_162902, int_162903)
    # Adding element type (line 41)
    int_162904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 18), tuple_162902, int_162904)
    
    # Processing the call keyword arguments (line 41)
    kwargs_162905 = {}
    # Getting the type of 'np' (line 41)
    np_162900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 41)
    zeros_162901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), np_162900, 'zeros')
    # Calling zeros(args, kwargs) (line 41)
    zeros_call_result_162906 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), zeros_162901, *[tuple_162902], **kwargs_162905)
    
    # Assigning a type to the variable 'd' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'd', zeros_call_result_162906)
    
    # Assigning a Num to a Subscript (line 42):
    
    # Assigning a Num to a Subscript (line 42):
    int_162907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'int')
    # Getting the type of 'd' (line 42)
    d_162908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'd')
    slice_162909 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 4), None, None, None)
    int_162910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'int')
    slice_162911 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 4), None, int_162910, None)
    # Storing an element on a container (line 42)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 4), d_162908, ((slice_162909, slice_162911), int_162907))
    
    # Assigning a Num to a Subscript (line 43):
    
    # Assigning a Num to a Subscript (line 43):
    int_162912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'int')
    # Getting the type of 'd' (line 43)
    d_162913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'd')
    slice_162914 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 43, 4), None, None, None)
    int_162915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 8), 'int')
    slice_162916 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 43, 4), int_162915, None, None)
    # Storing an element on a container (line 43)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), d_162913, ((slice_162914, slice_162916), int_162912))
    
    # Assigning a Num to a Subscript (line 44):
    
    # Assigning a Num to a Subscript (line 44):
    int_162917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'int')
    # Getting the type of 'd' (line 44)
    d_162918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'd')
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_162919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    int_162920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 6), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 6), tuple_162919, int_162920)
    # Adding element type (line 44)
    int_162921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 6), tuple_162919, int_162921)
    
    # Storing an element on a container (line 44)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 4), d_162918, (tuple_162919, int_162917))
    
    # Assigning a Num to a Subscript (line 45):
    
    # Assigning a Num to a Subscript (line 45):
    int_162922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 16), 'int')
    # Getting the type of 'd' (line 45)
    d_162923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'd')
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_162924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    int_162925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 6), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 6), tuple_162924, int_162925)
    # Adding element type (line 45)
    int_162926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 6), tuple_162924, int_162926)
    
    # Storing an element on a container (line 45)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 4), d_162923, (tuple_162924, int_162922))
    
    # Assigning a Num to a Subscript (line 46):
    
    # Assigning a Num to a Subscript (line 46):
    int_162927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 16), 'int')
    # Getting the type of 'd' (line 46)
    d_162928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'd')
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_162929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 6), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    int_162930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 6), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 6), tuple_162929, int_162930)
    # Adding element type (line 46)
    int_162931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 6), tuple_162929, int_162931)
    
    # Storing an element on a container (line 46)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 4), d_162928, (tuple_162929, int_162927))
    # Evaluating assert statement condition
    
    
    # Obtaining the type of the subscript
    int_162932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 44), 'int')
    
    # Call to label(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'd' (line 47)
    d_162935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'd', False)
    
    # Call to ones(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Obtaining an instance of the builtin type 'tuple' (line 47)
    tuple_162938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 47)
    # Adding element type (line 47)
    int_162939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 37), tuple_162938, int_162939)
    # Adding element type (line 47)
    int_162940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 37), tuple_162938, int_162940)
    
    # Processing the call keyword arguments (line 47)
    kwargs_162941 = {}
    # Getting the type of 'np' (line 47)
    np_162936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 28), 'np', False)
    # Obtaining the member 'ones' of a type (line 47)
    ones_162937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 28), np_162936, 'ones')
    # Calling ones(args, kwargs) (line 47)
    ones_call_result_162942 = invoke(stypy.reporting.localization.Localization(__file__, 47, 28), ones_162937, *[tuple_162938], **kwargs_162941)
    
    # Processing the call keyword arguments (line 47)
    kwargs_162943 = {}
    # Getting the type of 'ndimage' (line 47)
    ndimage_162933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'ndimage', False)
    # Obtaining the member 'label' of a type (line 47)
    label_162934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), ndimage_162933, 'label')
    # Calling label(args, kwargs) (line 47)
    label_call_result_162944 = invoke(stypy.reporting.localization.Localization(__file__, 47, 11), label_162934, *[d_162935, ones_call_result_162942], **kwargs_162943)
    
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___162945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), label_call_result_162944, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_162946 = invoke(stypy.reporting.localization.Localization(__file__, 47, 11), getitem___162945, int_162932)
    
    int_162947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 50), 'int')
    # Applying the binary operator '==' (line 47)
    result_eq_162948 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '==', subscript_call_result_162946, int_162947)
    
    
    # ################# End of 'test_gh_issue_3025(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gh_issue_3025' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_162949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_162949)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gh_issue_3025'
    return stypy_return_type_162949

# Assigning a type to the variable 'test_gh_issue_3025' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'test_gh_issue_3025', test_gh_issue_3025)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
