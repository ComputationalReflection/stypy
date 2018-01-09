
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: from numpy.lib import stride_tricks
5: 
6: 
7: def rolling(a, window):
8:     shape = (a.size - window + 1, window)
9:     strides = (a.itemsize, a.itemsize)
10:     l = locals().copy()
11:     for v in l:
12:         print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
13:     print "\n\n"
14:     return stride_tricks.as_strided(a, shape=shape, strides=strides)
15: 
16: 
17: Z = rolling(np.arange(10), 3)
18: #
19: # l = globals().copy()
20: # for v in l:
21: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.lib import stride_tricks' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_3 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.lib')

if (type(import_3) is not StypyTypeError):

    if (import_3 != 'pyd_module'):
        __import__(import_3)
        sys_modules_4 = sys.modules[import_3]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.lib', sys_modules_4.module_type_store, module_type_store, ['stride_tricks'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_4, sys_modules_4.module_type_store, module_type_store)
    else:
        from numpy.lib import stride_tricks

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.lib', None, module_type_store, ['stride_tricks'], [stride_tricks])

else:
    # Assigning a type to the variable 'numpy.lib' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.lib', import_3)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


@norecursion
def rolling(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rolling'
    module_type_store = module_type_store.open_function_context('rolling', 7, 0, False)
    
    # Passed parameters checking function
    rolling.stypy_localization = localization
    rolling.stypy_type_of_self = None
    rolling.stypy_type_store = module_type_store
    rolling.stypy_function_name = 'rolling'
    rolling.stypy_param_names_list = ['a', 'window']
    rolling.stypy_varargs_param_name = None
    rolling.stypy_kwargs_param_name = None
    rolling.stypy_call_defaults = defaults
    rolling.stypy_call_varargs = varargs
    rolling.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rolling', ['a', 'window'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rolling', localization, ['a', 'window'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rolling(...)' code ##################

    
    # Assigning a Tuple to a Name (line 8):
    
    # Obtaining an instance of the builtin type 'tuple' (line 8)
    tuple_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 8)
    # Adding element type (line 8)
    # Getting the type of 'a' (line 8)
    a_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'a')
    # Obtaining the member 'size' of a type (line 8)
    size_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 13), a_6, 'size')
    # Getting the type of 'window' (line 8)
    window_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 22), 'window')
    # Applying the binary operator '-' (line 8)
    result_sub_9 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 13), '-', size_7, window_8)
    
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 31), 'int')
    # Applying the binary operator '+' (line 8)
    result_add_11 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 29), '+', result_sub_9, int_10)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), tuple_5, result_add_11)
    # Adding element type (line 8)
    # Getting the type of 'window' (line 8)
    window_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 34), 'window')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), tuple_5, window_12)
    
    # Assigning a type to the variable 'shape' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'shape', tuple_5)
    
    # Assigning a Tuple to a Name (line 9):
    
    # Obtaining an instance of the builtin type 'tuple' (line 9)
    tuple_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 9)
    # Adding element type (line 9)
    # Getting the type of 'a' (line 9)
    a_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'a')
    # Obtaining the member 'itemsize' of a type (line 9)
    itemsize_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 15), a_14, 'itemsize')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), tuple_13, itemsize_15)
    # Adding element type (line 9)
    # Getting the type of 'a' (line 9)
    a_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 27), 'a')
    # Obtaining the member 'itemsize' of a type (line 9)
    itemsize_17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 27), a_16, 'itemsize')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), tuple_13, itemsize_17)
    
    # Assigning a type to the variable 'strides' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'strides', tuple_13)
    
    # Assigning a Call to a Name (line 10):
    
    # Call to copy(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_22 = {}
    
    # Call to locals(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_19 = {}
    # Getting the type of 'locals' (line 10)
    locals_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'locals', False)
    # Calling locals(args, kwargs) (line 10)
    locals_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), locals_18, *[], **kwargs_19)
    
    # Obtaining the member 'copy' of a type (line 10)
    copy_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), locals_call_result_20, 'copy')
    # Calling copy(args, kwargs) (line 10)
    copy_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), copy_21, *[], **kwargs_22)
    
    # Assigning a type to the variable 'l' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'l', copy_call_result_23)
    
    # Getting the type of 'l' (line 11)
    l_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'l')
    # Testing the type of a for loop iterable (line 11)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 11, 4), l_24)
    # Getting the type of the for loop variable (line 11)
    for_loop_var_25 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 11, 4), l_24)
    # Assigning a type to the variable 'v' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'v', for_loop_var_25)
    # SSA begins for a for statement (line 11)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    str_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'str', "'")
    # Getting the type of 'v' (line 12)
    v_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'v')
    # Applying the binary operator '+' (line 12)
    result_add_28 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 15), '+', str_26, v_27)
    
    str_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'str', "'")
    # Applying the binary operator '+' (line 12)
    result_add_30 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 23), '+', result_add_28, str_29)
    
    str_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'str', ': instance_of_class_name("')
    # Applying the binary operator '+' (line 12)
    result_add_32 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 29), '+', result_add_30, str_31)
    
    
    # Call to type(...): (line 12)
    # Processing the call arguments (line 12)
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 12)
    v_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 70), 'v', False)
    # Getting the type of 'l' (line 12)
    l_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 68), 'l', False)
    # Obtaining the member '__getitem__' of a type (line 12)
    getitem___36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 68), l_35, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 12)
    subscript_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 12, 68), getitem___36, v_34)
    
    # Processing the call keyword arguments (line 12)
    kwargs_38 = {}
    # Getting the type of 'type' (line 12)
    type_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 63), 'type', False)
    # Calling type(args, kwargs) (line 12)
    type_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 12, 63), type_33, *[subscript_call_result_37], **kwargs_38)
    
    # Obtaining the member '__name__' of a type (line 12)
    name___40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 63), type_call_result_39, '__name__')
    # Applying the binary operator '+' (line 12)
    result_add_41 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 61), '+', result_add_32, name___40)
    
    str_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 85), 'str', '"),')
    # Applying the binary operator '+' (line 12)
    result_add_43 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 83), '+', result_add_41, str_42)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    str_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'str', '\n\n')
    
    # Call to as_strided(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'a' (line 14)
    a_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 36), 'a', False)
    # Processing the call keyword arguments (line 14)
    # Getting the type of 'shape' (line 14)
    shape_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 45), 'shape', False)
    keyword_49 = shape_48
    # Getting the type of 'strides' (line 14)
    strides_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 60), 'strides', False)
    keyword_51 = strides_50
    kwargs_52 = {'strides': keyword_51, 'shape': keyword_49}
    # Getting the type of 'stride_tricks' (line 14)
    stride_tricks_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'stride_tricks', False)
    # Obtaining the member 'as_strided' of a type (line 14)
    as_strided_46 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 11), stride_tricks_45, 'as_strided')
    # Calling as_strided(args, kwargs) (line 14)
    as_strided_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 14, 11), as_strided_46, *[a_47], **kwargs_52)
    
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type', as_strided_call_result_53)
    
    # ################# End of 'rolling(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rolling' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rolling'
    return stypy_return_type_54

# Assigning a type to the variable 'rolling' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'rolling', rolling)

# Assigning a Call to a Name (line 17):

# Call to rolling(...): (line 17)
# Processing the call arguments (line 17)

# Call to arange(...): (line 17)
# Processing the call arguments (line 17)
int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'int')
# Processing the call keyword arguments (line 17)
kwargs_59 = {}
# Getting the type of 'np' (line 17)
np_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'np', False)
# Obtaining the member 'arange' of a type (line 17)
arange_57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), np_56, 'arange')
# Calling arange(args, kwargs) (line 17)
arange_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), arange_57, *[int_58], **kwargs_59)

int_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'int')
# Processing the call keyword arguments (line 17)
kwargs_62 = {}
# Getting the type of 'rolling' (line 17)
rolling_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'rolling', False)
# Calling rolling(args, kwargs) (line 17)
rolling_call_result_63 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), rolling_55, *[arange_call_result_60, int_61], **kwargs_62)

# Assigning a type to the variable 'Z' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'Z', rolling_call_result_63)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
