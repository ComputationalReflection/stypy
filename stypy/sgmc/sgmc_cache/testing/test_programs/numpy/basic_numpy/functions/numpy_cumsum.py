
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: 
6: def moving_average(a, n=3):
7:     ret = np.cumsum(a, dtype=float)
8:     ret[n:] = ret[n:] - ret[:-n]
9:     return ret[n - 1:] / n
10: 
11: 
12: Z = np.arange(20)
13: r = moving_average(Z, n=3)
14: 
15: # l = globals().copy()
16: # for v in l:
17: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_613 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_613) is not StypyTypeError):

    if (import_613 != 'pyd_module'):
        __import__(import_613)
        sys_modules_614 = sys.modules[import_613]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_614.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_613)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


@norecursion
def moving_average(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 24), 'int')
    defaults = [int_615]
    # Create a new context for function 'moving_average'
    module_type_store = module_type_store.open_function_context('moving_average', 6, 0, False)
    
    # Passed parameters checking function
    moving_average.stypy_localization = localization
    moving_average.stypy_type_of_self = None
    moving_average.stypy_type_store = module_type_store
    moving_average.stypy_function_name = 'moving_average'
    moving_average.stypy_param_names_list = ['a', 'n']
    moving_average.stypy_varargs_param_name = None
    moving_average.stypy_kwargs_param_name = None
    moving_average.stypy_call_defaults = defaults
    moving_average.stypy_call_varargs = varargs
    moving_average.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'moving_average', ['a', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'moving_average', localization, ['a', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'moving_average(...)' code ##################

    
    # Assigning a Call to a Name (line 7):
    
    # Call to cumsum(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'a' (line 7)
    a_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 20), 'a', False)
    # Processing the call keyword arguments (line 7)
    # Getting the type of 'float' (line 7)
    float_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 29), 'float', False)
    keyword_620 = float_619
    kwargs_621 = {'dtype': keyword_620}
    # Getting the type of 'np' (line 7)
    np_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 7)
    cumsum_617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 10), np_616, 'cumsum')
    # Calling cumsum(args, kwargs) (line 7)
    cumsum_call_result_622 = invoke(stypy.reporting.localization.Localization(__file__, 7, 10), cumsum_617, *[a_618], **kwargs_621)
    
    # Assigning a type to the variable 'ret' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'ret', cumsum_call_result_622)
    
    # Assigning a BinOp to a Subscript (line 8):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 8)
    n_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 18), 'n')
    slice_624 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 8, 14), n_623, None, None)
    # Getting the type of 'ret' (line 8)
    ret_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'ret')
    # Obtaining the member '__getitem__' of a type (line 8)
    getitem___626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 14), ret_625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 8)
    subscript_call_result_627 = invoke(stypy.reporting.localization.Localization(__file__, 8, 14), getitem___626, slice_624)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'n' (line 8)
    n_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 30), 'n')
    # Applying the 'usub' unary operator (line 8)
    result___neg___629 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 29), 'usub', n_628)
    
    slice_630 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 8, 24), None, result___neg___629, None)
    # Getting the type of 'ret' (line 8)
    ret_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 24), 'ret')
    # Obtaining the member '__getitem__' of a type (line 8)
    getitem___632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 24), ret_631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 8)
    subscript_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 8, 24), getitem___632, slice_630)
    
    # Applying the binary operator '-' (line 8)
    result_sub_634 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 14), '-', subscript_call_result_627, subscript_call_result_633)
    
    # Getting the type of 'ret' (line 8)
    ret_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'ret')
    # Getting the type of 'n' (line 8)
    n_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'n')
    slice_637 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 8, 4), n_636, None, None)
    # Storing an element on a container (line 8)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 4), ret_635, (slice_637, result_sub_634))
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 9)
    n_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'n')
    int_639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'int')
    # Applying the binary operator '-' (line 9)
    result_sub_640 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 15), '-', n_638, int_639)
    
    slice_641 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 9, 11), result_sub_640, None, None)
    # Getting the type of 'ret' (line 9)
    ret_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'ret')
    # Obtaining the member '__getitem__' of a type (line 9)
    getitem___643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 11), ret_642, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 9)
    subscript_call_result_644 = invoke(stypy.reporting.localization.Localization(__file__, 9, 11), getitem___643, slice_641)
    
    # Getting the type of 'n' (line 9)
    n_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 25), 'n')
    # Applying the binary operator 'div' (line 9)
    result_div_646 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 11), 'div', subscript_call_result_644, n_645)
    
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type', result_div_646)
    
    # ################# End of 'moving_average(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'moving_average' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_647)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'moving_average'
    return stypy_return_type_647

# Assigning a type to the variable 'moving_average' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'moving_average', moving_average)

# Assigning a Call to a Name (line 12):

# Call to arange(...): (line 12)
# Processing the call arguments (line 12)
int_650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
# Processing the call keyword arguments (line 12)
kwargs_651 = {}
# Getting the type of 'np' (line 12)
np_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 12)
arange_649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), np_648, 'arange')
# Calling arange(args, kwargs) (line 12)
arange_call_result_652 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), arange_649, *[int_650], **kwargs_651)

# Assigning a type to the variable 'Z' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'Z', arange_call_result_652)

# Assigning a Call to a Name (line 13):

# Call to moving_average(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'Z' (line 13)
Z_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'Z', False)
# Processing the call keyword arguments (line 13)
int_655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 24), 'int')
keyword_656 = int_655
kwargs_657 = {'n': keyword_656}
# Getting the type of 'moving_average' (line 13)
moving_average_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'moving_average', False)
# Calling moving_average(args, kwargs) (line 13)
moving_average_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), moving_average_653, *[Z_654], **kwargs_657)

# Assigning a type to the variable 'r' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r', moving_average_call_result_658)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
