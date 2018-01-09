
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # Author: Pearu Peterson, March 2002
3: #
4: 
5: from __future__ import division, print_function, absolute_import
6: 
7: __all__ = ['get_flinalg_funcs']
8: 
9: # The following ensures that possibly missing flavor (C or Fortran) is
10: # replaced with the available one. If none is available, exception
11: # is raised at the first attempt to use the resources.
12: try:
13:     from . import _flinalg
14: except ImportError:
15:     _flinalg = None
16: #    from numpy.distutils.misc_util import PostponedException
17: #    _flinalg = PostponedException()
18: #    print _flinalg.__doc__
19:     has_column_major_storage = lambda a:0
20: 
21: 
22: def has_column_major_storage(arr):
23:     return arr.flags['FORTRAN']
24: 
25: _type_conv = {'f':'s', 'd':'d', 'F':'c', 'D':'z'}  # 'd' will be default for 'i',..
26: 
27: 
28: def get_flinalg_funcs(names,arrays=(),debug=0):
29:     '''Return optimal available _flinalg function objects with
30:     names. arrays are used to determine optimal prefix.'''
31:     ordering = []
32:     for i in range(len(arrays)):
33:         t = arrays[i].dtype.char
34:         if t not in _type_conv:
35:             t = 'd'
36:         ordering.append((t,i))
37:     if ordering:
38:         ordering.sort()
39:         required_prefix = _type_conv[ordering[0][0]]
40:     else:
41:         required_prefix = 'd'
42:     # Some routines may require special treatment.
43:     # Handle them here before the default lookup.
44: 
45:     # Default lookup:
46:     if ordering and has_column_major_storage(arrays[ordering[0][1]]):
47:         suffix1,suffix2 = '_c','_r'
48:     else:
49:         suffix1,suffix2 = '_r','_c'
50: 
51:     funcs = []
52:     for name in names:
53:         func_name = required_prefix + name
54:         func = getattr(_flinalg,func_name+suffix1,
55:                        getattr(_flinalg,func_name+suffix2,None))
56:         funcs.append(func)
57:     return tuple(funcs)
58: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['get_flinalg_funcs']
module_type_store.set_exportable_members(['get_flinalg_funcs'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_20665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_20666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'get_flinalg_funcs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_20665, str_20666)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_20665)


# SSA begins for try-except statement (line 12)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))

# 'from scipy.linalg import _flinalg' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_20667 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'scipy.linalg')

if (type(import_20667) is not StypyTypeError):

    if (import_20667 != 'pyd_module'):
        __import__(import_20667)
        sys_modules_20668 = sys.modules[import_20667]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'scipy.linalg', sys_modules_20668.module_type_store, module_type_store, ['_flinalg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 4), __file__, sys_modules_20668, sys_modules_20668.module_type_store, module_type_store)
    else:
        from scipy.linalg import _flinalg

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'scipy.linalg', None, module_type_store, ['_flinalg'], [_flinalg])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'scipy.linalg', import_20667)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

# SSA branch for the except part of a try statement (line 12)
# SSA branch for the except 'ImportError' branch of a try statement (line 12)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 15):

# Assigning a Name to a Name (line 15):
# Getting the type of 'None' (line 15)
None_20669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'None')
# Assigning a type to the variable '_flinalg' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), '_flinalg', None_20669)

# Assigning a Lambda to a Name (line 19):

# Assigning a Lambda to a Name (line 19):

@norecursion
def _stypy_temp_lambda_7(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_7'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_7', 19, 31, True)
    # Passed parameters checking function
    _stypy_temp_lambda_7.stypy_localization = localization
    _stypy_temp_lambda_7.stypy_type_of_self = None
    _stypy_temp_lambda_7.stypy_type_store = module_type_store
    _stypy_temp_lambda_7.stypy_function_name = '_stypy_temp_lambda_7'
    _stypy_temp_lambda_7.stypy_param_names_list = ['a']
    _stypy_temp_lambda_7.stypy_varargs_param_name = None
    _stypy_temp_lambda_7.stypy_kwargs_param_name = None
    _stypy_temp_lambda_7.stypy_call_defaults = defaults
    _stypy_temp_lambda_7.stypy_call_varargs = varargs
    _stypy_temp_lambda_7.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_7', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_7', ['a'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    int_20670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 40), 'int')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'stypy_return_type', int_20670)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_7' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_20671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20671)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_7'
    return stypy_return_type_20671

# Assigning a type to the variable '_stypy_temp_lambda_7' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), '_stypy_temp_lambda_7', _stypy_temp_lambda_7)
# Getting the type of '_stypy_temp_lambda_7' (line 19)
_stypy_temp_lambda_7_20672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), '_stypy_temp_lambda_7')
# Assigning a type to the variable 'has_column_major_storage' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'has_column_major_storage', _stypy_temp_lambda_7_20672)
# SSA join for try-except statement (line 12)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def has_column_major_storage(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'has_column_major_storage'
    module_type_store = module_type_store.open_function_context('has_column_major_storage', 22, 0, False)
    
    # Passed parameters checking function
    has_column_major_storage.stypy_localization = localization
    has_column_major_storage.stypy_type_of_self = None
    has_column_major_storage.stypy_type_store = module_type_store
    has_column_major_storage.stypy_function_name = 'has_column_major_storage'
    has_column_major_storage.stypy_param_names_list = ['arr']
    has_column_major_storage.stypy_varargs_param_name = None
    has_column_major_storage.stypy_kwargs_param_name = None
    has_column_major_storage.stypy_call_defaults = defaults
    has_column_major_storage.stypy_call_varargs = varargs
    has_column_major_storage.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'has_column_major_storage', ['arr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'has_column_major_storage', localization, ['arr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'has_column_major_storage(...)' code ##################

    
    # Obtaining the type of the subscript
    str_20673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'str', 'FORTRAN')
    # Getting the type of 'arr' (line 23)
    arr_20674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'arr')
    # Obtaining the member 'flags' of a type (line 23)
    flags_20675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), arr_20674, 'flags')
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___20676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), flags_20675, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_20677 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), getitem___20676, str_20673)
    
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', subscript_call_result_20677)
    
    # ################# End of 'has_column_major_storage(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'has_column_major_storage' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_20678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20678)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'has_column_major_storage'
    return stypy_return_type_20678

# Assigning a type to the variable 'has_column_major_storage' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'has_column_major_storage', has_column_major_storage)

# Assigning a Dict to a Name (line 25):

# Assigning a Dict to a Name (line 25):

# Obtaining an instance of the builtin type 'dict' (line 25)
dict_20679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 25)
# Adding element type (key, value) (line 25)
str_20680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 14), 'str', 'f')
str_20681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'str', 's')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), dict_20679, (str_20680, str_20681))
# Adding element type (key, value) (line 25)
str_20682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'str', 'd')
str_20683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), dict_20679, (str_20682, str_20683))
# Adding element type (key, value) (line 25)
str_20684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 32), 'str', 'F')
str_20685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 36), 'str', 'c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), dict_20679, (str_20684, str_20685))
# Adding element type (key, value) (line 25)
str_20686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 41), 'str', 'D')
str_20687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 45), 'str', 'z')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), dict_20679, (str_20686, str_20687))

# Assigning a type to the variable '_type_conv' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '_type_conv', dict_20679)

@norecursion
def get_flinalg_funcs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 28)
    tuple_20688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 28)
    
    int_20689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 44), 'int')
    defaults = [tuple_20688, int_20689]
    # Create a new context for function 'get_flinalg_funcs'
    module_type_store = module_type_store.open_function_context('get_flinalg_funcs', 28, 0, False)
    
    # Passed parameters checking function
    get_flinalg_funcs.stypy_localization = localization
    get_flinalg_funcs.stypy_type_of_self = None
    get_flinalg_funcs.stypy_type_store = module_type_store
    get_flinalg_funcs.stypy_function_name = 'get_flinalg_funcs'
    get_flinalg_funcs.stypy_param_names_list = ['names', 'arrays', 'debug']
    get_flinalg_funcs.stypy_varargs_param_name = None
    get_flinalg_funcs.stypy_kwargs_param_name = None
    get_flinalg_funcs.stypy_call_defaults = defaults
    get_flinalg_funcs.stypy_call_varargs = varargs
    get_flinalg_funcs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_flinalg_funcs', ['names', 'arrays', 'debug'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_flinalg_funcs', localization, ['names', 'arrays', 'debug'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_flinalg_funcs(...)' code ##################

    str_20690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', 'Return optimal available _flinalg function objects with\n    names. arrays are used to determine optimal prefix.')
    
    # Assigning a List to a Name (line 31):
    
    # Assigning a List to a Name (line 31):
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_20691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    
    # Assigning a type to the variable 'ordering' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'ordering', list_20691)
    
    
    # Call to range(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to len(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'arrays' (line 32)
    arrays_20694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'arrays', False)
    # Processing the call keyword arguments (line 32)
    kwargs_20695 = {}
    # Getting the type of 'len' (line 32)
    len_20693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'len', False)
    # Calling len(args, kwargs) (line 32)
    len_call_result_20696 = invoke(stypy.reporting.localization.Localization(__file__, 32, 19), len_20693, *[arrays_20694], **kwargs_20695)
    
    # Processing the call keyword arguments (line 32)
    kwargs_20697 = {}
    # Getting the type of 'range' (line 32)
    range_20692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 13), 'range', False)
    # Calling range(args, kwargs) (line 32)
    range_call_result_20698 = invoke(stypy.reporting.localization.Localization(__file__, 32, 13), range_20692, *[len_call_result_20696], **kwargs_20697)
    
    # Testing the type of a for loop iterable (line 32)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 32, 4), range_call_result_20698)
    # Getting the type of the for loop variable (line 32)
    for_loop_var_20699 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 32, 4), range_call_result_20698)
    # Assigning a type to the variable 'i' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'i', for_loop_var_20699)
    # SSA begins for a for statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Attribute to a Name (line 33):
    
    # Assigning a Attribute to a Name (line 33):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 33)
    i_20700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'i')
    # Getting the type of 'arrays' (line 33)
    arrays_20701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'arrays')
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___20702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), arrays_20701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_20703 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), getitem___20702, i_20700)
    
    # Obtaining the member 'dtype' of a type (line 33)
    dtype_20704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), subscript_call_result_20703, 'dtype')
    # Obtaining the member 'char' of a type (line 33)
    char_20705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), dtype_20704, 'char')
    # Assigning a type to the variable 't' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 't', char_20705)
    
    
    # Getting the type of 't' (line 34)
    t_20706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 't')
    # Getting the type of '_type_conv' (line 34)
    _type_conv_20707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), '_type_conv')
    # Applying the binary operator 'notin' (line 34)
    result_contains_20708 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 11), 'notin', t_20706, _type_conv_20707)
    
    # Testing the type of an if condition (line 34)
    if_condition_20709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 8), result_contains_20708)
    # Assigning a type to the variable 'if_condition_20709' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'if_condition_20709', if_condition_20709)
    # SSA begins for if statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 35):
    
    # Assigning a Str to a Name (line 35):
    str_20710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'str', 'd')
    # Assigning a type to the variable 't' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 't', str_20710)
    # SSA join for if statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_20713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    # Getting the type of 't' (line 36)
    t_20714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), tuple_20713, t_20714)
    # Adding element type (line 36)
    # Getting the type of 'i' (line 36)
    i_20715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), tuple_20713, i_20715)
    
    # Processing the call keyword arguments (line 36)
    kwargs_20716 = {}
    # Getting the type of 'ordering' (line 36)
    ordering_20711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'ordering', False)
    # Obtaining the member 'append' of a type (line 36)
    append_20712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), ordering_20711, 'append')
    # Calling append(args, kwargs) (line 36)
    append_call_result_20717 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), append_20712, *[tuple_20713], **kwargs_20716)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ordering' (line 37)
    ordering_20718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'ordering')
    # Testing the type of an if condition (line 37)
    if_condition_20719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 4), ordering_20718)
    # Assigning a type to the variable 'if_condition_20719' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'if_condition_20719', if_condition_20719)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to sort(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_20722 = {}
    # Getting the type of 'ordering' (line 38)
    ordering_20720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'ordering', False)
    # Obtaining the member 'sort' of a type (line 38)
    sort_20721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), ordering_20720, 'sort')
    # Calling sort(args, kwargs) (line 38)
    sort_call_result_20723 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), sort_20721, *[], **kwargs_20722)
    
    
    # Assigning a Subscript to a Name (line 39):
    
    # Assigning a Subscript to a Name (line 39):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_20724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 49), 'int')
    
    # Obtaining the type of the subscript
    int_20725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'int')
    # Getting the type of 'ordering' (line 39)
    ordering_20726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'ordering')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___20727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 37), ordering_20726, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_20728 = invoke(stypy.reporting.localization.Localization(__file__, 39, 37), getitem___20727, int_20725)
    
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___20729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 37), subscript_call_result_20728, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_20730 = invoke(stypy.reporting.localization.Localization(__file__, 39, 37), getitem___20729, int_20724)
    
    # Getting the type of '_type_conv' (line 39)
    _type_conv_20731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), '_type_conv')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___20732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 26), _type_conv_20731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_20733 = invoke(stypy.reporting.localization.Localization(__file__, 39, 26), getitem___20732, subscript_call_result_20730)
    
    # Assigning a type to the variable 'required_prefix' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'required_prefix', subscript_call_result_20733)
    # SSA branch for the else part of an if statement (line 37)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 41):
    
    # Assigning a Str to a Name (line 41):
    str_20734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'str', 'd')
    # Assigning a type to the variable 'required_prefix' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'required_prefix', str_20734)
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'ordering' (line 46)
    ordering_20735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'ordering')
    
    # Call to has_column_major_storage(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_20737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 64), 'int')
    
    # Obtaining the type of the subscript
    int_20738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 61), 'int')
    # Getting the type of 'ordering' (line 46)
    ordering_20739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 52), 'ordering', False)
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___20740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 52), ordering_20739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_20741 = invoke(stypy.reporting.localization.Localization(__file__, 46, 52), getitem___20740, int_20738)
    
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___20742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 52), subscript_call_result_20741, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_20743 = invoke(stypy.reporting.localization.Localization(__file__, 46, 52), getitem___20742, int_20737)
    
    # Getting the type of 'arrays' (line 46)
    arrays_20744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'arrays', False)
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___20745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 45), arrays_20744, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_20746 = invoke(stypy.reporting.localization.Localization(__file__, 46, 45), getitem___20745, subscript_call_result_20743)
    
    # Processing the call keyword arguments (line 46)
    kwargs_20747 = {}
    # Getting the type of 'has_column_major_storage' (line 46)
    has_column_major_storage_20736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'has_column_major_storage', False)
    # Calling has_column_major_storage(args, kwargs) (line 46)
    has_column_major_storage_call_result_20748 = invoke(stypy.reporting.localization.Localization(__file__, 46, 20), has_column_major_storage_20736, *[subscript_call_result_20746], **kwargs_20747)
    
    # Applying the binary operator 'and' (line 46)
    result_and_keyword_20749 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 7), 'and', ordering_20735, has_column_major_storage_call_result_20748)
    
    # Testing the type of an if condition (line 46)
    if_condition_20750 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 4), result_and_keyword_20749)
    # Assigning a type to the variable 'if_condition_20750' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'if_condition_20750', if_condition_20750)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 47):
    
    # Assigning a Str to a Name (line 47):
    str_20751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 26), 'str', '_c')
    # Assigning a type to the variable 'tuple_assignment_20661' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_assignment_20661', str_20751)
    
    # Assigning a Str to a Name (line 47):
    str_20752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'str', '_r')
    # Assigning a type to the variable 'tuple_assignment_20662' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_assignment_20662', str_20752)
    
    # Assigning a Name to a Name (line 47):
    # Getting the type of 'tuple_assignment_20661' (line 47)
    tuple_assignment_20661_20753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_assignment_20661')
    # Assigning a type to the variable 'suffix1' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'suffix1', tuple_assignment_20661_20753)
    
    # Assigning a Name to a Name (line 47):
    # Getting the type of 'tuple_assignment_20662' (line 47)
    tuple_assignment_20662_20754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_assignment_20662')
    # Assigning a type to the variable 'suffix2' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'suffix2', tuple_assignment_20662_20754)
    # SSA branch for the else part of an if statement (line 46)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 49):
    
    # Assigning a Str to a Name (line 49):
    str_20755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 26), 'str', '_r')
    # Assigning a type to the variable 'tuple_assignment_20663' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_20663', str_20755)
    
    # Assigning a Str to a Name (line 49):
    str_20756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 31), 'str', '_c')
    # Assigning a type to the variable 'tuple_assignment_20664' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_20664', str_20756)
    
    # Assigning a Name to a Name (line 49):
    # Getting the type of 'tuple_assignment_20663' (line 49)
    tuple_assignment_20663_20757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_20663')
    # Assigning a type to the variable 'suffix1' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'suffix1', tuple_assignment_20663_20757)
    
    # Assigning a Name to a Name (line 49):
    # Getting the type of 'tuple_assignment_20664' (line 49)
    tuple_assignment_20664_20758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_20664')
    # Assigning a type to the variable 'suffix2' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'suffix2', tuple_assignment_20664_20758)
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 51):
    
    # Assigning a List to a Name (line 51):
    
    # Obtaining an instance of the builtin type 'list' (line 51)
    list_20759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 51)
    
    # Assigning a type to the variable 'funcs' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'funcs', list_20759)
    
    # Getting the type of 'names' (line 52)
    names_20760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'names')
    # Testing the type of a for loop iterable (line 52)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 4), names_20760)
    # Getting the type of the for loop variable (line 52)
    for_loop_var_20761 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 4), names_20760)
    # Assigning a type to the variable 'name' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'name', for_loop_var_20761)
    # SSA begins for a for statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 53):
    
    # Assigning a BinOp to a Name (line 53):
    # Getting the type of 'required_prefix' (line 53)
    required_prefix_20762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'required_prefix')
    # Getting the type of 'name' (line 53)
    name_20763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 38), 'name')
    # Applying the binary operator '+' (line 53)
    result_add_20764 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 20), '+', required_prefix_20762, name_20763)
    
    # Assigning a type to the variable 'func_name' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'func_name', result_add_20764)
    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to getattr(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of '_flinalg' (line 54)
    _flinalg_20766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), '_flinalg', False)
    # Getting the type of 'func_name' (line 54)
    func_name_20767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 32), 'func_name', False)
    # Getting the type of 'suffix1' (line 54)
    suffix1_20768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'suffix1', False)
    # Applying the binary operator '+' (line 54)
    result_add_20769 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 32), '+', func_name_20767, suffix1_20768)
    
    
    # Call to getattr(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of '_flinalg' (line 55)
    _flinalg_20771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 31), '_flinalg', False)
    # Getting the type of 'func_name' (line 55)
    func_name_20772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'func_name', False)
    # Getting the type of 'suffix2' (line 55)
    suffix2_20773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 50), 'suffix2', False)
    # Applying the binary operator '+' (line 55)
    result_add_20774 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 40), '+', func_name_20772, suffix2_20773)
    
    # Getting the type of 'None' (line 55)
    None_20775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 58), 'None', False)
    # Processing the call keyword arguments (line 55)
    kwargs_20776 = {}
    # Getting the type of 'getattr' (line 55)
    getattr_20770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'getattr', False)
    # Calling getattr(args, kwargs) (line 55)
    getattr_call_result_20777 = invoke(stypy.reporting.localization.Localization(__file__, 55, 23), getattr_20770, *[_flinalg_20771, result_add_20774, None_20775], **kwargs_20776)
    
    # Processing the call keyword arguments (line 54)
    kwargs_20778 = {}
    # Getting the type of 'getattr' (line 54)
    getattr_20765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'getattr', False)
    # Calling getattr(args, kwargs) (line 54)
    getattr_call_result_20779 = invoke(stypy.reporting.localization.Localization(__file__, 54, 15), getattr_20765, *[_flinalg_20766, result_add_20769, getattr_call_result_20777], **kwargs_20778)
    
    # Assigning a type to the variable 'func' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'func', getattr_call_result_20779)
    
    # Call to append(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'func' (line 56)
    func_20782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'func', False)
    # Processing the call keyword arguments (line 56)
    kwargs_20783 = {}
    # Getting the type of 'funcs' (line 56)
    funcs_20780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'funcs', False)
    # Obtaining the member 'append' of a type (line 56)
    append_20781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), funcs_20780, 'append')
    # Calling append(args, kwargs) (line 56)
    append_call_result_20784 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), append_20781, *[func_20782], **kwargs_20783)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to tuple(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'funcs' (line 57)
    funcs_20786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'funcs', False)
    # Processing the call keyword arguments (line 57)
    kwargs_20787 = {}
    # Getting the type of 'tuple' (line 57)
    tuple_20785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 57)
    tuple_call_result_20788 = invoke(stypy.reporting.localization.Localization(__file__, 57, 11), tuple_20785, *[funcs_20786], **kwargs_20787)
    
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type', tuple_call_result_20788)
    
    # ################# End of 'get_flinalg_funcs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_flinalg_funcs' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_20789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20789)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_flinalg_funcs'
    return stypy_return_type_20789

# Assigning a type to the variable 'get_flinalg_funcs' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'get_flinalg_funcs', get_flinalg_funcs)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
