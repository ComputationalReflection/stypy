
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import core_language_copy
2: import stypy_functions_copy
3: import functions_copy
4: 
5: '''
6: Helper functions to generate operator related nodes in the type inference AST
7: '''
8: 
9: # ##################################### OPERATORS #########################################
10: 
11: 
12: def create_binary_operator(op_name, op1, op2, line=0, column=0):
13:     localization = stypy_functions_copy.create_localization(line, column)
14: 
15:     binop_func = core_language_copy.create_Name("operator")
16:     binop = functions_copy.create_call(binop_func, [localization, op_name, op1, op2])
17: 
18:     return binop
19: 
20: 
21: def create_unary_operator(op_name, op, line=0, column=0):
22:     localization = stypy_functions_copy.create_localization(line, column)
23: 
24:     unop_func = core_language_copy.create_Name("operator")
25:     unop = functions_copy.create_call(unop_func, [localization, op_name, op])
26: 
27:     return unop
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import core_language_copy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16750 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'core_language_copy')

if (type(import_16750) is not StypyTypeError):

    if (import_16750 != 'pyd_module'):
        __import__(import_16750)
        sys_modules_16751 = sys.modules[import_16750]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'core_language_copy', sys_modules_16751.module_type_store, module_type_store)
    else:
        import core_language_copy

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'core_language_copy', core_language_copy, module_type_store)

else:
    # Assigning a type to the variable 'core_language_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'core_language_copy', import_16750)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import stypy_functions_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16752 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_functions_copy')

if (type(import_16752) is not StypyTypeError):

    if (import_16752 != 'pyd_module'):
        __import__(import_16752)
        sys_modules_16753 = sys.modules[import_16752]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_functions_copy', sys_modules_16753.module_type_store, module_type_store)
    else:
        import stypy_functions_copy

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_functions_copy', stypy_functions_copy, module_type_store)

else:
    # Assigning a type to the variable 'stypy_functions_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_functions_copy', import_16752)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import functions_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16754 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'functions_copy')

if (type(import_16754) is not StypyTypeError):

    if (import_16754 != 'pyd_module'):
        __import__(import_16754)
        sys_modules_16755 = sys.modules[import_16754]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'functions_copy', sys_modules_16755.module_type_store, module_type_store)
    else:
        import functions_copy

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'functions_copy', functions_copy, module_type_store)

else:
    # Assigning a type to the variable 'functions_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'functions_copy', import_16754)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_16756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nHelper functions to generate operator related nodes in the type inference AST\n')

@norecursion
def create_binary_operator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_16757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 51), 'int')
    int_16758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 61), 'int')
    defaults = [int_16757, int_16758]
    # Create a new context for function 'create_binary_operator'
    module_type_store = module_type_store.open_function_context('create_binary_operator', 12, 0, False)
    
    # Passed parameters checking function
    create_binary_operator.stypy_localization = localization
    create_binary_operator.stypy_type_of_self = None
    create_binary_operator.stypy_type_store = module_type_store
    create_binary_operator.stypy_function_name = 'create_binary_operator'
    create_binary_operator.stypy_param_names_list = ['op_name', 'op1', 'op2', 'line', 'column']
    create_binary_operator.stypy_varargs_param_name = None
    create_binary_operator.stypy_kwargs_param_name = None
    create_binary_operator.stypy_call_defaults = defaults
    create_binary_operator.stypy_call_varargs = varargs
    create_binary_operator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_binary_operator', ['op_name', 'op1', 'op2', 'line', 'column'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_binary_operator', localization, ['op_name', 'op1', 'op2', 'line', 'column'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_binary_operator(...)' code ##################

    
    # Assigning a Call to a Name (line 13):
    
    # Call to create_localization(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'line' (line 13)
    line_16761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 60), 'line', False)
    # Getting the type of 'column' (line 13)
    column_16762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 66), 'column', False)
    # Processing the call keyword arguments (line 13)
    kwargs_16763 = {}
    # Getting the type of 'stypy_functions_copy' (line 13)
    stypy_functions_copy_16759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'stypy_functions_copy', False)
    # Obtaining the member 'create_localization' of a type (line 13)
    create_localization_16760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 19), stypy_functions_copy_16759, 'create_localization')
    # Calling create_localization(args, kwargs) (line 13)
    create_localization_call_result_16764 = invoke(stypy.reporting.localization.Localization(__file__, 13, 19), create_localization_16760, *[line_16761, column_16762], **kwargs_16763)
    
    # Assigning a type to the variable 'localization' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'localization', create_localization_call_result_16764)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to create_Name(...): (line 15)
    # Processing the call arguments (line 15)
    str_16767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'str', 'operator')
    # Processing the call keyword arguments (line 15)
    kwargs_16768 = {}
    # Getting the type of 'core_language_copy' (line 15)
    core_language_copy_16765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 15)
    create_Name_16766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 17), core_language_copy_16765, 'create_Name')
    # Calling create_Name(args, kwargs) (line 15)
    create_Name_call_result_16769 = invoke(stypy.reporting.localization.Localization(__file__, 15, 17), create_Name_16766, *[str_16767], **kwargs_16768)
    
    # Assigning a type to the variable 'binop_func' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'binop_func', create_Name_call_result_16769)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to create_call(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'binop_func' (line 16)
    binop_func_16772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 39), 'binop_func', False)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_16773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    # Getting the type of 'localization' (line 16)
    localization_16774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 52), 'localization', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 51), list_16773, localization_16774)
    # Adding element type (line 16)
    # Getting the type of 'op_name' (line 16)
    op_name_16775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 66), 'op_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 51), list_16773, op_name_16775)
    # Adding element type (line 16)
    # Getting the type of 'op1' (line 16)
    op1_16776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 75), 'op1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 51), list_16773, op1_16776)
    # Adding element type (line 16)
    # Getting the type of 'op2' (line 16)
    op2_16777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 80), 'op2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 51), list_16773, op2_16777)
    
    # Processing the call keyword arguments (line 16)
    kwargs_16778 = {}
    # Getting the type of 'functions_copy' (line 16)
    functions_copy_16770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'functions_copy', False)
    # Obtaining the member 'create_call' of a type (line 16)
    create_call_16771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 12), functions_copy_16770, 'create_call')
    # Calling create_call(args, kwargs) (line 16)
    create_call_call_result_16779 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), create_call_16771, *[binop_func_16772, list_16773], **kwargs_16778)
    
    # Assigning a type to the variable 'binop' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'binop', create_call_call_result_16779)
    # Getting the type of 'binop' (line 18)
    binop_16780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'binop')
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', binop_16780)
    
    # ################# End of 'create_binary_operator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_binary_operator' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_16781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16781)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_binary_operator'
    return stypy_return_type_16781

# Assigning a type to the variable 'create_binary_operator' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'create_binary_operator', create_binary_operator)

@norecursion
def create_unary_operator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_16782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 44), 'int')
    int_16783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 54), 'int')
    defaults = [int_16782, int_16783]
    # Create a new context for function 'create_unary_operator'
    module_type_store = module_type_store.open_function_context('create_unary_operator', 21, 0, False)
    
    # Passed parameters checking function
    create_unary_operator.stypy_localization = localization
    create_unary_operator.stypy_type_of_self = None
    create_unary_operator.stypy_type_store = module_type_store
    create_unary_operator.stypy_function_name = 'create_unary_operator'
    create_unary_operator.stypy_param_names_list = ['op_name', 'op', 'line', 'column']
    create_unary_operator.stypy_varargs_param_name = None
    create_unary_operator.stypy_kwargs_param_name = None
    create_unary_operator.stypy_call_defaults = defaults
    create_unary_operator.stypy_call_varargs = varargs
    create_unary_operator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_unary_operator', ['op_name', 'op', 'line', 'column'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_unary_operator', localization, ['op_name', 'op', 'line', 'column'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_unary_operator(...)' code ##################

    
    # Assigning a Call to a Name (line 22):
    
    # Call to create_localization(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'line' (line 22)
    line_16786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 60), 'line', False)
    # Getting the type of 'column' (line 22)
    column_16787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 66), 'column', False)
    # Processing the call keyword arguments (line 22)
    kwargs_16788 = {}
    # Getting the type of 'stypy_functions_copy' (line 22)
    stypy_functions_copy_16784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'stypy_functions_copy', False)
    # Obtaining the member 'create_localization' of a type (line 22)
    create_localization_16785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 19), stypy_functions_copy_16784, 'create_localization')
    # Calling create_localization(args, kwargs) (line 22)
    create_localization_call_result_16789 = invoke(stypy.reporting.localization.Localization(__file__, 22, 19), create_localization_16785, *[line_16786, column_16787], **kwargs_16788)
    
    # Assigning a type to the variable 'localization' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'localization', create_localization_call_result_16789)
    
    # Assigning a Call to a Name (line 24):
    
    # Call to create_Name(...): (line 24)
    # Processing the call arguments (line 24)
    str_16792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 47), 'str', 'operator')
    # Processing the call keyword arguments (line 24)
    kwargs_16793 = {}
    # Getting the type of 'core_language_copy' (line 24)
    core_language_copy_16790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 24)
    create_Name_16791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), core_language_copy_16790, 'create_Name')
    # Calling create_Name(args, kwargs) (line 24)
    create_Name_call_result_16794 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), create_Name_16791, *[str_16792], **kwargs_16793)
    
    # Assigning a type to the variable 'unop_func' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'unop_func', create_Name_call_result_16794)
    
    # Assigning a Call to a Name (line 25):
    
    # Call to create_call(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'unop_func' (line 25)
    unop_func_16797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 38), 'unop_func', False)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_16798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    # Getting the type of 'localization' (line 25)
    localization_16799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 50), 'localization', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 49), list_16798, localization_16799)
    # Adding element type (line 25)
    # Getting the type of 'op_name' (line 25)
    op_name_16800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 64), 'op_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 49), list_16798, op_name_16800)
    # Adding element type (line 25)
    # Getting the type of 'op' (line 25)
    op_16801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 73), 'op', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 49), list_16798, op_16801)
    
    # Processing the call keyword arguments (line 25)
    kwargs_16802 = {}
    # Getting the type of 'functions_copy' (line 25)
    functions_copy_16795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'functions_copy', False)
    # Obtaining the member 'create_call' of a type (line 25)
    create_call_16796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), functions_copy_16795, 'create_call')
    # Calling create_call(args, kwargs) (line 25)
    create_call_call_result_16803 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), create_call_16796, *[unop_func_16797, list_16798], **kwargs_16802)
    
    # Assigning a type to the variable 'unop' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'unop', create_call_call_result_16803)
    # Getting the type of 'unop' (line 27)
    unop_16804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'unop')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type', unop_16804)
    
    # ################# End of 'create_unary_operator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_unary_operator' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_16805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16805)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_unary_operator'
    return stypy_return_type_16805

# Assigning a type to the variable 'create_unary_operator' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'create_unary_operator', create_unary_operator)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
