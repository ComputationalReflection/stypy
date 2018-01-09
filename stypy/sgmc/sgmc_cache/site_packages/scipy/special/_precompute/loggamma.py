
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Precompute series coefficients for log-Gamma.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: try:
5:     import mpmath
6: except ImportError:
7:     pass
8: 
9: 
10: def stirling_series(N):
11:     coeffs = []
12:     with mpmath.workdps(100):
13:         for n in range(1, N + 1):
14:             coeffs.append(mpmath.bernoulli(2*n)/(2*n*(2*n - 1)))
15:     return coeffs
16: 
17: 
18: def taylor_series_at_1(N):
19:     coeffs = []
20:     with mpmath.workdps(100):
21:         coeffs.append(-mpmath.euler)
22:         for n in range(2, N + 1):
23:             coeffs.append((-1)**n*mpmath.zeta(n)/n)
24:     return coeffs
25: 
26: 
27: def main():
28:     print(__doc__)
29:     print()
30:     stirling_coeffs = [mpmath.nstr(x, 20, min_fixed=0, max_fixed=0)
31:                        for x in stirling_series(8)[::-1]]
32:     taylor_coeffs = [mpmath.nstr(x, 20, min_fixed=0, max_fixed=0)
33:                      for x in taylor_series_at_1(23)[::-1]]
34:     print("Stirling series coefficients")
35:     print("----------------------------")
36:     print("\n".join(stirling_coeffs))
37:     print()
38:     print("Taylor series coefficients")
39:     print("--------------------------")
40:     print("\n".join(taylor_coeffs))
41:     print()
42: 
43: 
44: if __name__ == '__main__':
45:     main()
46:     
47: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_564508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Precompute series coefficients for log-Gamma.')


# SSA begins for try-except statement (line 4)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))

# 'import mpmath' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/_precompute/')
import_564509 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'mpmath')

if (type(import_564509) is not StypyTypeError):

    if (import_564509 != 'pyd_module'):
        __import__(import_564509)
        sys_modules_564510 = sys.modules[import_564509]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'mpmath', sys_modules_564510.module_type_store, module_type_store)
    else:
        import mpmath

        import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'mpmath', mpmath, module_type_store)

else:
    # Assigning a type to the variable 'mpmath' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'mpmath', import_564509)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/_precompute/')

# SSA branch for the except part of a try statement (line 4)
# SSA branch for the except 'ImportError' branch of a try statement (line 4)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 4)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def stirling_series(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'stirling_series'
    module_type_store = module_type_store.open_function_context('stirling_series', 10, 0, False)
    
    # Passed parameters checking function
    stirling_series.stypy_localization = localization
    stirling_series.stypy_type_of_self = None
    stirling_series.stypy_type_store = module_type_store
    stirling_series.stypy_function_name = 'stirling_series'
    stirling_series.stypy_param_names_list = ['N']
    stirling_series.stypy_varargs_param_name = None
    stirling_series.stypy_kwargs_param_name = None
    stirling_series.stypy_call_defaults = defaults
    stirling_series.stypy_call_varargs = varargs
    stirling_series.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'stirling_series', ['N'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'stirling_series', localization, ['N'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'stirling_series(...)' code ##################

    
    # Assigning a List to a Name (line 11):
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_564511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    
    # Assigning a type to the variable 'coeffs' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'coeffs', list_564511)
    
    # Call to workdps(...): (line 12)
    # Processing the call arguments (line 12)
    int_564514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 24), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_564515 = {}
    # Getting the type of 'mpmath' (line 12)
    mpmath_564512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'mpmath', False)
    # Obtaining the member 'workdps' of a type (line 12)
    workdps_564513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 9), mpmath_564512, 'workdps')
    # Calling workdps(args, kwargs) (line 12)
    workdps_call_result_564516 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), workdps_564513, *[int_564514], **kwargs_564515)
    
    with_564517 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 12, 9), workdps_call_result_564516, 'with parameter', '__enter__', '__exit__')

    if with_564517:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 12)
        enter___564518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 9), workdps_call_result_564516, '__enter__')
        with_enter_564519 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), enter___564518)
        
        
        # Call to range(...): (line 13)
        # Processing the call arguments (line 13)
        int_564521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'int')
        # Getting the type of 'N' (line 13)
        N_564522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 26), 'N', False)
        int_564523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 30), 'int')
        # Applying the binary operator '+' (line 13)
        result_add_564524 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 26), '+', N_564522, int_564523)
        
        # Processing the call keyword arguments (line 13)
        kwargs_564525 = {}
        # Getting the type of 'range' (line 13)
        range_564520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'range', False)
        # Calling range(args, kwargs) (line 13)
        range_call_result_564526 = invoke(stypy.reporting.localization.Localization(__file__, 13, 17), range_564520, *[int_564521, result_add_564524], **kwargs_564525)
        
        # Testing the type of a for loop iterable (line 13)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 13, 8), range_call_result_564526)
        # Getting the type of the for loop variable (line 13)
        for_loop_var_564527 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 13, 8), range_call_result_564526)
        # Assigning a type to the variable 'n' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'n', for_loop_var_564527)
        # SSA begins for a for statement (line 13)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 14)
        # Processing the call arguments (line 14)
        
        # Call to bernoulli(...): (line 14)
        # Processing the call arguments (line 14)
        int_564532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 43), 'int')
        # Getting the type of 'n' (line 14)
        n_564533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 45), 'n', False)
        # Applying the binary operator '*' (line 14)
        result_mul_564534 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 43), '*', int_564532, n_564533)
        
        # Processing the call keyword arguments (line 14)
        kwargs_564535 = {}
        # Getting the type of 'mpmath' (line 14)
        mpmath_564530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 26), 'mpmath', False)
        # Obtaining the member 'bernoulli' of a type (line 14)
        bernoulli_564531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 26), mpmath_564530, 'bernoulli')
        # Calling bernoulli(args, kwargs) (line 14)
        bernoulli_call_result_564536 = invoke(stypy.reporting.localization.Localization(__file__, 14, 26), bernoulli_564531, *[result_mul_564534], **kwargs_564535)
        
        int_564537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 49), 'int')
        # Getting the type of 'n' (line 14)
        n_564538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 51), 'n', False)
        # Applying the binary operator '*' (line 14)
        result_mul_564539 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 49), '*', int_564537, n_564538)
        
        int_564540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 54), 'int')
        # Getting the type of 'n' (line 14)
        n_564541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 56), 'n', False)
        # Applying the binary operator '*' (line 14)
        result_mul_564542 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 54), '*', int_564540, n_564541)
        
        int_564543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 60), 'int')
        # Applying the binary operator '-' (line 14)
        result_sub_564544 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 54), '-', result_mul_564542, int_564543)
        
        # Applying the binary operator '*' (line 14)
        result_mul_564545 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 52), '*', result_mul_564539, result_sub_564544)
        
        # Applying the binary operator 'div' (line 14)
        result_div_564546 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 26), 'div', bernoulli_call_result_564536, result_mul_564545)
        
        # Processing the call keyword arguments (line 14)
        kwargs_564547 = {}
        # Getting the type of 'coeffs' (line 14)
        coeffs_564528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'coeffs', False)
        # Obtaining the member 'append' of a type (line 14)
        append_564529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 12), coeffs_564528, 'append')
        # Calling append(args, kwargs) (line 14)
        append_call_result_564548 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), append_564529, *[result_div_564546], **kwargs_564547)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 12)
        exit___564549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 9), workdps_call_result_564516, '__exit__')
        with_exit_564550 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), exit___564549, None, None, None)

    # Getting the type of 'coeffs' (line 15)
    coeffs_564551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'coeffs')
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type', coeffs_564551)
    
    # ################# End of 'stirling_series(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'stirling_series' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_564552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_564552)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'stirling_series'
    return stypy_return_type_564552

# Assigning a type to the variable 'stirling_series' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stirling_series', stirling_series)

@norecursion
def taylor_series_at_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'taylor_series_at_1'
    module_type_store = module_type_store.open_function_context('taylor_series_at_1', 18, 0, False)
    
    # Passed parameters checking function
    taylor_series_at_1.stypy_localization = localization
    taylor_series_at_1.stypy_type_of_self = None
    taylor_series_at_1.stypy_type_store = module_type_store
    taylor_series_at_1.stypy_function_name = 'taylor_series_at_1'
    taylor_series_at_1.stypy_param_names_list = ['N']
    taylor_series_at_1.stypy_varargs_param_name = None
    taylor_series_at_1.stypy_kwargs_param_name = None
    taylor_series_at_1.stypy_call_defaults = defaults
    taylor_series_at_1.stypy_call_varargs = varargs
    taylor_series_at_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'taylor_series_at_1', ['N'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'taylor_series_at_1', localization, ['N'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'taylor_series_at_1(...)' code ##################

    
    # Assigning a List to a Name (line 19):
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_564553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    
    # Assigning a type to the variable 'coeffs' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'coeffs', list_564553)
    
    # Call to workdps(...): (line 20)
    # Processing the call arguments (line 20)
    int_564556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'int')
    # Processing the call keyword arguments (line 20)
    kwargs_564557 = {}
    # Getting the type of 'mpmath' (line 20)
    mpmath_564554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 9), 'mpmath', False)
    # Obtaining the member 'workdps' of a type (line 20)
    workdps_564555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 9), mpmath_564554, 'workdps')
    # Calling workdps(args, kwargs) (line 20)
    workdps_call_result_564558 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), workdps_564555, *[int_564556], **kwargs_564557)
    
    with_564559 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 20, 9), workdps_call_result_564558, 'with parameter', '__enter__', '__exit__')

    if with_564559:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 20)
        enter___564560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 9), workdps_call_result_564558, '__enter__')
        with_enter_564561 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), enter___564560)
        
        # Call to append(...): (line 21)
        # Processing the call arguments (line 21)
        
        # Getting the type of 'mpmath' (line 21)
        mpmath_564564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'mpmath', False)
        # Obtaining the member 'euler' of a type (line 21)
        euler_564565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 23), mpmath_564564, 'euler')
        # Applying the 'usub' unary operator (line 21)
        result___neg___564566 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 22), 'usub', euler_564565)
        
        # Processing the call keyword arguments (line 21)
        kwargs_564567 = {}
        # Getting the type of 'coeffs' (line 21)
        coeffs_564562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'coeffs', False)
        # Obtaining the member 'append' of a type (line 21)
        append_564563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), coeffs_564562, 'append')
        # Calling append(args, kwargs) (line 21)
        append_call_result_564568 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), append_564563, *[result___neg___564566], **kwargs_564567)
        
        
        
        # Call to range(...): (line 22)
        # Processing the call arguments (line 22)
        int_564570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'int')
        # Getting the type of 'N' (line 22)
        N_564571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'N', False)
        int_564572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'int')
        # Applying the binary operator '+' (line 22)
        result_add_564573 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 26), '+', N_564571, int_564572)
        
        # Processing the call keyword arguments (line 22)
        kwargs_564574 = {}
        # Getting the type of 'range' (line 22)
        range_564569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'range', False)
        # Calling range(args, kwargs) (line 22)
        range_call_result_564575 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), range_564569, *[int_564570, result_add_564573], **kwargs_564574)
        
        # Testing the type of a for loop iterable (line 22)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 22, 8), range_call_result_564575)
        # Getting the type of the for loop variable (line 22)
        for_loop_var_564576 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 22, 8), range_call_result_564575)
        # Assigning a type to the variable 'n' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'n', for_loop_var_564576)
        # SSA begins for a for statement (line 22)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 23)
        # Processing the call arguments (line 23)
        int_564579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'int')
        # Getting the type of 'n' (line 23)
        n_564580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 32), 'n', False)
        # Applying the binary operator '**' (line 23)
        result_pow_564581 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 26), '**', int_564579, n_564580)
        
        
        # Call to zeta(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'n' (line 23)
        n_564584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 46), 'n', False)
        # Processing the call keyword arguments (line 23)
        kwargs_564585 = {}
        # Getting the type of 'mpmath' (line 23)
        mpmath_564582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'mpmath', False)
        # Obtaining the member 'zeta' of a type (line 23)
        zeta_564583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 34), mpmath_564582, 'zeta')
        # Calling zeta(args, kwargs) (line 23)
        zeta_call_result_564586 = invoke(stypy.reporting.localization.Localization(__file__, 23, 34), zeta_564583, *[n_564584], **kwargs_564585)
        
        # Applying the binary operator '*' (line 23)
        result_mul_564587 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 26), '*', result_pow_564581, zeta_call_result_564586)
        
        # Getting the type of 'n' (line 23)
        n_564588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 49), 'n', False)
        # Applying the binary operator 'div' (line 23)
        result_div_564589 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 48), 'div', result_mul_564587, n_564588)
        
        # Processing the call keyword arguments (line 23)
        kwargs_564590 = {}
        # Getting the type of 'coeffs' (line 23)
        coeffs_564577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'coeffs', False)
        # Obtaining the member 'append' of a type (line 23)
        append_564578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), coeffs_564577, 'append')
        # Calling append(args, kwargs) (line 23)
        append_call_result_564591 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), append_564578, *[result_div_564589], **kwargs_564590)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 20)
        exit___564592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 9), workdps_call_result_564558, '__exit__')
        with_exit_564593 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), exit___564592, None, None, None)

    # Getting the type of 'coeffs' (line 24)
    coeffs_564594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'coeffs')
    # Assigning a type to the variable 'stypy_return_type' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type', coeffs_564594)
    
    # ################# End of 'taylor_series_at_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'taylor_series_at_1' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_564595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_564595)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'taylor_series_at_1'
    return stypy_return_type_564595

# Assigning a type to the variable 'taylor_series_at_1' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'taylor_series_at_1', taylor_series_at_1)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 27, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = []
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    # Call to print(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of '__doc__' (line 28)
    doc___564597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), '__doc__', False)
    # Processing the call keyword arguments (line 28)
    kwargs_564598 = {}
    # Getting the type of 'print' (line 28)
    print_564596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'print', False)
    # Calling print(args, kwargs) (line 28)
    print_call_result_564599 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), print_564596, *[doc___564597], **kwargs_564598)
    
    
    # Call to print(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_564601 = {}
    # Getting the type of 'print' (line 29)
    print_564600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'print', False)
    # Calling print(args, kwargs) (line 29)
    print_call_result_564602 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), print_564600, *[], **kwargs_564601)
    
    
    # Assigning a ListComp to a Name (line 30):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_564613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 53), 'int')
    slice_564614 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 31, 32), None, None, int_564613)
    
    # Call to stirling_series(...): (line 31)
    # Processing the call arguments (line 31)
    int_564616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 48), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_564617 = {}
    # Getting the type of 'stirling_series' (line 31)
    stirling_series_564615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 32), 'stirling_series', False)
    # Calling stirling_series(args, kwargs) (line 31)
    stirling_series_call_result_564618 = invoke(stypy.reporting.localization.Localization(__file__, 31, 32), stirling_series_564615, *[int_564616], **kwargs_564617)
    
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___564619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 32), stirling_series_call_result_564618, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_564620 = invoke(stypy.reporting.localization.Localization(__file__, 31, 32), getitem___564619, slice_564614)
    
    comprehension_564621 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 23), subscript_call_result_564620)
    # Assigning a type to the variable 'x' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'x', comprehension_564621)
    
    # Call to nstr(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'x' (line 30)
    x_564605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 35), 'x', False)
    int_564606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'int')
    # Processing the call keyword arguments (line 30)
    int_564607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 52), 'int')
    keyword_564608 = int_564607
    int_564609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 65), 'int')
    keyword_564610 = int_564609
    kwargs_564611 = {'max_fixed': keyword_564610, 'min_fixed': keyword_564608}
    # Getting the type of 'mpmath' (line 30)
    mpmath_564603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'mpmath', False)
    # Obtaining the member 'nstr' of a type (line 30)
    nstr_564604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 23), mpmath_564603, 'nstr')
    # Calling nstr(args, kwargs) (line 30)
    nstr_call_result_564612 = invoke(stypy.reporting.localization.Localization(__file__, 30, 23), nstr_564604, *[x_564605, int_564606], **kwargs_564611)
    
    list_564622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 23), list_564622, nstr_call_result_564612)
    # Assigning a type to the variable 'stirling_coeffs' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stirling_coeffs', list_564622)
    
    # Assigning a ListComp to a Name (line 32):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_564633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 55), 'int')
    slice_564634 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 33, 30), None, None, int_564633)
    
    # Call to taylor_series_at_1(...): (line 33)
    # Processing the call arguments (line 33)
    int_564636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 49), 'int')
    # Processing the call keyword arguments (line 33)
    kwargs_564637 = {}
    # Getting the type of 'taylor_series_at_1' (line 33)
    taylor_series_at_1_564635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 30), 'taylor_series_at_1', False)
    # Calling taylor_series_at_1(args, kwargs) (line 33)
    taylor_series_at_1_call_result_564638 = invoke(stypy.reporting.localization.Localization(__file__, 33, 30), taylor_series_at_1_564635, *[int_564636], **kwargs_564637)
    
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___564639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 30), taylor_series_at_1_call_result_564638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_564640 = invoke(stypy.reporting.localization.Localization(__file__, 33, 30), getitem___564639, slice_564634)
    
    comprehension_564641 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 21), subscript_call_result_564640)
    # Assigning a type to the variable 'x' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'x', comprehension_564641)
    
    # Call to nstr(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'x' (line 32)
    x_564625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'x', False)
    int_564626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 36), 'int')
    # Processing the call keyword arguments (line 32)
    int_564627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 50), 'int')
    keyword_564628 = int_564627
    int_564629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 63), 'int')
    keyword_564630 = int_564629
    kwargs_564631 = {'max_fixed': keyword_564630, 'min_fixed': keyword_564628}
    # Getting the type of 'mpmath' (line 32)
    mpmath_564623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'mpmath', False)
    # Obtaining the member 'nstr' of a type (line 32)
    nstr_564624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 21), mpmath_564623, 'nstr')
    # Calling nstr(args, kwargs) (line 32)
    nstr_call_result_564632 = invoke(stypy.reporting.localization.Localization(__file__, 32, 21), nstr_564624, *[x_564625, int_564626], **kwargs_564631)
    
    list_564642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 21), list_564642, nstr_call_result_564632)
    # Assigning a type to the variable 'taylor_coeffs' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'taylor_coeffs', list_564642)
    
    # Call to print(...): (line 34)
    # Processing the call arguments (line 34)
    str_564644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 10), 'str', 'Stirling series coefficients')
    # Processing the call keyword arguments (line 34)
    kwargs_564645 = {}
    # Getting the type of 'print' (line 34)
    print_564643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'print', False)
    # Calling print(args, kwargs) (line 34)
    print_call_result_564646 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), print_564643, *[str_564644], **kwargs_564645)
    
    
    # Call to print(...): (line 35)
    # Processing the call arguments (line 35)
    str_564648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 10), 'str', '----------------------------')
    # Processing the call keyword arguments (line 35)
    kwargs_564649 = {}
    # Getting the type of 'print' (line 35)
    print_564647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'print', False)
    # Calling print(args, kwargs) (line 35)
    print_call_result_564650 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), print_564647, *[str_564648], **kwargs_564649)
    
    
    # Call to print(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to join(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'stirling_coeffs' (line 36)
    stirling_coeffs_564654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'stirling_coeffs', False)
    # Processing the call keyword arguments (line 36)
    kwargs_564655 = {}
    str_564652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 10), 'str', '\n')
    # Obtaining the member 'join' of a type (line 36)
    join_564653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 10), str_564652, 'join')
    # Calling join(args, kwargs) (line 36)
    join_call_result_564656 = invoke(stypy.reporting.localization.Localization(__file__, 36, 10), join_564653, *[stirling_coeffs_564654], **kwargs_564655)
    
    # Processing the call keyword arguments (line 36)
    kwargs_564657 = {}
    # Getting the type of 'print' (line 36)
    print_564651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'print', False)
    # Calling print(args, kwargs) (line 36)
    print_call_result_564658 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), print_564651, *[join_call_result_564656], **kwargs_564657)
    
    
    # Call to print(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_564660 = {}
    # Getting the type of 'print' (line 37)
    print_564659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'print', False)
    # Calling print(args, kwargs) (line 37)
    print_call_result_564661 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), print_564659, *[], **kwargs_564660)
    
    
    # Call to print(...): (line 38)
    # Processing the call arguments (line 38)
    str_564663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 10), 'str', 'Taylor series coefficients')
    # Processing the call keyword arguments (line 38)
    kwargs_564664 = {}
    # Getting the type of 'print' (line 38)
    print_564662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'print', False)
    # Calling print(args, kwargs) (line 38)
    print_call_result_564665 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), print_564662, *[str_564663], **kwargs_564664)
    
    
    # Call to print(...): (line 39)
    # Processing the call arguments (line 39)
    str_564667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 10), 'str', '--------------------------')
    # Processing the call keyword arguments (line 39)
    kwargs_564668 = {}
    # Getting the type of 'print' (line 39)
    print_564666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'print', False)
    # Calling print(args, kwargs) (line 39)
    print_call_result_564669 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), print_564666, *[str_564667], **kwargs_564668)
    
    
    # Call to print(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Call to join(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'taylor_coeffs' (line 40)
    taylor_coeffs_564673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'taylor_coeffs', False)
    # Processing the call keyword arguments (line 40)
    kwargs_564674 = {}
    str_564671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 10), 'str', '\n')
    # Obtaining the member 'join' of a type (line 40)
    join_564672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 10), str_564671, 'join')
    # Calling join(args, kwargs) (line 40)
    join_call_result_564675 = invoke(stypy.reporting.localization.Localization(__file__, 40, 10), join_564672, *[taylor_coeffs_564673], **kwargs_564674)
    
    # Processing the call keyword arguments (line 40)
    kwargs_564676 = {}
    # Getting the type of 'print' (line 40)
    print_564670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'print', False)
    # Calling print(args, kwargs) (line 40)
    print_call_result_564677 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), print_564670, *[join_call_result_564675], **kwargs_564676)
    
    
    # Call to print(...): (line 41)
    # Processing the call keyword arguments (line 41)
    kwargs_564679 = {}
    # Getting the type of 'print' (line 41)
    print_564678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'print', False)
    # Calling print(args, kwargs) (line 41)
    print_call_result_564680 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), print_564678, *[], **kwargs_564679)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 27)
    stypy_return_type_564681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_564681)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_564681

# Assigning a type to the variable 'main' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'main', main)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_564683 = {}
    # Getting the type of 'main' (line 45)
    main_564682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'main', False)
    # Calling main(args, kwargs) (line 45)
    main_call_result_564684 = invoke(stypy.reporting.localization.Localization(__file__, 45, 4), main_564682, *[], **kwargs_564683)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
