
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from scipy._lib._numpy_compat import suppress_warnings
4: 
5: try:
6:     import mpmath as mp
7: except ImportError:
8:     pass
9: 
10: try:
11:     # Can remove when sympy #11255 is resolved; see
12:     # https://github.com/sympy/sympy/issues/11255
13:     with suppress_warnings() as sup:
14:         sup.filter(DeprecationWarning, "inspect.getargspec.. is deprecated")
15:         from sympy.abc import x
16: except ImportError:
17:     pass
18: 
19: 
20: def lagrange_inversion(a):
21:     '''Given a series
22: 
23:     f(x) = a[1]*x + a[2]*x**2 + ... + a[n-1]*x**(n - 1),
24: 
25:     use the Lagrange inversion formula to compute a series
26: 
27:     g(x) = b[1]*x + b[2]*x**2 + ... + b[n-1]*x**(n - 1)
28: 
29:     so that f(g(x)) = g(f(x)) = x mod x**n. We must have a[0] = 0, so
30:     necessarily b[0] = 0 too.
31: 
32:     The algorithm is naive and could be improved, but speed isn't an
33:     issue here and it's easy to read.
34: 
35:     '''
36:     n = len(a)
37:     f = sum(a[i]*x**i for i in range(len(a)))
38:     h = (x/f).series(x, 0, n).removeO()
39:     hpower = [h**0]
40:     for k in range(n):
41:         hpower.append((hpower[-1]*h).expand())
42:     b = [mp.mpf(0)]
43:     for k in range(1, n):
44:         b.append(hpower[k].coeff(x, k - 1)/k)
45:     b = map(lambda x: mp.mpf(x), b)
46:     return b
47: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/_precompute/')
import_564708 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib._numpy_compat')

if (type(import_564708) is not StypyTypeError):

    if (import_564708 != 'pyd_module'):
        __import__(import_564708)
        sys_modules_564709 = sys.modules[import_564708]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib._numpy_compat', sys_modules_564709.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_564709, sys_modules_564709.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib._numpy_compat', import_564708)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/_precompute/')



# SSA begins for try-except statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))

# 'import mpmath' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/_precompute/')
import_564710 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'mpmath')

if (type(import_564710) is not StypyTypeError):

    if (import_564710 != 'pyd_module'):
        __import__(import_564710)
        sys_modules_564711 = sys.modules[import_564710]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'mp', sys_modules_564711.module_type_store, module_type_store)
    else:
        import mpmath as mp

        import_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'mp', mpmath, module_type_store)

else:
    # Assigning a type to the variable 'mpmath' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'mpmath', import_564710)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/_precompute/')

# SSA branch for the except part of a try statement (line 5)
# SSA branch for the except 'ImportError' branch of a try statement (line 5)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 5)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 10)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to suppress_warnings(...): (line 13)
# Processing the call keyword arguments (line 13)
kwargs_564713 = {}
# Getting the type of 'suppress_warnings' (line 13)
suppress_warnings_564712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 'suppress_warnings', False)
# Calling suppress_warnings(args, kwargs) (line 13)
suppress_warnings_call_result_564714 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), suppress_warnings_564712, *[], **kwargs_564713)

with_564715 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 13, 9), suppress_warnings_call_result_564714, 'with parameter', '__enter__', '__exit__')

if with_564715:
    # Calling the __enter__ method to initiate a with section
    # Obtaining the member '__enter__' of a type (line 13)
    enter___564716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), suppress_warnings_call_result_564714, '__enter__')
    with_enter_564717 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), enter___564716)
    # Assigning a type to the variable 'sup' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 'sup', with_enter_564717)
    
    # Call to filter(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'DeprecationWarning' (line 14)
    DeprecationWarning_564720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'DeprecationWarning', False)
    str_564721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 39), 'str', 'inspect.getargspec.. is deprecated')
    # Processing the call keyword arguments (line 14)
    kwargs_564722 = {}
    # Getting the type of 'sup' (line 14)
    sup_564718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'sup', False)
    # Obtaining the member 'filter' of a type (line 14)
    filter_564719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), sup_564718, 'filter')
    # Calling filter(args, kwargs) (line 14)
    filter_call_result_564723 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), filter_564719, *[DeprecationWarning_564720, str_564721], **kwargs_564722)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 8))
    
    # 'from sympy.abc import x' statement (line 15)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/_precompute/')
    import_564724 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 8), 'sympy.abc')

    if (type(import_564724) is not StypyTypeError):

        if (import_564724 != 'pyd_module'):
            __import__(import_564724)
            sys_modules_564725 = sys.modules[import_564724]
            import_from_module(stypy.reporting.localization.Localization(__file__, 15, 8), 'sympy.abc', sys_modules_564725.module_type_store, module_type_store, ['x'])
            nest_module(stypy.reporting.localization.Localization(__file__, 15, 8), __file__, sys_modules_564725, sys_modules_564725.module_type_store, module_type_store)
        else:
            from sympy.abc import x

            import_from_module(stypy.reporting.localization.Localization(__file__, 15, 8), 'sympy.abc', None, module_type_store, ['x'], [x])

    else:
        # Assigning a type to the variable 'sympy.abc' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'sympy.abc', import_564724)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/_precompute/')
    
    # Calling the __exit__ method to finish a with section
    # Obtaining the member '__exit__' of a type (line 13)
    exit___564726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), suppress_warnings_call_result_564714, '__exit__')
    with_exit_564727 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), exit___564726, None, None, None)

# SSA branch for the except part of a try statement (line 10)
# SSA branch for the except 'ImportError' branch of a try statement (line 10)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 10)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def lagrange_inversion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lagrange_inversion'
    module_type_store = module_type_store.open_function_context('lagrange_inversion', 20, 0, False)
    
    # Passed parameters checking function
    lagrange_inversion.stypy_localization = localization
    lagrange_inversion.stypy_type_of_self = None
    lagrange_inversion.stypy_type_store = module_type_store
    lagrange_inversion.stypy_function_name = 'lagrange_inversion'
    lagrange_inversion.stypy_param_names_list = ['a']
    lagrange_inversion.stypy_varargs_param_name = None
    lagrange_inversion.stypy_kwargs_param_name = None
    lagrange_inversion.stypy_call_defaults = defaults
    lagrange_inversion.stypy_call_varargs = varargs
    lagrange_inversion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lagrange_inversion', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lagrange_inversion', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lagrange_inversion(...)' code ##################

    str_564728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', "Given a series\n\n    f(x) = a[1]*x + a[2]*x**2 + ... + a[n-1]*x**(n - 1),\n\n    use the Lagrange inversion formula to compute a series\n\n    g(x) = b[1]*x + b[2]*x**2 + ... + b[n-1]*x**(n - 1)\n\n    so that f(g(x)) = g(f(x)) = x mod x**n. We must have a[0] = 0, so\n    necessarily b[0] = 0 too.\n\n    The algorithm is naive and could be improved, but speed isn't an\n    issue here and it's easy to read.\n\n    ")
    
    # Assigning a Call to a Name (line 36):
    
    # Call to len(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'a' (line 36)
    a_564730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'a', False)
    # Processing the call keyword arguments (line 36)
    kwargs_564731 = {}
    # Getting the type of 'len' (line 36)
    len_564729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'len', False)
    # Calling len(args, kwargs) (line 36)
    len_call_result_564732 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), len_564729, *[a_564730], **kwargs_564731)
    
    # Assigning a type to the variable 'n' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'n', len_call_result_564732)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to sum(...): (line 37)
    # Processing the call arguments (line 37)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 37, 12, True)
    # Calculating comprehension expression
    
    # Call to range(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to len(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'a' (line 37)
    a_564744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 41), 'a', False)
    # Processing the call keyword arguments (line 37)
    kwargs_564745 = {}
    # Getting the type of 'len' (line 37)
    len_564743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 37), 'len', False)
    # Calling len(args, kwargs) (line 37)
    len_call_result_564746 = invoke(stypy.reporting.localization.Localization(__file__, 37, 37), len_564743, *[a_564744], **kwargs_564745)
    
    # Processing the call keyword arguments (line 37)
    kwargs_564747 = {}
    # Getting the type of 'range' (line 37)
    range_564742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'range', False)
    # Calling range(args, kwargs) (line 37)
    range_call_result_564748 = invoke(stypy.reporting.localization.Localization(__file__, 37, 31), range_564742, *[len_call_result_564746], **kwargs_564747)
    
    comprehension_564749 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 12), range_call_result_564748)
    # Assigning a type to the variable 'i' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'i', comprehension_564749)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 37)
    i_564734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'i', False)
    # Getting the type of 'a' (line 37)
    a_564735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___564736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), a_564735, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_564737 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), getitem___564736, i_564734)
    
    # Getting the type of 'x' (line 37)
    x_564738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'x', False)
    # Getting the type of 'i' (line 37)
    i_564739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'i', False)
    # Applying the binary operator '**' (line 37)
    result_pow_564740 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 17), '**', x_564738, i_564739)
    
    # Applying the binary operator '*' (line 37)
    result_mul_564741 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 12), '*', subscript_call_result_564737, result_pow_564740)
    
    list_564750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 12), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 12), list_564750, result_mul_564741)
    # Processing the call keyword arguments (line 37)
    kwargs_564751 = {}
    # Getting the type of 'sum' (line 37)
    sum_564733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'sum', False)
    # Calling sum(args, kwargs) (line 37)
    sum_call_result_564752 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), sum_564733, *[list_564750], **kwargs_564751)
    
    # Assigning a type to the variable 'f' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'f', sum_call_result_564752)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to removeO(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_564763 = {}
    
    # Call to series(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'x' (line 38)
    x_564757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'x', False)
    int_564758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'int')
    # Getting the type of 'n' (line 38)
    n_564759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'n', False)
    # Processing the call keyword arguments (line 38)
    kwargs_564760 = {}
    # Getting the type of 'x' (line 38)
    x_564753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'x', False)
    # Getting the type of 'f' (line 38)
    f_564754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'f', False)
    # Applying the binary operator 'div' (line 38)
    result_div_564755 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 9), 'div', x_564753, f_564754)
    
    # Obtaining the member 'series' of a type (line 38)
    series_564756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 9), result_div_564755, 'series')
    # Calling series(args, kwargs) (line 38)
    series_call_result_564761 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), series_564756, *[x_564757, int_564758, n_564759], **kwargs_564760)
    
    # Obtaining the member 'removeO' of a type (line 38)
    removeO_564762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 9), series_call_result_564761, 'removeO')
    # Calling removeO(args, kwargs) (line 38)
    removeO_call_result_564764 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), removeO_564762, *[], **kwargs_564763)
    
    # Assigning a type to the variable 'h' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'h', removeO_call_result_564764)
    
    # Assigning a List to a Name (line 39):
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_564765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    # Getting the type of 'h' (line 39)
    h_564766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'h')
    int_564767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'int')
    # Applying the binary operator '**' (line 39)
    result_pow_564768 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 14), '**', h_564766, int_564767)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 13), list_564765, result_pow_564768)
    
    # Assigning a type to the variable 'hpower' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'hpower', list_564765)
    
    
    # Call to range(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'n' (line 40)
    n_564770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'n', False)
    # Processing the call keyword arguments (line 40)
    kwargs_564771 = {}
    # Getting the type of 'range' (line 40)
    range_564769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'range', False)
    # Calling range(args, kwargs) (line 40)
    range_call_result_564772 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), range_564769, *[n_564770], **kwargs_564771)
    
    # Testing the type of a for loop iterable (line 40)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 4), range_call_result_564772)
    # Getting the type of the for loop variable (line 40)
    for_loop_var_564773 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 4), range_call_result_564772)
    # Assigning a type to the variable 'k' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'k', for_loop_var_564773)
    # SSA begins for a for statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Call to expand(...): (line 41)
    # Processing the call keyword arguments (line 41)
    kwargs_564783 = {}
    
    # Obtaining the type of the subscript
    int_564776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'int')
    # Getting the type of 'hpower' (line 41)
    hpower_564777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'hpower', False)
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___564778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), hpower_564777, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_564779 = invoke(stypy.reporting.localization.Localization(__file__, 41, 23), getitem___564778, int_564776)
    
    # Getting the type of 'h' (line 41)
    h_564780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'h', False)
    # Applying the binary operator '*' (line 41)
    result_mul_564781 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 23), '*', subscript_call_result_564779, h_564780)
    
    # Obtaining the member 'expand' of a type (line 41)
    expand_564782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), result_mul_564781, 'expand')
    # Calling expand(args, kwargs) (line 41)
    expand_call_result_564784 = invoke(stypy.reporting.localization.Localization(__file__, 41, 23), expand_564782, *[], **kwargs_564783)
    
    # Processing the call keyword arguments (line 41)
    kwargs_564785 = {}
    # Getting the type of 'hpower' (line 41)
    hpower_564774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'hpower', False)
    # Obtaining the member 'append' of a type (line 41)
    append_564775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), hpower_564774, 'append')
    # Calling append(args, kwargs) (line 41)
    append_call_result_564786 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), append_564775, *[expand_call_result_564784], **kwargs_564785)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 42):
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_564787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    
    # Call to mpf(...): (line 42)
    # Processing the call arguments (line 42)
    int_564790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'int')
    # Processing the call keyword arguments (line 42)
    kwargs_564791 = {}
    # Getting the type of 'mp' (line 42)
    mp_564788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 9), 'mp', False)
    # Obtaining the member 'mpf' of a type (line 42)
    mpf_564789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 9), mp_564788, 'mpf')
    # Calling mpf(args, kwargs) (line 42)
    mpf_call_result_564792 = invoke(stypy.reporting.localization.Localization(__file__, 42, 9), mpf_564789, *[int_564790], **kwargs_564791)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 8), list_564787, mpf_call_result_564792)
    
    # Assigning a type to the variable 'b' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'b', list_564787)
    
    
    # Call to range(...): (line 43)
    # Processing the call arguments (line 43)
    int_564794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
    # Getting the type of 'n' (line 43)
    n_564795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'n', False)
    # Processing the call keyword arguments (line 43)
    kwargs_564796 = {}
    # Getting the type of 'range' (line 43)
    range_564793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'range', False)
    # Calling range(args, kwargs) (line 43)
    range_call_result_564797 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), range_564793, *[int_564794, n_564795], **kwargs_564796)
    
    # Testing the type of a for loop iterable (line 43)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 4), range_call_result_564797)
    # Getting the type of the for loop variable (line 43)
    for_loop_var_564798 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 4), range_call_result_564797)
    # Assigning a type to the variable 'k' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'k', for_loop_var_564798)
    # SSA begins for a for statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to coeff(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'x' (line 44)
    x_564806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 33), 'x', False)
    # Getting the type of 'k' (line 44)
    k_564807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 36), 'k', False)
    int_564808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 40), 'int')
    # Applying the binary operator '-' (line 44)
    result_sub_564809 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 36), '-', k_564807, int_564808)
    
    # Processing the call keyword arguments (line 44)
    kwargs_564810 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 44)
    k_564801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'k', False)
    # Getting the type of 'hpower' (line 44)
    hpower_564802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'hpower', False)
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___564803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 17), hpower_564802, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_564804 = invoke(stypy.reporting.localization.Localization(__file__, 44, 17), getitem___564803, k_564801)
    
    # Obtaining the member 'coeff' of a type (line 44)
    coeff_564805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 17), subscript_call_result_564804, 'coeff')
    # Calling coeff(args, kwargs) (line 44)
    coeff_call_result_564811 = invoke(stypy.reporting.localization.Localization(__file__, 44, 17), coeff_564805, *[x_564806, result_sub_564809], **kwargs_564810)
    
    # Getting the type of 'k' (line 44)
    k_564812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 43), 'k', False)
    # Applying the binary operator 'div' (line 44)
    result_div_564813 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 17), 'div', coeff_call_result_564811, k_564812)
    
    # Processing the call keyword arguments (line 44)
    kwargs_564814 = {}
    # Getting the type of 'b' (line 44)
    b_564799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'b', False)
    # Obtaining the member 'append' of a type (line 44)
    append_564800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), b_564799, 'append')
    # Calling append(args, kwargs) (line 44)
    append_call_result_564815 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), append_564800, *[result_div_564813], **kwargs_564814)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 45):
    
    # Call to map(...): (line 45)
    # Processing the call arguments (line 45)

    @norecursion
    def _stypy_temp_lambda_486(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_486'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_486', 45, 12, True)
        # Passed parameters checking function
        _stypy_temp_lambda_486.stypy_localization = localization
        _stypy_temp_lambda_486.stypy_type_of_self = None
        _stypy_temp_lambda_486.stypy_type_store = module_type_store
        _stypy_temp_lambda_486.stypy_function_name = '_stypy_temp_lambda_486'
        _stypy_temp_lambda_486.stypy_param_names_list = ['x']
        _stypy_temp_lambda_486.stypy_varargs_param_name = None
        _stypy_temp_lambda_486.stypy_kwargs_param_name = None
        _stypy_temp_lambda_486.stypy_call_defaults = defaults
        _stypy_temp_lambda_486.stypy_call_varargs = varargs
        _stypy_temp_lambda_486.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_486', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_486', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to mpf(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'x' (line 45)
        x_564819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'x', False)
        # Processing the call keyword arguments (line 45)
        kwargs_564820 = {}
        # Getting the type of 'mp' (line 45)
        mp_564817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 45)
        mpf_564818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 22), mp_564817, 'mpf')
        # Calling mpf(args, kwargs) (line 45)
        mpf_call_result_564821 = invoke(stypy.reporting.localization.Localization(__file__, 45, 22), mpf_564818, *[x_564819], **kwargs_564820)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type', mpf_call_result_564821)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_486' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_564822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_564822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_486'
        return stypy_return_type_564822

    # Assigning a type to the variable '_stypy_temp_lambda_486' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), '_stypy_temp_lambda_486', _stypy_temp_lambda_486)
    # Getting the type of '_stypy_temp_lambda_486' (line 45)
    _stypy_temp_lambda_486_564823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), '_stypy_temp_lambda_486')
    # Getting the type of 'b' (line 45)
    b_564824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'b', False)
    # Processing the call keyword arguments (line 45)
    kwargs_564825 = {}
    # Getting the type of 'map' (line 45)
    map_564816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'map', False)
    # Calling map(args, kwargs) (line 45)
    map_call_result_564826 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), map_564816, *[_stypy_temp_lambda_486_564823, b_564824], **kwargs_564825)
    
    # Assigning a type to the variable 'b' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'b', map_call_result_564826)
    # Getting the type of 'b' (line 46)
    b_564827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'b')
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', b_564827)
    
    # ################# End of 'lagrange_inversion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lagrange_inversion' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_564828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_564828)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lagrange_inversion'
    return stypy_return_type_564828

# Assigning a type to the variable 'lagrange_inversion' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'lagrange_inversion', lagrange_inversion)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
