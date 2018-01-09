
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from numpy import zeros, asarray, eye, poly1d, hstack, r_
4: from scipy import linalg
5: 
6: __all__ = ["pade"]
7: 
8: def pade(an, m):
9:     '''
10:     Return Pade approximation to a polynomial as the ratio of two polynomials.
11: 
12:     Parameters
13:     ----------
14:     an : (N,) array_like
15:         Taylor series coefficients.
16:     m : int
17:         The order of the returned approximating polynomials.
18: 
19:     Returns
20:     -------
21:     p, q : Polynomial class
22:         The Pade approximation of the polynomial defined by `an` is
23:         ``p(x)/q(x)``.
24: 
25:     Examples
26:     --------
27:     >>> from scipy.interpolate import pade
28:     >>> e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0]
29:     >>> p, q = pade(e_exp, 2)
30: 
31:     >>> e_exp.reverse()
32:     >>> e_poly = np.poly1d(e_exp)
33: 
34:     Compare ``e_poly(x)`` and the Pade approximation ``p(x)/q(x)``
35: 
36:     >>> e_poly(1)
37:     2.7166666666666668
38: 
39:     >>> p(1)/q(1)
40:     2.7179487179487181
41: 
42:     '''
43:     an = asarray(an)
44:     N = len(an) - 1
45:     n = N - m
46:     if n < 0:
47:         raise ValueError("Order of q <m> must be smaller than len(an)-1.")
48:     Akj = eye(N+1, n+1)
49:     Bkj = zeros((N+1, m), 'd')
50:     for row in range(1, m+1):
51:         Bkj[row,:row] = -(an[:row])[::-1]
52:     for row in range(m+1, N+1):
53:         Bkj[row,:] = -(an[row-m:row])[::-1]
54:     C = hstack((Akj, Bkj))
55:     pq = linalg.solve(C, an)
56:     p = pq[:n+1]
57:     q = r_[1.0, pq[n+1:]]
58:     return poly1d(p[::-1]), poly1d(q[::-1])
59: 
60: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy import zeros, asarray, eye, poly1d, hstack, r_' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81751 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_81751) is not StypyTypeError):

    if (import_81751 != 'pyd_module'):
        __import__(import_81751)
        sys_modules_81752 = sys.modules[import_81751]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', sys_modules_81752.module_type_store, module_type_store, ['zeros', 'asarray', 'eye', 'poly1d', 'hstack', 'r_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_81752, sys_modules_81752.module_type_store, module_type_store)
    else:
        from numpy import zeros, asarray, eye, poly1d, hstack, r_

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', None, module_type_store, ['zeros', 'asarray', 'eye', 'poly1d', 'hstack', 'r_'], [zeros, asarray, eye, poly1d, hstack, r_])

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_81751)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy import linalg' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_81753 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy')

if (type(import_81753) is not StypyTypeError):

    if (import_81753 != 'pyd_module'):
        __import__(import_81753)
        sys_modules_81754 = sys.modules[import_81753]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy', sys_modules_81754.module_type_store, module_type_store, ['linalg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_81754, sys_modules_81754.module_type_store, module_type_store)
    else:
        from scipy import linalg

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy', None, module_type_store, ['linalg'], [linalg])

else:
    # Assigning a type to the variable 'scipy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy', import_81753)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


# Assigning a List to a Name (line 6):
__all__ = ['pade']
module_type_store.set_exportable_members(['pade'])

# Obtaining an instance of the builtin type 'list' (line 6)
list_81755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_81756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'str', 'pade')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_81755, str_81756)

# Assigning a type to the variable '__all__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__all__', list_81755)

@norecursion
def pade(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pade'
    module_type_store = module_type_store.open_function_context('pade', 8, 0, False)
    
    # Passed parameters checking function
    pade.stypy_localization = localization
    pade.stypy_type_of_self = None
    pade.stypy_type_store = module_type_store
    pade.stypy_function_name = 'pade'
    pade.stypy_param_names_list = ['an', 'm']
    pade.stypy_varargs_param_name = None
    pade.stypy_kwargs_param_name = None
    pade.stypy_call_defaults = defaults
    pade.stypy_call_varargs = varargs
    pade.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pade', ['an', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pade', localization, ['an', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pade(...)' code ##################

    str_81757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'str', '\n    Return Pade approximation to a polynomial as the ratio of two polynomials.\n\n    Parameters\n    ----------\n    an : (N,) array_like\n        Taylor series coefficients.\n    m : int\n        The order of the returned approximating polynomials.\n\n    Returns\n    -------\n    p, q : Polynomial class\n        The Pade approximation of the polynomial defined by `an` is\n        ``p(x)/q(x)``.\n\n    Examples\n    --------\n    >>> from scipy.interpolate import pade\n    >>> e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0]\n    >>> p, q = pade(e_exp, 2)\n\n    >>> e_exp.reverse()\n    >>> e_poly = np.poly1d(e_exp)\n\n    Compare ``e_poly(x)`` and the Pade approximation ``p(x)/q(x)``\n\n    >>> e_poly(1)\n    2.7166666666666668\n\n    >>> p(1)/q(1)\n    2.7179487179487181\n\n    ')
    
    # Assigning a Call to a Name (line 43):
    
    # Call to asarray(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'an' (line 43)
    an_81759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'an', False)
    # Processing the call keyword arguments (line 43)
    kwargs_81760 = {}
    # Getting the type of 'asarray' (line 43)
    asarray_81758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 9), 'asarray', False)
    # Calling asarray(args, kwargs) (line 43)
    asarray_call_result_81761 = invoke(stypy.reporting.localization.Localization(__file__, 43, 9), asarray_81758, *[an_81759], **kwargs_81760)
    
    # Assigning a type to the variable 'an' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'an', asarray_call_result_81761)
    
    # Assigning a BinOp to a Name (line 44):
    
    # Call to len(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'an' (line 44)
    an_81763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'an', False)
    # Processing the call keyword arguments (line 44)
    kwargs_81764 = {}
    # Getting the type of 'len' (line 44)
    len_81762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'len', False)
    # Calling len(args, kwargs) (line 44)
    len_call_result_81765 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), len_81762, *[an_81763], **kwargs_81764)
    
    int_81766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 18), 'int')
    # Applying the binary operator '-' (line 44)
    result_sub_81767 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 8), '-', len_call_result_81765, int_81766)
    
    # Assigning a type to the variable 'N' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'N', result_sub_81767)
    
    # Assigning a BinOp to a Name (line 45):
    # Getting the type of 'N' (line 45)
    N_81768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'N')
    # Getting the type of 'm' (line 45)
    m_81769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'm')
    # Applying the binary operator '-' (line 45)
    result_sub_81770 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 8), '-', N_81768, m_81769)
    
    # Assigning a type to the variable 'n' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'n', result_sub_81770)
    
    
    # Getting the type of 'n' (line 46)
    n_81771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'n')
    int_81772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 11), 'int')
    # Applying the binary operator '<' (line 46)
    result_lt_81773 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 7), '<', n_81771, int_81772)
    
    # Testing the type of an if condition (line 46)
    if_condition_81774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 4), result_lt_81773)
    # Assigning a type to the variable 'if_condition_81774' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'if_condition_81774', if_condition_81774)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 47)
    # Processing the call arguments (line 47)
    str_81776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'str', 'Order of q <m> must be smaller than len(an)-1.')
    # Processing the call keyword arguments (line 47)
    kwargs_81777 = {}
    # Getting the type of 'ValueError' (line 47)
    ValueError_81775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 47)
    ValueError_call_result_81778 = invoke(stypy.reporting.localization.Localization(__file__, 47, 14), ValueError_81775, *[str_81776], **kwargs_81777)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 47, 8), ValueError_call_result_81778, 'raise parameter', BaseException)
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 48):
    
    # Call to eye(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'N' (line 48)
    N_81780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'N', False)
    int_81781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 16), 'int')
    # Applying the binary operator '+' (line 48)
    result_add_81782 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 14), '+', N_81780, int_81781)
    
    # Getting the type of 'n' (line 48)
    n_81783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'n', False)
    int_81784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 21), 'int')
    # Applying the binary operator '+' (line 48)
    result_add_81785 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 19), '+', n_81783, int_81784)
    
    # Processing the call keyword arguments (line 48)
    kwargs_81786 = {}
    # Getting the type of 'eye' (line 48)
    eye_81779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'eye', False)
    # Calling eye(args, kwargs) (line 48)
    eye_call_result_81787 = invoke(stypy.reporting.localization.Localization(__file__, 48, 10), eye_81779, *[result_add_81782, result_add_81785], **kwargs_81786)
    
    # Assigning a type to the variable 'Akj' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'Akj', eye_call_result_81787)
    
    # Assigning a Call to a Name (line 49):
    
    # Call to zeros(...): (line 49)
    # Processing the call arguments (line 49)
    
    # Obtaining an instance of the builtin type 'tuple' (line 49)
    tuple_81789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 49)
    # Adding element type (line 49)
    # Getting the type of 'N' (line 49)
    N_81790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'N', False)
    int_81791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'int')
    # Applying the binary operator '+' (line 49)
    result_add_81792 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 17), '+', N_81790, int_81791)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 17), tuple_81789, result_add_81792)
    # Adding element type (line 49)
    # Getting the type of 'm' (line 49)
    m_81793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 17), tuple_81789, m_81793)
    
    str_81794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 26), 'str', 'd')
    # Processing the call keyword arguments (line 49)
    kwargs_81795 = {}
    # Getting the type of 'zeros' (line 49)
    zeros_81788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'zeros', False)
    # Calling zeros(args, kwargs) (line 49)
    zeros_call_result_81796 = invoke(stypy.reporting.localization.Localization(__file__, 49, 10), zeros_81788, *[tuple_81789, str_81794], **kwargs_81795)
    
    # Assigning a type to the variable 'Bkj' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'Bkj', zeros_call_result_81796)
    
    
    # Call to range(...): (line 50)
    # Processing the call arguments (line 50)
    int_81798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 21), 'int')
    # Getting the type of 'm' (line 50)
    m_81799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'm', False)
    int_81800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 26), 'int')
    # Applying the binary operator '+' (line 50)
    result_add_81801 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 24), '+', m_81799, int_81800)
    
    # Processing the call keyword arguments (line 50)
    kwargs_81802 = {}
    # Getting the type of 'range' (line 50)
    range_81797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'range', False)
    # Calling range(args, kwargs) (line 50)
    range_call_result_81803 = invoke(stypy.reporting.localization.Localization(__file__, 50, 15), range_81797, *[int_81798, result_add_81801], **kwargs_81802)
    
    # Testing the type of a for loop iterable (line 50)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 4), range_call_result_81803)
    # Getting the type of the for loop variable (line 50)
    for_loop_var_81804 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 4), range_call_result_81803)
    # Assigning a type to the variable 'row' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'row', for_loop_var_81804)
    # SSA begins for a for statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a UnaryOp to a Subscript (line 51):
    
    
    # Obtaining the type of the subscript
    int_81805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 38), 'int')
    slice_81806 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 51, 26), None, None, int_81805)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 51)
    row_81807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'row')
    slice_81808 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 51, 26), None, row_81807, None)
    # Getting the type of 'an' (line 51)
    an_81809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'an')
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___81810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 26), an_81809, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_81811 = invoke(stypy.reporting.localization.Localization(__file__, 51, 26), getitem___81810, slice_81808)
    
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___81812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 26), subscript_call_result_81811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_81813 = invoke(stypy.reporting.localization.Localization(__file__, 51, 26), getitem___81812, slice_81806)
    
    # Applying the 'usub' unary operator (line 51)
    result___neg___81814 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 24), 'usub', subscript_call_result_81813)
    
    # Getting the type of 'Bkj' (line 51)
    Bkj_81815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'Bkj')
    # Getting the type of 'row' (line 51)
    row_81816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'row')
    # Getting the type of 'row' (line 51)
    row_81817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'row')
    slice_81818 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 51, 8), None, row_81817, None)
    # Storing an element on a container (line 51)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), Bkj_81815, ((row_81816, slice_81818), result___neg___81814))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'm' (line 52)
    m_81820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'm', False)
    int_81821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'int')
    # Applying the binary operator '+' (line 52)
    result_add_81822 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 21), '+', m_81820, int_81821)
    
    # Getting the type of 'N' (line 52)
    N_81823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'N', False)
    int_81824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'int')
    # Applying the binary operator '+' (line 52)
    result_add_81825 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 26), '+', N_81823, int_81824)
    
    # Processing the call keyword arguments (line 52)
    kwargs_81826 = {}
    # Getting the type of 'range' (line 52)
    range_81819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'range', False)
    # Calling range(args, kwargs) (line 52)
    range_call_result_81827 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), range_81819, *[result_add_81822, result_add_81825], **kwargs_81826)
    
    # Testing the type of a for loop iterable (line 52)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 4), range_call_result_81827)
    # Getting the type of the for loop variable (line 52)
    for_loop_var_81828 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 4), range_call_result_81827)
    # Assigning a type to the variable 'row' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'row', for_loop_var_81828)
    # SSA begins for a for statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a UnaryOp to a Subscript (line 53):
    
    
    # Obtaining the type of the subscript
    int_81829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 40), 'int')
    slice_81830 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 53, 23), None, None, int_81829)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 53)
    row_81831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'row')
    # Getting the type of 'm' (line 53)
    m_81832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'm')
    # Applying the binary operator '-' (line 53)
    result_sub_81833 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 26), '-', row_81831, m_81832)
    
    # Getting the type of 'row' (line 53)
    row_81834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'row')
    slice_81835 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 53, 23), result_sub_81833, row_81834, None)
    # Getting the type of 'an' (line 53)
    an_81836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'an')
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___81837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 23), an_81836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_81838 = invoke(stypy.reporting.localization.Localization(__file__, 53, 23), getitem___81837, slice_81835)
    
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___81839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 23), subscript_call_result_81838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_81840 = invoke(stypy.reporting.localization.Localization(__file__, 53, 23), getitem___81839, slice_81830)
    
    # Applying the 'usub' unary operator (line 53)
    result___neg___81841 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 21), 'usub', subscript_call_result_81840)
    
    # Getting the type of 'Bkj' (line 53)
    Bkj_81842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'Bkj')
    # Getting the type of 'row' (line 53)
    row_81843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'row')
    slice_81844 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 53, 8), None, None, None)
    # Storing an element on a container (line 53)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 8), Bkj_81842, ((row_81843, slice_81844), result___neg___81841))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 54):
    
    # Call to hstack(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_81846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    # Getting the type of 'Akj' (line 54)
    Akj_81847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'Akj', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 16), tuple_81846, Akj_81847)
    # Adding element type (line 54)
    # Getting the type of 'Bkj' (line 54)
    Bkj_81848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'Bkj', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 16), tuple_81846, Bkj_81848)
    
    # Processing the call keyword arguments (line 54)
    kwargs_81849 = {}
    # Getting the type of 'hstack' (line 54)
    hstack_81845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'hstack', False)
    # Calling hstack(args, kwargs) (line 54)
    hstack_call_result_81850 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), hstack_81845, *[tuple_81846], **kwargs_81849)
    
    # Assigning a type to the variable 'C' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'C', hstack_call_result_81850)
    
    # Assigning a Call to a Name (line 55):
    
    # Call to solve(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'C' (line 55)
    C_81853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'C', False)
    # Getting the type of 'an' (line 55)
    an_81854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'an', False)
    # Processing the call keyword arguments (line 55)
    kwargs_81855 = {}
    # Getting the type of 'linalg' (line 55)
    linalg_81851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'linalg', False)
    # Obtaining the member 'solve' of a type (line 55)
    solve_81852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 9), linalg_81851, 'solve')
    # Calling solve(args, kwargs) (line 55)
    solve_call_result_81856 = invoke(stypy.reporting.localization.Localization(__file__, 55, 9), solve_81852, *[C_81853, an_81854], **kwargs_81855)
    
    # Assigning a type to the variable 'pq' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'pq', solve_call_result_81856)
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 56)
    n_81857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'n')
    int_81858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 14), 'int')
    # Applying the binary operator '+' (line 56)
    result_add_81859 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 12), '+', n_81857, int_81858)
    
    slice_81860 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 8), None, result_add_81859, None)
    # Getting the type of 'pq' (line 56)
    pq_81861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'pq')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___81862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), pq_81861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_81863 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___81862, slice_81860)
    
    # Assigning a type to the variable 'p' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'p', subscript_call_result_81863)
    
    # Assigning a Subscript to a Name (line 57):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 57)
    tuple_81864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 57)
    # Adding element type (line 57)
    float_81865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 11), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 11), tuple_81864, float_81865)
    # Adding element type (line 57)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 57)
    n_81866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'n')
    int_81867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'int')
    # Applying the binary operator '+' (line 57)
    result_add_81868 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 19), '+', n_81866, int_81867)
    
    slice_81869 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 57, 16), result_add_81868, None, None)
    # Getting the type of 'pq' (line 57)
    pq_81870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'pq')
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___81871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), pq_81870, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_81872 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), getitem___81871, slice_81869)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 11), tuple_81864, subscript_call_result_81872)
    
    # Getting the type of 'r_' (line 57)
    r__81873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'r_')
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___81874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), r__81873, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_81875 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), getitem___81874, tuple_81864)
    
    # Assigning a type to the variable 'q' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'q', subscript_call_result_81875)
    
    # Obtaining an instance of the builtin type 'tuple' (line 58)
    tuple_81876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 58)
    # Adding element type (line 58)
    
    # Call to poly1d(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Obtaining the type of the subscript
    int_81878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 22), 'int')
    slice_81879 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 58, 18), None, None, int_81878)
    # Getting the type of 'p' (line 58)
    p_81880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___81881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 18), p_81880, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 58)
    subscript_call_result_81882 = invoke(stypy.reporting.localization.Localization(__file__, 58, 18), getitem___81881, slice_81879)
    
    # Processing the call keyword arguments (line 58)
    kwargs_81883 = {}
    # Getting the type of 'poly1d' (line 58)
    poly1d_81877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 58)
    poly1d_call_result_81884 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), poly1d_81877, *[subscript_call_result_81882], **kwargs_81883)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 11), tuple_81876, poly1d_call_result_81884)
    # Adding element type (line 58)
    
    # Call to poly1d(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Obtaining the type of the subscript
    int_81886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 39), 'int')
    slice_81887 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 58, 35), None, None, int_81886)
    # Getting the type of 'q' (line 58)
    q_81888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 35), 'q', False)
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___81889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 35), q_81888, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 58)
    subscript_call_result_81890 = invoke(stypy.reporting.localization.Localization(__file__, 58, 35), getitem___81889, slice_81887)
    
    # Processing the call keyword arguments (line 58)
    kwargs_81891 = {}
    # Getting the type of 'poly1d' (line 58)
    poly1d_81885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'poly1d', False)
    # Calling poly1d(args, kwargs) (line 58)
    poly1d_call_result_81892 = invoke(stypy.reporting.localization.Localization(__file__, 58, 28), poly1d_81885, *[subscript_call_result_81890], **kwargs_81891)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 11), tuple_81876, poly1d_call_result_81892)
    
    # Assigning a type to the variable 'stypy_return_type' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type', tuple_81876)
    
    # ################# End of 'pade(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pade' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_81893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_81893)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pade'
    return stypy_return_type_81893

# Assigning a type to the variable 'pade' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pade', pade)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
