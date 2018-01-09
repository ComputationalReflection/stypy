
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Last Change: Sat Mar 21 02:00 PM 2009 J
2: 
3: # Copyright (c) 2001, 2002 Enthought, Inc.
4: #
5: # All rights reserved.
6: #
7: # Redistribution and use in source and binary forms, with or without
8: # modification, are permitted provided that the following conditions are met:
9: #
10: #   a. Redistributions of source code must retain the above copyright notice,
11: #      this list of conditions and the following disclaimer.
12: #   b. Redistributions in binary form must reproduce the above copyright
13: #      notice, this list of conditions and the following disclaimer in the
14: #      documentation and/or other materials provided with the distribution.
15: #   c. Neither the name of the Enthought nor the names of its contributors
16: #      may be used to endorse or promote products derived from this software
17: #      without specific prior written permission.
18: #
19: #
20: # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
21: # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
22: # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
23: # ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
24: # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
25: # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
26: # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
27: # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
28: # LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
29: # OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
30: # DAMAGE.
31: 
32: '''Some more special functions which may be useful for multivariate statistical
33: analysis.'''
34: 
35: from __future__ import division, print_function, absolute_import
36: 
37: import numpy as np
38: from scipy.special import gammaln as loggam
39: 
40: 
41: __all__ = ['multigammaln']
42: 
43: 
44: def multigammaln(a, d):
45:     r'''Returns the log of multivariate gamma, also sometimes called the
46:     generalized gamma.
47: 
48:     Parameters
49:     ----------
50:     a : ndarray
51:         The multivariate gamma is computed for each item of `a`.
52:     d : int
53:         The dimension of the space of integration.
54: 
55:     Returns
56:     -------
57:     res : ndarray
58:         The values of the log multivariate gamma at the given points `a`.
59: 
60:     Notes
61:     -----
62:     The formal definition of the multivariate gamma of dimension d for a real
63:     `a` is
64: 
65:     .. math::
66: 
67:         \Gamma_d(a) = \int_{A>0} e^{-tr(A)} |A|^{a - (d+1)/2} dA
68: 
69:     with the condition :math:`a > (d-1)/2`, and :math:`A > 0` being the set of
70:     all the positive definite matrices of dimension `d`.  Note that `a` is a
71:     scalar: the integrand only is multivariate, the argument is not (the
72:     function is defined over a subset of the real set).
73: 
74:     This can be proven to be equal to the much friendlier equation
75: 
76:     .. math::
77: 
78:         \Gamma_d(a) = \pi^{d(d-1)/4} \prod_{i=1}^{d} \Gamma(a - (i-1)/2).
79: 
80:     References
81:     ----------
82:     R. J. Muirhead, Aspects of multivariate statistical theory (Wiley Series in
83:     probability and mathematical statistics).
84: 
85:     '''
86:     a = np.asarray(a)
87:     if not np.isscalar(d) or (np.floor(d) != d):
88:         raise ValueError("d should be a positive integer (dimension)")
89:     if np.any(a <= 0.5 * (d - 1)):
90:         raise ValueError("condition a (%f) > 0.5 * (d-1) (%f) not met"
91:                          % (a, 0.5 * (d-1)))
92: 
93:     res = (d * (d-1) * 0.25) * np.log(np.pi)
94:     res += np.sum(loggam([(a - (j - 1.)/2) for j in range(1, d+1)]), axis=0)
95:     return res
96: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_503714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', 'Some more special functions which may be useful for multivariate statistical\nanalysis.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'import numpy' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_503715 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'numpy')

if (type(import_503715) is not StypyTypeError):

    if (import_503715 != 'pyd_module'):
        __import__(import_503715)
        sys_modules_503716 = sys.modules[import_503715]
        import_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'np', sys_modules_503716.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'numpy', import_503715)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from scipy.special import loggam' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_503717 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.special')

if (type(import_503717) is not StypyTypeError):

    if (import_503717 != 'pyd_module'):
        __import__(import_503717)
        sys_modules_503718 = sys.modules[import_503717]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.special', sys_modules_503718.module_type_store, module_type_store, ['gammaln'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_503718, sys_modules_503718.module_type_store, module_type_store)
    else:
        from scipy.special import gammaln as loggam

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.special', None, module_type_store, ['gammaln'], [loggam])

else:
    # Assigning a type to the variable 'scipy.special' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.special', import_503717)

# Adding an alias
module_type_store.add_alias('loggam', 'gammaln')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')


# Assigning a List to a Name (line 41):
__all__ = ['multigammaln']
module_type_store.set_exportable_members(['multigammaln'])

# Obtaining an instance of the builtin type 'list' (line 41)
list_503719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
str_503720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'str', 'multigammaln')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_503719, str_503720)

# Assigning a type to the variable '__all__' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '__all__', list_503719)

@norecursion
def multigammaln(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'multigammaln'
    module_type_store = module_type_store.open_function_context('multigammaln', 44, 0, False)
    
    # Passed parameters checking function
    multigammaln.stypy_localization = localization
    multigammaln.stypy_type_of_self = None
    multigammaln.stypy_type_store = module_type_store
    multigammaln.stypy_function_name = 'multigammaln'
    multigammaln.stypy_param_names_list = ['a', 'd']
    multigammaln.stypy_varargs_param_name = None
    multigammaln.stypy_kwargs_param_name = None
    multigammaln.stypy_call_defaults = defaults
    multigammaln.stypy_call_varargs = varargs
    multigammaln.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'multigammaln', ['a', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'multigammaln', localization, ['a', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'multigammaln(...)' code ##################

    str_503721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', 'Returns the log of multivariate gamma, also sometimes called the\n    generalized gamma.\n\n    Parameters\n    ----------\n    a : ndarray\n        The multivariate gamma is computed for each item of `a`.\n    d : int\n        The dimension of the space of integration.\n\n    Returns\n    -------\n    res : ndarray\n        The values of the log multivariate gamma at the given points `a`.\n\n    Notes\n    -----\n    The formal definition of the multivariate gamma of dimension d for a real\n    `a` is\n\n    .. math::\n\n        \\Gamma_d(a) = \\int_{A>0} e^{-tr(A)} |A|^{a - (d+1)/2} dA\n\n    with the condition :math:`a > (d-1)/2`, and :math:`A > 0` being the set of\n    all the positive definite matrices of dimension `d`.  Note that `a` is a\n    scalar: the integrand only is multivariate, the argument is not (the\n    function is defined over a subset of the real set).\n\n    This can be proven to be equal to the much friendlier equation\n\n    .. math::\n\n        \\Gamma_d(a) = \\pi^{d(d-1)/4} \\prod_{i=1}^{d} \\Gamma(a - (i-1)/2).\n\n    References\n    ----------\n    R. J. Muirhead, Aspects of multivariate statistical theory (Wiley Series in\n    probability and mathematical statistics).\n\n    ')
    
    # Assigning a Call to a Name (line 86):
    
    # Call to asarray(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'a' (line 86)
    a_503724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'a', False)
    # Processing the call keyword arguments (line 86)
    kwargs_503725 = {}
    # Getting the type of 'np' (line 86)
    np_503722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 86)
    asarray_503723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), np_503722, 'asarray')
    # Calling asarray(args, kwargs) (line 86)
    asarray_call_result_503726 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), asarray_503723, *[a_503724], **kwargs_503725)
    
    # Assigning a type to the variable 'a' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'a', asarray_call_result_503726)
    
    
    # Evaluating a boolean operation
    
    
    # Call to isscalar(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'd' (line 87)
    d_503729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'd', False)
    # Processing the call keyword arguments (line 87)
    kwargs_503730 = {}
    # Getting the type of 'np' (line 87)
    np_503727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'np', False)
    # Obtaining the member 'isscalar' of a type (line 87)
    isscalar_503728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 11), np_503727, 'isscalar')
    # Calling isscalar(args, kwargs) (line 87)
    isscalar_call_result_503731 = invoke(stypy.reporting.localization.Localization(__file__, 87, 11), isscalar_503728, *[d_503729], **kwargs_503730)
    
    # Applying the 'not' unary operator (line 87)
    result_not__503732 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 7), 'not', isscalar_call_result_503731)
    
    
    
    # Call to floor(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'd' (line 87)
    d_503735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 39), 'd', False)
    # Processing the call keyword arguments (line 87)
    kwargs_503736 = {}
    # Getting the type of 'np' (line 87)
    np_503733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 'np', False)
    # Obtaining the member 'floor' of a type (line 87)
    floor_503734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 30), np_503733, 'floor')
    # Calling floor(args, kwargs) (line 87)
    floor_call_result_503737 = invoke(stypy.reporting.localization.Localization(__file__, 87, 30), floor_503734, *[d_503735], **kwargs_503736)
    
    # Getting the type of 'd' (line 87)
    d_503738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 45), 'd')
    # Applying the binary operator '!=' (line 87)
    result_ne_503739 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 30), '!=', floor_call_result_503737, d_503738)
    
    # Applying the binary operator 'or' (line 87)
    result_or_keyword_503740 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 7), 'or', result_not__503732, result_ne_503739)
    
    # Testing the type of an if condition (line 87)
    if_condition_503741 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 4), result_or_keyword_503740)
    # Assigning a type to the variable 'if_condition_503741' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'if_condition_503741', if_condition_503741)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 88)
    # Processing the call arguments (line 88)
    str_503743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 25), 'str', 'd should be a positive integer (dimension)')
    # Processing the call keyword arguments (line 88)
    kwargs_503744 = {}
    # Getting the type of 'ValueError' (line 88)
    ValueError_503742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 88)
    ValueError_call_result_503745 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), ValueError_503742, *[str_503743], **kwargs_503744)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 88, 8), ValueError_call_result_503745, 'raise parameter', BaseException)
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 89)
    # Processing the call arguments (line 89)
    
    # Getting the type of 'a' (line 89)
    a_503748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 14), 'a', False)
    float_503749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 19), 'float')
    # Getting the type of 'd' (line 89)
    d_503750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'd', False)
    int_503751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 30), 'int')
    # Applying the binary operator '-' (line 89)
    result_sub_503752 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 26), '-', d_503750, int_503751)
    
    # Applying the binary operator '*' (line 89)
    result_mul_503753 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 19), '*', float_503749, result_sub_503752)
    
    # Applying the binary operator '<=' (line 89)
    result_le_503754 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 14), '<=', a_503748, result_mul_503753)
    
    # Processing the call keyword arguments (line 89)
    kwargs_503755 = {}
    # Getting the type of 'np' (line 89)
    np_503746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 89)
    any_503747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 7), np_503746, 'any')
    # Calling any(args, kwargs) (line 89)
    any_call_result_503756 = invoke(stypy.reporting.localization.Localization(__file__, 89, 7), any_503747, *[result_le_503754], **kwargs_503755)
    
    # Testing the type of an if condition (line 89)
    if_condition_503757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 4), any_call_result_503756)
    # Assigning a type to the variable 'if_condition_503757' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'if_condition_503757', if_condition_503757)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 90)
    # Processing the call arguments (line 90)
    str_503759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'str', 'condition a (%f) > 0.5 * (d-1) (%f) not met')
    
    # Obtaining an instance of the builtin type 'tuple' (line 91)
    tuple_503760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 91)
    # Adding element type (line 91)
    # Getting the type of 'a' (line 91)
    a_503761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 28), tuple_503760, a_503761)
    # Adding element type (line 91)
    float_503762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 31), 'float')
    # Getting the type of 'd' (line 91)
    d_503763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'd', False)
    int_503764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 40), 'int')
    # Applying the binary operator '-' (line 91)
    result_sub_503765 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 38), '-', d_503763, int_503764)
    
    # Applying the binary operator '*' (line 91)
    result_mul_503766 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 31), '*', float_503762, result_sub_503765)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 28), tuple_503760, result_mul_503766)
    
    # Applying the binary operator '%' (line 90)
    result_mod_503767 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 25), '%', str_503759, tuple_503760)
    
    # Processing the call keyword arguments (line 90)
    kwargs_503768 = {}
    # Getting the type of 'ValueError' (line 90)
    ValueError_503758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 90)
    ValueError_call_result_503769 = invoke(stypy.reporting.localization.Localization(__file__, 90, 14), ValueError_503758, *[result_mod_503767], **kwargs_503768)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 90, 8), ValueError_call_result_503769, 'raise parameter', BaseException)
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 93):
    # Getting the type of 'd' (line 93)
    d_503770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'd')
    # Getting the type of 'd' (line 93)
    d_503771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'd')
    int_503772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 18), 'int')
    # Applying the binary operator '-' (line 93)
    result_sub_503773 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 16), '-', d_503771, int_503772)
    
    # Applying the binary operator '*' (line 93)
    result_mul_503774 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 11), '*', d_503770, result_sub_503773)
    
    float_503775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 23), 'float')
    # Applying the binary operator '*' (line 93)
    result_mul_503776 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 21), '*', result_mul_503774, float_503775)
    
    
    # Call to log(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'np' (line 93)
    np_503779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'np', False)
    # Obtaining the member 'pi' of a type (line 93)
    pi_503780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 38), np_503779, 'pi')
    # Processing the call keyword arguments (line 93)
    kwargs_503781 = {}
    # Getting the type of 'np' (line 93)
    np_503777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'np', False)
    # Obtaining the member 'log' of a type (line 93)
    log_503778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), np_503777, 'log')
    # Calling log(args, kwargs) (line 93)
    log_call_result_503782 = invoke(stypy.reporting.localization.Localization(__file__, 93, 31), log_503778, *[pi_503780], **kwargs_503781)
    
    # Applying the binary operator '*' (line 93)
    result_mul_503783 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 10), '*', result_mul_503776, log_call_result_503782)
    
    # Assigning a type to the variable 'res' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'res', result_mul_503783)
    
    # Getting the type of 'res' (line 94)
    res_503784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'res')
    
    # Call to sum(...): (line 94)
    # Processing the call arguments (line 94)
    
    # Call to loggam(...): (line 94)
    # Processing the call arguments (line 94)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 94)
    # Processing the call arguments (line 94)
    int_503796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 58), 'int')
    # Getting the type of 'd' (line 94)
    d_503797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 61), 'd', False)
    int_503798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 63), 'int')
    # Applying the binary operator '+' (line 94)
    result_add_503799 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 61), '+', d_503797, int_503798)
    
    # Processing the call keyword arguments (line 94)
    kwargs_503800 = {}
    # Getting the type of 'range' (line 94)
    range_503795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 52), 'range', False)
    # Calling range(args, kwargs) (line 94)
    range_call_result_503801 = invoke(stypy.reporting.localization.Localization(__file__, 94, 52), range_503795, *[int_503796, result_add_503799], **kwargs_503800)
    
    comprehension_503802 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 26), range_call_result_503801)
    # Assigning a type to the variable 'j' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'j', comprehension_503802)
    # Getting the type of 'a' (line 94)
    a_503788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'a', False)
    # Getting the type of 'j' (line 94)
    j_503789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 32), 'j', False)
    float_503790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 36), 'float')
    # Applying the binary operator '-' (line 94)
    result_sub_503791 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 32), '-', j_503789, float_503790)
    
    int_503792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 40), 'int')
    # Applying the binary operator 'div' (line 94)
    result_div_503793 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 31), 'div', result_sub_503791, int_503792)
    
    # Applying the binary operator '-' (line 94)
    result_sub_503794 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 27), '-', a_503788, result_div_503793)
    
    list_503803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 26), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 26), list_503803, result_sub_503794)
    # Processing the call keyword arguments (line 94)
    kwargs_503804 = {}
    # Getting the type of 'loggam' (line 94)
    loggam_503787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'loggam', False)
    # Calling loggam(args, kwargs) (line 94)
    loggam_call_result_503805 = invoke(stypy.reporting.localization.Localization(__file__, 94, 18), loggam_503787, *[list_503803], **kwargs_503804)
    
    # Processing the call keyword arguments (line 94)
    int_503806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 74), 'int')
    keyword_503807 = int_503806
    kwargs_503808 = {'axis': keyword_503807}
    # Getting the type of 'np' (line 94)
    np_503785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'np', False)
    # Obtaining the member 'sum' of a type (line 94)
    sum_503786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 11), np_503785, 'sum')
    # Calling sum(args, kwargs) (line 94)
    sum_call_result_503809 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), sum_503786, *[loggam_call_result_503805], **kwargs_503808)
    
    # Applying the binary operator '+=' (line 94)
    result_iadd_503810 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 4), '+=', res_503784, sum_call_result_503809)
    # Assigning a type to the variable 'res' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'res', result_iadd_503810)
    
    # Getting the type of 'res' (line 95)
    res_503811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type', res_503811)
    
    # ################# End of 'multigammaln(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'multigammaln' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_503812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_503812)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'multigammaln'
    return stypy_return_type_503812

# Assigning a type to the variable 'multigammaln' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'multigammaln', multigammaln)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
