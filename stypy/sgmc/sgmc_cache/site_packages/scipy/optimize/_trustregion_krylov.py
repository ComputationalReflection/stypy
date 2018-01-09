
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ._trustregion import (_minimize_trust_region)
2: from ._trlib import (get_trlib_quadratic_subproblem)
3: 
4: __all__ = ['_minimize_trust_krylov']
5: 
6: def _minimize_trust_krylov(fun, x0, args=(), jac=None, hess=None, hessp=None,
7:                            inexact=True, **trust_region_options):
8:     '''
9:     Minimization of a scalar function of one or more variables using
10:     a nearly exact trust-region algorithm that only requires matrix
11:     vector products with the hessian matrix.
12: 
13:     Options
14:     -------
15:     inexact : bool, optional
16:         Accuracy to solve subproblems. If True requires less nonlinear
17:         iterations, but more vector products.
18: 
19:     .. versionadded:: 1.0.0
20:     '''
21: 
22:     if jac is None:
23:         raise ValueError('Jacobian is required for trust region ',
24:                          'exact minimization.')
25:     if hess is None and hessp is None:
26:         raise ValueError('Either the Hessian or the Hessian-vector product '
27:                          'is required for Krylov trust-region minimization')
28: 
29:     # tol_rel specifies the termination tolerance relative to the initial
30:     # gradient norm in the krylov subspace iteration.
31: 
32:     # - tol_rel_i specifies the tolerance for interior convergence.
33:     # - tol_rel_b specifies the tolerance for boundary convergence.
34:     #   in nonlinear programming applications it is not necessary to solve
35:     #   the boundary case as exact as the interior case.
36: 
37:     # - setting tol_rel_i=-2 leads to a forcing sequence in the krylov
38:     #   subspace iteration leading to quadratic convergence if eventually
39:     #   the trust region stays inactive.
40:     # - setting tol_rel_b=-3 leads to a forcing sequence in the krylov
41:     #   subspace iteration leading to superlinear convergence as long
42:     #   as the iterates hit the trust region boundary.
43: 
44:     # For details consult the documentation of trlib_krylov_min
45:     # in _trlib/trlib_krylov.h
46:     #
47:     # Optimality of this choice of parameters among a range of possibilites
48:     # has been tested on the unconstrained subset of the CUTEst library.
49: 
50:     if inexact:
51:         return _minimize_trust_region(fun, x0, args=args, jac=jac,
52:                                       hess=hess, hessp=hessp,
53:                                       subproblem=get_trlib_quadratic_subproblem(
54:                                           tol_rel_i=-2.0, tol_rel_b=-3.0,
55:                                           disp=trust_region_options.get('disp', False)
56:                                           ),
57:                                       **trust_region_options)
58:     else:
59:         return _minimize_trust_region(fun, x0, args=args, jac=jac,
60:                                       hess=hess, hessp=hessp,
61:                                       subproblem=get_trlib_quadratic_subproblem(
62:                                           tol_rel_i=1e-8, tol_rel_b=1e-6,
63:                                           disp=trust_region_options.get('disp', False)
64:                                           ),
65:                                       **trust_region_options)
66: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from scipy.optimize._trustregion import _minimize_trust_region' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204294 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy.optimize._trustregion')

if (type(import_204294) is not StypyTypeError):

    if (import_204294 != 'pyd_module'):
        __import__(import_204294)
        sys_modules_204295 = sys.modules[import_204294]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy.optimize._trustregion', sys_modules_204295.module_type_store, module_type_store, ['_minimize_trust_region'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_204295, sys_modules_204295.module_type_store, module_type_store)
    else:
        from scipy.optimize._trustregion import _minimize_trust_region

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy.optimize._trustregion', None, module_type_store, ['_minimize_trust_region'], [_minimize_trust_region])

else:
    # Assigning a type to the variable 'scipy.optimize._trustregion' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy.optimize._trustregion', import_204294)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from scipy.optimize._trlib import get_trlib_quadratic_subproblem' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204296 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.optimize._trlib')

if (type(import_204296) is not StypyTypeError):

    if (import_204296 != 'pyd_module'):
        __import__(import_204296)
        sys_modules_204297 = sys.modules[import_204296]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.optimize._trlib', sys_modules_204297.module_type_store, module_type_store, ['get_trlib_quadratic_subproblem'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_204297, sys_modules_204297.module_type_store, module_type_store)
    else:
        from scipy.optimize._trlib import get_trlib_quadratic_subproblem

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.optimize._trlib', None, module_type_store, ['get_trlib_quadratic_subproblem'], [get_trlib_quadratic_subproblem])

else:
    # Assigning a type to the variable 'scipy.optimize._trlib' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.optimize._trlib', import_204296)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a List to a Name (line 4):
__all__ = ['_minimize_trust_krylov']
module_type_store.set_exportable_members(['_minimize_trust_krylov'])

# Obtaining an instance of the builtin type 'list' (line 4)
list_204298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 4)
# Adding element type (line 4)
str_204299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'str', '_minimize_trust_krylov')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 10), list_204298, str_204299)

# Assigning a type to the variable '__all__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__all__', list_204298)

@norecursion
def _minimize_trust_krylov(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 6)
    tuple_204300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 6)
    
    # Getting the type of 'None' (line 6)
    None_204301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 49), 'None')
    # Getting the type of 'None' (line 6)
    None_204302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 60), 'None')
    # Getting the type of 'None' (line 6)
    None_204303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 72), 'None')
    # Getting the type of 'True' (line 7)
    True_204304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 35), 'True')
    defaults = [tuple_204300, None_204301, None_204302, None_204303, True_204304]
    # Create a new context for function '_minimize_trust_krylov'
    module_type_store = module_type_store.open_function_context('_minimize_trust_krylov', 6, 0, False)
    
    # Passed parameters checking function
    _minimize_trust_krylov.stypy_localization = localization
    _minimize_trust_krylov.stypy_type_of_self = None
    _minimize_trust_krylov.stypy_type_store = module_type_store
    _minimize_trust_krylov.stypy_function_name = '_minimize_trust_krylov'
    _minimize_trust_krylov.stypy_param_names_list = ['fun', 'x0', 'args', 'jac', 'hess', 'hessp', 'inexact']
    _minimize_trust_krylov.stypy_varargs_param_name = None
    _minimize_trust_krylov.stypy_kwargs_param_name = 'trust_region_options'
    _minimize_trust_krylov.stypy_call_defaults = defaults
    _minimize_trust_krylov.stypy_call_varargs = varargs
    _minimize_trust_krylov.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_minimize_trust_krylov', ['fun', 'x0', 'args', 'jac', 'hess', 'hessp', 'inexact'], None, 'trust_region_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_minimize_trust_krylov', localization, ['fun', 'x0', 'args', 'jac', 'hess', 'hessp', 'inexact'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_minimize_trust_krylov(...)' code ##################

    str_204305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', '\n    Minimization of a scalar function of one or more variables using\n    a nearly exact trust-region algorithm that only requires matrix\n    vector products with the hessian matrix.\n\n    Options\n    -------\n    inexact : bool, optional\n        Accuracy to solve subproblems. If True requires less nonlinear\n        iterations, but more vector products.\n\n    .. versionadded:: 1.0.0\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 22)
    # Getting the type of 'jac' (line 22)
    jac_204306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 7), 'jac')
    # Getting the type of 'None' (line 22)
    None_204307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 14), 'None')
    
    (may_be_204308, more_types_in_union_204309) = may_be_none(jac_204306, None_204307)

    if may_be_204308:

        if more_types_in_union_204309:
            # Runtime conditional SSA (line 22)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 23)
        # Processing the call arguments (line 23)
        str_204311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'str', 'Jacobian is required for trust region ')
        str_204312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'str', 'exact minimization.')
        # Processing the call keyword arguments (line 23)
        kwargs_204313 = {}
        # Getting the type of 'ValueError' (line 23)
        ValueError_204310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 23)
        ValueError_call_result_204314 = invoke(stypy.reporting.localization.Localization(__file__, 23, 14), ValueError_204310, *[str_204311, str_204312], **kwargs_204313)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 23, 8), ValueError_call_result_204314, 'raise parameter', BaseException)

        if more_types_in_union_204309:
            # SSA join for if statement (line 22)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'hess' (line 25)
    hess_204315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 7), 'hess')
    # Getting the type of 'None' (line 25)
    None_204316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'None')
    # Applying the binary operator 'is' (line 25)
    result_is__204317 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 7), 'is', hess_204315, None_204316)
    
    
    # Getting the type of 'hessp' (line 25)
    hessp_204318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'hessp')
    # Getting the type of 'None' (line 25)
    None_204319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 33), 'None')
    # Applying the binary operator 'is' (line 25)
    result_is__204320 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 24), 'is', hessp_204318, None_204319)
    
    # Applying the binary operator 'and' (line 25)
    result_and_keyword_204321 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 7), 'and', result_is__204317, result_is__204320)
    
    # Testing the type of an if condition (line 25)
    if_condition_204322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 4), result_and_keyword_204321)
    # Assigning a type to the variable 'if_condition_204322' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'if_condition_204322', if_condition_204322)
    # SSA begins for if statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 26)
    # Processing the call arguments (line 26)
    str_204324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'str', 'Either the Hessian or the Hessian-vector product is required for Krylov trust-region minimization')
    # Processing the call keyword arguments (line 26)
    kwargs_204325 = {}
    # Getting the type of 'ValueError' (line 26)
    ValueError_204323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 26)
    ValueError_call_result_204326 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), ValueError_204323, *[str_204324], **kwargs_204325)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 26, 8), ValueError_call_result_204326, 'raise parameter', BaseException)
    # SSA join for if statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'inexact' (line 50)
    inexact_204327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'inexact')
    # Testing the type of an if condition (line 50)
    if_condition_204328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), inexact_204327)
    # Assigning a type to the variable 'if_condition_204328' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'if_condition_204328', if_condition_204328)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _minimize_trust_region(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'fun' (line 51)
    fun_204330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 38), 'fun', False)
    # Getting the type of 'x0' (line 51)
    x0_204331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 43), 'x0', False)
    # Processing the call keyword arguments (line 51)
    # Getting the type of 'args' (line 51)
    args_204332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 52), 'args', False)
    keyword_204333 = args_204332
    # Getting the type of 'jac' (line 51)
    jac_204334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 62), 'jac', False)
    keyword_204335 = jac_204334
    # Getting the type of 'hess' (line 52)
    hess_204336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 43), 'hess', False)
    keyword_204337 = hess_204336
    # Getting the type of 'hessp' (line 52)
    hessp_204338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 55), 'hessp', False)
    keyword_204339 = hessp_204338
    
    # Call to get_trlib_quadratic_subproblem(...): (line 53)
    # Processing the call keyword arguments (line 53)
    float_204341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 52), 'float')
    keyword_204342 = float_204341
    float_204343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 68), 'float')
    keyword_204344 = float_204343
    
    # Call to get(...): (line 55)
    # Processing the call arguments (line 55)
    str_204347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 72), 'str', 'disp')
    # Getting the type of 'False' (line 55)
    False_204348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 80), 'False', False)
    # Processing the call keyword arguments (line 55)
    kwargs_204349 = {}
    # Getting the type of 'trust_region_options' (line 55)
    trust_region_options_204345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 47), 'trust_region_options', False)
    # Obtaining the member 'get' of a type (line 55)
    get_204346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 47), trust_region_options_204345, 'get')
    # Calling get(args, kwargs) (line 55)
    get_call_result_204350 = invoke(stypy.reporting.localization.Localization(__file__, 55, 47), get_204346, *[str_204347, False_204348], **kwargs_204349)
    
    keyword_204351 = get_call_result_204350
    kwargs_204352 = {'tol_rel_i': keyword_204342, 'disp': keyword_204351, 'tol_rel_b': keyword_204344}
    # Getting the type of 'get_trlib_quadratic_subproblem' (line 53)
    get_trlib_quadratic_subproblem_204340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 49), 'get_trlib_quadratic_subproblem', False)
    # Calling get_trlib_quadratic_subproblem(args, kwargs) (line 53)
    get_trlib_quadratic_subproblem_call_result_204353 = invoke(stypy.reporting.localization.Localization(__file__, 53, 49), get_trlib_quadratic_subproblem_204340, *[], **kwargs_204352)
    
    keyword_204354 = get_trlib_quadratic_subproblem_call_result_204353
    # Getting the type of 'trust_region_options' (line 57)
    trust_region_options_204355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 40), 'trust_region_options', False)
    kwargs_204356 = {'hessp': keyword_204339, 'args': keyword_204333, 'trust_region_options_204355': trust_region_options_204355, 'subproblem': keyword_204354, 'hess': keyword_204337, 'jac': keyword_204335}
    # Getting the type of '_minimize_trust_region' (line 51)
    _minimize_trust_region_204329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), '_minimize_trust_region', False)
    # Calling _minimize_trust_region(args, kwargs) (line 51)
    _minimize_trust_region_call_result_204357 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), _minimize_trust_region_204329, *[fun_204330, x0_204331], **kwargs_204356)
    
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', _minimize_trust_region_call_result_204357)
    # SSA branch for the else part of an if statement (line 50)
    module_type_store.open_ssa_branch('else')
    
    # Call to _minimize_trust_region(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'fun' (line 59)
    fun_204359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 38), 'fun', False)
    # Getting the type of 'x0' (line 59)
    x0_204360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 43), 'x0', False)
    # Processing the call keyword arguments (line 59)
    # Getting the type of 'args' (line 59)
    args_204361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 52), 'args', False)
    keyword_204362 = args_204361
    # Getting the type of 'jac' (line 59)
    jac_204363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 62), 'jac', False)
    keyword_204364 = jac_204363
    # Getting the type of 'hess' (line 60)
    hess_204365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 43), 'hess', False)
    keyword_204366 = hess_204365
    # Getting the type of 'hessp' (line 60)
    hessp_204367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 55), 'hessp', False)
    keyword_204368 = hessp_204367
    
    # Call to get_trlib_quadratic_subproblem(...): (line 61)
    # Processing the call keyword arguments (line 61)
    float_204370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 52), 'float')
    keyword_204371 = float_204370
    float_204372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 68), 'float')
    keyword_204373 = float_204372
    
    # Call to get(...): (line 63)
    # Processing the call arguments (line 63)
    str_204376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 72), 'str', 'disp')
    # Getting the type of 'False' (line 63)
    False_204377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 80), 'False', False)
    # Processing the call keyword arguments (line 63)
    kwargs_204378 = {}
    # Getting the type of 'trust_region_options' (line 63)
    trust_region_options_204374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'trust_region_options', False)
    # Obtaining the member 'get' of a type (line 63)
    get_204375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 47), trust_region_options_204374, 'get')
    # Calling get(args, kwargs) (line 63)
    get_call_result_204379 = invoke(stypy.reporting.localization.Localization(__file__, 63, 47), get_204375, *[str_204376, False_204377], **kwargs_204378)
    
    keyword_204380 = get_call_result_204379
    kwargs_204381 = {'tol_rel_i': keyword_204371, 'disp': keyword_204380, 'tol_rel_b': keyword_204373}
    # Getting the type of 'get_trlib_quadratic_subproblem' (line 61)
    get_trlib_quadratic_subproblem_204369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 49), 'get_trlib_quadratic_subproblem', False)
    # Calling get_trlib_quadratic_subproblem(args, kwargs) (line 61)
    get_trlib_quadratic_subproblem_call_result_204382 = invoke(stypy.reporting.localization.Localization(__file__, 61, 49), get_trlib_quadratic_subproblem_204369, *[], **kwargs_204381)
    
    keyword_204383 = get_trlib_quadratic_subproblem_call_result_204382
    # Getting the type of 'trust_region_options' (line 65)
    trust_region_options_204384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 40), 'trust_region_options', False)
    kwargs_204385 = {'args': keyword_204362, 'hessp': keyword_204368, 'trust_region_options_204384': trust_region_options_204384, 'subproblem': keyword_204383, 'hess': keyword_204366, 'jac': keyword_204364}
    # Getting the type of '_minimize_trust_region' (line 59)
    _minimize_trust_region_204358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), '_minimize_trust_region', False)
    # Calling _minimize_trust_region(args, kwargs) (line 59)
    _minimize_trust_region_call_result_204386 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), _minimize_trust_region_204358, *[fun_204359, x0_204360], **kwargs_204385)
    
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', _minimize_trust_region_call_result_204386)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_minimize_trust_krylov(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_minimize_trust_krylov' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_204387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204387)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_minimize_trust_krylov'
    return stypy_return_type_204387

# Assigning a type to the variable '_minimize_trust_krylov' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '_minimize_trust_krylov', _minimize_trust_krylov)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
