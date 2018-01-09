
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ._trlib import TRLIBQuadraticSubproblem
2: 
3: __all__ = ['TRLIBQuadraticSubproblem', 'get_trlib_quadratic_subproblem']
4: 
5: 
6: def get_trlib_quadratic_subproblem(tol_rel_i=-2.0, tol_rel_b=-3.0, disp=False):
7:     def subproblem_factory(x, fun, jac, hess, hessp):
8:         return TRLIBQuadraticSubproblem(x, fun, jac, hess, hessp,
9:                                         tol_rel_i=tol_rel_i,
10:                                         tol_rel_b=tol_rel_b,
11:                                         disp=disp)
12:     return subproblem_factory
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from scipy.optimize._trlib._trlib import TRLIBQuadraticSubproblem' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_trlib/')
import_255779 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy.optimize._trlib._trlib')

if (type(import_255779) is not StypyTypeError):

    if (import_255779 != 'pyd_module'):
        __import__(import_255779)
        sys_modules_255780 = sys.modules[import_255779]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy.optimize._trlib._trlib', sys_modules_255780.module_type_store, module_type_store, ['TRLIBQuadraticSubproblem'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_255780, sys_modules_255780.module_type_store, module_type_store)
    else:
        from scipy.optimize._trlib._trlib import TRLIBQuadraticSubproblem

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy.optimize._trlib._trlib', None, module_type_store, ['TRLIBQuadraticSubproblem'], [TRLIBQuadraticSubproblem])

else:
    # Assigning a type to the variable 'scipy.optimize._trlib._trlib' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy.optimize._trlib._trlib', import_255779)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_trlib/')


# Assigning a List to a Name (line 3):
__all__ = ['TRLIBQuadraticSubproblem', 'get_trlib_quadratic_subproblem']
module_type_store.set_exportable_members(['TRLIBQuadraticSubproblem', 'get_trlib_quadratic_subproblem'])

# Obtaining an instance of the builtin type 'list' (line 3)
list_255781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
str_255782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'TRLIBQuadraticSubproblem')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_255781, str_255782)
# Adding element type (line 3)
str_255783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 39), 'str', 'get_trlib_quadratic_subproblem')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_255781, str_255783)

# Assigning a type to the variable '__all__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__all__', list_255781)

@norecursion
def get_trlib_quadratic_subproblem(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_255784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 45), 'float')
    float_255785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 61), 'float')
    # Getting the type of 'False' (line 6)
    False_255786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 72), 'False')
    defaults = [float_255784, float_255785, False_255786]
    # Create a new context for function 'get_trlib_quadratic_subproblem'
    module_type_store = module_type_store.open_function_context('get_trlib_quadratic_subproblem', 6, 0, False)
    
    # Passed parameters checking function
    get_trlib_quadratic_subproblem.stypy_localization = localization
    get_trlib_quadratic_subproblem.stypy_type_of_self = None
    get_trlib_quadratic_subproblem.stypy_type_store = module_type_store
    get_trlib_quadratic_subproblem.stypy_function_name = 'get_trlib_quadratic_subproblem'
    get_trlib_quadratic_subproblem.stypy_param_names_list = ['tol_rel_i', 'tol_rel_b', 'disp']
    get_trlib_quadratic_subproblem.stypy_varargs_param_name = None
    get_trlib_quadratic_subproblem.stypy_kwargs_param_name = None
    get_trlib_quadratic_subproblem.stypy_call_defaults = defaults
    get_trlib_quadratic_subproblem.stypy_call_varargs = varargs
    get_trlib_quadratic_subproblem.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_trlib_quadratic_subproblem', ['tol_rel_i', 'tol_rel_b', 'disp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_trlib_quadratic_subproblem', localization, ['tol_rel_i', 'tol_rel_b', 'disp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_trlib_quadratic_subproblem(...)' code ##################


    @norecursion
    def subproblem_factory(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'subproblem_factory'
        module_type_store = module_type_store.open_function_context('subproblem_factory', 7, 4, False)
        
        # Passed parameters checking function
        subproblem_factory.stypy_localization = localization
        subproblem_factory.stypy_type_of_self = None
        subproblem_factory.stypy_type_store = module_type_store
        subproblem_factory.stypy_function_name = 'subproblem_factory'
        subproblem_factory.stypy_param_names_list = ['x', 'fun', 'jac', 'hess', 'hessp']
        subproblem_factory.stypy_varargs_param_name = None
        subproblem_factory.stypy_kwargs_param_name = None
        subproblem_factory.stypy_call_defaults = defaults
        subproblem_factory.stypy_call_varargs = varargs
        subproblem_factory.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'subproblem_factory', ['x', 'fun', 'jac', 'hess', 'hessp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'subproblem_factory', localization, ['x', 'fun', 'jac', 'hess', 'hessp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'subproblem_factory(...)' code ##################

        
        # Call to TRLIBQuadraticSubproblem(...): (line 8)
        # Processing the call arguments (line 8)
        # Getting the type of 'x' (line 8)
        x_255788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 40), 'x', False)
        # Getting the type of 'fun' (line 8)
        fun_255789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 43), 'fun', False)
        # Getting the type of 'jac' (line 8)
        jac_255790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 48), 'jac', False)
        # Getting the type of 'hess' (line 8)
        hess_255791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 53), 'hess', False)
        # Getting the type of 'hessp' (line 8)
        hessp_255792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 59), 'hessp', False)
        # Processing the call keyword arguments (line 8)
        # Getting the type of 'tol_rel_i' (line 9)
        tol_rel_i_255793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 50), 'tol_rel_i', False)
        keyword_255794 = tol_rel_i_255793
        # Getting the type of 'tol_rel_b' (line 10)
        tol_rel_b_255795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 50), 'tol_rel_b', False)
        keyword_255796 = tol_rel_b_255795
        # Getting the type of 'disp' (line 11)
        disp_255797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 45), 'disp', False)
        keyword_255798 = disp_255797
        kwargs_255799 = {'tol_rel_i': keyword_255794, 'disp': keyword_255798, 'tol_rel_b': keyword_255796}
        # Getting the type of 'TRLIBQuadraticSubproblem' (line 8)
        TRLIBQuadraticSubproblem_255787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'TRLIBQuadraticSubproblem', False)
        # Calling TRLIBQuadraticSubproblem(args, kwargs) (line 8)
        TRLIBQuadraticSubproblem_call_result_255800 = invoke(stypy.reporting.localization.Localization(__file__, 8, 15), TRLIBQuadraticSubproblem_255787, *[x_255788, fun_255789, jac_255790, hess_255791, hessp_255792], **kwargs_255799)
        
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', TRLIBQuadraticSubproblem_call_result_255800)
        
        # ################# End of 'subproblem_factory(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'subproblem_factory' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_255801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_255801)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'subproblem_factory'
        return stypy_return_type_255801

    # Assigning a type to the variable 'subproblem_factory' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'subproblem_factory', subproblem_factory)
    # Getting the type of 'subproblem_factory' (line 12)
    subproblem_factory_255802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'subproblem_factory')
    # Assigning a type to the variable 'stypy_return_type' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type', subproblem_factory_255802)
    
    # ################# End of 'get_trlib_quadratic_subproblem(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_trlib_quadratic_subproblem' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_255803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_255803)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_trlib_quadratic_subproblem'
    return stypy_return_type_255803

# Assigning a type to the variable 'get_trlib_quadratic_subproblem' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'get_trlib_quadratic_subproblem', get_trlib_quadratic_subproblem)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
