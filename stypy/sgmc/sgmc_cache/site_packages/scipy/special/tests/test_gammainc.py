
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: import numpy as np
3: from scipy.special import gammainc
4: from scipy.special._testutils import FuncData
5: 
6: 
7: def test_line():
8:     # Test on the line a = x where a simpler asymptotic expansion
9:     # (analog of DLMF 8.12.15) is available.
10: 
11:     def gammainc_line(x):
12:         c = np.array([-1/3, -1/540, 25/6048, 101/155520, 
13:                       -3184811/3695155200, -2745493/8151736420])
14:         res = 0
15:         xfac = 1
16:         for ck in c:
17:             res -= ck*xfac
18:             xfac /= x
19:         res /= np.sqrt(2*np.pi*x)
20:         res += 0.5
21:         return res
22: 
23:     x = np.logspace(np.log10(25), 300, 500)
24:     a = x.copy()
25:     dataset = np.vstack((a, x, gammainc_line(x))).T
26: 
27:     FuncData(gammainc, dataset, (0, 1), 2, rtol=1e-11).check()
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_539630 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_539630) is not StypyTypeError):

    if (import_539630 != 'pyd_module'):
        __import__(import_539630)
        sys_modules_539631 = sys.modules[import_539630]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_539631.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_539630)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.special import gammainc' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_539632 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special')

if (type(import_539632) is not StypyTypeError):

    if (import_539632 != 'pyd_module'):
        __import__(import_539632)
        sys_modules_539633 = sys.modules[import_539632]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special', sys_modules_539633.module_type_store, module_type_store, ['gammainc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_539633, sys_modules_539633.module_type_store, module_type_store)
    else:
        from scipy.special import gammainc

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special', None, module_type_store, ['gammainc'], [gammainc])

else:
    # Assigning a type to the variable 'scipy.special' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special', import_539632)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.special._testutils import FuncData' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_539634 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.special._testutils')

if (type(import_539634) is not StypyTypeError):

    if (import_539634 != 'pyd_module'):
        __import__(import_539634)
        sys_modules_539635 = sys.modules[import_539634]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.special._testutils', sys_modules_539635.module_type_store, module_type_store, ['FuncData'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_539635, sys_modules_539635.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import FuncData

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.special._testutils', None, module_type_store, ['FuncData'], [FuncData])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.special._testutils', import_539634)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_line(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_line'
    module_type_store = module_type_store.open_function_context('test_line', 7, 0, False)
    
    # Passed parameters checking function
    test_line.stypy_localization = localization
    test_line.stypy_type_of_self = None
    test_line.stypy_type_store = module_type_store
    test_line.stypy_function_name = 'test_line'
    test_line.stypy_param_names_list = []
    test_line.stypy_varargs_param_name = None
    test_line.stypy_kwargs_param_name = None
    test_line.stypy_call_defaults = defaults
    test_line.stypy_call_varargs = varargs
    test_line.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_line', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_line', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_line(...)' code ##################


    @norecursion
    def gammainc_line(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'gammainc_line'
        module_type_store = module_type_store.open_function_context('gammainc_line', 11, 4, False)
        
        # Passed parameters checking function
        gammainc_line.stypy_localization = localization
        gammainc_line.stypy_type_of_self = None
        gammainc_line.stypy_type_store = module_type_store
        gammainc_line.stypy_function_name = 'gammainc_line'
        gammainc_line.stypy_param_names_list = ['x']
        gammainc_line.stypy_varargs_param_name = None
        gammainc_line.stypy_kwargs_param_name = None
        gammainc_line.stypy_call_defaults = defaults
        gammainc_line.stypy_call_varargs = varargs
        gammainc_line.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'gammainc_line', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'gammainc_line', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'gammainc_line(...)' code ##################

        
        # Assigning a Call to a Name (line 12):
        
        # Call to array(...): (line 12)
        # Processing the call arguments (line 12)
        
        # Obtaining an instance of the builtin type 'list' (line 12)
        list_539638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 12)
        # Adding element type (line 12)
        int_539639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 22), 'int')
        int_539640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_539641 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 22), 'div', int_539639, int_539640)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_539638, result_div_539641)
        # Adding element type (line 12)
        int_539642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 28), 'int')
        int_539643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_539644 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 28), 'div', int_539642, int_539643)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_539638, result_div_539644)
        # Adding element type (line 12)
        int_539645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 36), 'int')
        int_539646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 39), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_539647 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 36), 'div', int_539645, int_539646)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_539638, result_div_539647)
        # Adding element type (line 12)
        int_539648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 45), 'int')
        int_539649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 49), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_539650 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 45), 'div', int_539648, int_539649)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_539638, result_div_539650)
        # Adding element type (line 12)
        int_539651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'int')
        long_539652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 31), 'long')
        # Applying the binary operator 'div' (line 13)
        result_div_539653 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 22), 'div', int_539651, long_539652)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_539638, result_div_539653)
        # Adding element type (line 12)
        int_539654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 43), 'int')
        long_539655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 52), 'long')
        # Applying the binary operator 'div' (line 13)
        result_div_539656 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 43), 'div', int_539654, long_539655)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_539638, result_div_539656)
        
        # Processing the call keyword arguments (line 12)
        kwargs_539657 = {}
        # Getting the type of 'np' (line 12)
        np_539636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 12)
        array_539637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 12), np_539636, 'array')
        # Calling array(args, kwargs) (line 12)
        array_call_result_539658 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), array_539637, *[list_539638], **kwargs_539657)
        
        # Assigning a type to the variable 'c' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'c', array_call_result_539658)
        
        # Assigning a Num to a Name (line 14):
        int_539659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
        # Assigning a type to the variable 'res' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'res', int_539659)
        
        # Assigning a Num to a Name (line 15):
        int_539660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'int')
        # Assigning a type to the variable 'xfac' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'xfac', int_539660)
        
        # Getting the type of 'c' (line 16)
        c_539661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'c')
        # Testing the type of a for loop iterable (line 16)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 8), c_539661)
        # Getting the type of the for loop variable (line 16)
        for_loop_var_539662 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 8), c_539661)
        # Assigning a type to the variable 'ck' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'ck', for_loop_var_539662)
        # SSA begins for a for statement (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'res' (line 17)
        res_539663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'res')
        # Getting the type of 'ck' (line 17)
        ck_539664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'ck')
        # Getting the type of 'xfac' (line 17)
        xfac_539665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'xfac')
        # Applying the binary operator '*' (line 17)
        result_mul_539666 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 19), '*', ck_539664, xfac_539665)
        
        # Applying the binary operator '-=' (line 17)
        result_isub_539667 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 12), '-=', res_539663, result_mul_539666)
        # Assigning a type to the variable 'res' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'res', result_isub_539667)
        
        
        # Getting the type of 'xfac' (line 18)
        xfac_539668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'xfac')
        # Getting the type of 'x' (line 18)
        x_539669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'x')
        # Applying the binary operator 'div=' (line 18)
        result_div_539670 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 12), 'div=', xfac_539668, x_539669)
        # Assigning a type to the variable 'xfac' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'xfac', result_div_539670)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'res' (line 19)
        res_539671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'res')
        
        # Call to sqrt(...): (line 19)
        # Processing the call arguments (line 19)
        int_539674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'int')
        # Getting the type of 'np' (line 19)
        np_539675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'np', False)
        # Obtaining the member 'pi' of a type (line 19)
        pi_539676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 25), np_539675, 'pi')
        # Applying the binary operator '*' (line 19)
        result_mul_539677 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 23), '*', int_539674, pi_539676)
        
        # Getting the type of 'x' (line 19)
        x_539678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'x', False)
        # Applying the binary operator '*' (line 19)
        result_mul_539679 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 30), '*', result_mul_539677, x_539678)
        
        # Processing the call keyword arguments (line 19)
        kwargs_539680 = {}
        # Getting the type of 'np' (line 19)
        np_539672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 19)
        sqrt_539673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), np_539672, 'sqrt')
        # Calling sqrt(args, kwargs) (line 19)
        sqrt_call_result_539681 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), sqrt_539673, *[result_mul_539679], **kwargs_539680)
        
        # Applying the binary operator 'div=' (line 19)
        result_div_539682 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 8), 'div=', res_539671, sqrt_call_result_539681)
        # Assigning a type to the variable 'res' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'res', result_div_539682)
        
        
        # Getting the type of 'res' (line 20)
        res_539683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'res')
        float_539684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'float')
        # Applying the binary operator '+=' (line 20)
        result_iadd_539685 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 8), '+=', res_539683, float_539684)
        # Assigning a type to the variable 'res' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'res', result_iadd_539685)
        
        # Getting the type of 'res' (line 21)
        res_539686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', res_539686)
        
        # ################# End of 'gammainc_line(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'gammainc_line' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_539687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539687)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'gammainc_line'
        return stypy_return_type_539687

    # Assigning a type to the variable 'gammainc_line' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'gammainc_line', gammainc_line)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to logspace(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to log10(...): (line 23)
    # Processing the call arguments (line 23)
    int_539692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_539693 = {}
    # Getting the type of 'np' (line 23)
    np_539690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'np', False)
    # Obtaining the member 'log10' of a type (line 23)
    log10_539691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 20), np_539690, 'log10')
    # Calling log10(args, kwargs) (line 23)
    log10_call_result_539694 = invoke(stypy.reporting.localization.Localization(__file__, 23, 20), log10_539691, *[int_539692], **kwargs_539693)
    
    int_539695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 34), 'int')
    int_539696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 39), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_539697 = {}
    # Getting the type of 'np' (line 23)
    np_539688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'np', False)
    # Obtaining the member 'logspace' of a type (line 23)
    logspace_539689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), np_539688, 'logspace')
    # Calling logspace(args, kwargs) (line 23)
    logspace_call_result_539698 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), logspace_539689, *[log10_call_result_539694, int_539695, int_539696], **kwargs_539697)
    
    # Assigning a type to the variable 'x' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'x', logspace_call_result_539698)
    
    # Assigning a Call to a Name (line 24):
    
    # Call to copy(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_539701 = {}
    # Getting the type of 'x' (line 24)
    x_539699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'x', False)
    # Obtaining the member 'copy' of a type (line 24)
    copy_539700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), x_539699, 'copy')
    # Calling copy(args, kwargs) (line 24)
    copy_call_result_539702 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), copy_539700, *[], **kwargs_539701)
    
    # Assigning a type to the variable 'a' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'a', copy_call_result_539702)
    
    # Assigning a Attribute to a Name (line 25):
    
    # Call to vstack(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Obtaining an instance of the builtin type 'tuple' (line 25)
    tuple_539705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 25)
    # Adding element type (line 25)
    # Getting the type of 'a' (line 25)
    a_539706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 25), tuple_539705, a_539706)
    # Adding element type (line 25)
    # Getting the type of 'x' (line 25)
    x_539707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 25), tuple_539705, x_539707)
    # Adding element type (line 25)
    
    # Call to gammainc_line(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'x' (line 25)
    x_539709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 45), 'x', False)
    # Processing the call keyword arguments (line 25)
    kwargs_539710 = {}
    # Getting the type of 'gammainc_line' (line 25)
    gammainc_line_539708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'gammainc_line', False)
    # Calling gammainc_line(args, kwargs) (line 25)
    gammainc_line_call_result_539711 = invoke(stypy.reporting.localization.Localization(__file__, 25, 31), gammainc_line_539708, *[x_539709], **kwargs_539710)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 25), tuple_539705, gammainc_line_call_result_539711)
    
    # Processing the call keyword arguments (line 25)
    kwargs_539712 = {}
    # Getting the type of 'np' (line 25)
    np_539703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'np', False)
    # Obtaining the member 'vstack' of a type (line 25)
    vstack_539704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 14), np_539703, 'vstack')
    # Calling vstack(args, kwargs) (line 25)
    vstack_call_result_539713 = invoke(stypy.reporting.localization.Localization(__file__, 25, 14), vstack_539704, *[tuple_539705], **kwargs_539712)
    
    # Obtaining the member 'T' of a type (line 25)
    T_539714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 14), vstack_call_result_539713, 'T')
    # Assigning a type to the variable 'dataset' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'dataset', T_539714)
    
    # Call to check(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_539727 = {}
    
    # Call to FuncData(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'gammainc' (line 27)
    gammainc_539716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'gammainc', False)
    # Getting the type of 'dataset' (line 27)
    dataset_539717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 'dataset', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_539718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    int_539719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 33), tuple_539718, int_539719)
    # Adding element type (line 27)
    int_539720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 33), tuple_539718, int_539720)
    
    int_539721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 40), 'int')
    # Processing the call keyword arguments (line 27)
    float_539722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 48), 'float')
    keyword_539723 = float_539722
    kwargs_539724 = {'rtol': keyword_539723}
    # Getting the type of 'FuncData' (line 27)
    FuncData_539715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 27)
    FuncData_call_result_539725 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), FuncData_539715, *[gammainc_539716, dataset_539717, tuple_539718, int_539721], **kwargs_539724)
    
    # Obtaining the member 'check' of a type (line 27)
    check_539726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), FuncData_call_result_539725, 'check')
    # Calling check(args, kwargs) (line 27)
    check_call_result_539728 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), check_539726, *[], **kwargs_539727)
    
    
    # ################# End of 'test_line(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_line' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_539729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_539729)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_line'
    return stypy_return_type_539729

# Assigning a type to the variable 'test_line' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'test_line', test_line)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
