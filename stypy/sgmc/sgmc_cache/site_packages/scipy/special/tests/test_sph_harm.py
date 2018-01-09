
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_allclose
5: import scipy.special as sc
6: 
7: 
8: def test_first_harmonics():
9:     # Test against explicit representations of the first four
10:     # spherical harmonics which use `theta` as the azimuthal angle,
11:     # `phi` as the polar angle, and include the Condon-Shortley
12:     # phase.
13: 
14:     # Notation is Ymn
15:     def Y00(theta, phi):
16:         return 0.5*np.sqrt(1/np.pi)
17: 
18:     def Yn11(theta, phi):
19:         return 0.5*np.sqrt(3/(2*np.pi))*np.exp(-1j*theta)*np.sin(phi)
20: 
21:     def Y01(theta, phi):
22:         return 0.5*np.sqrt(3/np.pi)*np.cos(phi)
23: 
24:     def Y11(theta, phi):
25:         return -0.5*np.sqrt(3/(2*np.pi))*np.exp(1j*theta)*np.sin(phi)
26: 
27:     harms = [Y00, Yn11, Y01, Y11]
28:     m = [0, -1, 0, 1]
29:     n = [0, 1, 1, 1]
30: 
31:     theta = np.linspace(0, 2*np.pi)
32:     phi = np.linspace(0, np.pi)
33:     theta, phi = np.meshgrid(theta, phi)
34: 
35:     for harm, m, n in zip(harms, m, n):
36:         assert_allclose(sc.sph_harm(m, n, theta, phi),
37:                         harm(theta, phi),
38:                         rtol=1e-15, atol=1e-15,
39:                         err_msg="Y^{}_{} incorrect".format(m, n))
40: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_562791 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_562791) is not StypyTypeError):

    if (import_562791 != 'pyd_module'):
        __import__(import_562791)
        sys_modules_562792 = sys.modules[import_562791]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_562792.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_562791)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_562793 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_562793) is not StypyTypeError):

    if (import_562793 != 'pyd_module'):
        __import__(import_562793)
        sys_modules_562794 = sys.modules[import_562793]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_562794.module_type_store, module_type_store, ['assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_562794, sys_modules_562794.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_allclose'], [assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_562793)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy.special' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_562795 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special')

if (type(import_562795) is not StypyTypeError):

    if (import_562795 != 'pyd_module'):
        __import__(import_562795)
        sys_modules_562796 = sys.modules[import_562795]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sc', sys_modules_562796.module_type_store, module_type_store)
    else:
        import scipy.special as sc

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sc', scipy.special, module_type_store)

else:
    # Assigning a type to the variable 'scipy.special' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', import_562795)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_first_harmonics(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_first_harmonics'
    module_type_store = module_type_store.open_function_context('test_first_harmonics', 8, 0, False)
    
    # Passed parameters checking function
    test_first_harmonics.stypy_localization = localization
    test_first_harmonics.stypy_type_of_self = None
    test_first_harmonics.stypy_type_store = module_type_store
    test_first_harmonics.stypy_function_name = 'test_first_harmonics'
    test_first_harmonics.stypy_param_names_list = []
    test_first_harmonics.stypy_varargs_param_name = None
    test_first_harmonics.stypy_kwargs_param_name = None
    test_first_harmonics.stypy_call_defaults = defaults
    test_first_harmonics.stypy_call_varargs = varargs
    test_first_harmonics.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_first_harmonics', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_first_harmonics', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_first_harmonics(...)' code ##################


    @norecursion
    def Y00(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'Y00'
        module_type_store = module_type_store.open_function_context('Y00', 15, 4, False)
        
        # Passed parameters checking function
        Y00.stypy_localization = localization
        Y00.stypy_type_of_self = None
        Y00.stypy_type_store = module_type_store
        Y00.stypy_function_name = 'Y00'
        Y00.stypy_param_names_list = ['theta', 'phi']
        Y00.stypy_varargs_param_name = None
        Y00.stypy_kwargs_param_name = None
        Y00.stypy_call_defaults = defaults
        Y00.stypy_call_varargs = varargs
        Y00.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'Y00', ['theta', 'phi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'Y00', localization, ['theta', 'phi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'Y00(...)' code ##################

        float_562797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'float')
        
        # Call to sqrt(...): (line 16)
        # Processing the call arguments (line 16)
        int_562800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'int')
        # Getting the type of 'np' (line 16)
        np_562801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 29), 'np', False)
        # Obtaining the member 'pi' of a type (line 16)
        pi_562802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 29), np_562801, 'pi')
        # Applying the binary operator 'div' (line 16)
        result_div_562803 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 27), 'div', int_562800, pi_562802)
        
        # Processing the call keyword arguments (line 16)
        kwargs_562804 = {}
        # Getting the type of 'np' (line 16)
        np_562798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 16)
        sqrt_562799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 19), np_562798, 'sqrt')
        # Calling sqrt(args, kwargs) (line 16)
        sqrt_call_result_562805 = invoke(stypy.reporting.localization.Localization(__file__, 16, 19), sqrt_562799, *[result_div_562803], **kwargs_562804)
        
        # Applying the binary operator '*' (line 16)
        result_mul_562806 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 15), '*', float_562797, sqrt_call_result_562805)
        
        # Assigning a type to the variable 'stypy_return_type' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type', result_mul_562806)
        
        # ################# End of 'Y00(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'Y00' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_562807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562807)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'Y00'
        return stypy_return_type_562807

    # Assigning a type to the variable 'Y00' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'Y00', Y00)

    @norecursion
    def Yn11(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'Yn11'
        module_type_store = module_type_store.open_function_context('Yn11', 18, 4, False)
        
        # Passed parameters checking function
        Yn11.stypy_localization = localization
        Yn11.stypy_type_of_self = None
        Yn11.stypy_type_store = module_type_store
        Yn11.stypy_function_name = 'Yn11'
        Yn11.stypy_param_names_list = ['theta', 'phi']
        Yn11.stypy_varargs_param_name = None
        Yn11.stypy_kwargs_param_name = None
        Yn11.stypy_call_defaults = defaults
        Yn11.stypy_call_varargs = varargs
        Yn11.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'Yn11', ['theta', 'phi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'Yn11', localization, ['theta', 'phi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'Yn11(...)' code ##################

        float_562808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'float')
        
        # Call to sqrt(...): (line 19)
        # Processing the call arguments (line 19)
        int_562811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'int')
        int_562812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 30), 'int')
        # Getting the type of 'np' (line 19)
        np_562813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 32), 'np', False)
        # Obtaining the member 'pi' of a type (line 19)
        pi_562814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 32), np_562813, 'pi')
        # Applying the binary operator '*' (line 19)
        result_mul_562815 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 30), '*', int_562812, pi_562814)
        
        # Applying the binary operator 'div' (line 19)
        result_div_562816 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 27), 'div', int_562811, result_mul_562815)
        
        # Processing the call keyword arguments (line 19)
        kwargs_562817 = {}
        # Getting the type of 'np' (line 19)
        np_562809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 19)
        sqrt_562810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), np_562809, 'sqrt')
        # Calling sqrt(args, kwargs) (line 19)
        sqrt_call_result_562818 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), sqrt_562810, *[result_div_562816], **kwargs_562817)
        
        # Applying the binary operator '*' (line 19)
        result_mul_562819 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 15), '*', float_562808, sqrt_call_result_562818)
        
        
        # Call to exp(...): (line 19)
        # Processing the call arguments (line 19)
        complex_562822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 47), 'complex')
        # Getting the type of 'theta' (line 19)
        theta_562823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 51), 'theta', False)
        # Applying the binary operator '*' (line 19)
        result_mul_562824 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 47), '*', complex_562822, theta_562823)
        
        # Processing the call keyword arguments (line 19)
        kwargs_562825 = {}
        # Getting the type of 'np' (line 19)
        np_562820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 40), 'np', False)
        # Obtaining the member 'exp' of a type (line 19)
        exp_562821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 40), np_562820, 'exp')
        # Calling exp(args, kwargs) (line 19)
        exp_call_result_562826 = invoke(stypy.reporting.localization.Localization(__file__, 19, 40), exp_562821, *[result_mul_562824], **kwargs_562825)
        
        # Applying the binary operator '*' (line 19)
        result_mul_562827 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 39), '*', result_mul_562819, exp_call_result_562826)
        
        
        # Call to sin(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'phi' (line 19)
        phi_562830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 65), 'phi', False)
        # Processing the call keyword arguments (line 19)
        kwargs_562831 = {}
        # Getting the type of 'np' (line 19)
        np_562828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 58), 'np', False)
        # Obtaining the member 'sin' of a type (line 19)
        sin_562829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 58), np_562828, 'sin')
        # Calling sin(args, kwargs) (line 19)
        sin_call_result_562832 = invoke(stypy.reporting.localization.Localization(__file__, 19, 58), sin_562829, *[phi_562830], **kwargs_562831)
        
        # Applying the binary operator '*' (line 19)
        result_mul_562833 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 57), '*', result_mul_562827, sin_call_result_562832)
        
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', result_mul_562833)
        
        # ################# End of 'Yn11(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'Yn11' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_562834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'Yn11'
        return stypy_return_type_562834

    # Assigning a type to the variable 'Yn11' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'Yn11', Yn11)

    @norecursion
    def Y01(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'Y01'
        module_type_store = module_type_store.open_function_context('Y01', 21, 4, False)
        
        # Passed parameters checking function
        Y01.stypy_localization = localization
        Y01.stypy_type_of_self = None
        Y01.stypy_type_store = module_type_store
        Y01.stypy_function_name = 'Y01'
        Y01.stypy_param_names_list = ['theta', 'phi']
        Y01.stypy_varargs_param_name = None
        Y01.stypy_kwargs_param_name = None
        Y01.stypy_call_defaults = defaults
        Y01.stypy_call_varargs = varargs
        Y01.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'Y01', ['theta', 'phi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'Y01', localization, ['theta', 'phi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'Y01(...)' code ##################

        float_562835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'float')
        
        # Call to sqrt(...): (line 22)
        # Processing the call arguments (line 22)
        int_562838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'int')
        # Getting the type of 'np' (line 22)
        np_562839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 29), 'np', False)
        # Obtaining the member 'pi' of a type (line 22)
        pi_562840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 29), np_562839, 'pi')
        # Applying the binary operator 'div' (line 22)
        result_div_562841 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 27), 'div', int_562838, pi_562840)
        
        # Processing the call keyword arguments (line 22)
        kwargs_562842 = {}
        # Getting the type of 'np' (line 22)
        np_562836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 22)
        sqrt_562837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 19), np_562836, 'sqrt')
        # Calling sqrt(args, kwargs) (line 22)
        sqrt_call_result_562843 = invoke(stypy.reporting.localization.Localization(__file__, 22, 19), sqrt_562837, *[result_div_562841], **kwargs_562842)
        
        # Applying the binary operator '*' (line 22)
        result_mul_562844 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 15), '*', float_562835, sqrt_call_result_562843)
        
        
        # Call to cos(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'phi' (line 22)
        phi_562847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 43), 'phi', False)
        # Processing the call keyword arguments (line 22)
        kwargs_562848 = {}
        # Getting the type of 'np' (line 22)
        np_562845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 36), 'np', False)
        # Obtaining the member 'cos' of a type (line 22)
        cos_562846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 36), np_562845, 'cos')
        # Calling cos(args, kwargs) (line 22)
        cos_call_result_562849 = invoke(stypy.reporting.localization.Localization(__file__, 22, 36), cos_562846, *[phi_562847], **kwargs_562848)
        
        # Applying the binary operator '*' (line 22)
        result_mul_562850 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 35), '*', result_mul_562844, cos_call_result_562849)
        
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', result_mul_562850)
        
        # ################# End of 'Y01(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'Y01' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_562851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'Y01'
        return stypy_return_type_562851

    # Assigning a type to the variable 'Y01' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'Y01', Y01)

    @norecursion
    def Y11(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'Y11'
        module_type_store = module_type_store.open_function_context('Y11', 24, 4, False)
        
        # Passed parameters checking function
        Y11.stypy_localization = localization
        Y11.stypy_type_of_self = None
        Y11.stypy_type_store = module_type_store
        Y11.stypy_function_name = 'Y11'
        Y11.stypy_param_names_list = ['theta', 'phi']
        Y11.stypy_varargs_param_name = None
        Y11.stypy_kwargs_param_name = None
        Y11.stypy_call_defaults = defaults
        Y11.stypy_call_varargs = varargs
        Y11.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'Y11', ['theta', 'phi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'Y11', localization, ['theta', 'phi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'Y11(...)' code ##################

        float_562852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'float')
        
        # Call to sqrt(...): (line 25)
        # Processing the call arguments (line 25)
        int_562855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'int')
        int_562856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'int')
        # Getting the type of 'np' (line 25)
        np_562857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 33), 'np', False)
        # Obtaining the member 'pi' of a type (line 25)
        pi_562858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 33), np_562857, 'pi')
        # Applying the binary operator '*' (line 25)
        result_mul_562859 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 31), '*', int_562856, pi_562858)
        
        # Applying the binary operator 'div' (line 25)
        result_div_562860 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 28), 'div', int_562855, result_mul_562859)
        
        # Processing the call keyword arguments (line 25)
        kwargs_562861 = {}
        # Getting the type of 'np' (line 25)
        np_562853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 25)
        sqrt_562854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 20), np_562853, 'sqrt')
        # Calling sqrt(args, kwargs) (line 25)
        sqrt_call_result_562862 = invoke(stypy.reporting.localization.Localization(__file__, 25, 20), sqrt_562854, *[result_div_562860], **kwargs_562861)
        
        # Applying the binary operator '*' (line 25)
        result_mul_562863 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), '*', float_562852, sqrt_call_result_562862)
        
        
        # Call to exp(...): (line 25)
        # Processing the call arguments (line 25)
        complex_562866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 48), 'complex')
        # Getting the type of 'theta' (line 25)
        theta_562867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 51), 'theta', False)
        # Applying the binary operator '*' (line 25)
        result_mul_562868 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 48), '*', complex_562866, theta_562867)
        
        # Processing the call keyword arguments (line 25)
        kwargs_562869 = {}
        # Getting the type of 'np' (line 25)
        np_562864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 41), 'np', False)
        # Obtaining the member 'exp' of a type (line 25)
        exp_562865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 41), np_562864, 'exp')
        # Calling exp(args, kwargs) (line 25)
        exp_call_result_562870 = invoke(stypy.reporting.localization.Localization(__file__, 25, 41), exp_562865, *[result_mul_562868], **kwargs_562869)
        
        # Applying the binary operator '*' (line 25)
        result_mul_562871 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 40), '*', result_mul_562863, exp_call_result_562870)
        
        
        # Call to sin(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'phi' (line 25)
        phi_562874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 65), 'phi', False)
        # Processing the call keyword arguments (line 25)
        kwargs_562875 = {}
        # Getting the type of 'np' (line 25)
        np_562872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 58), 'np', False)
        # Obtaining the member 'sin' of a type (line 25)
        sin_562873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 58), np_562872, 'sin')
        # Calling sin(args, kwargs) (line 25)
        sin_call_result_562876 = invoke(stypy.reporting.localization.Localization(__file__, 25, 58), sin_562873, *[phi_562874], **kwargs_562875)
        
        # Applying the binary operator '*' (line 25)
        result_mul_562877 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 57), '*', result_mul_562871, sin_call_result_562876)
        
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', result_mul_562877)
        
        # ################# End of 'Y11(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'Y11' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_562878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562878)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'Y11'
        return stypy_return_type_562878

    # Assigning a type to the variable 'Y11' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'Y11', Y11)
    
    # Assigning a List to a Name (line 27):
    
    # Assigning a List to a Name (line 27):
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_562879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    # Getting the type of 'Y00' (line 27)
    Y00_562880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'Y00')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 12), list_562879, Y00_562880)
    # Adding element type (line 27)
    # Getting the type of 'Yn11' (line 27)
    Yn11_562881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'Yn11')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 12), list_562879, Yn11_562881)
    # Adding element type (line 27)
    # Getting the type of 'Y01' (line 27)
    Y01_562882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'Y01')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 12), list_562879, Y01_562882)
    # Adding element type (line 27)
    # Getting the type of 'Y11' (line 27)
    Y11_562883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'Y11')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 12), list_562879, Y11_562883)
    
    # Assigning a type to the variable 'harms' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'harms', list_562879)
    
    # Assigning a List to a Name (line 28):
    
    # Assigning a List to a Name (line 28):
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_562884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_562885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_562884, int_562885)
    # Adding element type (line 28)
    int_562886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_562884, int_562886)
    # Adding element type (line 28)
    int_562887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_562884, int_562887)
    # Adding element type (line 28)
    int_562888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_562884, int_562888)
    
    # Assigning a type to the variable 'm' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'm', list_562884)
    
    # Assigning a List to a Name (line 29):
    
    # Assigning a List to a Name (line 29):
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_562889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    int_562890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_562889, int_562890)
    # Adding element type (line 29)
    int_562891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_562889, int_562891)
    # Adding element type (line 29)
    int_562892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_562889, int_562892)
    # Adding element type (line 29)
    int_562893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_562889, int_562893)
    
    # Assigning a type to the variable 'n' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'n', list_562889)
    
    # Assigning a Call to a Name (line 31):
    
    # Assigning a Call to a Name (line 31):
    
    # Call to linspace(...): (line 31)
    # Processing the call arguments (line 31)
    int_562896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'int')
    int_562897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 27), 'int')
    # Getting the type of 'np' (line 31)
    np_562898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 29), 'np', False)
    # Obtaining the member 'pi' of a type (line 31)
    pi_562899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 29), np_562898, 'pi')
    # Applying the binary operator '*' (line 31)
    result_mul_562900 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 27), '*', int_562897, pi_562899)
    
    # Processing the call keyword arguments (line 31)
    kwargs_562901 = {}
    # Getting the type of 'np' (line 31)
    np_562894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'np', False)
    # Obtaining the member 'linspace' of a type (line 31)
    linspace_562895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), np_562894, 'linspace')
    # Calling linspace(args, kwargs) (line 31)
    linspace_call_result_562902 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), linspace_562895, *[int_562896, result_mul_562900], **kwargs_562901)
    
    # Assigning a type to the variable 'theta' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'theta', linspace_call_result_562902)
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to linspace(...): (line 32)
    # Processing the call arguments (line 32)
    int_562905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 22), 'int')
    # Getting the type of 'np' (line 32)
    np_562906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'np', False)
    # Obtaining the member 'pi' of a type (line 32)
    pi_562907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 25), np_562906, 'pi')
    # Processing the call keyword arguments (line 32)
    kwargs_562908 = {}
    # Getting the type of 'np' (line 32)
    np_562903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 10), 'np', False)
    # Obtaining the member 'linspace' of a type (line 32)
    linspace_562904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 10), np_562903, 'linspace')
    # Calling linspace(args, kwargs) (line 32)
    linspace_call_result_562909 = invoke(stypy.reporting.localization.Localization(__file__, 32, 10), linspace_562904, *[int_562905, pi_562907], **kwargs_562908)
    
    # Assigning a type to the variable 'phi' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'phi', linspace_call_result_562909)
    
    # Assigning a Call to a Tuple (line 33):
    
    # Assigning a Subscript to a Name (line 33):
    
    # Obtaining the type of the subscript
    int_562910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'int')
    
    # Call to meshgrid(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'theta' (line 33)
    theta_562913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'theta', False)
    # Getting the type of 'phi' (line 33)
    phi_562914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 36), 'phi', False)
    # Processing the call keyword arguments (line 33)
    kwargs_562915 = {}
    # Getting the type of 'np' (line 33)
    np_562911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 33)
    meshgrid_562912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 17), np_562911, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 33)
    meshgrid_call_result_562916 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), meshgrid_562912, *[theta_562913, phi_562914], **kwargs_562915)
    
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___562917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), meshgrid_call_result_562916, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_562918 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), getitem___562917, int_562910)
    
    # Assigning a type to the variable 'tuple_var_assignment_562789' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'tuple_var_assignment_562789', subscript_call_result_562918)
    
    # Assigning a Subscript to a Name (line 33):
    
    # Obtaining the type of the subscript
    int_562919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'int')
    
    # Call to meshgrid(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'theta' (line 33)
    theta_562922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'theta', False)
    # Getting the type of 'phi' (line 33)
    phi_562923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 36), 'phi', False)
    # Processing the call keyword arguments (line 33)
    kwargs_562924 = {}
    # Getting the type of 'np' (line 33)
    np_562920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 33)
    meshgrid_562921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 17), np_562920, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 33)
    meshgrid_call_result_562925 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), meshgrid_562921, *[theta_562922, phi_562923], **kwargs_562924)
    
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___562926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), meshgrid_call_result_562925, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_562927 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), getitem___562926, int_562919)
    
    # Assigning a type to the variable 'tuple_var_assignment_562790' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'tuple_var_assignment_562790', subscript_call_result_562927)
    
    # Assigning a Name to a Name (line 33):
    # Getting the type of 'tuple_var_assignment_562789' (line 33)
    tuple_var_assignment_562789_562928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'tuple_var_assignment_562789')
    # Assigning a type to the variable 'theta' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'theta', tuple_var_assignment_562789_562928)
    
    # Assigning a Name to a Name (line 33):
    # Getting the type of 'tuple_var_assignment_562790' (line 33)
    tuple_var_assignment_562790_562929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'tuple_var_assignment_562790')
    # Assigning a type to the variable 'phi' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'phi', tuple_var_assignment_562790_562929)
    
    
    # Call to zip(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'harms' (line 35)
    harms_562931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'harms', False)
    # Getting the type of 'm' (line 35)
    m_562932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'm', False)
    # Getting the type of 'n' (line 35)
    n_562933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 36), 'n', False)
    # Processing the call keyword arguments (line 35)
    kwargs_562934 = {}
    # Getting the type of 'zip' (line 35)
    zip_562930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'zip', False)
    # Calling zip(args, kwargs) (line 35)
    zip_call_result_562935 = invoke(stypy.reporting.localization.Localization(__file__, 35, 22), zip_562930, *[harms_562931, m_562932, n_562933], **kwargs_562934)
    
    # Testing the type of a for loop iterable (line 35)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 4), zip_call_result_562935)
    # Getting the type of the for loop variable (line 35)
    for_loop_var_562936 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 4), zip_call_result_562935)
    # Assigning a type to the variable 'harm' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'harm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 4), for_loop_var_562936))
    # Assigning a type to the variable 'm' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 4), for_loop_var_562936))
    # Assigning a type to the variable 'n' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 4), for_loop_var_562936))
    # SSA begins for a for statement (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_allclose(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to sph_harm(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'm' (line 36)
    m_562940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'm', False)
    # Getting the type of 'n' (line 36)
    n_562941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 39), 'n', False)
    # Getting the type of 'theta' (line 36)
    theta_562942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 42), 'theta', False)
    # Getting the type of 'phi' (line 36)
    phi_562943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 49), 'phi', False)
    # Processing the call keyword arguments (line 36)
    kwargs_562944 = {}
    # Getting the type of 'sc' (line 36)
    sc_562938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'sc', False)
    # Obtaining the member 'sph_harm' of a type (line 36)
    sph_harm_562939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 24), sc_562938, 'sph_harm')
    # Calling sph_harm(args, kwargs) (line 36)
    sph_harm_call_result_562945 = invoke(stypy.reporting.localization.Localization(__file__, 36, 24), sph_harm_562939, *[m_562940, n_562941, theta_562942, phi_562943], **kwargs_562944)
    
    
    # Call to harm(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'theta' (line 37)
    theta_562947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'theta', False)
    # Getting the type of 'phi' (line 37)
    phi_562948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 36), 'phi', False)
    # Processing the call keyword arguments (line 37)
    kwargs_562949 = {}
    # Getting the type of 'harm' (line 37)
    harm_562946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'harm', False)
    # Calling harm(args, kwargs) (line 37)
    harm_call_result_562950 = invoke(stypy.reporting.localization.Localization(__file__, 37, 24), harm_562946, *[theta_562947, phi_562948], **kwargs_562949)
    
    # Processing the call keyword arguments (line 36)
    float_562951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 29), 'float')
    keyword_562952 = float_562951
    float_562953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 41), 'float')
    keyword_562954 = float_562953
    
    # Call to format(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'm' (line 39)
    m_562957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 59), 'm', False)
    # Getting the type of 'n' (line 39)
    n_562958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 62), 'n', False)
    # Processing the call keyword arguments (line 39)
    kwargs_562959 = {}
    str_562955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 32), 'str', 'Y^{}_{} incorrect')
    # Obtaining the member 'format' of a type (line 39)
    format_562956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 32), str_562955, 'format')
    # Calling format(args, kwargs) (line 39)
    format_call_result_562960 = invoke(stypy.reporting.localization.Localization(__file__, 39, 32), format_562956, *[m_562957, n_562958], **kwargs_562959)
    
    keyword_562961 = format_call_result_562960
    kwargs_562962 = {'rtol': keyword_562952, 'err_msg': keyword_562961, 'atol': keyword_562954}
    # Getting the type of 'assert_allclose' (line 36)
    assert_allclose_562937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 36)
    assert_allclose_call_result_562963 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assert_allclose_562937, *[sph_harm_call_result_562945, harm_call_result_562950], **kwargs_562962)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_first_harmonics(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_first_harmonics' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_562964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_562964)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_first_harmonics'
    return stypy_return_type_562964

# Assigning a type to the variable 'test_first_harmonics' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'test_first_harmonics', test_first_harmonics)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
