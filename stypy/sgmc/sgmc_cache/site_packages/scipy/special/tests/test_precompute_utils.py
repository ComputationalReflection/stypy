
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: import numpy as np
3: import pytest
4: 
5: from scipy.special._testutils import MissingModule, check_version
6: from scipy.special._mptestutils import mp_assert_allclose
7: from scipy.special._precompute.utils import lagrange_inversion
8: 
9: try:
10:     import sympy
11: except ImportError:
12:     sympy = MissingModule('sympy')
13: 
14: try:
15:     import mpmath as mp
16: except ImportError:
17:     mp = MissingModule('mpmath')
18: 
19: 
20: _is_32bit_platform = np.intp(0).itemsize < 8
21: 
22: 
23: @pytest.mark.slow
24: @check_version(sympy, '0.7')
25: @check_version(mp, '0.19')
26: class TestInversion(object):
27:     @pytest.mark.xfail(condition=_is_32bit_platform, reason="rtol only 2e-9, see gh-6938")
28:     def test_log(self):
29:         with mp.workdps(30):
30:             logcoeffs = mp.taylor(lambda x: mp.log(1 + x), 0, 10)
31:             expcoeffs = mp.taylor(lambda x: mp.exp(x) - 1, 0, 10)
32:             invlogcoeffs = lagrange_inversion(logcoeffs)
33:             mp_assert_allclose(invlogcoeffs, expcoeffs)
34: 
35:     @pytest.mark.xfail(condition=_is_32bit_platform, reason="rtol only 1e-15, see gh-6938")
36:     def test_sin(self):
37:         with mp.workdps(30):
38:             sincoeffs = mp.taylor(mp.sin, 0, 10)
39:             asincoeffs = mp.taylor(mp.asin, 0, 10)
40:             invsincoeffs = lagrange_inversion(sincoeffs)
41:             mp_assert_allclose(invsincoeffs, asincoeffs, atol=1e-30)
42: 
43: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559676 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_559676) is not StypyTypeError):

    if (import_559676 != 'pyd_module'):
        __import__(import_559676)
        sys_modules_559677 = sys.modules[import_559676]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_559677.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_559676)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import pytest' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559678 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest')

if (type(import_559678) is not StypyTypeError):

    if (import_559678 != 'pyd_module'):
        __import__(import_559678)
        sys_modules_559679 = sys.modules[import_559678]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest', sys_modules_559679.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest', import_559678)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.special._testutils import MissingModule, check_version' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559680 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils')

if (type(import_559680) is not StypyTypeError):

    if (import_559680 != 'pyd_module'):
        __import__(import_559680)
        sys_modules_559681 = sys.modules[import_559680]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils', sys_modules_559681.module_type_store, module_type_store, ['MissingModule', 'check_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_559681, sys_modules_559681.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import MissingModule, check_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils', None, module_type_store, ['MissingModule', 'check_version'], [MissingModule, check_version])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils', import_559680)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special._mptestutils import mp_assert_allclose' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._mptestutils')

if (type(import_559682) is not StypyTypeError):

    if (import_559682 != 'pyd_module'):
        __import__(import_559682)
        sys_modules_559683 = sys.modules[import_559682]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._mptestutils', sys_modules_559683.module_type_store, module_type_store, ['mp_assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_559683, sys_modules_559683.module_type_store, module_type_store)
    else:
        from scipy.special._mptestutils import mp_assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._mptestutils', None, module_type_store, ['mp_assert_allclose'], [mp_assert_allclose])

else:
    # Assigning a type to the variable 'scipy.special._mptestutils' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._mptestutils', import_559682)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.special._precompute.utils import lagrange_inversion' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._precompute.utils')

if (type(import_559684) is not StypyTypeError):

    if (import_559684 != 'pyd_module'):
        __import__(import_559684)
        sys_modules_559685 = sys.modules[import_559684]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._precompute.utils', sys_modules_559685.module_type_store, module_type_store, ['lagrange_inversion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_559685, sys_modules_559685.module_type_store, module_type_store)
    else:
        from scipy.special._precompute.utils import lagrange_inversion

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._precompute.utils', None, module_type_store, ['lagrange_inversion'], [lagrange_inversion])

else:
    # Assigning a type to the variable 'scipy.special._precompute.utils' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._precompute.utils', import_559684)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')



# SSA begins for try-except statement (line 9)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))

# 'import sympy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559686 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'sympy')

if (type(import_559686) is not StypyTypeError):

    if (import_559686 != 'pyd_module'):
        __import__(import_559686)
        sys_modules_559687 = sys.modules[import_559686]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'sympy', sys_modules_559687.module_type_store, module_type_store)
    else:
        import sympy

        import_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'sympy', sympy, module_type_store)

else:
    # Assigning a type to the variable 'sympy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'sympy', import_559686)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

# SSA branch for the except part of a try statement (line 9)
# SSA branch for the except 'ImportError' branch of a try statement (line 9)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 12):

# Call to MissingModule(...): (line 12)
# Processing the call arguments (line 12)
str_559689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'str', 'sympy')
# Processing the call keyword arguments (line 12)
kwargs_559690 = {}
# Getting the type of 'MissingModule' (line 12)
MissingModule_559688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'MissingModule', False)
# Calling MissingModule(args, kwargs) (line 12)
MissingModule_call_result_559691 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), MissingModule_559688, *[str_559689], **kwargs_559690)

# Assigning a type to the variable 'sympy' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'sympy', MissingModule_call_result_559691)
# SSA join for try-except statement (line 9)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 14)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 4))

# 'import mpmath' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559692 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'mpmath')

if (type(import_559692) is not StypyTypeError):

    if (import_559692 != 'pyd_module'):
        __import__(import_559692)
        sys_modules_559693 = sys.modules[import_559692]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'mp', sys_modules_559693.module_type_store, module_type_store)
    else:
        import mpmath as mp

        import_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'mp', mpmath, module_type_store)

else:
    # Assigning a type to the variable 'mpmath' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'mpmath', import_559692)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

# SSA branch for the except part of a try statement (line 14)
# SSA branch for the except 'ImportError' branch of a try statement (line 14)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 17):

# Call to MissingModule(...): (line 17)
# Processing the call arguments (line 17)
str_559695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'str', 'mpmath')
# Processing the call keyword arguments (line 17)
kwargs_559696 = {}
# Getting the type of 'MissingModule' (line 17)
MissingModule_559694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 9), 'MissingModule', False)
# Calling MissingModule(args, kwargs) (line 17)
MissingModule_call_result_559697 = invoke(stypy.reporting.localization.Localization(__file__, 17, 9), MissingModule_559694, *[str_559695], **kwargs_559696)

# Assigning a type to the variable 'mp' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'mp', MissingModule_call_result_559697)
# SSA join for try-except statement (line 14)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Compare to a Name (line 20):


# Call to intp(...): (line 20)
# Processing the call arguments (line 20)
int_559700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 29), 'int')
# Processing the call keyword arguments (line 20)
kwargs_559701 = {}
# Getting the type of 'np' (line 20)
np_559698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'np', False)
# Obtaining the member 'intp' of a type (line 20)
intp_559699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 21), np_559698, 'intp')
# Calling intp(args, kwargs) (line 20)
intp_call_result_559702 = invoke(stypy.reporting.localization.Localization(__file__, 20, 21), intp_559699, *[int_559700], **kwargs_559701)

# Obtaining the member 'itemsize' of a type (line 20)
itemsize_559703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 21), intp_call_result_559702, 'itemsize')
int_559704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 43), 'int')
# Applying the binary operator '<' (line 20)
result_lt_559705 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 21), '<', itemsize_559703, int_559704)

# Assigning a type to the variable '_is_32bit_platform' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '_is_32bit_platform', result_lt_559705)
# Declaration of the 'TestInversion' class

class TestInversion(object, ):

    @norecursion
    def test_log(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_log'
        module_type_store = module_type_store.open_function_context('test_log', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInversion.test_log.__dict__.__setitem__('stypy_localization', localization)
        TestInversion.test_log.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInversion.test_log.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInversion.test_log.__dict__.__setitem__('stypy_function_name', 'TestInversion.test_log')
        TestInversion.test_log.__dict__.__setitem__('stypy_param_names_list', [])
        TestInversion.test_log.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInversion.test_log.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInversion.test_log.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInversion.test_log.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInversion.test_log.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInversion.test_log.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInversion.test_log', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_log', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_log(...)' code ##################

        
        # Call to workdps(...): (line 29)
        # Processing the call arguments (line 29)
        int_559708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 24), 'int')
        # Processing the call keyword arguments (line 29)
        kwargs_559709 = {}
        # Getting the type of 'mp' (line 29)
        mp_559706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'mp', False)
        # Obtaining the member 'workdps' of a type (line 29)
        workdps_559707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 13), mp_559706, 'workdps')
        # Calling workdps(args, kwargs) (line 29)
        workdps_call_result_559710 = invoke(stypy.reporting.localization.Localization(__file__, 29, 13), workdps_559707, *[int_559708], **kwargs_559709)
        
        with_559711 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 29, 13), workdps_call_result_559710, 'with parameter', '__enter__', '__exit__')

        if with_559711:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 29)
            enter___559712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 13), workdps_call_result_559710, '__enter__')
            with_enter_559713 = invoke(stypy.reporting.localization.Localization(__file__, 29, 13), enter___559712)
            
            # Assigning a Call to a Name (line 30):
            
            # Call to taylor(...): (line 30)
            # Processing the call arguments (line 30)

            @norecursion
            def _stypy_temp_lambda_481(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_481'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_481', 30, 34, True)
                # Passed parameters checking function
                _stypy_temp_lambda_481.stypy_localization = localization
                _stypy_temp_lambda_481.stypy_type_of_self = None
                _stypy_temp_lambda_481.stypy_type_store = module_type_store
                _stypy_temp_lambda_481.stypy_function_name = '_stypy_temp_lambda_481'
                _stypy_temp_lambda_481.stypy_param_names_list = ['x']
                _stypy_temp_lambda_481.stypy_varargs_param_name = None
                _stypy_temp_lambda_481.stypy_kwargs_param_name = None
                _stypy_temp_lambda_481.stypy_call_defaults = defaults
                _stypy_temp_lambda_481.stypy_call_varargs = varargs
                _stypy_temp_lambda_481.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_481', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_481', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to log(...): (line 30)
                # Processing the call arguments (line 30)
                int_559718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 51), 'int')
                # Getting the type of 'x' (line 30)
                x_559719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 55), 'x', False)
                # Applying the binary operator '+' (line 30)
                result_add_559720 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 51), '+', int_559718, x_559719)
                
                # Processing the call keyword arguments (line 30)
                kwargs_559721 = {}
                # Getting the type of 'mp' (line 30)
                mp_559716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 44), 'mp', False)
                # Obtaining the member 'log' of a type (line 30)
                log_559717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 44), mp_559716, 'log')
                # Calling log(args, kwargs) (line 30)
                log_call_result_559722 = invoke(stypy.reporting.localization.Localization(__file__, 30, 44), log_559717, *[result_add_559720], **kwargs_559721)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 30)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'stypy_return_type', log_call_result_559722)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_481' in the type store
                # Getting the type of 'stypy_return_type' (line 30)
                stypy_return_type_559723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_559723)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_481'
                return stypy_return_type_559723

            # Assigning a type to the variable '_stypy_temp_lambda_481' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), '_stypy_temp_lambda_481', _stypy_temp_lambda_481)
            # Getting the type of '_stypy_temp_lambda_481' (line 30)
            _stypy_temp_lambda_481_559724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), '_stypy_temp_lambda_481')
            int_559725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 59), 'int')
            int_559726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 62), 'int')
            # Processing the call keyword arguments (line 30)
            kwargs_559727 = {}
            # Getting the type of 'mp' (line 30)
            mp_559714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'mp', False)
            # Obtaining the member 'taylor' of a type (line 30)
            taylor_559715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 24), mp_559714, 'taylor')
            # Calling taylor(args, kwargs) (line 30)
            taylor_call_result_559728 = invoke(stypy.reporting.localization.Localization(__file__, 30, 24), taylor_559715, *[_stypy_temp_lambda_481_559724, int_559725, int_559726], **kwargs_559727)
            
            # Assigning a type to the variable 'logcoeffs' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'logcoeffs', taylor_call_result_559728)
            
            # Assigning a Call to a Name (line 31):
            
            # Call to taylor(...): (line 31)
            # Processing the call arguments (line 31)

            @norecursion
            def _stypy_temp_lambda_482(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_482'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_482', 31, 34, True)
                # Passed parameters checking function
                _stypy_temp_lambda_482.stypy_localization = localization
                _stypy_temp_lambda_482.stypy_type_of_self = None
                _stypy_temp_lambda_482.stypy_type_store = module_type_store
                _stypy_temp_lambda_482.stypy_function_name = '_stypy_temp_lambda_482'
                _stypy_temp_lambda_482.stypy_param_names_list = ['x']
                _stypy_temp_lambda_482.stypy_varargs_param_name = None
                _stypy_temp_lambda_482.stypy_kwargs_param_name = None
                _stypy_temp_lambda_482.stypy_call_defaults = defaults
                _stypy_temp_lambda_482.stypy_call_varargs = varargs
                _stypy_temp_lambda_482.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_482', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_482', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to exp(...): (line 31)
                # Processing the call arguments (line 31)
                # Getting the type of 'x' (line 31)
                x_559733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 51), 'x', False)
                # Processing the call keyword arguments (line 31)
                kwargs_559734 = {}
                # Getting the type of 'mp' (line 31)
                mp_559731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 44), 'mp', False)
                # Obtaining the member 'exp' of a type (line 31)
                exp_559732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 44), mp_559731, 'exp')
                # Calling exp(args, kwargs) (line 31)
                exp_call_result_559735 = invoke(stypy.reporting.localization.Localization(__file__, 31, 44), exp_559732, *[x_559733], **kwargs_559734)
                
                int_559736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 56), 'int')
                # Applying the binary operator '-' (line 31)
                result_sub_559737 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 44), '-', exp_call_result_559735, int_559736)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 31)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 'stypy_return_type', result_sub_559737)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_482' in the type store
                # Getting the type of 'stypy_return_type' (line 31)
                stypy_return_type_559738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_559738)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_482'
                return stypy_return_type_559738

            # Assigning a type to the variable '_stypy_temp_lambda_482' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), '_stypy_temp_lambda_482', _stypy_temp_lambda_482)
            # Getting the type of '_stypy_temp_lambda_482' (line 31)
            _stypy_temp_lambda_482_559739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), '_stypy_temp_lambda_482')
            int_559740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 59), 'int')
            int_559741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 62), 'int')
            # Processing the call keyword arguments (line 31)
            kwargs_559742 = {}
            # Getting the type of 'mp' (line 31)
            mp_559729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'mp', False)
            # Obtaining the member 'taylor' of a type (line 31)
            taylor_559730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 24), mp_559729, 'taylor')
            # Calling taylor(args, kwargs) (line 31)
            taylor_call_result_559743 = invoke(stypy.reporting.localization.Localization(__file__, 31, 24), taylor_559730, *[_stypy_temp_lambda_482_559739, int_559740, int_559741], **kwargs_559742)
            
            # Assigning a type to the variable 'expcoeffs' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'expcoeffs', taylor_call_result_559743)
            
            # Assigning a Call to a Name (line 32):
            
            # Call to lagrange_inversion(...): (line 32)
            # Processing the call arguments (line 32)
            # Getting the type of 'logcoeffs' (line 32)
            logcoeffs_559745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 46), 'logcoeffs', False)
            # Processing the call keyword arguments (line 32)
            kwargs_559746 = {}
            # Getting the type of 'lagrange_inversion' (line 32)
            lagrange_inversion_559744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 27), 'lagrange_inversion', False)
            # Calling lagrange_inversion(args, kwargs) (line 32)
            lagrange_inversion_call_result_559747 = invoke(stypy.reporting.localization.Localization(__file__, 32, 27), lagrange_inversion_559744, *[logcoeffs_559745], **kwargs_559746)
            
            # Assigning a type to the variable 'invlogcoeffs' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'invlogcoeffs', lagrange_inversion_call_result_559747)
            
            # Call to mp_assert_allclose(...): (line 33)
            # Processing the call arguments (line 33)
            # Getting the type of 'invlogcoeffs' (line 33)
            invlogcoeffs_559749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'invlogcoeffs', False)
            # Getting the type of 'expcoeffs' (line 33)
            expcoeffs_559750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 45), 'expcoeffs', False)
            # Processing the call keyword arguments (line 33)
            kwargs_559751 = {}
            # Getting the type of 'mp_assert_allclose' (line 33)
            mp_assert_allclose_559748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'mp_assert_allclose', False)
            # Calling mp_assert_allclose(args, kwargs) (line 33)
            mp_assert_allclose_call_result_559752 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), mp_assert_allclose_559748, *[invlogcoeffs_559749, expcoeffs_559750], **kwargs_559751)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 29)
            exit___559753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 13), workdps_call_result_559710, '__exit__')
            with_exit_559754 = invoke(stypy.reporting.localization.Localization(__file__, 29, 13), exit___559753, None, None, None)

        
        # ################# End of 'test_log(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_log' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_559755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_559755)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_log'
        return stypy_return_type_559755


    @norecursion
    def test_sin(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sin'
        module_type_store = module_type_store.open_function_context('test_sin', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestInversion.test_sin.__dict__.__setitem__('stypy_localization', localization)
        TestInversion.test_sin.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestInversion.test_sin.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestInversion.test_sin.__dict__.__setitem__('stypy_function_name', 'TestInversion.test_sin')
        TestInversion.test_sin.__dict__.__setitem__('stypy_param_names_list', [])
        TestInversion.test_sin.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestInversion.test_sin.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestInversion.test_sin.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestInversion.test_sin.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestInversion.test_sin.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestInversion.test_sin.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInversion.test_sin', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sin', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sin(...)' code ##################

        
        # Call to workdps(...): (line 37)
        # Processing the call arguments (line 37)
        int_559758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'int')
        # Processing the call keyword arguments (line 37)
        kwargs_559759 = {}
        # Getting the type of 'mp' (line 37)
        mp_559756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'mp', False)
        # Obtaining the member 'workdps' of a type (line 37)
        workdps_559757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), mp_559756, 'workdps')
        # Calling workdps(args, kwargs) (line 37)
        workdps_call_result_559760 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), workdps_559757, *[int_559758], **kwargs_559759)
        
        with_559761 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 37, 13), workdps_call_result_559760, 'with parameter', '__enter__', '__exit__')

        if with_559761:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 37)
            enter___559762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), workdps_call_result_559760, '__enter__')
            with_enter_559763 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), enter___559762)
            
            # Assigning a Call to a Name (line 38):
            
            # Call to taylor(...): (line 38)
            # Processing the call arguments (line 38)
            # Getting the type of 'mp' (line 38)
            mp_559766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 34), 'mp', False)
            # Obtaining the member 'sin' of a type (line 38)
            sin_559767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 34), mp_559766, 'sin')
            int_559768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 42), 'int')
            int_559769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 45), 'int')
            # Processing the call keyword arguments (line 38)
            kwargs_559770 = {}
            # Getting the type of 'mp' (line 38)
            mp_559764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'mp', False)
            # Obtaining the member 'taylor' of a type (line 38)
            taylor_559765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), mp_559764, 'taylor')
            # Calling taylor(args, kwargs) (line 38)
            taylor_call_result_559771 = invoke(stypy.reporting.localization.Localization(__file__, 38, 24), taylor_559765, *[sin_559767, int_559768, int_559769], **kwargs_559770)
            
            # Assigning a type to the variable 'sincoeffs' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'sincoeffs', taylor_call_result_559771)
            
            # Assigning a Call to a Name (line 39):
            
            # Call to taylor(...): (line 39)
            # Processing the call arguments (line 39)
            # Getting the type of 'mp' (line 39)
            mp_559774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 35), 'mp', False)
            # Obtaining the member 'asin' of a type (line 39)
            asin_559775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 35), mp_559774, 'asin')
            int_559776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 44), 'int')
            int_559777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 47), 'int')
            # Processing the call keyword arguments (line 39)
            kwargs_559778 = {}
            # Getting the type of 'mp' (line 39)
            mp_559772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'mp', False)
            # Obtaining the member 'taylor' of a type (line 39)
            taylor_559773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 25), mp_559772, 'taylor')
            # Calling taylor(args, kwargs) (line 39)
            taylor_call_result_559779 = invoke(stypy.reporting.localization.Localization(__file__, 39, 25), taylor_559773, *[asin_559775, int_559776, int_559777], **kwargs_559778)
            
            # Assigning a type to the variable 'asincoeffs' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'asincoeffs', taylor_call_result_559779)
            
            # Assigning a Call to a Name (line 40):
            
            # Call to lagrange_inversion(...): (line 40)
            # Processing the call arguments (line 40)
            # Getting the type of 'sincoeffs' (line 40)
            sincoeffs_559781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 46), 'sincoeffs', False)
            # Processing the call keyword arguments (line 40)
            kwargs_559782 = {}
            # Getting the type of 'lagrange_inversion' (line 40)
            lagrange_inversion_559780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'lagrange_inversion', False)
            # Calling lagrange_inversion(args, kwargs) (line 40)
            lagrange_inversion_call_result_559783 = invoke(stypy.reporting.localization.Localization(__file__, 40, 27), lagrange_inversion_559780, *[sincoeffs_559781], **kwargs_559782)
            
            # Assigning a type to the variable 'invsincoeffs' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'invsincoeffs', lagrange_inversion_call_result_559783)
            
            # Call to mp_assert_allclose(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'invsincoeffs' (line 41)
            invsincoeffs_559785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'invsincoeffs', False)
            # Getting the type of 'asincoeffs' (line 41)
            asincoeffs_559786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 45), 'asincoeffs', False)
            # Processing the call keyword arguments (line 41)
            float_559787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 62), 'float')
            keyword_559788 = float_559787
            kwargs_559789 = {'atol': keyword_559788}
            # Getting the type of 'mp_assert_allclose' (line 41)
            mp_assert_allclose_559784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'mp_assert_allclose', False)
            # Calling mp_assert_allclose(args, kwargs) (line 41)
            mp_assert_allclose_call_result_559790 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), mp_assert_allclose_559784, *[invsincoeffs_559785, asincoeffs_559786], **kwargs_559789)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 37)
            exit___559791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), workdps_call_result_559760, '__exit__')
            with_exit_559792 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), exit___559791, None, None, None)

        
        # ################# End of 'test_sin(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sin' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_559793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_559793)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sin'
        return stypy_return_type_559793


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 0, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestInversion.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TestInversion' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'TestInversion', TestInversion)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
