
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: import numpy as np
3: from numpy.testing import assert_equal, assert_allclose
4: from scipy.integrate import odeint
5: import scipy.integrate._test_odeint_banded as banded5x5
6: 
7: 
8: def rhs(y, t):
9:     dydt = np.zeros_like(y)
10:     banded5x5.banded5x5(t, y, dydt)
11:     return dydt
12: 
13: 
14: def jac(y, t):
15:     n = len(y)
16:     jac = np.zeros((n, n), order='F')
17:     banded5x5.banded5x5_jac(t, y, 1, 1, jac)
18:     return jac
19: 
20: 
21: def bjac(y, t):
22:     n = len(y)
23:     bjac = np.zeros((4, n), order='F')
24:     banded5x5.banded5x5_bjac(t, y, 1, 1, bjac)
25:     return bjac
26: 
27: 
28: JACTYPE_FULL = 1
29: JACTYPE_BANDED = 4
30: 
31: 
32: def check_odeint(jactype):
33:     if jactype == JACTYPE_FULL:
34:         ml = None
35:         mu = None
36:         jacobian = jac
37:     elif jactype == JACTYPE_BANDED:
38:         ml = 2
39:         mu = 1
40:         jacobian = bjac
41:     else:
42:         raise ValueError("invalid jactype: %r" % (jactype,))
43: 
44:     y0 = np.arange(1.0, 6.0)
45:     # These tolerances must match the tolerances used in banded5x5.f.
46:     rtol = 1e-11
47:     atol = 1e-13
48:     dt = 0.125
49:     nsteps = 64
50:     t = dt * np.arange(nsteps+1)
51: 
52:     sol, info = odeint(rhs, y0, t,
53:                        Dfun=jacobian, ml=ml, mu=mu,
54:                        atol=atol, rtol=rtol, full_output=True)
55:     yfinal = sol[-1]
56:     odeint_nst = info['nst'][-1]
57:     odeint_nfe = info['nfe'][-1]
58:     odeint_nje = info['nje'][-1]
59: 
60:     y1 = y0.copy()
61:     # Pure Fortran solution.  y1 is modified in-place.
62:     nst, nfe, nje = banded5x5.banded5x5_solve(y1, nsteps, dt, jactype)
63: 
64:     # It is likely that yfinal and y1 are *exactly* the same, but
65:     # we'll be cautious and use assert_allclose.
66:     assert_allclose(yfinal, y1, rtol=1e-12)
67:     assert_equal((odeint_nst, odeint_nfe, odeint_nje), (nst, nfe, nje))
68: 
69: 
70: def test_odeint_full_jac():
71:     check_odeint(JACTYPE_FULL)
72: 
73: 
74: def test_odeint_banded_jac():
75:     check_odeint(JACTYPE_BANDED)
76: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49028 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_49028) is not StypyTypeError):

    if (import_49028 != 'pyd_module'):
        __import__(import_49028)
        sys_modules_49029 = sys.modules[import_49028]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_49029.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_49028)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.testing import assert_equal, assert_allclose' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49030 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_49030) is not StypyTypeError):

    if (import_49030 != 'pyd_module'):
        __import__(import_49030)
        sys_modules_49031 = sys.modules[import_49030]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', sys_modules_49031.module_type_store, module_type_store, ['assert_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_49031, sys_modules_49031.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose'], [assert_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_49030)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.integrate import odeint' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49032 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate')

if (type(import_49032) is not StypyTypeError):

    if (import_49032 != 'pyd_module'):
        __import__(import_49032)
        sys_modules_49033 = sys.modules[import_49032]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate', sys_modules_49033.module_type_store, module_type_store, ['odeint'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_49033, sys_modules_49033.module_type_store, module_type_store)
    else:
        from scipy.integrate import odeint

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate', None, module_type_store, ['odeint'], [odeint])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate', import_49032)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy.integrate._test_odeint_banded' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49034 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.integrate._test_odeint_banded')

if (type(import_49034) is not StypyTypeError):

    if (import_49034 != 'pyd_module'):
        __import__(import_49034)
        sys_modules_49035 = sys.modules[import_49034]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'banded5x5', sys_modules_49035.module_type_store, module_type_store)
    else:
        import scipy.integrate._test_odeint_banded as banded5x5

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'banded5x5', scipy.integrate._test_odeint_banded, module_type_store)

else:
    # Assigning a type to the variable 'scipy.integrate._test_odeint_banded' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.integrate._test_odeint_banded', import_49034)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')


@norecursion
def rhs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rhs'
    module_type_store = module_type_store.open_function_context('rhs', 8, 0, False)
    
    # Passed parameters checking function
    rhs.stypy_localization = localization
    rhs.stypy_type_of_self = None
    rhs.stypy_type_store = module_type_store
    rhs.stypy_function_name = 'rhs'
    rhs.stypy_param_names_list = ['y', 't']
    rhs.stypy_varargs_param_name = None
    rhs.stypy_kwargs_param_name = None
    rhs.stypy_call_defaults = defaults
    rhs.stypy_call_varargs = varargs
    rhs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rhs', ['y', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rhs', localization, ['y', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rhs(...)' code ##################

    
    # Assigning a Call to a Name (line 9):
    
    # Assigning a Call to a Name (line 9):
    
    # Call to zeros_like(...): (line 9)
    # Processing the call arguments (line 9)
    # Getting the type of 'y' (line 9)
    y_49038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 25), 'y', False)
    # Processing the call keyword arguments (line 9)
    kwargs_49039 = {}
    # Getting the type of 'np' (line 9)
    np_49036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 9)
    zeros_like_49037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 11), np_49036, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 9)
    zeros_like_call_result_49040 = invoke(stypy.reporting.localization.Localization(__file__, 9, 11), zeros_like_49037, *[y_49038], **kwargs_49039)
    
    # Assigning a type to the variable 'dydt' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'dydt', zeros_like_call_result_49040)
    
    # Call to banded5x5(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 't' (line 10)
    t_49043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 24), 't', False)
    # Getting the type of 'y' (line 10)
    y_49044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 27), 'y', False)
    # Getting the type of 'dydt' (line 10)
    dydt_49045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 30), 'dydt', False)
    # Processing the call keyword arguments (line 10)
    kwargs_49046 = {}
    # Getting the type of 'banded5x5' (line 10)
    banded5x5_49041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'banded5x5', False)
    # Obtaining the member 'banded5x5' of a type (line 10)
    banded5x5_49042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), banded5x5_49041, 'banded5x5')
    # Calling banded5x5(args, kwargs) (line 10)
    banded5x5_call_result_49047 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), banded5x5_49042, *[t_49043, y_49044, dydt_49045], **kwargs_49046)
    
    # Getting the type of 'dydt' (line 11)
    dydt_49048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'dydt')
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', dydt_49048)
    
    # ################# End of 'rhs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rhs' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_49049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_49049)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rhs'
    return stypy_return_type_49049

# Assigning a type to the variable 'rhs' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'rhs', rhs)

@norecursion
def jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'jac'
    module_type_store = module_type_store.open_function_context('jac', 14, 0, False)
    
    # Passed parameters checking function
    jac.stypy_localization = localization
    jac.stypy_type_of_self = None
    jac.stypy_type_store = module_type_store
    jac.stypy_function_name = 'jac'
    jac.stypy_param_names_list = ['y', 't']
    jac.stypy_varargs_param_name = None
    jac.stypy_kwargs_param_name = None
    jac.stypy_call_defaults = defaults
    jac.stypy_call_varargs = varargs
    jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'jac', ['y', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'jac', localization, ['y', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'jac(...)' code ##################

    
    # Assigning a Call to a Name (line 15):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to len(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'y' (line 15)
    y_49051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'y', False)
    # Processing the call keyword arguments (line 15)
    kwargs_49052 = {}
    # Getting the type of 'len' (line 15)
    len_49050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'len', False)
    # Calling len(args, kwargs) (line 15)
    len_call_result_49053 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), len_49050, *[y_49051], **kwargs_49052)
    
    # Assigning a type to the variable 'n' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'n', len_call_result_49053)
    
    # Assigning a Call to a Name (line 16):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to zeros(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_49056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    # Getting the type of 'n' (line 16)
    n_49057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), tuple_49056, n_49057)
    # Adding element type (line 16)
    # Getting the type of 'n' (line 16)
    n_49058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 20), tuple_49056, n_49058)
    
    # Processing the call keyword arguments (line 16)
    str_49059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'str', 'F')
    keyword_49060 = str_49059
    kwargs_49061 = {'order': keyword_49060}
    # Getting the type of 'np' (line 16)
    np_49054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 16)
    zeros_49055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 10), np_49054, 'zeros')
    # Calling zeros(args, kwargs) (line 16)
    zeros_call_result_49062 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), zeros_49055, *[tuple_49056], **kwargs_49061)
    
    # Assigning a type to the variable 'jac' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'jac', zeros_call_result_49062)
    
    # Call to banded5x5_jac(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 't' (line 17)
    t_49065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 28), 't', False)
    # Getting the type of 'y' (line 17)
    y_49066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 31), 'y', False)
    int_49067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'int')
    int_49068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 37), 'int')
    # Getting the type of 'jac' (line 17)
    jac_49069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 40), 'jac', False)
    # Processing the call keyword arguments (line 17)
    kwargs_49070 = {}
    # Getting the type of 'banded5x5' (line 17)
    banded5x5_49063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'banded5x5', False)
    # Obtaining the member 'banded5x5_jac' of a type (line 17)
    banded5x5_jac_49064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), banded5x5_49063, 'banded5x5_jac')
    # Calling banded5x5_jac(args, kwargs) (line 17)
    banded5x5_jac_call_result_49071 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), banded5x5_jac_49064, *[t_49065, y_49066, int_49067, int_49068, jac_49069], **kwargs_49070)
    
    # Getting the type of 'jac' (line 18)
    jac_49072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'jac')
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', jac_49072)
    
    # ################# End of 'jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'jac' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_49073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_49073)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'jac'
    return stypy_return_type_49073

# Assigning a type to the variable 'jac' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'jac', jac)

@norecursion
def bjac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'bjac'
    module_type_store = module_type_store.open_function_context('bjac', 21, 0, False)
    
    # Passed parameters checking function
    bjac.stypy_localization = localization
    bjac.stypy_type_of_self = None
    bjac.stypy_type_store = module_type_store
    bjac.stypy_function_name = 'bjac'
    bjac.stypy_param_names_list = ['y', 't']
    bjac.stypy_varargs_param_name = None
    bjac.stypy_kwargs_param_name = None
    bjac.stypy_call_defaults = defaults
    bjac.stypy_call_varargs = varargs
    bjac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bjac', ['y', 't'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bjac', localization, ['y', 't'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bjac(...)' code ##################

    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to len(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'y' (line 22)
    y_49075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'y', False)
    # Processing the call keyword arguments (line 22)
    kwargs_49076 = {}
    # Getting the type of 'len' (line 22)
    len_49074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'len', False)
    # Calling len(args, kwargs) (line 22)
    len_call_result_49077 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), len_49074, *[y_49075], **kwargs_49076)
    
    # Assigning a type to the variable 'n' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'n', len_call_result_49077)
    
    # Assigning a Call to a Name (line 23):
    
    # Assigning a Call to a Name (line 23):
    
    # Call to zeros(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_49080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    int_49081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 21), tuple_49080, int_49081)
    # Adding element type (line 23)
    # Getting the type of 'n' (line 23)
    n_49082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 21), tuple_49080, n_49082)
    
    # Processing the call keyword arguments (line 23)
    str_49083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 34), 'str', 'F')
    keyword_49084 = str_49083
    kwargs_49085 = {'order': keyword_49084}
    # Getting the type of 'np' (line 23)
    np_49078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'np', False)
    # Obtaining the member 'zeros' of a type (line 23)
    zeros_49079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), np_49078, 'zeros')
    # Calling zeros(args, kwargs) (line 23)
    zeros_call_result_49086 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), zeros_49079, *[tuple_49080], **kwargs_49085)
    
    # Assigning a type to the variable 'bjac' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'bjac', zeros_call_result_49086)
    
    # Call to banded5x5_bjac(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 't' (line 24)
    t_49089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 't', False)
    # Getting the type of 'y' (line 24)
    y_49090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 32), 'y', False)
    int_49091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 35), 'int')
    int_49092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 38), 'int')
    # Getting the type of 'bjac' (line 24)
    bjac_49093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'bjac', False)
    # Processing the call keyword arguments (line 24)
    kwargs_49094 = {}
    # Getting the type of 'banded5x5' (line 24)
    banded5x5_49087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'banded5x5', False)
    # Obtaining the member 'banded5x5_bjac' of a type (line 24)
    banded5x5_bjac_49088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), banded5x5_49087, 'banded5x5_bjac')
    # Calling banded5x5_bjac(args, kwargs) (line 24)
    banded5x5_bjac_call_result_49095 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), banded5x5_bjac_49088, *[t_49089, y_49090, int_49091, int_49092, bjac_49093], **kwargs_49094)
    
    # Getting the type of 'bjac' (line 25)
    bjac_49096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'bjac')
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', bjac_49096)
    
    # ################# End of 'bjac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bjac' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_49097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_49097)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bjac'
    return stypy_return_type_49097

# Assigning a type to the variable 'bjac' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'bjac', bjac)

# Assigning a Num to a Name (line 28):

# Assigning a Num to a Name (line 28):
int_49098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'int')
# Assigning a type to the variable 'JACTYPE_FULL' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'JACTYPE_FULL', int_49098)

# Assigning a Num to a Name (line 29):

# Assigning a Num to a Name (line 29):
int_49099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'int')
# Assigning a type to the variable 'JACTYPE_BANDED' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'JACTYPE_BANDED', int_49099)

@norecursion
def check_odeint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_odeint'
    module_type_store = module_type_store.open_function_context('check_odeint', 32, 0, False)
    
    # Passed parameters checking function
    check_odeint.stypy_localization = localization
    check_odeint.stypy_type_of_self = None
    check_odeint.stypy_type_store = module_type_store
    check_odeint.stypy_function_name = 'check_odeint'
    check_odeint.stypy_param_names_list = ['jactype']
    check_odeint.stypy_varargs_param_name = None
    check_odeint.stypy_kwargs_param_name = None
    check_odeint.stypy_call_defaults = defaults
    check_odeint.stypy_call_varargs = varargs
    check_odeint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_odeint', ['jactype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_odeint', localization, ['jactype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_odeint(...)' code ##################

    
    
    # Getting the type of 'jactype' (line 33)
    jactype_49100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'jactype')
    # Getting the type of 'JACTYPE_FULL' (line 33)
    JACTYPE_FULL_49101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'JACTYPE_FULL')
    # Applying the binary operator '==' (line 33)
    result_eq_49102 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 7), '==', jactype_49100, JACTYPE_FULL_49101)
    
    # Testing the type of an if condition (line 33)
    if_condition_49103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), result_eq_49102)
    # Assigning a type to the variable 'if_condition_49103' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_49103', if_condition_49103)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 34):
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'None' (line 34)
    None_49104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'None')
    # Assigning a type to the variable 'ml' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'ml', None_49104)
    
    # Assigning a Name to a Name (line 35):
    
    # Assigning a Name to a Name (line 35):
    # Getting the type of 'None' (line 35)
    None_49105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'None')
    # Assigning a type to the variable 'mu' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'mu', None_49105)
    
    # Assigning a Name to a Name (line 36):
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'jac' (line 36)
    jac_49106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'jac')
    # Assigning a type to the variable 'jacobian' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'jacobian', jac_49106)
    # SSA branch for the else part of an if statement (line 33)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'jactype' (line 37)
    jactype_49107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'jactype')
    # Getting the type of 'JACTYPE_BANDED' (line 37)
    JACTYPE_BANDED_49108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'JACTYPE_BANDED')
    # Applying the binary operator '==' (line 37)
    result_eq_49109 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 9), '==', jactype_49107, JACTYPE_BANDED_49108)
    
    # Testing the type of an if condition (line 37)
    if_condition_49110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 9), result_eq_49109)
    # Assigning a type to the variable 'if_condition_49110' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'if_condition_49110', if_condition_49110)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 38):
    
    # Assigning a Num to a Name (line 38):
    int_49111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'int')
    # Assigning a type to the variable 'ml' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'ml', int_49111)
    
    # Assigning a Num to a Name (line 39):
    
    # Assigning a Num to a Name (line 39):
    int_49112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 13), 'int')
    # Assigning a type to the variable 'mu' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'mu', int_49112)
    
    # Assigning a Name to a Name (line 40):
    
    # Assigning a Name to a Name (line 40):
    # Getting the type of 'bjac' (line 40)
    bjac_49113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'bjac')
    # Assigning a type to the variable 'jacobian' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'jacobian', bjac_49113)
    # SSA branch for the else part of an if statement (line 37)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 42)
    # Processing the call arguments (line 42)
    str_49115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'str', 'invalid jactype: %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_49116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    # Getting the type of 'jactype' (line 42)
    jactype_49117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 50), 'jactype', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 50), tuple_49116, jactype_49117)
    
    # Applying the binary operator '%' (line 42)
    result_mod_49118 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 25), '%', str_49115, tuple_49116)
    
    # Processing the call keyword arguments (line 42)
    kwargs_49119 = {}
    # Getting the type of 'ValueError' (line 42)
    ValueError_49114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 42)
    ValueError_call_result_49120 = invoke(stypy.reporting.localization.Localization(__file__, 42, 14), ValueError_49114, *[result_mod_49118], **kwargs_49119)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 42, 8), ValueError_call_result_49120, 'raise parameter', BaseException)
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 44):
    
    # Assigning a Call to a Name (line 44):
    
    # Call to arange(...): (line 44)
    # Processing the call arguments (line 44)
    float_49123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'float')
    float_49124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'float')
    # Processing the call keyword arguments (line 44)
    kwargs_49125 = {}
    # Getting the type of 'np' (line 44)
    np_49121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 9), 'np', False)
    # Obtaining the member 'arange' of a type (line 44)
    arange_49122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 9), np_49121, 'arange')
    # Calling arange(args, kwargs) (line 44)
    arange_call_result_49126 = invoke(stypy.reporting.localization.Localization(__file__, 44, 9), arange_49122, *[float_49123, float_49124], **kwargs_49125)
    
    # Assigning a type to the variable 'y0' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'y0', arange_call_result_49126)
    
    # Assigning a Num to a Name (line 46):
    
    # Assigning a Num to a Name (line 46):
    float_49127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 11), 'float')
    # Assigning a type to the variable 'rtol' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'rtol', float_49127)
    
    # Assigning a Num to a Name (line 47):
    
    # Assigning a Num to a Name (line 47):
    float_49128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'float')
    # Assigning a type to the variable 'atol' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'atol', float_49128)
    
    # Assigning a Num to a Name (line 48):
    
    # Assigning a Num to a Name (line 48):
    float_49129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 9), 'float')
    # Assigning a type to the variable 'dt' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'dt', float_49129)
    
    # Assigning a Num to a Name (line 49):
    
    # Assigning a Num to a Name (line 49):
    int_49130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 13), 'int')
    # Assigning a type to the variable 'nsteps' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'nsteps', int_49130)
    
    # Assigning a BinOp to a Name (line 50):
    
    # Assigning a BinOp to a Name (line 50):
    # Getting the type of 'dt' (line 50)
    dt_49131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'dt')
    
    # Call to arange(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'nsteps' (line 50)
    nsteps_49134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 23), 'nsteps', False)
    int_49135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'int')
    # Applying the binary operator '+' (line 50)
    result_add_49136 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 23), '+', nsteps_49134, int_49135)
    
    # Processing the call keyword arguments (line 50)
    kwargs_49137 = {}
    # Getting the type of 'np' (line 50)
    np_49132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'np', False)
    # Obtaining the member 'arange' of a type (line 50)
    arange_49133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 13), np_49132, 'arange')
    # Calling arange(args, kwargs) (line 50)
    arange_call_result_49138 = invoke(stypy.reporting.localization.Localization(__file__, 50, 13), arange_49133, *[result_add_49136], **kwargs_49137)
    
    # Applying the binary operator '*' (line 50)
    result_mul_49139 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 8), '*', dt_49131, arange_call_result_49138)
    
    # Assigning a type to the variable 't' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 't', result_mul_49139)
    
    # Assigning a Call to a Tuple (line 52):
    
    # Assigning a Subscript to a Name (line 52):
    
    # Obtaining the type of the subscript
    int_49140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'int')
    
    # Call to odeint(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'rhs' (line 52)
    rhs_49142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'rhs', False)
    # Getting the type of 'y0' (line 52)
    y0_49143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'y0', False)
    # Getting the type of 't' (line 52)
    t_49144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 32), 't', False)
    # Processing the call keyword arguments (line 52)
    # Getting the type of 'jacobian' (line 53)
    jacobian_49145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'jacobian', False)
    keyword_49146 = jacobian_49145
    # Getting the type of 'ml' (line 53)
    ml_49147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 41), 'ml', False)
    keyword_49148 = ml_49147
    # Getting the type of 'mu' (line 53)
    mu_49149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 48), 'mu', False)
    keyword_49150 = mu_49149
    # Getting the type of 'atol' (line 54)
    atol_49151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'atol', False)
    keyword_49152 = atol_49151
    # Getting the type of 'rtol' (line 54)
    rtol_49153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 'rtol', False)
    keyword_49154 = rtol_49153
    # Getting the type of 'True' (line 54)
    True_49155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 57), 'True', False)
    keyword_49156 = True_49155
    kwargs_49157 = {'full_output': keyword_49156, 'ml': keyword_49148, 'Dfun': keyword_49146, 'mu': keyword_49150, 'rtol': keyword_49154, 'atol': keyword_49152}
    # Getting the type of 'odeint' (line 52)
    odeint_49141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'odeint', False)
    # Calling odeint(args, kwargs) (line 52)
    odeint_call_result_49158 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), odeint_49141, *[rhs_49142, y0_49143, t_49144], **kwargs_49157)
    
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___49159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), odeint_call_result_49158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 52)
    subscript_call_result_49160 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), getitem___49159, int_49140)
    
    # Assigning a type to the variable 'tuple_var_assignment_49023' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'tuple_var_assignment_49023', subscript_call_result_49160)
    
    # Assigning a Subscript to a Name (line 52):
    
    # Obtaining the type of the subscript
    int_49161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'int')
    
    # Call to odeint(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'rhs' (line 52)
    rhs_49163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'rhs', False)
    # Getting the type of 'y0' (line 52)
    y0_49164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'y0', False)
    # Getting the type of 't' (line 52)
    t_49165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 32), 't', False)
    # Processing the call keyword arguments (line 52)
    # Getting the type of 'jacobian' (line 53)
    jacobian_49166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'jacobian', False)
    keyword_49167 = jacobian_49166
    # Getting the type of 'ml' (line 53)
    ml_49168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 41), 'ml', False)
    keyword_49169 = ml_49168
    # Getting the type of 'mu' (line 53)
    mu_49170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 48), 'mu', False)
    keyword_49171 = mu_49170
    # Getting the type of 'atol' (line 54)
    atol_49172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'atol', False)
    keyword_49173 = atol_49172
    # Getting the type of 'rtol' (line 54)
    rtol_49174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 'rtol', False)
    keyword_49175 = rtol_49174
    # Getting the type of 'True' (line 54)
    True_49176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 57), 'True', False)
    keyword_49177 = True_49176
    kwargs_49178 = {'full_output': keyword_49177, 'ml': keyword_49169, 'Dfun': keyword_49167, 'mu': keyword_49171, 'rtol': keyword_49175, 'atol': keyword_49173}
    # Getting the type of 'odeint' (line 52)
    odeint_49162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'odeint', False)
    # Calling odeint(args, kwargs) (line 52)
    odeint_call_result_49179 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), odeint_49162, *[rhs_49163, y0_49164, t_49165], **kwargs_49178)
    
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___49180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), odeint_call_result_49179, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 52)
    subscript_call_result_49181 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), getitem___49180, int_49161)
    
    # Assigning a type to the variable 'tuple_var_assignment_49024' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'tuple_var_assignment_49024', subscript_call_result_49181)
    
    # Assigning a Name to a Name (line 52):
    # Getting the type of 'tuple_var_assignment_49023' (line 52)
    tuple_var_assignment_49023_49182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'tuple_var_assignment_49023')
    # Assigning a type to the variable 'sol' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'sol', tuple_var_assignment_49023_49182)
    
    # Assigning a Name to a Name (line 52):
    # Getting the type of 'tuple_var_assignment_49024' (line 52)
    tuple_var_assignment_49024_49183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'tuple_var_assignment_49024')
    # Assigning a type to the variable 'info' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 9), 'info', tuple_var_assignment_49024_49183)
    
    # Assigning a Subscript to a Name (line 55):
    
    # Assigning a Subscript to a Name (line 55):
    
    # Obtaining the type of the subscript
    int_49184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 17), 'int')
    # Getting the type of 'sol' (line 55)
    sol_49185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'sol')
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___49186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 13), sol_49185, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_49187 = invoke(stypy.reporting.localization.Localization(__file__, 55, 13), getitem___49186, int_49184)
    
    # Assigning a type to the variable 'yfinal' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'yfinal', subscript_call_result_49187)
    
    # Assigning a Subscript to a Name (line 56):
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_49188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 29), 'int')
    
    # Obtaining the type of the subscript
    str_49189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'str', 'nst')
    # Getting the type of 'info' (line 56)
    info_49190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'info')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___49191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), info_49190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_49192 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), getitem___49191, str_49189)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___49193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), subscript_call_result_49192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_49194 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), getitem___49193, int_49188)
    
    # Assigning a type to the variable 'odeint_nst' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'odeint_nst', subscript_call_result_49194)
    
    # Assigning a Subscript to a Name (line 57):
    
    # Assigning a Subscript to a Name (line 57):
    
    # Obtaining the type of the subscript
    int_49195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 29), 'int')
    
    # Obtaining the type of the subscript
    str_49196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'str', 'nfe')
    # Getting the type of 'info' (line 57)
    info_49197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'info')
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___49198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 17), info_49197, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_49199 = invoke(stypy.reporting.localization.Localization(__file__, 57, 17), getitem___49198, str_49196)
    
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___49200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 17), subscript_call_result_49199, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_49201 = invoke(stypy.reporting.localization.Localization(__file__, 57, 17), getitem___49200, int_49195)
    
    # Assigning a type to the variable 'odeint_nfe' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'odeint_nfe', subscript_call_result_49201)
    
    # Assigning a Subscript to a Name (line 58):
    
    # Assigning a Subscript to a Name (line 58):
    
    # Obtaining the type of the subscript
    int_49202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'int')
    
    # Obtaining the type of the subscript
    str_49203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 22), 'str', 'nje')
    # Getting the type of 'info' (line 58)
    info_49204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'info')
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___49205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 17), info_49204, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 58)
    subscript_call_result_49206 = invoke(stypy.reporting.localization.Localization(__file__, 58, 17), getitem___49205, str_49203)
    
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___49207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 17), subscript_call_result_49206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 58)
    subscript_call_result_49208 = invoke(stypy.reporting.localization.Localization(__file__, 58, 17), getitem___49207, int_49202)
    
    # Assigning a type to the variable 'odeint_nje' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'odeint_nje', subscript_call_result_49208)
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to copy(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_49211 = {}
    # Getting the type of 'y0' (line 60)
    y0_49209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 9), 'y0', False)
    # Obtaining the member 'copy' of a type (line 60)
    copy_49210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 9), y0_49209, 'copy')
    # Calling copy(args, kwargs) (line 60)
    copy_call_result_49212 = invoke(stypy.reporting.localization.Localization(__file__, 60, 9), copy_49210, *[], **kwargs_49211)
    
    # Assigning a type to the variable 'y1' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'y1', copy_call_result_49212)
    
    # Assigning a Call to a Tuple (line 62):
    
    # Assigning a Subscript to a Name (line 62):
    
    # Obtaining the type of the subscript
    int_49213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'int')
    
    # Call to banded5x5_solve(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'y1' (line 62)
    y1_49216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 46), 'y1', False)
    # Getting the type of 'nsteps' (line 62)
    nsteps_49217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 50), 'nsteps', False)
    # Getting the type of 'dt' (line 62)
    dt_49218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 58), 'dt', False)
    # Getting the type of 'jactype' (line 62)
    jactype_49219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 62), 'jactype', False)
    # Processing the call keyword arguments (line 62)
    kwargs_49220 = {}
    # Getting the type of 'banded5x5' (line 62)
    banded5x5_49214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'banded5x5', False)
    # Obtaining the member 'banded5x5_solve' of a type (line 62)
    banded5x5_solve_49215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 20), banded5x5_49214, 'banded5x5_solve')
    # Calling banded5x5_solve(args, kwargs) (line 62)
    banded5x5_solve_call_result_49221 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), banded5x5_solve_49215, *[y1_49216, nsteps_49217, dt_49218, jactype_49219], **kwargs_49220)
    
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___49222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), banded5x5_solve_call_result_49221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_49223 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), getitem___49222, int_49213)
    
    # Assigning a type to the variable 'tuple_var_assignment_49025' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'tuple_var_assignment_49025', subscript_call_result_49223)
    
    # Assigning a Subscript to a Name (line 62):
    
    # Obtaining the type of the subscript
    int_49224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'int')
    
    # Call to banded5x5_solve(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'y1' (line 62)
    y1_49227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 46), 'y1', False)
    # Getting the type of 'nsteps' (line 62)
    nsteps_49228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 50), 'nsteps', False)
    # Getting the type of 'dt' (line 62)
    dt_49229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 58), 'dt', False)
    # Getting the type of 'jactype' (line 62)
    jactype_49230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 62), 'jactype', False)
    # Processing the call keyword arguments (line 62)
    kwargs_49231 = {}
    # Getting the type of 'banded5x5' (line 62)
    banded5x5_49225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'banded5x5', False)
    # Obtaining the member 'banded5x5_solve' of a type (line 62)
    banded5x5_solve_49226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 20), banded5x5_49225, 'banded5x5_solve')
    # Calling banded5x5_solve(args, kwargs) (line 62)
    banded5x5_solve_call_result_49232 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), banded5x5_solve_49226, *[y1_49227, nsteps_49228, dt_49229, jactype_49230], **kwargs_49231)
    
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___49233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), banded5x5_solve_call_result_49232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_49234 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), getitem___49233, int_49224)
    
    # Assigning a type to the variable 'tuple_var_assignment_49026' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'tuple_var_assignment_49026', subscript_call_result_49234)
    
    # Assigning a Subscript to a Name (line 62):
    
    # Obtaining the type of the subscript
    int_49235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'int')
    
    # Call to banded5x5_solve(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'y1' (line 62)
    y1_49238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 46), 'y1', False)
    # Getting the type of 'nsteps' (line 62)
    nsteps_49239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 50), 'nsteps', False)
    # Getting the type of 'dt' (line 62)
    dt_49240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 58), 'dt', False)
    # Getting the type of 'jactype' (line 62)
    jactype_49241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 62), 'jactype', False)
    # Processing the call keyword arguments (line 62)
    kwargs_49242 = {}
    # Getting the type of 'banded5x5' (line 62)
    banded5x5_49236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'banded5x5', False)
    # Obtaining the member 'banded5x5_solve' of a type (line 62)
    banded5x5_solve_49237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 20), banded5x5_49236, 'banded5x5_solve')
    # Calling banded5x5_solve(args, kwargs) (line 62)
    banded5x5_solve_call_result_49243 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), banded5x5_solve_49237, *[y1_49238, nsteps_49239, dt_49240, jactype_49241], **kwargs_49242)
    
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___49244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), banded5x5_solve_call_result_49243, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_49245 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), getitem___49244, int_49235)
    
    # Assigning a type to the variable 'tuple_var_assignment_49027' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'tuple_var_assignment_49027', subscript_call_result_49245)
    
    # Assigning a Name to a Name (line 62):
    # Getting the type of 'tuple_var_assignment_49025' (line 62)
    tuple_var_assignment_49025_49246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'tuple_var_assignment_49025')
    # Assigning a type to the variable 'nst' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'nst', tuple_var_assignment_49025_49246)
    
    # Assigning a Name to a Name (line 62):
    # Getting the type of 'tuple_var_assignment_49026' (line 62)
    tuple_var_assignment_49026_49247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'tuple_var_assignment_49026')
    # Assigning a type to the variable 'nfe' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 9), 'nfe', tuple_var_assignment_49026_49247)
    
    # Assigning a Name to a Name (line 62):
    # Getting the type of 'tuple_var_assignment_49027' (line 62)
    tuple_var_assignment_49027_49248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'tuple_var_assignment_49027')
    # Assigning a type to the variable 'nje' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 14), 'nje', tuple_var_assignment_49027_49248)
    
    # Call to assert_allclose(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'yfinal' (line 66)
    yfinal_49250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'yfinal', False)
    # Getting the type of 'y1' (line 66)
    y1_49251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'y1', False)
    # Processing the call keyword arguments (line 66)
    float_49252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 37), 'float')
    keyword_49253 = float_49252
    kwargs_49254 = {'rtol': keyword_49253}
    # Getting the type of 'assert_allclose' (line 66)
    assert_allclose_49249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 66)
    assert_allclose_call_result_49255 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), assert_allclose_49249, *[yfinal_49250, y1_49251], **kwargs_49254)
    
    
    # Call to assert_equal(...): (line 67)
    # Processing the call arguments (line 67)
    
    # Obtaining an instance of the builtin type 'tuple' (line 67)
    tuple_49257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 67)
    # Adding element type (line 67)
    # Getting the type of 'odeint_nst' (line 67)
    odeint_nst_49258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'odeint_nst', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 18), tuple_49257, odeint_nst_49258)
    # Adding element type (line 67)
    # Getting the type of 'odeint_nfe' (line 67)
    odeint_nfe_49259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'odeint_nfe', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 18), tuple_49257, odeint_nfe_49259)
    # Adding element type (line 67)
    # Getting the type of 'odeint_nje' (line 67)
    odeint_nje_49260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 42), 'odeint_nje', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 18), tuple_49257, odeint_nje_49260)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 67)
    tuple_49261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 67)
    # Adding element type (line 67)
    # Getting the type of 'nst' (line 67)
    nst_49262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 56), 'nst', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 56), tuple_49261, nst_49262)
    # Adding element type (line 67)
    # Getting the type of 'nfe' (line 67)
    nfe_49263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 61), 'nfe', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 56), tuple_49261, nfe_49263)
    # Adding element type (line 67)
    # Getting the type of 'nje' (line 67)
    nje_49264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 66), 'nje', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 56), tuple_49261, nje_49264)
    
    # Processing the call keyword arguments (line 67)
    kwargs_49265 = {}
    # Getting the type of 'assert_equal' (line 67)
    assert_equal_49256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 67)
    assert_equal_call_result_49266 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), assert_equal_49256, *[tuple_49257, tuple_49261], **kwargs_49265)
    
    
    # ################# End of 'check_odeint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_odeint' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_49267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_49267)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_odeint'
    return stypy_return_type_49267

# Assigning a type to the variable 'check_odeint' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'check_odeint', check_odeint)

@norecursion
def test_odeint_full_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_odeint_full_jac'
    module_type_store = module_type_store.open_function_context('test_odeint_full_jac', 70, 0, False)
    
    # Passed parameters checking function
    test_odeint_full_jac.stypy_localization = localization
    test_odeint_full_jac.stypy_type_of_self = None
    test_odeint_full_jac.stypy_type_store = module_type_store
    test_odeint_full_jac.stypy_function_name = 'test_odeint_full_jac'
    test_odeint_full_jac.stypy_param_names_list = []
    test_odeint_full_jac.stypy_varargs_param_name = None
    test_odeint_full_jac.stypy_kwargs_param_name = None
    test_odeint_full_jac.stypy_call_defaults = defaults
    test_odeint_full_jac.stypy_call_varargs = varargs
    test_odeint_full_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_odeint_full_jac', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_odeint_full_jac', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_odeint_full_jac(...)' code ##################

    
    # Call to check_odeint(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'JACTYPE_FULL' (line 71)
    JACTYPE_FULL_49269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'JACTYPE_FULL', False)
    # Processing the call keyword arguments (line 71)
    kwargs_49270 = {}
    # Getting the type of 'check_odeint' (line 71)
    check_odeint_49268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'check_odeint', False)
    # Calling check_odeint(args, kwargs) (line 71)
    check_odeint_call_result_49271 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), check_odeint_49268, *[JACTYPE_FULL_49269], **kwargs_49270)
    
    
    # ################# End of 'test_odeint_full_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_odeint_full_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_49272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_49272)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_odeint_full_jac'
    return stypy_return_type_49272

# Assigning a type to the variable 'test_odeint_full_jac' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'test_odeint_full_jac', test_odeint_full_jac)

@norecursion
def test_odeint_banded_jac(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_odeint_banded_jac'
    module_type_store = module_type_store.open_function_context('test_odeint_banded_jac', 74, 0, False)
    
    # Passed parameters checking function
    test_odeint_banded_jac.stypy_localization = localization
    test_odeint_banded_jac.stypy_type_of_self = None
    test_odeint_banded_jac.stypy_type_store = module_type_store
    test_odeint_banded_jac.stypy_function_name = 'test_odeint_banded_jac'
    test_odeint_banded_jac.stypy_param_names_list = []
    test_odeint_banded_jac.stypy_varargs_param_name = None
    test_odeint_banded_jac.stypy_kwargs_param_name = None
    test_odeint_banded_jac.stypy_call_defaults = defaults
    test_odeint_banded_jac.stypy_call_varargs = varargs
    test_odeint_banded_jac.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_odeint_banded_jac', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_odeint_banded_jac', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_odeint_banded_jac(...)' code ##################

    
    # Call to check_odeint(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'JACTYPE_BANDED' (line 75)
    JACTYPE_BANDED_49274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'JACTYPE_BANDED', False)
    # Processing the call keyword arguments (line 75)
    kwargs_49275 = {}
    # Getting the type of 'check_odeint' (line 75)
    check_odeint_49273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'check_odeint', False)
    # Calling check_odeint(args, kwargs) (line 75)
    check_odeint_call_result_49276 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), check_odeint_49273, *[JACTYPE_BANDED_49274], **kwargs_49275)
    
    
    # ################# End of 'test_odeint_banded_jac(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_odeint_banded_jac' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_49277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_49277)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_odeint_banded_jac'
    return stypy_return_type_49277

# Assigning a type to the variable 'test_odeint_banded_jac' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'test_odeint_banded_jac', test_odeint_banded_jac)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
