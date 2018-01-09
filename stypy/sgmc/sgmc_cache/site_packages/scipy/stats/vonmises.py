
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: import scipy.stats
5: from scipy.special import i0
6: 
7: 
8: def von_mises_cdf_series(k,x,p):
9:     x = float(x)
10:     s = np.sin(x)
11:     c = np.cos(x)
12:     sn = np.sin(p*x)
13:     cn = np.cos(p*x)
14:     R = 0
15:     V = 0
16:     for n in range(p-1,0,-1):
17:         sn, cn = sn*c - cn*s, cn*c + sn*s
18:         R = 1./(2*n/k + R)
19:         V = R*(sn/n+V)
20: 
21:     return 0.5+x/(2*np.pi) + V/np.pi
22: 
23: 
24: def von_mises_cdf_normalapprox(k, x):
25:     b = np.sqrt(2/np.pi)*np.exp(k)/i0(k)
26:     z = b*np.sin(x/2.)
27:     return scipy.stats.norm.cdf(z)
28: 
29: 
30: def von_mises_cdf(k,x):
31:     ix = 2*np.pi*np.round(x/(2*np.pi))
32:     x = x-ix
33:     k = float(k)
34: 
35:     # These values should give 12 decimal digits
36:     CK = 50
37:     a = [28., 0.5, 100., 5.0]
38: 
39:     if k < CK:
40:         p = int(np.ceil(a[0]+a[1]*k-a[2]/(k+a[3])))
41: 
42:         F = np.clip(von_mises_cdf_series(k,x,p),0,1)
43:     else:
44:         F = von_mises_cdf_normalapprox(k, x)
45: 
46:     return F+ix
47: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_589028 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_589028) is not StypyTypeError):

    if (import_589028 != 'pyd_module'):
        __import__(import_589028)
        sys_modules_589029 = sys.modules[import_589028]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_589029.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_589028)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import scipy.stats' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_589030 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.stats')

if (type(import_589030) is not StypyTypeError):

    if (import_589030 != 'pyd_module'):
        __import__(import_589030)
        sys_modules_589031 = sys.modules[import_589030]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.stats', sys_modules_589031.module_type_store, module_type_store)
    else:
        import scipy.stats

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.stats', scipy.stats, module_type_store)

else:
    # Assigning a type to the variable 'scipy.stats' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.stats', import_589030)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.special import i0' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_589032 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special')

if (type(import_589032) is not StypyTypeError):

    if (import_589032 != 'pyd_module'):
        __import__(import_589032)
        sys_modules_589033 = sys.modules[import_589032]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', sys_modules_589033.module_type_store, module_type_store, ['i0'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_589033, sys_modules_589033.module_type_store, module_type_store)
    else:
        from scipy.special import i0

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', None, module_type_store, ['i0'], [i0])

else:
    # Assigning a type to the variable 'scipy.special' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', import_589032)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


@norecursion
def von_mises_cdf_series(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'von_mises_cdf_series'
    module_type_store = module_type_store.open_function_context('von_mises_cdf_series', 8, 0, False)
    
    # Passed parameters checking function
    von_mises_cdf_series.stypy_localization = localization
    von_mises_cdf_series.stypy_type_of_self = None
    von_mises_cdf_series.stypy_type_store = module_type_store
    von_mises_cdf_series.stypy_function_name = 'von_mises_cdf_series'
    von_mises_cdf_series.stypy_param_names_list = ['k', 'x', 'p']
    von_mises_cdf_series.stypy_varargs_param_name = None
    von_mises_cdf_series.stypy_kwargs_param_name = None
    von_mises_cdf_series.stypy_call_defaults = defaults
    von_mises_cdf_series.stypy_call_varargs = varargs
    von_mises_cdf_series.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'von_mises_cdf_series', ['k', 'x', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'von_mises_cdf_series', localization, ['k', 'x', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'von_mises_cdf_series(...)' code ##################

    
    # Assigning a Call to a Name (line 9):
    
    # Assigning a Call to a Name (line 9):
    
    # Call to float(...): (line 9)
    # Processing the call arguments (line 9)
    # Getting the type of 'x' (line 9)
    x_589035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 14), 'x', False)
    # Processing the call keyword arguments (line 9)
    kwargs_589036 = {}
    # Getting the type of 'float' (line 9)
    float_589034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'float', False)
    # Calling float(args, kwargs) (line 9)
    float_call_result_589037 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), float_589034, *[x_589035], **kwargs_589036)
    
    # Assigning a type to the variable 'x' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'x', float_call_result_589037)
    
    # Assigning a Call to a Name (line 10):
    
    # Assigning a Call to a Name (line 10):
    
    # Call to sin(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'x' (line 10)
    x_589040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'x', False)
    # Processing the call keyword arguments (line 10)
    kwargs_589041 = {}
    # Getting the type of 'np' (line 10)
    np_589038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'np', False)
    # Obtaining the member 'sin' of a type (line 10)
    sin_589039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), np_589038, 'sin')
    # Calling sin(args, kwargs) (line 10)
    sin_call_result_589042 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), sin_589039, *[x_589040], **kwargs_589041)
    
    # Assigning a type to the variable 's' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 's', sin_call_result_589042)
    
    # Assigning a Call to a Name (line 11):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to cos(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'x' (line 11)
    x_589045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'x', False)
    # Processing the call keyword arguments (line 11)
    kwargs_589046 = {}
    # Getting the type of 'np' (line 11)
    np_589043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'np', False)
    # Obtaining the member 'cos' of a type (line 11)
    cos_589044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), np_589043, 'cos')
    # Calling cos(args, kwargs) (line 11)
    cos_call_result_589047 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), cos_589044, *[x_589045], **kwargs_589046)
    
    # Assigning a type to the variable 'c' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'c', cos_call_result_589047)
    
    # Assigning a Call to a Name (line 12):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to sin(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'p' (line 12)
    p_589050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), 'p', False)
    # Getting the type of 'x' (line 12)
    x_589051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'x', False)
    # Applying the binary operator '*' (line 12)
    result_mul_589052 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 16), '*', p_589050, x_589051)
    
    # Processing the call keyword arguments (line 12)
    kwargs_589053 = {}
    # Getting the type of 'np' (line 12)
    np_589048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'np', False)
    # Obtaining the member 'sin' of a type (line 12)
    sin_589049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 9), np_589048, 'sin')
    # Calling sin(args, kwargs) (line 12)
    sin_call_result_589054 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), sin_589049, *[result_mul_589052], **kwargs_589053)
    
    # Assigning a type to the variable 'sn' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'sn', sin_call_result_589054)
    
    # Assigning a Call to a Name (line 13):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to cos(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'p' (line 13)
    p_589057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'p', False)
    # Getting the type of 'x' (line 13)
    x_589058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 18), 'x', False)
    # Applying the binary operator '*' (line 13)
    result_mul_589059 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 16), '*', p_589057, x_589058)
    
    # Processing the call keyword arguments (line 13)
    kwargs_589060 = {}
    # Getting the type of 'np' (line 13)
    np_589055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 'np', False)
    # Obtaining the member 'cos' of a type (line 13)
    cos_589056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), np_589055, 'cos')
    # Calling cos(args, kwargs) (line 13)
    cos_call_result_589061 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), cos_589056, *[result_mul_589059], **kwargs_589060)
    
    # Assigning a type to the variable 'cn' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'cn', cos_call_result_589061)
    
    # Assigning a Num to a Name (line 14):
    
    # Assigning a Num to a Name (line 14):
    int_589062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 8), 'int')
    # Assigning a type to the variable 'R' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'R', int_589062)
    
    # Assigning a Num to a Name (line 15):
    
    # Assigning a Num to a Name (line 15):
    int_589063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'int')
    # Assigning a type to the variable 'V' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'V', int_589063)
    
    
    # Call to range(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'p' (line 16)
    p_589065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'p', False)
    int_589066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'int')
    # Applying the binary operator '-' (line 16)
    result_sub_589067 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 19), '-', p_589065, int_589066)
    
    int_589068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'int')
    int_589069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_589070 = {}
    # Getting the type of 'range' (line 16)
    range_589064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 13), 'range', False)
    # Calling range(args, kwargs) (line 16)
    range_call_result_589071 = invoke(stypy.reporting.localization.Localization(__file__, 16, 13), range_589064, *[result_sub_589067, int_589068, int_589069], **kwargs_589070)
    
    # Testing the type of a for loop iterable (line 16)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 4), range_call_result_589071)
    # Getting the type of the for loop variable (line 16)
    for_loop_var_589072 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 4), range_call_result_589071)
    # Assigning a type to the variable 'n' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'n', for_loop_var_589072)
    # SSA begins for a for statement (line 16)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Tuple to a Tuple (line 17):
    
    # Assigning a BinOp to a Name (line 17):
    # Getting the type of 'sn' (line 17)
    sn_589073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'sn')
    # Getting the type of 'c' (line 17)
    c_589074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'c')
    # Applying the binary operator '*' (line 17)
    result_mul_589075 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 17), '*', sn_589073, c_589074)
    
    # Getting the type of 'cn' (line 17)
    cn_589076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'cn')
    # Getting the type of 's' (line 17)
    s_589077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 27), 's')
    # Applying the binary operator '*' (line 17)
    result_mul_589078 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 24), '*', cn_589076, s_589077)
    
    # Applying the binary operator '-' (line 17)
    result_sub_589079 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 17), '-', result_mul_589075, result_mul_589078)
    
    # Assigning a type to the variable 'tuple_assignment_589026' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_assignment_589026', result_sub_589079)
    
    # Assigning a BinOp to a Name (line 17):
    # Getting the type of 'cn' (line 17)
    cn_589080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 30), 'cn')
    # Getting the type of 'c' (line 17)
    c_589081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 33), 'c')
    # Applying the binary operator '*' (line 17)
    result_mul_589082 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 30), '*', cn_589080, c_589081)
    
    # Getting the type of 'sn' (line 17)
    sn_589083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 37), 'sn')
    # Getting the type of 's' (line 17)
    s_589084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 40), 's')
    # Applying the binary operator '*' (line 17)
    result_mul_589085 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 37), '*', sn_589083, s_589084)
    
    # Applying the binary operator '+' (line 17)
    result_add_589086 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 30), '+', result_mul_589082, result_mul_589085)
    
    # Assigning a type to the variable 'tuple_assignment_589027' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_assignment_589027', result_add_589086)
    
    # Assigning a Name to a Name (line 17):
    # Getting the type of 'tuple_assignment_589026' (line 17)
    tuple_assignment_589026_589087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_assignment_589026')
    # Assigning a type to the variable 'sn' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'sn', tuple_assignment_589026_589087)
    
    # Assigning a Name to a Name (line 17):
    # Getting the type of 'tuple_assignment_589027' (line 17)
    tuple_assignment_589027_589088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_assignment_589027')
    # Assigning a type to the variable 'cn' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'cn', tuple_assignment_589027_589088)
    
    # Assigning a BinOp to a Name (line 18):
    
    # Assigning a BinOp to a Name (line 18):
    float_589089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 12), 'float')
    int_589090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'int')
    # Getting the type of 'n' (line 18)
    n_589091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'n')
    # Applying the binary operator '*' (line 18)
    result_mul_589092 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 16), '*', int_589090, n_589091)
    
    # Getting the type of 'k' (line 18)
    k_589093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'k')
    # Applying the binary operator 'div' (line 18)
    result_div_589094 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 19), 'div', result_mul_589092, k_589093)
    
    # Getting the type of 'R' (line 18)
    R_589095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'R')
    # Applying the binary operator '+' (line 18)
    result_add_589096 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 16), '+', result_div_589094, R_589095)
    
    # Applying the binary operator 'div' (line 18)
    result_div_589097 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 12), 'div', float_589089, result_add_589096)
    
    # Assigning a type to the variable 'R' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'R', result_div_589097)
    
    # Assigning a BinOp to a Name (line 19):
    
    # Assigning a BinOp to a Name (line 19):
    # Getting the type of 'R' (line 19)
    R_589098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'R')
    # Getting the type of 'sn' (line 19)
    sn_589099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'sn')
    # Getting the type of 'n' (line 19)
    n_589100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 18), 'n')
    # Applying the binary operator 'div' (line 19)
    result_div_589101 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 15), 'div', sn_589099, n_589100)
    
    # Getting the type of 'V' (line 19)
    V_589102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'V')
    # Applying the binary operator '+' (line 19)
    result_add_589103 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 15), '+', result_div_589101, V_589102)
    
    # Applying the binary operator '*' (line 19)
    result_mul_589104 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 12), '*', R_589098, result_add_589103)
    
    # Assigning a type to the variable 'V' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'V', result_mul_589104)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    float_589105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'float')
    # Getting the type of 'x' (line 21)
    x_589106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'x')
    int_589107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'int')
    # Getting the type of 'np' (line 21)
    np_589108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'np')
    # Obtaining the member 'pi' of a type (line 21)
    pi_589109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 20), np_589108, 'pi')
    # Applying the binary operator '*' (line 21)
    result_mul_589110 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 18), '*', int_589107, pi_589109)
    
    # Applying the binary operator 'div' (line 21)
    result_div_589111 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 15), 'div', x_589106, result_mul_589110)
    
    # Applying the binary operator '+' (line 21)
    result_add_589112 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 11), '+', float_589105, result_div_589111)
    
    # Getting the type of 'V' (line 21)
    V_589113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 29), 'V')
    # Getting the type of 'np' (line 21)
    np_589114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), 'np')
    # Obtaining the member 'pi' of a type (line 21)
    pi_589115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 31), np_589114, 'pi')
    # Applying the binary operator 'div' (line 21)
    result_div_589116 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 29), 'div', V_589113, pi_589115)
    
    # Applying the binary operator '+' (line 21)
    result_add_589117 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 27), '+', result_add_589112, result_div_589116)
    
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type', result_add_589117)
    
    # ################# End of 'von_mises_cdf_series(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'von_mises_cdf_series' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_589118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_589118)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'von_mises_cdf_series'
    return stypy_return_type_589118

# Assigning a type to the variable 'von_mises_cdf_series' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'von_mises_cdf_series', von_mises_cdf_series)

@norecursion
def von_mises_cdf_normalapprox(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'von_mises_cdf_normalapprox'
    module_type_store = module_type_store.open_function_context('von_mises_cdf_normalapprox', 24, 0, False)
    
    # Passed parameters checking function
    von_mises_cdf_normalapprox.stypy_localization = localization
    von_mises_cdf_normalapprox.stypy_type_of_self = None
    von_mises_cdf_normalapprox.stypy_type_store = module_type_store
    von_mises_cdf_normalapprox.stypy_function_name = 'von_mises_cdf_normalapprox'
    von_mises_cdf_normalapprox.stypy_param_names_list = ['k', 'x']
    von_mises_cdf_normalapprox.stypy_varargs_param_name = None
    von_mises_cdf_normalapprox.stypy_kwargs_param_name = None
    von_mises_cdf_normalapprox.stypy_call_defaults = defaults
    von_mises_cdf_normalapprox.stypy_call_varargs = varargs
    von_mises_cdf_normalapprox.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'von_mises_cdf_normalapprox', ['k', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'von_mises_cdf_normalapprox', localization, ['k', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'von_mises_cdf_normalapprox(...)' code ##################

    
    # Assigning a BinOp to a Name (line 25):
    
    # Assigning a BinOp to a Name (line 25):
    
    # Call to sqrt(...): (line 25)
    # Processing the call arguments (line 25)
    int_589121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'int')
    # Getting the type of 'np' (line 25)
    np_589122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'np', False)
    # Obtaining the member 'pi' of a type (line 25)
    pi_589123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 18), np_589122, 'pi')
    # Applying the binary operator 'div' (line 25)
    result_div_589124 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 16), 'div', int_589121, pi_589123)
    
    # Processing the call keyword arguments (line 25)
    kwargs_589125 = {}
    # Getting the type of 'np' (line 25)
    np_589119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 25)
    sqrt_589120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), np_589119, 'sqrt')
    # Calling sqrt(args, kwargs) (line 25)
    sqrt_call_result_589126 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), sqrt_589120, *[result_div_589124], **kwargs_589125)
    
    
    # Call to exp(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'k' (line 25)
    k_589129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'k', False)
    # Processing the call keyword arguments (line 25)
    kwargs_589130 = {}
    # Getting the type of 'np' (line 25)
    np_589127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'np', False)
    # Obtaining the member 'exp' of a type (line 25)
    exp_589128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 25), np_589127, 'exp')
    # Calling exp(args, kwargs) (line 25)
    exp_call_result_589131 = invoke(stypy.reporting.localization.Localization(__file__, 25, 25), exp_589128, *[k_589129], **kwargs_589130)
    
    # Applying the binary operator '*' (line 25)
    result_mul_589132 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 8), '*', sqrt_call_result_589126, exp_call_result_589131)
    
    
    # Call to i0(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'k' (line 25)
    k_589134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 38), 'k', False)
    # Processing the call keyword arguments (line 25)
    kwargs_589135 = {}
    # Getting the type of 'i0' (line 25)
    i0_589133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 35), 'i0', False)
    # Calling i0(args, kwargs) (line 25)
    i0_call_result_589136 = invoke(stypy.reporting.localization.Localization(__file__, 25, 35), i0_589133, *[k_589134], **kwargs_589135)
    
    # Applying the binary operator 'div' (line 25)
    result_div_589137 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 34), 'div', result_mul_589132, i0_call_result_589136)
    
    # Assigning a type to the variable 'b' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'b', result_div_589137)
    
    # Assigning a BinOp to a Name (line 26):
    
    # Assigning a BinOp to a Name (line 26):
    # Getting the type of 'b' (line 26)
    b_589138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'b')
    
    # Call to sin(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'x' (line 26)
    x_589141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'x', False)
    float_589142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'float')
    # Applying the binary operator 'div' (line 26)
    result_div_589143 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 17), 'div', x_589141, float_589142)
    
    # Processing the call keyword arguments (line 26)
    kwargs_589144 = {}
    # Getting the type of 'np' (line 26)
    np_589139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'np', False)
    # Obtaining the member 'sin' of a type (line 26)
    sin_589140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 10), np_589139, 'sin')
    # Calling sin(args, kwargs) (line 26)
    sin_call_result_589145 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), sin_589140, *[result_div_589143], **kwargs_589144)
    
    # Applying the binary operator '*' (line 26)
    result_mul_589146 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 8), '*', b_589138, sin_call_result_589145)
    
    # Assigning a type to the variable 'z' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'z', result_mul_589146)
    
    # Call to cdf(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'z' (line 27)
    z_589151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 32), 'z', False)
    # Processing the call keyword arguments (line 27)
    kwargs_589152 = {}
    # Getting the type of 'scipy' (line 27)
    scipy_589147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'scipy', False)
    # Obtaining the member 'stats' of a type (line 27)
    stats_589148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 11), scipy_589147, 'stats')
    # Obtaining the member 'norm' of a type (line 27)
    norm_589149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 11), stats_589148, 'norm')
    # Obtaining the member 'cdf' of a type (line 27)
    cdf_589150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 11), norm_589149, 'cdf')
    # Calling cdf(args, kwargs) (line 27)
    cdf_call_result_589153 = invoke(stypy.reporting.localization.Localization(__file__, 27, 11), cdf_589150, *[z_589151], **kwargs_589152)
    
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type', cdf_call_result_589153)
    
    # ################# End of 'von_mises_cdf_normalapprox(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'von_mises_cdf_normalapprox' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_589154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_589154)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'von_mises_cdf_normalapprox'
    return stypy_return_type_589154

# Assigning a type to the variable 'von_mises_cdf_normalapprox' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'von_mises_cdf_normalapprox', von_mises_cdf_normalapprox)

@norecursion
def von_mises_cdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'von_mises_cdf'
    module_type_store = module_type_store.open_function_context('von_mises_cdf', 30, 0, False)
    
    # Passed parameters checking function
    von_mises_cdf.stypy_localization = localization
    von_mises_cdf.stypy_type_of_self = None
    von_mises_cdf.stypy_type_store = module_type_store
    von_mises_cdf.stypy_function_name = 'von_mises_cdf'
    von_mises_cdf.stypy_param_names_list = ['k', 'x']
    von_mises_cdf.stypy_varargs_param_name = None
    von_mises_cdf.stypy_kwargs_param_name = None
    von_mises_cdf.stypy_call_defaults = defaults
    von_mises_cdf.stypy_call_varargs = varargs
    von_mises_cdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'von_mises_cdf', ['k', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'von_mises_cdf', localization, ['k', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'von_mises_cdf(...)' code ##################

    
    # Assigning a BinOp to a Name (line 31):
    
    # Assigning a BinOp to a Name (line 31):
    int_589155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'int')
    # Getting the type of 'np' (line 31)
    np_589156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'np')
    # Obtaining the member 'pi' of a type (line 31)
    pi_589157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), np_589156, 'pi')
    # Applying the binary operator '*' (line 31)
    result_mul_589158 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 9), '*', int_589155, pi_589157)
    
    
    # Call to round(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'x' (line 31)
    x_589161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'x', False)
    int_589162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'int')
    # Getting the type of 'np' (line 31)
    np_589163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'np', False)
    # Obtaining the member 'pi' of a type (line 31)
    pi_589164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 31), np_589163, 'pi')
    # Applying the binary operator '*' (line 31)
    result_mul_589165 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 29), '*', int_589162, pi_589164)
    
    # Applying the binary operator 'div' (line 31)
    result_div_589166 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 26), 'div', x_589161, result_mul_589165)
    
    # Processing the call keyword arguments (line 31)
    kwargs_589167 = {}
    # Getting the type of 'np' (line 31)
    np_589159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'np', False)
    # Obtaining the member 'round' of a type (line 31)
    round_589160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 17), np_589159, 'round')
    # Calling round(args, kwargs) (line 31)
    round_call_result_589168 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), round_589160, *[result_div_589166], **kwargs_589167)
    
    # Applying the binary operator '*' (line 31)
    result_mul_589169 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 16), '*', result_mul_589158, round_call_result_589168)
    
    # Assigning a type to the variable 'ix' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'ix', result_mul_589169)
    
    # Assigning a BinOp to a Name (line 32):
    
    # Assigning a BinOp to a Name (line 32):
    # Getting the type of 'x' (line 32)
    x_589170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'x')
    # Getting the type of 'ix' (line 32)
    ix_589171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 10), 'ix')
    # Applying the binary operator '-' (line 32)
    result_sub_589172 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 8), '-', x_589170, ix_589171)
    
    # Assigning a type to the variable 'x' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'x', result_sub_589172)
    
    # Assigning a Call to a Name (line 33):
    
    # Assigning a Call to a Name (line 33):
    
    # Call to float(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'k' (line 33)
    k_589174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'k', False)
    # Processing the call keyword arguments (line 33)
    kwargs_589175 = {}
    # Getting the type of 'float' (line 33)
    float_589173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'float', False)
    # Calling float(args, kwargs) (line 33)
    float_call_result_589176 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), float_589173, *[k_589174], **kwargs_589175)
    
    # Assigning a type to the variable 'k' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'k', float_call_result_589176)
    
    # Assigning a Num to a Name (line 36):
    
    # Assigning a Num to a Name (line 36):
    int_589177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'int')
    # Assigning a type to the variable 'CK' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'CK', int_589177)
    
    # Assigning a List to a Name (line 37):
    
    # Assigning a List to a Name (line 37):
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_589178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    float_589179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 8), list_589178, float_589179)
    # Adding element type (line 37)
    float_589180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 14), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 8), list_589178, float_589180)
    # Adding element type (line 37)
    float_589181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 8), list_589178, float_589181)
    # Adding element type (line 37)
    float_589182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 8), list_589178, float_589182)
    
    # Assigning a type to the variable 'a' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'a', list_589178)
    
    
    # Getting the type of 'k' (line 39)
    k_589183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 7), 'k')
    # Getting the type of 'CK' (line 39)
    CK_589184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'CK')
    # Applying the binary operator '<' (line 39)
    result_lt_589185 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 7), '<', k_589183, CK_589184)
    
    # Testing the type of an if condition (line 39)
    if_condition_589186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 4), result_lt_589185)
    # Assigning a type to the variable 'if_condition_589186' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'if_condition_589186', if_condition_589186)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to int(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Call to ceil(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Obtaining the type of the subscript
    int_589190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'int')
    # Getting the type of 'a' (line 40)
    a_589191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___589192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 24), a_589191, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_589193 = invoke(stypy.reporting.localization.Localization(__file__, 40, 24), getitem___589192, int_589190)
    
    
    # Obtaining the type of the subscript
    int_589194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 31), 'int')
    # Getting the type of 'a' (line 40)
    a_589195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 29), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___589196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 29), a_589195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_589197 = invoke(stypy.reporting.localization.Localization(__file__, 40, 29), getitem___589196, int_589194)
    
    # Getting the type of 'k' (line 40)
    k_589198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'k', False)
    # Applying the binary operator '*' (line 40)
    result_mul_589199 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 29), '*', subscript_call_result_589197, k_589198)
    
    # Applying the binary operator '+' (line 40)
    result_add_589200 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 24), '+', subscript_call_result_589193, result_mul_589199)
    
    
    # Obtaining the type of the subscript
    int_589201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'int')
    # Getting the type of 'a' (line 40)
    a_589202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 36), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___589203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 36), a_589202, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_589204 = invoke(stypy.reporting.localization.Localization(__file__, 40, 36), getitem___589203, int_589201)
    
    # Getting the type of 'k' (line 40)
    k_589205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 42), 'k', False)
    
    # Obtaining the type of the subscript
    int_589206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 46), 'int')
    # Getting the type of 'a' (line 40)
    a_589207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 44), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___589208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 44), a_589207, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_589209 = invoke(stypy.reporting.localization.Localization(__file__, 40, 44), getitem___589208, int_589206)
    
    # Applying the binary operator '+' (line 40)
    result_add_589210 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 42), '+', k_589205, subscript_call_result_589209)
    
    # Applying the binary operator 'div' (line 40)
    result_div_589211 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 36), 'div', subscript_call_result_589204, result_add_589210)
    
    # Applying the binary operator '-' (line 40)
    result_sub_589212 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 35), '-', result_add_589200, result_div_589211)
    
    # Processing the call keyword arguments (line 40)
    kwargs_589213 = {}
    # Getting the type of 'np' (line 40)
    np_589188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'np', False)
    # Obtaining the member 'ceil' of a type (line 40)
    ceil_589189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), np_589188, 'ceil')
    # Calling ceil(args, kwargs) (line 40)
    ceil_call_result_589214 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), ceil_589189, *[result_sub_589212], **kwargs_589213)
    
    # Processing the call keyword arguments (line 40)
    kwargs_589215 = {}
    # Getting the type of 'int' (line 40)
    int_589187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'int', False)
    # Calling int(args, kwargs) (line 40)
    int_call_result_589216 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), int_589187, *[ceil_call_result_589214], **kwargs_589215)
    
    # Assigning a type to the variable 'p' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'p', int_call_result_589216)
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to clip(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to von_mises_cdf_series(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'k' (line 42)
    k_589220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), 'k', False)
    # Getting the type of 'x' (line 42)
    x_589221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 43), 'x', False)
    # Getting the type of 'p' (line 42)
    p_589222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 45), 'p', False)
    # Processing the call keyword arguments (line 42)
    kwargs_589223 = {}
    # Getting the type of 'von_mises_cdf_series' (line 42)
    von_mises_cdf_series_589219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'von_mises_cdf_series', False)
    # Calling von_mises_cdf_series(args, kwargs) (line 42)
    von_mises_cdf_series_call_result_589224 = invoke(stypy.reporting.localization.Localization(__file__, 42, 20), von_mises_cdf_series_589219, *[k_589220, x_589221, p_589222], **kwargs_589223)
    
    int_589225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 48), 'int')
    int_589226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 50), 'int')
    # Processing the call keyword arguments (line 42)
    kwargs_589227 = {}
    # Getting the type of 'np' (line 42)
    np_589217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'np', False)
    # Obtaining the member 'clip' of a type (line 42)
    clip_589218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), np_589217, 'clip')
    # Calling clip(args, kwargs) (line 42)
    clip_call_result_589228 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), clip_589218, *[von_mises_cdf_series_call_result_589224, int_589225, int_589226], **kwargs_589227)
    
    # Assigning a type to the variable 'F' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'F', clip_call_result_589228)
    # SSA branch for the else part of an if statement (line 39)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 44):
    
    # Assigning a Call to a Name (line 44):
    
    # Call to von_mises_cdf_normalapprox(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'k' (line 44)
    k_589230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'k', False)
    # Getting the type of 'x' (line 44)
    x_589231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 42), 'x', False)
    # Processing the call keyword arguments (line 44)
    kwargs_589232 = {}
    # Getting the type of 'von_mises_cdf_normalapprox' (line 44)
    von_mises_cdf_normalapprox_589229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'von_mises_cdf_normalapprox', False)
    # Calling von_mises_cdf_normalapprox(args, kwargs) (line 44)
    von_mises_cdf_normalapprox_call_result_589233 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), von_mises_cdf_normalapprox_589229, *[k_589230, x_589231], **kwargs_589232)
    
    # Assigning a type to the variable 'F' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'F', von_mises_cdf_normalapprox_call_result_589233)
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'F' (line 46)
    F_589234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'F')
    # Getting the type of 'ix' (line 46)
    ix_589235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'ix')
    # Applying the binary operator '+' (line 46)
    result_add_589236 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 11), '+', F_589234, ix_589235)
    
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', result_add_589236)
    
    # ################# End of 'von_mises_cdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'von_mises_cdf' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_589237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_589237)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'von_mises_cdf'
    return stypy_return_type_589237

# Assigning a type to the variable 'von_mises_cdf' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'von_mises_cdf', von_mises_cdf)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
