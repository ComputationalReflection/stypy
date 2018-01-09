
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Parameters used in test and benchmark methods '''
2: from __future__ import division, print_function, absolute_import
3: 
4: from random import random
5: 
6: from scipy.optimize import zeros as cc
7: 
8: 
9: def f1(x):
10:     return x*(x-1.)
11: 
12: 
13: def f2(x):
14:     return x**2 - 1
15: 
16: 
17: def f3(x):
18:     return x*(x-1.)*(x-2.)*(x-3.)
19: 
20: 
21: def f4(x):
22:     if x > 1:
23:         return 1.0 + .1*x
24:     if x < 1:
25:         return -1.0 + .1*x
26:     return 0
27: 
28: 
29: def f5(x):
30:     if x != 1:
31:         return 1.0/(1. - x)
32:     return 0
33: 
34: 
35: def f6(x):
36:     if x > 1:
37:         return random()
38:     elif x < 1:
39:         return -random()
40:     else:
41:         return 0
42: 
43: description = '''
44: f2 is a symmetric parabola, x**2 - 1
45: f3 is a quartic polynomial with large hump in interval
46: f4 is step function with a discontinuity at 1
47: f5 is a hyperbola with vertical asymptote at 1
48: f6 has random values positive to left of 1, negative to right
49: 
50: of course these are not real problems. They just test how the
51: 'good' solvers behave in bad circumstances where bisection is
52: really the best. A good solver should not be much worse than
53: bisection in such circumstance, while being faster for smooth
54: monotone sorts of functions.
55: '''
56: 
57: methods = [cc.bisect,cc.ridder,cc.brenth,cc.brentq]
58: mstrings = ['cc.bisect','cc.ridder','cc.brenth','cc.brentq']
59: functions = [f2,f3,f4,f5,f6]
60: fstrings = ['f2','f3','f4','f5','f6']
61: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_204632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', ' Parameters used in test and benchmark methods ')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from random import random' statement (line 4)
try:
    from random import random

except:
    random = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'random', None, module_type_store, ['random'], [random])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.optimize import cc' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204633 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize')

if (type(import_204633) is not StypyTypeError):

    if (import_204633 != 'pyd_module'):
        __import__(import_204633)
        sys_modules_204634 = sys.modules[import_204633]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', sys_modules_204634.module_type_store, module_type_store, ['zeros'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_204634, sys_modules_204634.module_type_store, module_type_store)
    else:
        from scipy.optimize import zeros as cc

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', None, module_type_store, ['zeros'], [cc])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', import_204633)

# Adding an alias
module_type_store.add_alias('cc', 'zeros')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


@norecursion
def f1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f1'
    module_type_store = module_type_store.open_function_context('f1', 9, 0, False)
    
    # Passed parameters checking function
    f1.stypy_localization = localization
    f1.stypy_type_of_self = None
    f1.stypy_type_store = module_type_store
    f1.stypy_function_name = 'f1'
    f1.stypy_param_names_list = ['x']
    f1.stypy_varargs_param_name = None
    f1.stypy_kwargs_param_name = None
    f1.stypy_call_defaults = defaults
    f1.stypy_call_varargs = varargs
    f1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f1', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f1', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f1(...)' code ##################

    # Getting the type of 'x' (line 10)
    x_204635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'x')
    # Getting the type of 'x' (line 10)
    x_204636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'x')
    float_204637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'float')
    # Applying the binary operator '-' (line 10)
    result_sub_204638 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 14), '-', x_204636, float_204637)
    
    # Applying the binary operator '*' (line 10)
    result_mul_204639 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 11), '*', x_204635, result_sub_204638)
    
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type', result_mul_204639)
    
    # ################# End of 'f1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f1' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_204640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204640)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f1'
    return stypy_return_type_204640

# Assigning a type to the variable 'f1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'f1', f1)

@norecursion
def f2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f2'
    module_type_store = module_type_store.open_function_context('f2', 13, 0, False)
    
    # Passed parameters checking function
    f2.stypy_localization = localization
    f2.stypy_type_of_self = None
    f2.stypy_type_store = module_type_store
    f2.stypy_function_name = 'f2'
    f2.stypy_param_names_list = ['x']
    f2.stypy_varargs_param_name = None
    f2.stypy_kwargs_param_name = None
    f2.stypy_call_defaults = defaults
    f2.stypy_call_varargs = varargs
    f2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f2', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f2', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f2(...)' code ##################

    # Getting the type of 'x' (line 14)
    x_204641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'x')
    int_204642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
    # Applying the binary operator '**' (line 14)
    result_pow_204643 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 11), '**', x_204641, int_204642)
    
    int_204644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'int')
    # Applying the binary operator '-' (line 14)
    result_sub_204645 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 11), '-', result_pow_204643, int_204644)
    
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type', result_sub_204645)
    
    # ################# End of 'f2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f2' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_204646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204646)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f2'
    return stypy_return_type_204646

# Assigning a type to the variable 'f2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'f2', f2)

@norecursion
def f3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f3'
    module_type_store = module_type_store.open_function_context('f3', 17, 0, False)
    
    # Passed parameters checking function
    f3.stypy_localization = localization
    f3.stypy_type_of_self = None
    f3.stypy_type_store = module_type_store
    f3.stypy_function_name = 'f3'
    f3.stypy_param_names_list = ['x']
    f3.stypy_varargs_param_name = None
    f3.stypy_kwargs_param_name = None
    f3.stypy_call_defaults = defaults
    f3.stypy_call_varargs = varargs
    f3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f3', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f3', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f3(...)' code ##################

    # Getting the type of 'x' (line 18)
    x_204647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'x')
    # Getting the type of 'x' (line 18)
    x_204648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'x')
    float_204649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'float')
    # Applying the binary operator '-' (line 18)
    result_sub_204650 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 14), '-', x_204648, float_204649)
    
    # Applying the binary operator '*' (line 18)
    result_mul_204651 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), '*', x_204647, result_sub_204650)
    
    # Getting the type of 'x' (line 18)
    x_204652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'x')
    float_204653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 23), 'float')
    # Applying the binary operator '-' (line 18)
    result_sub_204654 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 21), '-', x_204652, float_204653)
    
    # Applying the binary operator '*' (line 18)
    result_mul_204655 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 19), '*', result_mul_204651, result_sub_204654)
    
    # Getting the type of 'x' (line 18)
    x_204656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 28), 'x')
    float_204657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'float')
    # Applying the binary operator '-' (line 18)
    result_sub_204658 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 28), '-', x_204656, float_204657)
    
    # Applying the binary operator '*' (line 18)
    result_mul_204659 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 26), '*', result_mul_204655, result_sub_204658)
    
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', result_mul_204659)
    
    # ################# End of 'f3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f3' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_204660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204660)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f3'
    return stypy_return_type_204660

# Assigning a type to the variable 'f3' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'f3', f3)

@norecursion
def f4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f4'
    module_type_store = module_type_store.open_function_context('f4', 21, 0, False)
    
    # Passed parameters checking function
    f4.stypy_localization = localization
    f4.stypy_type_of_self = None
    f4.stypy_type_store = module_type_store
    f4.stypy_function_name = 'f4'
    f4.stypy_param_names_list = ['x']
    f4.stypy_varargs_param_name = None
    f4.stypy_kwargs_param_name = None
    f4.stypy_call_defaults = defaults
    f4.stypy_call_varargs = varargs
    f4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f4', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f4', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f4(...)' code ##################

    
    
    # Getting the type of 'x' (line 22)
    x_204661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 7), 'x')
    int_204662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'int')
    # Applying the binary operator '>' (line 22)
    result_gt_204663 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 7), '>', x_204661, int_204662)
    
    # Testing the type of an if condition (line 22)
    if_condition_204664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 4), result_gt_204663)
    # Assigning a type to the variable 'if_condition_204664' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'if_condition_204664', if_condition_204664)
    # SSA begins for if statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    float_204665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'float')
    float_204666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'float')
    # Getting the type of 'x' (line 23)
    x_204667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'x')
    # Applying the binary operator '*' (line 23)
    result_mul_204668 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 21), '*', float_204666, x_204667)
    
    # Applying the binary operator '+' (line 23)
    result_add_204669 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 15), '+', float_204665, result_mul_204668)
    
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', result_add_204669)
    # SSA join for if statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 24)
    x_204670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 7), 'x')
    int_204671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'int')
    # Applying the binary operator '<' (line 24)
    result_lt_204672 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 7), '<', x_204670, int_204671)
    
    # Testing the type of an if condition (line 24)
    if_condition_204673 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 4), result_lt_204672)
    # Assigning a type to the variable 'if_condition_204673' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'if_condition_204673', if_condition_204673)
    # SSA begins for if statement (line 24)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    float_204674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'float')
    float_204675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'float')
    # Getting the type of 'x' (line 25)
    x_204676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'x')
    # Applying the binary operator '*' (line 25)
    result_mul_204677 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 22), '*', float_204675, x_204676)
    
    # Applying the binary operator '+' (line 25)
    result_add_204678 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), '+', float_204674, result_mul_204677)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', result_add_204678)
    # SSA join for if statement (line 24)
    module_type_store = module_type_store.join_ssa_context()
    
    int_204679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', int_204679)
    
    # ################# End of 'f4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f4' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_204680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204680)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f4'
    return stypy_return_type_204680

# Assigning a type to the variable 'f4' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'f4', f4)

@norecursion
def f5(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f5'
    module_type_store = module_type_store.open_function_context('f5', 29, 0, False)
    
    # Passed parameters checking function
    f5.stypy_localization = localization
    f5.stypy_type_of_self = None
    f5.stypy_type_store = module_type_store
    f5.stypy_function_name = 'f5'
    f5.stypy_param_names_list = ['x']
    f5.stypy_varargs_param_name = None
    f5.stypy_kwargs_param_name = None
    f5.stypy_call_defaults = defaults
    f5.stypy_call_varargs = varargs
    f5.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f5', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f5', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f5(...)' code ##################

    
    
    # Getting the type of 'x' (line 30)
    x_204681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'x')
    int_204682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 12), 'int')
    # Applying the binary operator '!=' (line 30)
    result_ne_204683 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 7), '!=', x_204681, int_204682)
    
    # Testing the type of an if condition (line 30)
    if_condition_204684 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 4), result_ne_204683)
    # Assigning a type to the variable 'if_condition_204684' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'if_condition_204684', if_condition_204684)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    float_204685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 15), 'float')
    float_204686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'float')
    # Getting the type of 'x' (line 31)
    x_204687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'x')
    # Applying the binary operator '-' (line 31)
    result_sub_204688 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 20), '-', float_204686, x_204687)
    
    # Applying the binary operator 'div' (line 31)
    result_div_204689 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), 'div', float_204685, result_sub_204688)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', result_div_204689)
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    int_204690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', int_204690)
    
    # ################# End of 'f5(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f5' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_204691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204691)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f5'
    return stypy_return_type_204691

# Assigning a type to the variable 'f5' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'f5', f5)

@norecursion
def f6(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f6'
    module_type_store = module_type_store.open_function_context('f6', 35, 0, False)
    
    # Passed parameters checking function
    f6.stypy_localization = localization
    f6.stypy_type_of_self = None
    f6.stypy_type_store = module_type_store
    f6.stypy_function_name = 'f6'
    f6.stypy_param_names_list = ['x']
    f6.stypy_varargs_param_name = None
    f6.stypy_kwargs_param_name = None
    f6.stypy_call_defaults = defaults
    f6.stypy_call_varargs = varargs
    f6.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f6', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f6', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f6(...)' code ##################

    
    
    # Getting the type of 'x' (line 36)
    x_204692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 7), 'x')
    int_204693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'int')
    # Applying the binary operator '>' (line 36)
    result_gt_204694 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 7), '>', x_204692, int_204693)
    
    # Testing the type of an if condition (line 36)
    if_condition_204695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 4), result_gt_204694)
    # Assigning a type to the variable 'if_condition_204695' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'if_condition_204695', if_condition_204695)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to random(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_204697 = {}
    # Getting the type of 'random' (line 37)
    random_204696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'random', False)
    # Calling random(args, kwargs) (line 37)
    random_call_result_204698 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), random_204696, *[], **kwargs_204697)
    
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', random_call_result_204698)
    # SSA branch for the else part of an if statement (line 36)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'x' (line 38)
    x_204699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'x')
    int_204700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'int')
    # Applying the binary operator '<' (line 38)
    result_lt_204701 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 9), '<', x_204699, int_204700)
    
    # Testing the type of an if condition (line 38)
    if_condition_204702 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 9), result_lt_204701)
    # Assigning a type to the variable 'if_condition_204702' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'if_condition_204702', if_condition_204702)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to random(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_204704 = {}
    # Getting the type of 'random' (line 39)
    random_204703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'random', False)
    # Calling random(args, kwargs) (line 39)
    random_call_result_204705 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), random_204703, *[], **kwargs_204704)
    
    # Applying the 'usub' unary operator (line 39)
    result___neg___204706 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), 'usub', random_call_result_204705)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', result___neg___204706)
    # SSA branch for the else part of an if statement (line 38)
    module_type_store.open_ssa_branch('else')
    int_204707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', int_204707)
    # SSA join for if statement (line 38)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'f6(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f6' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_204708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204708)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f6'
    return stypy_return_type_204708

# Assigning a type to the variable 'f6' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'f6', f6)

# Assigning a Str to a Name (line 43):
str_204709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', "\nf2 is a symmetric parabola, x**2 - 1\nf3 is a quartic polynomial with large hump in interval\nf4 is step function with a discontinuity at 1\nf5 is a hyperbola with vertical asymptote at 1\nf6 has random values positive to left of 1, negative to right\n\nof course these are not real problems. They just test how the\n'good' solvers behave in bad circumstances where bisection is\nreally the best. A good solver should not be much worse than\nbisection in such circumstance, while being faster for smooth\nmonotone sorts of functions.\n")
# Assigning a type to the variable 'description' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'description', str_204709)

# Assigning a List to a Name (line 57):

# Obtaining an instance of the builtin type 'list' (line 57)
list_204710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 57)
# Adding element type (line 57)
# Getting the type of 'cc' (line 57)
cc_204711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'cc')
# Obtaining the member 'bisect' of a type (line 57)
bisect_204712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), cc_204711, 'bisect')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_204710, bisect_204712)
# Adding element type (line 57)
# Getting the type of 'cc' (line 57)
cc_204713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'cc')
# Obtaining the member 'ridder' of a type (line 57)
ridder_204714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 21), cc_204713, 'ridder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_204710, ridder_204714)
# Adding element type (line 57)
# Getting the type of 'cc' (line 57)
cc_204715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 31), 'cc')
# Obtaining the member 'brenth' of a type (line 57)
brenth_204716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 31), cc_204715, 'brenth')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_204710, brenth_204716)
# Adding element type (line 57)
# Getting the type of 'cc' (line 57)
cc_204717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 41), 'cc')
# Obtaining the member 'brentq' of a type (line 57)
brentq_204718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 41), cc_204717, 'brentq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 10), list_204710, brentq_204718)

# Assigning a type to the variable 'methods' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'methods', list_204710)

# Assigning a List to a Name (line 58):

# Obtaining an instance of the builtin type 'list' (line 58)
list_204719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 58)
# Adding element type (line 58)
str_204720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 12), 'str', 'cc.bisect')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 11), list_204719, str_204720)
# Adding element type (line 58)
str_204721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 24), 'str', 'cc.ridder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 11), list_204719, str_204721)
# Adding element type (line 58)
str_204722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 36), 'str', 'cc.brenth')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 11), list_204719, str_204722)
# Adding element type (line 58)
str_204723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 48), 'str', 'cc.brentq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 11), list_204719, str_204723)

# Assigning a type to the variable 'mstrings' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'mstrings', list_204719)

# Assigning a List to a Name (line 59):

# Obtaining an instance of the builtin type 'list' (line 59)
list_204724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 59)
# Adding element type (line 59)
# Getting the type of 'f2' (line 59)
f2_204725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'f2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 12), list_204724, f2_204725)
# Adding element type (line 59)
# Getting the type of 'f3' (line 59)
f3_204726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'f3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 12), list_204724, f3_204726)
# Adding element type (line 59)
# Getting the type of 'f4' (line 59)
f4_204727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'f4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 12), list_204724, f4_204727)
# Adding element type (line 59)
# Getting the type of 'f5' (line 59)
f5_204728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'f5')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 12), list_204724, f5_204728)
# Adding element type (line 59)
# Getting the type of 'f6' (line 59)
f6_204729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'f6')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 12), list_204724, f6_204729)

# Assigning a type to the variable 'functions' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'functions', list_204724)

# Assigning a List to a Name (line 60):

# Obtaining an instance of the builtin type 'list' (line 60)
list_204730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)
# Adding element type (line 60)
str_204731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'str', 'f2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 11), list_204730, str_204731)
# Adding element type (line 60)
str_204732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 17), 'str', 'f3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 11), list_204730, str_204732)
# Adding element type (line 60)
str_204733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'str', 'f4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 11), list_204730, str_204733)
# Adding element type (line 60)
str_204734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 27), 'str', 'f5')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 11), list_204730, str_204734)
# Adding element type (line 60)
str_204735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'str', 'f6')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 11), list_204730, str_204735)

# Assigning a type to the variable 'fstrings' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'fstrings', list_204730)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
