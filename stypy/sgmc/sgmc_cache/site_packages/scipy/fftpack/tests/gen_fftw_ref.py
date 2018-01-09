
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from subprocess import Popen, PIPE, STDOUT
4: 
5: import numpy as np
6: 
7: SZ = [2, 3, 4, 8, 12, 15, 16, 17, 32, 64, 128, 256, 512, 1024]
8: 
9: 
10: def gen_data(dt):
11:     arrays = {}
12: 
13:     if dt == np.double:
14:         pg = './fftw_double'
15:     elif dt == np.float32:
16:         pg = './fftw_single'
17:     else:
18:         raise ValueError("unknown: %s" % dt)
19:     # Generate test data using FFTW for reference
20:     for type in [1, 2, 3, 4, 5, 6, 7, 8]:
21:         arrays[type] = {}
22:         for sz in SZ:
23:             a = Popen([pg, str(type), str(sz)], stdout=PIPE, stderr=STDOUT)
24:             st = [i.strip() for i in a.stdout.readlines()]
25:             arrays[type][sz] = np.fromstring(",".join(st), sep=',', dtype=dt)
26: 
27:     return arrays
28: 
29: # generate single precision data
30: data = gen_data(np.float32)
31: filename = 'fftw_single_ref'
32: # Save ref data into npz format
33: d = {'sizes': SZ}
34: for type in [1, 2, 3, 4]:
35:     for sz in SZ:
36:         d['dct_%d_%d' % (type, sz)] = data[type][sz]
37: 
38: d['sizes'] = SZ
39: for type in [5, 6, 7, 8]:
40:     for sz in SZ:
41:         d['dst_%d_%d' % (type-4, sz)] = data[type][sz]
42: np.savez(filename, **d)
43: 
44: 
45: # generate double precision data
46: data = gen_data(np.float64)
47: filename = 'fftw_double_ref'
48: # Save ref data into npz format
49: d = {'sizes': SZ}
50: for type in [1, 2, 3, 4]:
51:     for sz in SZ:
52:         d['dct_%d_%d' % (type, sz)] = data[type][sz]
53: 
54: d['sizes'] = SZ
55: for type in [5, 6, 7, 8]:
56:     for sz in SZ:
57:         d['dst_%d_%d' % (type-4, sz)] = data[type][sz]
58: np.savez(filename, **d)
59: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from subprocess import Popen, PIPE, STDOUT' statement (line 3)
try:
    from subprocess import Popen, PIPE, STDOUT

except:
    Popen = UndefinedType
    PIPE = UndefinedType
    STDOUT = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'subprocess', None, module_type_store, ['Popen', 'PIPE', 'STDOUT'], [Popen, PIPE, STDOUT])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_18725 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_18725) is not StypyTypeError):

    if (import_18725 != 'pyd_module'):
        __import__(import_18725)
        sys_modules_18726 = sys.modules[import_18725]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_18726.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_18725)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')


# Assigning a List to a Name (line 7):

# Obtaining an instance of the builtin type 'list' (line 7)
list_18727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_18728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18728)
# Adding element type (line 7)
int_18729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18729)
# Adding element type (line 7)
int_18730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18730)
# Adding element type (line 7)
int_18731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18731)
# Adding element type (line 7)
int_18732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18732)
# Adding element type (line 7)
int_18733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18733)
# Adding element type (line 7)
int_18734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18734)
# Adding element type (line 7)
int_18735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18735)
# Adding element type (line 7)
int_18736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18736)
# Adding element type (line 7)
int_18737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18737)
# Adding element type (line 7)
int_18738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18738)
# Adding element type (line 7)
int_18739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 47), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18739)
# Adding element type (line 7)
int_18740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18740)
# Adding element type (line 7)
int_18741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 5), list_18727, int_18741)

# Assigning a type to the variable 'SZ' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'SZ', list_18727)

@norecursion
def gen_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gen_data'
    module_type_store = module_type_store.open_function_context('gen_data', 10, 0, False)
    
    # Passed parameters checking function
    gen_data.stypy_localization = localization
    gen_data.stypy_type_of_self = None
    gen_data.stypy_type_store = module_type_store
    gen_data.stypy_function_name = 'gen_data'
    gen_data.stypy_param_names_list = ['dt']
    gen_data.stypy_varargs_param_name = None
    gen_data.stypy_kwargs_param_name = None
    gen_data.stypy_call_defaults = defaults
    gen_data.stypy_call_varargs = varargs
    gen_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gen_data', ['dt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gen_data', localization, ['dt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gen_data(...)' code ##################

    
    # Assigning a Dict to a Name (line 11):
    
    # Obtaining an instance of the builtin type 'dict' (line 11)
    dict_18742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 11)
    
    # Assigning a type to the variable 'arrays' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'arrays', dict_18742)
    
    
    # Getting the type of 'dt' (line 13)
    dt_18743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 7), 'dt')
    # Getting the type of 'np' (line 13)
    np_18744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'np')
    # Obtaining the member 'double' of a type (line 13)
    double_18745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 13), np_18744, 'double')
    # Applying the binary operator '==' (line 13)
    result_eq_18746 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 7), '==', dt_18743, double_18745)
    
    # Testing the type of an if condition (line 13)
    if_condition_18747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 4), result_eq_18746)
    # Assigning a type to the variable 'if_condition_18747' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'if_condition_18747', if_condition_18747)
    # SSA begins for if statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 14):
    str_18748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 13), 'str', './fftw_double')
    # Assigning a type to the variable 'pg' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'pg', str_18748)
    # SSA branch for the else part of an if statement (line 13)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dt' (line 15)
    dt_18749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'dt')
    # Getting the type of 'np' (line 15)
    np_18750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'np')
    # Obtaining the member 'float32' of a type (line 15)
    float32_18751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 15), np_18750, 'float32')
    # Applying the binary operator '==' (line 15)
    result_eq_18752 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 9), '==', dt_18749, float32_18751)
    
    # Testing the type of an if condition (line 15)
    if_condition_18753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 9), result_eq_18752)
    # Assigning a type to the variable 'if_condition_18753' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'if_condition_18753', if_condition_18753)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 16):
    str_18754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'str', './fftw_single')
    # Assigning a type to the variable 'pg' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'pg', str_18754)
    # SSA branch for the else part of an if statement (line 15)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 18)
    # Processing the call arguments (line 18)
    str_18756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'str', 'unknown: %s')
    # Getting the type of 'dt' (line 18)
    dt_18757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 41), 'dt', False)
    # Applying the binary operator '%' (line 18)
    result_mod_18758 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 25), '%', str_18756, dt_18757)
    
    # Processing the call keyword arguments (line 18)
    kwargs_18759 = {}
    # Getting the type of 'ValueError' (line 18)
    ValueError_18755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 18)
    ValueError_call_result_18760 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), ValueError_18755, *[result_mod_18758], **kwargs_18759)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 18, 8), ValueError_call_result_18760, 'raise parameter', BaseException)
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 13)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_18761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    int_18762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_18761, int_18762)
    # Adding element type (line 20)
    int_18763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_18761, int_18763)
    # Adding element type (line 20)
    int_18764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_18761, int_18764)
    # Adding element type (line 20)
    int_18765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_18761, int_18765)
    # Adding element type (line 20)
    int_18766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_18761, int_18766)
    # Adding element type (line 20)
    int_18767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_18761, int_18767)
    # Adding element type (line 20)
    int_18768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_18761, int_18768)
    # Adding element type (line 20)
    int_18769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_18761, int_18769)
    
    # Testing the type of a for loop iterable (line 20)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 4), list_18761)
    # Getting the type of the for loop variable (line 20)
    for_loop_var_18770 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 4), list_18761)
    # Assigning a type to the variable 'type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'type', for_loop_var_18770)
    # SSA begins for a for statement (line 20)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Dict to a Subscript (line 21):
    
    # Obtaining an instance of the builtin type 'dict' (line 21)
    dict_18771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 21)
    
    # Getting the type of 'arrays' (line 21)
    arrays_18772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'arrays')
    # Getting the type of 'type' (line 21)
    type_18773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'type')
    # Storing an element on a container (line 21)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), arrays_18772, (type_18773, dict_18771))
    
    # Getting the type of 'SZ' (line 22)
    SZ_18774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'SZ')
    # Testing the type of a for loop iterable (line 22)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 22, 8), SZ_18774)
    # Getting the type of the for loop variable (line 22)
    for_loop_var_18775 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 22, 8), SZ_18774)
    # Assigning a type to the variable 'sz' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'sz', for_loop_var_18775)
    # SSA begins for a for statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 23):
    
    # Call to Popen(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_18777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    # Getting the type of 'pg' (line 23)
    pg_18778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'pg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), list_18777, pg_18778)
    # Adding element type (line 23)
    
    # Call to str(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'type' (line 23)
    type_18780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 31), 'type', False)
    # Processing the call keyword arguments (line 23)
    kwargs_18781 = {}
    # Getting the type of 'str' (line 23)
    str_18779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 27), 'str', False)
    # Calling str(args, kwargs) (line 23)
    str_call_result_18782 = invoke(stypy.reporting.localization.Localization(__file__, 23, 27), str_18779, *[type_18780], **kwargs_18781)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), list_18777, str_call_result_18782)
    # Adding element type (line 23)
    
    # Call to str(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'sz' (line 23)
    sz_18784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 42), 'sz', False)
    # Processing the call keyword arguments (line 23)
    kwargs_18785 = {}
    # Getting the type of 'str' (line 23)
    str_18783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 38), 'str', False)
    # Calling str(args, kwargs) (line 23)
    str_call_result_18786 = invoke(stypy.reporting.localization.Localization(__file__, 23, 38), str_18783, *[sz_18784], **kwargs_18785)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), list_18777, str_call_result_18786)
    
    # Processing the call keyword arguments (line 23)
    # Getting the type of 'PIPE' (line 23)
    PIPE_18787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 55), 'PIPE', False)
    keyword_18788 = PIPE_18787
    # Getting the type of 'STDOUT' (line 23)
    STDOUT_18789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 68), 'STDOUT', False)
    keyword_18790 = STDOUT_18789
    kwargs_18791 = {'stderr': keyword_18790, 'stdout': keyword_18788}
    # Getting the type of 'Popen' (line 23)
    Popen_18776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'Popen', False)
    # Calling Popen(args, kwargs) (line 23)
    Popen_call_result_18792 = invoke(stypy.reporting.localization.Localization(__file__, 23, 16), Popen_18776, *[list_18777], **kwargs_18791)
    
    # Assigning a type to the variable 'a' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'a', Popen_call_result_18792)
    
    # Assigning a ListComp to a Name (line 24):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to readlines(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_18800 = {}
    # Getting the type of 'a' (line 24)
    a_18797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 37), 'a', False)
    # Obtaining the member 'stdout' of a type (line 24)
    stdout_18798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 37), a_18797, 'stdout')
    # Obtaining the member 'readlines' of a type (line 24)
    readlines_18799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 37), stdout_18798, 'readlines')
    # Calling readlines(args, kwargs) (line 24)
    readlines_call_result_18801 = invoke(stypy.reporting.localization.Localization(__file__, 24, 37), readlines_18799, *[], **kwargs_18800)
    
    comprehension_18802 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 18), readlines_call_result_18801)
    # Assigning a type to the variable 'i' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'i', comprehension_18802)
    
    # Call to strip(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_18795 = {}
    # Getting the type of 'i' (line 24)
    i_18793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'i', False)
    # Obtaining the member 'strip' of a type (line 24)
    strip_18794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 18), i_18793, 'strip')
    # Calling strip(args, kwargs) (line 24)
    strip_call_result_18796 = invoke(stypy.reporting.localization.Localization(__file__, 24, 18), strip_18794, *[], **kwargs_18795)
    
    list_18803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 18), list_18803, strip_call_result_18796)
    # Assigning a type to the variable 'st' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'st', list_18803)
    
    # Assigning a Call to a Subscript (line 25):
    
    # Call to fromstring(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to join(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'st' (line 25)
    st_18808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 54), 'st', False)
    # Processing the call keyword arguments (line 25)
    kwargs_18809 = {}
    str_18806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 45), 'str', ',')
    # Obtaining the member 'join' of a type (line 25)
    join_18807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 45), str_18806, 'join')
    # Calling join(args, kwargs) (line 25)
    join_call_result_18810 = invoke(stypy.reporting.localization.Localization(__file__, 25, 45), join_18807, *[st_18808], **kwargs_18809)
    
    # Processing the call keyword arguments (line 25)
    str_18811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 63), 'str', ',')
    keyword_18812 = str_18811
    # Getting the type of 'dt' (line 25)
    dt_18813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 74), 'dt', False)
    keyword_18814 = dt_18813
    kwargs_18815 = {'dtype': keyword_18814, 'sep': keyword_18812}
    # Getting the type of 'np' (line 25)
    np_18804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'np', False)
    # Obtaining the member 'fromstring' of a type (line 25)
    fromstring_18805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 31), np_18804, 'fromstring')
    # Calling fromstring(args, kwargs) (line 25)
    fromstring_call_result_18816 = invoke(stypy.reporting.localization.Localization(__file__, 25, 31), fromstring_18805, *[join_call_result_18810], **kwargs_18815)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'type' (line 25)
    type_18817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'type')
    # Getting the type of 'arrays' (line 25)
    arrays_18818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'arrays')
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___18819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), arrays_18818, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_18820 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), getitem___18819, type_18817)
    
    # Getting the type of 'sz' (line 25)
    sz_18821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'sz')
    # Storing an element on a container (line 25)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), subscript_call_result_18820, (sz_18821, fromstring_call_result_18816))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'arrays' (line 27)
    arrays_18822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'arrays')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type', arrays_18822)
    
    # ################# End of 'gen_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gen_data' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_18823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18823)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gen_data'
    return stypy_return_type_18823

# Assigning a type to the variable 'gen_data' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'gen_data', gen_data)

# Assigning a Call to a Name (line 30):

# Call to gen_data(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of 'np' (line 30)
np_18825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'np', False)
# Obtaining the member 'float32' of a type (line 30)
float32_18826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 16), np_18825, 'float32')
# Processing the call keyword arguments (line 30)
kwargs_18827 = {}
# Getting the type of 'gen_data' (line 30)
gen_data_18824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'gen_data', False)
# Calling gen_data(args, kwargs) (line 30)
gen_data_call_result_18828 = invoke(stypy.reporting.localization.Localization(__file__, 30, 7), gen_data_18824, *[float32_18826], **kwargs_18827)

# Assigning a type to the variable 'data' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'data', gen_data_call_result_18828)

# Assigning a Str to a Name (line 31):
str_18829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 11), 'str', 'fftw_single_ref')
# Assigning a type to the variable 'filename' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'filename', str_18829)

# Assigning a Dict to a Name (line 33):

# Obtaining an instance of the builtin type 'dict' (line 33)
dict_18830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 33)
# Adding element type (key, value) (line 33)
str_18831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 5), 'str', 'sizes')
# Getting the type of 'SZ' (line 33)
SZ_18832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'SZ')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), dict_18830, (str_18831, SZ_18832))

# Assigning a type to the variable 'd' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'd', dict_18830)


# Obtaining an instance of the builtin type 'list' (line 34)
list_18833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)
int_18834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 12), list_18833, int_18834)
# Adding element type (line 34)
int_18835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 12), list_18833, int_18835)
# Adding element type (line 34)
int_18836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 12), list_18833, int_18836)
# Adding element type (line 34)
int_18837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 12), list_18833, int_18837)

# Testing the type of a for loop iterable (line 34)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 0), list_18833)
# Getting the type of the for loop variable (line 34)
for_loop_var_18838 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 0), list_18833)
# Assigning a type to the variable 'type' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'type', for_loop_var_18838)
# SSA begins for a for statement (line 34)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Getting the type of 'SZ' (line 35)
SZ_18839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'SZ')
# Testing the type of a for loop iterable (line 35)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 4), SZ_18839)
# Getting the type of the for loop variable (line 35)
for_loop_var_18840 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 4), SZ_18839)
# Assigning a type to the variable 'sz' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'sz', for_loop_var_18840)
# SSA begins for a for statement (line 35)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Subscript to a Subscript (line 36):

# Obtaining the type of the subscript
# Getting the type of 'sz' (line 36)
sz_18841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 49), 'sz')

# Obtaining the type of the subscript
# Getting the type of 'type' (line 36)
type_18842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 43), 'type')
# Getting the type of 'data' (line 36)
data_18843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 38), 'data')
# Obtaining the member '__getitem__' of a type (line 36)
getitem___18844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 38), data_18843, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 36)
subscript_call_result_18845 = invoke(stypy.reporting.localization.Localization(__file__, 36, 38), getitem___18844, type_18842)

# Obtaining the member '__getitem__' of a type (line 36)
getitem___18846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 38), subscript_call_result_18845, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 36)
subscript_call_result_18847 = invoke(stypy.reporting.localization.Localization(__file__, 36, 38), getitem___18846, sz_18841)

# Getting the type of 'd' (line 36)
d_18848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'd')
str_18849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 10), 'str', 'dct_%d_%d')

# Obtaining an instance of the builtin type 'tuple' (line 36)
tuple_18850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 36)
# Adding element type (line 36)
# Getting the type of 'type' (line 36)
type_18851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), tuple_18850, type_18851)
# Adding element type (line 36)
# Getting the type of 'sz' (line 36)
sz_18852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'sz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), tuple_18850, sz_18852)

# Applying the binary operator '%' (line 36)
result_mod_18853 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 10), '%', str_18849, tuple_18850)

# Storing an element on a container (line 36)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), d_18848, (result_mod_18853, subscript_call_result_18847))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Subscript (line 38):
# Getting the type of 'SZ' (line 38)
SZ_18854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'SZ')
# Getting the type of 'd' (line 38)
d_18855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'd')
str_18856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 2), 'str', 'sizes')
# Storing an element on a container (line 38)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 0), d_18855, (str_18856, SZ_18854))


# Obtaining an instance of the builtin type 'list' (line 39)
list_18857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 39)
# Adding element type (line 39)
int_18858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 12), list_18857, int_18858)
# Adding element type (line 39)
int_18859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 12), list_18857, int_18859)
# Adding element type (line 39)
int_18860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 12), list_18857, int_18860)
# Adding element type (line 39)
int_18861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 12), list_18857, int_18861)

# Testing the type of a for loop iterable (line 39)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 0), list_18857)
# Getting the type of the for loop variable (line 39)
for_loop_var_18862 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 0), list_18857)
# Assigning a type to the variable 'type' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'type', for_loop_var_18862)
# SSA begins for a for statement (line 39)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Getting the type of 'SZ' (line 40)
SZ_18863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'SZ')
# Testing the type of a for loop iterable (line 40)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 4), SZ_18863)
# Getting the type of the for loop variable (line 40)
for_loop_var_18864 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 4), SZ_18863)
# Assigning a type to the variable 'sz' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'sz', for_loop_var_18864)
# SSA begins for a for statement (line 40)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Subscript to a Subscript (line 41):

# Obtaining the type of the subscript
# Getting the type of 'sz' (line 41)
sz_18865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 51), 'sz')

# Obtaining the type of the subscript
# Getting the type of 'type' (line 41)
type_18866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 45), 'type')
# Getting the type of 'data' (line 41)
data_18867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), 'data')
# Obtaining the member '__getitem__' of a type (line 41)
getitem___18868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 40), data_18867, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 41)
subscript_call_result_18869 = invoke(stypy.reporting.localization.Localization(__file__, 41, 40), getitem___18868, type_18866)

# Obtaining the member '__getitem__' of a type (line 41)
getitem___18870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 40), subscript_call_result_18869, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 41)
subscript_call_result_18871 = invoke(stypy.reporting.localization.Localization(__file__, 41, 40), getitem___18870, sz_18865)

# Getting the type of 'd' (line 41)
d_18872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'd')
str_18873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 10), 'str', 'dst_%d_%d')

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_18874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
# Getting the type of 'type' (line 41)
type_18875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'type')
int_18876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'int')
# Applying the binary operator '-' (line 41)
result_sub_18877 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 25), '-', type_18875, int_18876)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 25), tuple_18874, result_sub_18877)
# Adding element type (line 41)
# Getting the type of 'sz' (line 41)
sz_18878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 33), 'sz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 25), tuple_18874, sz_18878)

# Applying the binary operator '%' (line 41)
result_mod_18879 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 10), '%', str_18873, tuple_18874)

# Storing an element on a container (line 41)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 8), d_18872, (result_mod_18879, subscript_call_result_18871))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Call to savez(...): (line 42)
# Processing the call arguments (line 42)
# Getting the type of 'filename' (line 42)
filename_18882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 9), 'filename', False)
# Processing the call keyword arguments (line 42)
# Getting the type of 'd' (line 42)
d_18883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'd', False)
kwargs_18884 = {'d_18883': d_18883}
# Getting the type of 'np' (line 42)
np_18880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'np', False)
# Obtaining the member 'savez' of a type (line 42)
savez_18881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 0), np_18880, 'savez')
# Calling savez(args, kwargs) (line 42)
savez_call_result_18885 = invoke(stypy.reporting.localization.Localization(__file__, 42, 0), savez_18881, *[filename_18882], **kwargs_18884)


# Assigning a Call to a Name (line 46):

# Call to gen_data(...): (line 46)
# Processing the call arguments (line 46)
# Getting the type of 'np' (line 46)
np_18887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'np', False)
# Obtaining the member 'float64' of a type (line 46)
float64_18888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), np_18887, 'float64')
# Processing the call keyword arguments (line 46)
kwargs_18889 = {}
# Getting the type of 'gen_data' (line 46)
gen_data_18886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'gen_data', False)
# Calling gen_data(args, kwargs) (line 46)
gen_data_call_result_18890 = invoke(stypy.reporting.localization.Localization(__file__, 46, 7), gen_data_18886, *[float64_18888], **kwargs_18889)

# Assigning a type to the variable 'data' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'data', gen_data_call_result_18890)

# Assigning a Str to a Name (line 47):
str_18891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'str', 'fftw_double_ref')
# Assigning a type to the variable 'filename' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'filename', str_18891)

# Assigning a Dict to a Name (line 49):

# Obtaining an instance of the builtin type 'dict' (line 49)
dict_18892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 49)
# Adding element type (key, value) (line 49)
str_18893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 5), 'str', 'sizes')
# Getting the type of 'SZ' (line 49)
SZ_18894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 14), 'SZ')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 4), dict_18892, (str_18893, SZ_18894))

# Assigning a type to the variable 'd' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'd', dict_18892)


# Obtaining an instance of the builtin type 'list' (line 50)
list_18895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 50)
# Adding element type (line 50)
int_18896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), list_18895, int_18896)
# Adding element type (line 50)
int_18897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), list_18895, int_18897)
# Adding element type (line 50)
int_18898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), list_18895, int_18898)
# Adding element type (line 50)
int_18899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), list_18895, int_18899)

# Testing the type of a for loop iterable (line 50)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 0), list_18895)
# Getting the type of the for loop variable (line 50)
for_loop_var_18900 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 0), list_18895)
# Assigning a type to the variable 'type' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'type', for_loop_var_18900)
# SSA begins for a for statement (line 50)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Getting the type of 'SZ' (line 51)
SZ_18901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'SZ')
# Testing the type of a for loop iterable (line 51)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 4), SZ_18901)
# Getting the type of the for loop variable (line 51)
for_loop_var_18902 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 4), SZ_18901)
# Assigning a type to the variable 'sz' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'sz', for_loop_var_18902)
# SSA begins for a for statement (line 51)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Subscript to a Subscript (line 52):

# Obtaining the type of the subscript
# Getting the type of 'sz' (line 52)
sz_18903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 49), 'sz')

# Obtaining the type of the subscript
# Getting the type of 'type' (line 52)
type_18904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 43), 'type')
# Getting the type of 'data' (line 52)
data_18905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 38), 'data')
# Obtaining the member '__getitem__' of a type (line 52)
getitem___18906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 38), data_18905, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 52)
subscript_call_result_18907 = invoke(stypy.reporting.localization.Localization(__file__, 52, 38), getitem___18906, type_18904)

# Obtaining the member '__getitem__' of a type (line 52)
getitem___18908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 38), subscript_call_result_18907, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 52)
subscript_call_result_18909 = invoke(stypy.reporting.localization.Localization(__file__, 52, 38), getitem___18908, sz_18903)

# Getting the type of 'd' (line 52)
d_18910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'd')
str_18911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 10), 'str', 'dct_%d_%d')

# Obtaining an instance of the builtin type 'tuple' (line 52)
tuple_18912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 52)
# Adding element type (line 52)
# Getting the type of 'type' (line 52)
type_18913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 25), tuple_18912, type_18913)
# Adding element type (line 52)
# Getting the type of 'sz' (line 52)
sz_18914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'sz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 25), tuple_18912, sz_18914)

# Applying the binary operator '%' (line 52)
result_mod_18915 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 10), '%', str_18911, tuple_18912)

# Storing an element on a container (line 52)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 8), d_18910, (result_mod_18915, subscript_call_result_18909))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Subscript (line 54):
# Getting the type of 'SZ' (line 54)
SZ_18916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'SZ')
# Getting the type of 'd' (line 54)
d_18917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'd')
str_18918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 2), 'str', 'sizes')
# Storing an element on a container (line 54)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 0), d_18917, (str_18918, SZ_18916))


# Obtaining an instance of the builtin type 'list' (line 55)
list_18919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 55)
# Adding element type (line 55)
int_18920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 12), list_18919, int_18920)
# Adding element type (line 55)
int_18921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 12), list_18919, int_18921)
# Adding element type (line 55)
int_18922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 12), list_18919, int_18922)
# Adding element type (line 55)
int_18923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 12), list_18919, int_18923)

# Testing the type of a for loop iterable (line 55)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 0), list_18919)
# Getting the type of the for loop variable (line 55)
for_loop_var_18924 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 0), list_18919)
# Assigning a type to the variable 'type' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'type', for_loop_var_18924)
# SSA begins for a for statement (line 55)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Getting the type of 'SZ' (line 56)
SZ_18925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'SZ')
# Testing the type of a for loop iterable (line 56)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 56, 4), SZ_18925)
# Getting the type of the for loop variable (line 56)
for_loop_var_18926 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 56, 4), SZ_18925)
# Assigning a type to the variable 'sz' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'sz', for_loop_var_18926)
# SSA begins for a for statement (line 56)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Subscript to a Subscript (line 57):

# Obtaining the type of the subscript
# Getting the type of 'sz' (line 57)
sz_18927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 51), 'sz')

# Obtaining the type of the subscript
# Getting the type of 'type' (line 57)
type_18928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'type')
# Getting the type of 'data' (line 57)
data_18929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 40), 'data')
# Obtaining the member '__getitem__' of a type (line 57)
getitem___18930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 40), data_18929, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 57)
subscript_call_result_18931 = invoke(stypy.reporting.localization.Localization(__file__, 57, 40), getitem___18930, type_18928)

# Obtaining the member '__getitem__' of a type (line 57)
getitem___18932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 40), subscript_call_result_18931, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 57)
subscript_call_result_18933 = invoke(stypy.reporting.localization.Localization(__file__, 57, 40), getitem___18932, sz_18927)

# Getting the type of 'd' (line 57)
d_18934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'd')
str_18935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 10), 'str', 'dst_%d_%d')

# Obtaining an instance of the builtin type 'tuple' (line 57)
tuple_18936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 57)
# Adding element type (line 57)
# Getting the type of 'type' (line 57)
type_18937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'type')
int_18938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'int')
# Applying the binary operator '-' (line 57)
result_sub_18939 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 25), '-', type_18937, int_18938)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 25), tuple_18936, result_sub_18939)
# Adding element type (line 57)
# Getting the type of 'sz' (line 57)
sz_18940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 33), 'sz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 25), tuple_18936, sz_18940)

# Applying the binary operator '%' (line 57)
result_mod_18941 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 10), '%', str_18935, tuple_18936)

# Storing an element on a container (line 57)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 8), d_18934, (result_mod_18941, subscript_call_result_18933))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Call to savez(...): (line 58)
# Processing the call arguments (line 58)
# Getting the type of 'filename' (line 58)
filename_18944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'filename', False)
# Processing the call keyword arguments (line 58)
# Getting the type of 'd' (line 58)
d_18945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'd', False)
kwargs_18946 = {'d_18945': d_18945}
# Getting the type of 'np' (line 58)
np_18942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'np', False)
# Obtaining the member 'savez' of a type (line 58)
savez_18943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 0), np_18942, 'savez')
# Calling savez(args, kwargs) (line 58)
savez_call_result_18947 = invoke(stypy.reporting.localization.Localization(__file__, 58, 0), savez_18943, *[filename_18944], **kwargs_18946)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
