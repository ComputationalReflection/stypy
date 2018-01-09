
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import other_module_to_import_mixed
2: 
3: def f():
4:     return 3
5: 
6: r1 = f()
7: 
8: r2 = other_module_to_import_mixed.global_a
9: r3 = other_module_to_import_mixed.f_parent()
10: r4 = other_module_to_import_mixed.my_func
11: r5 = other_module_to_import_mixed.secondary_module_mixed
12: r6 = r5.number
13: 
14: 
15: r7 = other_module_to_import_mixed.my_func2
16: r7b = r7(2)
17: r8 = other_module_to_import_mixed.my_func3
18: r8b = r8()
19: r9 = other_module_to_import_mixed.secondary_module_mixed
20: r10 = r9.time.clock()
21: r11 = other_module_to_import_mixed.secondary_module_mixed.time
22: r12 = other_module_to_import_mixed.secondary_module_mixed.clock_func
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import other_module_to_import_mixed' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5046 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'other_module_to_import_mixed')

if (type(import_5046) is not StypyTypeError):

    if (import_5046 != 'pyd_module'):
        __import__(import_5046)
        sys_modules_5047 = sys.modules[import_5046]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'other_module_to_import_mixed', sys_modules_5047.module_type_store, module_type_store)
    else:
        import other_module_to_import_mixed

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'other_module_to_import_mixed', other_module_to_import_mixed, module_type_store)

else:
    # Assigning a type to the variable 'other_module_to_import_mixed' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'other_module_to_import_mixed', import_5046)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


@norecursion
def f(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f'
    module_type_store = module_type_store.open_function_context('f', 3, 0, False)
    
    # Passed parameters checking function
    f.stypy_localization = localization
    f.stypy_type_of_self = None
    f.stypy_type_store = module_type_store
    f.stypy_function_name = 'f'
    f.stypy_param_names_list = []
    f.stypy_varargs_param_name = None
    f.stypy_kwargs_param_name = None
    f.stypy_call_defaults = defaults
    f.stypy_call_varargs = varargs
    f.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f(...)' code ##################

    int_5048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_return_type', int_5048)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_5049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5049)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_5049

# Assigning a type to the variable 'f' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'f', f)

# Assigning a Call to a Name (line 6):

# Call to f(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_5051 = {}
# Getting the type of 'f' (line 6)
f_5050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'f', False)
# Calling f(args, kwargs) (line 6)
f_call_result_5052 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), f_5050, *[], **kwargs_5051)

# Assigning a type to the variable 'r1' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r1', f_call_result_5052)

# Assigning a Attribute to a Name (line 8):
# Getting the type of 'other_module_to_import_mixed' (line 8)
other_module_to_import_mixed_5053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'other_module_to_import_mixed')
# Obtaining the member 'global_a' of a type (line 8)
global_a_5054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), other_module_to_import_mixed_5053, 'global_a')
# Assigning a type to the variable 'r2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r2', global_a_5054)

# Assigning a Call to a Name (line 9):

# Call to f_parent(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_5057 = {}
# Getting the type of 'other_module_to_import_mixed' (line 9)
other_module_to_import_mixed_5055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'other_module_to_import_mixed', False)
# Obtaining the member 'f_parent' of a type (line 9)
f_parent_5056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), other_module_to_import_mixed_5055, 'f_parent')
# Calling f_parent(args, kwargs) (line 9)
f_parent_call_result_5058 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), f_parent_5056, *[], **kwargs_5057)

# Assigning a type to the variable 'r3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r3', f_parent_call_result_5058)

# Assigning a Attribute to a Name (line 10):
# Getting the type of 'other_module_to_import_mixed' (line 10)
other_module_to_import_mixed_5059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'other_module_to_import_mixed')
# Obtaining the member 'my_func' of a type (line 10)
my_func_5060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), other_module_to_import_mixed_5059, 'my_func')
# Assigning a type to the variable 'r4' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r4', my_func_5060)

# Assigning a Attribute to a Name (line 11):
# Getting the type of 'other_module_to_import_mixed' (line 11)
other_module_to_import_mixed_5061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'other_module_to_import_mixed')
# Obtaining the member 'secondary_module_mixed' of a type (line 11)
secondary_module_mixed_5062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), other_module_to_import_mixed_5061, 'secondary_module_mixed')
# Assigning a type to the variable 'r5' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r5', secondary_module_mixed_5062)

# Assigning a Attribute to a Name (line 12):
# Getting the type of 'r5' (line 12)
r5_5063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'r5')
# Obtaining the member 'number' of a type (line 12)
number_5064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), r5_5063, 'number')
# Assigning a type to the variable 'r6' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r6', number_5064)

# Assigning a Attribute to a Name (line 15):
# Getting the type of 'other_module_to_import_mixed' (line 15)
other_module_to_import_mixed_5065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'other_module_to_import_mixed')
# Obtaining the member 'my_func2' of a type (line 15)
my_func2_5066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), other_module_to_import_mixed_5065, 'my_func2')
# Assigning a type to the variable 'r7' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r7', my_func2_5066)

# Assigning a Call to a Name (line 16):

# Call to r7(...): (line 16)
# Processing the call arguments (line 16)
int_5068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 9), 'int')
# Processing the call keyword arguments (line 16)
kwargs_5069 = {}
# Getting the type of 'r7' (line 16)
r7_5067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 6), 'r7', False)
# Calling r7(args, kwargs) (line 16)
r7_call_result_5070 = invoke(stypy.reporting.localization.Localization(__file__, 16, 6), r7_5067, *[int_5068], **kwargs_5069)

# Assigning a type to the variable 'r7b' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r7b', r7_call_result_5070)

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'other_module_to_import_mixed' (line 17)
other_module_to_import_mixed_5071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'other_module_to_import_mixed')
# Obtaining the member 'my_func3' of a type (line 17)
my_func3_5072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), other_module_to_import_mixed_5071, 'my_func3')
# Assigning a type to the variable 'r8' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r8', my_func3_5072)

# Assigning a Call to a Name (line 18):

# Call to r8(...): (line 18)
# Processing the call keyword arguments (line 18)
kwargs_5074 = {}
# Getting the type of 'r8' (line 18)
r8_5073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 6), 'r8', False)
# Calling r8(args, kwargs) (line 18)
r8_call_result_5075 = invoke(stypy.reporting.localization.Localization(__file__, 18, 6), r8_5073, *[], **kwargs_5074)

# Assigning a type to the variable 'r8b' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r8b', r8_call_result_5075)

# Assigning a Attribute to a Name (line 19):
# Getting the type of 'other_module_to_import_mixed' (line 19)
other_module_to_import_mixed_5076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'other_module_to_import_mixed')
# Obtaining the member 'secondary_module_mixed' of a type (line 19)
secondary_module_mixed_5077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), other_module_to_import_mixed_5076, 'secondary_module_mixed')
# Assigning a type to the variable 'r9' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r9', secondary_module_mixed_5077)

# Assigning a Call to a Name (line 20):

# Call to clock(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_5081 = {}
# Getting the type of 'r9' (line 20)
r9_5078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 6), 'r9', False)
# Obtaining the member 'time' of a type (line 20)
time_5079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 6), r9_5078, 'time')
# Obtaining the member 'clock' of a type (line 20)
clock_5080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 6), time_5079, 'clock')
# Calling clock(args, kwargs) (line 20)
clock_call_result_5082 = invoke(stypy.reporting.localization.Localization(__file__, 20, 6), clock_5080, *[], **kwargs_5081)

# Assigning a type to the variable 'r10' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'r10', clock_call_result_5082)

# Assigning a Attribute to a Name (line 21):
# Getting the type of 'other_module_to_import_mixed' (line 21)
other_module_to_import_mixed_5083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 6), 'other_module_to_import_mixed')
# Obtaining the member 'secondary_module_mixed' of a type (line 21)
secondary_module_mixed_5084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 6), other_module_to_import_mixed_5083, 'secondary_module_mixed')
# Obtaining the member 'time' of a type (line 21)
time_5085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 6), secondary_module_mixed_5084, 'time')
# Assigning a type to the variable 'r11' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r11', time_5085)

# Assigning a Attribute to a Name (line 22):
# Getting the type of 'other_module_to_import_mixed' (line 22)
other_module_to_import_mixed_5086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 6), 'other_module_to_import_mixed')
# Obtaining the member 'secondary_module_mixed' of a type (line 22)
secondary_module_mixed_5087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 6), other_module_to_import_mixed_5086, 'secondary_module_mixed')
# Obtaining the member 'clock_func' of a type (line 22)
clock_func_5088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 6), secondary_module_mixed_5087, 'clock_func')
# Assigning a type to the variable 'r12' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r12', clock_func_5088)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
