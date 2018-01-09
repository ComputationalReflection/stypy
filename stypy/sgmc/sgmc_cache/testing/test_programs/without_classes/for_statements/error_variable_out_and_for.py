
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: acum = "3"
2: 
3: for i in range(4):
4:     acum = i
5: 
6: print acum[4]  # Not reported
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 1):
str_8085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 7), 'str', '3')
# Assigning a type to the variable 'acum' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'acum', str_8085)


# Call to range(...): (line 3)
# Processing the call arguments (line 3)
int_8087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'int')
# Processing the call keyword arguments (line 3)
kwargs_8088 = {}
# Getting the type of 'range' (line 3)
range_8086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 9), 'range', False)
# Calling range(args, kwargs) (line 3)
range_call_result_8089 = invoke(stypy.reporting.localization.Localization(__file__, 3, 9), range_8086, *[int_8087], **kwargs_8088)

# Testing the type of a for loop iterable (line 3)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 3, 0), range_call_result_8089)
# Getting the type of the for loop variable (line 3)
for_loop_var_8090 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 3, 0), range_call_result_8089)
# Assigning a type to the variable 'i' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'i', for_loop_var_8090)
# SSA begins for a for statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Name to a Name (line 4):
# Getting the type of 'i' (line 4)
i_8091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 11), 'i')
# Assigning a type to the variable 'acum' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'acum', i_8091)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Obtaining the type of the subscript
int_8092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'int')
# Getting the type of 'acum' (line 6)
acum_8093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 6), 'acum')
# Obtaining the member '__getitem__' of a type (line 6)
getitem___8094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 6), acum_8093, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_8095 = invoke(stypy.reporting.localization.Localization(__file__, 6, 6), getitem___8094, int_8092)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
