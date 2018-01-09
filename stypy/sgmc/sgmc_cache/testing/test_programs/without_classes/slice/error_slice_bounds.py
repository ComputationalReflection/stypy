
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: Array1Glob = [0] * 51
2: 
3: sl = Array1Glob["a":]  # Unreported. slice creation indexes are not checked
4: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a BinOp to a Name (line 1):

# Obtaining an instance of the builtin type 'list' (line 1)
list_8002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
int_8003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 13), list_8002, int_8003)

int_8004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 19), 'int')
# Applying the binary operator '*' (line 1)
result_mul_8005 = python_operator(stypy.reporting.localization.Localization(__file__, 1, 13), '*', list_8002, int_8004)

# Assigning a type to the variable 'Array1Glob' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'Array1Glob', result_mul_8005)

# Assigning a Subscript to a Name (line 3):

# Obtaining the type of the subscript
str_8006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 16), 'str', 'a')
slice_8007 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 3, 5), str_8006, None, None)
# Getting the type of 'Array1Glob' (line 3)
Array1Glob_8008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'Array1Glob')
# Obtaining the member '__getitem__' of a type (line 3)
getitem___8009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 5), Array1Glob_8008, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 3)
subscript_call_result_8010 = invoke(stypy.reporting.localization.Localization(__file__, 3, 5), getitem___8009, slice_8007)

# Assigning a type to the variable 'sl' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'sl', subscript_call_result_8010)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
