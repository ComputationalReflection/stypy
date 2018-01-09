
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: 
4: Array1Glob = [0] * 51
5: 
6: sl = Array1Glob[:]

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a BinOp to a Name (line 4):

# Obtaining an instance of the builtin type 'list' (line 4)
list_6409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 4)
# Adding element type (line 4)
int_6410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 13), list_6409, int_6410)

int_6411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 19), 'int')
# Applying the binary operator '*' (line 4)
result_mul_6412 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 13), '*', list_6409, int_6411)

# Assigning a type to the variable 'Array1Glob' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'Array1Glob', result_mul_6412)

# Assigning a Subscript to a Name (line 6):

# Obtaining the type of the subscript
slice_6413 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 6, 5), None, None, None)
# Getting the type of 'Array1Glob' (line 6)
Array1Glob_6414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'Array1Glob')
# Obtaining the member '__getitem__' of a type (line 6)
getitem___6415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), Array1Glob_6414, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_6416 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), getitem___6415, slice_6413)

# Assigning a type to the variable 'sl' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'sl', subscript_call_result_6416)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
