
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #https://www.python.org/dev/peps/pep-0289/
2: 
3: r = sum(x*x for x in range(10))
4: 
5: print r
6: 
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Call to a Name (line 3):

# Call to sum(...): (line 3)
# Processing the call arguments (line 3)
# Calculating generator expression
module_type_store = module_type_store.open_function_context('list comprehension expression', 3, 8, True)
# Calculating comprehension expression

# Call to range(...): (line 3)
# Processing the call arguments (line 3)
int_2767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 27), 'int')
# Processing the call keyword arguments (line 3)
kwargs_2768 = {}
# Getting the type of 'range' (line 3)
range_2766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 21), 'range', False)
# Calling range(args, kwargs) (line 3)
range_call_result_2769 = invoke(stypy.reporting.localization.Localization(__file__, 3, 21), range_2766, *[int_2767], **kwargs_2768)

comprehension_2770 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), range_call_result_2769)
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'x', comprehension_2770)
# Getting the type of 'x' (line 3)
x_2763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'x', False)
# Getting the type of 'x' (line 3)
x_2764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 10), 'x', False)
# Applying the binary operator '*' (line 3)
result_mul_2765 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 8), '*', x_2763, x_2764)

list_2771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'list')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 8), list_2771, result_mul_2765)
# Processing the call keyword arguments (line 3)
kwargs_2772 = {}
# Getting the type of 'sum' (line 3)
sum_2762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'sum', False)
# Calling sum(args, kwargs) (line 3)
sum_call_result_2773 = invoke(stypy.reporting.localization.Localization(__file__, 3, 4), sum_2762, *[list_2771], **kwargs_2772)

# Assigning a type to the variable 'r' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'r', sum_call_result_2773)
# Getting the type of 'r' (line 5)
r_2774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 6), 'r')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
