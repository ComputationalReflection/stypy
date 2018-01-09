
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: _mapper = [(1, 4, 'a'), (2, 5, 'b'), (3, 6, 'c')]
3: 
4: 
5: (_defaulttype, _defaultfunc, _defaultfill) = zip(*_mapper)
6: 
7: r1 = _defaulttype
8: r2 = _defaultfunc
9: r3 = _defaultfill
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 2):

# Assigning a List to a Name (line 2):

# Obtaining an instance of the builtin type 'list' (line 2)
list_591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
int_593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 12), tuple_592, int_593)
# Adding element type (line 2)
int_594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 12), tuple_592, int_594)
# Adding element type (line 2)
str_595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 18), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 12), tuple_592, str_595)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_591, tuple_592)
# Adding element type (line 2)

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
int_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 25), tuple_596, int_597)
# Adding element type (line 2)
int_598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 25), tuple_596, int_598)
# Adding element type (line 2)
str_599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 31), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 25), tuple_596, str_599)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_591, tuple_596)
# Adding element type (line 2)

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
int_601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 38), tuple_600, int_601)
# Adding element type (line 2)
int_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 38), tuple_600, int_602)
# Adding element type (line 2)
str_603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 44), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 38), tuple_600, str_603)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_591, tuple_600)

# Assigning a type to the variable '_mapper' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '_mapper', list_591)

# Assigning a Call to a Tuple (line 5):

# Assigning a Subscript to a Name (line 5):

# Obtaining the type of the subscript
int_604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'int')

# Call to zip(...): (line 5)
# Getting the type of '_mapper' (line 5)
_mapper_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 50), '_mapper', False)
# Processing the call keyword arguments (line 5)
kwargs_607 = {}
# Getting the type of 'zip' (line 5)
zip_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 45), 'zip', False)
# Calling zip(args, kwargs) (line 5)
zip_call_result_608 = invoke(stypy.reporting.localization.Localization(__file__, 5, 45), zip_605, *[_mapper_606], **kwargs_607)

# Obtaining the member '__getitem__' of a type (line 5)
getitem___609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 0), zip_call_result_608, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 5, 0), getitem___609, int_604)

# Assigning a type to the variable 'tuple_var_assignment_588' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_var_assignment_588', subscript_call_result_610)

# Assigning a Subscript to a Name (line 5):

# Obtaining the type of the subscript
int_611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'int')

# Call to zip(...): (line 5)
# Getting the type of '_mapper' (line 5)
_mapper_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 50), '_mapper', False)
# Processing the call keyword arguments (line 5)
kwargs_614 = {}
# Getting the type of 'zip' (line 5)
zip_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 45), 'zip', False)
# Calling zip(args, kwargs) (line 5)
zip_call_result_615 = invoke(stypy.reporting.localization.Localization(__file__, 5, 45), zip_612, *[_mapper_613], **kwargs_614)

# Obtaining the member '__getitem__' of a type (line 5)
getitem___616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 0), zip_call_result_615, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_617 = invoke(stypy.reporting.localization.Localization(__file__, 5, 0), getitem___616, int_611)

# Assigning a type to the variable 'tuple_var_assignment_589' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_var_assignment_589', subscript_call_result_617)

# Assigning a Subscript to a Name (line 5):

# Obtaining the type of the subscript
int_618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'int')

# Call to zip(...): (line 5)
# Getting the type of '_mapper' (line 5)
_mapper_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 50), '_mapper', False)
# Processing the call keyword arguments (line 5)
kwargs_621 = {}
# Getting the type of 'zip' (line 5)
zip_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 45), 'zip', False)
# Calling zip(args, kwargs) (line 5)
zip_call_result_622 = invoke(stypy.reporting.localization.Localization(__file__, 5, 45), zip_619, *[_mapper_620], **kwargs_621)

# Obtaining the member '__getitem__' of a type (line 5)
getitem___623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 0), zip_call_result_622, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_624 = invoke(stypy.reporting.localization.Localization(__file__, 5, 0), getitem___623, int_618)

# Assigning a type to the variable 'tuple_var_assignment_590' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_var_assignment_590', subscript_call_result_624)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_var_assignment_588' (line 5)
tuple_var_assignment_588_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_var_assignment_588')
# Assigning a type to the variable '_defaulttype' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 1), '_defaulttype', tuple_var_assignment_588_625)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_var_assignment_589' (line 5)
tuple_var_assignment_589_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_var_assignment_589')
# Assigning a type to the variable '_defaultfunc' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 15), '_defaultfunc', tuple_var_assignment_589_626)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_var_assignment_590' (line 5)
tuple_var_assignment_590_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_var_assignment_590')
# Assigning a type to the variable '_defaultfill' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 29), '_defaultfill', tuple_var_assignment_590_627)

# Assigning a Name to a Name (line 7):

# Assigning a Name to a Name (line 7):
# Getting the type of '_defaulttype' (line 7)
_defaulttype_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), '_defaulttype')
# Assigning a type to the variable 'r1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r1', _defaulttype_628)

# Assigning a Name to a Name (line 8):

# Assigning a Name to a Name (line 8):
# Getting the type of '_defaultfunc' (line 8)
_defaultfunc_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), '_defaultfunc')
# Assigning a type to the variable 'r2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r2', _defaultfunc_629)

# Assigning a Name to a Name (line 9):

# Assigning a Name to a Name (line 9):
# Getting the type of '_defaultfill' (line 9)
_defaultfill_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), '_defaultfill')
# Assigning a type to the variable 'r3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r3', _defaultfill_630)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
