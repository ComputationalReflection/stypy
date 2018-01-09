
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: c = 4, 5
4: 
5: a, b = 5, 3
6: 
7: (x,y)=(z,w)=(4,5)

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Tuple to a Name (line 3):

# Assigning a Tuple to a Name (line 3):

# Obtaining an instance of the builtin type 'tuple' (line 3)
tuple_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3)
# Adding element type (line 3)
int_574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 4), tuple_573, int_574)
# Adding element type (line 3)
int_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 4), tuple_573, int_575)

# Assigning a type to the variable 'c' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'c', tuple_573)

# Assigning a Tuple to a Tuple (line 5):

# Assigning a Num to a Name (line 5):
int_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 7), 'int')
# Assigning a type to the variable 'tuple_assignment_567' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_567', int_576)

# Assigning a Num to a Name (line 5):
int_577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 10), 'int')
# Assigning a type to the variable 'tuple_assignment_568' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_568', int_577)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_assignment_567' (line 5)
tuple_assignment_567_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_567')
# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', tuple_assignment_567_578)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_assignment_568' (line 5)
tuple_assignment_568_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_568')
# Assigning a type to the variable 'b' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 3), 'b', tuple_assignment_568_579)

# Multiple assignment of 2 elements.

# Assigning a Num to a Name (line 7):
int_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'int')
# Assigning a type to the variable 'tuple_assignment_569' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_569', int_580)

# Assigning a Num to a Name (line 7):
int_581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
# Assigning a type to the variable 'tuple_assignment_570' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_570', int_581)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_assignment_569' (line 7)
tuple_assignment_569_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_569')
# Assigning a type to the variable 'z' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 7), 'z', tuple_assignment_569_582)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_assignment_570' (line 7)
tuple_assignment_570_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_570')
# Assigning a type to the variable 'w' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'w', tuple_assignment_570_583)

# Assigning a Name to a Name (line 7):
# Getting the type of 'z' (line 7)
z_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 7), 'z')
# Assigning a type to the variable 'tuple_assignment_571' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_571', z_584)

# Assigning a Name to a Name (line 7):
# Getting the type of 'w' (line 7)
w_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'w')
# Assigning a type to the variable 'tuple_assignment_572' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_572', w_585)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_assignment_571' (line 7)
tuple_assignment_571_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_571')
# Assigning a type to the variable 'x' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 1), 'x', tuple_assignment_571_586)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_assignment_572' (line 7)
tuple_assignment_572_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_572')
# Assigning a type to the variable 'y' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 3), 'y', tuple_assignment_572_587)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
