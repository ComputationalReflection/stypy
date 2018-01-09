
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Contains information about the dateutil version.
3: '''
4: 
5: VERSION_MAJOR = 2
6: VERSION_MINOR = 6
7: VERSION_PATCH = 1
8: 
9: VERSION_TUPLE = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)
10: VERSION = '.'.join(map(str, VERSION_TUPLE))
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_320176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nContains information about the dateutil version.\n')

# Assigning a Num to a Name (line 5):
int_320177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'int')
# Assigning a type to the variable 'VERSION_MAJOR' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'VERSION_MAJOR', int_320177)

# Assigning a Num to a Name (line 6):
int_320178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 16), 'int')
# Assigning a type to the variable 'VERSION_MINOR' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'VERSION_MINOR', int_320178)

# Assigning a Num to a Name (line 7):
int_320179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 16), 'int')
# Assigning a type to the variable 'VERSION_PATCH' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'VERSION_PATCH', int_320179)

# Assigning a Tuple to a Name (line 9):

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_320180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
# Getting the type of 'VERSION_MAJOR' (line 9)
VERSION_MAJOR_320181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 17), 'VERSION_MAJOR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), tuple_320180, VERSION_MAJOR_320181)
# Adding element type (line 9)
# Getting the type of 'VERSION_MINOR' (line 9)
VERSION_MINOR_320182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 32), 'VERSION_MINOR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), tuple_320180, VERSION_MINOR_320182)
# Adding element type (line 9)
# Getting the type of 'VERSION_PATCH' (line 9)
VERSION_PATCH_320183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 47), 'VERSION_PATCH')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), tuple_320180, VERSION_PATCH_320183)

# Assigning a type to the variable 'VERSION_TUPLE' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'VERSION_TUPLE', tuple_320180)

# Assigning a Call to a Name (line 10):

# Call to join(...): (line 10)
# Processing the call arguments (line 10)

# Call to map(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'str' (line 10)
str_320187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 23), 'str', False)
# Getting the type of 'VERSION_TUPLE' (line 10)
VERSION_TUPLE_320188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 28), 'VERSION_TUPLE', False)
# Processing the call keyword arguments (line 10)
kwargs_320189 = {}
# Getting the type of 'map' (line 10)
map_320186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 19), 'map', False)
# Calling map(args, kwargs) (line 10)
map_call_result_320190 = invoke(stypy.reporting.localization.Localization(__file__, 10, 19), map_320186, *[str_320187, VERSION_TUPLE_320188], **kwargs_320189)

# Processing the call keyword arguments (line 10)
kwargs_320191 = {}
str_320184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'str', '.')
# Obtaining the member 'join' of a type (line 10)
join_320185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 10), str_320184, 'join')
# Calling join(args, kwargs) (line 10)
join_call_result_320192 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), join_320185, *[map_call_result_320190], **kwargs_320191)

# Assigning a type to the variable 'VERSION' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'VERSION', join_call_result_320192)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
