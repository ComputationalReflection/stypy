
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: __version__ = "2"
3: loops = 100
4: benchtime = 1000
5: stones = 10
6: 
7: print "Pystone(%s) time for %d passes = %g" % (__version__, loops, benchtime)
8: print "This machine benchmarks at %g pystones/second" % stones

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_6377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 14), 'str', '2')
# Assigning a type to the variable '__version__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__version__', str_6377)

# Assigning a Num to a Name (line 3):
int_6378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'int')
# Assigning a type to the variable 'loops' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'loops', int_6378)

# Assigning a Num to a Name (line 4):
int_6379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Assigning a type to the variable 'benchtime' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'benchtime', int_6379)

# Assigning a Num to a Name (line 5):
int_6380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 9), 'int')
# Assigning a type to the variable 'stones' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stones', int_6380)
str_6381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 6), 'str', 'Pystone(%s) time for %d passes = %g')

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_6382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 47), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
# Getting the type of '__version__' (line 7)
version___6383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 47), '__version__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), tuple_6382, version___6383)
# Adding element type (line 7)
# Getting the type of 'loops' (line 7)
loops_6384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 60), 'loops')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), tuple_6382, loops_6384)
# Adding element type (line 7)
# Getting the type of 'benchtime' (line 7)
benchtime_6385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 67), 'benchtime')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), tuple_6382, benchtime_6385)

# Applying the binary operator '%' (line 7)
result_mod_6386 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 6), '%', str_6381, tuple_6382)

str_6387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 6), 'str', 'This machine benchmarks at %g pystones/second')
# Getting the type of 'stones' (line 8)
stones_6388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 56), 'stones')
# Applying the binary operator '%' (line 8)
result_mod_6389 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 6), '%', str_6387, stones_6388)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
