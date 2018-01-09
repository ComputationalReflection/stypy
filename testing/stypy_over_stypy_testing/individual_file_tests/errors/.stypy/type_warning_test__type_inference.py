
"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy.errors_copy.type_warning_copy import TypeWarning
2: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization
3: 
4: warn1 = TypeWarning(Localization("foo_file.py", 1, 1), "foo")
5: warn2 = TypeWarning.instance(Localization("foo_file.py", 1, 1), "foo")
6: # #warn3 = TypeWarning(None, "foo")
7: #
8: # r1 = warn1.print_warning_msgs()
9: # r2 = warn1.reset_warning_msgs()
10: # r3 = warn1.get_warning_msgs()

"""

# Import the stypy library
from stypy import *

# Create the module type store
type_store = TypeStore(__file__)

################## Begin of the type inference program ##################

# Importing from 'stypy_copy.errors_copy.type_warning_copy' module (line 1)
import_from_module(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 1, 0), 'stypy_copy.errors_copy.type_warning_copy', type_store, *['TypeWarning'])

# Importing from 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy' module (line 2)
import_from_module(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 2, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy', type_store, *['Localization'])


# Assignment to a Name from a Call

# Calling 'TypeWarning' (line 4)
# Processing call arguments (line 4)

# Calling 'Localization' (line 4)
# Processing call arguments (line 4)
_stypy_temp_3 = get_builtin_type(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 4, 33), 'str', 'foo_file.py')
_stypy_temp_4 = get_builtin_type(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 4, 48), 'int', 1)
_stypy_temp_5 = get_builtin_type(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 4, 51), 'int', 1)
# Processing call keyword arguments (line 4)
_stypy_temp_6 = {}
# Getting the type of 'Localization' (line 4)
_stypy_temp_2 = type_store.get_type_of(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 4, 20), 'Localization')
# Performing the call (line 4)
_stypy_temp_7 = _stypy_temp_2.invoke(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 4, 20), *[_stypy_temp_3, _stypy_temp_4, _stypy_temp_5], **_stypy_temp_6)

_stypy_temp_8 = get_builtin_type(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 4, 55), 'str', 'foo')
# Processing call keyword arguments (line 4)
_stypy_temp_9 = {}
# Getting the type of 'TypeWarning' (line 4)
_stypy_temp_1 = type_store.get_type_of(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 4, 8), 'TypeWarning')
# Performing the call (line 4)
_stypy_temp_10 = _stypy_temp_1.invoke(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 4, 8), *[_stypy_temp_7, _stypy_temp_8], **_stypy_temp_9)

# Type assignment (line 4)
type_store.set_type_of(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 4, 0), 'warn1', _stypy_temp_10)

# Assignment to a Name from a Call

# Calling 'instance' (line 5)
# Processing call arguments (line 5)

# Calling 'Localization' (line 5)
# Processing call arguments (line 5)
_stypy_temp_14 = get_builtin_type(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 5, 42), 'str', 'foo_file.py')
_stypy_temp_15 = get_builtin_type(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 5, 57), 'int', 1)
_stypy_temp_16 = get_builtin_type(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 5, 60), 'int', 1)
# Processing call keyword arguments (line 5)
_stypy_temp_17 = {}
# Getting the type of 'Localization' (line 5)
_stypy_temp_13 = type_store.get_type_of(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 5, 29), 'Localization')
# Performing the call (line 5)
_stypy_temp_18 = _stypy_temp_13.invoke(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 5, 29), *[_stypy_temp_14, _stypy_temp_15, _stypy_temp_16], **_stypy_temp_17)

_stypy_temp_19 = get_builtin_type(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 5, 64), 'str', 'foo')
# Processing call keyword arguments (line 5)
_stypy_temp_20 = {}
# Getting the type of 'TypeWarning' (line 5)
_stypy_temp_11 = type_store.get_type_of(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 5, 8), 'TypeWarning')
# Obtaining the member 'instance' of a type (line 5)
_stypy_temp_12 = _stypy_temp_11.get_type_of_member(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 5, 8), 'instance')
# Performing the call (line 5)
_stypy_temp_21 = _stypy_temp_12.invoke(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 5, 8), *[_stypy_temp_18, _stypy_temp_19], **_stypy_temp_20)

# Type assignment (line 5)
type_store.set_type_of(stypy.python_lib.python_types.type_inference.localization.Localization(__file__, 5, 0), 'warn2', _stypy_temp_21)

################## End of the type inference program ##################

module_errors = stypy.errors.type_error.TypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
