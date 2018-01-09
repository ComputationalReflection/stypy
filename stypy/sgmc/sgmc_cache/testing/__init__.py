
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: # -*- coding: utf-8 -*-
3: __all__ = ['model_testing', 'code_generation_testing', 'testing_parameters']

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 3):
__all__ = ['model_testing', 'code_generation_testing', 'testing_parameters']
module_type_store.set_exportable_members(['model_testing', 'code_generation_testing', 'testing_parameters'])

# Obtaining an instance of the builtin type 'list' (line 3)
list_4960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
str_4961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'model_testing')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_4960, str_4961)
# Adding element type (line 3)
str_4962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 28), 'str', 'code_generation_testing')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_4960, str_4962)
# Adding element type (line 3)
str_4963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 55), 'str', 'testing_parameters')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_4960, str_4963)

# Assigning a type to the variable '__all__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__all__', list_4960)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
