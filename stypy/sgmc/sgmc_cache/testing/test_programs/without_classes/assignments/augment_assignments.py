
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: s = ""
3: s += str(3)
4: s += str(5)
5: s += str(7)
6: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 4), 'str', '')
# Assigning a type to the variable 's' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 's', str_91)

# Getting the type of 's' (line 3)
s_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 's')

# Call to str(...): (line 3)
# Processing the call arguments (line 3)
int_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 9), 'int')
# Processing the call keyword arguments (line 3)
kwargs_95 = {}
# Getting the type of 'str' (line 3)
str_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'str', False)
# Calling str(args, kwargs) (line 3)
str_call_result_96 = invoke(stypy.reporting.localization.Localization(__file__, 3, 5), str_93, *[int_94], **kwargs_95)

# Applying the binary operator '+=' (line 3)
result_iadd_97 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 0), '+=', s_92, str_call_result_96)
# Assigning a type to the variable 's' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 's', result_iadd_97)


# Getting the type of 's' (line 4)
s_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 's')

# Call to str(...): (line 4)
# Processing the call arguments (line 4)
int_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 9), 'int')
# Processing the call keyword arguments (line 4)
kwargs_101 = {}
# Getting the type of 'str' (line 4)
str_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 5), 'str', False)
# Calling str(args, kwargs) (line 4)
str_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 4, 5), str_99, *[int_100], **kwargs_101)

# Applying the binary operator '+=' (line 4)
result_iadd_103 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 0), '+=', s_98, str_call_result_102)
# Assigning a type to the variable 's' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 's', result_iadd_103)


# Getting the type of 's' (line 5)
s_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 's')

# Call to str(...): (line 5)
# Processing the call arguments (line 5)
int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 9), 'int')
# Processing the call keyword arguments (line 5)
kwargs_107 = {}
# Getting the type of 'str' (line 5)
str_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'str', False)
# Calling str(args, kwargs) (line 5)
str_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 5, 5), str_105, *[int_106], **kwargs_107)

# Applying the binary operator '+=' (line 5)
result_iadd_109 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 0), '+=', s_104, str_call_result_108)
# Assigning a type to the variable 's' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 's', result_iadd_109)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
