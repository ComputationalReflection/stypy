
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: from math import cos as coseno
4: 
5: r = coseno(30)
6: 
7: 
8: import math as matematicas
9: 
10: r2 = matematicas.acos(3)
11: 
12: import module_to_import as mimodulo
13: 
14: r3 = mimodulo.global_a
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from math import coseno' statement (line 3)
try:
    from math import cos as coseno

except:
    coseno = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'math', None, module_type_store, ['cos'], [coseno])
# Adding an alias
module_type_store.add_alias('coseno', 'cos')


# Assigning a Call to a Name (line 5):

# Call to coseno(...): (line 5)
# Processing the call arguments (line 5)
int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 11), 'int')
# Processing the call keyword arguments (line 5)
kwargs_3 = {}
# Getting the type of 'coseno' (line 5)
coseno_1 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'coseno', False)
# Calling coseno(args, kwargs) (line 5)
coseno_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), coseno_1, *[int_2], **kwargs_3)

# Assigning a type to the variable 'r' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r', coseno_call_result_4)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import math' statement (line 8)
import math as matematicas

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matematicas', matematicas, module_type_store)


# Assigning a Call to a Name (line 10):

# Call to acos(...): (line 10)
# Processing the call arguments (line 10)
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 22), 'int')
# Processing the call keyword arguments (line 10)
kwargs_8 = {}
# Getting the type of 'matematicas' (line 10)
matematicas_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'matematicas', False)
# Obtaining the member 'acos' of a type (line 10)
acos_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), matematicas_5, 'acos')
# Calling acos(args, kwargs) (line 10)
acos_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), acos_6, *[int_7], **kwargs_8)

# Assigning a type to the variable 'r2' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r2', acos_call_result_9)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import module_to_import' statement (line 12)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/without_classes/imports/')
import_10 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'module_to_import')

if (type(import_10) is not StypyTypeError):

    if (import_10 != 'pyd_module'):
        __import__(import_10)
        sys_modules_11 = sys.modules[import_10]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'mimodulo', sys_modules_11.module_type_store, module_type_store)
    else:
        import module_to_import as mimodulo

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'mimodulo', mimodulo, module_type_store)

else:
    # Assigning a type to the variable 'module_to_import' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'module_to_import', import_10)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/without_classes/imports/')


# Assigning a Attribute to a Name (line 14):
# Getting the type of 'mimodulo' (line 14)
mimodulo_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'mimodulo')
# Obtaining the member 'global_a' of a type (line 14)
global_a_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), mimodulo_12, 'global_a')
# Assigning a type to the variable 'r3' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r3', global_a_13)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
