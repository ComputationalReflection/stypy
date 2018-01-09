
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.uniform(0, 1, 10)
6: z = 0.5
7: m = Z.flat[np.abs(Z - z).argmin()]
8: 
9: # l = globals().copy()
10: # for v in l:
11: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_572 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_572) is not StypyTypeError):

    if (import_572 != 'pyd_module'):
        __import__(import_572)
        sys_modules_573 = sys.modules[import_572]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_573.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_572)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to uniform(...): (line 5)
# Processing the call arguments (line 5)
int_577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
int_578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')
int_579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 28), 'int')
# Processing the call keyword arguments (line 5)
kwargs_580 = {}
# Getting the type of 'np' (line 5)
np_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_574, 'random')
# Obtaining the member 'uniform' of a type (line 5)
uniform_576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_575, 'uniform')
# Calling uniform(args, kwargs) (line 5)
uniform_call_result_581 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), uniform_576, *[int_577, int_578, int_579], **kwargs_580)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', uniform_call_result_581)

# Assigning a Num to a Name (line 6):
float_582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'float')
# Assigning a type to the variable 'z' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'z', float_582)

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript

# Call to argmin(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_591 = {}

# Call to abs(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'Z' (line 7)
Z_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), 'Z', False)
# Getting the type of 'z' (line 7)
z_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 22), 'z', False)
# Applying the binary operator '-' (line 7)
result_sub_587 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 18), '-', Z_585, z_586)

# Processing the call keyword arguments (line 7)
kwargs_588 = {}
# Getting the type of 'np' (line 7)
np_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'np', False)
# Obtaining the member 'abs' of a type (line 7)
abs_584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 11), np_583, 'abs')
# Calling abs(args, kwargs) (line 7)
abs_call_result_589 = invoke(stypy.reporting.localization.Localization(__file__, 7, 11), abs_584, *[result_sub_587], **kwargs_588)

# Obtaining the member 'argmin' of a type (line 7)
argmin_590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 11), abs_call_result_589, 'argmin')
# Calling argmin(args, kwargs) (line 7)
argmin_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 7, 11), argmin_590, *[], **kwargs_591)

# Getting the type of 'Z' (line 7)
Z_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'Z')
# Obtaining the member 'flat' of a type (line 7)
flat_594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), Z_593, 'flat')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), flat_594, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), getitem___595, argmin_call_result_592)

# Assigning a type to the variable 'm' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'm', subscript_call_result_596)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
