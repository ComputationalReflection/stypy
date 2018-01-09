
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.randint(0, 10, (3, 3))
6: 
7: r = (Z[Z[:, 1].argsort()])
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
import_37 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_37) is not StypyTypeError):

    if (import_37 != 'pyd_module'):
        __import__(import_37)
        sys_modules_38 = sys.modules[import_37]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_38.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_37)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to randint(...): (line 5)
# Processing the call arguments (line 5)
int_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 30), tuple_44, int_45)
# Adding element type (line 5)
int_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 30), tuple_44, int_46)

# Processing the call keyword arguments (line 5)
kwargs_47 = {}
# Getting the type of 'np' (line 5)
np_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_39, 'random')
# Obtaining the member 'randint' of a type (line 5)
randint_41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_40, 'randint')
# Calling randint(args, kwargs) (line 5)
randint_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), randint_41, *[int_42, int_43, tuple_44], **kwargs_47)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', randint_call_result_48)

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript

# Call to argsort(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_55 = {}

# Obtaining the type of the subscript
slice_49 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 7, 7), None, None, None)
int_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'int')
# Getting the type of 'Z' (line 7)
Z_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 7), 'Z', False)
# Obtaining the member '__getitem__' of a type (line 7)
getitem___52 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 7), Z_51, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 7, 7), getitem___52, (slice_49, int_50))

# Obtaining the member 'argsort' of a type (line 7)
argsort_54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 7), subscript_call_result_53, 'argsort')
# Calling argsort(args, kwargs) (line 7)
argsort_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 7, 7), argsort_54, *[], **kwargs_55)

# Getting the type of 'Z' (line 7)
Z_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'Z')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___58 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), Z_57, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_59 = invoke(stypy.reporting.localization.Localization(__file__, 7, 5), getitem___58, argsort_call_result_56)

# Assigning a type to the variable 'r' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r', subscript_call_result_59)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
