
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.random((3, 3, 3))
6: 
7: Z2 = np.random.random((10, 10))
8: Zmin, Zmax = Z2.min(), Z2.max()
9: 
10: # l = globals().copy()
11: # for v in l:
12: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2145 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2145) is not StypyTypeError):

    if (import_2145 != 'pyd_module'):
        __import__(import_2145)
        sys_modules_2146 = sys.modules[import_2145]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2146.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2145)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Assigning a Call to a Name (line 5):

# Call to random(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_2150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_2151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), tuple_2150, int_2151)
# Adding element type (line 5)
int_2152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), tuple_2150, int_2152)
# Adding element type (line 5)
int_2153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), tuple_2150, int_2153)

# Processing the call keyword arguments (line 5)
kwargs_2154 = {}
# Getting the type of 'np' (line 5)
np_2147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_2148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_2147, 'random')
# Obtaining the member 'random' of a type (line 5)
random_2149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_2148, 'random')
# Calling random(args, kwargs) (line 5)
random_call_result_2155 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), random_2149, *[tuple_2150], **kwargs_2154)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', random_call_result_2155)

# Assigning a Call to a Name (line 7):

# Assigning a Call to a Name (line 7):

# Call to random(...): (line 7)
# Processing the call arguments (line 7)

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_2159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
int_2160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 23), tuple_2159, int_2160)
# Adding element type (line 7)
int_2161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 23), tuple_2159, int_2161)

# Processing the call keyword arguments (line 7)
kwargs_2162 = {}
# Getting the type of 'np' (line 7)
np_2156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'np', False)
# Obtaining the member 'random' of a type (line 7)
random_2157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), np_2156, 'random')
# Obtaining the member 'random' of a type (line 7)
random_2158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), random_2157, 'random')
# Calling random(args, kwargs) (line 7)
random_call_result_2163 = invoke(stypy.reporting.localization.Localization(__file__, 7, 5), random_2158, *[tuple_2159], **kwargs_2162)

# Assigning a type to the variable 'Z2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'Z2', random_call_result_2163)

# Assigning a Tuple to a Tuple (line 8):

# Assigning a Call to a Name (line 8):

# Call to min(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_2166 = {}
# Getting the type of 'Z2' (line 8)
Z2_2164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'Z2', False)
# Obtaining the member 'min' of a type (line 8)
min_2165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 13), Z2_2164, 'min')
# Calling min(args, kwargs) (line 8)
min_call_result_2167 = invoke(stypy.reporting.localization.Localization(__file__, 8, 13), min_2165, *[], **kwargs_2166)

# Assigning a type to the variable 'tuple_assignment_2143' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'tuple_assignment_2143', min_call_result_2167)

# Assigning a Call to a Name (line 8):

# Call to max(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_2170 = {}
# Getting the type of 'Z2' (line 8)
Z2_2168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 23), 'Z2', False)
# Obtaining the member 'max' of a type (line 8)
max_2169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 23), Z2_2168, 'max')
# Calling max(args, kwargs) (line 8)
max_call_result_2171 = invoke(stypy.reporting.localization.Localization(__file__, 8, 23), max_2169, *[], **kwargs_2170)

# Assigning a type to the variable 'tuple_assignment_2144' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'tuple_assignment_2144', max_call_result_2171)

# Assigning a Name to a Name (line 8):
# Getting the type of 'tuple_assignment_2143' (line 8)
tuple_assignment_2143_2172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'tuple_assignment_2143')
# Assigning a type to the variable 'Zmin' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'Zmin', tuple_assignment_2143_2172)

# Assigning a Name to a Name (line 8):
# Getting the type of 'tuple_assignment_2144' (line 8)
tuple_assignment_2144_2173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'tuple_assignment_2144')
# Assigning a type to the variable 'Zmax' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 6), 'Zmax', tuple_assignment_2144_2173)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
