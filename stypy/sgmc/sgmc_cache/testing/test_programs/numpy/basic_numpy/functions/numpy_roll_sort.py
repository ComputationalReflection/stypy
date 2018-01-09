
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: faces = np.random.randint(0, 100, (10, 3))
6: F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
7: F = F.reshape(len(F) * 3, 2)
8: F = np.sort(F, axis=1)
9: G = F.view(dtype=[('p0', F.dtype), ('p1', F.dtype)])
10: G = np.unique(G)
11: #
12: # l = globals().copy()
13: # for v in l:
14: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2280 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2280) is not StypyTypeError):

    if (import_2280 != 'pyd_module'):
        __import__(import_2280)
        sys_modules_2281 = sys.modules[import_2280]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2281.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2280)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to randint(...): (line 5)
# Processing the call arguments (line 5)
int_2285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
int_2286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_2287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_2288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 35), tuple_2287, int_2288)
# Adding element type (line 5)
int_2289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 35), tuple_2287, int_2289)

# Processing the call keyword arguments (line 5)
kwargs_2290 = {}
# Getting the type of 'np' (line 5)
np_2282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_2283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 8), np_2282, 'random')
# Obtaining the member 'randint' of a type (line 5)
randint_2284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 8), random_2283, 'randint')
# Calling randint(args, kwargs) (line 5)
randint_call_result_2291 = invoke(stypy.reporting.localization.Localization(__file__, 5, 8), randint_2284, *[int_2285, int_2286, tuple_2287], **kwargs_2290)

# Assigning a type to the variable 'faces' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'faces', randint_call_result_2291)

# Assigning a Call to a Name (line 6):

# Call to roll(...): (line 6)
# Processing the call arguments (line 6)

# Call to repeat(...): (line 6)
# Processing the call arguments (line 6)
int_2296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'int')
# Processing the call keyword arguments (line 6)
int_2297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'int')
keyword_2298 = int_2297
kwargs_2299 = {'axis': keyword_2298}
# Getting the type of 'faces' (line 6)
faces_2294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 12), 'faces', False)
# Obtaining the member 'repeat' of a type (line 6)
repeat_2295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 12), faces_2294, 'repeat')
# Calling repeat(args, kwargs) (line 6)
repeat_call_result_2300 = invoke(stypy.reporting.localization.Localization(__file__, 6, 12), repeat_2295, *[int_2296], **kwargs_2299)

int_2301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 37), 'int')
# Processing the call keyword arguments (line 6)
int_2302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 46), 'int')
keyword_2303 = int_2302
kwargs_2304 = {'axis': keyword_2303}
# Getting the type of 'np' (line 6)
np_2292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'roll' of a type (line 6)
roll_2293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_2292, 'roll')
# Calling roll(args, kwargs) (line 6)
roll_call_result_2305 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), roll_2293, *[repeat_call_result_2300, int_2301], **kwargs_2304)

# Assigning a type to the variable 'F' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'F', roll_call_result_2305)

# Assigning a Call to a Name (line 7):

# Call to reshape(...): (line 7)
# Processing the call arguments (line 7)

# Call to len(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'F' (line 7)
F_2309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), 'F', False)
# Processing the call keyword arguments (line 7)
kwargs_2310 = {}
# Getting the type of 'len' (line 7)
len_2308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'len', False)
# Calling len(args, kwargs) (line 7)
len_call_result_2311 = invoke(stypy.reporting.localization.Localization(__file__, 7, 14), len_2308, *[F_2309], **kwargs_2310)

int_2312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 23), 'int')
# Applying the binary operator '*' (line 7)
result_mul_2313 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 14), '*', len_call_result_2311, int_2312)

int_2314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'int')
# Processing the call keyword arguments (line 7)
kwargs_2315 = {}
# Getting the type of 'F' (line 7)
F_2306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'F', False)
# Obtaining the member 'reshape' of a type (line 7)
reshape_2307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), F_2306, 'reshape')
# Calling reshape(args, kwargs) (line 7)
reshape_call_result_2316 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), reshape_2307, *[result_mul_2313, int_2314], **kwargs_2315)

# Assigning a type to the variable 'F' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'F', reshape_call_result_2316)

# Assigning a Call to a Name (line 8):

# Call to sort(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'F' (line 8)
F_2319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'F', False)
# Processing the call keyword arguments (line 8)
int_2320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
keyword_2321 = int_2320
kwargs_2322 = {'axis': keyword_2321}
# Getting the type of 'np' (line 8)
np_2317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'np', False)
# Obtaining the member 'sort' of a type (line 8)
sort_2318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), np_2317, 'sort')
# Calling sort(args, kwargs) (line 8)
sort_call_result_2323 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), sort_2318, *[F_2319], **kwargs_2322)

# Assigning a type to the variable 'F' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'F', sort_call_result_2323)

# Assigning a Call to a Name (line 9):

# Call to view(...): (line 9)
# Processing the call keyword arguments (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_2326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_2327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
str_2328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'str', 'p0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 19), tuple_2327, str_2328)
# Adding element type (line 9)
# Getting the type of 'F' (line 9)
F_2329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 25), 'F', False)
# Obtaining the member 'dtype' of a type (line 9)
dtype_2330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 25), F_2329, 'dtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 19), tuple_2327, dtype_2330)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_2326, tuple_2327)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_2331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
str_2332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 36), 'str', 'p1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 36), tuple_2331, str_2332)
# Adding element type (line 9)
# Getting the type of 'F' (line 9)
F_2333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 42), 'F', False)
# Obtaining the member 'dtype' of a type (line 9)
dtype_2334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 42), F_2333, 'dtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 36), tuple_2331, dtype_2334)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_2326, tuple_2331)

keyword_2335 = list_2326
kwargs_2336 = {'dtype': keyword_2335}
# Getting the type of 'F' (line 9)
F_2324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'F', False)
# Obtaining the member 'view' of a type (line 9)
view_2325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), F_2324, 'view')
# Calling view(args, kwargs) (line 9)
view_call_result_2337 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), view_2325, *[], **kwargs_2336)

# Assigning a type to the variable 'G' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'G', view_call_result_2337)

# Assigning a Call to a Name (line 10):

# Call to unique(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'G' (line 10)
G_2340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'G', False)
# Processing the call keyword arguments (line 10)
kwargs_2341 = {}
# Getting the type of 'np' (line 10)
np_2338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'np', False)
# Obtaining the member 'unique' of a type (line 10)
unique_2339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), np_2338, 'unique')
# Calling unique(args, kwargs) (line 10)
unique_call_result_2342 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), unique_2339, *[G_2340], **kwargs_2341)

# Assigning a type to the variable 'G' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'G', unique_call_result_2342)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
