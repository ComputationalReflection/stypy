
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: import numpy as np
3: 
4: A = np.random.randint(0, 10, (3, 4, 3, 4))
5: sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
6: 
7: # l = globals().copy()
8: # for v in l:
9: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2248 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_2248) is not StypyTypeError):

    if (import_2248 != 'pyd_module'):
        __import__(import_2248)
        sys_modules_2249 = sys.modules[import_2248]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_2249.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_2248)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 4):

# Call to randint(...): (line 4)
# Processing the call arguments (line 4)
int_2253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 22), 'int')
int_2254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 25), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 4)
tuple_2255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 30), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4)
# Adding element type (line 4)
int_2256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 30), tuple_2255, int_2256)
# Adding element type (line 4)
int_2257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 30), tuple_2255, int_2257)
# Adding element type (line 4)
int_2258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 30), tuple_2255, int_2258)
# Adding element type (line 4)
int_2259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 30), tuple_2255, int_2259)

# Processing the call keyword arguments (line 4)
kwargs_2260 = {}
# Getting the type of 'np' (line 4)
np_2250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'np', False)
# Obtaining the member 'random' of a type (line 4)
random_2251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), np_2250, 'random')
# Obtaining the member 'randint' of a type (line 4)
randint_2252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), random_2251, 'randint')
# Calling randint(args, kwargs) (line 4)
randint_call_result_2261 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), randint_2252, *[int_2253, int_2254, tuple_2255], **kwargs_2260)

# Assigning a type to the variable 'A' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'A', randint_call_result_2261)

# Assigning a Call to a Name (line 5):

# Call to sum(...): (line 5)
# Processing the call keyword arguments (line 5)
int_2276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 47), 'int')
keyword_2277 = int_2276
kwargs_2278 = {'axis': keyword_2277}

# Call to reshape(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining the type of the subscript
int_2264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')
slice_2265 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 5, 16), None, int_2264, None)
# Getting the type of 'A' (line 5)
A_2266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 16), 'A', False)
# Obtaining the member 'shape' of a type (line 5)
shape_2267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 16), A_2266, 'shape')
# Obtaining the member '__getitem__' of a type (line 5)
getitem___2268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 16), shape_2267, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_2269 = invoke(stypy.reporting.localization.Localization(__file__, 5, 16), getitem___2268, slice_2265)


# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_2270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_2271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 32), tuple_2270, int_2271)

# Applying the binary operator '+' (line 5)
result_add_2272 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 16), '+', subscript_call_result_2269, tuple_2270)

# Processing the call keyword arguments (line 5)
kwargs_2273 = {}
# Getting the type of 'A' (line 5)
A_2262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 6), 'A', False)
# Obtaining the member 'reshape' of a type (line 5)
reshape_2263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 6), A_2262, 'reshape')
# Calling reshape(args, kwargs) (line 5)
reshape_call_result_2274 = invoke(stypy.reporting.localization.Localization(__file__, 5, 6), reshape_2263, *[result_add_2272], **kwargs_2273)

# Obtaining the member 'sum' of a type (line 5)
sum_2275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 6), reshape_call_result_2274, 'sum')
# Calling sum(args, kwargs) (line 5)
sum_call_result_2279 = invoke(stypy.reporting.localization.Localization(__file__, 5, 6), sum_2275, *[], **kwargs_2278)

# Assigning a type to the variable 'sum' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'sum', sum_call_result_2279)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
