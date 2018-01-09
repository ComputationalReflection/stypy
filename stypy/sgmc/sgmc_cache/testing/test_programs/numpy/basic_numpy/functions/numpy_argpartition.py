
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.arange(10000)
6: x2 = np.random.shuffle(Z)
7: n = 5
8: 
9: # Slow
10: r = (Z[np.argsort(Z)[-n:]])
11: 
12: # Fast
13: r2 = (Z[np.argpartition(-Z, n)[:n]])
14: 
15: 
16: # l = globals().copy()
17: # for v in l:
18: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_255 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_255) is not StypyTypeError):

    if (import_255 != 'pyd_module'):
        __import__(import_255)
        sys_modules_256 = sys.modules[import_255]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_256.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_255)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to arange(...): (line 5)
# Processing the call arguments (line 5)
int_259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
# Processing the call keyword arguments (line 5)
kwargs_260 = {}
# Getting the type of 'np' (line 5)
np_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 5)
arange_258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_257, 'arange')
# Calling arange(args, kwargs) (line 5)
arange_call_result_261 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), arange_258, *[int_259], **kwargs_260)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', arange_call_result_261)

# Assigning a Call to a Name (line 6):

# Call to shuffle(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'Z' (line 6)
Z_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 23), 'Z', False)
# Processing the call keyword arguments (line 6)
kwargs_266 = {}
# Getting the type of 'np' (line 6)
np_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'np', False)
# Obtaining the member 'random' of a type (line 6)
random_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), np_262, 'random')
# Obtaining the member 'shuffle' of a type (line 6)
shuffle_264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), random_263, 'shuffle')
# Calling shuffle(args, kwargs) (line 6)
shuffle_call_result_267 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), shuffle_264, *[Z_265], **kwargs_266)

# Assigning a type to the variable 'x2' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x2', shuffle_call_result_267)

# Assigning a Num to a Name (line 7):
int_268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 4), 'int')
# Assigning a type to the variable 'n' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'n', int_268)

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript

# Obtaining the type of the subscript

# Getting the type of 'n' (line 10)
n_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 22), 'n')
# Applying the 'usub' unary operator (line 10)
result___neg___270 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 21), 'usub', n_269)

slice_271 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 10, 7), result___neg___270, None, None)

# Call to argsort(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'Z' (line 10)
Z_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), 'Z', False)
# Processing the call keyword arguments (line 10)
kwargs_275 = {}
# Getting the type of 'np' (line 10)
np_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 7), 'np', False)
# Obtaining the member 'argsort' of a type (line 10)
argsort_273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 7), np_272, 'argsort')
# Calling argsort(args, kwargs) (line 10)
argsort_call_result_276 = invoke(stypy.reporting.localization.Localization(__file__, 10, 7), argsort_273, *[Z_274], **kwargs_275)

# Obtaining the member '__getitem__' of a type (line 10)
getitem___277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 7), argsort_call_result_276, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 10, 7), getitem___277, slice_271)

# Getting the type of 'Z' (line 10)
Z_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'Z')
# Obtaining the member '__getitem__' of a type (line 10)
getitem___280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), Z_279, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_281 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), getitem___280, subscript_call_result_278)

# Assigning a type to the variable 'r' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r', subscript_call_result_281)

# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript

# Obtaining the type of the subscript
# Getting the type of 'n' (line 13)
n_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 32), 'n')
slice_283 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 13, 8), None, n_282, None)

# Call to argpartition(...): (line 13)
# Processing the call arguments (line 13)

# Getting the type of 'Z' (line 13)
Z_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'Z', False)
# Applying the 'usub' unary operator (line 13)
result___neg___287 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 24), 'usub', Z_286)

# Getting the type of 'n' (line 13)
n_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'n', False)
# Processing the call keyword arguments (line 13)
kwargs_289 = {}
# Getting the type of 'np' (line 13)
np_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'np', False)
# Obtaining the member 'argpartition' of a type (line 13)
argpartition_285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), np_284, 'argpartition')
# Calling argpartition(args, kwargs) (line 13)
argpartition_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), argpartition_285, *[result___neg___287, n_288], **kwargs_289)

# Obtaining the member '__getitem__' of a type (line 13)
getitem___291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), argpartition_call_result_290, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), getitem___291, slice_283)

# Getting the type of 'Z' (line 13)
Z_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 6), 'Z')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 6), Z_293, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_295 = invoke(stypy.reporting.localization.Localization(__file__, 13, 6), getitem___294, subscript_call_result_292)

# Assigning a type to the variable 'r2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r2', subscript_call_result_295)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
