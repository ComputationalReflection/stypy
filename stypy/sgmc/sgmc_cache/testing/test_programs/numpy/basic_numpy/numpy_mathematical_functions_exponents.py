
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # https://docs.scipy.org/doc/numpy/reference/routines.math.html
2: 
3: import numpy as np
4: 
5: x = 2.1
6: x1 = 3.4
7: x2 = 5.6
8: 
9: # Exponents and logarithms
10: r1 = np.exp(x)  # Calculate the exponential of all elements in the input array.
11: r2 = np.expm1(x)  # Calculate exp(x) - 1 for all elements in the array.
12: r3 = np.exp2(x)  # Calculate 2**p for all p in the input array.
13: r4 = np.log(x)  # Natural logarithm, element-wise.
14: r5 = np.log10(x)  # Return the base 10 logarithm of the input array, element-wise.
15: r6 = np.log2(x)  # Base-2 logarithm of x.
16: r7 = np.log1p(x)  # Return the natural logarithm of one plus the input array, element-wise.
17: r8 = np.logaddexp(x1, x2)  # Logarithm of the sum of exponentiations of the inputs.
18: r9 = np.logaddexp2(x1, x2)  # Logarithm of the sum of exponentiations of the inputs in base-2.
19: 
20: x = [2.1, 4.5, 6.7]
21: x1 = [3.4, 7.8, 9.8]
22: x2 = [5.6, 2.3, 6.7]
23: 
24: r10 = np.exp(x)  # Calculate the exponential of all elements in the input array.
25: r11 = np.expm1(x)  # Calculate exp(x) - 1 for all elements in the array.
26: r12 = np.exp2(x)  # Calculate 2**p for all p in the input array.
27: r13 = np.log(x)  # Natural logarithm, element-wise.
28: r14 = np.log10(x)  # Return the base 10 logarithm of the input array, element-wise.
29: r15 = np.log2(x)  # Base-2 logarithm of x.
30: r16 = np.log1p(x)  # Return the natural logarithm of one plus the input array, element-wise.
31: r17 = np.logaddexp(x1, x2)  # Logarithm of the sum of exponentiations of the inputs.
32: r18 = np.logaddexp2(x1, x2)  # Logarithm of the sum of exponentiations of the inputs in base-2.
33: 
34: # l = globals().copy()
35: # for v in l:
36: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_231 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_231) is not StypyTypeError):

    if (import_231 != 'pyd_module'):
        __import__(import_231)
        sys_modules_232 = sys.modules[import_231]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_232.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_231)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Num to a Name (line 5):
float_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'float')
# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', float_233)

# Assigning a Num to a Name (line 6):
float_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 5), 'float')
# Assigning a type to the variable 'x1' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x1', float_234)

# Assigning a Num to a Name (line 7):
float_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 5), 'float')
# Assigning a type to the variable 'x2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x2', float_235)

# Assigning a Call to a Name (line 10):

# Call to exp(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'x' (line 10)
x_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'x', False)
# Processing the call keyword arguments (line 10)
kwargs_239 = {}
# Getting the type of 'np' (line 10)
np_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'np', False)
# Obtaining the member 'exp' of a type (line 10)
exp_237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), np_236, 'exp')
# Calling exp(args, kwargs) (line 10)
exp_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), exp_237, *[x_238], **kwargs_239)

# Assigning a type to the variable 'r1' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r1', exp_call_result_240)

# Assigning a Call to a Name (line 11):

# Call to expm1(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'x' (line 11)
x_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'x', False)
# Processing the call keyword arguments (line 11)
kwargs_244 = {}
# Getting the type of 'np' (line 11)
np_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'expm1' of a type (line 11)
expm1_242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_241, 'expm1')
# Calling expm1(args, kwargs) (line 11)
expm1_call_result_245 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), expm1_242, *[x_243], **kwargs_244)

# Assigning a type to the variable 'r2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r2', expm1_call_result_245)

# Assigning a Call to a Name (line 12):

# Call to exp2(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'x' (line 12)
x_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'x', False)
# Processing the call keyword arguments (line 12)
kwargs_249 = {}
# Getting the type of 'np' (line 12)
np_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'np', False)
# Obtaining the member 'exp2' of a type (line 12)
exp2_247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), np_246, 'exp2')
# Calling exp2(args, kwargs) (line 12)
exp2_call_result_250 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), exp2_247, *[x_248], **kwargs_249)

# Assigning a type to the variable 'r3' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r3', exp2_call_result_250)

# Assigning a Call to a Name (line 13):

# Call to log(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'x' (line 13)
x_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'x', False)
# Processing the call keyword arguments (line 13)
kwargs_254 = {}
# Getting the type of 'np' (line 13)
np_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'np', False)
# Obtaining the member 'log' of a type (line 13)
log_252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), np_251, 'log')
# Calling log(args, kwargs) (line 13)
log_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), log_252, *[x_253], **kwargs_254)

# Assigning a type to the variable 'r4' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r4', log_call_result_255)

# Assigning a Call to a Name (line 14):

# Call to log10(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'x' (line 14)
x_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'x', False)
# Processing the call keyword arguments (line 14)
kwargs_259 = {}
# Getting the type of 'np' (line 14)
np_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'np', False)
# Obtaining the member 'log10' of a type (line 14)
log10_257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), np_256, 'log10')
# Calling log10(args, kwargs) (line 14)
log10_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), log10_257, *[x_258], **kwargs_259)

# Assigning a type to the variable 'r5' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r5', log10_call_result_260)

# Assigning a Call to a Name (line 15):

# Call to log2(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'x' (line 15)
x_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'x', False)
# Processing the call keyword arguments (line 15)
kwargs_264 = {}
# Getting the type of 'np' (line 15)
np_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np', False)
# Obtaining the member 'log2' of a type (line 15)
log2_262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_261, 'log2')
# Calling log2(args, kwargs) (line 15)
log2_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), log2_262, *[x_263], **kwargs_264)

# Assigning a type to the variable 'r6' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r6', log2_call_result_265)

# Assigning a Call to a Name (line 16):

# Call to log1p(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'x' (line 16)
x_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'x', False)
# Processing the call keyword arguments (line 16)
kwargs_269 = {}
# Getting the type of 'np' (line 16)
np_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'np', False)
# Obtaining the member 'log1p' of a type (line 16)
log1p_267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 5), np_266, 'log1p')
# Calling log1p(args, kwargs) (line 16)
log1p_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 16, 5), log1p_267, *[x_268], **kwargs_269)

# Assigning a type to the variable 'r7' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r7', log1p_call_result_270)

# Assigning a Call to a Name (line 17):

# Call to logaddexp(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of 'x1' (line 17)
x1_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 18), 'x1', False)
# Getting the type of 'x2' (line 17)
x2_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'x2', False)
# Processing the call keyword arguments (line 17)
kwargs_275 = {}
# Getting the type of 'np' (line 17)
np_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'logaddexp' of a type (line 17)
logaddexp_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_271, 'logaddexp')
# Calling logaddexp(args, kwargs) (line 17)
logaddexp_call_result_276 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), logaddexp_272, *[x1_273, x2_274], **kwargs_275)

# Assigning a type to the variable 'r8' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r8', logaddexp_call_result_276)

# Assigning a Call to a Name (line 18):

# Call to logaddexp2(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of 'x1' (line 18)
x1_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'x1', False)
# Getting the type of 'x2' (line 18)
x2_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'x2', False)
# Processing the call keyword arguments (line 18)
kwargs_281 = {}
# Getting the type of 'np' (line 18)
np_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'np', False)
# Obtaining the member 'logaddexp2' of a type (line 18)
logaddexp2_278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), np_277, 'logaddexp2')
# Calling logaddexp2(args, kwargs) (line 18)
logaddexp2_call_result_282 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), logaddexp2_278, *[x1_279, x2_280], **kwargs_281)

# Assigning a type to the variable 'r9' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r9', logaddexp2_call_result_282)

# Assigning a List to a Name (line 20):

# Obtaining an instance of the builtin type 'list' (line 20)
list_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
float_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), list_283, float_284)
# Adding element type (line 20)
float_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), list_283, float_285)
# Adding element type (line 20)
float_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), list_283, float_286)

# Assigning a type to the variable 'x' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'x', list_283)

# Assigning a List to a Name (line 21):

# Obtaining an instance of the builtin type 'list' (line 21)
list_287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
float_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 6), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 5), list_287, float_288)
# Adding element type (line 21)
float_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 5), list_287, float_289)
# Adding element type (line 21)
float_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 5), list_287, float_290)

# Assigning a type to the variable 'x1' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'x1', list_287)

# Assigning a List to a Name (line 22):

# Obtaining an instance of the builtin type 'list' (line 22)
list_291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
float_292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 6), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 5), list_291, float_292)
# Adding element type (line 22)
float_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 5), list_291, float_293)
# Adding element type (line 22)
float_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 5), list_291, float_294)

# Assigning a type to the variable 'x2' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'x2', list_291)

# Assigning a Call to a Name (line 24):

# Call to exp(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'x' (line 24)
x_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'x', False)
# Processing the call keyword arguments (line 24)
kwargs_298 = {}
# Getting the type of 'np' (line 24)
np_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 6), 'np', False)
# Obtaining the member 'exp' of a type (line 24)
exp_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 6), np_295, 'exp')
# Calling exp(args, kwargs) (line 24)
exp_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 24, 6), exp_296, *[x_297], **kwargs_298)

# Assigning a type to the variable 'r10' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'r10', exp_call_result_299)

# Assigning a Call to a Name (line 25):

# Call to expm1(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'x' (line 25)
x_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'x', False)
# Processing the call keyword arguments (line 25)
kwargs_303 = {}
# Getting the type of 'np' (line 25)
np_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 6), 'np', False)
# Obtaining the member 'expm1' of a type (line 25)
expm1_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 6), np_300, 'expm1')
# Calling expm1(args, kwargs) (line 25)
expm1_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 25, 6), expm1_301, *[x_302], **kwargs_303)

# Assigning a type to the variable 'r11' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'r11', expm1_call_result_304)

# Assigning a Call to a Name (line 26):

# Call to exp2(...): (line 26)
# Processing the call arguments (line 26)
# Getting the type of 'x' (line 26)
x_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'x', False)
# Processing the call keyword arguments (line 26)
kwargs_308 = {}
# Getting the type of 'np' (line 26)
np_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 6), 'np', False)
# Obtaining the member 'exp2' of a type (line 26)
exp2_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 6), np_305, 'exp2')
# Calling exp2(args, kwargs) (line 26)
exp2_call_result_309 = invoke(stypy.reporting.localization.Localization(__file__, 26, 6), exp2_306, *[x_307], **kwargs_308)

# Assigning a type to the variable 'r12' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'r12', exp2_call_result_309)

# Assigning a Call to a Name (line 27):

# Call to log(...): (line 27)
# Processing the call arguments (line 27)
# Getting the type of 'x' (line 27)
x_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'x', False)
# Processing the call keyword arguments (line 27)
kwargs_313 = {}
# Getting the type of 'np' (line 27)
np_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 6), 'np', False)
# Obtaining the member 'log' of a type (line 27)
log_311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 6), np_310, 'log')
# Calling log(args, kwargs) (line 27)
log_call_result_314 = invoke(stypy.reporting.localization.Localization(__file__, 27, 6), log_311, *[x_312], **kwargs_313)

# Assigning a type to the variable 'r13' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'r13', log_call_result_314)

# Assigning a Call to a Name (line 28):

# Call to log10(...): (line 28)
# Processing the call arguments (line 28)
# Getting the type of 'x' (line 28)
x_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'x', False)
# Processing the call keyword arguments (line 28)
kwargs_318 = {}
# Getting the type of 'np' (line 28)
np_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 6), 'np', False)
# Obtaining the member 'log10' of a type (line 28)
log10_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 6), np_315, 'log10')
# Calling log10(args, kwargs) (line 28)
log10_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 28, 6), log10_316, *[x_317], **kwargs_318)

# Assigning a type to the variable 'r14' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'r14', log10_call_result_319)

# Assigning a Call to a Name (line 29):

# Call to log2(...): (line 29)
# Processing the call arguments (line 29)
# Getting the type of 'x' (line 29)
x_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'x', False)
# Processing the call keyword arguments (line 29)
kwargs_323 = {}
# Getting the type of 'np' (line 29)
np_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 6), 'np', False)
# Obtaining the member 'log2' of a type (line 29)
log2_321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 6), np_320, 'log2')
# Calling log2(args, kwargs) (line 29)
log2_call_result_324 = invoke(stypy.reporting.localization.Localization(__file__, 29, 6), log2_321, *[x_322], **kwargs_323)

# Assigning a type to the variable 'r15' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'r15', log2_call_result_324)

# Assigning a Call to a Name (line 30):

# Call to log1p(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of 'x' (line 30)
x_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'x', False)
# Processing the call keyword arguments (line 30)
kwargs_328 = {}
# Getting the type of 'np' (line 30)
np_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 6), 'np', False)
# Obtaining the member 'log1p' of a type (line 30)
log1p_326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 6), np_325, 'log1p')
# Calling log1p(args, kwargs) (line 30)
log1p_call_result_329 = invoke(stypy.reporting.localization.Localization(__file__, 30, 6), log1p_326, *[x_327], **kwargs_328)

# Assigning a type to the variable 'r16' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'r16', log1p_call_result_329)

# Assigning a Call to a Name (line 31):

# Call to logaddexp(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of 'x1' (line 31)
x1_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'x1', False)
# Getting the type of 'x2' (line 31)
x2_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'x2', False)
# Processing the call keyword arguments (line 31)
kwargs_334 = {}
# Getting the type of 'np' (line 31)
np_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 6), 'np', False)
# Obtaining the member 'logaddexp' of a type (line 31)
logaddexp_331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 6), np_330, 'logaddexp')
# Calling logaddexp(args, kwargs) (line 31)
logaddexp_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 31, 6), logaddexp_331, *[x1_332, x2_333], **kwargs_334)

# Assigning a type to the variable 'r17' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'r17', logaddexp_call_result_335)

# Assigning a Call to a Name (line 32):

# Call to logaddexp2(...): (line 32)
# Processing the call arguments (line 32)
# Getting the type of 'x1' (line 32)
x1_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'x1', False)
# Getting the type of 'x2' (line 32)
x2_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'x2', False)
# Processing the call keyword arguments (line 32)
kwargs_340 = {}
# Getting the type of 'np' (line 32)
np_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 6), 'np', False)
# Obtaining the member 'logaddexp2' of a type (line 32)
logaddexp2_337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 6), np_336, 'logaddexp2')
# Calling logaddexp2(args, kwargs) (line 32)
logaddexp2_call_result_341 = invoke(stypy.reporting.localization.Localization(__file__, 32, 6), logaddexp2_337, *[x1_338, x2_339], **kwargs_340)

# Assigning a type to the variable 'r18' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'r18', logaddexp2_call_result_341)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
