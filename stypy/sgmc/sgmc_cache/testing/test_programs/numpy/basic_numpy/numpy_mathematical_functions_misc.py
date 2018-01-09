
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # https://docs.scipy.org/doc/numpy/reference/routines.math.html
2: 
3: 
4: import numpy as np
5: 
6: a = 2.1
7: a_min = 1
8: a_max = 3
9: v = 3.4
10: x = 3.5
11: xp = 5.6
12: fp = 5.8
13: x1 = 3.6
14: x2 = 3.7
15: 
16: # Miscellaneous
17: r1 = np.convolve(a, v)  # Returns the discrete, linear convolution of two one-dimensional sequences.
18: r2 = np.clip(a, a_min, a_max)  # Clip (limit) the values in an array.
19: r3 = np.sqrt(x)  # Return the positive square-root of an array, element-wise.
20: r4 = np.square(x)  # Return the element-wise square of the input.
21: r5 = np.absolute(x)  # Calculate the absolute value element-wise.
22: r6 = np.fabs(x)  # Compute the absolute values element-wise.
23: r7 = np.sign(x)  # Returns an element-wise indication of the sign of a number.
24: r8 = np.maximum(x1, x2)  # Element-wise maximum of array elements.
25: r9 = np.minimum(x1, x2)  # Element-wise minimum of array elements.
26: r10 = np.fmax(x1, x2)  # Element-wise maximum of array elements.
27: r11 = np.fmin(x1, x2)  # Element-wise minimum of array elements.
28: r12 = np.nan_to_num(x)  # Replace nan with zero and inf with finite numbers.
29: r13 = np.real_if_close(a)  # If complex input returns a real array if complex parts are close to zero.
30: # Type error
31: r14b = np.interp(x, xp, fp) 	#One-dimensional linear interpolation.
32: 
33: a = [2.1, 3.4, 5.6]
34: a_min = 1
35: a_max = 3
36: v = 3.4
37: x = [3.5, 3.6, 3.7, 3.8]
38: xp = 5.6
39: fp = 5.8
40: x1 = 3.6
41: x2 = 3.7
42: 
43: r14 = np.convolve(a, v)  # Returns the discrete, linear convolution of two one-dimensional sequences.
44: r15 = np.clip(a, a_min, a_max)  # Clip (limit) the values in an array.
45: r16 = np.sqrt(x)  # Return the positive square-root of an array, element-wise.
46: r17 = np.square(x)  # Return the element-wise square of the input.
47: r18 = np.absolute(x)  # Calculate the absolute value element-wise.
48: r19 = np.fabs(x)  # Compute the absolute values element-wise.
49: r20 = np.sign(x)  # Returns an element-wise indication of the sign of a number.
50: r21 = np.maximum(x1, x2)  # Element-wise maximum of array elements.
51: r22 = np.minimum(x1, x2)  # Element-wise minimum of array elements.
52: r23 = np.fmax(x1, x2)  # Element-wise maximum of array elements.
53: r24 = np.fmin(x1, x2)  # Element-wise minimum of array elements.
54: r25 = np.nan_to_num(x)  # Replace nan with zero and inf with finite numbers.
55: r26 = np.real_if_close(a)  # If complex input returns a real array if complex parts are close to zero.
56: 
57: xp = [1, 2, 3]
58: fp = [3, 2, 0]
59: r27 = np.interp(2.5, xp, fp)
60: 
61: r28 = np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
62: r29 = np.array([3., 3., 2.5, 0.56, 0.])
63: UNDEF = -99.0
64: r30 = np.interp(3.14, xp, fp, right=UNDEF)
65: 
66: # l = globals().copy()
67: # for v in l:
68: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
69: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_455 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_455) is not StypyTypeError):

    if (import_455 != 'pyd_module'):
        __import__(import_455)
        sys_modules_456 = sys.modules[import_455]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_456.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_455)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Num to a Name (line 6):
float_457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'float')
# Assigning a type to the variable 'a' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'a', float_457)

# Assigning a Num to a Name (line 7):
int_458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 8), 'int')
# Assigning a type to the variable 'a_min' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'a_min', int_458)

# Assigning a Num to a Name (line 8):
int_459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'int')
# Assigning a type to the variable 'a_max' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'a_max', int_459)

# Assigning a Num to a Name (line 9):
float_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'float')
# Assigning a type to the variable 'v' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'v', float_460)

# Assigning a Num to a Name (line 10):
float_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'float')
# Assigning a type to the variable 'x' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'x', float_461)

# Assigning a Num to a Name (line 11):
float_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 5), 'float')
# Assigning a type to the variable 'xp' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'xp', float_462)

# Assigning a Num to a Name (line 12):
float_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 5), 'float')
# Assigning a type to the variable 'fp' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'fp', float_463)

# Assigning a Num to a Name (line 13):
float_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 5), 'float')
# Assigning a type to the variable 'x1' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'x1', float_464)

# Assigning a Num to a Name (line 14):
float_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 5), 'float')
# Assigning a type to the variable 'x2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'x2', float_465)

# Assigning a Call to a Name (line 17):

# Call to convolve(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of 'a' (line 17)
a_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'a', False)
# Getting the type of 'v' (line 17)
v_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'v', False)
# Processing the call keyword arguments (line 17)
kwargs_470 = {}
# Getting the type of 'np' (line 17)
np_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'convolve' of a type (line 17)
convolve_467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_466, 'convolve')
# Calling convolve(args, kwargs) (line 17)
convolve_call_result_471 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), convolve_467, *[a_468, v_469], **kwargs_470)

# Assigning a type to the variable 'r1' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r1', convolve_call_result_471)

# Assigning a Call to a Name (line 18):

# Call to clip(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of 'a' (line 18)
a_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'a', False)
# Getting the type of 'a_min' (line 18)
a_min_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'a_min', False)
# Getting the type of 'a_max' (line 18)
a_max_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'a_max', False)
# Processing the call keyword arguments (line 18)
kwargs_477 = {}
# Getting the type of 'np' (line 18)
np_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'np', False)
# Obtaining the member 'clip' of a type (line 18)
clip_473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), np_472, 'clip')
# Calling clip(args, kwargs) (line 18)
clip_call_result_478 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), clip_473, *[a_474, a_min_475, a_max_476], **kwargs_477)

# Assigning a type to the variable 'r2' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r2', clip_call_result_478)

# Assigning a Call to a Name (line 19):

# Call to sqrt(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'x' (line 19)
x_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'x', False)
# Processing the call keyword arguments (line 19)
kwargs_482 = {}
# Getting the type of 'np' (line 19)
np_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'np', False)
# Obtaining the member 'sqrt' of a type (line 19)
sqrt_480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), np_479, 'sqrt')
# Calling sqrt(args, kwargs) (line 19)
sqrt_call_result_483 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), sqrt_480, *[x_481], **kwargs_482)

# Assigning a type to the variable 'r3' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r3', sqrt_call_result_483)

# Assigning a Call to a Name (line 20):

# Call to square(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'x' (line 20)
x_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'x', False)
# Processing the call keyword arguments (line 20)
kwargs_487 = {}
# Getting the type of 'np' (line 20)
np_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 5), 'np', False)
# Obtaining the member 'square' of a type (line 20)
square_485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 5), np_484, 'square')
# Calling square(args, kwargs) (line 20)
square_call_result_488 = invoke(stypy.reporting.localization.Localization(__file__, 20, 5), square_485, *[x_486], **kwargs_487)

# Assigning a type to the variable 'r4' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'r4', square_call_result_488)

# Assigning a Call to a Name (line 21):

# Call to absolute(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'x' (line 21)
x_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'x', False)
# Processing the call keyword arguments (line 21)
kwargs_492 = {}
# Getting the type of 'np' (line 21)
np_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'np', False)
# Obtaining the member 'absolute' of a type (line 21)
absolute_490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), np_489, 'absolute')
# Calling absolute(args, kwargs) (line 21)
absolute_call_result_493 = invoke(stypy.reporting.localization.Localization(__file__, 21, 5), absolute_490, *[x_491], **kwargs_492)

# Assigning a type to the variable 'r5' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r5', absolute_call_result_493)

# Assigning a Call to a Name (line 22):

# Call to fabs(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'x' (line 22)
x_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 13), 'x', False)
# Processing the call keyword arguments (line 22)
kwargs_497 = {}
# Getting the type of 'np' (line 22)
np_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'np', False)
# Obtaining the member 'fabs' of a type (line 22)
fabs_495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 5), np_494, 'fabs')
# Calling fabs(args, kwargs) (line 22)
fabs_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 22, 5), fabs_495, *[x_496], **kwargs_497)

# Assigning a type to the variable 'r6' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r6', fabs_call_result_498)

# Assigning a Call to a Name (line 23):

# Call to sign(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'x' (line 23)
x_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'x', False)
# Processing the call keyword arguments (line 23)
kwargs_502 = {}
# Getting the type of 'np' (line 23)
np_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'np', False)
# Obtaining the member 'sign' of a type (line 23)
sign_500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 5), np_499, 'sign')
# Calling sign(args, kwargs) (line 23)
sign_call_result_503 = invoke(stypy.reporting.localization.Localization(__file__, 23, 5), sign_500, *[x_501], **kwargs_502)

# Assigning a type to the variable 'r7' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r7', sign_call_result_503)

# Assigning a Call to a Name (line 24):

# Call to maximum(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'x1' (line 24)
x1_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'x1', False)
# Getting the type of 'x2' (line 24)
x2_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'x2', False)
# Processing the call keyword arguments (line 24)
kwargs_508 = {}
# Getting the type of 'np' (line 24)
np_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 5), 'np', False)
# Obtaining the member 'maximum' of a type (line 24)
maximum_505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 5), np_504, 'maximum')
# Calling maximum(args, kwargs) (line 24)
maximum_call_result_509 = invoke(stypy.reporting.localization.Localization(__file__, 24, 5), maximum_505, *[x1_506, x2_507], **kwargs_508)

# Assigning a type to the variable 'r8' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'r8', maximum_call_result_509)

# Assigning a Call to a Name (line 25):

# Call to minimum(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'x1' (line 25)
x1_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'x1', False)
# Getting the type of 'x2' (line 25)
x2_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'x2', False)
# Processing the call keyword arguments (line 25)
kwargs_514 = {}
# Getting the type of 'np' (line 25)
np_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 5), 'np', False)
# Obtaining the member 'minimum' of a type (line 25)
minimum_511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 5), np_510, 'minimum')
# Calling minimum(args, kwargs) (line 25)
minimum_call_result_515 = invoke(stypy.reporting.localization.Localization(__file__, 25, 5), minimum_511, *[x1_512, x2_513], **kwargs_514)

# Assigning a type to the variable 'r9' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'r9', minimum_call_result_515)

# Assigning a Call to a Name (line 26):

# Call to fmax(...): (line 26)
# Processing the call arguments (line 26)
# Getting the type of 'x1' (line 26)
x1_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'x1', False)
# Getting the type of 'x2' (line 26)
x2_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'x2', False)
# Processing the call keyword arguments (line 26)
kwargs_520 = {}
# Getting the type of 'np' (line 26)
np_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 6), 'np', False)
# Obtaining the member 'fmax' of a type (line 26)
fmax_517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 6), np_516, 'fmax')
# Calling fmax(args, kwargs) (line 26)
fmax_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 26, 6), fmax_517, *[x1_518, x2_519], **kwargs_520)

# Assigning a type to the variable 'r10' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'r10', fmax_call_result_521)

# Assigning a Call to a Name (line 27):

# Call to fmin(...): (line 27)
# Processing the call arguments (line 27)
# Getting the type of 'x1' (line 27)
x1_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'x1', False)
# Getting the type of 'x2' (line 27)
x2_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'x2', False)
# Processing the call keyword arguments (line 27)
kwargs_526 = {}
# Getting the type of 'np' (line 27)
np_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 6), 'np', False)
# Obtaining the member 'fmin' of a type (line 27)
fmin_523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 6), np_522, 'fmin')
# Calling fmin(args, kwargs) (line 27)
fmin_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 27, 6), fmin_523, *[x1_524, x2_525], **kwargs_526)

# Assigning a type to the variable 'r11' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'r11', fmin_call_result_527)

# Assigning a Call to a Name (line 28):

# Call to nan_to_num(...): (line 28)
# Processing the call arguments (line 28)
# Getting the type of 'x' (line 28)
x_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'x', False)
# Processing the call keyword arguments (line 28)
kwargs_531 = {}
# Getting the type of 'np' (line 28)
np_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 6), 'np', False)
# Obtaining the member 'nan_to_num' of a type (line 28)
nan_to_num_529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 6), np_528, 'nan_to_num')
# Calling nan_to_num(args, kwargs) (line 28)
nan_to_num_call_result_532 = invoke(stypy.reporting.localization.Localization(__file__, 28, 6), nan_to_num_529, *[x_530], **kwargs_531)

# Assigning a type to the variable 'r12' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'r12', nan_to_num_call_result_532)

# Assigning a Call to a Name (line 29):

# Call to real_if_close(...): (line 29)
# Processing the call arguments (line 29)
# Getting the type of 'a' (line 29)
a_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'a', False)
# Processing the call keyword arguments (line 29)
kwargs_536 = {}
# Getting the type of 'np' (line 29)
np_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 6), 'np', False)
# Obtaining the member 'real_if_close' of a type (line 29)
real_if_close_534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 6), np_533, 'real_if_close')
# Calling real_if_close(args, kwargs) (line 29)
real_if_close_call_result_537 = invoke(stypy.reporting.localization.Localization(__file__, 29, 6), real_if_close_534, *[a_535], **kwargs_536)

# Assigning a type to the variable 'r13' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'r13', real_if_close_call_result_537)

# Assigning a Call to a Name (line 31):

# Call to interp(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of 'x' (line 31)
x_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'x', False)
# Getting the type of 'xp' (line 31)
xp_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'xp', False)
# Getting the type of 'fp' (line 31)
fp_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'fp', False)
# Processing the call keyword arguments (line 31)
kwargs_543 = {}
# Getting the type of 'np' (line 31)
np_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'np', False)
# Obtaining the member 'interp' of a type (line 31)
interp_539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 7), np_538, 'interp')
# Calling interp(args, kwargs) (line 31)
interp_call_result_544 = invoke(stypy.reporting.localization.Localization(__file__, 31, 7), interp_539, *[x_540, xp_541, fp_542], **kwargs_543)

# Assigning a type to the variable 'r14b' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'r14b', interp_call_result_544)

# Assigning a List to a Name (line 33):

# Obtaining an instance of the builtin type 'list' (line 33)
list_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
float_546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), list_545, float_546)
# Adding element type (line 33)
float_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), list_545, float_547)
# Adding element type (line 33)
float_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), list_545, float_548)

# Assigning a type to the variable 'a' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'a', list_545)

# Assigning a Num to a Name (line 34):
int_549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'int')
# Assigning a type to the variable 'a_min' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'a_min', int_549)

# Assigning a Num to a Name (line 35):
int_550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 8), 'int')
# Assigning a type to the variable 'a_max' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'a_max', int_550)

# Assigning a Num to a Name (line 36):
float_551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'float')
# Assigning a type to the variable 'v' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'v', float_551)

# Assigning a List to a Name (line 37):

# Obtaining an instance of the builtin type 'list' (line 37)
list_552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
float_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 4), list_552, float_553)
# Adding element type (line 37)
float_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 4), list_552, float_554)
# Adding element type (line 37)
float_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 4), list_552, float_555)
# Adding element type (line 37)
float_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 4), list_552, float_556)

# Assigning a type to the variable 'x' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'x', list_552)

# Assigning a Num to a Name (line 38):
float_557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 5), 'float')
# Assigning a type to the variable 'xp' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'xp', float_557)

# Assigning a Num to a Name (line 39):
float_558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 5), 'float')
# Assigning a type to the variable 'fp' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'fp', float_558)

# Assigning a Num to a Name (line 40):
float_559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 5), 'float')
# Assigning a type to the variable 'x1' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'x1', float_559)

# Assigning a Num to a Name (line 41):
float_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 5), 'float')
# Assigning a type to the variable 'x2' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'x2', float_560)

# Assigning a Call to a Name (line 43):

# Call to convolve(...): (line 43)
# Processing the call arguments (line 43)
# Getting the type of 'a' (line 43)
a_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'a', False)
# Getting the type of 'v' (line 43)
v_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'v', False)
# Processing the call keyword arguments (line 43)
kwargs_565 = {}
# Getting the type of 'np' (line 43)
np_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 6), 'np', False)
# Obtaining the member 'convolve' of a type (line 43)
convolve_562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 6), np_561, 'convolve')
# Calling convolve(args, kwargs) (line 43)
convolve_call_result_566 = invoke(stypy.reporting.localization.Localization(__file__, 43, 6), convolve_562, *[a_563, v_564], **kwargs_565)

# Assigning a type to the variable 'r14' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'r14', convolve_call_result_566)

# Assigning a Call to a Name (line 44):

# Call to clip(...): (line 44)
# Processing the call arguments (line 44)
# Getting the type of 'a' (line 44)
a_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'a', False)
# Getting the type of 'a_min' (line 44)
a_min_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'a_min', False)
# Getting the type of 'a_max' (line 44)
a_max_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'a_max', False)
# Processing the call keyword arguments (line 44)
kwargs_572 = {}
# Getting the type of 'np' (line 44)
np_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 6), 'np', False)
# Obtaining the member 'clip' of a type (line 44)
clip_568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 6), np_567, 'clip')
# Calling clip(args, kwargs) (line 44)
clip_call_result_573 = invoke(stypy.reporting.localization.Localization(__file__, 44, 6), clip_568, *[a_569, a_min_570, a_max_571], **kwargs_572)

# Assigning a type to the variable 'r15' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'r15', clip_call_result_573)

# Assigning a Call to a Name (line 45):

# Call to sqrt(...): (line 45)
# Processing the call arguments (line 45)
# Getting the type of 'x' (line 45)
x_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), 'x', False)
# Processing the call keyword arguments (line 45)
kwargs_577 = {}
# Getting the type of 'np' (line 45)
np_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 6), 'np', False)
# Obtaining the member 'sqrt' of a type (line 45)
sqrt_575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 6), np_574, 'sqrt')
# Calling sqrt(args, kwargs) (line 45)
sqrt_call_result_578 = invoke(stypy.reporting.localization.Localization(__file__, 45, 6), sqrt_575, *[x_576], **kwargs_577)

# Assigning a type to the variable 'r16' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'r16', sqrt_call_result_578)

# Assigning a Call to a Name (line 46):

# Call to square(...): (line 46)
# Processing the call arguments (line 46)
# Getting the type of 'x' (line 46)
x_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'x', False)
# Processing the call keyword arguments (line 46)
kwargs_582 = {}
# Getting the type of 'np' (line 46)
np_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 6), 'np', False)
# Obtaining the member 'square' of a type (line 46)
square_580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 6), np_579, 'square')
# Calling square(args, kwargs) (line 46)
square_call_result_583 = invoke(stypy.reporting.localization.Localization(__file__, 46, 6), square_580, *[x_581], **kwargs_582)

# Assigning a type to the variable 'r17' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'r17', square_call_result_583)

# Assigning a Call to a Name (line 47):

# Call to absolute(...): (line 47)
# Processing the call arguments (line 47)
# Getting the type of 'x' (line 47)
x_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'x', False)
# Processing the call keyword arguments (line 47)
kwargs_587 = {}
# Getting the type of 'np' (line 47)
np_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 6), 'np', False)
# Obtaining the member 'absolute' of a type (line 47)
absolute_585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 6), np_584, 'absolute')
# Calling absolute(args, kwargs) (line 47)
absolute_call_result_588 = invoke(stypy.reporting.localization.Localization(__file__, 47, 6), absolute_585, *[x_586], **kwargs_587)

# Assigning a type to the variable 'r18' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'r18', absolute_call_result_588)

# Assigning a Call to a Name (line 48):

# Call to fabs(...): (line 48)
# Processing the call arguments (line 48)
# Getting the type of 'x' (line 48)
x_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'x', False)
# Processing the call keyword arguments (line 48)
kwargs_592 = {}
# Getting the type of 'np' (line 48)
np_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 6), 'np', False)
# Obtaining the member 'fabs' of a type (line 48)
fabs_590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 6), np_589, 'fabs')
# Calling fabs(args, kwargs) (line 48)
fabs_call_result_593 = invoke(stypy.reporting.localization.Localization(__file__, 48, 6), fabs_590, *[x_591], **kwargs_592)

# Assigning a type to the variable 'r19' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'r19', fabs_call_result_593)

# Assigning a Call to a Name (line 49):

# Call to sign(...): (line 49)
# Processing the call arguments (line 49)
# Getting the type of 'x' (line 49)
x_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 14), 'x', False)
# Processing the call keyword arguments (line 49)
kwargs_597 = {}
# Getting the type of 'np' (line 49)
np_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 6), 'np', False)
# Obtaining the member 'sign' of a type (line 49)
sign_595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 6), np_594, 'sign')
# Calling sign(args, kwargs) (line 49)
sign_call_result_598 = invoke(stypy.reporting.localization.Localization(__file__, 49, 6), sign_595, *[x_596], **kwargs_597)

# Assigning a type to the variable 'r20' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'r20', sign_call_result_598)

# Assigning a Call to a Name (line 50):

# Call to maximum(...): (line 50)
# Processing the call arguments (line 50)
# Getting the type of 'x1' (line 50)
x1_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'x1', False)
# Getting the type of 'x2' (line 50)
x2_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'x2', False)
# Processing the call keyword arguments (line 50)
kwargs_603 = {}
# Getting the type of 'np' (line 50)
np_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 6), 'np', False)
# Obtaining the member 'maximum' of a type (line 50)
maximum_600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 6), np_599, 'maximum')
# Calling maximum(args, kwargs) (line 50)
maximum_call_result_604 = invoke(stypy.reporting.localization.Localization(__file__, 50, 6), maximum_600, *[x1_601, x2_602], **kwargs_603)

# Assigning a type to the variable 'r21' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'r21', maximum_call_result_604)

# Assigning a Call to a Name (line 51):

# Call to minimum(...): (line 51)
# Processing the call arguments (line 51)
# Getting the type of 'x1' (line 51)
x1_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'x1', False)
# Getting the type of 'x2' (line 51)
x2_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'x2', False)
# Processing the call keyword arguments (line 51)
kwargs_609 = {}
# Getting the type of 'np' (line 51)
np_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 6), 'np', False)
# Obtaining the member 'minimum' of a type (line 51)
minimum_606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 6), np_605, 'minimum')
# Calling minimum(args, kwargs) (line 51)
minimum_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 51, 6), minimum_606, *[x1_607, x2_608], **kwargs_609)

# Assigning a type to the variable 'r22' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'r22', minimum_call_result_610)

# Assigning a Call to a Name (line 52):

# Call to fmax(...): (line 52)
# Processing the call arguments (line 52)
# Getting the type of 'x1' (line 52)
x1_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 14), 'x1', False)
# Getting the type of 'x2' (line 52)
x2_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'x2', False)
# Processing the call keyword arguments (line 52)
kwargs_615 = {}
# Getting the type of 'np' (line 52)
np_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 6), 'np', False)
# Obtaining the member 'fmax' of a type (line 52)
fmax_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 6), np_611, 'fmax')
# Calling fmax(args, kwargs) (line 52)
fmax_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 52, 6), fmax_612, *[x1_613, x2_614], **kwargs_615)

# Assigning a type to the variable 'r23' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'r23', fmax_call_result_616)

# Assigning a Call to a Name (line 53):

# Call to fmin(...): (line 53)
# Processing the call arguments (line 53)
# Getting the type of 'x1' (line 53)
x1_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'x1', False)
# Getting the type of 'x2' (line 53)
x2_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 18), 'x2', False)
# Processing the call keyword arguments (line 53)
kwargs_621 = {}
# Getting the type of 'np' (line 53)
np_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 6), 'np', False)
# Obtaining the member 'fmin' of a type (line 53)
fmin_618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 6), np_617, 'fmin')
# Calling fmin(args, kwargs) (line 53)
fmin_call_result_622 = invoke(stypy.reporting.localization.Localization(__file__, 53, 6), fmin_618, *[x1_619, x2_620], **kwargs_621)

# Assigning a type to the variable 'r24' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'r24', fmin_call_result_622)

# Assigning a Call to a Name (line 54):

# Call to nan_to_num(...): (line 54)
# Processing the call arguments (line 54)
# Getting the type of 'x' (line 54)
x_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'x', False)
# Processing the call keyword arguments (line 54)
kwargs_626 = {}
# Getting the type of 'np' (line 54)
np_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 6), 'np', False)
# Obtaining the member 'nan_to_num' of a type (line 54)
nan_to_num_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 6), np_623, 'nan_to_num')
# Calling nan_to_num(args, kwargs) (line 54)
nan_to_num_call_result_627 = invoke(stypy.reporting.localization.Localization(__file__, 54, 6), nan_to_num_624, *[x_625], **kwargs_626)

# Assigning a type to the variable 'r25' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'r25', nan_to_num_call_result_627)

# Assigning a Call to a Name (line 55):

# Call to real_if_close(...): (line 55)
# Processing the call arguments (line 55)
# Getting the type of 'a' (line 55)
a_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'a', False)
# Processing the call keyword arguments (line 55)
kwargs_631 = {}
# Getting the type of 'np' (line 55)
np_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 6), 'np', False)
# Obtaining the member 'real_if_close' of a type (line 55)
real_if_close_629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 6), np_628, 'real_if_close')
# Calling real_if_close(args, kwargs) (line 55)
real_if_close_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 55, 6), real_if_close_629, *[a_630], **kwargs_631)

# Assigning a type to the variable 'r26' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'r26', real_if_close_call_result_632)

# Assigning a List to a Name (line 57):

# Obtaining an instance of the builtin type 'list' (line 57)
list_633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 57)
# Adding element type (line 57)
int_634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 5), list_633, int_634)
# Adding element type (line 57)
int_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 5), list_633, int_635)
# Adding element type (line 57)
int_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 5), list_633, int_636)

# Assigning a type to the variable 'xp' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'xp', list_633)

# Assigning a List to a Name (line 58):

# Obtaining an instance of the builtin type 'list' (line 58)
list_637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 58)
# Adding element type (line 58)
int_638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 5), list_637, int_638)
# Adding element type (line 58)
int_639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 5), list_637, int_639)
# Adding element type (line 58)
int_640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 5), list_637, int_640)

# Assigning a type to the variable 'fp' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'fp', list_637)

# Assigning a Call to a Name (line 59):

# Call to interp(...): (line 59)
# Processing the call arguments (line 59)
float_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 16), 'float')
# Getting the type of 'xp' (line 59)
xp_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'xp', False)
# Getting the type of 'fp' (line 59)
fp_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'fp', False)
# Processing the call keyword arguments (line 59)
kwargs_646 = {}
# Getting the type of 'np' (line 59)
np_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 6), 'np', False)
# Obtaining the member 'interp' of a type (line 59)
interp_642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 6), np_641, 'interp')
# Calling interp(args, kwargs) (line 59)
interp_call_result_647 = invoke(stypy.reporting.localization.Localization(__file__, 59, 6), interp_642, *[float_643, xp_644, fp_645], **kwargs_646)

# Assigning a type to the variable 'r27' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'r27', interp_call_result_647)

# Assigning a Call to a Name (line 61):

# Call to interp(...): (line 61)
# Processing the call arguments (line 61)

# Obtaining an instance of the builtin type 'list' (line 61)
list_650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 61)
# Adding element type (line 61)
int_651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), list_650, int_651)
# Adding element type (line 61)
int_652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), list_650, int_652)
# Adding element type (line 61)
float_653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), list_650, float_653)
# Adding element type (line 61)
float_654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), list_650, float_654)
# Adding element type (line 61)
float_655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), list_650, float_655)

# Getting the type of 'xp' (line 61)
xp_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 41), 'xp', False)
# Getting the type of 'fp' (line 61)
fp_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 45), 'fp', False)
# Processing the call keyword arguments (line 61)
kwargs_658 = {}
# Getting the type of 'np' (line 61)
np_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 6), 'np', False)
# Obtaining the member 'interp' of a type (line 61)
interp_649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 6), np_648, 'interp')
# Calling interp(args, kwargs) (line 61)
interp_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 61, 6), interp_649, *[list_650, xp_656, fp_657], **kwargs_658)

# Assigning a type to the variable 'r28' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'r28', interp_call_result_659)

# Assigning a Call to a Name (line 62):

# Call to array(...): (line 62)
# Processing the call arguments (line 62)

# Obtaining an instance of the builtin type 'list' (line 62)
list_662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 62)
# Adding element type (line 62)
float_663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), list_662, float_663)
# Adding element type (line 62)
float_664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), list_662, float_664)
# Adding element type (line 62)
float_665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), list_662, float_665)
# Adding element type (line 62)
float_666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), list_662, float_666)
# Adding element type (line 62)
float_667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), list_662, float_667)

# Processing the call keyword arguments (line 62)
kwargs_668 = {}
# Getting the type of 'np' (line 62)
np_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 6), 'np', False)
# Obtaining the member 'array' of a type (line 62)
array_661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 6), np_660, 'array')
# Calling array(args, kwargs) (line 62)
array_call_result_669 = invoke(stypy.reporting.localization.Localization(__file__, 62, 6), array_661, *[list_662], **kwargs_668)

# Assigning a type to the variable 'r29' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'r29', array_call_result_669)

# Assigning a Num to a Name (line 63):
float_670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'float')
# Assigning a type to the variable 'UNDEF' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'UNDEF', float_670)

# Assigning a Call to a Name (line 64):

# Call to interp(...): (line 64)
# Processing the call arguments (line 64)
float_673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'float')
# Getting the type of 'xp' (line 64)
xp_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'xp', False)
# Getting the type of 'fp' (line 64)
fp_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'fp', False)
# Processing the call keyword arguments (line 64)
# Getting the type of 'UNDEF' (line 64)
UNDEF_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 36), 'UNDEF', False)
keyword_677 = UNDEF_676
kwargs_678 = {'right': keyword_677}
# Getting the type of 'np' (line 64)
np_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 6), 'np', False)
# Obtaining the member 'interp' of a type (line 64)
interp_672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 6), np_671, 'interp')
# Calling interp(args, kwargs) (line 64)
interp_call_result_679 = invoke(stypy.reporting.localization.Localization(__file__, 64, 6), interp_672, *[float_673, xp_674, fp_675], **kwargs_678)

# Assigning a type to the variable 'r30' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'r30', interp_call_result_679)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
