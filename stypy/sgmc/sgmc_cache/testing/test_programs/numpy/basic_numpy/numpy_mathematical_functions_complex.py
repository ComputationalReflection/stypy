
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # https://docs.scipy.org/doc/numpy/reference/routines.math.html
2: 
3: import numpy as np
4: 
5: z = 2.1j + 3
6: 
7: # Handling complex numbers
8: r1 = np.angle(z)  # Return the angle of the complex argument.
9: r2 = np.real(z)  # Return the real part of the elements of the array.
10: r3 = np.imag(z)  # Return the imaginary part of the elements of the array.
11: r4 = np.conj(z)  # Return the complex conjugate, element-wise.
12: 
13: z = [2.1j + 3, 3.1j + 3, 4.1j + 3]
14: 
15: r5 = np.angle(z)  # Return the angle of the complex argument.
16: r6 = np.real(z)  # Return the real part of the elements of the array.
17: r7 = np.imag(z)  # Return the imaginary part of the elements of the array.
18: r8 = np.conj(z)  # Return the complex conjugate, element-wise.
19: 
20: 
21: # l = globals().copy()
22: # for v in l:
23: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_176 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_176) is not StypyTypeError):

    if (import_176 != 'pyd_module'):
        __import__(import_176)
        sys_modules_177 = sys.modules[import_176]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_177.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_176)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a BinOp to a Name (line 5):
complex_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'complex')
int_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 11), 'int')
# Applying the binary operator '+' (line 5)
result_add_180 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 4), '+', complex_178, int_179)

# Assigning a type to the variable 'z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'z', result_add_180)

# Assigning a Call to a Name (line 8):

# Call to angle(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'z' (line 8)
z_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'z', False)
# Processing the call keyword arguments (line 8)
kwargs_184 = {}
# Getting the type of 'np' (line 8)
np_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'np', False)
# Obtaining the member 'angle' of a type (line 8)
angle_182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), np_181, 'angle')
# Calling angle(args, kwargs) (line 8)
angle_call_result_185 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), angle_182, *[z_183], **kwargs_184)

# Assigning a type to the variable 'r1' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r1', angle_call_result_185)

# Assigning a Call to a Name (line 9):

# Call to real(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'z' (line 9)
z_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'z', False)
# Processing the call keyword arguments (line 9)
kwargs_189 = {}
# Getting the type of 'np' (line 9)
np_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'real' of a type (line 9)
real_187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_186, 'real')
# Calling real(args, kwargs) (line 9)
real_call_result_190 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), real_187, *[z_188], **kwargs_189)

# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', real_call_result_190)

# Assigning a Call to a Name (line 10):

# Call to imag(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'z' (line 10)
z_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'z', False)
# Processing the call keyword arguments (line 10)
kwargs_194 = {}
# Getting the type of 'np' (line 10)
np_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'np', False)
# Obtaining the member 'imag' of a type (line 10)
imag_192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), np_191, 'imag')
# Calling imag(args, kwargs) (line 10)
imag_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), imag_192, *[z_193], **kwargs_194)

# Assigning a type to the variable 'r3' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r3', imag_call_result_195)

# Assigning a Call to a Name (line 11):

# Call to conj(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'z' (line 11)
z_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'z', False)
# Processing the call keyword arguments (line 11)
kwargs_199 = {}
# Getting the type of 'np' (line 11)
np_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'conj' of a type (line 11)
conj_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_196, 'conj')
# Calling conj(args, kwargs) (line 11)
conj_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), conj_197, *[z_198], **kwargs_199)

# Assigning a type to the variable 'r4' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r4', conj_call_result_200)

# Assigning a List to a Name (line 13):

# Obtaining an instance of the builtin type 'list' (line 13)
list_201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
complex_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 5), 'complex')
int_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'int')
# Applying the binary operator '+' (line 13)
result_add_204 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 5), '+', complex_202, int_203)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_201, result_add_204)
# Adding element type (line 13)
complex_205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'complex')
int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'int')
# Applying the binary operator '+' (line 13)
result_add_207 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 15), '+', complex_205, int_206)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_201, result_add_207)
# Adding element type (line 13)
complex_208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'complex')
int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 32), 'int')
# Applying the binary operator '+' (line 13)
result_add_210 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 25), '+', complex_208, int_209)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), list_201, result_add_210)

# Assigning a type to the variable 'z' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'z', list_201)

# Assigning a Call to a Name (line 15):

# Call to angle(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'z' (line 15)
z_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'z', False)
# Processing the call keyword arguments (line 15)
kwargs_214 = {}
# Getting the type of 'np' (line 15)
np_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np', False)
# Obtaining the member 'angle' of a type (line 15)
angle_212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_211, 'angle')
# Calling angle(args, kwargs) (line 15)
angle_call_result_215 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), angle_212, *[z_213], **kwargs_214)

# Assigning a type to the variable 'r5' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r5', angle_call_result_215)

# Assigning a Call to a Name (line 16):

# Call to real(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'z' (line 16)
z_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 13), 'z', False)
# Processing the call keyword arguments (line 16)
kwargs_219 = {}
# Getting the type of 'np' (line 16)
np_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'np', False)
# Obtaining the member 'real' of a type (line 16)
real_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 5), np_216, 'real')
# Calling real(args, kwargs) (line 16)
real_call_result_220 = invoke(stypy.reporting.localization.Localization(__file__, 16, 5), real_217, *[z_218], **kwargs_219)

# Assigning a type to the variable 'r6' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r6', real_call_result_220)

# Assigning a Call to a Name (line 17):

# Call to imag(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of 'z' (line 17)
z_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 13), 'z', False)
# Processing the call keyword arguments (line 17)
kwargs_224 = {}
# Getting the type of 'np' (line 17)
np_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'imag' of a type (line 17)
imag_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_221, 'imag')
# Calling imag(args, kwargs) (line 17)
imag_call_result_225 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), imag_222, *[z_223], **kwargs_224)

# Assigning a type to the variable 'r7' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r7', imag_call_result_225)

# Assigning a Call to a Name (line 18):

# Call to conj(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of 'z' (line 18)
z_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'z', False)
# Processing the call keyword arguments (line 18)
kwargs_229 = {}
# Getting the type of 'np' (line 18)
np_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'np', False)
# Obtaining the member 'conj' of a type (line 18)
conj_227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), np_226, 'conj')
# Calling conj(args, kwargs) (line 18)
conj_call_result_230 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), conj_227, *[z_228], **kwargs_229)

# Assigning a type to the variable 'r8' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r8', conj_call_result_230)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
