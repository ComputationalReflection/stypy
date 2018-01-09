
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.python-course.eu/numpy.php
2: 
3: import time
4: 
5: import numpy as np
6: 
7: size_of_vec = 1000
8: 
9: 
10: def pure_python_version():
11:     t1 = time.time()
12:     X = range(size_of_vec)
13:     Y = range(size_of_vec)
14:     Z = []
15:     for i in range(len(X)):
16:         Z.append(X[i] + Y[i])
17: 
18:     l = locals().copy()
19:     for v in l:
20:         print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
21:     print("\n\n")
22:     return time.time() - t1
23: 
24: 
25: def numpy_version():
26:     t1 = time.time()
27:     X = np.arange(size_of_vec)
28:     Y = np.arange(size_of_vec)
29:     Z = X + Y
30: 
31:     l = locals().copy()
32:     for v in l:
33:         print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
34:     print("\n\n")
35:     return time.time() - t1
36: 
37: 
38: t1 = pure_python_version()
39: t2 = numpy_version()
40: 
41: # l = globals().copy()
42: # for v in l:
43: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
44: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import time' statement (line 3)
import time

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_127 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_127) is not StypyTypeError):

    if (import_127 != 'pyd_module'):
        __import__(import_127)
        sys_modules_128 = sys.modules[import_127]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_128.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_127)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Num to a Name (line 7):
int_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'int')
# Assigning a type to the variable 'size_of_vec' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'size_of_vec', int_129)

@norecursion
def pure_python_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pure_python_version'
    module_type_store = module_type_store.open_function_context('pure_python_version', 10, 0, False)
    
    # Passed parameters checking function
    pure_python_version.stypy_localization = localization
    pure_python_version.stypy_type_of_self = None
    pure_python_version.stypy_type_store = module_type_store
    pure_python_version.stypy_function_name = 'pure_python_version'
    pure_python_version.stypy_param_names_list = []
    pure_python_version.stypy_varargs_param_name = None
    pure_python_version.stypy_kwargs_param_name = None
    pure_python_version.stypy_call_defaults = defaults
    pure_python_version.stypy_call_varargs = varargs
    pure_python_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pure_python_version', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pure_python_version', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pure_python_version(...)' code ##################

    
    # Assigning a Call to a Name (line 11):
    
    # Call to time(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_132 = {}
    # Getting the type of 'time' (line 11)
    time_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 9), 'time', False)
    # Obtaining the member 'time' of a type (line 11)
    time_131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 9), time_130, 'time')
    # Calling time(args, kwargs) (line 11)
    time_call_result_133 = invoke(stypy.reporting.localization.Localization(__file__, 11, 9), time_131, *[], **kwargs_132)
    
    # Assigning a type to the variable 't1' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 't1', time_call_result_133)
    
    # Assigning a Call to a Name (line 12):
    
    # Call to range(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'size_of_vec' (line 12)
    size_of_vec_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'size_of_vec', False)
    # Processing the call keyword arguments (line 12)
    kwargs_136 = {}
    # Getting the type of 'range' (line 12)
    range_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'range', False)
    # Calling range(args, kwargs) (line 12)
    range_call_result_137 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), range_134, *[size_of_vec_135], **kwargs_136)
    
    # Assigning a type to the variable 'X' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'X', range_call_result_137)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to range(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'size_of_vec' (line 13)
    size_of_vec_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'size_of_vec', False)
    # Processing the call keyword arguments (line 13)
    kwargs_140 = {}
    # Getting the type of 'range' (line 13)
    range_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'range', False)
    # Calling range(args, kwargs) (line 13)
    range_call_result_141 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), range_138, *[size_of_vec_139], **kwargs_140)
    
    # Assigning a type to the variable 'Y' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'Y', range_call_result_141)
    
    # Assigning a List to a Name (line 14):
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    
    # Assigning a type to the variable 'Z' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'Z', list_142)
    
    
    # Call to range(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to len(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'X' (line 15)
    X_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'X', False)
    # Processing the call keyword arguments (line 15)
    kwargs_146 = {}
    # Getting the type of 'len' (line 15)
    len_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'len', False)
    # Calling len(args, kwargs) (line 15)
    len_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 15, 19), len_144, *[X_145], **kwargs_146)
    
    # Processing the call keyword arguments (line 15)
    kwargs_148 = {}
    # Getting the type of 'range' (line 15)
    range_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'range', False)
    # Calling range(args, kwargs) (line 15)
    range_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 15, 13), range_143, *[len_call_result_147], **kwargs_148)
    
    # Testing the type of a for loop iterable (line 15)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 15, 4), range_call_result_149)
    # Getting the type of the for loop variable (line 15)
    for_loop_var_150 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 15, 4), range_call_result_149)
    # Assigning a type to the variable 'i' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'i', for_loop_var_150)
    # SSA begins for a for statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 16)
    i_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'i', False)
    # Getting the type of 'X' (line 16)
    X_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'X', False)
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 17), X_154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 16, 17), getitem___155, i_153)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 16)
    i_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 26), 'i', False)
    # Getting the type of 'Y' (line 16)
    Y_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'Y', False)
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 24), Y_158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_160 = invoke(stypy.reporting.localization.Localization(__file__, 16, 24), getitem___159, i_157)
    
    # Applying the binary operator '+' (line 16)
    result_add_161 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 17), '+', subscript_call_result_156, subscript_call_result_160)
    
    # Processing the call keyword arguments (line 16)
    kwargs_162 = {}
    # Getting the type of 'Z' (line 16)
    Z_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'Z', False)
    # Obtaining the member 'append' of a type (line 16)
    append_152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), Z_151, 'append')
    # Calling append(args, kwargs) (line 16)
    append_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), append_152, *[result_add_161], **kwargs_162)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 18):
    
    # Call to copy(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_168 = {}
    
    # Call to locals(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_165 = {}
    # Getting the type of 'locals' (line 18)
    locals_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'locals', False)
    # Calling locals(args, kwargs) (line 18)
    locals_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), locals_164, *[], **kwargs_165)
    
    # Obtaining the member 'copy' of a type (line 18)
    copy_167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), locals_call_result_166, 'copy')
    # Calling copy(args, kwargs) (line 18)
    copy_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), copy_167, *[], **kwargs_168)
    
    # Assigning a type to the variable 'l' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'l', copy_call_result_169)
    
    # Getting the type of 'l' (line 19)
    l_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'l')
    # Testing the type of a for loop iterable (line 19)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 4), l_170)
    # Getting the type of the for loop variable (line 19)
    for_loop_var_171 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 4), l_170)
    # Assigning a type to the variable 'v' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'v', for_loop_var_171)
    # SSA begins for a for statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    str_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'str', "'")
    # Getting the type of 'v' (line 20)
    v_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'v')
    # Applying the binary operator '+' (line 20)
    result_add_174 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 15), '+', str_172, v_173)
    
    str_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'str', "'")
    # Applying the binary operator '+' (line 20)
    result_add_176 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 23), '+', result_add_174, str_175)
    
    str_177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 31), 'str', ': instance_of_class_name("')
    # Applying the binary operator '+' (line 20)
    result_add_178 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 29), '+', result_add_176, str_177)
    
    
    # Call to type(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 20)
    v_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 70), 'v', False)
    # Getting the type of 'l' (line 20)
    l_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 68), 'l', False)
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 68), l_181, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_183 = invoke(stypy.reporting.localization.Localization(__file__, 20, 68), getitem___182, v_180)
    
    # Processing the call keyword arguments (line 20)
    kwargs_184 = {}
    # Getting the type of 'type' (line 20)
    type_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 63), 'type', False)
    # Calling type(args, kwargs) (line 20)
    type_call_result_185 = invoke(stypy.reporting.localization.Localization(__file__, 20, 63), type_179, *[subscript_call_result_183], **kwargs_184)
    
    # Obtaining the member '__name__' of a type (line 20)
    name___186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 63), type_call_result_185, '__name__')
    # Applying the binary operator '+' (line 20)
    result_add_187 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 61), '+', result_add_178, name___186)
    
    str_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 85), 'str', '"),')
    # Applying the binary operator '+' (line 20)
    result_add_189 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 83), '+', result_add_187, str_188)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    str_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'str', '\n\n')
    
    # Call to time(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_193 = {}
    # Getting the type of 'time' (line 22)
    time_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'time', False)
    # Obtaining the member 'time' of a type (line 22)
    time_192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), time_191, 'time')
    # Calling time(args, kwargs) (line 22)
    time_call_result_194 = invoke(stypy.reporting.localization.Localization(__file__, 22, 11), time_192, *[], **kwargs_193)
    
    # Getting the type of 't1' (line 22)
    t1_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 't1')
    # Applying the binary operator '-' (line 22)
    result_sub_196 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '-', time_call_result_194, t1_195)
    
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', result_sub_196)
    
    # ################# End of 'pure_python_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pure_python_version' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_197)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pure_python_version'
    return stypy_return_type_197

# Assigning a type to the variable 'pure_python_version' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'pure_python_version', pure_python_version)

@norecursion
def numpy_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'numpy_version'
    module_type_store = module_type_store.open_function_context('numpy_version', 25, 0, False)
    
    # Passed parameters checking function
    numpy_version.stypy_localization = localization
    numpy_version.stypy_type_of_self = None
    numpy_version.stypy_type_store = module_type_store
    numpy_version.stypy_function_name = 'numpy_version'
    numpy_version.stypy_param_names_list = []
    numpy_version.stypy_varargs_param_name = None
    numpy_version.stypy_kwargs_param_name = None
    numpy_version.stypy_call_defaults = defaults
    numpy_version.stypy_call_varargs = varargs
    numpy_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'numpy_version', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'numpy_version', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'numpy_version(...)' code ##################

    
    # Assigning a Call to a Name (line 26):
    
    # Call to time(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_200 = {}
    # Getting the type of 'time' (line 26)
    time_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'time', False)
    # Obtaining the member 'time' of a type (line 26)
    time_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 9), time_198, 'time')
    # Calling time(args, kwargs) (line 26)
    time_call_result_201 = invoke(stypy.reporting.localization.Localization(__file__, 26, 9), time_199, *[], **kwargs_200)
    
    # Assigning a type to the variable 't1' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 't1', time_call_result_201)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to arange(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'size_of_vec' (line 27)
    size_of_vec_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'size_of_vec', False)
    # Processing the call keyword arguments (line 27)
    kwargs_205 = {}
    # Getting the type of 'np' (line 27)
    np_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 27)
    arange_203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), np_202, 'arange')
    # Calling arange(args, kwargs) (line 27)
    arange_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), arange_203, *[size_of_vec_204], **kwargs_205)
    
    # Assigning a type to the variable 'X' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'X', arange_call_result_206)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to arange(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'size_of_vec' (line 28)
    size_of_vec_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'size_of_vec', False)
    # Processing the call keyword arguments (line 28)
    kwargs_210 = {}
    # Getting the type of 'np' (line 28)
    np_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 28)
    arange_208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), np_207, 'arange')
    # Calling arange(args, kwargs) (line 28)
    arange_call_result_211 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), arange_208, *[size_of_vec_209], **kwargs_210)
    
    # Assigning a type to the variable 'Y' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'Y', arange_call_result_211)
    
    # Assigning a BinOp to a Name (line 29):
    # Getting the type of 'X' (line 29)
    X_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'X')
    # Getting the type of 'Y' (line 29)
    Y_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'Y')
    # Applying the binary operator '+' (line 29)
    result_add_214 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 8), '+', X_212, Y_213)
    
    # Assigning a type to the variable 'Z' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'Z', result_add_214)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to copy(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_219 = {}
    
    # Call to locals(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_216 = {}
    # Getting the type of 'locals' (line 31)
    locals_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'locals', False)
    # Calling locals(args, kwargs) (line 31)
    locals_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), locals_215, *[], **kwargs_216)
    
    # Obtaining the member 'copy' of a type (line 31)
    copy_218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), locals_call_result_217, 'copy')
    # Calling copy(args, kwargs) (line 31)
    copy_call_result_220 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), copy_218, *[], **kwargs_219)
    
    # Assigning a type to the variable 'l' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'l', copy_call_result_220)
    
    # Getting the type of 'l' (line 32)
    l_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 13), 'l')
    # Testing the type of a for loop iterable (line 32)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 32, 4), l_221)
    # Getting the type of the for loop variable (line 32)
    for_loop_var_222 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 32, 4), l_221)
    # Assigning a type to the variable 'v' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'v', for_loop_var_222)
    # SSA begins for a for statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    str_223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'str', "'")
    # Getting the type of 'v' (line 33)
    v_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'v')
    # Applying the binary operator '+' (line 33)
    result_add_225 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 15), '+', str_223, v_224)
    
    str_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'str', "'")
    # Applying the binary operator '+' (line 33)
    result_add_227 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 23), '+', result_add_225, str_226)
    
    str_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'str', ': instance_of_class_name("')
    # Applying the binary operator '+' (line 33)
    result_add_229 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 29), '+', result_add_227, str_228)
    
    
    # Call to type(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 33)
    v_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 70), 'v', False)
    # Getting the type of 'l' (line 33)
    l_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 68), 'l', False)
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 68), l_232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 33, 68), getitem___233, v_231)
    
    # Processing the call keyword arguments (line 33)
    kwargs_235 = {}
    # Getting the type of 'type' (line 33)
    type_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 63), 'type', False)
    # Calling type(args, kwargs) (line 33)
    type_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 33, 63), type_230, *[subscript_call_result_234], **kwargs_235)
    
    # Obtaining the member '__name__' of a type (line 33)
    name___237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 63), type_call_result_236, '__name__')
    # Applying the binary operator '+' (line 33)
    result_add_238 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 61), '+', result_add_229, name___237)
    
    str_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 85), 'str', '"),')
    # Applying the binary operator '+' (line 33)
    result_add_240 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 83), '+', result_add_238, str_239)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    str_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 10), 'str', '\n\n')
    
    # Call to time(...): (line 35)
    # Processing the call keyword arguments (line 35)
    kwargs_244 = {}
    # Getting the type of 'time' (line 35)
    time_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'time', False)
    # Obtaining the member 'time' of a type (line 35)
    time_243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 11), time_242, 'time')
    # Calling time(args, kwargs) (line 35)
    time_call_result_245 = invoke(stypy.reporting.localization.Localization(__file__, 35, 11), time_243, *[], **kwargs_244)
    
    # Getting the type of 't1' (line 35)
    t1_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 't1')
    # Applying the binary operator '-' (line 35)
    result_sub_247 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 11), '-', time_call_result_245, t1_246)
    
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', result_sub_247)
    
    # ################# End of 'numpy_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'numpy_version' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'numpy_version'
    return stypy_return_type_248

# Assigning a type to the variable 'numpy_version' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy_version', numpy_version)

# Assigning a Call to a Name (line 38):

# Call to pure_python_version(...): (line 38)
# Processing the call keyword arguments (line 38)
kwargs_250 = {}
# Getting the type of 'pure_python_version' (line 38)
pure_python_version_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 5), 'pure_python_version', False)
# Calling pure_python_version(args, kwargs) (line 38)
pure_python_version_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 38, 5), pure_python_version_249, *[], **kwargs_250)

# Assigning a type to the variable 't1' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 't1', pure_python_version_call_result_251)

# Assigning a Call to a Name (line 39):

# Call to numpy_version(...): (line 39)
# Processing the call keyword arguments (line 39)
kwargs_253 = {}
# Getting the type of 'numpy_version' (line 39)
numpy_version_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 5), 'numpy_version', False)
# Calling numpy_version(args, kwargs) (line 39)
numpy_version_call_result_254 = invoke(stypy.reporting.localization.Localization(__file__, 39, 5), numpy_version_252, *[], **kwargs_253)

# Assigning a type to the variable 't2' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 't2', numpy_version_call_result_254)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
