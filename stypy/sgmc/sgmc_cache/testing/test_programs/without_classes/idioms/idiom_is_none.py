
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: import os
3: import sys
4: 
5: import types
6: 
7: def test_package(package, depth):
8:     if package is None:
9:         f = sys._getframe(1 + depth)
10:         package_path = f.f_locals.get('__file__', None)
11:         if package_path is None:
12:             raise AssertionError
13:         package_path = os.path.dirname(package_path)
14:         package_name = f.f_locals.get('__name__', None)
15:     elif isinstance(package, type(os)):
16:         package_path = os.path.dirname(package.__file__)
17:         package_name = getattr(package, '__name__', None)
18:     else:
19:         package_path = str(package)
20: 
21: test_package("os", 1)
22: test_package(os, 1)
23: test_package(None, 1)
24: 
25: r = "3"
26: 
27: if not r is None:
28:     r2 = 3
29: else:
30:     r2 = 3.0
31: 
32: rb = None
33: 
34: if not rb is None:
35:     r3 = 3
36: else:
37:     r3 = 3.0
38: 
39: 
40: 
41: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import types' statement (line 5)
import types

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'types', types, module_type_store)


@norecursion
def test_package(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_package'
    module_type_store = module_type_store.open_function_context('test_package', 7, 0, False)
    
    # Passed parameters checking function
    test_package.stypy_localization = localization
    test_package.stypy_type_of_self = None
    test_package.stypy_type_store = module_type_store
    test_package.stypy_function_name = 'test_package'
    test_package.stypy_param_names_list = ['package', 'depth']
    test_package.stypy_varargs_param_name = None
    test_package.stypy_kwargs_param_name = None
    test_package.stypy_call_defaults = defaults
    test_package.stypy_call_varargs = varargs
    test_package.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_package', ['package', 'depth'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_package', localization, ['package', 'depth'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_package(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 8)
    # Getting the type of 'package' (line 8)
    package_3081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 7), 'package')
    # Getting the type of 'None' (line 8)
    None_3082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 18), 'None')
    
    (may_be_3083, more_types_in_union_3084) = may_be_none(package_3081, None_3082)

    if may_be_3083:

        if more_types_in_union_3084:
            # Runtime conditional SSA (line 8)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 9):
        
        # Call to _getframe(...): (line 9)
        # Processing the call arguments (line 9)
        int_3087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 26), 'int')
        # Getting the type of 'depth' (line 9)
        depth_3088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 30), 'depth', False)
        # Applying the binary operator '+' (line 9)
        result_add_3089 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 26), '+', int_3087, depth_3088)
        
        # Processing the call keyword arguments (line 9)
        kwargs_3090 = {}
        # Getting the type of 'sys' (line 9)
        sys_3085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'sys', False)
        # Obtaining the member '_getframe' of a type (line 9)
        _getframe_3086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 12), sys_3085, '_getframe')
        # Calling _getframe(args, kwargs) (line 9)
        _getframe_call_result_3091 = invoke(stypy.reporting.localization.Localization(__file__, 9, 12), _getframe_3086, *[result_add_3089], **kwargs_3090)
        
        # Assigning a type to the variable 'f' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'f', _getframe_call_result_3091)
        
        # Assigning a Call to a Name (line 10):
        
        # Call to get(...): (line 10)
        # Processing the call arguments (line 10)
        str_3095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 38), 'str', '__file__')
        # Getting the type of 'None' (line 10)
        None_3096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 50), 'None', False)
        # Processing the call keyword arguments (line 10)
        kwargs_3097 = {}
        # Getting the type of 'f' (line 10)
        f_3092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 23), 'f', False)
        # Obtaining the member 'f_locals' of a type (line 10)
        f_locals_3093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 23), f_3092, 'f_locals')
        # Obtaining the member 'get' of a type (line 10)
        get_3094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 23), f_locals_3093, 'get')
        # Calling get(args, kwargs) (line 10)
        get_call_result_3098 = invoke(stypy.reporting.localization.Localization(__file__, 10, 23), get_3094, *[str_3095, None_3096], **kwargs_3097)
        
        # Assigning a type to the variable 'package_path' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'package_path', get_call_result_3098)
        
        # Type idiom detected: calculating its left and rigth part (line 11)
        # Getting the type of 'package_path' (line 11)
        package_path_3099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'package_path')
        # Getting the type of 'None' (line 11)
        None_3100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 27), 'None')
        
        (may_be_3101, more_types_in_union_3102) = may_be_none(package_path_3099, None_3100)

        if may_be_3101:

            if more_types_in_union_3102:
                # Runtime conditional SSA (line 11)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'AssertionError' (line 12)
            AssertionError_3103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'AssertionError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 12, 12), AssertionError_3103, 'raise parameter', BaseException)

            if more_types_in_union_3102:
                # SSA join for if statement (line 11)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 13):
        
        # Call to dirname(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'package_path' (line 13)
        package_path_3107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 39), 'package_path', False)
        # Processing the call keyword arguments (line 13)
        kwargs_3108 = {}
        # Getting the type of 'os' (line 13)
        os_3104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 13)
        path_3105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 23), os_3104, 'path')
        # Obtaining the member 'dirname' of a type (line 13)
        dirname_3106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 23), path_3105, 'dirname')
        # Calling dirname(args, kwargs) (line 13)
        dirname_call_result_3109 = invoke(stypy.reporting.localization.Localization(__file__, 13, 23), dirname_3106, *[package_path_3107], **kwargs_3108)
        
        # Assigning a type to the variable 'package_path' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'package_path', dirname_call_result_3109)
        
        # Assigning a Call to a Name (line 14):
        
        # Call to get(...): (line 14)
        # Processing the call arguments (line 14)
        str_3113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 38), 'str', '__name__')
        # Getting the type of 'None' (line 14)
        None_3114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 50), 'None', False)
        # Processing the call keyword arguments (line 14)
        kwargs_3115 = {}
        # Getting the type of 'f' (line 14)
        f_3110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'f', False)
        # Obtaining the member 'f_locals' of a type (line 14)
        f_locals_3111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 23), f_3110, 'f_locals')
        # Obtaining the member 'get' of a type (line 14)
        get_3112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 23), f_locals_3111, 'get')
        # Calling get(args, kwargs) (line 14)
        get_call_result_3116 = invoke(stypy.reporting.localization.Localization(__file__, 14, 23), get_3112, *[str_3113, None_3114], **kwargs_3115)
        
        # Assigning a type to the variable 'package_name' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'package_name', get_call_result_3116)

        if more_types_in_union_3084:
            # Runtime conditional SSA for else branch (line 8)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_3083) or more_types_in_union_3084):
        
        # Type idiom detected: calculating its left and rigth part (line 15)
        
        # Call to type(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'os' (line 15)
        os_3118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 34), 'os', False)
        # Processing the call keyword arguments (line 15)
        kwargs_3119 = {}
        # Getting the type of 'type' (line 15)
        type_3117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 29), 'type', False)
        # Calling type(args, kwargs) (line 15)
        type_call_result_3120 = invoke(stypy.reporting.localization.Localization(__file__, 15, 29), type_3117, *[os_3118], **kwargs_3119)
        
        # Getting the type of 'package' (line 15)
        package_3121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'package')
        
        (may_be_3122, more_types_in_union_3123) = may_be_subtype(type_call_result_3120, package_3121)

        if may_be_3122:

            if more_types_in_union_3123:
                # Runtime conditional SSA (line 15)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'package' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'package', remove_not_subtype_from_union(package_3121, type(os)))
            
            # Assigning a Call to a Name (line 16):
            
            # Call to dirname(...): (line 16)
            # Processing the call arguments (line 16)
            # Getting the type of 'package' (line 16)
            package_3127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 39), 'package', False)
            # Obtaining the member '__file__' of a type (line 16)
            file___3128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 39), package_3127, '__file__')
            # Processing the call keyword arguments (line 16)
            kwargs_3129 = {}
            # Getting the type of 'os' (line 16)
            os_3124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'os', False)
            # Obtaining the member 'path' of a type (line 16)
            path_3125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 23), os_3124, 'path')
            # Obtaining the member 'dirname' of a type (line 16)
            dirname_3126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 23), path_3125, 'dirname')
            # Calling dirname(args, kwargs) (line 16)
            dirname_call_result_3130 = invoke(stypy.reporting.localization.Localization(__file__, 16, 23), dirname_3126, *[file___3128], **kwargs_3129)
            
            # Assigning a type to the variable 'package_path' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'package_path', dirname_call_result_3130)
            
            # Assigning a Call to a Name (line 17):
            
            # Call to getattr(...): (line 17)
            # Processing the call arguments (line 17)
            # Getting the type of 'package' (line 17)
            package_3132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 31), 'package', False)
            str_3133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 40), 'str', '__name__')
            # Getting the type of 'None' (line 17)
            None_3134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 52), 'None', False)
            # Processing the call keyword arguments (line 17)
            kwargs_3135 = {}
            # Getting the type of 'getattr' (line 17)
            getattr_3131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'getattr', False)
            # Calling getattr(args, kwargs) (line 17)
            getattr_call_result_3136 = invoke(stypy.reporting.localization.Localization(__file__, 17, 23), getattr_3131, *[package_3132, str_3133, None_3134], **kwargs_3135)
            
            # Assigning a type to the variable 'package_name' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'package_name', getattr_call_result_3136)

            if more_types_in_union_3123:
                # Runtime conditional SSA for else branch (line 15)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_3122) or more_types_in_union_3123):
            # Assigning a type to the variable 'package' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'package', remove_subtype_from_union(package_3121, type(os)))
            
            # Assigning a Call to a Name (line 19):
            
            # Call to str(...): (line 19)
            # Processing the call arguments (line 19)
            # Getting the type of 'package' (line 19)
            package_3138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'package', False)
            # Processing the call keyword arguments (line 19)
            kwargs_3139 = {}
            # Getting the type of 'str' (line 19)
            str_3137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'str', False)
            # Calling str(args, kwargs) (line 19)
            str_call_result_3140 = invoke(stypy.reporting.localization.Localization(__file__, 19, 23), str_3137, *[package_3138], **kwargs_3139)
            
            # Assigning a type to the variable 'package_path' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'package_path', str_call_result_3140)

            if (may_be_3122 and more_types_in_union_3123):
                # SSA join for if statement (line 15)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_3083 and more_types_in_union_3084):
            # SSA join for if statement (line 8)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'test_package(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_package' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_3141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3141)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_package'
    return stypy_return_type_3141

# Assigning a type to the variable 'test_package' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'test_package', test_package)

# Call to test_package(...): (line 21)
# Processing the call arguments (line 21)
str_3143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 13), 'str', 'os')
int_3144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'int')
# Processing the call keyword arguments (line 21)
kwargs_3145 = {}
# Getting the type of 'test_package' (line 21)
test_package_3142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'test_package', False)
# Calling test_package(args, kwargs) (line 21)
test_package_call_result_3146 = invoke(stypy.reporting.localization.Localization(__file__, 21, 0), test_package_3142, *[str_3143, int_3144], **kwargs_3145)


# Call to test_package(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'os' (line 22)
os_3148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 13), 'os', False)
int_3149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
# Processing the call keyword arguments (line 22)
kwargs_3150 = {}
# Getting the type of 'test_package' (line 22)
test_package_3147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'test_package', False)
# Calling test_package(args, kwargs) (line 22)
test_package_call_result_3151 = invoke(stypy.reporting.localization.Localization(__file__, 22, 0), test_package_3147, *[os_3148, int_3149], **kwargs_3150)


# Call to test_package(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'None' (line 23)
None_3153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'None', False)
int_3154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'int')
# Processing the call keyword arguments (line 23)
kwargs_3155 = {}
# Getting the type of 'test_package' (line 23)
test_package_3152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'test_package', False)
# Calling test_package(args, kwargs) (line 23)
test_package_call_result_3156 = invoke(stypy.reporting.localization.Localization(__file__, 23, 0), test_package_3152, *[None_3153, int_3154], **kwargs_3155)


# Assigning a Str to a Name (line 25):
str_3157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', '3')
# Assigning a type to the variable 'r' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'r', str_3157)

# Type idiom detected: calculating its left and rigth part (line 27)
# Getting the type of 'r' (line 27)
r_3158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'r')
# Getting the type of 'None' (line 27)
None_3159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'None')

(may_be_3160, more_types_in_union_3161) = may_not_be_none(r_3158, None_3159)

if may_be_3160:

    if more_types_in_union_3161:
        # Runtime conditional SSA (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Num to a Name (line 28):
    int_3162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'int')
    # Assigning a type to the variable 'r2' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'r2', int_3162)

    if more_types_in_union_3161:
        # Runtime conditional SSA for else branch (line 27)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_3160) or more_types_in_union_3161):
    
    # Assigning a Num to a Name (line 30):
    float_3163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'float')
    # Assigning a type to the variable 'r2' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'r2', float_3163)

    if (may_be_3160 and more_types_in_union_3161):
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a Name to a Name (line 32):
# Getting the type of 'None' (line 32)
None_3164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 5), 'None')
# Assigning a type to the variable 'rb' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'rb', None_3164)

# Type idiom detected: calculating its left and rigth part (line 34)
# Getting the type of 'rb' (line 34)
rb_3165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 7), 'rb')
# Getting the type of 'None' (line 34)
None_3166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'None')

(may_be_3167, more_types_in_union_3168) = may_not_be_none(rb_3165, None_3166)

if may_be_3167:

    if more_types_in_union_3168:
        # Runtime conditional SSA (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Num to a Name (line 35):
    int_3169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'int')
    # Assigning a type to the variable 'r3' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'r3', int_3169)

    if more_types_in_union_3168:
        # Runtime conditional SSA for else branch (line 34)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_3167) or more_types_in_union_3168):
    
    # Assigning a Num to a Name (line 37):
    float_3170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'float')
    # Assigning a type to the variable 'r3' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'r3', float_3170)

    if (may_be_3167 and more_types_in_union_3168):
        # SSA join for if statement (line 34)
        module_type_store = module_type_store.join_ssa_context()




# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
