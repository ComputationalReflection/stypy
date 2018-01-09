
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: theInt = 3
2: theStr = "hi"
3: theBool = True
4: theComplex = complex(1, 2)
5: if True:
6:     union = 3
7: else:
8:     union = "hi"
9: 
10: def idiom(a):
11:     '''
12:     if type(a) is int: return 3
13:     if type(a) is str: return "hi"
14:     return True
15:     '''
16:     if type(a) is int:
17:         result = 3
18:     else:
19:         if type(a) is str:
20:             result = "hi"
21:         else:
22:             result = True
23:     return result
24: 
25: bigUnion = 3 if True else "a" if False else True
26: intOrBool = int() if True else False
27: intStrComplex = int() if True else str() if False else complex()
28: 
29: r = idiom(theInt)
30: r2 = idiom(theStr)
31: r3 = idiom(union)
32: r4 = idiom(theBool)
33: r5 = idiom(theComplex)
34: r6 = idiom(intOrBool)
35: r7 = idiom(bigUnion)
36: r8 = idiom(intStrComplex)
37: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_3199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'int')
# Assigning a type to the variable 'theInt' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'theInt', int_3199)

# Assigning a Str to a Name (line 2):
str_3200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'str', 'hi')
# Assigning a type to the variable 'theStr' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'theStr', str_3200)

# Assigning a Name to a Name (line 3):
# Getting the type of 'True' (line 3)
True_3201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 10), 'True')
# Assigning a type to the variable 'theBool' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'theBool', True_3201)

# Assigning a Call to a Name (line 4):

# Call to complex(...): (line 4)
# Processing the call arguments (line 4)
int_3203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 21), 'int')
int_3204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 24), 'int')
# Processing the call keyword arguments (line 4)
kwargs_3205 = {}
# Getting the type of 'complex' (line 4)
complex_3202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 13), 'complex', False)
# Calling complex(args, kwargs) (line 4)
complex_call_result_3206 = invoke(stypy.reporting.localization.Localization(__file__, 4, 13), complex_3202, *[int_3203, int_3204], **kwargs_3205)

# Assigning a type to the variable 'theComplex' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'theComplex', complex_call_result_3206)

# Getting the type of 'True' (line 5)
True_3207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 3), 'True')
# Testing the type of an if condition (line 5)
if_condition_3208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 5, 0), True_3207)
# Assigning a type to the variable 'if_condition_3208' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'if_condition_3208', if_condition_3208)
# SSA begins for if statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 6):
int_3209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'int')
# Assigning a type to the variable 'union' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'union', int_3209)
# SSA branch for the else part of an if statement (line 5)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 8):
str_3210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'str', 'hi')
# Assigning a type to the variable 'union' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'union', str_3210)
# SSA join for if statement (line 5)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def idiom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idiom'
    module_type_store = module_type_store.open_function_context('idiom', 10, 0, False)
    
    # Passed parameters checking function
    idiom.stypy_localization = localization
    idiom.stypy_type_of_self = None
    idiom.stypy_type_store = module_type_store
    idiom.stypy_function_name = 'idiom'
    idiom.stypy_param_names_list = ['a']
    idiom.stypy_varargs_param_name = None
    idiom.stypy_kwargs_param_name = None
    idiom.stypy_call_defaults = defaults
    idiom.stypy_call_varargs = varargs
    idiom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idiom', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idiom', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idiom(...)' code ##################

    str_3211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', '\n    if type(a) is int: return 3\n    if type(a) is str: return "hi"\n    return True\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 16)
    # Getting the type of 'a' (line 16)
    a_3212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'a')
    # Getting the type of 'int' (line 16)
    int_3213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'int')
    
    (may_be_3214, more_types_in_union_3215) = may_be_type(a_3212, int_3213)

    if may_be_3214:

        if more_types_in_union_3215:
            # Runtime conditional SSA (line 16)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'a', int_3213())
        
        # Assigning a Num to a Name (line 17):
        int_3216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'int')
        # Assigning a type to the variable 'result' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'result', int_3216)

        if more_types_in_union_3215:
            # Runtime conditional SSA for else branch (line 16)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_3214) or more_types_in_union_3215):
        # Getting the type of 'a' (line 16)
        a_3217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'a')
        # Assigning a type to the variable 'a' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'a', remove_type_from_union(a_3217, int_3213))
        
        # Type idiom detected: calculating its left and rigth part (line 19)
        # Getting the type of 'a' (line 19)
        a_3218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'a')
        # Getting the type of 'str' (line 19)
        str_3219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'str')
        
        (may_be_3220, more_types_in_union_3221) = may_be_type(a_3218, str_3219)

        if may_be_3220:

            if more_types_in_union_3221:
                # Runtime conditional SSA (line 19)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'a' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'a', str_3219())
            
            # Assigning a Str to a Name (line 20):
            str_3222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'str', 'hi')
            # Assigning a type to the variable 'result' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'result', str_3222)

            if more_types_in_union_3221:
                # Runtime conditional SSA for else branch (line 19)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_3220) or more_types_in_union_3221):
            # Getting the type of 'a' (line 19)
            a_3223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'a')
            # Assigning a type to the variable 'a' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'a', remove_type_from_union(a_3223, str_3219))
            
            # Assigning a Name to a Name (line 22):
            # Getting the type of 'True' (line 22)
            True_3224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'True')
            # Assigning a type to the variable 'result' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'result', True_3224)

            if (may_be_3220 and more_types_in_union_3221):
                # SSA join for if statement (line 19)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_3214 and more_types_in_union_3215):
            # SSA join for if statement (line 16)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'result' (line 23)
    result_3225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', result_3225)
    
    # ################# End of 'idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_3226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3226)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idiom'
    return stypy_return_type_3226

# Assigning a type to the variable 'idiom' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'idiom', idiom)

# Assigning a IfExp to a Name (line 25):

# Getting the type of 'True' (line 25)
True_3227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'True')
# Testing the type of an if expression (line 25)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 11), True_3227)
# SSA begins for if expression (line 25)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
int_3228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'int')
# SSA branch for the else part of an if expression (line 25)
module_type_store.open_ssa_branch('if expression else')

# Getting the type of 'False' (line 25)
False_3229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 33), 'False')
# Testing the type of an if expression (line 25)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 26), False_3229)
# SSA begins for if expression (line 25)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
str_3230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'str', 'a')
# SSA branch for the else part of an if expression (line 25)
module_type_store.open_ssa_branch('if expression else')
# Getting the type of 'True' (line 25)
True_3231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 44), 'True')
# SSA join for if expression (line 25)
module_type_store = module_type_store.join_ssa_context()
if_exp_3232 = union_type.UnionType.add(str_3230, True_3231)

# SSA join for if expression (line 25)
module_type_store = module_type_store.join_ssa_context()
if_exp_3233 = union_type.UnionType.add(int_3228, if_exp_3232)

# Assigning a type to the variable 'bigUnion' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'bigUnion', if_exp_3233)

# Assigning a IfExp to a Name (line 26):

# Getting the type of 'True' (line 26)
True_3234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'True')
# Testing the type of an if expression (line 26)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 12), True_3234)
# SSA begins for if expression (line 26)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')

# Call to int(...): (line 26)
# Processing the call keyword arguments (line 26)
kwargs_3236 = {}
# Getting the type of 'int' (line 26)
int_3235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'int', False)
# Calling int(args, kwargs) (line 26)
int_call_result_3237 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), int_3235, *[], **kwargs_3236)

# SSA branch for the else part of an if expression (line 26)
module_type_store.open_ssa_branch('if expression else')
# Getting the type of 'False' (line 26)
False_3238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'False')
# SSA join for if expression (line 26)
module_type_store = module_type_store.join_ssa_context()
if_exp_3239 = union_type.UnionType.add(int_call_result_3237, False_3238)

# Assigning a type to the variable 'intOrBool' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'intOrBool', if_exp_3239)

# Assigning a IfExp to a Name (line 27):

# Getting the type of 'True' (line 27)
True_3240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'True')
# Testing the type of an if expression (line 27)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 16), True_3240)
# SSA begins for if expression (line 27)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')

# Call to int(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_3242 = {}
# Getting the type of 'int' (line 27)
int_3241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'int', False)
# Calling int(args, kwargs) (line 27)
int_call_result_3243 = invoke(stypy.reporting.localization.Localization(__file__, 27, 16), int_3241, *[], **kwargs_3242)

# SSA branch for the else part of an if expression (line 27)
module_type_store.open_ssa_branch('if expression else')

# Getting the type of 'False' (line 27)
False_3244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 44), 'False')
# Testing the type of an if expression (line 27)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 35), False_3244)
# SSA begins for if expression (line 27)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')

# Call to str(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_3246 = {}
# Getting the type of 'str' (line 27)
str_3245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 35), 'str', False)
# Calling str(args, kwargs) (line 27)
str_call_result_3247 = invoke(stypy.reporting.localization.Localization(__file__, 27, 35), str_3245, *[], **kwargs_3246)

# SSA branch for the else part of an if expression (line 27)
module_type_store.open_ssa_branch('if expression else')

# Call to complex(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_3249 = {}
# Getting the type of 'complex' (line 27)
complex_3248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 55), 'complex', False)
# Calling complex(args, kwargs) (line 27)
complex_call_result_3250 = invoke(stypy.reporting.localization.Localization(__file__, 27, 55), complex_3248, *[], **kwargs_3249)

# SSA join for if expression (line 27)
module_type_store = module_type_store.join_ssa_context()
if_exp_3251 = union_type.UnionType.add(str_call_result_3247, complex_call_result_3250)

# SSA join for if expression (line 27)
module_type_store = module_type_store.join_ssa_context()
if_exp_3252 = union_type.UnionType.add(int_call_result_3243, if_exp_3251)

# Assigning a type to the variable 'intStrComplex' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'intStrComplex', if_exp_3252)

# Assigning a Call to a Name (line 29):

# Call to idiom(...): (line 29)
# Processing the call arguments (line 29)
# Getting the type of 'theInt' (line 29)
theInt_3254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'theInt', False)
# Processing the call keyword arguments (line 29)
kwargs_3255 = {}
# Getting the type of 'idiom' (line 29)
idiom_3253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'idiom', False)
# Calling idiom(args, kwargs) (line 29)
idiom_call_result_3256 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), idiom_3253, *[theInt_3254], **kwargs_3255)

# Assigning a type to the variable 'r' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'r', idiom_call_result_3256)

# Assigning a Call to a Name (line 30):

# Call to idiom(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of 'theStr' (line 30)
theStr_3258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'theStr', False)
# Processing the call keyword arguments (line 30)
kwargs_3259 = {}
# Getting the type of 'idiom' (line 30)
idiom_3257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 5), 'idiom', False)
# Calling idiom(args, kwargs) (line 30)
idiom_call_result_3260 = invoke(stypy.reporting.localization.Localization(__file__, 30, 5), idiom_3257, *[theStr_3258], **kwargs_3259)

# Assigning a type to the variable 'r2' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'r2', idiom_call_result_3260)

# Assigning a Call to a Name (line 31):

# Call to idiom(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of 'union' (line 31)
union_3262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'union', False)
# Processing the call keyword arguments (line 31)
kwargs_3263 = {}
# Getting the type of 'idiom' (line 31)
idiom_3261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 5), 'idiom', False)
# Calling idiom(args, kwargs) (line 31)
idiom_call_result_3264 = invoke(stypy.reporting.localization.Localization(__file__, 31, 5), idiom_3261, *[union_3262], **kwargs_3263)

# Assigning a type to the variable 'r3' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'r3', idiom_call_result_3264)

# Assigning a Call to a Name (line 32):

# Call to idiom(...): (line 32)
# Processing the call arguments (line 32)
# Getting the type of 'theBool' (line 32)
theBool_3266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'theBool', False)
# Processing the call keyword arguments (line 32)
kwargs_3267 = {}
# Getting the type of 'idiom' (line 32)
idiom_3265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 5), 'idiom', False)
# Calling idiom(args, kwargs) (line 32)
idiom_call_result_3268 = invoke(stypy.reporting.localization.Localization(__file__, 32, 5), idiom_3265, *[theBool_3266], **kwargs_3267)

# Assigning a type to the variable 'r4' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'r4', idiom_call_result_3268)

# Assigning a Call to a Name (line 33):

# Call to idiom(...): (line 33)
# Processing the call arguments (line 33)
# Getting the type of 'theComplex' (line 33)
theComplex_3270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'theComplex', False)
# Processing the call keyword arguments (line 33)
kwargs_3271 = {}
# Getting the type of 'idiom' (line 33)
idiom_3269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 5), 'idiom', False)
# Calling idiom(args, kwargs) (line 33)
idiom_call_result_3272 = invoke(stypy.reporting.localization.Localization(__file__, 33, 5), idiom_3269, *[theComplex_3270], **kwargs_3271)

# Assigning a type to the variable 'r5' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'r5', idiom_call_result_3272)

# Assigning a Call to a Name (line 34):

# Call to idiom(...): (line 34)
# Processing the call arguments (line 34)
# Getting the type of 'intOrBool' (line 34)
intOrBool_3274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'intOrBool', False)
# Processing the call keyword arguments (line 34)
kwargs_3275 = {}
# Getting the type of 'idiom' (line 34)
idiom_3273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 5), 'idiom', False)
# Calling idiom(args, kwargs) (line 34)
idiom_call_result_3276 = invoke(stypy.reporting.localization.Localization(__file__, 34, 5), idiom_3273, *[intOrBool_3274], **kwargs_3275)

# Assigning a type to the variable 'r6' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'r6', idiom_call_result_3276)

# Assigning a Call to a Name (line 35):

# Call to idiom(...): (line 35)
# Processing the call arguments (line 35)
# Getting the type of 'bigUnion' (line 35)
bigUnion_3278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'bigUnion', False)
# Processing the call keyword arguments (line 35)
kwargs_3279 = {}
# Getting the type of 'idiom' (line 35)
idiom_3277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 5), 'idiom', False)
# Calling idiom(args, kwargs) (line 35)
idiom_call_result_3280 = invoke(stypy.reporting.localization.Localization(__file__, 35, 5), idiom_3277, *[bigUnion_3278], **kwargs_3279)

# Assigning a type to the variable 'r7' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'r7', idiom_call_result_3280)

# Assigning a Call to a Name (line 36):

# Call to idiom(...): (line 36)
# Processing the call arguments (line 36)
# Getting the type of 'intStrComplex' (line 36)
intStrComplex_3282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'intStrComplex', False)
# Processing the call keyword arguments (line 36)
kwargs_3283 = {}
# Getting the type of 'idiom' (line 36)
idiom_3281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 5), 'idiom', False)
# Calling idiom(args, kwargs) (line 36)
idiom_call_result_3284 = invoke(stypy.reporting.localization.Localization(__file__, 36, 5), idiom_3281, *[intStrComplex_3282], **kwargs_3283)

# Assigning a type to the variable 'r8' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'r8', idiom_call_result_3284)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
