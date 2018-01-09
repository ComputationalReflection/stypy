
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: r = "3"
2: 
3: if not r is None:
4:     r2 = 3
5: else:
6:     r2 = 3.0
7: 
8: rb = None
9: 
10: if rb is not None:
11:     r3 = 3
12: else:
13:     r3 = 3.0
14: 
15: rc = None
16: 
17: if rc == None:
18:     r4 = 3
19: else:
20:     r4 = 3.0
21: 
22: rd = None
23: 
24: if not rc == None:
25:     r5 = 3
26: else:
27:     r5 = 3.0

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 1):
str_3171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 4), 'str', '3')
# Assigning a type to the variable 'r' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'r', str_3171)

# Type idiom detected: calculating its left and rigth part (line 3)
# Getting the type of 'r' (line 3)
r_3172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 7), 'r')
# Getting the type of 'None' (line 3)
None_3173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 12), 'None')

(may_be_3174, more_types_in_union_3175) = may_not_be_none(r_3172, None_3173)

if may_be_3174:

    if more_types_in_union_3175:
        # Runtime conditional SSA (line 3)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Num to a Name (line 4):
    int_3176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 9), 'int')
    # Assigning a type to the variable 'r2' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'r2', int_3176)

    if more_types_in_union_3175:
        # Runtime conditional SSA for else branch (line 3)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_3174) or more_types_in_union_3175):
    
    # Assigning a Num to a Name (line 6):
    float_3177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 9), 'float')
    # Assigning a type to the variable 'r2' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'r2', float_3177)

    if (may_be_3174 and more_types_in_union_3175):
        # SSA join for if statement (line 3)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a Name to a Name (line 8):
# Getting the type of 'None' (line 8)
None_3178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'None')
# Assigning a type to the variable 'rb' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'rb', None_3178)

# Type idiom detected: calculating its left and rigth part (line 10)
# Getting the type of 'rb' (line 10)
rb_3179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'rb')
# Getting the type of 'None' (line 10)
None_3180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'None')

(may_be_3181, more_types_in_union_3182) = may_not_be_none(rb_3179, None_3180)

if may_be_3181:

    if more_types_in_union_3182:
        # Runtime conditional SSA (line 10)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Num to a Name (line 11):
    int_3183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 9), 'int')
    # Assigning a type to the variable 'r3' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'r3', int_3183)

    if more_types_in_union_3182:
        # Runtime conditional SSA for else branch (line 10)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_3181) or more_types_in_union_3182):
    
    # Assigning a Num to a Name (line 13):
    float_3184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'float')
    # Assigning a type to the variable 'r3' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'r3', float_3184)

    if (may_be_3181 and more_types_in_union_3182):
        # SSA join for if statement (line 10)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a Name to a Name (line 15):
# Getting the type of 'None' (line 15)
None_3185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'None')
# Assigning a type to the variable 'rc' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'rc', None_3185)

# Type idiom detected: calculating its left and rigth part (line 17)
# Getting the type of 'rc' (line 17)
rc_3186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'rc')
# Getting the type of 'None' (line 17)
None_3187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 9), 'None')

(may_be_3188, more_types_in_union_3189) = may_be_none(rc_3186, None_3187)

if may_be_3188:

    if more_types_in_union_3189:
        # Runtime conditional SSA (line 17)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Num to a Name (line 18):
    int_3190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 9), 'int')
    # Assigning a type to the variable 'r4' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'r4', int_3190)

    if more_types_in_union_3189:
        # Runtime conditional SSA for else branch (line 17)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_3188) or more_types_in_union_3189):
    
    # Assigning a Num to a Name (line 20):
    float_3191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'float')
    # Assigning a type to the variable 'r4' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'r4', float_3191)

    if (may_be_3188 and more_types_in_union_3189):
        # SSA join for if statement (line 17)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a Name to a Name (line 22):
# Getting the type of 'None' (line 22)
None_3192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'None')
# Assigning a type to the variable 'rd' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'rd', None_3192)

# Type idiom detected: calculating its left and rigth part (line 24)
# Getting the type of 'rc' (line 24)
rc_3193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'rc')
# Getting the type of 'None' (line 24)
None_3194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'None')

(may_be_3195, more_types_in_union_3196) = may_not_be_none(rc_3193, None_3194)

if may_be_3195:

    if more_types_in_union_3196:
        # Runtime conditional SSA (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Num to a Name (line 25):
    int_3197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'int')
    # Assigning a type to the variable 'r5' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'r5', int_3197)

    if more_types_in_union_3196:
        # Runtime conditional SSA for else branch (line 24)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_3195) or more_types_in_union_3196):
    
    # Assigning a Num to a Name (line 27):
    float_3198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'float')
    # Assigning a type to the variable 'r5' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'r5', float_3198)

    if (may_be_3195 and more_types_in_union_3196):
        # SSA join for if statement (line 24)
        module_type_store = module_type_store.join_ssa_context()




# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
