
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: from math import fabs
3: 
4: 
5: try:
6:     from math import kos, sin, fabs
7: except Exception as ex:
8:     print ex
9: 
10:     from math import cos
11: 
12: r = cos(30)
13: r2 = sin(45)
14: print r2
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from math import fabs' statement (line 2)
try:
    from math import fabs

except:
    fabs = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'math', None, module_type_store, ['fabs'], [fabs])



# SSA begins for try-except statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))

# 'from math import kos, sin, fabs' statement (line 6)
try:
    from math import kos, sin, fabs

except:
    kos = UndefinedType
    sin = UndefinedType
    fabs = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'math', None, module_type_store, ['kos', 'sin', 'fabs'], [kos, sin, fabs])

# SSA branch for the except part of a try statement (line 5)
# SSA branch for the except 'Exception' branch of a try statement (line 5)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'Exception' (line 7)
Exception_1 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 7), 'Exception')
# Assigning a type to the variable 'ex' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'ex', Exception_1)
# Getting the type of 'ex' (line 8)
ex_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'ex')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))

# 'from math import cos' statement (line 10)
try:
    from math import cos

except:
    cos = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'math', None, module_type_store, ['cos'], [cos])

# SSA join for try-except statement (line 5)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 12):

# Call to cos(...): (line 12)
# Processing the call arguments (line 12)
int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 8), 'int')
# Processing the call keyword arguments (line 12)
kwargs_5 = {}
# Getting the type of 'cos' (line 12)
cos_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'cos', False)
# Calling cos(args, kwargs) (line 12)
cos_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), cos_3, *[int_4], **kwargs_5)

# Assigning a type to the variable 'r' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r', cos_call_result_6)

# Assigning a Call to a Name (line 13):

# Call to sin(...): (line 13)
# Processing the call arguments (line 13)
int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'int')
# Processing the call keyword arguments (line 13)
kwargs_9 = {}
# Getting the type of 'sin' (line 13)
sin_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'sin', False)
# Calling sin(args, kwargs) (line 13)
sin_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), sin_7, *[int_8], **kwargs_9)

# Assigning a type to the variable 'r2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r2', sin_call_result_10)
# Getting the type of 'r2' (line 14)
r2_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 6), 'r2')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
