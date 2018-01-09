
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: class C:
4:     def __init__(self):
5:         pass
6: 
7:     r = "hi"
8: 
9: 
10: C.r = 5
11: 
12: c = C()
13: 
14: x = c.r == 5
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'C' class

class C:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 4, 4, False)
        # Assigning a type to the variable 'self' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'C' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'C', C)

# Assigning a Str to a Name (line 7):
str_1218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 8), 'str', 'hi')
# Getting the type of 'C'
C_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'r' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1219, 'r', str_1218)

# Assigning a Num to a Attribute (line 10):
int_1220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 6), 'int')
# Getting the type of 'C' (line 10)
C_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'C')
# Setting the type of the member 'r' of a type (line 10)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 0), C_1221, 'r', int_1220)

# Assigning a Call to a Name (line 12):

# Call to C(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_1223 = {}
# Getting the type of 'C' (line 12)
C_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'C', False)
# Calling C(args, kwargs) (line 12)
C_call_result_1224 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), C_1222, *[], **kwargs_1223)

# Assigning a type to the variable 'c' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'c', C_call_result_1224)

# Assigning a Compare to a Name (line 14):

# Getting the type of 'c' (line 14)
c_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'c')
# Obtaining the member 'r' of a type (line 14)
r_1226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), c_1225, 'r')
int_1227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'int')
# Applying the binary operator '==' (line 14)
result_eq_1228 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 4), '==', r_1226, int_1227)

# Assigning a type to the variable 'x' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'x', result_eq_1228)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
