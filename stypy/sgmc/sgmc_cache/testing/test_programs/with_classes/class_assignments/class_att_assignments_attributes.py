
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: class Argsort:
3:     __doc__ = "argsort doc"
4: 
5: class ndarray:
6:     argsort = Argsort()
7: 
8: class C:
9:     argsort = Argsort()
10: 
11:     def __init__(self):
12:         pass
13: 
14:     argsort.__doc__ = ndarray.argsort.__doc__
15:     argsort.__doc__.__doc__ = ndarray.argsort.__doc__
16: 
17: c = C()
18: 
19: x = c.argsort.__doc__
20: y = c.argsort.__doc__.capitalize()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Argsort' class

class Argsort:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 2, 0, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Argsort.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Argsort' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'Argsort', Argsort)

# Assigning a Str to a Name (line 3):
str_1229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 14), 'str', 'argsort doc')
# Getting the type of 'Argsort'
Argsort_1230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Argsort')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Argsort_1230, '__doc__', str_1229)
# Declaration of the 'ndarray' class

class ndarray:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 5, 0, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ndarray.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ndarray' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'ndarray', ndarray)

# Assigning a Call to a Name (line 6):

# Call to Argsort(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_1232 = {}
# Getting the type of 'Argsort' (line 6)
Argsort_1231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'Argsort', False)
# Calling Argsort(args, kwargs) (line 6)
Argsort_call_result_1233 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), Argsort_1231, *[], **kwargs_1232)

# Getting the type of 'ndarray'
ndarray_1234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ndarray')
# Setting the type of the member 'argsort' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ndarray_1234, 'argsort', Argsort_call_result_1233)
# Declaration of the 'C' class

class C:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
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


# Assigning a type to the variable 'C' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'C', C)

# Assigning a Call to a Name (line 9):

# Call to Argsort(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_1236 = {}
# Getting the type of 'Argsort' (line 9)
Argsort_1235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 14), 'Argsort', False)
# Calling Argsort(args, kwargs) (line 9)
Argsort_call_result_1237 = invoke(stypy.reporting.localization.Localization(__file__, 9, 14), Argsort_1235, *[], **kwargs_1236)

# Getting the type of 'C'
C_1238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'argsort' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1238, 'argsort', Argsort_call_result_1237)

# Assigning a Attribute to a Attribute (line 14):
# Getting the type of 'ndarray' (line 14)
ndarray_1239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'ndarray')
# Obtaining the member 'argsort' of a type (line 14)
argsort_1240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 22), ndarray_1239, 'argsort')
# Obtaining the member '__doc__' of a type (line 14)
doc___1241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 22), argsort_1240, '__doc__')
# Getting the type of 'C'
C_1242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'argsort' of a type
argsort_1243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1242, 'argsort')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), argsort_1243, '__doc__', doc___1241)

# Assigning a Attribute to a Attribute (line 15):
# Getting the type of 'ndarray' (line 15)
ndarray_1244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 30), 'ndarray')
# Obtaining the member 'argsort' of a type (line 15)
argsort_1245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 30), ndarray_1244, 'argsort')
# Obtaining the member '__doc__' of a type (line 15)
doc___1246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 30), argsort_1245, '__doc__')
# Getting the type of 'C'
C_1247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'argsort' of a type
argsort_1248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1247, 'argsort')
# Obtaining the member '__doc__' of a type
doc___1249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), argsort_1248, '__doc__')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), doc___1249, '__doc__', doc___1246)

# Assigning a Call to a Name (line 17):

# Call to C(...): (line 17)
# Processing the call keyword arguments (line 17)
kwargs_1251 = {}
# Getting the type of 'C' (line 17)
C_1250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'C', False)
# Calling C(args, kwargs) (line 17)
C_call_result_1252 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), C_1250, *[], **kwargs_1251)

# Assigning a type to the variable 'c' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'c', C_call_result_1252)

# Assigning a Attribute to a Name (line 19):
# Getting the type of 'c' (line 19)
c_1253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'c')
# Obtaining the member 'argsort' of a type (line 19)
argsort_1254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), c_1253, 'argsort')
# Obtaining the member '__doc__' of a type (line 19)
doc___1255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), argsort_1254, '__doc__')
# Assigning a type to the variable 'x' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'x', doc___1255)

# Assigning a Call to a Name (line 20):

# Call to capitalize(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_1260 = {}
# Getting the type of 'c' (line 20)
c_1256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'c', False)
# Obtaining the member 'argsort' of a type (line 20)
argsort_1257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), c_1256, 'argsort')
# Obtaining the member '__doc__' of a type (line 20)
doc___1258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), argsort_1257, '__doc__')
# Obtaining the member 'capitalize' of a type (line 20)
capitalize_1259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), doc___1258, 'capitalize')
# Calling capitalize(args, kwargs) (line 20)
capitalize_call_result_1261 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), capitalize_1259, *[], **kwargs_1260)

# Assigning a type to the variable 'y' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'y', capitalize_call_result_1261)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
