
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: _mapper = [(1, 4, 'a'), (2, 5, 'b'), (3, 6, 'c')]
3: 
4: 
5: class Foo:
6:     (_defaulttype, _defaultfunc, _defaultfill) = zip(*_mapper)
7: 
8: f = Foo()
9: 
10: r1 = f._defaulttype
11: r2 = f._defaultfunc
12: r3 = f._defaultfill
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 2):

# Assigning a List to a Name (line 2):

# Obtaining an instance of the builtin type 'list' (line 2)
list_2408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_2409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
int_2410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 12), tuple_2409, int_2410)
# Adding element type (line 2)
int_2411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 12), tuple_2409, int_2411)
# Adding element type (line 2)
str_2412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 18), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 12), tuple_2409, str_2412)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_2408, tuple_2409)
# Adding element type (line 2)

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_2413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
int_2414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 25), tuple_2413, int_2414)
# Adding element type (line 2)
int_2415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 25), tuple_2413, int_2415)
# Adding element type (line 2)
str_2416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 31), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 25), tuple_2413, str_2416)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_2408, tuple_2413)
# Adding element type (line 2)

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_2417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
int_2418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 38), tuple_2417, int_2418)
# Adding element type (line 2)
int_2419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 38), tuple_2417, int_2419)
# Adding element type (line 2)
str_2420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 44), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 38), tuple_2417, str_2420)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_2408, tuple_2417)

# Assigning a type to the variable '_mapper' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '_mapper', list_2408)
# Declaration of the 'Foo' class

class Foo:
    
    # Assigning a Call to a Tuple (line 6):
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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Foo' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Foo', Foo)

# Assigning a Subscript to a Name (line 6):

# Obtaining the type of the subscript
int_2421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'int')

# Call to zip(...): (line 6)
# Getting the type of '_mapper' (line 6)
_mapper_2423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 54), '_mapper', False)
# Processing the call keyword arguments (line 6)
kwargs_2424 = {}
# Getting the type of 'zip' (line 6)
zip_2422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 49), 'zip', False)
# Calling zip(args, kwargs) (line 6)
zip_call_result_2425 = invoke(stypy.reporting.localization.Localization(__file__, 6, 49), zip_2422, *[_mapper_2423], **kwargs_2424)

# Obtaining the member '__getitem__' of a type (line 6)
getitem___2426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), zip_call_result_2425, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_2427 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), getitem___2426, int_2421)

# Getting the type of 'Foo'
Foo_2428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member 'tuple_var_assignment_2405' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_2428, 'tuple_var_assignment_2405', subscript_call_result_2427)

# Assigning a Subscript to a Name (line 6):

# Obtaining the type of the subscript
int_2429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'int')

# Call to zip(...): (line 6)
# Getting the type of '_mapper' (line 6)
_mapper_2431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 54), '_mapper', False)
# Processing the call keyword arguments (line 6)
kwargs_2432 = {}
# Getting the type of 'zip' (line 6)
zip_2430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 49), 'zip', False)
# Calling zip(args, kwargs) (line 6)
zip_call_result_2433 = invoke(stypy.reporting.localization.Localization(__file__, 6, 49), zip_2430, *[_mapper_2431], **kwargs_2432)

# Obtaining the member '__getitem__' of a type (line 6)
getitem___2434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), zip_call_result_2433, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_2435 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), getitem___2434, int_2429)

# Getting the type of 'Foo'
Foo_2436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member 'tuple_var_assignment_2406' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_2436, 'tuple_var_assignment_2406', subscript_call_result_2435)

# Assigning a Subscript to a Name (line 6):

# Obtaining the type of the subscript
int_2437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'int')

# Call to zip(...): (line 6)
# Getting the type of '_mapper' (line 6)
_mapper_2439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 54), '_mapper', False)
# Processing the call keyword arguments (line 6)
kwargs_2440 = {}
# Getting the type of 'zip' (line 6)
zip_2438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 49), 'zip', False)
# Calling zip(args, kwargs) (line 6)
zip_call_result_2441 = invoke(stypy.reporting.localization.Localization(__file__, 6, 49), zip_2438, *[_mapper_2439], **kwargs_2440)

# Obtaining the member '__getitem__' of a type (line 6)
getitem___2442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), zip_call_result_2441, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_2443 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), getitem___2442, int_2437)

# Getting the type of 'Foo'
Foo_2444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member 'tuple_var_assignment_2407' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_2444, 'tuple_var_assignment_2407', subscript_call_result_2443)

# Assigning a Name to a Name (line 6):
# Getting the type of 'Foo'
Foo_2445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Obtaining the member 'tuple_var_assignment_2405' of a type
tuple_var_assignment_2405_2446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_2445, 'tuple_var_assignment_2405')
# Getting the type of 'Foo'
Foo_2447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member '_defaulttype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_2447, '_defaulttype', tuple_var_assignment_2405_2446)

# Assigning a Name to a Name (line 6):
# Getting the type of 'Foo'
Foo_2448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Obtaining the member 'tuple_var_assignment_2406' of a type
tuple_var_assignment_2406_2449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_2448, 'tuple_var_assignment_2406')
# Getting the type of 'Foo'
Foo_2450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member '_defaultfunc' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_2450, '_defaultfunc', tuple_var_assignment_2406_2449)

# Assigning a Name to a Name (line 6):
# Getting the type of 'Foo'
Foo_2451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Obtaining the member 'tuple_var_assignment_2407' of a type
tuple_var_assignment_2407_2452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_2451, 'tuple_var_assignment_2407')
# Getting the type of 'Foo'
Foo_2453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member '_defaultfill' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_2453, '_defaultfill', tuple_var_assignment_2407_2452)

# Assigning a Call to a Name (line 8):

# Assigning a Call to a Name (line 8):

# Call to Foo(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_2455 = {}
# Getting the type of 'Foo' (line 8)
Foo_2454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Foo', False)
# Calling Foo(args, kwargs) (line 8)
Foo_call_result_2456 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), Foo_2454, *[], **kwargs_2455)

# Assigning a type to the variable 'f' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'f', Foo_call_result_2456)

# Assigning a Attribute to a Name (line 10):

# Assigning a Attribute to a Name (line 10):
# Getting the type of 'f' (line 10)
f_2457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'f')
# Obtaining the member '_defaulttype' of a type (line 10)
_defaulttype_2458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), f_2457, '_defaulttype')
# Assigning a type to the variable 'r1' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r1', _defaulttype_2458)

# Assigning a Attribute to a Name (line 11):

# Assigning a Attribute to a Name (line 11):
# Getting the type of 'f' (line 11)
f_2459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'f')
# Obtaining the member '_defaultfunc' of a type (line 11)
_defaultfunc_2460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), f_2459, '_defaultfunc')
# Assigning a type to the variable 'r2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r2', _defaultfunc_2460)

# Assigning a Attribute to a Name (line 12):

# Assigning a Attribute to a Name (line 12):
# Getting the type of 'f' (line 12)
f_2461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'f')
# Obtaining the member '_defaultfill' of a type (line 12)
_defaultfill_2462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), f_2461, '_defaultfill')
# Assigning a type to the variable 'r3' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r3', _defaultfill_2462)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
