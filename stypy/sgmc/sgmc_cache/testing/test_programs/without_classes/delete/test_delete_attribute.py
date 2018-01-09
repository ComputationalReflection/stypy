
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: class Nested:
3:     def __init__(self):
4:         self.a = 3
5: 
6: class Foo:
7:     def __init__(self):
8:         self.att = Nested()
9: 
10:     def met(self):
11:         return self.att
12: 
13: 
14: 
15: f = Foo()
16: 
17: x1 = f.att.a
18: del f.att.a
19: x2 = f.att.a
20: 
21: y1 = f.att
22: del f.att
23: y2 = f.att
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Nested' class

class Nested:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 3, 4, False)
        # Assigning a type to the variable 'self' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Nested.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 4):
        int_6253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 17), 'int')
        # Getting the type of 'self' (line 4)
        self_6254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 8), 'self')
        # Setting the type of the member 'a' of a type (line 4)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 8), self_6254, 'a', int_6253)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Nested' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'Nested', Nested)
# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 7, 4, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
        
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

        
        # Assigning a Call to a Attribute (line 8):
        
        # Call to Nested(...): (line 8)
        # Processing the call keyword arguments (line 8)
        kwargs_6256 = {}
        # Getting the type of 'Nested' (line 8)
        Nested_6255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'Nested', False)
        # Calling Nested(args, kwargs) (line 8)
        Nested_call_result_6257 = invoke(stypy.reporting.localization.Localization(__file__, 8, 19), Nested_6255, *[], **kwargs_6256)
        
        # Getting the type of 'self' (line 8)
        self_6258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self')
        # Setting the type of the member 'att' of a type (line 8)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), self_6258, 'att', Nested_call_result_6257)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def met(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'met'
        module_type_store = module_type_store.open_function_context('met', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo.met.__dict__.__setitem__('stypy_localization', localization)
        Foo.met.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo.met.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.met.__dict__.__setitem__('stypy_function_name', 'Foo.met')
        Foo.met.__dict__.__setitem__('stypy_param_names_list', [])
        Foo.met.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo.met.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.met.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.met.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.met.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.met.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.met', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'met', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'met(...)' code ##################

        # Getting the type of 'self' (line 11)
        self_6259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'self')
        # Obtaining the member 'att' of a type (line 11)
        att_6260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 15), self_6259, 'att')
        # Assigning a type to the variable 'stypy_return_type' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', att_6260)
        
        # ################# End of 'met(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'met' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_6261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'met'
        return stypy_return_type_6261


# Assigning a type to the variable 'Foo' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Foo', Foo)

# Assigning a Call to a Name (line 15):

# Call to Foo(...): (line 15)
# Processing the call keyword arguments (line 15)
kwargs_6263 = {}
# Getting the type of 'Foo' (line 15)
Foo_6262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'Foo', False)
# Calling Foo(args, kwargs) (line 15)
Foo_call_result_6264 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), Foo_6262, *[], **kwargs_6263)

# Assigning a type to the variable 'f' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'f', Foo_call_result_6264)

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'f' (line 17)
f_6265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'f')
# Obtaining the member 'att' of a type (line 17)
att_6266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), f_6265, 'att')
# Obtaining the member 'a' of a type (line 17)
a_6267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), att_6266, 'a')
# Assigning a type to the variable 'x1' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'x1', a_6267)
# Deleting a member
# Getting the type of 'f' (line 18)
f_6268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'f')
# Obtaining the member 'att' of a type (line 18)
att_6269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), f_6268, 'att')
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 18, 0), att_6269, 'a')

# Assigning a Attribute to a Name (line 19):
# Getting the type of 'f' (line 19)
f_6270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'f')
# Obtaining the member 'att' of a type (line 19)
att_6271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), f_6270, 'att')
# Obtaining the member 'a' of a type (line 19)
a_6272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), att_6271, 'a')
# Assigning a type to the variable 'x2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'x2', a_6272)

# Assigning a Attribute to a Name (line 21):
# Getting the type of 'f' (line 21)
f_6273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'f')
# Obtaining the member 'att' of a type (line 21)
att_6274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), f_6273, 'att')
# Assigning a type to the variable 'y1' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'y1', att_6274)
# Deleting a member
# Getting the type of 'f' (line 22)
f_6275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'f')
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 22, 0), f_6275, 'att')

# Assigning a Attribute to a Name (line 23):
# Getting the type of 'f' (line 23)
f_6276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'f')
# Obtaining the member 'att' of a type (line 23)
att_6277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 5), f_6276, 'att')
# Assigning a type to the variable 'y2' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'y2', att_6277)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
