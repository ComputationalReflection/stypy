
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class Foo:
2:     att = 3
3: 
4:     def met(self):
5:         self.my_att = 3
6: 
7:         return 3
8: 
9:     def met_class(self):
10:         self.my_att = 3
11:         Foo.class_att = True
12:         return 3
13: 
14: 
15: f1 = Foo()
16: # f1.met() # Not called!
17: r1 = f1.my_att  # No error reported, but it is an error because met was not called
18: 
19: f2 = Foo()
20: 
21: f2.met_class()
22: r2 = Foo.class_att  # This is reported as an error even calling met()!
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def met(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'met'
        module_type_store = module_type_store.open_function_context('met', 4, 4, False)
        # Assigning a type to the variable 'self' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'self', type_of_self)
        
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

        
        # Assigning a Num to a Attribute (line 5):
        int_6890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
        # Getting the type of 'self' (line 5)
        self_6891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'self')
        # Setting the type of the member 'my_att' of a type (line 5)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 8), self_6891, 'my_att', int_6890)
        int_6892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', int_6892)
        
        # ################# End of 'met(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'met' in the type store
        # Getting the type of 'stypy_return_type' (line 4)
        stypy_return_type_6893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6893)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'met'
        return stypy_return_type_6893


    @norecursion
    def met_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'met_class'
        module_type_store = module_type_store.open_function_context('met_class', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo.met_class.__dict__.__setitem__('stypy_localization', localization)
        Foo.met_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo.met_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.met_class.__dict__.__setitem__('stypy_function_name', 'Foo.met_class')
        Foo.met_class.__dict__.__setitem__('stypy_param_names_list', [])
        Foo.met_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo.met_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.met_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.met_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.met_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.met_class.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.met_class', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'met_class', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'met_class(...)' code ##################

        
        # Assigning a Num to a Attribute (line 10):
        int_6894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 22), 'int')
        # Getting the type of 'self' (line 10)
        self_6895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self')
        # Setting the type of the member 'my_att' of a type (line 10)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), self_6895, 'my_att', int_6894)
        
        # Assigning a Name to a Attribute (line 11):
        # Getting the type of 'True' (line 11)
        True_6896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'True')
        # Getting the type of 'Foo' (line 11)
        Foo_6897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'Foo')
        # Setting the type of the member 'class_att' of a type (line 11)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), Foo_6897, 'class_att', True_6896)
        int_6898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type', int_6898)
        
        # ################# End of 'met_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'met_class' in the type store
        # Getting the type of 'stypy_return_type' (line 9)
        stypy_return_type_6899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6899)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'met_class'
        return stypy_return_type_6899


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1, 0, False)
        # Assigning a type to the variable 'self' (line 2)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'self', type_of_self)
        
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


# Assigning a type to the variable 'Foo' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'Foo', Foo)

# Assigning a Num to a Name (line 2):
int_6900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'int')
# Getting the type of 'Foo'
Foo_6901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member 'att' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_6901, 'att', int_6900)

# Assigning a Call to a Name (line 15):

# Call to Foo(...): (line 15)
# Processing the call keyword arguments (line 15)
kwargs_6903 = {}
# Getting the type of 'Foo' (line 15)
Foo_6902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'Foo', False)
# Calling Foo(args, kwargs) (line 15)
Foo_call_result_6904 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), Foo_6902, *[], **kwargs_6903)

# Assigning a type to the variable 'f1' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'f1', Foo_call_result_6904)

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'f1' (line 17)
f1_6905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'f1')
# Obtaining the member 'my_att' of a type (line 17)
my_att_6906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), f1_6905, 'my_att')
# Assigning a type to the variable 'r1' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r1', my_att_6906)

# Assigning a Call to a Name (line 19):

# Call to Foo(...): (line 19)
# Processing the call keyword arguments (line 19)
kwargs_6908 = {}
# Getting the type of 'Foo' (line 19)
Foo_6907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'Foo', False)
# Calling Foo(args, kwargs) (line 19)
Foo_call_result_6909 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), Foo_6907, *[], **kwargs_6908)

# Assigning a type to the variable 'f2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'f2', Foo_call_result_6909)

# Call to met_class(...): (line 21)
# Processing the call keyword arguments (line 21)
kwargs_6912 = {}
# Getting the type of 'f2' (line 21)
f2_6910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'f2', False)
# Obtaining the member 'met_class' of a type (line 21)
met_class_6911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 0), f2_6910, 'met_class')
# Calling met_class(args, kwargs) (line 21)
met_class_call_result_6913 = invoke(stypy.reporting.localization.Localization(__file__, 21, 0), met_class_6911, *[], **kwargs_6912)


# Assigning a Attribute to a Name (line 22):
# Getting the type of 'Foo' (line 22)
Foo_6914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'Foo')
# Obtaining the member 'class_att' of a type (line 22)
class_att_6915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 5), Foo_6914, 'class_att')
# Assigning a type to the variable 'r2' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r2', class_att_6915)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
