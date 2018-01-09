
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
9: 
10: f = Foo()
11: 
12: del f.my_att  # Not reported and met was not called
13: 
14: a = 0
15: if a > 0:
16:     f.xx = 3
17: else:
18:     f.yy = 5
19: 
20: del f.yy  # Not detected
21: del f.xx
22: 
23: del list.__doc__  # Failure not detected
24: 

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
        int_7186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
        # Getting the type of 'self' (line 5)
        self_7187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'self')
        # Setting the type of the member 'my_att' of a type (line 5)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 8), self_7187, 'my_att', int_7186)
        int_7188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', int_7188)
        
        # ################# End of 'met(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'met' in the type store
        # Getting the type of 'stypy_return_type' (line 4)
        stypy_return_type_7189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'met'
        return stypy_return_type_7189


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
int_7190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'int')
# Getting the type of 'Foo'
Foo_7191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member 'att' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_7191, 'att', int_7190)

# Assigning a Call to a Name (line 10):

# Call to Foo(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_7193 = {}
# Getting the type of 'Foo' (line 10)
Foo_7192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Foo', False)
# Calling Foo(args, kwargs) (line 10)
Foo_call_result_7194 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), Foo_7192, *[], **kwargs_7193)

# Assigning a type to the variable 'f' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'f', Foo_call_result_7194)
# Deleting a member
# Getting the type of 'f' (line 12)
f_7195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'f')
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 12, 0), f_7195, 'my_att')

# Assigning a Num to a Name (line 14):
int_7196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'int')
# Assigning a type to the variable 'a' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'a', int_7196)


# Getting the type of 'a' (line 15)
a_7197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 3), 'a')
int_7198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 7), 'int')
# Applying the binary operator '>' (line 15)
result_gt_7199 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 3), '>', a_7197, int_7198)

# Testing the type of an if condition (line 15)
if_condition_7200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 0), result_gt_7199)
# Assigning a type to the variable 'if_condition_7200' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'if_condition_7200', if_condition_7200)
# SSA begins for if statement (line 15)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Attribute (line 16):
int_7201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'int')
# Getting the type of 'f' (line 16)
f_7202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'f')
# Setting the type of the member 'xx' of a type (line 16)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), f_7202, 'xx', int_7201)
# SSA branch for the else part of an if statement (line 15)
module_type_store.open_ssa_branch('else')

# Assigning a Num to a Attribute (line 18):
int_7203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'int')
# Getting the type of 'f' (line 18)
f_7204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'f')
# Setting the type of the member 'yy' of a type (line 18)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), f_7204, 'yy', int_7203)
# SSA join for if statement (line 15)
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
# Getting the type of 'f' (line 20)
f_7205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'f')
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 20, 0), f_7205, 'yy')
# Deleting a member
# Getting the type of 'f' (line 21)
f_7206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'f')
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 21, 0), f_7206, 'xx')
# Deleting a member
# Getting the type of 'list' (line 23)
list_7207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'list')
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 23, 0), list_7207, '__doc__')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
