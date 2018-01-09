
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: class Foo:
4:     att = "sample"
5: 
6:     def met(self):
7:         return self.att
8: 
9: 
10: 
11: f = Foo()
12: 
13: att_predelete = f.att
14: met_predelete = f.met
15: met_result_predelete = f.met()
16: func_predelete = f.met
17: 
18: del Foo.att
19: 
20: att_postdelete = f.att
21: met_result_postdelete = f.met()
22: 
23: del Foo.met
24: 
25: met_postdelete = f.met
26: func_result_postdelete = func_predelete()

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
        module_type_store = module_type_store.open_function_context('met', 6, 4, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'self', type_of_self)
        
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

        # Getting the type of 'self' (line 7)
        self_6222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'self')
        # Obtaining the member 'att' of a type (line 7)
        att_6223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 15), self_6222, 'att')
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', att_6223)
        
        # ################# End of 'met(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'met' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_6224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6224)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'met'
        return stypy_return_type_6224


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 3, 0, False)
        # Assigning a type to the variable 'self' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'self', type_of_self)
        
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


# Assigning a type to the variable 'Foo' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'Foo', Foo)

# Assigning a Str to a Name (line 4):
str_6225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'sample')
# Getting the type of 'Foo'
Foo_6226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member 'att' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_6226, 'att', str_6225)

# Assigning a Call to a Name (line 11):

# Call to Foo(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_6228 = {}
# Getting the type of 'Foo' (line 11)
Foo_6227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'Foo', False)
# Calling Foo(args, kwargs) (line 11)
Foo_call_result_6229 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), Foo_6227, *[], **kwargs_6228)

# Assigning a type to the variable 'f' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'f', Foo_call_result_6229)

# Assigning a Attribute to a Name (line 13):
# Getting the type of 'f' (line 13)
f_6230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'f')
# Obtaining the member 'att' of a type (line 13)
att_6231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 16), f_6230, 'att')
# Assigning a type to the variable 'att_predelete' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'att_predelete', att_6231)

# Assigning a Attribute to a Name (line 14):
# Getting the type of 'f' (line 14)
f_6232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'f')
# Obtaining the member 'met' of a type (line 14)
met_6233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 16), f_6232, 'met')
# Assigning a type to the variable 'met_predelete' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'met_predelete', met_6233)

# Assigning a Call to a Name (line 15):

# Call to met(...): (line 15)
# Processing the call keyword arguments (line 15)
kwargs_6236 = {}
# Getting the type of 'f' (line 15)
f_6234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'f', False)
# Obtaining the member 'met' of a type (line 15)
met_6235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 23), f_6234, 'met')
# Calling met(args, kwargs) (line 15)
met_call_result_6237 = invoke(stypy.reporting.localization.Localization(__file__, 15, 23), met_6235, *[], **kwargs_6236)

# Assigning a type to the variable 'met_result_predelete' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'met_result_predelete', met_call_result_6237)

# Assigning a Attribute to a Name (line 16):
# Getting the type of 'f' (line 16)
f_6238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'f')
# Obtaining the member 'met' of a type (line 16)
met_6239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 17), f_6238, 'met')
# Assigning a type to the variable 'func_predelete' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'func_predelete', met_6239)
# Deleting a member
# Getting the type of 'Foo' (line 18)
Foo_6240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'Foo')
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 18, 0), Foo_6240, 'att')

# Assigning a Attribute to a Name (line 20):
# Getting the type of 'f' (line 20)
f_6241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'f')
# Obtaining the member 'att' of a type (line 20)
att_6242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 17), f_6241, 'att')
# Assigning a type to the variable 'att_postdelete' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'att_postdelete', att_6242)

# Assigning a Call to a Name (line 21):

# Call to met(...): (line 21)
# Processing the call keyword arguments (line 21)
kwargs_6245 = {}
# Getting the type of 'f' (line 21)
f_6243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'f', False)
# Obtaining the member 'met' of a type (line 21)
met_6244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 24), f_6243, 'met')
# Calling met(args, kwargs) (line 21)
met_call_result_6246 = invoke(stypy.reporting.localization.Localization(__file__, 21, 24), met_6244, *[], **kwargs_6245)

# Assigning a type to the variable 'met_result_postdelete' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'met_result_postdelete', met_call_result_6246)
# Deleting a member
# Getting the type of 'Foo' (line 23)
Foo_6247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'Foo')
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 23, 0), Foo_6247, 'met')

# Assigning a Attribute to a Name (line 25):
# Getting the type of 'f' (line 25)
f_6248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'f')
# Obtaining the member 'met' of a type (line 25)
met_6249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 17), f_6248, 'met')
# Assigning a type to the variable 'met_postdelete' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'met_postdelete', met_6249)

# Assigning a Call to a Name (line 26):

# Call to func_predelete(...): (line 26)
# Processing the call keyword arguments (line 26)
kwargs_6251 = {}
# Getting the type of 'func_predelete' (line 26)
func_predelete_6250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'func_predelete', False)
# Calling func_predelete(args, kwargs) (line 26)
func_predelete_call_result_6252 = invoke(stypy.reporting.localization.Localization(__file__, 26, 25), func_predelete_6250, *[], **kwargs_6251)

# Assigning a type to the variable 'func_result_postdelete' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'func_result_postdelete', func_predelete_call_result_6252)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
