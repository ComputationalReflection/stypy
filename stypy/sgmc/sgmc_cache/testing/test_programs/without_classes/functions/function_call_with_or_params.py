
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: class Foo:
3:     def __init__(self, txt, params=None):
4:         self.txt = txt
5:         self._extras = params or {}
6: 
7:     def pars(self, x=3):
8:         return x
9: 
10: f = Foo("test")
11: 
12: r = f.pars()
13: 
14: r2 = f._extras

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 3)
        None_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 35), 'None')
        defaults = [None_984]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 3, 4, False)
        # Assigning a type to the variable 'self' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', ['txt', 'params'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['txt', 'params'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 4):
        # Getting the type of 'txt' (line 4)
        txt_985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 19), 'txt')
        # Getting the type of 'self' (line 4)
        self_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 8), 'self')
        # Setting the type of the member 'txt' of a type (line 4)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 8), self_986, 'txt', txt_985)
        
        # Assigning a BoolOp to a Attribute (line 5):
        
        # Evaluating a boolean operation
        # Getting the type of 'params' (line 5)
        params_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 23), 'params')
        
        # Obtaining an instance of the builtin type 'dict' (line 5)
        dict_988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 33), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 5)
        
        # Applying the binary operator 'or' (line 5)
        result_or_keyword_989 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 23), 'or', params_987, dict_988)
        
        # Getting the type of 'self' (line 5)
        self_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'self')
        # Setting the type of the member '_extras' of a type (line 5)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 8), self_990, '_extras', result_or_keyword_989)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def pars(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 21), 'int')
        defaults = [int_991]
        # Create a new context for function 'pars'
        module_type_store = module_type_store.open_function_context('pars', 7, 4, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo.pars.__dict__.__setitem__('stypy_localization', localization)
        Foo.pars.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo.pars.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.pars.__dict__.__setitem__('stypy_function_name', 'Foo.pars')
        Foo.pars.__dict__.__setitem__('stypy_param_names_list', ['x'])
        Foo.pars.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo.pars.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.pars.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.pars.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.pars.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.pars.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.pars', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pars', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pars(...)' code ##################

        # Getting the type of 'x' (line 8)
        x_992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', x_992)
        
        # ################# End of 'pars(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pars' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_993)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pars'
        return stypy_return_type_993


# Assigning a type to the variable 'Foo' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'Foo', Foo)

# Assigning a Call to a Name (line 10):

# Call to Foo(...): (line 10)
# Processing the call arguments (line 10)
str_995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'test')
# Processing the call keyword arguments (line 10)
kwargs_996 = {}
# Getting the type of 'Foo' (line 10)
Foo_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Foo', False)
# Calling Foo(args, kwargs) (line 10)
Foo_call_result_997 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), Foo_994, *[str_995], **kwargs_996)

# Assigning a type to the variable 'f' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'f', Foo_call_result_997)

# Assigning a Call to a Name (line 12):

# Call to pars(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_1000 = {}
# Getting the type of 'f' (line 12)
f_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'f', False)
# Obtaining the member 'pars' of a type (line 12)
pars_999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), f_998, 'pars')
# Calling pars(args, kwargs) (line 12)
pars_call_result_1001 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), pars_999, *[], **kwargs_1000)

# Assigning a type to the variable 'r' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r', pars_call_result_1001)

# Assigning a Attribute to a Name (line 14):
# Getting the type of 'f' (line 14)
f_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'f')
# Obtaining the member '_extras' of a type (line 14)
_extras_1003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), f_1002, '_extras')
# Assigning a type to the variable 'r2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r2', _extras_1003)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
