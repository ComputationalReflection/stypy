
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class Foo:
2:     def __init__(self):
3:         pass
4: 
5: 
6: def func(x):
7:     return x
8: 
9: 
10: def class_func(cls, x):
11:     return x
12: 
13: 
14: f = Foo()
15: f2 = Foo()
16: 
17: f.a = 3
18: r1 = f.a
19: r2 = f2.a  # Reported
20: 
21: if f.a > 0:
22:     f.att1 = 4
23: else:
24:     f.att2 = "hi"
25: 
26: r3 = f.att1
27: r4 = f.att2  # Not reported
28: 
29: Foo.class_a = "hi"
30: 
31: r5 = f.class_a  # Incorrectly reported. Prints "hi"
32: 
33: 

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
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 2, 4, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'self', type_of_self)
        
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

@norecursion
def func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'func'
    module_type_store = module_type_store.open_function_context('func', 6, 0, False)
    
    # Passed parameters checking function
    func.stypy_localization = localization
    func.stypy_type_of_self = None
    func.stypy_type_store = module_type_store
    func.stypy_function_name = 'func'
    func.stypy_param_names_list = ['x']
    func.stypy_varargs_param_name = None
    func.stypy_kwargs_param_name = None
    func.stypy_call_defaults = defaults
    func.stypy_call_varargs = varargs
    func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'func', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'func', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'func(...)' code ##################

    # Getting the type of 'x' (line 7)
    x_6857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type', x_6857)
    
    # ################# End of 'func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'func' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_6858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6858)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'func'
    return stypy_return_type_6858

# Assigning a type to the variable 'func' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'func', func)

@norecursion
def class_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'class_func'
    module_type_store = module_type_store.open_function_context('class_func', 10, 0, False)
    
    # Passed parameters checking function
    class_func.stypy_localization = localization
    class_func.stypy_type_of_self = None
    class_func.stypy_type_store = module_type_store
    class_func.stypy_function_name = 'class_func'
    class_func.stypy_param_names_list = ['cls', 'x']
    class_func.stypy_varargs_param_name = None
    class_func.stypy_kwargs_param_name = None
    class_func.stypy_call_defaults = defaults
    class_func.stypy_call_varargs = varargs
    class_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'class_func', ['cls', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'class_func', localization, ['cls', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'class_func(...)' code ##################

    # Getting the type of 'x' (line 11)
    x_6859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', x_6859)
    
    # ################# End of 'class_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'class_func' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_6860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6860)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'class_func'
    return stypy_return_type_6860

# Assigning a type to the variable 'class_func' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'class_func', class_func)

# Assigning a Call to a Name (line 14):

# Call to Foo(...): (line 14)
# Processing the call keyword arguments (line 14)
kwargs_6862 = {}
# Getting the type of 'Foo' (line 14)
Foo_6861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'Foo', False)
# Calling Foo(args, kwargs) (line 14)
Foo_call_result_6863 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), Foo_6861, *[], **kwargs_6862)

# Assigning a type to the variable 'f' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'f', Foo_call_result_6863)

# Assigning a Call to a Name (line 15):

# Call to Foo(...): (line 15)
# Processing the call keyword arguments (line 15)
kwargs_6865 = {}
# Getting the type of 'Foo' (line 15)
Foo_6864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'Foo', False)
# Calling Foo(args, kwargs) (line 15)
Foo_call_result_6866 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), Foo_6864, *[], **kwargs_6865)

# Assigning a type to the variable 'f2' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'f2', Foo_call_result_6866)

# Assigning a Num to a Attribute (line 17):
int_6867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 6), 'int')
# Getting the type of 'f' (line 17)
f_6868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'f')
# Setting the type of the member 'a' of a type (line 17)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 0), f_6868, 'a', int_6867)

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'f' (line 18)
f_6869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'f')
# Obtaining the member 'a' of a type (line 18)
a_6870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), f_6869, 'a')
# Assigning a type to the variable 'r1' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r1', a_6870)

# Assigning a Attribute to a Name (line 19):
# Getting the type of 'f2' (line 19)
f2_6871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'f2')
# Obtaining the member 'a' of a type (line 19)
a_6872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), f2_6871, 'a')
# Assigning a type to the variable 'r2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r2', a_6872)


# Getting the type of 'f' (line 21)
f_6873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 3), 'f')
# Obtaining the member 'a' of a type (line 21)
a_6874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 3), f_6873, 'a')
int_6875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'int')
# Applying the binary operator '>' (line 21)
result_gt_6876 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 3), '>', a_6874, int_6875)

# Testing the type of an if condition (line 21)
if_condition_6877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 0), result_gt_6876)
# Assigning a type to the variable 'if_condition_6877' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'if_condition_6877', if_condition_6877)
# SSA begins for if statement (line 21)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Attribute (line 22):
int_6878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'int')
# Getting the type of 'f' (line 22)
f_6879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'f')
# Setting the type of the member 'att1' of a type (line 22)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), f_6879, 'att1', int_6878)
# SSA branch for the else part of an if statement (line 21)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Attribute (line 24):
str_6880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'str', 'hi')
# Getting the type of 'f' (line 24)
f_6881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'f')
# Setting the type of the member 'att2' of a type (line 24)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), f_6881, 'att2', str_6880)
# SSA join for if statement (line 21)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Attribute to a Name (line 26):
# Getting the type of 'f' (line 26)
f_6882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'f')
# Obtaining the member 'att1' of a type (line 26)
att1_6883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 5), f_6882, 'att1')
# Assigning a type to the variable 'r3' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'r3', att1_6883)

# Assigning a Attribute to a Name (line 27):
# Getting the type of 'f' (line 27)
f_6884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 5), 'f')
# Obtaining the member 'att2' of a type (line 27)
att2_6885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 5), f_6884, 'att2')
# Assigning a type to the variable 'r4' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'r4', att2_6885)

# Assigning a Str to a Attribute (line 29):
str_6886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 14), 'str', 'hi')
# Getting the type of 'Foo' (line 29)
Foo_6887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'Foo')
# Setting the type of the member 'class_a' of a type (line 29)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 0), Foo_6887, 'class_a', str_6886)

# Assigning a Attribute to a Name (line 31):
# Getting the type of 'f' (line 31)
f_6888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 5), 'f')
# Obtaining the member 'class_a' of a type (line 31)
class_a_6889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 5), f_6888, 'class_a')
# Assigning a type to the variable 'r5' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'r5', class_a_6889)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
