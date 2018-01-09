
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''Prints type-coercion tables for the built-in NumPy types
3: 
4: '''
5: from __future__ import division, absolute_import, print_function
6: 
7: import numpy as np
8: 
9: # Generic object that can be added, but doesn't do anything else
10: class GenericObject(object):
11:     def __init__(self, v):
12:         self.v = v
13: 
14:     def __add__(self, other):
15:         return self
16: 
17:     def __radd__(self, other):
18:         return self
19: 
20:     dtype = np.dtype('O')
21: 
22: def print_cancast_table(ntypes):
23:     print('X', end=' ')
24:     for char in ntypes:
25:         print(char, end=' ')
26:     print()
27:     for row in ntypes:
28:         print(row, end=' ')
29:         for col in ntypes:
30:             print(int(np.can_cast(row, col)), end=' ')
31:         print()
32: 
33: def print_coercion_table(ntypes, inputfirstvalue, inputsecondvalue, firstarray, use_promote_types=False):
34:     print('+', end=' ')
35:     for char in ntypes:
36:         print(char, end=' ')
37:     print()
38:     for row in ntypes:
39:         if row == 'O':
40:             rowtype = GenericObject
41:         else:
42:             rowtype = np.obj2sctype(row)
43: 
44:         print(row, end=' ')
45:         for col in ntypes:
46:             if col == 'O':
47:                 coltype = GenericObject
48:             else:
49:                 coltype = np.obj2sctype(col)
50:             try:
51:                 if firstarray:
52:                     rowvalue = np.array([rowtype(inputfirstvalue)], dtype=rowtype)
53:                 else:
54:                     rowvalue = rowtype(inputfirstvalue)
55:                 colvalue = coltype(inputsecondvalue)
56:                 if use_promote_types:
57:                     char = np.promote_types(rowvalue.dtype, colvalue.dtype).char
58:                 else:
59:                     value = np.add(rowvalue, colvalue)
60:                     if isinstance(value, np.ndarray):
61:                         char = value.dtype.char
62:                     else:
63:                         char = np.dtype(type(value)).char
64:             except ValueError:
65:                 char = '!'
66:             except OverflowError:
67:                 char = '@'
68:             except TypeError:
69:                 char = '#'
70:             print(char, end=' ')
71:         print()
72: 
73: print("can cast")
74: print_cancast_table(np.typecodes['All'])
75: print()
76: print("In these tables, ValueError is '!', OverflowError is '@', TypeError is '#'")
77: print()
78: print("scalar + scalar")
79: print_coercion_table(np.typecodes['All'], 0, 0, False)
80: print()
81: print("scalar + neg scalar")
82: print_coercion_table(np.typecodes['All'], 0, -1, False)
83: print()
84: print("array + scalar")
85: print_coercion_table(np.typecodes['All'], 0, 0, True)
86: print()
87: print("array + neg scalar")
88: print_coercion_table(np.typecodes['All'], 0, -1, True)
89: print()
90: print("promote_types")
91: print_coercion_table(np.typecodes['All'], 0, 0, False, True)
92: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_182501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', 'Prints type-coercion tables for the built-in NumPy types\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_182502 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_182502) is not StypyTypeError):

    if (import_182502 != 'pyd_module'):
        __import__(import_182502)
        sys_modules_182503 = sys.modules[import_182502]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_182503.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_182502)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

# Declaration of the 'GenericObject' class

class GenericObject(object, ):

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GenericObject.__init__', ['v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 12):
        # Getting the type of 'v' (line 12)
        v_182504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'v')
        # Getting the type of 'self' (line 12)
        self_182505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self')
        # Setting the type of the member 'v' of a type (line 12)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), self_182505, 'v', v_182504)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GenericObject.__add__.__dict__.__setitem__('stypy_localization', localization)
        GenericObject.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GenericObject.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GenericObject.__add__.__dict__.__setitem__('stypy_function_name', 'GenericObject.__add__')
        GenericObject.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        GenericObject.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GenericObject.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GenericObject.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GenericObject.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GenericObject.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GenericObject.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GenericObject.__add__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        # Getting the type of 'self' (line 15)
        self_182506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', self_182506)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_182507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_182507)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_182507


    @norecursion
    def __radd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__radd__'
        module_type_store = module_type_store.open_function_context('__radd__', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GenericObject.__radd__.__dict__.__setitem__('stypy_localization', localization)
        GenericObject.__radd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GenericObject.__radd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GenericObject.__radd__.__dict__.__setitem__('stypy_function_name', 'GenericObject.__radd__')
        GenericObject.__radd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        GenericObject.__radd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GenericObject.__radd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GenericObject.__radd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GenericObject.__radd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GenericObject.__radd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GenericObject.__radd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GenericObject.__radd__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__radd__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__radd__(...)' code ##################

        # Getting the type of 'self' (line 18)
        self_182508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type', self_182508)
        
        # ################# End of '__radd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__radd__' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_182509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_182509)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__radd__'
        return stypy_return_type_182509


# Assigning a type to the variable 'GenericObject' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'GenericObject', GenericObject)

# Assigning a Call to a Name (line 20):

# Call to dtype(...): (line 20)
# Processing the call arguments (line 20)
str_182512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'str', 'O')
# Processing the call keyword arguments (line 20)
kwargs_182513 = {}
# Getting the type of 'np' (line 20)
np_182510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'np', False)
# Obtaining the member 'dtype' of a type (line 20)
dtype_182511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), np_182510, 'dtype')
# Calling dtype(args, kwargs) (line 20)
dtype_call_result_182514 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), dtype_182511, *[str_182512], **kwargs_182513)

# Getting the type of 'GenericObject'
GenericObject_182515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'GenericObject')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), GenericObject_182515, 'dtype', dtype_call_result_182514)

@norecursion
def print_cancast_table(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_cancast_table'
    module_type_store = module_type_store.open_function_context('print_cancast_table', 22, 0, False)
    
    # Passed parameters checking function
    print_cancast_table.stypy_localization = localization
    print_cancast_table.stypy_type_of_self = None
    print_cancast_table.stypy_type_store = module_type_store
    print_cancast_table.stypy_function_name = 'print_cancast_table'
    print_cancast_table.stypy_param_names_list = ['ntypes']
    print_cancast_table.stypy_varargs_param_name = None
    print_cancast_table.stypy_kwargs_param_name = None
    print_cancast_table.stypy_call_defaults = defaults
    print_cancast_table.stypy_call_varargs = varargs
    print_cancast_table.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_cancast_table', ['ntypes'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_cancast_table', localization, ['ntypes'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_cancast_table(...)' code ##################

    
    # Call to print(...): (line 23)
    # Processing the call arguments (line 23)
    str_182517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'str', 'X')
    # Processing the call keyword arguments (line 23)
    str_182518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'str', ' ')
    keyword_182519 = str_182518
    kwargs_182520 = {'end': keyword_182519}
    # Getting the type of 'print' (line 23)
    print_182516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'print', False)
    # Calling print(args, kwargs) (line 23)
    print_call_result_182521 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), print_182516, *[str_182517], **kwargs_182520)
    
    
    # Getting the type of 'ntypes' (line 24)
    ntypes_182522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'ntypes')
    # Testing the type of a for loop iterable (line 24)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 24, 4), ntypes_182522)
    # Getting the type of the for loop variable (line 24)
    for_loop_var_182523 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 24, 4), ntypes_182522)
    # Assigning a type to the variable 'char' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'char', for_loop_var_182523)
    # SSA begins for a for statement (line 24)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'char' (line 25)
    char_182525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'char', False)
    # Processing the call keyword arguments (line 25)
    str_182526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'str', ' ')
    keyword_182527 = str_182526
    kwargs_182528 = {'end': keyword_182527}
    # Getting the type of 'print' (line 25)
    print_182524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'print', False)
    # Calling print(args, kwargs) (line 25)
    print_call_result_182529 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), print_182524, *[char_182525], **kwargs_182528)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_182531 = {}
    # Getting the type of 'print' (line 26)
    print_182530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'print', False)
    # Calling print(args, kwargs) (line 26)
    print_call_result_182532 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), print_182530, *[], **kwargs_182531)
    
    
    # Getting the type of 'ntypes' (line 27)
    ntypes_182533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'ntypes')
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 4), ntypes_182533)
    # Getting the type of the for loop variable (line 27)
    for_loop_var_182534 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 4), ntypes_182533)
    # Assigning a type to the variable 'row' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'row', for_loop_var_182534)
    # SSA begins for a for statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'row' (line 28)
    row_182536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'row', False)
    # Processing the call keyword arguments (line 28)
    str_182537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 23), 'str', ' ')
    keyword_182538 = str_182537
    kwargs_182539 = {'end': keyword_182538}
    # Getting the type of 'print' (line 28)
    print_182535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'print', False)
    # Calling print(args, kwargs) (line 28)
    print_call_result_182540 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), print_182535, *[row_182536], **kwargs_182539)
    
    
    # Getting the type of 'ntypes' (line 29)
    ntypes_182541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'ntypes')
    # Testing the type of a for loop iterable (line 29)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 29, 8), ntypes_182541)
    # Getting the type of the for loop variable (line 29)
    for_loop_var_182542 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 29, 8), ntypes_182541)
    # Assigning a type to the variable 'col' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'col', for_loop_var_182542)
    # SSA begins for a for statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Call to int(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Call to can_cast(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'row' (line 30)
    row_182547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'row', False)
    # Getting the type of 'col' (line 30)
    col_182548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'col', False)
    # Processing the call keyword arguments (line 30)
    kwargs_182549 = {}
    # Getting the type of 'np' (line 30)
    np_182545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'np', False)
    # Obtaining the member 'can_cast' of a type (line 30)
    can_cast_182546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 22), np_182545, 'can_cast')
    # Calling can_cast(args, kwargs) (line 30)
    can_cast_call_result_182550 = invoke(stypy.reporting.localization.Localization(__file__, 30, 22), can_cast_182546, *[row_182547, col_182548], **kwargs_182549)
    
    # Processing the call keyword arguments (line 30)
    kwargs_182551 = {}
    # Getting the type of 'int' (line 30)
    int_182544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 18), 'int', False)
    # Calling int(args, kwargs) (line 30)
    int_call_result_182552 = invoke(stypy.reporting.localization.Localization(__file__, 30, 18), int_182544, *[can_cast_call_result_182550], **kwargs_182551)
    
    # Processing the call keyword arguments (line 30)
    str_182553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 50), 'str', ' ')
    keyword_182554 = str_182553
    kwargs_182555 = {'end': keyword_182554}
    # Getting the type of 'print' (line 30)
    print_182543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'print', False)
    # Calling print(args, kwargs) (line 30)
    print_call_result_182556 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), print_182543, *[int_call_result_182552], **kwargs_182555)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_182558 = {}
    # Getting the type of 'print' (line 31)
    print_182557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'print', False)
    # Calling print(args, kwargs) (line 31)
    print_call_result_182559 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), print_182557, *[], **kwargs_182558)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'print_cancast_table(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_cancast_table' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_182560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_182560)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_cancast_table'
    return stypy_return_type_182560

# Assigning a type to the variable 'print_cancast_table' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'print_cancast_table', print_cancast_table)

@norecursion
def print_coercion_table(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 33)
    False_182561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 98), 'False')
    defaults = [False_182561]
    # Create a new context for function 'print_coercion_table'
    module_type_store = module_type_store.open_function_context('print_coercion_table', 33, 0, False)
    
    # Passed parameters checking function
    print_coercion_table.stypy_localization = localization
    print_coercion_table.stypy_type_of_self = None
    print_coercion_table.stypy_type_store = module_type_store
    print_coercion_table.stypy_function_name = 'print_coercion_table'
    print_coercion_table.stypy_param_names_list = ['ntypes', 'inputfirstvalue', 'inputsecondvalue', 'firstarray', 'use_promote_types']
    print_coercion_table.stypy_varargs_param_name = None
    print_coercion_table.stypy_kwargs_param_name = None
    print_coercion_table.stypy_call_defaults = defaults
    print_coercion_table.stypy_call_varargs = varargs
    print_coercion_table.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_coercion_table', ['ntypes', 'inputfirstvalue', 'inputsecondvalue', 'firstarray', 'use_promote_types'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_coercion_table', localization, ['ntypes', 'inputfirstvalue', 'inputsecondvalue', 'firstarray', 'use_promote_types'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_coercion_table(...)' code ##################

    
    # Call to print(...): (line 34)
    # Processing the call arguments (line 34)
    str_182563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 10), 'str', '+')
    # Processing the call keyword arguments (line 34)
    str_182564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'str', ' ')
    keyword_182565 = str_182564
    kwargs_182566 = {'end': keyword_182565}
    # Getting the type of 'print' (line 34)
    print_182562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'print', False)
    # Calling print(args, kwargs) (line 34)
    print_call_result_182567 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), print_182562, *[str_182563], **kwargs_182566)
    
    
    # Getting the type of 'ntypes' (line 35)
    ntypes_182568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'ntypes')
    # Testing the type of a for loop iterable (line 35)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 4), ntypes_182568)
    # Getting the type of the for loop variable (line 35)
    for_loop_var_182569 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 4), ntypes_182568)
    # Assigning a type to the variable 'char' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'char', for_loop_var_182569)
    # SSA begins for a for statement (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'char' (line 36)
    char_182571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 14), 'char', False)
    # Processing the call keyword arguments (line 36)
    str_182572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 24), 'str', ' ')
    keyword_182573 = str_182572
    kwargs_182574 = {'end': keyword_182573}
    # Getting the type of 'print' (line 36)
    print_182570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'print', False)
    # Calling print(args, kwargs) (line 36)
    print_call_result_182575 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), print_182570, *[char_182571], **kwargs_182574)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_182577 = {}
    # Getting the type of 'print' (line 37)
    print_182576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'print', False)
    # Calling print(args, kwargs) (line 37)
    print_call_result_182578 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), print_182576, *[], **kwargs_182577)
    
    
    # Getting the type of 'ntypes' (line 38)
    ntypes_182579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'ntypes')
    # Testing the type of a for loop iterable (line 38)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 4), ntypes_182579)
    # Getting the type of the for loop variable (line 38)
    for_loop_var_182580 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 4), ntypes_182579)
    # Assigning a type to the variable 'row' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'row', for_loop_var_182580)
    # SSA begins for a for statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'row' (line 39)
    row_182581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'row')
    str_182582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'str', 'O')
    # Applying the binary operator '==' (line 39)
    result_eq_182583 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 11), '==', row_182581, str_182582)
    
    # Testing the type of an if condition (line 39)
    if_condition_182584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 8), result_eq_182583)
    # Assigning a type to the variable 'if_condition_182584' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'if_condition_182584', if_condition_182584)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 40):
    # Getting the type of 'GenericObject' (line 40)
    GenericObject_182585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'GenericObject')
    # Assigning a type to the variable 'rowtype' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'rowtype', GenericObject_182585)
    # SSA branch for the else part of an if statement (line 39)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 42):
    
    # Call to obj2sctype(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'row' (line 42)
    row_182588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'row', False)
    # Processing the call keyword arguments (line 42)
    kwargs_182589 = {}
    # Getting the type of 'np' (line 42)
    np_182586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'np', False)
    # Obtaining the member 'obj2sctype' of a type (line 42)
    obj2sctype_182587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 22), np_182586, 'obj2sctype')
    # Calling obj2sctype(args, kwargs) (line 42)
    obj2sctype_call_result_182590 = invoke(stypy.reporting.localization.Localization(__file__, 42, 22), obj2sctype_182587, *[row_182588], **kwargs_182589)
    
    # Assigning a type to the variable 'rowtype' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'rowtype', obj2sctype_call_result_182590)
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'row' (line 44)
    row_182592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'row', False)
    # Processing the call keyword arguments (line 44)
    str_182593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'str', ' ')
    keyword_182594 = str_182593
    kwargs_182595 = {'end': keyword_182594}
    # Getting the type of 'print' (line 44)
    print_182591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'print', False)
    # Calling print(args, kwargs) (line 44)
    print_call_result_182596 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), print_182591, *[row_182592], **kwargs_182595)
    
    
    # Getting the type of 'ntypes' (line 45)
    ntypes_182597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'ntypes')
    # Testing the type of a for loop iterable (line 45)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 8), ntypes_182597)
    # Getting the type of the for loop variable (line 45)
    for_loop_var_182598 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 8), ntypes_182597)
    # Assigning a type to the variable 'col' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'col', for_loop_var_182598)
    # SSA begins for a for statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'col' (line 46)
    col_182599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'col')
    str_182600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'str', 'O')
    # Applying the binary operator '==' (line 46)
    result_eq_182601 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 15), '==', col_182599, str_182600)
    
    # Testing the type of an if condition (line 46)
    if_condition_182602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 12), result_eq_182601)
    # Assigning a type to the variable 'if_condition_182602' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'if_condition_182602', if_condition_182602)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 47):
    # Getting the type of 'GenericObject' (line 47)
    GenericObject_182603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'GenericObject')
    # Assigning a type to the variable 'coltype' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'coltype', GenericObject_182603)
    # SSA branch for the else part of an if statement (line 46)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 49):
    
    # Call to obj2sctype(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'col' (line 49)
    col_182606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 40), 'col', False)
    # Processing the call keyword arguments (line 49)
    kwargs_182607 = {}
    # Getting the type of 'np' (line 49)
    np_182604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'np', False)
    # Obtaining the member 'obj2sctype' of a type (line 49)
    obj2sctype_182605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 26), np_182604, 'obj2sctype')
    # Calling obj2sctype(args, kwargs) (line 49)
    obj2sctype_call_result_182608 = invoke(stypy.reporting.localization.Localization(__file__, 49, 26), obj2sctype_182605, *[col_182606], **kwargs_182607)
    
    # Assigning a type to the variable 'coltype' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'coltype', obj2sctype_call_result_182608)
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Getting the type of 'firstarray' (line 51)
    firstarray_182609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'firstarray')
    # Testing the type of an if condition (line 51)
    if_condition_182610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 16), firstarray_182609)
    # Assigning a type to the variable 'if_condition_182610' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'if_condition_182610', if_condition_182610)
    # SSA begins for if statement (line 51)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 52):
    
    # Call to array(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Obtaining an instance of the builtin type 'list' (line 52)
    list_182613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 52)
    # Adding element type (line 52)
    
    # Call to rowtype(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'inputfirstvalue' (line 52)
    inputfirstvalue_182615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 49), 'inputfirstvalue', False)
    # Processing the call keyword arguments (line 52)
    kwargs_182616 = {}
    # Getting the type of 'rowtype' (line 52)
    rowtype_182614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'rowtype', False)
    # Calling rowtype(args, kwargs) (line 52)
    rowtype_call_result_182617 = invoke(stypy.reporting.localization.Localization(__file__, 52, 41), rowtype_182614, *[inputfirstvalue_182615], **kwargs_182616)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 40), list_182613, rowtype_call_result_182617)
    
    # Processing the call keyword arguments (line 52)
    # Getting the type of 'rowtype' (line 52)
    rowtype_182618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 74), 'rowtype', False)
    keyword_182619 = rowtype_182618
    kwargs_182620 = {'dtype': keyword_182619}
    # Getting the type of 'np' (line 52)
    np_182611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'np', False)
    # Obtaining the member 'array' of a type (line 52)
    array_182612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 31), np_182611, 'array')
    # Calling array(args, kwargs) (line 52)
    array_call_result_182621 = invoke(stypy.reporting.localization.Localization(__file__, 52, 31), array_182612, *[list_182613], **kwargs_182620)
    
    # Assigning a type to the variable 'rowvalue' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'rowvalue', array_call_result_182621)
    # SSA branch for the else part of an if statement (line 51)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 54):
    
    # Call to rowtype(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'inputfirstvalue' (line 54)
    inputfirstvalue_182623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 'inputfirstvalue', False)
    # Processing the call keyword arguments (line 54)
    kwargs_182624 = {}
    # Getting the type of 'rowtype' (line 54)
    rowtype_182622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 31), 'rowtype', False)
    # Calling rowtype(args, kwargs) (line 54)
    rowtype_call_result_182625 = invoke(stypy.reporting.localization.Localization(__file__, 54, 31), rowtype_182622, *[inputfirstvalue_182623], **kwargs_182624)
    
    # Assigning a type to the variable 'rowvalue' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'rowvalue', rowtype_call_result_182625)
    # SSA join for if statement (line 51)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 55):
    
    # Call to coltype(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'inputsecondvalue' (line 55)
    inputsecondvalue_182627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'inputsecondvalue', False)
    # Processing the call keyword arguments (line 55)
    kwargs_182628 = {}
    # Getting the type of 'coltype' (line 55)
    coltype_182626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'coltype', False)
    # Calling coltype(args, kwargs) (line 55)
    coltype_call_result_182629 = invoke(stypy.reporting.localization.Localization(__file__, 55, 27), coltype_182626, *[inputsecondvalue_182627], **kwargs_182628)
    
    # Assigning a type to the variable 'colvalue' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'colvalue', coltype_call_result_182629)
    
    # Getting the type of 'use_promote_types' (line 56)
    use_promote_types_182630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'use_promote_types')
    # Testing the type of an if condition (line 56)
    if_condition_182631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 16), use_promote_types_182630)
    # Assigning a type to the variable 'if_condition_182631' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'if_condition_182631', if_condition_182631)
    # SSA begins for if statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 57):
    
    # Call to promote_types(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'rowvalue' (line 57)
    rowvalue_182634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 44), 'rowvalue', False)
    # Obtaining the member 'dtype' of a type (line 57)
    dtype_182635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 44), rowvalue_182634, 'dtype')
    # Getting the type of 'colvalue' (line 57)
    colvalue_182636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 60), 'colvalue', False)
    # Obtaining the member 'dtype' of a type (line 57)
    dtype_182637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 60), colvalue_182636, 'dtype')
    # Processing the call keyword arguments (line 57)
    kwargs_182638 = {}
    # Getting the type of 'np' (line 57)
    np_182632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 27), 'np', False)
    # Obtaining the member 'promote_types' of a type (line 57)
    promote_types_182633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 27), np_182632, 'promote_types')
    # Calling promote_types(args, kwargs) (line 57)
    promote_types_call_result_182639 = invoke(stypy.reporting.localization.Localization(__file__, 57, 27), promote_types_182633, *[dtype_182635, dtype_182637], **kwargs_182638)
    
    # Obtaining the member 'char' of a type (line 57)
    char_182640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 27), promote_types_call_result_182639, 'char')
    # Assigning a type to the variable 'char' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'char', char_182640)
    # SSA branch for the else part of an if statement (line 56)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 59):
    
    # Call to add(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'rowvalue' (line 59)
    rowvalue_182643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 35), 'rowvalue', False)
    # Getting the type of 'colvalue' (line 59)
    colvalue_182644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 45), 'colvalue', False)
    # Processing the call keyword arguments (line 59)
    kwargs_182645 = {}
    # Getting the type of 'np' (line 59)
    np_182641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'np', False)
    # Obtaining the member 'add' of a type (line 59)
    add_182642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 28), np_182641, 'add')
    # Calling add(args, kwargs) (line 59)
    add_call_result_182646 = invoke(stypy.reporting.localization.Localization(__file__, 59, 28), add_182642, *[rowvalue_182643, colvalue_182644], **kwargs_182645)
    
    # Assigning a type to the variable 'value' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'value', add_call_result_182646)
    
    
    # Call to isinstance(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'value' (line 60)
    value_182648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'value', False)
    # Getting the type of 'np' (line 60)
    np_182649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 41), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 60)
    ndarray_182650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 41), np_182649, 'ndarray')
    # Processing the call keyword arguments (line 60)
    kwargs_182651 = {}
    # Getting the type of 'isinstance' (line 60)
    isinstance_182647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 60)
    isinstance_call_result_182652 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), isinstance_182647, *[value_182648, ndarray_182650], **kwargs_182651)
    
    # Testing the type of an if condition (line 60)
    if_condition_182653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 20), isinstance_call_result_182652)
    # Assigning a type to the variable 'if_condition_182653' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'if_condition_182653', if_condition_182653)
    # SSA begins for if statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 61):
    # Getting the type of 'value' (line 61)
    value_182654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 31), 'value')
    # Obtaining the member 'dtype' of a type (line 61)
    dtype_182655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 31), value_182654, 'dtype')
    # Obtaining the member 'char' of a type (line 61)
    char_182656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 31), dtype_182655, 'char')
    # Assigning a type to the variable 'char' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'char', char_182656)
    # SSA branch for the else part of an if statement (line 60)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 63):
    
    # Call to dtype(...): (line 63)
    # Processing the call arguments (line 63)
    
    # Call to type(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'value' (line 63)
    value_182660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 45), 'value', False)
    # Processing the call keyword arguments (line 63)
    kwargs_182661 = {}
    # Getting the type of 'type' (line 63)
    type_182659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'type', False)
    # Calling type(args, kwargs) (line 63)
    type_call_result_182662 = invoke(stypy.reporting.localization.Localization(__file__, 63, 40), type_182659, *[value_182660], **kwargs_182661)
    
    # Processing the call keyword arguments (line 63)
    kwargs_182663 = {}
    # Getting the type of 'np' (line 63)
    np_182657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'np', False)
    # Obtaining the member 'dtype' of a type (line 63)
    dtype_182658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 31), np_182657, 'dtype')
    # Calling dtype(args, kwargs) (line 63)
    dtype_call_result_182664 = invoke(stypy.reporting.localization.Localization(__file__, 63, 31), dtype_182658, *[type_call_result_182662], **kwargs_182663)
    
    # Obtaining the member 'char' of a type (line 63)
    char_182665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 31), dtype_call_result_182664, 'char')
    # Assigning a type to the variable 'char' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'char', char_182665)
    # SSA join for if statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 56)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 50)
    # SSA branch for the except 'ValueError' branch of a try statement (line 50)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Str to a Name (line 65):
    str_182666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'str', '!')
    # Assigning a type to the variable 'char' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'char', str_182666)
    # SSA branch for the except 'OverflowError' branch of a try statement (line 50)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Str to a Name (line 67):
    str_182667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'str', '@')
    # Assigning a type to the variable 'char' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'char', str_182667)
    # SSA branch for the except 'TypeError' branch of a try statement (line 50)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Str to a Name (line 69):
    str_182668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'str', '#')
    # Assigning a type to the variable 'char' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'char', str_182668)
    # SSA join for try-except statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'char' (line 70)
    char_182670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'char', False)
    # Processing the call keyword arguments (line 70)
    str_182671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'str', ' ')
    keyword_182672 = str_182671
    kwargs_182673 = {'end': keyword_182672}
    # Getting the type of 'print' (line 70)
    print_182669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'print', False)
    # Calling print(args, kwargs) (line 70)
    print_call_result_182674 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), print_182669, *[char_182670], **kwargs_182673)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 71)
    # Processing the call keyword arguments (line 71)
    kwargs_182676 = {}
    # Getting the type of 'print' (line 71)
    print_182675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'print', False)
    # Calling print(args, kwargs) (line 71)
    print_call_result_182677 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), print_182675, *[], **kwargs_182676)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'print_coercion_table(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_coercion_table' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_182678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_182678)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_coercion_table'
    return stypy_return_type_182678

# Assigning a type to the variable 'print_coercion_table' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'print_coercion_table', print_coercion_table)

# Call to print(...): (line 73)
# Processing the call arguments (line 73)
str_182680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 6), 'str', 'can cast')
# Processing the call keyword arguments (line 73)
kwargs_182681 = {}
# Getting the type of 'print' (line 73)
print_182679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'print', False)
# Calling print(args, kwargs) (line 73)
print_call_result_182682 = invoke(stypy.reporting.localization.Localization(__file__, 73, 0), print_182679, *[str_182680], **kwargs_182681)


# Call to print_cancast_table(...): (line 74)
# Processing the call arguments (line 74)

# Obtaining the type of the subscript
str_182684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 33), 'str', 'All')
# Getting the type of 'np' (line 74)
np_182685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'np', False)
# Obtaining the member 'typecodes' of a type (line 74)
typecodes_182686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 20), np_182685, 'typecodes')
# Obtaining the member '__getitem__' of a type (line 74)
getitem___182687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 20), typecodes_182686, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 74)
subscript_call_result_182688 = invoke(stypy.reporting.localization.Localization(__file__, 74, 20), getitem___182687, str_182684)

# Processing the call keyword arguments (line 74)
kwargs_182689 = {}
# Getting the type of 'print_cancast_table' (line 74)
print_cancast_table_182683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'print_cancast_table', False)
# Calling print_cancast_table(args, kwargs) (line 74)
print_cancast_table_call_result_182690 = invoke(stypy.reporting.localization.Localization(__file__, 74, 0), print_cancast_table_182683, *[subscript_call_result_182688], **kwargs_182689)


# Call to print(...): (line 75)
# Processing the call keyword arguments (line 75)
kwargs_182692 = {}
# Getting the type of 'print' (line 75)
print_182691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'print', False)
# Calling print(args, kwargs) (line 75)
print_call_result_182693 = invoke(stypy.reporting.localization.Localization(__file__, 75, 0), print_182691, *[], **kwargs_182692)


# Call to print(...): (line 76)
# Processing the call arguments (line 76)
str_182695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 6), 'str', "In these tables, ValueError is '!', OverflowError is '@', TypeError is '#'")
# Processing the call keyword arguments (line 76)
kwargs_182696 = {}
# Getting the type of 'print' (line 76)
print_182694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'print', False)
# Calling print(args, kwargs) (line 76)
print_call_result_182697 = invoke(stypy.reporting.localization.Localization(__file__, 76, 0), print_182694, *[str_182695], **kwargs_182696)


# Call to print(...): (line 77)
# Processing the call keyword arguments (line 77)
kwargs_182699 = {}
# Getting the type of 'print' (line 77)
print_182698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'print', False)
# Calling print(args, kwargs) (line 77)
print_call_result_182700 = invoke(stypy.reporting.localization.Localization(__file__, 77, 0), print_182698, *[], **kwargs_182699)


# Call to print(...): (line 78)
# Processing the call arguments (line 78)
str_182702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 6), 'str', 'scalar + scalar')
# Processing the call keyword arguments (line 78)
kwargs_182703 = {}
# Getting the type of 'print' (line 78)
print_182701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'print', False)
# Calling print(args, kwargs) (line 78)
print_call_result_182704 = invoke(stypy.reporting.localization.Localization(__file__, 78, 0), print_182701, *[str_182702], **kwargs_182703)


# Call to print_coercion_table(...): (line 79)
# Processing the call arguments (line 79)

# Obtaining the type of the subscript
str_182706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 34), 'str', 'All')
# Getting the type of 'np' (line 79)
np_182707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'np', False)
# Obtaining the member 'typecodes' of a type (line 79)
typecodes_182708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 21), np_182707, 'typecodes')
# Obtaining the member '__getitem__' of a type (line 79)
getitem___182709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 21), typecodes_182708, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 79)
subscript_call_result_182710 = invoke(stypy.reporting.localization.Localization(__file__, 79, 21), getitem___182709, str_182706)

int_182711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 42), 'int')
int_182712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 45), 'int')
# Getting the type of 'False' (line 79)
False_182713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 48), 'False', False)
# Processing the call keyword arguments (line 79)
kwargs_182714 = {}
# Getting the type of 'print_coercion_table' (line 79)
print_coercion_table_182705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'print_coercion_table', False)
# Calling print_coercion_table(args, kwargs) (line 79)
print_coercion_table_call_result_182715 = invoke(stypy.reporting.localization.Localization(__file__, 79, 0), print_coercion_table_182705, *[subscript_call_result_182710, int_182711, int_182712, False_182713], **kwargs_182714)


# Call to print(...): (line 80)
# Processing the call keyword arguments (line 80)
kwargs_182717 = {}
# Getting the type of 'print' (line 80)
print_182716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'print', False)
# Calling print(args, kwargs) (line 80)
print_call_result_182718 = invoke(stypy.reporting.localization.Localization(__file__, 80, 0), print_182716, *[], **kwargs_182717)


# Call to print(...): (line 81)
# Processing the call arguments (line 81)
str_182720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 6), 'str', 'scalar + neg scalar')
# Processing the call keyword arguments (line 81)
kwargs_182721 = {}
# Getting the type of 'print' (line 81)
print_182719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'print', False)
# Calling print(args, kwargs) (line 81)
print_call_result_182722 = invoke(stypy.reporting.localization.Localization(__file__, 81, 0), print_182719, *[str_182720], **kwargs_182721)


# Call to print_coercion_table(...): (line 82)
# Processing the call arguments (line 82)

# Obtaining the type of the subscript
str_182724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 34), 'str', 'All')
# Getting the type of 'np' (line 82)
np_182725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'np', False)
# Obtaining the member 'typecodes' of a type (line 82)
typecodes_182726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), np_182725, 'typecodes')
# Obtaining the member '__getitem__' of a type (line 82)
getitem___182727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), typecodes_182726, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 82)
subscript_call_result_182728 = invoke(stypy.reporting.localization.Localization(__file__, 82, 21), getitem___182727, str_182724)

int_182729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 42), 'int')
int_182730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 45), 'int')
# Getting the type of 'False' (line 82)
False_182731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 49), 'False', False)
# Processing the call keyword arguments (line 82)
kwargs_182732 = {}
# Getting the type of 'print_coercion_table' (line 82)
print_coercion_table_182723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'print_coercion_table', False)
# Calling print_coercion_table(args, kwargs) (line 82)
print_coercion_table_call_result_182733 = invoke(stypy.reporting.localization.Localization(__file__, 82, 0), print_coercion_table_182723, *[subscript_call_result_182728, int_182729, int_182730, False_182731], **kwargs_182732)


# Call to print(...): (line 83)
# Processing the call keyword arguments (line 83)
kwargs_182735 = {}
# Getting the type of 'print' (line 83)
print_182734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'print', False)
# Calling print(args, kwargs) (line 83)
print_call_result_182736 = invoke(stypy.reporting.localization.Localization(__file__, 83, 0), print_182734, *[], **kwargs_182735)


# Call to print(...): (line 84)
# Processing the call arguments (line 84)
str_182738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 6), 'str', 'array + scalar')
# Processing the call keyword arguments (line 84)
kwargs_182739 = {}
# Getting the type of 'print' (line 84)
print_182737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'print', False)
# Calling print(args, kwargs) (line 84)
print_call_result_182740 = invoke(stypy.reporting.localization.Localization(__file__, 84, 0), print_182737, *[str_182738], **kwargs_182739)


# Call to print_coercion_table(...): (line 85)
# Processing the call arguments (line 85)

# Obtaining the type of the subscript
str_182742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 34), 'str', 'All')
# Getting the type of 'np' (line 85)
np_182743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'np', False)
# Obtaining the member 'typecodes' of a type (line 85)
typecodes_182744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), np_182743, 'typecodes')
# Obtaining the member '__getitem__' of a type (line 85)
getitem___182745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), typecodes_182744, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 85)
subscript_call_result_182746 = invoke(stypy.reporting.localization.Localization(__file__, 85, 21), getitem___182745, str_182742)

int_182747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 42), 'int')
int_182748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 45), 'int')
# Getting the type of 'True' (line 85)
True_182749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 48), 'True', False)
# Processing the call keyword arguments (line 85)
kwargs_182750 = {}
# Getting the type of 'print_coercion_table' (line 85)
print_coercion_table_182741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'print_coercion_table', False)
# Calling print_coercion_table(args, kwargs) (line 85)
print_coercion_table_call_result_182751 = invoke(stypy.reporting.localization.Localization(__file__, 85, 0), print_coercion_table_182741, *[subscript_call_result_182746, int_182747, int_182748, True_182749], **kwargs_182750)


# Call to print(...): (line 86)
# Processing the call keyword arguments (line 86)
kwargs_182753 = {}
# Getting the type of 'print' (line 86)
print_182752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'print', False)
# Calling print(args, kwargs) (line 86)
print_call_result_182754 = invoke(stypy.reporting.localization.Localization(__file__, 86, 0), print_182752, *[], **kwargs_182753)


# Call to print(...): (line 87)
# Processing the call arguments (line 87)
str_182756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 6), 'str', 'array + neg scalar')
# Processing the call keyword arguments (line 87)
kwargs_182757 = {}
# Getting the type of 'print' (line 87)
print_182755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'print', False)
# Calling print(args, kwargs) (line 87)
print_call_result_182758 = invoke(stypy.reporting.localization.Localization(__file__, 87, 0), print_182755, *[str_182756], **kwargs_182757)


# Call to print_coercion_table(...): (line 88)
# Processing the call arguments (line 88)

# Obtaining the type of the subscript
str_182760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 34), 'str', 'All')
# Getting the type of 'np' (line 88)
np_182761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 21), 'np', False)
# Obtaining the member 'typecodes' of a type (line 88)
typecodes_182762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 21), np_182761, 'typecodes')
# Obtaining the member '__getitem__' of a type (line 88)
getitem___182763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 21), typecodes_182762, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 88)
subscript_call_result_182764 = invoke(stypy.reporting.localization.Localization(__file__, 88, 21), getitem___182763, str_182760)

int_182765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 42), 'int')
int_182766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 45), 'int')
# Getting the type of 'True' (line 88)
True_182767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 49), 'True', False)
# Processing the call keyword arguments (line 88)
kwargs_182768 = {}
# Getting the type of 'print_coercion_table' (line 88)
print_coercion_table_182759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'print_coercion_table', False)
# Calling print_coercion_table(args, kwargs) (line 88)
print_coercion_table_call_result_182769 = invoke(stypy.reporting.localization.Localization(__file__, 88, 0), print_coercion_table_182759, *[subscript_call_result_182764, int_182765, int_182766, True_182767], **kwargs_182768)


# Call to print(...): (line 89)
# Processing the call keyword arguments (line 89)
kwargs_182771 = {}
# Getting the type of 'print' (line 89)
print_182770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'print', False)
# Calling print(args, kwargs) (line 89)
print_call_result_182772 = invoke(stypy.reporting.localization.Localization(__file__, 89, 0), print_182770, *[], **kwargs_182771)


# Call to print(...): (line 90)
# Processing the call arguments (line 90)
str_182774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 6), 'str', 'promote_types')
# Processing the call keyword arguments (line 90)
kwargs_182775 = {}
# Getting the type of 'print' (line 90)
print_182773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'print', False)
# Calling print(args, kwargs) (line 90)
print_call_result_182776 = invoke(stypy.reporting.localization.Localization(__file__, 90, 0), print_182773, *[str_182774], **kwargs_182775)


# Call to print_coercion_table(...): (line 91)
# Processing the call arguments (line 91)

# Obtaining the type of the subscript
str_182778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 34), 'str', 'All')
# Getting the type of 'np' (line 91)
np_182779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'np', False)
# Obtaining the member 'typecodes' of a type (line 91)
typecodes_182780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 21), np_182779, 'typecodes')
# Obtaining the member '__getitem__' of a type (line 91)
getitem___182781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 21), typecodes_182780, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 91)
subscript_call_result_182782 = invoke(stypy.reporting.localization.Localization(__file__, 91, 21), getitem___182781, str_182778)

int_182783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 42), 'int')
int_182784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 45), 'int')
# Getting the type of 'False' (line 91)
False_182785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 48), 'False', False)
# Getting the type of 'True' (line 91)
True_182786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 55), 'True', False)
# Processing the call keyword arguments (line 91)
kwargs_182787 = {}
# Getting the type of 'print_coercion_table' (line 91)
print_coercion_table_182777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'print_coercion_table', False)
# Calling print_coercion_table(args, kwargs) (line 91)
print_coercion_table_call_result_182788 = invoke(stypy.reporting.localization.Localization(__file__, 91, 0), print_coercion_table_182777, *[subscript_call_result_182782, int_182783, int_182784, False_182785, True_182786], **kwargs_182787)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
