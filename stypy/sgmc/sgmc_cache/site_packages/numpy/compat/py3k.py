
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Python 3 compatibility tools.
3: 
4: '''
5: from __future__ import division, absolute_import, print_function
6: 
7: __all__ = ['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar',
8:            'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested',
9:            'asstr', 'open_latin1', 'long', 'basestring', 'sixu',
10:            'integer_types']
11: 
12: import sys
13: 
14: if sys.version_info[0] >= 3:
15:     import io
16: 
17:     long = int
18:     integer_types = (int,)
19:     basestring = str
20:     unicode = str
21:     bytes = bytes
22: 
23:     def asunicode(s):
24:         if isinstance(s, bytes):
25:             return s.decode('latin1')
26:         return str(s)
27: 
28:     def asbytes(s):
29:         if isinstance(s, bytes):
30:             return s
31:         return str(s).encode('latin1')
32: 
33:     def asstr(s):
34:         if isinstance(s, bytes):
35:             return s.decode('latin1')
36:         return str(s)
37: 
38:     def isfileobj(f):
39:         return isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter))
40: 
41:     def open_latin1(filename, mode='r'):
42:         return open(filename, mode=mode, encoding='iso-8859-1')
43: 
44:     def sixu(s):
45:         return s
46: 
47:     strchar = 'U'
48: 
49: 
50: else:
51:     bytes = str
52:     long = long
53:     basestring = basestring
54:     unicode = unicode
55:     integer_types = (int, long)
56:     asbytes = str
57:     asstr = str
58:     strchar = 'S'
59: 
60:     def isfileobj(f):
61:         return isinstance(f, file)
62: 
63:     def asunicode(s):
64:         if isinstance(s, unicode):
65:             return s
66:         return str(s).decode('ascii')
67: 
68:     def open_latin1(filename, mode='r'):
69:         return open(filename, mode=mode)
70: 
71:     def sixu(s):
72:         return unicode(s, 'unicode_escape')
73: 
74: 
75: def getexception():
76:     return sys.exc_info()[1]
77: 
78: def asbytes_nested(x):
79:     if hasattr(x, '__iter__') and not isinstance(x, (bytes, unicode)):
80:         return [asbytes_nested(y) for y in x]
81:     else:
82:         return asbytes(x)
83: 
84: def asunicode_nested(x):
85:     if hasattr(x, '__iter__') and not isinstance(x, (bytes, unicode)):
86:         return [asunicode_nested(y) for y in x]
87:     else:
88:         return asunicode(x)
89: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_25621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nPython 3 compatibility tools.\n\n')

# Assigning a List to a Name (line 7):
__all__ = ['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar', 'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested', 'asstr', 'open_latin1', 'long', 'basestring', 'sixu', 'integer_types']
module_type_store.set_exportable_members(['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar', 'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested', 'asstr', 'open_latin1', 'long', 'basestring', 'sixu', 'integer_types'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_25622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_25623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'bytes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25623)
# Adding element type (line 7)
str_25624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'str', 'asbytes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25624)
# Adding element type (line 7)
str_25625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 31), 'str', 'isfileobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25625)
# Adding element type (line 7)
str_25626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 44), 'str', 'getexception')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25626)
# Adding element type (line 7)
str_25627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 60), 'str', 'strchar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25627)
# Adding element type (line 7)
str_25628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'unicode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25628)
# Adding element type (line 7)
str_25629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 22), 'str', 'asunicode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25629)
# Adding element type (line 7)
str_25630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 35), 'str', 'asbytes_nested')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25630)
# Adding element type (line 7)
str_25631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 53), 'str', 'asunicode_nested')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25631)
# Adding element type (line 7)
str_25632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'asstr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25632)
# Adding element type (line 7)
str_25633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'str', 'open_latin1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25633)
# Adding element type (line 7)
str_25634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 35), 'str', 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25634)
# Adding element type (line 7)
str_25635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 43), 'str', 'basestring')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25635)
# Adding element type (line 7)
str_25636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 57), 'str', 'sixu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25636)
# Adding element type (line 7)
str_25637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'integer_types')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25622, str_25637)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_25622)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import sys' statement (line 12)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'sys', sys, module_type_store)




# Obtaining the type of the subscript
int_25638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'int')
# Getting the type of 'sys' (line 14)
sys_25639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 14)
version_info_25640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 3), sys_25639, 'version_info')
# Obtaining the member '__getitem__' of a type (line 14)
getitem___25641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 3), version_info_25640, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_25642 = invoke(stypy.reporting.localization.Localization(__file__, 14, 3), getitem___25641, int_25638)

int_25643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'int')
# Applying the binary operator '>=' (line 14)
result_ge_25644 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 3), '>=', subscript_call_result_25642, int_25643)

# Testing the type of an if condition (line 14)
if_condition_25645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 0), result_ge_25644)
# Assigning a type to the variable 'if_condition_25645' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'if_condition_25645', if_condition_25645)
# SSA begins for if statement (line 14)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 4))

# 'import io' statement (line 15)
import io

import_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'io', io, module_type_store)


# Assigning a Name to a Name (line 17):
# Getting the type of 'int' (line 17)
int_25646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'int')
# Assigning a type to the variable 'long' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'long', int_25646)

# Assigning a Tuple to a Name (line 18):

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_25647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
# Getting the type of 'int' (line 18)
int_25648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 21), tuple_25647, int_25648)

# Assigning a type to the variable 'integer_types' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'integer_types', tuple_25647)

# Assigning a Name to a Name (line 19):
# Getting the type of 'str' (line 19)
str_25649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'str')
# Assigning a type to the variable 'basestring' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'basestring', str_25649)

# Assigning a Name to a Name (line 20):
# Getting the type of 'str' (line 20)
str_25650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'str')
# Assigning a type to the variable 'unicode' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'unicode', str_25650)

# Assigning a Name to a Name (line 21):
# Getting the type of 'bytes' (line 21)
bytes_25651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'bytes')
# Assigning a type to the variable 'bytes' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'bytes', bytes_25651)

@norecursion
def asunicode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'asunicode'
    module_type_store = module_type_store.open_function_context('asunicode', 23, 4, False)
    
    # Passed parameters checking function
    asunicode.stypy_localization = localization
    asunicode.stypy_type_of_self = None
    asunicode.stypy_type_store = module_type_store
    asunicode.stypy_function_name = 'asunicode'
    asunicode.stypy_param_names_list = ['s']
    asunicode.stypy_varargs_param_name = None
    asunicode.stypy_kwargs_param_name = None
    asunicode.stypy_call_defaults = defaults
    asunicode.stypy_call_varargs = varargs
    asunicode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asunicode', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asunicode', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asunicode(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 24)
    # Getting the type of 'bytes' (line 24)
    bytes_25652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'bytes')
    # Getting the type of 's' (line 24)
    s_25653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 's')
    
    (may_be_25654, more_types_in_union_25655) = may_be_subtype(bytes_25652, s_25653)

    if may_be_25654:

        if more_types_in_union_25655:
            # Runtime conditional SSA (line 24)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 's' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 's', remove_not_subtype_from_union(s_25653, bytes))
        
        # Call to decode(...): (line 25)
        # Processing the call arguments (line 25)
        str_25658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'str', 'latin1')
        # Processing the call keyword arguments (line 25)
        kwargs_25659 = {}
        # Getting the type of 's' (line 25)
        s_25656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 's', False)
        # Obtaining the member 'decode' of a type (line 25)
        decode_25657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), s_25656, 'decode')
        # Calling decode(args, kwargs) (line 25)
        decode_call_result_25660 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), decode_25657, *[str_25658], **kwargs_25659)
        
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'stypy_return_type', decode_call_result_25660)

        if more_types_in_union_25655:
            # SSA join for if statement (line 24)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to str(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 's' (line 26)
    s_25662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 19), 's', False)
    # Processing the call keyword arguments (line 26)
    kwargs_25663 = {}
    # Getting the type of 'str' (line 26)
    str_25661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'str', False)
    # Calling str(args, kwargs) (line 26)
    str_call_result_25664 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), str_25661, *[s_25662], **kwargs_25663)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', str_call_result_25664)
    
    # ################# End of 'asunicode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asunicode' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_25665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25665)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asunicode'
    return stypy_return_type_25665

# Assigning a type to the variable 'asunicode' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'asunicode', asunicode)

@norecursion
def asbytes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'asbytes'
    module_type_store = module_type_store.open_function_context('asbytes', 28, 4, False)
    
    # Passed parameters checking function
    asbytes.stypy_localization = localization
    asbytes.stypy_type_of_self = None
    asbytes.stypy_type_store = module_type_store
    asbytes.stypy_function_name = 'asbytes'
    asbytes.stypy_param_names_list = ['s']
    asbytes.stypy_varargs_param_name = None
    asbytes.stypy_kwargs_param_name = None
    asbytes.stypy_call_defaults = defaults
    asbytes.stypy_call_varargs = varargs
    asbytes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asbytes', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asbytes', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asbytes(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 29)
    # Getting the type of 'bytes' (line 29)
    bytes_25666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 25), 'bytes')
    # Getting the type of 's' (line 29)
    s_25667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 's')
    
    (may_be_25668, more_types_in_union_25669) = may_be_subtype(bytes_25666, s_25667)

    if may_be_25668:

        if more_types_in_union_25669:
            # Runtime conditional SSA (line 29)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 's' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 's', remove_not_subtype_from_union(s_25667, bytes))
        # Getting the type of 's' (line 30)
        s_25670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'stypy_return_type', s_25670)

        if more_types_in_union_25669:
            # SSA join for if statement (line 29)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to encode(...): (line 31)
    # Processing the call arguments (line 31)
    str_25676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'str', 'latin1')
    # Processing the call keyword arguments (line 31)
    kwargs_25677 = {}
    
    # Call to str(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 's' (line 31)
    s_25672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 's', False)
    # Processing the call keyword arguments (line 31)
    kwargs_25673 = {}
    # Getting the type of 'str' (line 31)
    str_25671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'str', False)
    # Calling str(args, kwargs) (line 31)
    str_call_result_25674 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), str_25671, *[s_25672], **kwargs_25673)
    
    # Obtaining the member 'encode' of a type (line 31)
    encode_25675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), str_call_result_25674, 'encode')
    # Calling encode(args, kwargs) (line 31)
    encode_call_result_25678 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), encode_25675, *[str_25676], **kwargs_25677)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', encode_call_result_25678)
    
    # ################# End of 'asbytes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asbytes' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_25679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25679)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asbytes'
    return stypy_return_type_25679

# Assigning a type to the variable 'asbytes' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'asbytes', asbytes)

@norecursion
def asstr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'asstr'
    module_type_store = module_type_store.open_function_context('asstr', 33, 4, False)
    
    # Passed parameters checking function
    asstr.stypy_localization = localization
    asstr.stypy_type_of_self = None
    asstr.stypy_type_store = module_type_store
    asstr.stypy_function_name = 'asstr'
    asstr.stypy_param_names_list = ['s']
    asstr.stypy_varargs_param_name = None
    asstr.stypy_kwargs_param_name = None
    asstr.stypy_call_defaults = defaults
    asstr.stypy_call_varargs = varargs
    asstr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asstr', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asstr', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asstr(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 34)
    # Getting the type of 'bytes' (line 34)
    bytes_25680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'bytes')
    # Getting the type of 's' (line 34)
    s_25681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 22), 's')
    
    (may_be_25682, more_types_in_union_25683) = may_be_subtype(bytes_25680, s_25681)

    if may_be_25682:

        if more_types_in_union_25683:
            # Runtime conditional SSA (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 's' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 's', remove_not_subtype_from_union(s_25681, bytes))
        
        # Call to decode(...): (line 35)
        # Processing the call arguments (line 35)
        str_25686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'str', 'latin1')
        # Processing the call keyword arguments (line 35)
        kwargs_25687 = {}
        # Getting the type of 's' (line 35)
        s_25684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 's', False)
        # Obtaining the member 'decode' of a type (line 35)
        decode_25685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 19), s_25684, 'decode')
        # Calling decode(args, kwargs) (line 35)
        decode_call_result_25688 = invoke(stypy.reporting.localization.Localization(__file__, 35, 19), decode_25685, *[str_25686], **kwargs_25687)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'stypy_return_type', decode_call_result_25688)

        if more_types_in_union_25683:
            # SSA join for if statement (line 34)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to str(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 's' (line 36)
    s_25690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 's', False)
    # Processing the call keyword arguments (line 36)
    kwargs_25691 = {}
    # Getting the type of 'str' (line 36)
    str_25689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'str', False)
    # Calling str(args, kwargs) (line 36)
    str_call_result_25692 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), str_25689, *[s_25690], **kwargs_25691)
    
    # Assigning a type to the variable 'stypy_return_type' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', str_call_result_25692)
    
    # ################# End of 'asstr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asstr' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_25693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25693)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asstr'
    return stypy_return_type_25693

# Assigning a type to the variable 'asstr' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'asstr', asstr)

@norecursion
def isfileobj(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isfileobj'
    module_type_store = module_type_store.open_function_context('isfileobj', 38, 4, False)
    
    # Passed parameters checking function
    isfileobj.stypy_localization = localization
    isfileobj.stypy_type_of_self = None
    isfileobj.stypy_type_store = module_type_store
    isfileobj.stypy_function_name = 'isfileobj'
    isfileobj.stypy_param_names_list = ['f']
    isfileobj.stypy_varargs_param_name = None
    isfileobj.stypy_kwargs_param_name = None
    isfileobj.stypy_call_defaults = defaults
    isfileobj.stypy_call_varargs = varargs
    isfileobj.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isfileobj', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isfileobj', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isfileobj(...)' code ##################

    
    # Call to isinstance(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'f' (line 39)
    f_25695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'f', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 39)
    tuple_25696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 39)
    # Adding element type (line 39)
    # Getting the type of 'io' (line 39)
    io_25697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 30), 'io', False)
    # Obtaining the member 'FileIO' of a type (line 39)
    FileIO_25698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 30), io_25697, 'FileIO')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 30), tuple_25696, FileIO_25698)
    # Adding element type (line 39)
    # Getting the type of 'io' (line 39)
    io_25699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 41), 'io', False)
    # Obtaining the member 'BufferedReader' of a type (line 39)
    BufferedReader_25700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 41), io_25699, 'BufferedReader')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 30), tuple_25696, BufferedReader_25700)
    # Adding element type (line 39)
    # Getting the type of 'io' (line 39)
    io_25701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 60), 'io', False)
    # Obtaining the member 'BufferedWriter' of a type (line 39)
    BufferedWriter_25702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 60), io_25701, 'BufferedWriter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 30), tuple_25696, BufferedWriter_25702)
    
    # Processing the call keyword arguments (line 39)
    kwargs_25703 = {}
    # Getting the type of 'isinstance' (line 39)
    isinstance_25694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 39)
    isinstance_call_result_25704 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), isinstance_25694, *[f_25695, tuple_25696], **kwargs_25703)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', isinstance_call_result_25704)
    
    # ################# End of 'isfileobj(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isfileobj' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_25705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25705)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isfileobj'
    return stypy_return_type_25705

# Assigning a type to the variable 'isfileobj' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'isfileobj', isfileobj)

@norecursion
def open_latin1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_25706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 35), 'str', 'r')
    defaults = [str_25706]
    # Create a new context for function 'open_latin1'
    module_type_store = module_type_store.open_function_context('open_latin1', 41, 4, False)
    
    # Passed parameters checking function
    open_latin1.stypy_localization = localization
    open_latin1.stypy_type_of_self = None
    open_latin1.stypy_type_store = module_type_store
    open_latin1.stypy_function_name = 'open_latin1'
    open_latin1.stypy_param_names_list = ['filename', 'mode']
    open_latin1.stypy_varargs_param_name = None
    open_latin1.stypy_kwargs_param_name = None
    open_latin1.stypy_call_defaults = defaults
    open_latin1.stypy_call_varargs = varargs
    open_latin1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'open_latin1', ['filename', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'open_latin1', localization, ['filename', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'open_latin1(...)' code ##################

    
    # Call to open(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'filename' (line 42)
    filename_25708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'filename', False)
    # Processing the call keyword arguments (line 42)
    # Getting the type of 'mode' (line 42)
    mode_25709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 35), 'mode', False)
    keyword_25710 = mode_25709
    str_25711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 50), 'str', 'iso-8859-1')
    keyword_25712 = str_25711
    kwargs_25713 = {'mode': keyword_25710, 'encoding': keyword_25712}
    # Getting the type of 'open' (line 42)
    open_25707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'open', False)
    # Calling open(args, kwargs) (line 42)
    open_call_result_25714 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), open_25707, *[filename_25708], **kwargs_25713)
    
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', open_call_result_25714)
    
    # ################# End of 'open_latin1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'open_latin1' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_25715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25715)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'open_latin1'
    return stypy_return_type_25715

# Assigning a type to the variable 'open_latin1' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'open_latin1', open_latin1)

@norecursion
def sixu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sixu'
    module_type_store = module_type_store.open_function_context('sixu', 44, 4, False)
    
    # Passed parameters checking function
    sixu.stypy_localization = localization
    sixu.stypy_type_of_self = None
    sixu.stypy_type_store = module_type_store
    sixu.stypy_function_name = 'sixu'
    sixu.stypy_param_names_list = ['s']
    sixu.stypy_varargs_param_name = None
    sixu.stypy_kwargs_param_name = None
    sixu.stypy_call_defaults = defaults
    sixu.stypy_call_varargs = varargs
    sixu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sixu', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sixu', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sixu(...)' code ##################

    # Getting the type of 's' (line 45)
    s_25716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', s_25716)
    
    # ################# End of 'sixu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sixu' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_25717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25717)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sixu'
    return stypy_return_type_25717

# Assigning a type to the variable 'sixu' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'sixu', sixu)

# Assigning a Str to a Name (line 47):
str_25718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 14), 'str', 'U')
# Assigning a type to the variable 'strchar' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'strchar', str_25718)
# SSA branch for the else part of an if statement (line 14)
module_type_store.open_ssa_branch('else')

# Assigning a Name to a Name (line 51):
# Getting the type of 'str' (line 51)
str_25719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'str')
# Assigning a type to the variable 'bytes' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'bytes', str_25719)

# Assigning a Name to a Name (line 52):
# Getting the type of 'long' (line 52)
long_25720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'long')
# Assigning a type to the variable 'long' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'long', long_25720)

# Assigning a Name to a Name (line 53):
# Getting the type of 'basestring' (line 53)
basestring_25721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'basestring')
# Assigning a type to the variable 'basestring' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'basestring', basestring_25721)

# Assigning a Name to a Name (line 54):
# Getting the type of 'unicode' (line 54)
unicode_25722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 14), 'unicode')
# Assigning a type to the variable 'unicode' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'unicode', unicode_25722)

# Assigning a Tuple to a Name (line 55):

# Obtaining an instance of the builtin type 'tuple' (line 55)
tuple_25723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 55)
# Adding element type (line 55)
# Getting the type of 'int' (line 55)
int_25724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), tuple_25723, int_25724)
# Adding element type (line 55)
# Getting the type of 'long' (line 55)
long_25725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 26), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), tuple_25723, long_25725)

# Assigning a type to the variable 'integer_types' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'integer_types', tuple_25723)

# Assigning a Name to a Name (line 56):
# Getting the type of 'str' (line 56)
str_25726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'str')
# Assigning a type to the variable 'asbytes' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'asbytes', str_25726)

# Assigning a Name to a Name (line 57):
# Getting the type of 'str' (line 57)
str_25727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'str')
# Assigning a type to the variable 'asstr' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'asstr', str_25727)

# Assigning a Str to a Name (line 58):
str_25728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 14), 'str', 'S')
# Assigning a type to the variable 'strchar' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'strchar', str_25728)

@norecursion
def isfileobj(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isfileobj'
    module_type_store = module_type_store.open_function_context('isfileobj', 60, 4, False)
    
    # Passed parameters checking function
    isfileobj.stypy_localization = localization
    isfileobj.stypy_type_of_self = None
    isfileobj.stypy_type_store = module_type_store
    isfileobj.stypy_function_name = 'isfileobj'
    isfileobj.stypy_param_names_list = ['f']
    isfileobj.stypy_varargs_param_name = None
    isfileobj.stypy_kwargs_param_name = None
    isfileobj.stypy_call_defaults = defaults
    isfileobj.stypy_call_varargs = varargs
    isfileobj.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isfileobj', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isfileobj', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isfileobj(...)' code ##################

    
    # Call to isinstance(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'f' (line 61)
    f_25730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'f', False)
    # Getting the type of 'file' (line 61)
    file_25731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'file', False)
    # Processing the call keyword arguments (line 61)
    kwargs_25732 = {}
    # Getting the type of 'isinstance' (line 61)
    isinstance_25729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 61)
    isinstance_call_result_25733 = invoke(stypy.reporting.localization.Localization(__file__, 61, 15), isinstance_25729, *[f_25730, file_25731], **kwargs_25732)
    
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', isinstance_call_result_25733)
    
    # ################# End of 'isfileobj(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isfileobj' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_25734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25734)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isfileobj'
    return stypy_return_type_25734

# Assigning a type to the variable 'isfileobj' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'isfileobj', isfileobj)

@norecursion
def asunicode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'asunicode'
    module_type_store = module_type_store.open_function_context('asunicode', 63, 4, False)
    
    # Passed parameters checking function
    asunicode.stypy_localization = localization
    asunicode.stypy_type_of_self = None
    asunicode.stypy_type_store = module_type_store
    asunicode.stypy_function_name = 'asunicode'
    asunicode.stypy_param_names_list = ['s']
    asunicode.stypy_varargs_param_name = None
    asunicode.stypy_kwargs_param_name = None
    asunicode.stypy_call_defaults = defaults
    asunicode.stypy_call_varargs = varargs
    asunicode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asunicode', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asunicode', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asunicode(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 64)
    # Getting the type of 'unicode' (line 64)
    unicode_25735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'unicode')
    # Getting the type of 's' (line 64)
    s_25736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 's')
    
    (may_be_25737, more_types_in_union_25738) = may_be_subtype(unicode_25735, s_25736)

    if may_be_25737:

        if more_types_in_union_25738:
            # Runtime conditional SSA (line 64)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 's' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 's', remove_not_subtype_from_union(s_25736, unicode))
        # Getting the type of 's' (line 65)
        s_25739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', s_25739)

        if more_types_in_union_25738:
            # SSA join for if statement (line 64)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to decode(...): (line 66)
    # Processing the call arguments (line 66)
    str_25745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 29), 'str', 'ascii')
    # Processing the call keyword arguments (line 66)
    kwargs_25746 = {}
    
    # Call to str(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 's' (line 66)
    s_25741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 's', False)
    # Processing the call keyword arguments (line 66)
    kwargs_25742 = {}
    # Getting the type of 'str' (line 66)
    str_25740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'str', False)
    # Calling str(args, kwargs) (line 66)
    str_call_result_25743 = invoke(stypy.reporting.localization.Localization(__file__, 66, 15), str_25740, *[s_25741], **kwargs_25742)
    
    # Obtaining the member 'decode' of a type (line 66)
    decode_25744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 15), str_call_result_25743, 'decode')
    # Calling decode(args, kwargs) (line 66)
    decode_call_result_25747 = invoke(stypy.reporting.localization.Localization(__file__, 66, 15), decode_25744, *[str_25745], **kwargs_25746)
    
    # Assigning a type to the variable 'stypy_return_type' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'stypy_return_type', decode_call_result_25747)
    
    # ################# End of 'asunicode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asunicode' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_25748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25748)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asunicode'
    return stypy_return_type_25748

# Assigning a type to the variable 'asunicode' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'asunicode', asunicode)

@norecursion
def open_latin1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_25749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 35), 'str', 'r')
    defaults = [str_25749]
    # Create a new context for function 'open_latin1'
    module_type_store = module_type_store.open_function_context('open_latin1', 68, 4, False)
    
    # Passed parameters checking function
    open_latin1.stypy_localization = localization
    open_latin1.stypy_type_of_self = None
    open_latin1.stypy_type_store = module_type_store
    open_latin1.stypy_function_name = 'open_latin1'
    open_latin1.stypy_param_names_list = ['filename', 'mode']
    open_latin1.stypy_varargs_param_name = None
    open_latin1.stypy_kwargs_param_name = None
    open_latin1.stypy_call_defaults = defaults
    open_latin1.stypy_call_varargs = varargs
    open_latin1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'open_latin1', ['filename', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'open_latin1', localization, ['filename', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'open_latin1(...)' code ##################

    
    # Call to open(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'filename' (line 69)
    filename_25751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'filename', False)
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'mode' (line 69)
    mode_25752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 35), 'mode', False)
    keyword_25753 = mode_25752
    kwargs_25754 = {'mode': keyword_25753}
    # Getting the type of 'open' (line 69)
    open_25750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'open', False)
    # Calling open(args, kwargs) (line 69)
    open_call_result_25755 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), open_25750, *[filename_25751], **kwargs_25754)
    
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', open_call_result_25755)
    
    # ################# End of 'open_latin1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'open_latin1' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_25756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25756)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'open_latin1'
    return stypy_return_type_25756

# Assigning a type to the variable 'open_latin1' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'open_latin1', open_latin1)

@norecursion
def sixu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sixu'
    module_type_store = module_type_store.open_function_context('sixu', 71, 4, False)
    
    # Passed parameters checking function
    sixu.stypy_localization = localization
    sixu.stypy_type_of_self = None
    sixu.stypy_type_store = module_type_store
    sixu.stypy_function_name = 'sixu'
    sixu.stypy_param_names_list = ['s']
    sixu.stypy_varargs_param_name = None
    sixu.stypy_kwargs_param_name = None
    sixu.stypy_call_defaults = defaults
    sixu.stypy_call_varargs = varargs
    sixu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sixu', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sixu', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sixu(...)' code ##################

    
    # Call to unicode(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 's' (line 72)
    s_25758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 's', False)
    str_25759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 26), 'str', 'unicode_escape')
    # Processing the call keyword arguments (line 72)
    kwargs_25760 = {}
    # Getting the type of 'unicode' (line 72)
    unicode_25757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'unicode', False)
    # Calling unicode(args, kwargs) (line 72)
    unicode_call_result_25761 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), unicode_25757, *[s_25758, str_25759], **kwargs_25760)
    
    # Assigning a type to the variable 'stypy_return_type' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', unicode_call_result_25761)
    
    # ################# End of 'sixu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sixu' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_25762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25762)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sixu'
    return stypy_return_type_25762

# Assigning a type to the variable 'sixu' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'sixu', sixu)
# SSA join for if statement (line 14)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def getexception(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getexception'
    module_type_store = module_type_store.open_function_context('getexception', 75, 0, False)
    
    # Passed parameters checking function
    getexception.stypy_localization = localization
    getexception.stypy_type_of_self = None
    getexception.stypy_type_store = module_type_store
    getexception.stypy_function_name = 'getexception'
    getexception.stypy_param_names_list = []
    getexception.stypy_varargs_param_name = None
    getexception.stypy_kwargs_param_name = None
    getexception.stypy_call_defaults = defaults
    getexception.stypy_call_varargs = varargs
    getexception.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getexception', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getexception', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getexception(...)' code ##################

    
    # Obtaining the type of the subscript
    int_25763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 26), 'int')
    
    # Call to exc_info(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_25766 = {}
    # Getting the type of 'sys' (line 76)
    sys_25764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'sys', False)
    # Obtaining the member 'exc_info' of a type (line 76)
    exc_info_25765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), sys_25764, 'exc_info')
    # Calling exc_info(args, kwargs) (line 76)
    exc_info_call_result_25767 = invoke(stypy.reporting.localization.Localization(__file__, 76, 11), exc_info_25765, *[], **kwargs_25766)
    
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___25768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), exc_info_call_result_25767, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_25769 = invoke(stypy.reporting.localization.Localization(__file__, 76, 11), getitem___25768, int_25763)
    
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type', subscript_call_result_25769)
    
    # ################# End of 'getexception(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getexception' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_25770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25770)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getexception'
    return stypy_return_type_25770

# Assigning a type to the variable 'getexception' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'getexception', getexception)

@norecursion
def asbytes_nested(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'asbytes_nested'
    module_type_store = module_type_store.open_function_context('asbytes_nested', 78, 0, False)
    
    # Passed parameters checking function
    asbytes_nested.stypy_localization = localization
    asbytes_nested.stypy_type_of_self = None
    asbytes_nested.stypy_type_store = module_type_store
    asbytes_nested.stypy_function_name = 'asbytes_nested'
    asbytes_nested.stypy_param_names_list = ['x']
    asbytes_nested.stypy_varargs_param_name = None
    asbytes_nested.stypy_kwargs_param_name = None
    asbytes_nested.stypy_call_defaults = defaults
    asbytes_nested.stypy_call_varargs = varargs
    asbytes_nested.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asbytes_nested', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asbytes_nested', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asbytes_nested(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'x' (line 79)
    x_25772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'x', False)
    str_25773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 18), 'str', '__iter__')
    # Processing the call keyword arguments (line 79)
    kwargs_25774 = {}
    # Getting the type of 'hasattr' (line 79)
    hasattr_25771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 79)
    hasattr_call_result_25775 = invoke(stypy.reporting.localization.Localization(__file__, 79, 7), hasattr_25771, *[x_25772, str_25773], **kwargs_25774)
    
    
    
    # Call to isinstance(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'x' (line 79)
    x_25777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 49), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 79)
    tuple_25778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 79)
    # Adding element type (line 79)
    # Getting the type of 'bytes' (line 79)
    bytes_25779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 53), 'bytes', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 53), tuple_25778, bytes_25779)
    # Adding element type (line 79)
    # Getting the type of 'unicode' (line 79)
    unicode_25780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 60), 'unicode', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 53), tuple_25778, unicode_25780)
    
    # Processing the call keyword arguments (line 79)
    kwargs_25781 = {}
    # Getting the type of 'isinstance' (line 79)
    isinstance_25776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 38), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 79)
    isinstance_call_result_25782 = invoke(stypy.reporting.localization.Localization(__file__, 79, 38), isinstance_25776, *[x_25777, tuple_25778], **kwargs_25781)
    
    # Applying the 'not' unary operator (line 79)
    result_not__25783 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 34), 'not', isinstance_call_result_25782)
    
    # Applying the binary operator 'and' (line 79)
    result_and_keyword_25784 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 7), 'and', hasattr_call_result_25775, result_not__25783)
    
    # Testing the type of an if condition (line 79)
    if_condition_25785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 4), result_and_keyword_25784)
    # Assigning a type to the variable 'if_condition_25785' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'if_condition_25785', if_condition_25785)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'x' (line 80)
    x_25790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 43), 'x')
    comprehension_25791 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 16), x_25790)
    # Assigning a type to the variable 'y' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'y', comprehension_25791)
    
    # Call to asbytes_nested(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'y' (line 80)
    y_25787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'y', False)
    # Processing the call keyword arguments (line 80)
    kwargs_25788 = {}
    # Getting the type of 'asbytes_nested' (line 80)
    asbytes_nested_25786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'asbytes_nested', False)
    # Calling asbytes_nested(args, kwargs) (line 80)
    asbytes_nested_call_result_25789 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), asbytes_nested_25786, *[y_25787], **kwargs_25788)
    
    list_25792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 16), list_25792, asbytes_nested_call_result_25789)
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', list_25792)
    # SSA branch for the else part of an if statement (line 79)
    module_type_store.open_ssa_branch('else')
    
    # Call to asbytes(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'x' (line 82)
    x_25794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'x', False)
    # Processing the call keyword arguments (line 82)
    kwargs_25795 = {}
    # Getting the type of 'asbytes' (line 82)
    asbytes_25793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 82)
    asbytes_call_result_25796 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), asbytes_25793, *[x_25794], **kwargs_25795)
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', asbytes_call_result_25796)
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'asbytes_nested(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asbytes_nested' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_25797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25797)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asbytes_nested'
    return stypy_return_type_25797

# Assigning a type to the variable 'asbytes_nested' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'asbytes_nested', asbytes_nested)

@norecursion
def asunicode_nested(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'asunicode_nested'
    module_type_store = module_type_store.open_function_context('asunicode_nested', 84, 0, False)
    
    # Passed parameters checking function
    asunicode_nested.stypy_localization = localization
    asunicode_nested.stypy_type_of_self = None
    asunicode_nested.stypy_type_store = module_type_store
    asunicode_nested.stypy_function_name = 'asunicode_nested'
    asunicode_nested.stypy_param_names_list = ['x']
    asunicode_nested.stypy_varargs_param_name = None
    asunicode_nested.stypy_kwargs_param_name = None
    asunicode_nested.stypy_call_defaults = defaults
    asunicode_nested.stypy_call_varargs = varargs
    asunicode_nested.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asunicode_nested', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asunicode_nested', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asunicode_nested(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'x' (line 85)
    x_25799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'x', False)
    str_25800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'str', '__iter__')
    # Processing the call keyword arguments (line 85)
    kwargs_25801 = {}
    # Getting the type of 'hasattr' (line 85)
    hasattr_25798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 85)
    hasattr_call_result_25802 = invoke(stypy.reporting.localization.Localization(__file__, 85, 7), hasattr_25798, *[x_25799, str_25800], **kwargs_25801)
    
    
    
    # Call to isinstance(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'x' (line 85)
    x_25804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 49), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_25805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    # Getting the type of 'bytes' (line 85)
    bytes_25806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 53), 'bytes', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 53), tuple_25805, bytes_25806)
    # Adding element type (line 85)
    # Getting the type of 'unicode' (line 85)
    unicode_25807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 60), 'unicode', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 53), tuple_25805, unicode_25807)
    
    # Processing the call keyword arguments (line 85)
    kwargs_25808 = {}
    # Getting the type of 'isinstance' (line 85)
    isinstance_25803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 38), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 85)
    isinstance_call_result_25809 = invoke(stypy.reporting.localization.Localization(__file__, 85, 38), isinstance_25803, *[x_25804, tuple_25805], **kwargs_25808)
    
    # Applying the 'not' unary operator (line 85)
    result_not__25810 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 34), 'not', isinstance_call_result_25809)
    
    # Applying the binary operator 'and' (line 85)
    result_and_keyword_25811 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 7), 'and', hasattr_call_result_25802, result_not__25810)
    
    # Testing the type of an if condition (line 85)
    if_condition_25812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 4), result_and_keyword_25811)
    # Assigning a type to the variable 'if_condition_25812' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'if_condition_25812', if_condition_25812)
    # SSA begins for if statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'x' (line 86)
    x_25817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 45), 'x')
    comprehension_25818 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 16), x_25817)
    # Assigning a type to the variable 'y' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'y', comprehension_25818)
    
    # Call to asunicode_nested(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'y' (line 86)
    y_25814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 33), 'y', False)
    # Processing the call keyword arguments (line 86)
    kwargs_25815 = {}
    # Getting the type of 'asunicode_nested' (line 86)
    asunicode_nested_25813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'asunicode_nested', False)
    # Calling asunicode_nested(args, kwargs) (line 86)
    asunicode_nested_call_result_25816 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), asunicode_nested_25813, *[y_25814], **kwargs_25815)
    
    list_25819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 16), list_25819, asunicode_nested_call_result_25816)
    # Assigning a type to the variable 'stypy_return_type' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', list_25819)
    # SSA branch for the else part of an if statement (line 85)
    module_type_store.open_ssa_branch('else')
    
    # Call to asunicode(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'x' (line 88)
    x_25821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'x', False)
    # Processing the call keyword arguments (line 88)
    kwargs_25822 = {}
    # Getting the type of 'asunicode' (line 88)
    asunicode_25820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'asunicode', False)
    # Calling asunicode(args, kwargs) (line 88)
    asunicode_call_result_25823 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), asunicode_25820, *[x_25821], **kwargs_25822)
    
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', asunicode_call_result_25823)
    # SSA join for if statement (line 85)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'asunicode_nested(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asunicode_nested' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_25824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25824)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asunicode_nested'
    return stypy_return_type_25824

# Assigning a type to the variable 'asunicode_nested' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'asunicode_nested', asunicode_nested)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
