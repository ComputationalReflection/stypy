
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Byteorder utilities for system - numpy byteorder encoding
2: 
3: Converts a variety of string codes for little endian, big endian,
4: native byte order and swapped byte order to explicit numpy endian
5: codes - one of '<' (little endian) or '>' (big endian)
6: 
7: '''
8: from __future__ import division, print_function, absolute_import
9: 
10: import sys
11: 
12: sys_is_le = sys.byteorder == 'little'
13: native_code = sys_is_le and '<' or '>'
14: swapped_code = sys_is_le and '>' or '<'
15: 
16: aliases = {'little': ('little', '<', 'l', 'le'),
17:            'big': ('big', '>', 'b', 'be'),
18:            'native': ('native', '='),
19:            'swapped': ('swapped', 'S')}
20: 
21: 
22: def to_numpy_code(code):
23:     '''
24:     Convert various order codings to numpy format.
25: 
26:     Parameters
27:     ----------
28:     code : str
29:         The code to convert. It is converted to lower case before parsing.
30:         Legal values are:
31:         'little', 'big', 'l', 'b', 'le', 'be', '<', '>', 'native', '=',
32:         'swapped', 's'.
33: 
34:     Returns
35:     -------
36:     out_code : {'<', '>'}
37:         Here '<' is the numpy dtype code for little endian,
38:         and '>' is the code for big endian.
39: 
40:     Examples
41:     --------
42:     >>> import sys
43:     >>> sys_is_le == (sys.byteorder == 'little')
44:     True
45:     >>> to_numpy_code('big')
46:     '>'
47:     >>> to_numpy_code('little')
48:     '<'
49:     >>> nc = to_numpy_code('native')
50:     >>> nc == '<' if sys_is_le else nc == '>'
51:     True
52:     >>> sc = to_numpy_code('swapped')
53:     >>> sc == '>' if sys_is_le else sc == '<'
54:     True
55: 
56:     '''
57:     code = code.lower()
58:     if code is None:
59:         return native_code
60:     if code in aliases['little']:
61:         return '<'
62:     elif code in aliases['big']:
63:         return '>'
64:     elif code in aliases['native']:
65:         return native_code
66:     elif code in aliases['swapped']:
67:         return swapped_code
68:     else:
69:         raise ValueError(
70:             'We cannot handle byte order %s' % code)
71: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_133360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', " Byteorder utilities for system - numpy byteorder encoding\n\nConverts a variety of string codes for little endian, big endian,\nnative byte order and swapped byte order to explicit numpy endian\ncodes - one of '<' (little endian) or '>' (big endian)\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import sys' statement (line 10)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sys', sys, module_type_store)


# Assigning a Compare to a Name (line 12):

# Getting the type of 'sys' (line 12)
sys_133361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'sys')
# Obtaining the member 'byteorder' of a type (line 12)
byteorder_133362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 12), sys_133361, 'byteorder')
str_133363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 29), 'str', 'little')
# Applying the binary operator '==' (line 12)
result_eq_133364 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), '==', byteorder_133362, str_133363)

# Assigning a type to the variable 'sys_is_le' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'sys_is_le', result_eq_133364)

# Assigning a BoolOp to a Name (line 13):

# Evaluating a boolean operation

# Evaluating a boolean operation
# Getting the type of 'sys_is_le' (line 13)
sys_is_le_133365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'sys_is_le')
str_133366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 28), 'str', '<')
# Applying the binary operator 'and' (line 13)
result_and_keyword_133367 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 14), 'and', sys_is_le_133365, str_133366)

str_133368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 35), 'str', '>')
# Applying the binary operator 'or' (line 13)
result_or_keyword_133369 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 14), 'or', result_and_keyword_133367, str_133368)

# Assigning a type to the variable 'native_code' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'native_code', result_or_keyword_133369)

# Assigning a BoolOp to a Name (line 14):

# Evaluating a boolean operation

# Evaluating a boolean operation
# Getting the type of 'sys_is_le' (line 14)
sys_is_le_133370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'sys_is_le')
str_133371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'str', '>')
# Applying the binary operator 'and' (line 14)
result_and_keyword_133372 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 15), 'and', sys_is_le_133370, str_133371)

str_133373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'str', '<')
# Applying the binary operator 'or' (line 14)
result_or_keyword_133374 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 15), 'or', result_and_keyword_133372, str_133373)

# Assigning a type to the variable 'swapped_code' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'swapped_code', result_or_keyword_133374)

# Assigning a Dict to a Name (line 16):

# Obtaining an instance of the builtin type 'dict' (line 16)
dict_133375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 16)
# Adding element type (key, value) (line 16)
str_133376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'little')

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_133377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)
str_133378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'str', 'little')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), tuple_133377, str_133378)
# Adding element type (line 16)
str_133379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 32), 'str', '<')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), tuple_133377, str_133379)
# Adding element type (line 16)
str_133380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 37), 'str', 'l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), tuple_133377, str_133380)
# Adding element type (line 16)
str_133381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 42), 'str', 'le')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), tuple_133377, str_133381)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), dict_133375, (str_133376, tuple_133377))
# Adding element type (key, value) (line 16)
str_133382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'big')

# Obtaining an instance of the builtin type 'tuple' (line 17)
tuple_133383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 17)
# Adding element type (line 17)
str_133384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'str', 'big')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 19), tuple_133383, str_133384)
# Adding element type (line 17)
str_133385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', '>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 19), tuple_133383, str_133385)
# Adding element type (line 17)
str_133386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 19), tuple_133383, str_133386)
# Adding element type (line 17)
str_133387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 36), 'str', 'be')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 19), tuple_133383, str_133387)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), dict_133375, (str_133382, tuple_133383))
# Adding element type (key, value) (line 16)
str_133388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'str', 'native')

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_133389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
str_133390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'str', 'native')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 22), tuple_133389, str_133390)
# Adding element type (line 18)
str_133391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'str', '=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 22), tuple_133389, str_133391)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), dict_133375, (str_133388, tuple_133389))
# Adding element type (key, value) (line 16)
str_133392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'str', 'swapped')

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_133393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
str_133394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'str', 'swapped')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), tuple_133393, str_133394)
# Adding element type (line 19)
str_133395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 34), 'str', 'S')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), tuple_133393, str_133395)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), dict_133375, (str_133392, tuple_133393))

# Assigning a type to the variable 'aliases' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'aliases', dict_133375)

@norecursion
def to_numpy_code(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'to_numpy_code'
    module_type_store = module_type_store.open_function_context('to_numpy_code', 22, 0, False)
    
    # Passed parameters checking function
    to_numpy_code.stypy_localization = localization
    to_numpy_code.stypy_type_of_self = None
    to_numpy_code.stypy_type_store = module_type_store
    to_numpy_code.stypy_function_name = 'to_numpy_code'
    to_numpy_code.stypy_param_names_list = ['code']
    to_numpy_code.stypy_varargs_param_name = None
    to_numpy_code.stypy_kwargs_param_name = None
    to_numpy_code.stypy_call_defaults = defaults
    to_numpy_code.stypy_call_varargs = varargs
    to_numpy_code.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'to_numpy_code', ['code'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'to_numpy_code', localization, ['code'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'to_numpy_code(...)' code ##################

    str_133396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', "\n    Convert various order codings to numpy format.\n\n    Parameters\n    ----------\n    code : str\n        The code to convert. It is converted to lower case before parsing.\n        Legal values are:\n        'little', 'big', 'l', 'b', 'le', 'be', '<', '>', 'native', '=',\n        'swapped', 's'.\n\n    Returns\n    -------\n    out_code : {'<', '>'}\n        Here '<' is the numpy dtype code for little endian,\n        and '>' is the code for big endian.\n\n    Examples\n    --------\n    >>> import sys\n    >>> sys_is_le == (sys.byteorder == 'little')\n    True\n    >>> to_numpy_code('big')\n    '>'\n    >>> to_numpy_code('little')\n    '<'\n    >>> nc = to_numpy_code('native')\n    >>> nc == '<' if sys_is_le else nc == '>'\n    True\n    >>> sc = to_numpy_code('swapped')\n    >>> sc == '>' if sys_is_le else sc == '<'\n    True\n\n    ")
    
    # Assigning a Call to a Name (line 57):
    
    # Call to lower(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_133399 = {}
    # Getting the type of 'code' (line 57)
    code_133397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'code', False)
    # Obtaining the member 'lower' of a type (line 57)
    lower_133398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), code_133397, 'lower')
    # Calling lower(args, kwargs) (line 57)
    lower_call_result_133400 = invoke(stypy.reporting.localization.Localization(__file__, 57, 11), lower_133398, *[], **kwargs_133399)
    
    # Assigning a type to the variable 'code' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'code', lower_call_result_133400)
    
    # Type idiom detected: calculating its left and rigth part (line 58)
    # Getting the type of 'code' (line 58)
    code_133401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'code')
    # Getting the type of 'None' (line 58)
    None_133402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'None')
    
    (may_be_133403, more_types_in_union_133404) = may_be_none(code_133401, None_133402)

    if may_be_133403:

        if more_types_in_union_133404:
            # Runtime conditional SSA (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'native_code' (line 59)
        native_code_133405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'native_code')
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', native_code_133405)

        if more_types_in_union_133404:
            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'code' (line 60)
    code_133406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 7), 'code')
    
    # Obtaining the type of the subscript
    str_133407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'str', 'little')
    # Getting the type of 'aliases' (line 60)
    aliases_133408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'aliases')
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___133409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), aliases_133408, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_133410 = invoke(stypy.reporting.localization.Localization(__file__, 60, 15), getitem___133409, str_133407)
    
    # Applying the binary operator 'in' (line 60)
    result_contains_133411 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 7), 'in', code_133406, subscript_call_result_133410)
    
    # Testing the type of an if condition (line 60)
    if_condition_133412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 4), result_contains_133411)
    # Assigning a type to the variable 'if_condition_133412' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'if_condition_133412', if_condition_133412)
    # SSA begins for if statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_133413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'str', '<')
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', str_133413)
    # SSA branch for the else part of an if statement (line 60)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'code' (line 62)
    code_133414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 9), 'code')
    
    # Obtaining the type of the subscript
    str_133415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 25), 'str', 'big')
    # Getting the type of 'aliases' (line 62)
    aliases_133416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'aliases')
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___133417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 17), aliases_133416, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_133418 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), getitem___133417, str_133415)
    
    # Applying the binary operator 'in' (line 62)
    result_contains_133419 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 9), 'in', code_133414, subscript_call_result_133418)
    
    # Testing the type of an if condition (line 62)
    if_condition_133420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 9), result_contains_133419)
    # Assigning a type to the variable 'if_condition_133420' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 9), 'if_condition_133420', if_condition_133420)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_133421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 15), 'str', '>')
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', str_133421)
    # SSA branch for the else part of an if statement (line 62)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'code' (line 64)
    code_133422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 9), 'code')
    
    # Obtaining the type of the subscript
    str_133423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'str', 'native')
    # Getting the type of 'aliases' (line 64)
    aliases_133424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'aliases')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___133425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 17), aliases_133424, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_133426 = invoke(stypy.reporting.localization.Localization(__file__, 64, 17), getitem___133425, str_133423)
    
    # Applying the binary operator 'in' (line 64)
    result_contains_133427 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 9), 'in', code_133422, subscript_call_result_133426)
    
    # Testing the type of an if condition (line 64)
    if_condition_133428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 9), result_contains_133427)
    # Assigning a type to the variable 'if_condition_133428' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 9), 'if_condition_133428', if_condition_133428)
    # SSA begins for if statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'native_code' (line 65)
    native_code_133429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'native_code')
    # Assigning a type to the variable 'stypy_return_type' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', native_code_133429)
    # SSA branch for the else part of an if statement (line 64)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'code' (line 66)
    code_133430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 9), 'code')
    
    # Obtaining the type of the subscript
    str_133431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'str', 'swapped')
    # Getting the type of 'aliases' (line 66)
    aliases_133432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'aliases')
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___133433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 17), aliases_133432, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_133434 = invoke(stypy.reporting.localization.Localization(__file__, 66, 17), getitem___133433, str_133431)
    
    # Applying the binary operator 'in' (line 66)
    result_contains_133435 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 9), 'in', code_133430, subscript_call_result_133434)
    
    # Testing the type of an if condition (line 66)
    if_condition_133436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 9), result_contains_133435)
    # Assigning a type to the variable 'if_condition_133436' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 9), 'if_condition_133436', if_condition_133436)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'swapped_code' (line 67)
    swapped_code_133437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'swapped_code')
    # Assigning a type to the variable 'stypy_return_type' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', swapped_code_133437)
    # SSA branch for the else part of an if statement (line 66)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 69)
    # Processing the call arguments (line 69)
    str_133439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'str', 'We cannot handle byte order %s')
    # Getting the type of 'code' (line 70)
    code_133440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 47), 'code', False)
    # Applying the binary operator '%' (line 70)
    result_mod_133441 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 12), '%', str_133439, code_133440)
    
    # Processing the call keyword arguments (line 69)
    kwargs_133442 = {}
    # Getting the type of 'ValueError' (line 69)
    ValueError_133438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 69)
    ValueError_call_result_133443 = invoke(stypy.reporting.localization.Localization(__file__, 69, 14), ValueError_133438, *[result_mod_133441], **kwargs_133442)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 69, 8), ValueError_call_result_133443, 'raise parameter', BaseException)
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 64)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'to_numpy_code(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'to_numpy_code' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_133444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_133444)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'to_numpy_code'
    return stypy_return_type_133444

# Assigning a type to the variable 'to_numpy_code' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'to_numpy_code', to_numpy_code)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
