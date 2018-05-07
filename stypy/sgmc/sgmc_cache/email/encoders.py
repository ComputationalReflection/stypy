
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Encodings and related functions.'''
6: 
7: __all__ = [
8:     'encode_7or8bit',
9:     'encode_base64',
10:     'encode_noop',
11:     'encode_quopri',
12:     ]
13: 
14: import base64
15: 
16: from quopri import encodestring as _encodestring
17: 
18: 
19: 
20: def _qencode(s):
21:     enc = _encodestring(s, quotetabs=True)
22:     # Must encode spaces, which quopri.encodestring() doesn't do
23:     return enc.replace(' ', '=20')
24: 
25: 
26: def _bencode(s):
27:     # We can't quite use base64.encodestring() since it tacks on a "courtesy
28:     # newline".  Blech!
29:     if not s:
30:         return s
31:     hasnewline = (s[-1] == '\n')
32:     value = base64.encodestring(s)
33:     if not hasnewline and value[-1] == '\n':
34:         return value[:-1]
35:     return value
36: 
37: 
38: 
39: def encode_base64(msg):
40:     '''Encode the message's payload in Base64.
41: 
42:     Also, add an appropriate Content-Transfer-Encoding header.
43:     '''
44:     orig = msg.get_payload()
45:     encdata = _bencode(orig)
46:     msg.set_payload(encdata)
47:     msg['Content-Transfer-Encoding'] = 'base64'
48: 
49: 
50: 
51: def encode_quopri(msg):
52:     '''Encode the message's payload in quoted-printable.
53: 
54:     Also, add an appropriate Content-Transfer-Encoding header.
55:     '''
56:     orig = msg.get_payload()
57:     encdata = _qencode(orig)
58:     msg.set_payload(encdata)
59:     msg['Content-Transfer-Encoding'] = 'quoted-printable'
60: 
61: 
62: 
63: def encode_7or8bit(msg):
64:     '''Set the Content-Transfer-Encoding header to 7bit or 8bit.'''
65:     orig = msg.get_payload()
66:     if orig is None:
67:         # There's no payload.  For backwards compatibility we use 7bit
68:         msg['Content-Transfer-Encoding'] = '7bit'
69:         return
70:     # We play a trick to make this go fast.  If encoding to ASCII succeeds, we
71:     # know the data must be 7bit, otherwise treat it as 8bit.
72:     try:
73:         orig.encode('ascii')
74:     except UnicodeError:
75:         msg['Content-Transfer-Encoding'] = '8bit'
76:     else:
77:         msg['Content-Transfer-Encoding'] = '7bit'
78: 
79: 
80: 
81: def encode_noop(msg):
82:     '''Do nothing.'''
83: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_12801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Encodings and related functions.')

# Assigning a List to a Name (line 7):
__all__ = ['encode_7or8bit', 'encode_base64', 'encode_noop', 'encode_quopri']
module_type_store.set_exportable_members(['encode_7or8bit', 'encode_base64', 'encode_noop', 'encode_quopri'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_12802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_12803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'str', 'encode_7or8bit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_12802, str_12803)
# Adding element type (line 7)
str_12804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'str', 'encode_base64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_12802, str_12804)
# Adding element type (line 7)
str_12805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'str', 'encode_noop')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_12802, str_12805)
# Adding element type (line 7)
str_12806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'str', 'encode_quopri')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_12802, str_12806)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_12802)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import base64' statement (line 14)
import base64

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'base64', base64, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from quopri import _encodestring' statement (line 16)
try:
    from quopri import encodestring as _encodestring

except:
    _encodestring = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'quopri', None, module_type_store, ['encodestring'], [_encodestring])
# Adding an alias
module_type_store.add_alias('_encodestring', 'encodestring')


@norecursion
def _qencode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_qencode'
    module_type_store = module_type_store.open_function_context('_qencode', 20, 0, False)
    
    # Passed parameters checking function
    _qencode.stypy_localization = localization
    _qencode.stypy_type_of_self = None
    _qencode.stypy_type_store = module_type_store
    _qencode.stypy_function_name = '_qencode'
    _qencode.stypy_param_names_list = ['s']
    _qencode.stypy_varargs_param_name = None
    _qencode.stypy_kwargs_param_name = None
    _qencode.stypy_call_defaults = defaults
    _qencode.stypy_call_varargs = varargs
    _qencode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_qencode', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_qencode', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_qencode(...)' code ##################

    
    # Assigning a Call to a Name (line 21):
    
    # Call to _encodestring(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 's' (line 21)
    s_12808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 's', False)
    # Processing the call keyword arguments (line 21)
    # Getting the type of 'True' (line 21)
    True_12809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 37), 'True', False)
    keyword_12810 = True_12809
    kwargs_12811 = {'quotetabs': keyword_12810}
    # Getting the type of '_encodestring' (line 21)
    _encodestring_12807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), '_encodestring', False)
    # Calling _encodestring(args, kwargs) (line 21)
    _encodestring_call_result_12812 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), _encodestring_12807, *[s_12808], **kwargs_12811)
    
    # Assigning a type to the variable 'enc' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'enc', _encodestring_call_result_12812)
    
    # Call to replace(...): (line 23)
    # Processing the call arguments (line 23)
    str_12815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'str', ' ')
    str_12816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 28), 'str', '=20')
    # Processing the call keyword arguments (line 23)
    kwargs_12817 = {}
    # Getting the type of 'enc' (line 23)
    enc_12813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'enc', False)
    # Obtaining the member 'replace' of a type (line 23)
    replace_12814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), enc_12813, 'replace')
    # Calling replace(args, kwargs) (line 23)
    replace_call_result_12818 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), replace_12814, *[str_12815, str_12816], **kwargs_12817)
    
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', replace_call_result_12818)
    
    # ################# End of '_qencode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_qencode' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_12819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12819)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_qencode'
    return stypy_return_type_12819

# Assigning a type to the variable '_qencode' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '_qencode', _qencode)

@norecursion
def _bencode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_bencode'
    module_type_store = module_type_store.open_function_context('_bencode', 26, 0, False)
    
    # Passed parameters checking function
    _bencode.stypy_localization = localization
    _bencode.stypy_type_of_self = None
    _bencode.stypy_type_store = module_type_store
    _bencode.stypy_function_name = '_bencode'
    _bencode.stypy_param_names_list = ['s']
    _bencode.stypy_varargs_param_name = None
    _bencode.stypy_kwargs_param_name = None
    _bencode.stypy_call_defaults = defaults
    _bencode.stypy_call_varargs = varargs
    _bencode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_bencode', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_bencode', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_bencode(...)' code ##################

    
    # Getting the type of 's' (line 29)
    s_12820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 's')
    # Applying the 'not' unary operator (line 29)
    result_not__12821 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 7), 'not', s_12820)
    
    # Testing if the type of an if condition is none (line 29)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 4), result_not__12821):
        pass
    else:
        
        # Testing the type of an if condition (line 29)
        if_condition_12822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 4), result_not__12821)
        # Assigning a type to the variable 'if_condition_12822' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'if_condition_12822', if_condition_12822)
        # SSA begins for if statement (line 29)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 's' (line 30)
        s_12823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', s_12823)
        # SSA join for if statement (line 29)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Compare to a Name (line 31):
    
    
    # Obtaining the type of the subscript
    int_12824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'int')
    # Getting the type of 's' (line 31)
    s_12825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 's')
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___12826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 18), s_12825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_12827 = invoke(stypy.reporting.localization.Localization(__file__, 31, 18), getitem___12826, int_12824)
    
    str_12828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 27), 'str', '\n')
    # Applying the binary operator '==' (line 31)
    result_eq_12829 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 18), '==', subscript_call_result_12827, str_12828)
    
    # Assigning a type to the variable 'hasnewline' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'hasnewline', result_eq_12829)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to encodestring(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 's' (line 32)
    s_12832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), 's', False)
    # Processing the call keyword arguments (line 32)
    kwargs_12833 = {}
    # Getting the type of 'base64' (line 32)
    base64_12830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'base64', False)
    # Obtaining the member 'encodestring' of a type (line 32)
    encodestring_12831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), base64_12830, 'encodestring')
    # Calling encodestring(args, kwargs) (line 32)
    encodestring_call_result_12834 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), encodestring_12831, *[s_12832], **kwargs_12833)
    
    # Assigning a type to the variable 'value' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'value', encodestring_call_result_12834)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'hasnewline' (line 33)
    hasnewline_12835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'hasnewline')
    # Applying the 'not' unary operator (line 33)
    result_not__12836 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 7), 'not', hasnewline_12835)
    
    
    
    # Obtaining the type of the subscript
    int_12837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'int')
    # Getting the type of 'value' (line 33)
    value_12838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 26), 'value')
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___12839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 26), value_12838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_12840 = invoke(stypy.reporting.localization.Localization(__file__, 33, 26), getitem___12839, int_12837)
    
    str_12841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 39), 'str', '\n')
    # Applying the binary operator '==' (line 33)
    result_eq_12842 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 26), '==', subscript_call_result_12840, str_12841)
    
    # Applying the binary operator 'and' (line 33)
    result_and_keyword_12843 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 7), 'and', result_not__12836, result_eq_12842)
    
    # Testing if the type of an if condition is none (line 33)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 33, 4), result_and_keyword_12843):
        pass
    else:
        
        # Testing the type of an if condition (line 33)
        if_condition_12844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), result_and_keyword_12843)
        # Assigning a type to the variable 'if_condition_12844' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_12844', if_condition_12844)
        # SSA begins for if statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        int_12845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'int')
        slice_12846 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 34, 15), None, int_12845, None)
        # Getting the type of 'value' (line 34)
        value_12847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'value')
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___12848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), value_12847, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_12849 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), getitem___12848, slice_12846)
        
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', subscript_call_result_12849)
        # SSA join for if statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'value' (line 35)
    value_12850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'value')
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', value_12850)
    
    # ################# End of '_bencode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_bencode' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_12851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12851)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_bencode'
    return stypy_return_type_12851

# Assigning a type to the variable '_bencode' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '_bencode', _bencode)

@norecursion
def encode_base64(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'encode_base64'
    module_type_store = module_type_store.open_function_context('encode_base64', 39, 0, False)
    
    # Passed parameters checking function
    encode_base64.stypy_localization = localization
    encode_base64.stypy_type_of_self = None
    encode_base64.stypy_type_store = module_type_store
    encode_base64.stypy_function_name = 'encode_base64'
    encode_base64.stypy_param_names_list = ['msg']
    encode_base64.stypy_varargs_param_name = None
    encode_base64.stypy_kwargs_param_name = None
    encode_base64.stypy_call_defaults = defaults
    encode_base64.stypy_call_varargs = varargs
    encode_base64.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode_base64', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode_base64', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode_base64(...)' code ##################

    str_12852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', "Encode the message's payload in Base64.\n\n    Also, add an appropriate Content-Transfer-Encoding header.\n    ")
    
    # Assigning a Call to a Name (line 44):
    
    # Call to get_payload(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_12855 = {}
    # Getting the type of 'msg' (line 44)
    msg_12853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'msg', False)
    # Obtaining the member 'get_payload' of a type (line 44)
    get_payload_12854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), msg_12853, 'get_payload')
    # Calling get_payload(args, kwargs) (line 44)
    get_payload_call_result_12856 = invoke(stypy.reporting.localization.Localization(__file__, 44, 11), get_payload_12854, *[], **kwargs_12855)
    
    # Assigning a type to the variable 'orig' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'orig', get_payload_call_result_12856)
    
    # Assigning a Call to a Name (line 45):
    
    # Call to _bencode(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'orig' (line 45)
    orig_12858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'orig', False)
    # Processing the call keyword arguments (line 45)
    kwargs_12859 = {}
    # Getting the type of '_bencode' (line 45)
    _bencode_12857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), '_bencode', False)
    # Calling _bencode(args, kwargs) (line 45)
    _bencode_call_result_12860 = invoke(stypy.reporting.localization.Localization(__file__, 45, 14), _bencode_12857, *[orig_12858], **kwargs_12859)
    
    # Assigning a type to the variable 'encdata' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'encdata', _bencode_call_result_12860)
    
    # Call to set_payload(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'encdata' (line 46)
    encdata_12863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'encdata', False)
    # Processing the call keyword arguments (line 46)
    kwargs_12864 = {}
    # Getting the type of 'msg' (line 46)
    msg_12861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'msg', False)
    # Obtaining the member 'set_payload' of a type (line 46)
    set_payload_12862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), msg_12861, 'set_payload')
    # Calling set_payload(args, kwargs) (line 46)
    set_payload_call_result_12865 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), set_payload_12862, *[encdata_12863], **kwargs_12864)
    
    
    # Assigning a Str to a Subscript (line 47):
    str_12866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 39), 'str', 'base64')
    # Getting the type of 'msg' (line 47)
    msg_12867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'msg')
    str_12868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'str', 'Content-Transfer-Encoding')
    # Storing an element on a container (line 47)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 4), msg_12867, (str_12868, str_12866))
    
    # ################# End of 'encode_base64(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode_base64' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_12869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12869)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode_base64'
    return stypy_return_type_12869

# Assigning a type to the variable 'encode_base64' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'encode_base64', encode_base64)

@norecursion
def encode_quopri(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'encode_quopri'
    module_type_store = module_type_store.open_function_context('encode_quopri', 51, 0, False)
    
    # Passed parameters checking function
    encode_quopri.stypy_localization = localization
    encode_quopri.stypy_type_of_self = None
    encode_quopri.stypy_type_store = module_type_store
    encode_quopri.stypy_function_name = 'encode_quopri'
    encode_quopri.stypy_param_names_list = ['msg']
    encode_quopri.stypy_varargs_param_name = None
    encode_quopri.stypy_kwargs_param_name = None
    encode_quopri.stypy_call_defaults = defaults
    encode_quopri.stypy_call_varargs = varargs
    encode_quopri.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode_quopri', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode_quopri', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode_quopri(...)' code ##################

    str_12870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', "Encode the message's payload in quoted-printable.\n\n    Also, add an appropriate Content-Transfer-Encoding header.\n    ")
    
    # Assigning a Call to a Name (line 56):
    
    # Call to get_payload(...): (line 56)
    # Processing the call keyword arguments (line 56)
    kwargs_12873 = {}
    # Getting the type of 'msg' (line 56)
    msg_12871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'msg', False)
    # Obtaining the member 'get_payload' of a type (line 56)
    get_payload_12872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), msg_12871, 'get_payload')
    # Calling get_payload(args, kwargs) (line 56)
    get_payload_call_result_12874 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), get_payload_12872, *[], **kwargs_12873)
    
    # Assigning a type to the variable 'orig' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'orig', get_payload_call_result_12874)
    
    # Assigning a Call to a Name (line 57):
    
    # Call to _qencode(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'orig' (line 57)
    orig_12876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'orig', False)
    # Processing the call keyword arguments (line 57)
    kwargs_12877 = {}
    # Getting the type of '_qencode' (line 57)
    _qencode_12875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), '_qencode', False)
    # Calling _qencode(args, kwargs) (line 57)
    _qencode_call_result_12878 = invoke(stypy.reporting.localization.Localization(__file__, 57, 14), _qencode_12875, *[orig_12876], **kwargs_12877)
    
    # Assigning a type to the variable 'encdata' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'encdata', _qencode_call_result_12878)
    
    # Call to set_payload(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'encdata' (line 58)
    encdata_12881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'encdata', False)
    # Processing the call keyword arguments (line 58)
    kwargs_12882 = {}
    # Getting the type of 'msg' (line 58)
    msg_12879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'msg', False)
    # Obtaining the member 'set_payload' of a type (line 58)
    set_payload_12880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), msg_12879, 'set_payload')
    # Calling set_payload(args, kwargs) (line 58)
    set_payload_call_result_12883 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), set_payload_12880, *[encdata_12881], **kwargs_12882)
    
    
    # Assigning a Str to a Subscript (line 59):
    str_12884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 39), 'str', 'quoted-printable')
    # Getting the type of 'msg' (line 59)
    msg_12885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'msg')
    str_12886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'str', 'Content-Transfer-Encoding')
    # Storing an element on a container (line 59)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 4), msg_12885, (str_12886, str_12884))
    
    # ################# End of 'encode_quopri(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode_quopri' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_12887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12887)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode_quopri'
    return stypy_return_type_12887

# Assigning a type to the variable 'encode_quopri' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'encode_quopri', encode_quopri)

@norecursion
def encode_7or8bit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'encode_7or8bit'
    module_type_store = module_type_store.open_function_context('encode_7or8bit', 63, 0, False)
    
    # Passed parameters checking function
    encode_7or8bit.stypy_localization = localization
    encode_7or8bit.stypy_type_of_self = None
    encode_7or8bit.stypy_type_store = module_type_store
    encode_7or8bit.stypy_function_name = 'encode_7or8bit'
    encode_7or8bit.stypy_param_names_list = ['msg']
    encode_7or8bit.stypy_varargs_param_name = None
    encode_7or8bit.stypy_kwargs_param_name = None
    encode_7or8bit.stypy_call_defaults = defaults
    encode_7or8bit.stypy_call_varargs = varargs
    encode_7or8bit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode_7or8bit', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode_7or8bit', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode_7or8bit(...)' code ##################

    str_12888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'str', 'Set the Content-Transfer-Encoding header to 7bit or 8bit.')
    
    # Assigning a Call to a Name (line 65):
    
    # Call to get_payload(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_12891 = {}
    # Getting the type of 'msg' (line 65)
    msg_12889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'msg', False)
    # Obtaining the member 'get_payload' of a type (line 65)
    get_payload_12890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), msg_12889, 'get_payload')
    # Calling get_payload(args, kwargs) (line 65)
    get_payload_call_result_12892 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), get_payload_12890, *[], **kwargs_12891)
    
    # Assigning a type to the variable 'orig' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'orig', get_payload_call_result_12892)
    
    # Type idiom detected: calculating its left and rigth part (line 66)
    # Getting the type of 'orig' (line 66)
    orig_12893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'orig')
    # Getting the type of 'None' (line 66)
    None_12894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'None')
    
    (may_be_12895, more_types_in_union_12896) = may_be_none(orig_12893, None_12894)

    if may_be_12895:

        if more_types_in_union_12896:
            # Runtime conditional SSA (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Subscript (line 68):
        str_12897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 43), 'str', '7bit')
        # Getting the type of 'msg' (line 68)
        msg_12898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'msg')
        str_12899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 12), 'str', 'Content-Transfer-Encoding')
        # Storing an element on a container (line 68)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 8), msg_12898, (str_12899, str_12897))
        # Assigning a type to the variable 'stypy_return_type' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', types.NoneType)

        if more_types_in_union_12896:
            # SSA join for if statement (line 66)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'orig' (line 66)
    orig_12900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'orig')
    # Assigning a type to the variable 'orig' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'orig', remove_type_from_union(orig_12900, types.NoneType))
    
    
    # SSA begins for try-except statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to encode(...): (line 73)
    # Processing the call arguments (line 73)
    str_12903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'str', 'ascii')
    # Processing the call keyword arguments (line 73)
    kwargs_12904 = {}
    # Getting the type of 'orig' (line 73)
    orig_12901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'orig', False)
    # Obtaining the member 'encode' of a type (line 73)
    encode_12902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), orig_12901, 'encode')
    # Calling encode(args, kwargs) (line 73)
    encode_call_result_12905 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), encode_12902, *[str_12903], **kwargs_12904)
    
    # SSA branch for the except part of a try statement (line 72)
    # SSA branch for the except 'UnicodeError' branch of a try statement (line 72)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Str to a Subscript (line 75):
    str_12906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 43), 'str', '8bit')
    # Getting the type of 'msg' (line 75)
    msg_12907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'msg')
    str_12908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 12), 'str', 'Content-Transfer-Encoding')
    # Storing an element on a container (line 75)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 8), msg_12907, (str_12908, str_12906))
    # SSA branch for the else branch of a try statement (line 72)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a Str to a Subscript (line 77):
    str_12909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 43), 'str', '7bit')
    # Getting the type of 'msg' (line 77)
    msg_12910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'msg')
    str_12911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 12), 'str', 'Content-Transfer-Encoding')
    # Storing an element on a container (line 77)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 8), msg_12910, (str_12911, str_12909))
    # SSA join for try-except statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'encode_7or8bit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode_7or8bit' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_12912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12912)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode_7or8bit'
    return stypy_return_type_12912

# Assigning a type to the variable 'encode_7or8bit' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'encode_7or8bit', encode_7or8bit)

@norecursion
def encode_noop(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'encode_noop'
    module_type_store = module_type_store.open_function_context('encode_noop', 81, 0, False)
    
    # Passed parameters checking function
    encode_noop.stypy_localization = localization
    encode_noop.stypy_type_of_self = None
    encode_noop.stypy_type_store = module_type_store
    encode_noop.stypy_function_name = 'encode_noop'
    encode_noop.stypy_param_names_list = ['msg']
    encode_noop.stypy_varargs_param_name = None
    encode_noop.stypy_kwargs_param_name = None
    encode_noop.stypy_call_defaults = defaults
    encode_noop.stypy_call_varargs = varargs
    encode_noop.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode_noop', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode_noop', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode_noop(...)' code ##################

    str_12913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'str', 'Do nothing.')
    
    # ################# End of 'encode_noop(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode_noop' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_12914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12914)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode_noop'
    return stypy_return_type_12914

# Assigning a type to the variable 'encode_noop' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'encode_noop', encode_noop)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
