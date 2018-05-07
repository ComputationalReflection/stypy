
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2002-2006 Python Software Foundation
2: # Author: Ben Gertzfield
3: # Contact: email-sig@python.org
4: 
5: '''Base64 content transfer encoding per RFCs 2045-2047.
6: 
7: This module handles the content transfer encoding method defined in RFC 2045
8: to encode arbitrary 8-bit data using the three 8-bit bytes in four 7-bit
9: characters encoding known as Base64.
10: 
11: It is used in the MIME standards for email to attach images, audio, and text
12: using some 8-bit character sets to messages.
13: 
14: This module provides an interface to encode and decode both headers and bodies
15: with Base64 encoding.
16: 
17: RFC 2045 defines a method for including character set information in an
18: `encoded-word' in a header.  This method is commonly used for 8-bit real names
19: in To:, From:, Cc:, etc. fields, as well as Subject: lines.
20: 
21: This module does not do the line wrapping or end-of-line character conversion
22: necessary for proper internationalized headers; it only does dumb encoding and
23: decoding.  To deal with the various line wrapping issues, use the email.header
24: module.
25: '''
26: 
27: __all__ = [
28:     'base64_len',
29:     'body_decode',
30:     'body_encode',
31:     'decode',
32:     'decodestring',
33:     'encode',
34:     'encodestring',
35:     'header_encode',
36:     ]
37: 
38: 
39: from binascii import b2a_base64, a2b_base64
40: from email.utils import fix_eols
41: 
42: CRLF = '\r\n'
43: NL = '\n'
44: EMPTYSTRING = ''
45: 
46: # See also Charset.py
47: MISC_LEN = 7
48: 
49: 
50: 
51: # Helpers
52: def base64_len(s):
53:     '''Return the length of s when it is encoded with base64.'''
54:     groups_of_3, leftover = divmod(len(s), 3)
55:     # 4 bytes out for each 3 bytes (or nonzero fraction thereof) in.
56:     # Thanks, Tim!
57:     n = groups_of_3 * 4
58:     if leftover:
59:         n += 4
60:     return n
61: 
62: 
63: 
64: def header_encode(header, charset='iso-8859-1', keep_eols=False,
65:                   maxlinelen=76, eol=NL):
66:     '''Encode a single header line with Base64 encoding in a given charset.
67: 
68:     Defined in RFC 2045, this Base64 encoding is identical to normal Base64
69:     encoding, except that each line must be intelligently wrapped (respecting
70:     the Base64 encoding), and subsequent lines must start with a space.
71: 
72:     charset names the character set to use to encode the header.  It defaults
73:     to iso-8859-1.
74: 
75:     End-of-line characters (\\r, \\n, \\r\\n) will be automatically converted
76:     to the canonical email line separator \\r\\n unless the keep_eols
77:     parameter is True (the default is False).
78: 
79:     Each line of the header will be terminated in the value of eol, which
80:     defaults to "\\n".  Set this to "\\r\\n" if you are using the result of
81:     this function directly in email.
82: 
83:     The resulting string will be in the form:
84: 
85:     "=?charset?b?WW/5ciBtYXp66XLrIHf8eiBhIGhhbXBzdGHuciBBIFlv+XIgbWF6euly?=\\n
86:       =?charset?b?6yB3/HogYSBoYW1wc3Rh7nIgQkMgWW/5ciBtYXp66XLrIHf8eiBhIGhh?="
87: 
88:     with each line wrapped at, at most, maxlinelen characters (defaults to 76
89:     characters).
90:     '''
91:     # Return empty headers unchanged
92:     if not header:
93:         return header
94: 
95:     if not keep_eols:
96:         header = fix_eols(header)
97: 
98:     # Base64 encode each line, in encoded chunks no greater than maxlinelen in
99:     # length, after the RFC chrome is added in.
100:     base64ed = []
101:     max_encoded = maxlinelen - len(charset) - MISC_LEN
102:     max_unencoded = max_encoded * 3 // 4
103: 
104:     for i in range(0, len(header), max_unencoded):
105:         base64ed.append(b2a_base64(header[i:i+max_unencoded]))
106: 
107:     # Now add the RFC chrome to each encoded chunk
108:     lines = []
109:     for line in base64ed:
110:         # Ignore the last character of each line if it is a newline
111:         if line.endswith(NL):
112:             line = line[:-1]
113:         # Add the chrome
114:         lines.append('=?%s?b?%s?=' % (charset, line))
115:     # Glue the lines together and return it.  BAW: should we be able to
116:     # specify the leading whitespace in the joiner?
117:     joiner = eol + ' '
118:     return joiner.join(lines)
119: 
120: 
121: 
122: def encode(s, binary=True, maxlinelen=76, eol=NL):
123:     '''Encode a string with base64.
124: 
125:     Each line will be wrapped at, at most, maxlinelen characters (defaults to
126:     76 characters).
127: 
128:     If binary is False, end-of-line characters will be converted to the
129:     canonical email end-of-line sequence \\r\\n.  Otherwise they will be left
130:     verbatim (this is the default).
131: 
132:     Each line of encoded text will end with eol, which defaults to "\\n".  Set
133:     this to "\\r\\n" if you will be using the result of this function directly
134:     in an email.
135:     '''
136:     if not s:
137:         return s
138: 
139:     if not binary:
140:         s = fix_eols(s)
141: 
142:     encvec = []
143:     max_unencoded = maxlinelen * 3 // 4
144:     for i in range(0, len(s), max_unencoded):
145:         # BAW: should encode() inherit b2a_base64()'s dubious behavior in
146:         # adding a newline to the encoded string?
147:         enc = b2a_base64(s[i:i + max_unencoded])
148:         if enc.endswith(NL) and eol != NL:
149:             enc = enc[:-1] + eol
150:         encvec.append(enc)
151:     return EMPTYSTRING.join(encvec)
152: 
153: 
154: # For convenience and backwards compatibility w/ standard base64 module
155: body_encode = encode
156: encodestring = encode
157: 
158: 
159: 
160: def decode(s, convert_eols=None):
161:     '''Decode a raw base64 string.
162: 
163:     If convert_eols is set to a string value, all canonical email linefeeds,
164:     e.g. "\\r\\n", in the decoded text will be converted to the value of
165:     convert_eols.  os.linesep is a good choice for convert_eols if you are
166:     decoding a text attachment.
167: 
168:     This function does not parse a full MIME header value encoded with
169:     base64 (like =?iso-8859-1?b?bmloISBuaWgh?=) -- please use the high
170:     level email.header class for that functionality.
171:     '''
172:     if not s:
173:         return s
174: 
175:     dec = a2b_base64(s)
176:     if convert_eols:
177:         return dec.replace(CRLF, convert_eols)
178:     return dec
179: 
180: 
181: # For convenience and backwards compatibility w/ standard base64 module
182: body_decode = decode
183: decodestring = decode
184: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_11952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', "Base64 content transfer encoding per RFCs 2045-2047.\n\nThis module handles the content transfer encoding method defined in RFC 2045\nto encode arbitrary 8-bit data using the three 8-bit bytes in four 7-bit\ncharacters encoding known as Base64.\n\nIt is used in the MIME standards for email to attach images, audio, and text\nusing some 8-bit character sets to messages.\n\nThis module provides an interface to encode and decode both headers and bodies\nwith Base64 encoding.\n\nRFC 2045 defines a method for including character set information in an\n`encoded-word' in a header.  This method is commonly used for 8-bit real names\nin To:, From:, Cc:, etc. fields, as well as Subject: lines.\n\nThis module does not do the line wrapping or end-of-line character conversion\nnecessary for proper internationalized headers; it only does dumb encoding and\ndecoding.  To deal with the various line wrapping issues, use the email.header\nmodule.\n")

# Assigning a List to a Name (line 27):

# Assigning a List to a Name (line 27):
__all__ = ['base64_len', 'body_decode', 'body_encode', 'decode', 'decodestring', 'encode', 'encodestring', 'header_encode']
module_type_store.set_exportable_members(['base64_len', 'body_decode', 'body_encode', 'decode', 'decodestring', 'encode', 'encodestring', 'header_encode'])

# Obtaining an instance of the builtin type 'list' (line 27)
list_11953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
str_11954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'base64_len')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_11953, str_11954)
# Adding element type (line 27)
str_11955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', 'body_decode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_11953, str_11955)
# Adding element type (line 27)
str_11956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', 'body_encode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_11953, str_11956)
# Adding element type (line 27)
str_11957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'str', 'decode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_11953, str_11957)
# Adding element type (line 27)
str_11958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', 'decodestring')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_11953, str_11958)
# Adding element type (line 27)
str_11959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'str', 'encode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_11953, str_11959)
# Adding element type (line 27)
str_11960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'str', 'encodestring')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_11953, str_11960)
# Adding element type (line 27)
str_11961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', 'header_encode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_11953, str_11961)

# Assigning a type to the variable '__all__' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), '__all__', list_11953)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'from binascii import b2a_base64, a2b_base64' statement (line 39)
try:
    from binascii import b2a_base64, a2b_base64

except:
    b2a_base64 = UndefinedType
    a2b_base64 = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'binascii', None, module_type_store, ['b2a_base64', 'a2b_base64'], [b2a_base64, a2b_base64])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'from email.utils import fix_eols' statement (line 40)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_11962 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'email.utils')

if (type(import_11962) is not StypyTypeError):

    if (import_11962 != 'pyd_module'):
        __import__(import_11962)
        sys_modules_11963 = sys.modules[import_11962]
        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'email.utils', sys_modules_11963.module_type_store, module_type_store, ['fix_eols'])
        nest_module(stypy.reporting.localization.Localization(__file__, 40, 0), __file__, sys_modules_11963, sys_modules_11963.module_type_store, module_type_store)
    else:
        from email.utils import fix_eols

        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'email.utils', None, module_type_store, ['fix_eols'], [fix_eols])

else:
    # Assigning a type to the variable 'email.utils' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'email.utils', import_11962)

remove_current_file_folder_from_path('C:/Python27/lib/email/')


# Assigning a Str to a Name (line 42):

# Assigning a Str to a Name (line 42):
str_11964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 7), 'str', '\r\n')
# Assigning a type to the variable 'CRLF' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'CRLF', str_11964)

# Assigning a Str to a Name (line 43):

# Assigning a Str to a Name (line 43):
str_11965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 5), 'str', '\n')
# Assigning a type to the variable 'NL' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'NL', str_11965)

# Assigning a Str to a Name (line 44):

# Assigning a Str to a Name (line 44):
str_11966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 14), 'str', '')
# Assigning a type to the variable 'EMPTYSTRING' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'EMPTYSTRING', str_11966)

# Assigning a Num to a Name (line 47):

# Assigning a Num to a Name (line 47):
int_11967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'int')
# Assigning a type to the variable 'MISC_LEN' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'MISC_LEN', int_11967)

@norecursion
def base64_len(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'base64_len'
    module_type_store = module_type_store.open_function_context('base64_len', 52, 0, False)
    
    # Passed parameters checking function
    base64_len.stypy_localization = localization
    base64_len.stypy_type_of_self = None
    base64_len.stypy_type_store = module_type_store
    base64_len.stypy_function_name = 'base64_len'
    base64_len.stypy_param_names_list = ['s']
    base64_len.stypy_varargs_param_name = None
    base64_len.stypy_kwargs_param_name = None
    base64_len.stypy_call_defaults = defaults
    base64_len.stypy_call_varargs = varargs
    base64_len.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'base64_len', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'base64_len', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'base64_len(...)' code ##################

    str_11968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 4), 'str', 'Return the length of s when it is encoded with base64.')
    
    # Assigning a Call to a Tuple (line 54):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Call to len(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 's' (line 54)
    s_11971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 's', False)
    # Processing the call keyword arguments (line 54)
    kwargs_11972 = {}
    # Getting the type of 'len' (line 54)
    len_11970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 35), 'len', False)
    # Calling len(args, kwargs) (line 54)
    len_call_result_11973 = invoke(stypy.reporting.localization.Localization(__file__, 54, 35), len_11970, *[s_11971], **kwargs_11972)
    
    int_11974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 43), 'int')
    # Processing the call keyword arguments (line 54)
    kwargs_11975 = {}
    # Getting the type of 'divmod' (line 54)
    divmod_11969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'divmod', False)
    # Calling divmod(args, kwargs) (line 54)
    divmod_call_result_11976 = invoke(stypy.reporting.localization.Localization(__file__, 54, 28), divmod_11969, *[len_call_result_11973, int_11974], **kwargs_11975)
    
    # Assigning a type to the variable 'call_assignment_11949' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'call_assignment_11949', divmod_call_result_11976)
    
    # Assigning a Call to a Name (line 54):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_11949' (line 54)
    call_assignment_11949_11977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'call_assignment_11949', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_11978 = stypy_get_value_from_tuple(call_assignment_11949_11977, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_11950' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'call_assignment_11950', stypy_get_value_from_tuple_call_result_11978)
    
    # Assigning a Name to a Name (line 54):
    # Getting the type of 'call_assignment_11950' (line 54)
    call_assignment_11950_11979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'call_assignment_11950')
    # Assigning a type to the variable 'groups_of_3' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'groups_of_3', call_assignment_11950_11979)
    
    # Assigning a Call to a Name (line 54):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_11949' (line 54)
    call_assignment_11949_11980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'call_assignment_11949', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_11981 = stypy_get_value_from_tuple(call_assignment_11949_11980, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_11951' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'call_assignment_11951', stypy_get_value_from_tuple_call_result_11981)
    
    # Assigning a Name to a Name (line 54):
    # Getting the type of 'call_assignment_11951' (line 54)
    call_assignment_11951_11982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'call_assignment_11951')
    # Assigning a type to the variable 'leftover' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'leftover', call_assignment_11951_11982)
    
    # Assigning a BinOp to a Name (line 57):
    
    # Assigning a BinOp to a Name (line 57):
    # Getting the type of 'groups_of_3' (line 57)
    groups_of_3_11983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'groups_of_3')
    int_11984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'int')
    # Applying the binary operator '*' (line 57)
    result_mul_11985 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 8), '*', groups_of_3_11983, int_11984)
    
    # Assigning a type to the variable 'n' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'n', result_mul_11985)
    # Getting the type of 'leftover' (line 58)
    leftover_11986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'leftover')
    # Testing if the type of an if condition is none (line 58)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 4), leftover_11986):
        pass
    else:
        
        # Testing the type of an if condition (line 58)
        if_condition_11987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), leftover_11986)
        # Assigning a type to the variable 'if_condition_11987' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'if_condition_11987', if_condition_11987)
        # SSA begins for if statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'n' (line 59)
        n_11988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'n')
        int_11989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 13), 'int')
        # Applying the binary operator '+=' (line 59)
        result_iadd_11990 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 8), '+=', n_11988, int_11989)
        # Assigning a type to the variable 'n' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'n', result_iadd_11990)
        
        # SSA join for if statement (line 58)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'n' (line 60)
    n_11991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'n')
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', n_11991)
    
    # ################# End of 'base64_len(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'base64_len' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_11992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11992)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'base64_len'
    return stypy_return_type_11992

# Assigning a type to the variable 'base64_len' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'base64_len', base64_len)

@norecursion
def header_encode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_11993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 34), 'str', 'iso-8859-1')
    # Getting the type of 'False' (line 64)
    False_11994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 58), 'False')
    int_11995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'int')
    # Getting the type of 'NL' (line 65)
    NL_11996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 37), 'NL')
    defaults = [str_11993, False_11994, int_11995, NL_11996]
    # Create a new context for function 'header_encode'
    module_type_store = module_type_store.open_function_context('header_encode', 64, 0, False)
    
    # Passed parameters checking function
    header_encode.stypy_localization = localization
    header_encode.stypy_type_of_self = None
    header_encode.stypy_type_store = module_type_store
    header_encode.stypy_function_name = 'header_encode'
    header_encode.stypy_param_names_list = ['header', 'charset', 'keep_eols', 'maxlinelen', 'eol']
    header_encode.stypy_varargs_param_name = None
    header_encode.stypy_kwargs_param_name = None
    header_encode.stypy_call_defaults = defaults
    header_encode.stypy_call_varargs = varargs
    header_encode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'header_encode', ['header', 'charset', 'keep_eols', 'maxlinelen', 'eol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'header_encode', localization, ['header', 'charset', 'keep_eols', 'maxlinelen', 'eol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'header_encode(...)' code ##################

    str_11997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'str', 'Encode a single header line with Base64 encoding in a given charset.\n\n    Defined in RFC 2045, this Base64 encoding is identical to normal Base64\n    encoding, except that each line must be intelligently wrapped (respecting\n    the Base64 encoding), and subsequent lines must start with a space.\n\n    charset names the character set to use to encode the header.  It defaults\n    to iso-8859-1.\n\n    End-of-line characters (\\r, \\n, \\r\\n) will be automatically converted\n    to the canonical email line separator \\r\\n unless the keep_eols\n    parameter is True (the default is False).\n\n    Each line of the header will be terminated in the value of eol, which\n    defaults to "\\n".  Set this to "\\r\\n" if you are using the result of\n    this function directly in email.\n\n    The resulting string will be in the form:\n\n    "=?charset?b?WW/5ciBtYXp66XLrIHf8eiBhIGhhbXBzdGHuciBBIFlv+XIgbWF6euly?=\\n\n      =?charset?b?6yB3/HogYSBoYW1wc3Rh7nIgQkMgWW/5ciBtYXp66XLrIHf8eiBhIGhh?="\n\n    with each line wrapped at, at most, maxlinelen characters (defaults to 76\n    characters).\n    ')
    
    # Getting the type of 'header' (line 92)
    header_11998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'header')
    # Applying the 'not' unary operator (line 92)
    result_not__11999 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 7), 'not', header_11998)
    
    # Testing if the type of an if condition is none (line 92)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 92, 4), result_not__11999):
        pass
    else:
        
        # Testing the type of an if condition (line 92)
        if_condition_12000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 4), result_not__11999)
        # Assigning a type to the variable 'if_condition_12000' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'if_condition_12000', if_condition_12000)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'header' (line 93)
        header_12001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'header')
        # Assigning a type to the variable 'stypy_return_type' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stypy_return_type', header_12001)
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'keep_eols' (line 95)
    keep_eols_12002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'keep_eols')
    # Applying the 'not' unary operator (line 95)
    result_not__12003 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 7), 'not', keep_eols_12002)
    
    # Testing if the type of an if condition is none (line 95)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 95, 4), result_not__12003):
        pass
    else:
        
        # Testing the type of an if condition (line 95)
        if_condition_12004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 4), result_not__12003)
        # Assigning a type to the variable 'if_condition_12004' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'if_condition_12004', if_condition_12004)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to fix_eols(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'header' (line 96)
        header_12006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'header', False)
        # Processing the call keyword arguments (line 96)
        kwargs_12007 = {}
        # Getting the type of 'fix_eols' (line 96)
        fix_eols_12005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'fix_eols', False)
        # Calling fix_eols(args, kwargs) (line 96)
        fix_eols_call_result_12008 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), fix_eols_12005, *[header_12006], **kwargs_12007)
        
        # Assigning a type to the variable 'header' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'header', fix_eols_call_result_12008)
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 100):
    
    # Assigning a List to a Name (line 100):
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_12009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    
    # Assigning a type to the variable 'base64ed' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'base64ed', list_12009)
    
    # Assigning a BinOp to a Name (line 101):
    
    # Assigning a BinOp to a Name (line 101):
    # Getting the type of 'maxlinelen' (line 101)
    maxlinelen_12010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'maxlinelen')
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'charset' (line 101)
    charset_12012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'charset', False)
    # Processing the call keyword arguments (line 101)
    kwargs_12013 = {}
    # Getting the type of 'len' (line 101)
    len_12011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_12014 = invoke(stypy.reporting.localization.Localization(__file__, 101, 31), len_12011, *[charset_12012], **kwargs_12013)
    
    # Applying the binary operator '-' (line 101)
    result_sub_12015 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 18), '-', maxlinelen_12010, len_call_result_12014)
    
    # Getting the type of 'MISC_LEN' (line 101)
    MISC_LEN_12016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 46), 'MISC_LEN')
    # Applying the binary operator '-' (line 101)
    result_sub_12017 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 44), '-', result_sub_12015, MISC_LEN_12016)
    
    # Assigning a type to the variable 'max_encoded' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'max_encoded', result_sub_12017)
    
    # Assigning a BinOp to a Name (line 102):
    
    # Assigning a BinOp to a Name (line 102):
    # Getting the type of 'max_encoded' (line 102)
    max_encoded_12018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'max_encoded')
    int_12019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 34), 'int')
    # Applying the binary operator '*' (line 102)
    result_mul_12020 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 20), '*', max_encoded_12018, int_12019)
    
    int_12021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 39), 'int')
    # Applying the binary operator '//' (line 102)
    result_floordiv_12022 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 36), '//', result_mul_12020, int_12021)
    
    # Assigning a type to the variable 'max_unencoded' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'max_unencoded', result_floordiv_12022)
    
    
    # Call to range(...): (line 104)
    # Processing the call arguments (line 104)
    int_12024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 19), 'int')
    
    # Call to len(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'header' (line 104)
    header_12026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'header', False)
    # Processing the call keyword arguments (line 104)
    kwargs_12027 = {}
    # Getting the type of 'len' (line 104)
    len_12025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 22), 'len', False)
    # Calling len(args, kwargs) (line 104)
    len_call_result_12028 = invoke(stypy.reporting.localization.Localization(__file__, 104, 22), len_12025, *[header_12026], **kwargs_12027)
    
    # Getting the type of 'max_unencoded' (line 104)
    max_unencoded_12029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'max_unencoded', False)
    # Processing the call keyword arguments (line 104)
    kwargs_12030 = {}
    # Getting the type of 'range' (line 104)
    range_12023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'range', False)
    # Calling range(args, kwargs) (line 104)
    range_call_result_12031 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), range_12023, *[int_12024, len_call_result_12028, max_unencoded_12029], **kwargs_12030)
    
    # Assigning a type to the variable 'range_call_result_12031' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'range_call_result_12031', range_call_result_12031)
    # Testing if the for loop is going to be iterated (line 104)
    # Testing the type of a for loop iterable (line 104)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 104, 4), range_call_result_12031)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 104, 4), range_call_result_12031):
        # Getting the type of the for loop variable (line 104)
        for_loop_var_12032 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 104, 4), range_call_result_12031)
        # Assigning a type to the variable 'i' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'i', for_loop_var_12032)
        # SSA begins for a for statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to b2a_base64(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 105)
        i_12036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 42), 'i', False)
        # Getting the type of 'i' (line 105)
        i_12037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 44), 'i', False)
        # Getting the type of 'max_unencoded' (line 105)
        max_unencoded_12038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 46), 'max_unencoded', False)
        # Applying the binary operator '+' (line 105)
        result_add_12039 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 44), '+', i_12037, max_unencoded_12038)
        
        slice_12040 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 105, 35), i_12036, result_add_12039, None)
        # Getting the type of 'header' (line 105)
        header_12041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 35), 'header', False)
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___12042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 35), header_12041, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_12043 = invoke(stypy.reporting.localization.Localization(__file__, 105, 35), getitem___12042, slice_12040)
        
        # Processing the call keyword arguments (line 105)
        kwargs_12044 = {}
        # Getting the type of 'b2a_base64' (line 105)
        b2a_base64_12035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'b2a_base64', False)
        # Calling b2a_base64(args, kwargs) (line 105)
        b2a_base64_call_result_12045 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), b2a_base64_12035, *[subscript_call_result_12043], **kwargs_12044)
        
        # Processing the call keyword arguments (line 105)
        kwargs_12046 = {}
        # Getting the type of 'base64ed' (line 105)
        base64ed_12033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'base64ed', False)
        # Obtaining the member 'append' of a type (line 105)
        append_12034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), base64ed_12033, 'append')
        # Calling append(args, kwargs) (line 105)
        append_call_result_12047 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), append_12034, *[b2a_base64_call_result_12045], **kwargs_12046)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a List to a Name (line 108):
    
    # Assigning a List to a Name (line 108):
    
    # Obtaining an instance of the builtin type 'list' (line 108)
    list_12048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 108)
    
    # Assigning a type to the variable 'lines' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'lines', list_12048)
    
    # Getting the type of 'base64ed' (line 109)
    base64ed_12049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'base64ed')
    # Assigning a type to the variable 'base64ed_12049' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'base64ed_12049', base64ed_12049)
    # Testing if the for loop is going to be iterated (line 109)
    # Testing the type of a for loop iterable (line 109)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 4), base64ed_12049)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 109, 4), base64ed_12049):
        # Getting the type of the for loop variable (line 109)
        for_loop_var_12050 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 4), base64ed_12049)
        # Assigning a type to the variable 'line' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'line', for_loop_var_12050)
        # SSA begins for a for statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to endswith(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'NL' (line 111)
        NL_12053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'NL', False)
        # Processing the call keyword arguments (line 111)
        kwargs_12054 = {}
        # Getting the type of 'line' (line 111)
        line_12051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'line', False)
        # Obtaining the member 'endswith' of a type (line 111)
        endswith_12052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 11), line_12051, 'endswith')
        # Calling endswith(args, kwargs) (line 111)
        endswith_call_result_12055 = invoke(stypy.reporting.localization.Localization(__file__, 111, 11), endswith_12052, *[NL_12053], **kwargs_12054)
        
        # Testing if the type of an if condition is none (line 111)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 111, 8), endswith_call_result_12055):
            pass
        else:
            
            # Testing the type of an if condition (line 111)
            if_condition_12056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 8), endswith_call_result_12055)
            # Assigning a type to the variable 'if_condition_12056' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'if_condition_12056', if_condition_12056)
            # SSA begins for if statement (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 112):
            
            # Assigning a Subscript to a Name (line 112):
            
            # Obtaining the type of the subscript
            int_12057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 25), 'int')
            slice_12058 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 112, 19), None, int_12057, None)
            # Getting the type of 'line' (line 112)
            line_12059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'line')
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___12060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), line_12059, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_12061 = invoke(stypy.reporting.localization.Localization(__file__, 112, 19), getitem___12060, slice_12058)
            
            # Assigning a type to the variable 'line' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'line', subscript_call_result_12061)
            # SSA join for if statement (line 111)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to append(...): (line 114)
        # Processing the call arguments (line 114)
        str_12064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 21), 'str', '=?%s?b?%s?=')
        
        # Obtaining an instance of the builtin type 'tuple' (line 114)
        tuple_12065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 114)
        # Adding element type (line 114)
        # Getting the type of 'charset' (line 114)
        charset_12066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'charset', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), tuple_12065, charset_12066)
        # Adding element type (line 114)
        # Getting the type of 'line' (line 114)
        line_12067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 47), 'line', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), tuple_12065, line_12067)
        
        # Applying the binary operator '%' (line 114)
        result_mod_12068 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 21), '%', str_12064, tuple_12065)
        
        # Processing the call keyword arguments (line 114)
        kwargs_12069 = {}
        # Getting the type of 'lines' (line 114)
        lines_12062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 114)
        append_12063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), lines_12062, 'append')
        # Calling append(args, kwargs) (line 114)
        append_call_result_12070 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), append_12063, *[result_mod_12068], **kwargs_12069)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a BinOp to a Name (line 117):
    
    # Assigning a BinOp to a Name (line 117):
    # Getting the type of 'eol' (line 117)
    eol_12071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'eol')
    str_12072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 19), 'str', ' ')
    # Applying the binary operator '+' (line 117)
    result_add_12073 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 13), '+', eol_12071, str_12072)
    
    # Assigning a type to the variable 'joiner' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'joiner', result_add_12073)
    
    # Call to join(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'lines' (line 118)
    lines_12076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 23), 'lines', False)
    # Processing the call keyword arguments (line 118)
    kwargs_12077 = {}
    # Getting the type of 'joiner' (line 118)
    joiner_12074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'joiner', False)
    # Obtaining the member 'join' of a type (line 118)
    join_12075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 11), joiner_12074, 'join')
    # Calling join(args, kwargs) (line 118)
    join_call_result_12078 = invoke(stypy.reporting.localization.Localization(__file__, 118, 11), join_12075, *[lines_12076], **kwargs_12077)
    
    # Assigning a type to the variable 'stypy_return_type' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type', join_call_result_12078)
    
    # ################# End of 'header_encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'header_encode' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_12079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12079)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'header_encode'
    return stypy_return_type_12079

# Assigning a type to the variable 'header_encode' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'header_encode', header_encode)

@norecursion
def encode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 122)
    True_12080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'True')
    int_12081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 38), 'int')
    # Getting the type of 'NL' (line 122)
    NL_12082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 46), 'NL')
    defaults = [True_12080, int_12081, NL_12082]
    # Create a new context for function 'encode'
    module_type_store = module_type_store.open_function_context('encode', 122, 0, False)
    
    # Passed parameters checking function
    encode.stypy_localization = localization
    encode.stypy_type_of_self = None
    encode.stypy_type_store = module_type_store
    encode.stypy_function_name = 'encode'
    encode.stypy_param_names_list = ['s', 'binary', 'maxlinelen', 'eol']
    encode.stypy_varargs_param_name = None
    encode.stypy_kwargs_param_name = None
    encode.stypy_call_defaults = defaults
    encode.stypy_call_varargs = varargs
    encode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode', ['s', 'binary', 'maxlinelen', 'eol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode', localization, ['s', 'binary', 'maxlinelen', 'eol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode(...)' code ##################

    str_12083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, (-1)), 'str', 'Encode a string with base64.\n\n    Each line will be wrapped at, at most, maxlinelen characters (defaults to\n    76 characters).\n\n    If binary is False, end-of-line characters will be converted to the\n    canonical email end-of-line sequence \\r\\n.  Otherwise they will be left\n    verbatim (this is the default).\n\n    Each line of encoded text will end with eol, which defaults to "\\n".  Set\n    this to "\\r\\n" if you will be using the result of this function directly\n    in an email.\n    ')
    
    # Getting the type of 's' (line 136)
    s_12084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 's')
    # Applying the 'not' unary operator (line 136)
    result_not__12085 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 7), 'not', s_12084)
    
    # Testing if the type of an if condition is none (line 136)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 136, 4), result_not__12085):
        pass
    else:
        
        # Testing the type of an if condition (line 136)
        if_condition_12086 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 4), result_not__12085)
        # Assigning a type to the variable 'if_condition_12086' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'if_condition_12086', if_condition_12086)
        # SSA begins for if statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 's' (line 137)
        s_12087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stypy_return_type', s_12087)
        # SSA join for if statement (line 136)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'binary' (line 139)
    binary_12088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'binary')
    # Applying the 'not' unary operator (line 139)
    result_not__12089 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 7), 'not', binary_12088)
    
    # Testing if the type of an if condition is none (line 139)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 139, 4), result_not__12089):
        pass
    else:
        
        # Testing the type of an if condition (line 139)
        if_condition_12090 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 4), result_not__12089)
        # Assigning a type to the variable 'if_condition_12090' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'if_condition_12090', if_condition_12090)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to fix_eols(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 's' (line 140)
        s_12092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 's', False)
        # Processing the call keyword arguments (line 140)
        kwargs_12093 = {}
        # Getting the type of 'fix_eols' (line 140)
        fix_eols_12091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'fix_eols', False)
        # Calling fix_eols(args, kwargs) (line 140)
        fix_eols_call_result_12094 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), fix_eols_12091, *[s_12092], **kwargs_12093)
        
        # Assigning a type to the variable 's' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 's', fix_eols_call_result_12094)
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 142):
    
    # Assigning a List to a Name (line 142):
    
    # Obtaining an instance of the builtin type 'list' (line 142)
    list_12095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 142)
    
    # Assigning a type to the variable 'encvec' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'encvec', list_12095)
    
    # Assigning a BinOp to a Name (line 143):
    
    # Assigning a BinOp to a Name (line 143):
    # Getting the type of 'maxlinelen' (line 143)
    maxlinelen_12096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'maxlinelen')
    int_12097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 33), 'int')
    # Applying the binary operator '*' (line 143)
    result_mul_12098 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 20), '*', maxlinelen_12096, int_12097)
    
    int_12099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 38), 'int')
    # Applying the binary operator '//' (line 143)
    result_floordiv_12100 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 35), '//', result_mul_12098, int_12099)
    
    # Assigning a type to the variable 'max_unencoded' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'max_unencoded', result_floordiv_12100)
    
    
    # Call to range(...): (line 144)
    # Processing the call arguments (line 144)
    int_12102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 19), 'int')
    
    # Call to len(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 's' (line 144)
    s_12104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 's', False)
    # Processing the call keyword arguments (line 144)
    kwargs_12105 = {}
    # Getting the type of 'len' (line 144)
    len_12103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'len', False)
    # Calling len(args, kwargs) (line 144)
    len_call_result_12106 = invoke(stypy.reporting.localization.Localization(__file__, 144, 22), len_12103, *[s_12104], **kwargs_12105)
    
    # Getting the type of 'max_unencoded' (line 144)
    max_unencoded_12107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 30), 'max_unencoded', False)
    # Processing the call keyword arguments (line 144)
    kwargs_12108 = {}
    # Getting the type of 'range' (line 144)
    range_12101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 13), 'range', False)
    # Calling range(args, kwargs) (line 144)
    range_call_result_12109 = invoke(stypy.reporting.localization.Localization(__file__, 144, 13), range_12101, *[int_12102, len_call_result_12106, max_unencoded_12107], **kwargs_12108)
    
    # Assigning a type to the variable 'range_call_result_12109' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'range_call_result_12109', range_call_result_12109)
    # Testing if the for loop is going to be iterated (line 144)
    # Testing the type of a for loop iterable (line 144)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 144, 4), range_call_result_12109)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 144, 4), range_call_result_12109):
        # Getting the type of the for loop variable (line 144)
        for_loop_var_12110 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 144, 4), range_call_result_12109)
        # Assigning a type to the variable 'i' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'i', for_loop_var_12110)
        # SSA begins for a for statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to b2a_base64(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 147)
        i_12112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'i', False)
        # Getting the type of 'i' (line 147)
        i_12113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 29), 'i', False)
        # Getting the type of 'max_unencoded' (line 147)
        max_unencoded_12114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'max_unencoded', False)
        # Applying the binary operator '+' (line 147)
        result_add_12115 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 29), '+', i_12113, max_unencoded_12114)
        
        slice_12116 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 147, 25), i_12112, result_add_12115, None)
        # Getting the type of 's' (line 147)
        s_12117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 's', False)
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___12118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), s_12117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_12119 = invoke(stypy.reporting.localization.Localization(__file__, 147, 25), getitem___12118, slice_12116)
        
        # Processing the call keyword arguments (line 147)
        kwargs_12120 = {}
        # Getting the type of 'b2a_base64' (line 147)
        b2a_base64_12111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'b2a_base64', False)
        # Calling b2a_base64(args, kwargs) (line 147)
        b2a_base64_call_result_12121 = invoke(stypy.reporting.localization.Localization(__file__, 147, 14), b2a_base64_12111, *[subscript_call_result_12119], **kwargs_12120)
        
        # Assigning a type to the variable 'enc' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'enc', b2a_base64_call_result_12121)
        
        # Evaluating a boolean operation
        
        # Call to endswith(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'NL' (line 148)
        NL_12124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'NL', False)
        # Processing the call keyword arguments (line 148)
        kwargs_12125 = {}
        # Getting the type of 'enc' (line 148)
        enc_12122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'enc', False)
        # Obtaining the member 'endswith' of a type (line 148)
        endswith_12123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 11), enc_12122, 'endswith')
        # Calling endswith(args, kwargs) (line 148)
        endswith_call_result_12126 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), endswith_12123, *[NL_12124], **kwargs_12125)
        
        
        # Getting the type of 'eol' (line 148)
        eol_12127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), 'eol')
        # Getting the type of 'NL' (line 148)
        NL_12128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 39), 'NL')
        # Applying the binary operator '!=' (line 148)
        result_ne_12129 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 32), '!=', eol_12127, NL_12128)
        
        # Applying the binary operator 'and' (line 148)
        result_and_keyword_12130 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 11), 'and', endswith_call_result_12126, result_ne_12129)
        
        # Testing if the type of an if condition is none (line 148)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 148, 8), result_and_keyword_12130):
            pass
        else:
            
            # Testing the type of an if condition (line 148)
            if_condition_12131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 8), result_and_keyword_12130)
            # Assigning a type to the variable 'if_condition_12131' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'if_condition_12131', if_condition_12131)
            # SSA begins for if statement (line 148)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 149):
            
            # Assigning a BinOp to a Name (line 149):
            
            # Obtaining the type of the subscript
            int_12132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 23), 'int')
            slice_12133 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 149, 18), None, int_12132, None)
            # Getting the type of 'enc' (line 149)
            enc_12134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'enc')
            # Obtaining the member '__getitem__' of a type (line 149)
            getitem___12135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 18), enc_12134, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 149)
            subscript_call_result_12136 = invoke(stypy.reporting.localization.Localization(__file__, 149, 18), getitem___12135, slice_12133)
            
            # Getting the type of 'eol' (line 149)
            eol_12137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 29), 'eol')
            # Applying the binary operator '+' (line 149)
            result_add_12138 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 18), '+', subscript_call_result_12136, eol_12137)
            
            # Assigning a type to the variable 'enc' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'enc', result_add_12138)
            # SSA join for if statement (line 148)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to append(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'enc' (line 150)
        enc_12141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 22), 'enc', False)
        # Processing the call keyword arguments (line 150)
        kwargs_12142 = {}
        # Getting the type of 'encvec' (line 150)
        encvec_12139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'encvec', False)
        # Obtaining the member 'append' of a type (line 150)
        append_12140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), encvec_12139, 'append')
        # Calling append(args, kwargs) (line 150)
        append_call_result_12143 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), append_12140, *[enc_12141], **kwargs_12142)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to join(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'encvec' (line 151)
    encvec_12146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 28), 'encvec', False)
    # Processing the call keyword arguments (line 151)
    kwargs_12147 = {}
    # Getting the type of 'EMPTYSTRING' (line 151)
    EMPTYSTRING_12144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'EMPTYSTRING', False)
    # Obtaining the member 'join' of a type (line 151)
    join_12145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), EMPTYSTRING_12144, 'join')
    # Calling join(args, kwargs) (line 151)
    join_call_result_12148 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), join_12145, *[encvec_12146], **kwargs_12147)
    
    # Assigning a type to the variable 'stypy_return_type' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type', join_call_result_12148)
    
    # ################# End of 'encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_12149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12149)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode'
    return stypy_return_type_12149

# Assigning a type to the variable 'encode' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'encode', encode)

# Assigning a Name to a Name (line 155):

# Assigning a Name to a Name (line 155):
# Getting the type of 'encode' (line 155)
encode_12150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'encode')
# Assigning a type to the variable 'body_encode' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'body_encode', encode_12150)

# Assigning a Name to a Name (line 156):

# Assigning a Name to a Name (line 156):
# Getting the type of 'encode' (line 156)
encode_12151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'encode')
# Assigning a type to the variable 'encodestring' (line 156)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 0), 'encodestring', encode_12151)

@norecursion
def decode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 160)
    None_12152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), 'None')
    defaults = [None_12152]
    # Create a new context for function 'decode'
    module_type_store = module_type_store.open_function_context('decode', 160, 0, False)
    
    # Passed parameters checking function
    decode.stypy_localization = localization
    decode.stypy_type_of_self = None
    decode.stypy_type_store = module_type_store
    decode.stypy_function_name = 'decode'
    decode.stypy_param_names_list = ['s', 'convert_eols']
    decode.stypy_varargs_param_name = None
    decode.stypy_kwargs_param_name = None
    decode.stypy_call_defaults = defaults
    decode.stypy_call_varargs = varargs
    decode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decode', ['s', 'convert_eols'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decode', localization, ['s', 'convert_eols'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decode(...)' code ##################

    str_12153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, (-1)), 'str', 'Decode a raw base64 string.\n\n    If convert_eols is set to a string value, all canonical email linefeeds,\n    e.g. "\\r\\n", in the decoded text will be converted to the value of\n    convert_eols.  os.linesep is a good choice for convert_eols if you are\n    decoding a text attachment.\n\n    This function does not parse a full MIME header value encoded with\n    base64 (like =?iso-8859-1?b?bmloISBuaWgh?=) -- please use the high\n    level email.header class for that functionality.\n    ')
    
    # Getting the type of 's' (line 172)
    s_12154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 's')
    # Applying the 'not' unary operator (line 172)
    result_not__12155 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 7), 'not', s_12154)
    
    # Testing if the type of an if condition is none (line 172)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 172, 4), result_not__12155):
        pass
    else:
        
        # Testing the type of an if condition (line 172)
        if_condition_12156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 4), result_not__12155)
        # Assigning a type to the variable 'if_condition_12156' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'if_condition_12156', if_condition_12156)
        # SSA begins for if statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 's' (line 173)
        s_12157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', s_12157)
        # SSA join for if statement (line 172)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to a2b_base64(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 's' (line 175)
    s_12159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 's', False)
    # Processing the call keyword arguments (line 175)
    kwargs_12160 = {}
    # Getting the type of 'a2b_base64' (line 175)
    a2b_base64_12158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 10), 'a2b_base64', False)
    # Calling a2b_base64(args, kwargs) (line 175)
    a2b_base64_call_result_12161 = invoke(stypy.reporting.localization.Localization(__file__, 175, 10), a2b_base64_12158, *[s_12159], **kwargs_12160)
    
    # Assigning a type to the variable 'dec' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'dec', a2b_base64_call_result_12161)
    # Getting the type of 'convert_eols' (line 176)
    convert_eols_12162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 7), 'convert_eols')
    # Testing if the type of an if condition is none (line 176)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 176, 4), convert_eols_12162):
        pass
    else:
        
        # Testing the type of an if condition (line 176)
        if_condition_12163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 4), convert_eols_12162)
        # Assigning a type to the variable 'if_condition_12163' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'if_condition_12163', if_condition_12163)
        # SSA begins for if statement (line 176)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to replace(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'CRLF' (line 177)
        CRLF_12166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 27), 'CRLF', False)
        # Getting the type of 'convert_eols' (line 177)
        convert_eols_12167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 33), 'convert_eols', False)
        # Processing the call keyword arguments (line 177)
        kwargs_12168 = {}
        # Getting the type of 'dec' (line 177)
        dec_12164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'dec', False)
        # Obtaining the member 'replace' of a type (line 177)
        replace_12165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 15), dec_12164, 'replace')
        # Calling replace(args, kwargs) (line 177)
        replace_call_result_12169 = invoke(stypy.reporting.localization.Localization(__file__, 177, 15), replace_12165, *[CRLF_12166, convert_eols_12167], **kwargs_12168)
        
        # Assigning a type to the variable 'stypy_return_type' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'stypy_return_type', replace_call_result_12169)
        # SSA join for if statement (line 176)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'dec' (line 178)
    dec_12170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'dec')
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type', dec_12170)
    
    # ################# End of 'decode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_12171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12171)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode'
    return stypy_return_type_12171

# Assigning a type to the variable 'decode' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'decode', decode)

# Assigning a Name to a Name (line 182):

# Assigning a Name to a Name (line 182):
# Getting the type of 'decode' (line 182)
decode_12172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 14), 'decode')
# Assigning a type to the variable 'body_decode' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'body_decode', decode_12172)

# Assigning a Name to a Name (line 183):

# Assigning a Name to a Name (line 183):
# Getting the type of 'decode' (line 183)
decode_12173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'decode')
# Assigning a type to the variable 'decodestring' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'decodestring', decode_12173)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
