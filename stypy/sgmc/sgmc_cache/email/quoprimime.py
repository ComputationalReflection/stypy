
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Ben Gertzfield
3: # Contact: email-sig@python.org
4: 
5: '''Quoted-printable content transfer encoding per RFCs 2045-2047.
6: 
7: This module handles the content transfer encoding method defined in RFC 2045
8: to encode US ASCII-like 8-bit data called `quoted-printable'.  It is used to
9: safely encode text that is in a character set similar to the 7-bit US ASCII
10: character set, but that includes some 8-bit characters that are normally not
11: allowed in email bodies or headers.
12: 
13: Quoted-printable is very space-inefficient for encoding binary files; use the
14: email.base64mime module for that instead.
15: 
16: This module provides an interface to encode and decode both headers and bodies
17: with quoted-printable encoding.
18: 
19: RFC 2045 defines a method for including character set information in an
20: `encoded-word' in a header.  This method is commonly used for 8-bit real names
21: in To:/From:/Cc: etc. fields, as well as Subject: lines.
22: 
23: This module does not do the line wrapping or end-of-line character
24: conversion necessary for proper internationalized headers; it only
25: does dumb encoding and decoding.  To deal with the various line
26: wrapping issues, use the email.header module.
27: '''
28: 
29: __all__ = [
30:     'body_decode',
31:     'body_encode',
32:     'body_quopri_check',
33:     'body_quopri_len',
34:     'decode',
35:     'decodestring',
36:     'encode',
37:     'encodestring',
38:     'header_decode',
39:     'header_encode',
40:     'header_quopri_check',
41:     'header_quopri_len',
42:     'quote',
43:     'unquote',
44:     ]
45: 
46: import re
47: 
48: from string import hexdigits
49: from email.utils import fix_eols
50: 
51: CRLF = '\r\n'
52: NL = '\n'
53: 
54: # See also Charset.py
55: MISC_LEN = 7
56: 
57: hqre = re.compile(r'[^-a-zA-Z0-9!*+/ ]')
58: bqre = re.compile(r'[^ !-<>-~\t]')
59: 
60: 
61: 
62: # Helpers
63: def header_quopri_check(c):
64:     '''Return True if the character should be escaped with header quopri.'''
65:     return bool(hqre.match(c))
66: 
67: 
68: def body_quopri_check(c):
69:     '''Return True if the character should be escaped with body quopri.'''
70:     return bool(bqre.match(c))
71: 
72: 
73: def header_quopri_len(s):
74:     '''Return the length of str when it is encoded with header quopri.'''
75:     count = 0
76:     for c in s:
77:         if hqre.match(c):
78:             count += 3
79:         else:
80:             count += 1
81:     return count
82: 
83: 
84: def body_quopri_len(str):
85:     '''Return the length of str when it is encoded with body quopri.'''
86:     count = 0
87:     for c in str:
88:         if bqre.match(c):
89:             count += 3
90:         else:
91:             count += 1
92:     return count
93: 
94: 
95: def _max_append(L, s, maxlen, extra=''):
96:     if not L:
97:         L.append(s.lstrip())
98:     elif len(L[-1]) + len(s) <= maxlen:
99:         L[-1] += extra + s
100:     else:
101:         L.append(s.lstrip())
102: 
103: 
104: def unquote(s):
105:     '''Turn a string in the form =AB to the ASCII character with value 0xab'''
106:     return chr(int(s[1:3], 16))
107: 
108: 
109: def quote(c):
110:     return "=%02X" % ord(c)
111: 
112: 
113: 
114: def header_encode(header, charset="iso-8859-1", keep_eols=False,
115:                   maxlinelen=76, eol=NL):
116:     '''Encode a single header line with quoted-printable (like) encoding.
117: 
118:     Defined in RFC 2045, this `Q' encoding is similar to quoted-printable, but
119:     used specifically for email header fields to allow charsets with mostly 7
120:     bit characters (and some 8 bit) to remain more or less readable in non-RFC
121:     2045 aware mail clients.
122: 
123:     charset names the character set to use to encode the header.  It defaults
124:     to iso-8859-1.
125: 
126:     The resulting string will be in the form:
127: 
128:     "=?charset?q?I_f=E2rt_in_your_g=E8n=E8ral_dire=E7tion?\\n
129:       =?charset?q?Silly_=C8nglish_Kn=EEghts?="
130: 
131:     with each line wrapped safely at, at most, maxlinelen characters (defaults
132:     to 76 characters).  If maxlinelen is None, the entire string is encoded in
133:     one chunk with no splitting.
134: 
135:     End-of-line characters (\\r, \\n, \\r\\n) will be automatically converted
136:     to the canonical email line separator \\r\\n unless the keep_eols
137:     parameter is True (the default is False).
138: 
139:     Each line of the header will be terminated in the value of eol, which
140:     defaults to "\\n".  Set this to "\\r\\n" if you are using the result of
141:     this function directly in email.
142:     '''
143:     # Return empty headers unchanged
144:     if not header:
145:         return header
146: 
147:     if not keep_eols:
148:         header = fix_eols(header)
149: 
150:     # Quopri encode each line, in encoded chunks no greater than maxlinelen in
151:     # length, after the RFC chrome is added in.
152:     quoted = []
153:     if maxlinelen is None:
154:         # An obnoxiously large number that's good enough
155:         max_encoded = 100000
156:     else:
157:         max_encoded = maxlinelen - len(charset) - MISC_LEN - 1
158: 
159:     for c in header:
160:         # Space may be represented as _ instead of =20 for readability
161:         if c == ' ':
162:             _max_append(quoted, '_', max_encoded)
163:         # These characters can be included verbatim
164:         elif not hqre.match(c):
165:             _max_append(quoted, c, max_encoded)
166:         # Otherwise, replace with hex value like =E2
167:         else:
168:             _max_append(quoted, "=%02X" % ord(c), max_encoded)
169: 
170:     # Now add the RFC chrome to each encoded chunk and glue the chunks
171:     # together.  BAW: should we be able to specify the leading whitespace in
172:     # the joiner?
173:     joiner = eol + ' '
174:     return joiner.join(['=?%s?q?%s?=' % (charset, line) for line in quoted])
175: 
176: 
177: 
178: def encode(body, binary=False, maxlinelen=76, eol=NL):
179:     '''Encode with quoted-printable, wrapping at maxlinelen characters.
180: 
181:     If binary is False (the default), end-of-line characters will be converted
182:     to the canonical email end-of-line sequence \\r\\n.  Otherwise they will
183:     be left verbatim.
184: 
185:     Each line of encoded text will end with eol, which defaults to "\\n".  Set
186:     this to "\\r\\n" if you will be using the result of this function directly
187:     in an email.
188: 
189:     Each line will be wrapped at, at most, maxlinelen characters (defaults to
190:     76 characters).  Long lines will have the `soft linefeed' quoted-printable
191:     character "=" appended to them, so the decoded text will be identical to
192:     the original text.
193:     '''
194:     if not body:
195:         return body
196: 
197:     if not binary:
198:         body = fix_eols(body)
199: 
200:     # BAW: We're accumulating the body text by string concatenation.  That
201:     # can't be very efficient, but I don't have time now to rewrite it.  It
202:     # just feels like this algorithm could be more efficient.
203:     encoded_body = ''
204:     lineno = -1
205:     # Preserve line endings here so we can check later to see an eol needs to
206:     # be added to the output later.
207:     lines = body.splitlines(1)
208:     for line in lines:
209:         # But strip off line-endings for processing this line.
210:         if line.endswith(CRLF):
211:             line = line[:-2]
212:         elif line[-1] in CRLF:
213:             line = line[:-1]
214: 
215:         lineno += 1
216:         encoded_line = ''
217:         prev = None
218:         linelen = len(line)
219:         # Now we need to examine every character to see if it needs to be
220:         # quopri encoded.  BAW: again, string concatenation is inefficient.
221:         for j in range(linelen):
222:             c = line[j]
223:             prev = c
224:             if bqre.match(c):
225:                 c = quote(c)
226:             elif j+1 == linelen:
227:                 # Check for whitespace at end of line; special case
228:                 if c not in ' \t':
229:                     encoded_line += c
230:                 prev = c
231:                 continue
232:             # Check to see to see if the line has reached its maximum length
233:             if len(encoded_line) + len(c) >= maxlinelen:
234:                 encoded_body += encoded_line + '=' + eol
235:                 encoded_line = ''
236:             encoded_line += c
237:         # Now at end of line..
238:         if prev and prev in ' \t':
239:             # Special case for whitespace at end of file
240:             if lineno + 1 == len(lines):
241:                 prev = quote(prev)
242:                 if len(encoded_line) + len(prev) > maxlinelen:
243:                     encoded_body += encoded_line + '=' + eol + prev
244:                 else:
245:                     encoded_body += encoded_line + prev
246:             # Just normal whitespace at end of line
247:             else:
248:                 encoded_body += encoded_line + prev + '=' + eol
249:             encoded_line = ''
250:         # Now look at the line we just finished and it has a line ending, we
251:         # need to add eol to the end of the line.
252:         if lines[lineno].endswith(CRLF) or lines[lineno][-1] in CRLF:
253:             encoded_body += encoded_line + eol
254:         else:
255:             encoded_body += encoded_line
256:         encoded_line = ''
257:     return encoded_body
258: 
259: 
260: # For convenience and backwards compatibility w/ standard base64 module
261: body_encode = encode
262: encodestring = encode
263: 
264: 
265: 
266: # BAW: I'm not sure if the intent was for the signature of this function to be
267: # the same as base64MIME.decode() or not...
268: def decode(encoded, eol=NL):
269:     '''Decode a quoted-printable string.
270: 
271:     Lines are separated with eol, which defaults to \\n.
272:     '''
273:     if not encoded:
274:         return encoded
275:     # BAW: see comment in encode() above.  Again, we're building up the
276:     # decoded string with string concatenation, which could be done much more
277:     # efficiently.
278:     decoded = ''
279: 
280:     for line in encoded.splitlines():
281:         line = line.rstrip()
282:         if not line:
283:             decoded += eol
284:             continue
285: 
286:         i = 0
287:         n = len(line)
288:         while i < n:
289:             c = line[i]
290:             if c != '=':
291:                 decoded += c
292:                 i += 1
293:             # Otherwise, c == "=".  Are we at the end of the line?  If so, add
294:             # a soft line break.
295:             elif i+1 == n:
296:                 i += 1
297:                 continue
298:             # Decode if in form =AB
299:             elif i+2 < n and line[i+1] in hexdigits and line[i+2] in hexdigits:
300:                 decoded += unquote(line[i:i+3])
301:                 i += 3
302:             # Otherwise, not in form =AB, pass literally
303:             else:
304:                 decoded += c
305:                 i += 1
306: 
307:             if i == n:
308:                 decoded += eol
309:     # Special case if original string did not end with eol
310:     if not encoded.endswith(eol) and decoded.endswith(eol):
311:         decoded = decoded[:-1]
312:     return decoded
313: 
314: 
315: # For convenience and backwards compatibility w/ standard base64 module
316: body_decode = decode
317: decodestring = decode
318: 
319: 
320: 
321: def _unquote_match(match):
322:     '''Turn a match in the form =AB to the ASCII character with value 0xab'''
323:     s = match.group(0)
324:     return unquote(s)
325: 
326: 
327: # Header decoding is done a bit differently
328: def header_decode(s):
329:     '''Decode a string encoded with RFC 2045 MIME header `Q' encoding.
330: 
331:     This function does not parse a full MIME header value encoded with
332:     quoted-printable (like =?iso-8859-1?q?Hello_World?=) -- please use
333:     the high level email.header class for that functionality.
334:     '''
335:     s = s.replace('_', ' ')
336:     return re.sub(r'=[a-fA-F0-9]{2}', _unquote_match, s)
337: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_17731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', "Quoted-printable content transfer encoding per RFCs 2045-2047.\n\nThis module handles the content transfer encoding method defined in RFC 2045\nto encode US ASCII-like 8-bit data called `quoted-printable'.  It is used to\nsafely encode text that is in a character set similar to the 7-bit US ASCII\ncharacter set, but that includes some 8-bit characters that are normally not\nallowed in email bodies or headers.\n\nQuoted-printable is very space-inefficient for encoding binary files; use the\nemail.base64mime module for that instead.\n\nThis module provides an interface to encode and decode both headers and bodies\nwith quoted-printable encoding.\n\nRFC 2045 defines a method for including character set information in an\n`encoded-word' in a header.  This method is commonly used for 8-bit real names\nin To:/From:/Cc: etc. fields, as well as Subject: lines.\n\nThis module does not do the line wrapping or end-of-line character\nconversion necessary for proper internationalized headers; it only\ndoes dumb encoding and decoding.  To deal with the various line\nwrapping issues, use the email.header module.\n")

# Assigning a List to a Name (line 29):
__all__ = ['body_decode', 'body_encode', 'body_quopri_check', 'body_quopri_len', 'decode', 'decodestring', 'encode', 'encodestring', 'header_decode', 'header_encode', 'header_quopri_check', 'header_quopri_len', 'quote', 'unquote']
module_type_store.set_exportable_members(['body_decode', 'body_encode', 'body_quopri_check', 'body_quopri_len', 'decode', 'decodestring', 'encode', 'encodestring', 'header_decode', 'header_encode', 'header_quopri_check', 'header_quopri_len', 'quote', 'unquote'])

# Obtaining an instance of the builtin type 'list' (line 29)
list_17732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
str_17733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', 'body_decode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17733)
# Adding element type (line 29)
str_17734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'str', 'body_encode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17734)
# Adding element type (line 29)
str_17735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', 'body_quopri_check')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17735)
# Adding element type (line 29)
str_17736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'str', 'body_quopri_len')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17736)
# Adding element type (line 29)
str_17737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'str', 'decode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17737)
# Adding element type (line 29)
str_17738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', 'decodestring')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17738)
# Adding element type (line 29)
str_17739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'encode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17739)
# Adding element type (line 29)
str_17740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'str', 'encodestring')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17740)
# Adding element type (line 29)
str_17741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'str', 'header_decode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17741)
# Adding element type (line 29)
str_17742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 4), 'str', 'header_encode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17742)
# Adding element type (line 29)
str_17743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'str', 'header_quopri_check')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17743)
# Adding element type (line 29)
str_17744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'str', 'header_quopri_len')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17744)
# Adding element type (line 29)
str_17745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'str', 'quote')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17745)
# Adding element type (line 29)
str_17746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'str', 'unquote')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_17732, str_17746)

# Assigning a type to the variable '__all__' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '__all__', list_17732)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 0))

# 'import re' statement (line 46)
import re

import_module(stypy.reporting.localization.Localization(__file__, 46, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 0))

# 'from string import hexdigits' statement (line 48)
try:
    from string import hexdigits

except:
    hexdigits = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'string', None, module_type_store, ['hexdigits'], [hexdigits])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 49, 0))

# 'from email.utils import fix_eols' statement (line 49)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_17747 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'email.utils')

if (type(import_17747) is not StypyTypeError):

    if (import_17747 != 'pyd_module'):
        __import__(import_17747)
        sys_modules_17748 = sys.modules[import_17747]
        import_from_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'email.utils', sys_modules_17748.module_type_store, module_type_store, ['fix_eols'])
        nest_module(stypy.reporting.localization.Localization(__file__, 49, 0), __file__, sys_modules_17748, sys_modules_17748.module_type_store, module_type_store)
    else:
        from email.utils import fix_eols

        import_from_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'email.utils', None, module_type_store, ['fix_eols'], [fix_eols])

else:
    # Assigning a type to the variable 'email.utils' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'email.utils', import_17747)

remove_current_file_folder_from_path('C:/Python27/lib/email/')


# Assigning a Str to a Name (line 51):
str_17749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 7), 'str', '\r\n')
# Assigning a type to the variable 'CRLF' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'CRLF', str_17749)

# Assigning a Str to a Name (line 52):
str_17750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 5), 'str', '\n')
# Assigning a type to the variable 'NL' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'NL', str_17750)

# Assigning a Num to a Name (line 55):
int_17751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 11), 'int')
# Assigning a type to the variable 'MISC_LEN' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'MISC_LEN', int_17751)

# Assigning a Call to a Name (line 57):

# Call to compile(...): (line 57)
# Processing the call arguments (line 57)
str_17754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 18), 'str', '[^-a-zA-Z0-9!*+/ ]')
# Processing the call keyword arguments (line 57)
kwargs_17755 = {}
# Getting the type of 're' (line 57)
re_17752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 're', False)
# Obtaining the member 'compile' of a type (line 57)
compile_17753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 7), re_17752, 'compile')
# Calling compile(args, kwargs) (line 57)
compile_call_result_17756 = invoke(stypy.reporting.localization.Localization(__file__, 57, 7), compile_17753, *[str_17754], **kwargs_17755)

# Assigning a type to the variable 'hqre' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'hqre', compile_call_result_17756)

# Assigning a Call to a Name (line 58):

# Call to compile(...): (line 58)
# Processing the call arguments (line 58)
str_17759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 18), 'str', '[^ !-<>-~\\t]')
# Processing the call keyword arguments (line 58)
kwargs_17760 = {}
# Getting the type of 're' (line 58)
re_17757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 're', False)
# Obtaining the member 'compile' of a type (line 58)
compile_17758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 7), re_17757, 'compile')
# Calling compile(args, kwargs) (line 58)
compile_call_result_17761 = invoke(stypy.reporting.localization.Localization(__file__, 58, 7), compile_17758, *[str_17759], **kwargs_17760)

# Assigning a type to the variable 'bqre' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'bqre', compile_call_result_17761)

@norecursion
def header_quopri_check(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'header_quopri_check'
    module_type_store = module_type_store.open_function_context('header_quopri_check', 63, 0, False)
    
    # Passed parameters checking function
    header_quopri_check.stypy_localization = localization
    header_quopri_check.stypy_type_of_self = None
    header_quopri_check.stypy_type_store = module_type_store
    header_quopri_check.stypy_function_name = 'header_quopri_check'
    header_quopri_check.stypy_param_names_list = ['c']
    header_quopri_check.stypy_varargs_param_name = None
    header_quopri_check.stypy_kwargs_param_name = None
    header_quopri_check.stypy_call_defaults = defaults
    header_quopri_check.stypy_call_varargs = varargs
    header_quopri_check.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'header_quopri_check', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'header_quopri_check', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'header_quopri_check(...)' code ##################

    str_17762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'str', 'Return True if the character should be escaped with header quopri.')
    
    # Call to bool(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Call to match(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'c' (line 65)
    c_17766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'c', False)
    # Processing the call keyword arguments (line 65)
    kwargs_17767 = {}
    # Getting the type of 'hqre' (line 65)
    hqre_17764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'hqre', False)
    # Obtaining the member 'match' of a type (line 65)
    match_17765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), hqre_17764, 'match')
    # Calling match(args, kwargs) (line 65)
    match_call_result_17768 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), match_17765, *[c_17766], **kwargs_17767)
    
    # Processing the call keyword arguments (line 65)
    kwargs_17769 = {}
    # Getting the type of 'bool' (line 65)
    bool_17763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'bool', False)
    # Calling bool(args, kwargs) (line 65)
    bool_call_result_17770 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), bool_17763, *[match_call_result_17768], **kwargs_17769)
    
    # Assigning a type to the variable 'stypy_return_type' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type', bool_call_result_17770)
    
    # ################# End of 'header_quopri_check(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'header_quopri_check' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_17771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17771)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'header_quopri_check'
    return stypy_return_type_17771

# Assigning a type to the variable 'header_quopri_check' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'header_quopri_check', header_quopri_check)

@norecursion
def body_quopri_check(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'body_quopri_check'
    module_type_store = module_type_store.open_function_context('body_quopri_check', 68, 0, False)
    
    # Passed parameters checking function
    body_quopri_check.stypy_localization = localization
    body_quopri_check.stypy_type_of_self = None
    body_quopri_check.stypy_type_store = module_type_store
    body_quopri_check.stypy_function_name = 'body_quopri_check'
    body_quopri_check.stypy_param_names_list = ['c']
    body_quopri_check.stypy_varargs_param_name = None
    body_quopri_check.stypy_kwargs_param_name = None
    body_quopri_check.stypy_call_defaults = defaults
    body_quopri_check.stypy_call_varargs = varargs
    body_quopri_check.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'body_quopri_check', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'body_quopri_check', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'body_quopri_check(...)' code ##################

    str_17772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'str', 'Return True if the character should be escaped with body quopri.')
    
    # Call to bool(...): (line 70)
    # Processing the call arguments (line 70)
    
    # Call to match(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'c' (line 70)
    c_17776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'c', False)
    # Processing the call keyword arguments (line 70)
    kwargs_17777 = {}
    # Getting the type of 'bqre' (line 70)
    bqre_17774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'bqre', False)
    # Obtaining the member 'match' of a type (line 70)
    match_17775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), bqre_17774, 'match')
    # Calling match(args, kwargs) (line 70)
    match_call_result_17778 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), match_17775, *[c_17776], **kwargs_17777)
    
    # Processing the call keyword arguments (line 70)
    kwargs_17779 = {}
    # Getting the type of 'bool' (line 70)
    bool_17773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'bool', False)
    # Calling bool(args, kwargs) (line 70)
    bool_call_result_17780 = invoke(stypy.reporting.localization.Localization(__file__, 70, 11), bool_17773, *[match_call_result_17778], **kwargs_17779)
    
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', bool_call_result_17780)
    
    # ################# End of 'body_quopri_check(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'body_quopri_check' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_17781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17781)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'body_quopri_check'
    return stypy_return_type_17781

# Assigning a type to the variable 'body_quopri_check' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'body_quopri_check', body_quopri_check)

@norecursion
def header_quopri_len(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'header_quopri_len'
    module_type_store = module_type_store.open_function_context('header_quopri_len', 73, 0, False)
    
    # Passed parameters checking function
    header_quopri_len.stypy_localization = localization
    header_quopri_len.stypy_type_of_self = None
    header_quopri_len.stypy_type_store = module_type_store
    header_quopri_len.stypy_function_name = 'header_quopri_len'
    header_quopri_len.stypy_param_names_list = ['s']
    header_quopri_len.stypy_varargs_param_name = None
    header_quopri_len.stypy_kwargs_param_name = None
    header_quopri_len.stypy_call_defaults = defaults
    header_quopri_len.stypy_call_varargs = varargs
    header_quopri_len.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'header_quopri_len', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'header_quopri_len', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'header_quopri_len(...)' code ##################

    str_17782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'str', 'Return the length of str when it is encoded with header quopri.')
    
    # Assigning a Num to a Name (line 75):
    int_17783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 12), 'int')
    # Assigning a type to the variable 'count' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'count', int_17783)
    
    # Getting the type of 's' (line 76)
    s_17784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 's')
    # Assigning a type to the variable 's_17784' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 's_17784', s_17784)
    # Testing if the for loop is going to be iterated (line 76)
    # Testing the type of a for loop iterable (line 76)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 76, 4), s_17784)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 76, 4), s_17784):
        # Getting the type of the for loop variable (line 76)
        for_loop_var_17785 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 76, 4), s_17784)
        # Assigning a type to the variable 'c' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'c', for_loop_var_17785)
        # SSA begins for a for statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to match(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'c' (line 77)
        c_17788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'c', False)
        # Processing the call keyword arguments (line 77)
        kwargs_17789 = {}
        # Getting the type of 'hqre' (line 77)
        hqre_17786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'hqre', False)
        # Obtaining the member 'match' of a type (line 77)
        match_17787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 11), hqre_17786, 'match')
        # Calling match(args, kwargs) (line 77)
        match_call_result_17790 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), match_17787, *[c_17788], **kwargs_17789)
        
        # Testing if the type of an if condition is none (line 77)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 8), match_call_result_17790):
            
            # Getting the type of 'count' (line 80)
            count_17795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'count')
            int_17796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'int')
            # Applying the binary operator '+=' (line 80)
            result_iadd_17797 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 12), '+=', count_17795, int_17796)
            # Assigning a type to the variable 'count' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'count', result_iadd_17797)
            
        else:
            
            # Testing the type of an if condition (line 77)
            if_condition_17791 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 8), match_call_result_17790)
            # Assigning a type to the variable 'if_condition_17791' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'if_condition_17791', if_condition_17791)
            # SSA begins for if statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'count' (line 78)
            count_17792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'count')
            int_17793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'int')
            # Applying the binary operator '+=' (line 78)
            result_iadd_17794 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 12), '+=', count_17792, int_17793)
            # Assigning a type to the variable 'count' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'count', result_iadd_17794)
            
            # SSA branch for the else part of an if statement (line 77)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'count' (line 80)
            count_17795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'count')
            int_17796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'int')
            # Applying the binary operator '+=' (line 80)
            result_iadd_17797 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 12), '+=', count_17795, int_17796)
            # Assigning a type to the variable 'count' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'count', result_iadd_17797)
            
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'count' (line 81)
    count_17798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'count')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', count_17798)
    
    # ################# End of 'header_quopri_len(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'header_quopri_len' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_17799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17799)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'header_quopri_len'
    return stypy_return_type_17799

# Assigning a type to the variable 'header_quopri_len' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'header_quopri_len', header_quopri_len)

@norecursion
def body_quopri_len(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'body_quopri_len'
    module_type_store = module_type_store.open_function_context('body_quopri_len', 84, 0, False)
    
    # Passed parameters checking function
    body_quopri_len.stypy_localization = localization
    body_quopri_len.stypy_type_of_self = None
    body_quopri_len.stypy_type_store = module_type_store
    body_quopri_len.stypy_function_name = 'body_quopri_len'
    body_quopri_len.stypy_param_names_list = ['str']
    body_quopri_len.stypy_varargs_param_name = None
    body_quopri_len.stypy_kwargs_param_name = None
    body_quopri_len.stypy_call_defaults = defaults
    body_quopri_len.stypy_call_varargs = varargs
    body_quopri_len.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'body_quopri_len', ['str'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'body_quopri_len', localization, ['str'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'body_quopri_len(...)' code ##################

    str_17800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'str', 'Return the length of str when it is encoded with body quopri.')
    
    # Assigning a Num to a Name (line 86):
    int_17801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 12), 'int')
    # Assigning a type to the variable 'count' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'count', int_17801)
    
    # Getting the type of 'str' (line 87)
    str_17802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'str')
    # Assigning a type to the variable 'str_17802' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'str_17802', str_17802)
    # Testing if the for loop is going to be iterated (line 87)
    # Testing the type of a for loop iterable (line 87)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 4), str_17802)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 4), str_17802):
        # Getting the type of the for loop variable (line 87)
        for_loop_var_17803 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 4), str_17802)
        # Assigning a type to the variable 'c' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'c', for_loop_var_17803)
        # SSA begins for a for statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to match(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'c' (line 88)
        c_17806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'c', False)
        # Processing the call keyword arguments (line 88)
        kwargs_17807 = {}
        # Getting the type of 'bqre' (line 88)
        bqre_17804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'bqre', False)
        # Obtaining the member 'match' of a type (line 88)
        match_17805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 11), bqre_17804, 'match')
        # Calling match(args, kwargs) (line 88)
        match_call_result_17808 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), match_17805, *[c_17806], **kwargs_17807)
        
        # Testing if the type of an if condition is none (line 88)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 8), match_call_result_17808):
            
            # Getting the type of 'count' (line 91)
            count_17813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'count')
            int_17814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'int')
            # Applying the binary operator '+=' (line 91)
            result_iadd_17815 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 12), '+=', count_17813, int_17814)
            # Assigning a type to the variable 'count' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'count', result_iadd_17815)
            
        else:
            
            # Testing the type of an if condition (line 88)
            if_condition_17809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 8), match_call_result_17808)
            # Assigning a type to the variable 'if_condition_17809' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'if_condition_17809', if_condition_17809)
            # SSA begins for if statement (line 88)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'count' (line 89)
            count_17810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'count')
            int_17811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'int')
            # Applying the binary operator '+=' (line 89)
            result_iadd_17812 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 12), '+=', count_17810, int_17811)
            # Assigning a type to the variable 'count' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'count', result_iadd_17812)
            
            # SSA branch for the else part of an if statement (line 88)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'count' (line 91)
            count_17813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'count')
            int_17814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'int')
            # Applying the binary operator '+=' (line 91)
            result_iadd_17815 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 12), '+=', count_17813, int_17814)
            # Assigning a type to the variable 'count' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'count', result_iadd_17815)
            
            # SSA join for if statement (line 88)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'count' (line 92)
    count_17816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'count')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', count_17816)
    
    # ################# End of 'body_quopri_len(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'body_quopri_len' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_17817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17817)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'body_quopri_len'
    return stypy_return_type_17817

# Assigning a type to the variable 'body_quopri_len' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'body_quopri_len', body_quopri_len)

@norecursion
def _max_append(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_17818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 36), 'str', '')
    defaults = [str_17818]
    # Create a new context for function '_max_append'
    module_type_store = module_type_store.open_function_context('_max_append', 95, 0, False)
    
    # Passed parameters checking function
    _max_append.stypy_localization = localization
    _max_append.stypy_type_of_self = None
    _max_append.stypy_type_store = module_type_store
    _max_append.stypy_function_name = '_max_append'
    _max_append.stypy_param_names_list = ['L', 's', 'maxlen', 'extra']
    _max_append.stypy_varargs_param_name = None
    _max_append.stypy_kwargs_param_name = None
    _max_append.stypy_call_defaults = defaults
    _max_append.stypy_call_varargs = varargs
    _max_append.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_max_append', ['L', 's', 'maxlen', 'extra'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_max_append', localization, ['L', 's', 'maxlen', 'extra'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_max_append(...)' code ##################

    
    # Getting the type of 'L' (line 96)
    L_17819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'L')
    # Applying the 'not' unary operator (line 96)
    result_not__17820 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), 'not', L_17819)
    
    # Testing if the type of an if condition is none (line 96)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 4), result_not__17820):
        
        
        # Call to len(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Obtaining the type of the subscript
        int_17831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 15), 'int')
        # Getting the type of 'L' (line 98)
        L_17832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'L', False)
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___17833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 13), L_17832, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_17834 = invoke(stypy.reporting.localization.Localization(__file__, 98, 13), getitem___17833, int_17831)
        
        # Processing the call keyword arguments (line 98)
        kwargs_17835 = {}
        # Getting the type of 'len' (line 98)
        len_17830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'len', False)
        # Calling len(args, kwargs) (line 98)
        len_call_result_17836 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), len_17830, *[subscript_call_result_17834], **kwargs_17835)
        
        
        # Call to len(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 's' (line 98)
        s_17838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 's', False)
        # Processing the call keyword arguments (line 98)
        kwargs_17839 = {}
        # Getting the type of 'len' (line 98)
        len_17837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'len', False)
        # Calling len(args, kwargs) (line 98)
        len_call_result_17840 = invoke(stypy.reporting.localization.Localization(__file__, 98, 22), len_17837, *[s_17838], **kwargs_17839)
        
        # Applying the binary operator '+' (line 98)
        result_add_17841 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 9), '+', len_call_result_17836, len_call_result_17840)
        
        # Getting the type of 'maxlen' (line 98)
        maxlen_17842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 32), 'maxlen')
        # Applying the binary operator '<=' (line 98)
        result_le_17843 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 9), '<=', result_add_17841, maxlen_17842)
        
        # Testing if the type of an if condition is none (line 98)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 98, 9), result_le_17843):
            
            # Call to append(...): (line 101)
            # Processing the call arguments (line 101)
            
            # Call to lstrip(...): (line 101)
            # Processing the call keyword arguments (line 101)
            kwargs_17860 = {}
            # Getting the type of 's' (line 101)
            s_17858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 's', False)
            # Obtaining the member 'lstrip' of a type (line 101)
            lstrip_17859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), s_17858, 'lstrip')
            # Calling lstrip(args, kwargs) (line 101)
            lstrip_call_result_17861 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), lstrip_17859, *[], **kwargs_17860)
            
            # Processing the call keyword arguments (line 101)
            kwargs_17862 = {}
            # Getting the type of 'L' (line 101)
            L_17856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'L', False)
            # Obtaining the member 'append' of a type (line 101)
            append_17857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), L_17856, 'append')
            # Calling append(args, kwargs) (line 101)
            append_call_result_17863 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), append_17857, *[lstrip_call_result_17861], **kwargs_17862)
            
        else:
            
            # Testing the type of an if condition (line 98)
            if_condition_17844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 9), result_le_17843)
            # Assigning a type to the variable 'if_condition_17844' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'if_condition_17844', if_condition_17844)
            # SSA begins for if statement (line 98)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'L' (line 99)
            L_17845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'L')
            
            # Obtaining the type of the subscript
            int_17846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 10), 'int')
            # Getting the type of 'L' (line 99)
            L_17847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'L')
            # Obtaining the member '__getitem__' of a type (line 99)
            getitem___17848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), L_17847, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 99)
            subscript_call_result_17849 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), getitem___17848, int_17846)
            
            # Getting the type of 'extra' (line 99)
            extra_17850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'extra')
            # Getting the type of 's' (line 99)
            s_17851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 's')
            # Applying the binary operator '+' (line 99)
            result_add_17852 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 17), '+', extra_17850, s_17851)
            
            # Applying the binary operator '+=' (line 99)
            result_iadd_17853 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 8), '+=', subscript_call_result_17849, result_add_17852)
            # Getting the type of 'L' (line 99)
            L_17854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'L')
            int_17855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 10), 'int')
            # Storing an element on a container (line 99)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 8), L_17854, (int_17855, result_iadd_17853))
            
            # SSA branch for the else part of an if statement (line 98)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 101)
            # Processing the call arguments (line 101)
            
            # Call to lstrip(...): (line 101)
            # Processing the call keyword arguments (line 101)
            kwargs_17860 = {}
            # Getting the type of 's' (line 101)
            s_17858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 's', False)
            # Obtaining the member 'lstrip' of a type (line 101)
            lstrip_17859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), s_17858, 'lstrip')
            # Calling lstrip(args, kwargs) (line 101)
            lstrip_call_result_17861 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), lstrip_17859, *[], **kwargs_17860)
            
            # Processing the call keyword arguments (line 101)
            kwargs_17862 = {}
            # Getting the type of 'L' (line 101)
            L_17856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'L', False)
            # Obtaining the member 'append' of a type (line 101)
            append_17857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), L_17856, 'append')
            # Calling append(args, kwargs) (line 101)
            append_call_result_17863 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), append_17857, *[lstrip_call_result_17861], **kwargs_17862)
            
            # SSA join for if statement (line 98)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 96)
        if_condition_17821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_not__17820)
        # Assigning a type to the variable 'if_condition_17821' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_17821', if_condition_17821)
        # SSA begins for if statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to lstrip(...): (line 97)
        # Processing the call keyword arguments (line 97)
        kwargs_17826 = {}
        # Getting the type of 's' (line 97)
        s_17824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 's', False)
        # Obtaining the member 'lstrip' of a type (line 97)
        lstrip_17825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 17), s_17824, 'lstrip')
        # Calling lstrip(args, kwargs) (line 97)
        lstrip_call_result_17827 = invoke(stypy.reporting.localization.Localization(__file__, 97, 17), lstrip_17825, *[], **kwargs_17826)
        
        # Processing the call keyword arguments (line 97)
        kwargs_17828 = {}
        # Getting the type of 'L' (line 97)
        L_17822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'L', False)
        # Obtaining the member 'append' of a type (line 97)
        append_17823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), L_17822, 'append')
        # Calling append(args, kwargs) (line 97)
        append_call_result_17829 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), append_17823, *[lstrip_call_result_17827], **kwargs_17828)
        
        # SSA branch for the else part of an if statement (line 96)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to len(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Obtaining the type of the subscript
        int_17831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 15), 'int')
        # Getting the type of 'L' (line 98)
        L_17832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'L', False)
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___17833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 13), L_17832, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_17834 = invoke(stypy.reporting.localization.Localization(__file__, 98, 13), getitem___17833, int_17831)
        
        # Processing the call keyword arguments (line 98)
        kwargs_17835 = {}
        # Getting the type of 'len' (line 98)
        len_17830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'len', False)
        # Calling len(args, kwargs) (line 98)
        len_call_result_17836 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), len_17830, *[subscript_call_result_17834], **kwargs_17835)
        
        
        # Call to len(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 's' (line 98)
        s_17838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 's', False)
        # Processing the call keyword arguments (line 98)
        kwargs_17839 = {}
        # Getting the type of 'len' (line 98)
        len_17837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'len', False)
        # Calling len(args, kwargs) (line 98)
        len_call_result_17840 = invoke(stypy.reporting.localization.Localization(__file__, 98, 22), len_17837, *[s_17838], **kwargs_17839)
        
        # Applying the binary operator '+' (line 98)
        result_add_17841 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 9), '+', len_call_result_17836, len_call_result_17840)
        
        # Getting the type of 'maxlen' (line 98)
        maxlen_17842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 32), 'maxlen')
        # Applying the binary operator '<=' (line 98)
        result_le_17843 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 9), '<=', result_add_17841, maxlen_17842)
        
        # Testing if the type of an if condition is none (line 98)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 98, 9), result_le_17843):
            
            # Call to append(...): (line 101)
            # Processing the call arguments (line 101)
            
            # Call to lstrip(...): (line 101)
            # Processing the call keyword arguments (line 101)
            kwargs_17860 = {}
            # Getting the type of 's' (line 101)
            s_17858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 's', False)
            # Obtaining the member 'lstrip' of a type (line 101)
            lstrip_17859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), s_17858, 'lstrip')
            # Calling lstrip(args, kwargs) (line 101)
            lstrip_call_result_17861 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), lstrip_17859, *[], **kwargs_17860)
            
            # Processing the call keyword arguments (line 101)
            kwargs_17862 = {}
            # Getting the type of 'L' (line 101)
            L_17856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'L', False)
            # Obtaining the member 'append' of a type (line 101)
            append_17857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), L_17856, 'append')
            # Calling append(args, kwargs) (line 101)
            append_call_result_17863 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), append_17857, *[lstrip_call_result_17861], **kwargs_17862)
            
        else:
            
            # Testing the type of an if condition (line 98)
            if_condition_17844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 9), result_le_17843)
            # Assigning a type to the variable 'if_condition_17844' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'if_condition_17844', if_condition_17844)
            # SSA begins for if statement (line 98)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'L' (line 99)
            L_17845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'L')
            
            # Obtaining the type of the subscript
            int_17846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 10), 'int')
            # Getting the type of 'L' (line 99)
            L_17847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'L')
            # Obtaining the member '__getitem__' of a type (line 99)
            getitem___17848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), L_17847, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 99)
            subscript_call_result_17849 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), getitem___17848, int_17846)
            
            # Getting the type of 'extra' (line 99)
            extra_17850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'extra')
            # Getting the type of 's' (line 99)
            s_17851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 's')
            # Applying the binary operator '+' (line 99)
            result_add_17852 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 17), '+', extra_17850, s_17851)
            
            # Applying the binary operator '+=' (line 99)
            result_iadd_17853 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 8), '+=', subscript_call_result_17849, result_add_17852)
            # Getting the type of 'L' (line 99)
            L_17854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'L')
            int_17855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 10), 'int')
            # Storing an element on a container (line 99)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 8), L_17854, (int_17855, result_iadd_17853))
            
            # SSA branch for the else part of an if statement (line 98)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 101)
            # Processing the call arguments (line 101)
            
            # Call to lstrip(...): (line 101)
            # Processing the call keyword arguments (line 101)
            kwargs_17860 = {}
            # Getting the type of 's' (line 101)
            s_17858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 's', False)
            # Obtaining the member 'lstrip' of a type (line 101)
            lstrip_17859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), s_17858, 'lstrip')
            # Calling lstrip(args, kwargs) (line 101)
            lstrip_call_result_17861 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), lstrip_17859, *[], **kwargs_17860)
            
            # Processing the call keyword arguments (line 101)
            kwargs_17862 = {}
            # Getting the type of 'L' (line 101)
            L_17856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'L', False)
            # Obtaining the member 'append' of a type (line 101)
            append_17857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), L_17856, 'append')
            # Calling append(args, kwargs) (line 101)
            append_call_result_17863 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), append_17857, *[lstrip_call_result_17861], **kwargs_17862)
            
            # SSA join for if statement (line 98)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of '_max_append(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_max_append' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_17864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17864)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_max_append'
    return stypy_return_type_17864

# Assigning a type to the variable '_max_append' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), '_max_append', _max_append)

@norecursion
def unquote(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unquote'
    module_type_store = module_type_store.open_function_context('unquote', 104, 0, False)
    
    # Passed parameters checking function
    unquote.stypy_localization = localization
    unquote.stypy_type_of_self = None
    unquote.stypy_type_store = module_type_store
    unquote.stypy_function_name = 'unquote'
    unquote.stypy_param_names_list = ['s']
    unquote.stypy_varargs_param_name = None
    unquote.stypy_kwargs_param_name = None
    unquote.stypy_call_defaults = defaults
    unquote.stypy_call_varargs = varargs
    unquote.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unquote', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unquote', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unquote(...)' code ##################

    str_17865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 4), 'str', 'Turn a string in the form =AB to the ASCII character with value 0xab')
    
    # Call to chr(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Call to int(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Obtaining the type of the subscript
    int_17868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 21), 'int')
    int_17869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 23), 'int')
    slice_17870 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 106, 19), int_17868, int_17869, None)
    # Getting the type of 's' (line 106)
    s_17871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 's', False)
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___17872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 19), s_17871, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_17873 = invoke(stypy.reporting.localization.Localization(__file__, 106, 19), getitem___17872, slice_17870)
    
    int_17874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 27), 'int')
    # Processing the call keyword arguments (line 106)
    kwargs_17875 = {}
    # Getting the type of 'int' (line 106)
    int_17867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'int', False)
    # Calling int(args, kwargs) (line 106)
    int_call_result_17876 = invoke(stypy.reporting.localization.Localization(__file__, 106, 15), int_17867, *[subscript_call_result_17873, int_17874], **kwargs_17875)
    
    # Processing the call keyword arguments (line 106)
    kwargs_17877 = {}
    # Getting the type of 'chr' (line 106)
    chr_17866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'chr', False)
    # Calling chr(args, kwargs) (line 106)
    chr_call_result_17878 = invoke(stypy.reporting.localization.Localization(__file__, 106, 11), chr_17866, *[int_call_result_17876], **kwargs_17877)
    
    # Assigning a type to the variable 'stypy_return_type' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type', chr_call_result_17878)
    
    # ################# End of 'unquote(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unquote' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_17879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17879)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unquote'
    return stypy_return_type_17879

# Assigning a type to the variable 'unquote' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'unquote', unquote)

@norecursion
def quote(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'quote'
    module_type_store = module_type_store.open_function_context('quote', 109, 0, False)
    
    # Passed parameters checking function
    quote.stypy_localization = localization
    quote.stypy_type_of_self = None
    quote.stypy_type_store = module_type_store
    quote.stypy_function_name = 'quote'
    quote.stypy_param_names_list = ['c']
    quote.stypy_varargs_param_name = None
    quote.stypy_kwargs_param_name = None
    quote.stypy_call_defaults = defaults
    quote.stypy_call_varargs = varargs
    quote.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'quote', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'quote', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'quote(...)' code ##################

    str_17880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 11), 'str', '=%02X')
    
    # Call to ord(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'c' (line 110)
    c_17882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'c', False)
    # Processing the call keyword arguments (line 110)
    kwargs_17883 = {}
    # Getting the type of 'ord' (line 110)
    ord_17881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'ord', False)
    # Calling ord(args, kwargs) (line 110)
    ord_call_result_17884 = invoke(stypy.reporting.localization.Localization(__file__, 110, 21), ord_17881, *[c_17882], **kwargs_17883)
    
    # Applying the binary operator '%' (line 110)
    result_mod_17885 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 11), '%', str_17880, ord_call_result_17884)
    
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', result_mod_17885)
    
    # ################# End of 'quote(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'quote' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_17886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17886)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'quote'
    return stypy_return_type_17886

# Assigning a type to the variable 'quote' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'quote', quote)

@norecursion
def header_encode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_17887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 34), 'str', 'iso-8859-1')
    # Getting the type of 'False' (line 114)
    False_17888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 58), 'False')
    int_17889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 29), 'int')
    # Getting the type of 'NL' (line 115)
    NL_17890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'NL')
    defaults = [str_17887, False_17888, int_17889, NL_17890]
    # Create a new context for function 'header_encode'
    module_type_store = module_type_store.open_function_context('header_encode', 114, 0, False)
    
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

    str_17891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, (-1)), 'str', 'Encode a single header line with quoted-printable (like) encoding.\n\n    Defined in RFC 2045, this `Q\' encoding is similar to quoted-printable, but\n    used specifically for email header fields to allow charsets with mostly 7\n    bit characters (and some 8 bit) to remain more or less readable in non-RFC\n    2045 aware mail clients.\n\n    charset names the character set to use to encode the header.  It defaults\n    to iso-8859-1.\n\n    The resulting string will be in the form:\n\n    "=?charset?q?I_f=E2rt_in_your_g=E8n=E8ral_dire=E7tion?\\n\n      =?charset?q?Silly_=C8nglish_Kn=EEghts?="\n\n    with each line wrapped safely at, at most, maxlinelen characters (defaults\n    to 76 characters).  If maxlinelen is None, the entire string is encoded in\n    one chunk with no splitting.\n\n    End-of-line characters (\\r, \\n, \\r\\n) will be automatically converted\n    to the canonical email line separator \\r\\n unless the keep_eols\n    parameter is True (the default is False).\n\n    Each line of the header will be terminated in the value of eol, which\n    defaults to "\\n".  Set this to "\\r\\n" if you are using the result of\n    this function directly in email.\n    ')
    
    # Getting the type of 'header' (line 144)
    header_17892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'header')
    # Applying the 'not' unary operator (line 144)
    result_not__17893 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 7), 'not', header_17892)
    
    # Testing if the type of an if condition is none (line 144)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 144, 4), result_not__17893):
        pass
    else:
        
        # Testing the type of an if condition (line 144)
        if_condition_17894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 4), result_not__17893)
        # Assigning a type to the variable 'if_condition_17894' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'if_condition_17894', if_condition_17894)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'header' (line 145)
        header_17895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'header')
        # Assigning a type to the variable 'stypy_return_type' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'stypy_return_type', header_17895)
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'keep_eols' (line 147)
    keep_eols_17896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'keep_eols')
    # Applying the 'not' unary operator (line 147)
    result_not__17897 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 7), 'not', keep_eols_17896)
    
    # Testing if the type of an if condition is none (line 147)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 147, 4), result_not__17897):
        pass
    else:
        
        # Testing the type of an if condition (line 147)
        if_condition_17898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 4), result_not__17897)
        # Assigning a type to the variable 'if_condition_17898' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'if_condition_17898', if_condition_17898)
        # SSA begins for if statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 148):
        
        # Call to fix_eols(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'header' (line 148)
        header_17900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'header', False)
        # Processing the call keyword arguments (line 148)
        kwargs_17901 = {}
        # Getting the type of 'fix_eols' (line 148)
        fix_eols_17899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 17), 'fix_eols', False)
        # Calling fix_eols(args, kwargs) (line 148)
        fix_eols_call_result_17902 = invoke(stypy.reporting.localization.Localization(__file__, 148, 17), fix_eols_17899, *[header_17900], **kwargs_17901)
        
        # Assigning a type to the variable 'header' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'header', fix_eols_call_result_17902)
        # SSA join for if statement (line 147)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 152):
    
    # Obtaining an instance of the builtin type 'list' (line 152)
    list_17903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 152)
    
    # Assigning a type to the variable 'quoted' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'quoted', list_17903)
    
    # Type idiom detected: calculating its left and rigth part (line 153)
    # Getting the type of 'maxlinelen' (line 153)
    maxlinelen_17904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 7), 'maxlinelen')
    # Getting the type of 'None' (line 153)
    None_17905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'None')
    
    (may_be_17906, more_types_in_union_17907) = may_be_none(maxlinelen_17904, None_17905)

    if may_be_17906:

        if more_types_in_union_17907:
            # Runtime conditional SSA (line 153)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 155):
        int_17908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 22), 'int')
        # Assigning a type to the variable 'max_encoded' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'max_encoded', int_17908)

        if more_types_in_union_17907:
            # Runtime conditional SSA for else branch (line 153)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_17906) or more_types_in_union_17907):
        
        # Assigning a BinOp to a Name (line 157):
        # Getting the type of 'maxlinelen' (line 157)
        maxlinelen_17909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'maxlinelen')
        
        # Call to len(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'charset' (line 157)
        charset_17911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 39), 'charset', False)
        # Processing the call keyword arguments (line 157)
        kwargs_17912 = {}
        # Getting the type of 'len' (line 157)
        len_17910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 35), 'len', False)
        # Calling len(args, kwargs) (line 157)
        len_call_result_17913 = invoke(stypy.reporting.localization.Localization(__file__, 157, 35), len_17910, *[charset_17911], **kwargs_17912)
        
        # Applying the binary operator '-' (line 157)
        result_sub_17914 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 22), '-', maxlinelen_17909, len_call_result_17913)
        
        # Getting the type of 'MISC_LEN' (line 157)
        MISC_LEN_17915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 50), 'MISC_LEN')
        # Applying the binary operator '-' (line 157)
        result_sub_17916 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 48), '-', result_sub_17914, MISC_LEN_17915)
        
        int_17917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 61), 'int')
        # Applying the binary operator '-' (line 157)
        result_sub_17918 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 59), '-', result_sub_17916, int_17917)
        
        # Assigning a type to the variable 'max_encoded' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'max_encoded', result_sub_17918)

        if (may_be_17906 and more_types_in_union_17907):
            # SSA join for if statement (line 153)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'header' (line 159)
    header_17919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'header')
    # Assigning a type to the variable 'header_17919' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'header_17919', header_17919)
    # Testing if the for loop is going to be iterated (line 159)
    # Testing the type of a for loop iterable (line 159)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 4), header_17919)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 159, 4), header_17919):
        # Getting the type of the for loop variable (line 159)
        for_loop_var_17920 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 4), header_17919)
        # Assigning a type to the variable 'c' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'c', for_loop_var_17920)
        # SSA begins for a for statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'c' (line 161)
        c_17921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'c')
        str_17922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'str', ' ')
        # Applying the binary operator '==' (line 161)
        result_eq_17923 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), '==', c_17921, str_17922)
        
        # Testing if the type of an if condition is none (line 161)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 8), result_eq_17923):
            
            
            # Call to match(...): (line 164)
            # Processing the call arguments (line 164)
            # Getting the type of 'c' (line 164)
            c_17933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'c', False)
            # Processing the call keyword arguments (line 164)
            kwargs_17934 = {}
            # Getting the type of 'hqre' (line 164)
            hqre_17931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), 'hqre', False)
            # Obtaining the member 'match' of a type (line 164)
            match_17932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 17), hqre_17931, 'match')
            # Calling match(args, kwargs) (line 164)
            match_call_result_17935 = invoke(stypy.reporting.localization.Localization(__file__, 164, 17), match_17932, *[c_17933], **kwargs_17934)
            
            # Applying the 'not' unary operator (line 164)
            result_not__17936 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 13), 'not', match_call_result_17935)
            
            # Testing if the type of an if condition is none (line 164)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 164, 13), result_not__17936):
                
                # Call to _max_append(...): (line 168)
                # Processing the call arguments (line 168)
                # Getting the type of 'quoted' (line 168)
                quoted_17945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'quoted', False)
                str_17946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 32), 'str', '=%02X')
                
                # Call to ord(...): (line 168)
                # Processing the call arguments (line 168)
                # Getting the type of 'c' (line 168)
                c_17948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 46), 'c', False)
                # Processing the call keyword arguments (line 168)
                kwargs_17949 = {}
                # Getting the type of 'ord' (line 168)
                ord_17947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 42), 'ord', False)
                # Calling ord(args, kwargs) (line 168)
                ord_call_result_17950 = invoke(stypy.reporting.localization.Localization(__file__, 168, 42), ord_17947, *[c_17948], **kwargs_17949)
                
                # Applying the binary operator '%' (line 168)
                result_mod_17951 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 32), '%', str_17946, ord_call_result_17950)
                
                # Getting the type of 'max_encoded' (line 168)
                max_encoded_17952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 50), 'max_encoded', False)
                # Processing the call keyword arguments (line 168)
                kwargs_17953 = {}
                # Getting the type of '_max_append' (line 168)
                _max_append_17944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), '_max_append', False)
                # Calling _max_append(args, kwargs) (line 168)
                _max_append_call_result_17954 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), _max_append_17944, *[quoted_17945, result_mod_17951, max_encoded_17952], **kwargs_17953)
                
            else:
                
                # Testing the type of an if condition (line 164)
                if_condition_17937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 13), result_not__17936)
                # Assigning a type to the variable 'if_condition_17937' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'if_condition_17937', if_condition_17937)
                # SSA begins for if statement (line 164)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to _max_append(...): (line 165)
                # Processing the call arguments (line 165)
                # Getting the type of 'quoted' (line 165)
                quoted_17939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'quoted', False)
                # Getting the type of 'c' (line 165)
                c_17940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'c', False)
                # Getting the type of 'max_encoded' (line 165)
                max_encoded_17941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 35), 'max_encoded', False)
                # Processing the call keyword arguments (line 165)
                kwargs_17942 = {}
                # Getting the type of '_max_append' (line 165)
                _max_append_17938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), '_max_append', False)
                # Calling _max_append(args, kwargs) (line 165)
                _max_append_call_result_17943 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), _max_append_17938, *[quoted_17939, c_17940, max_encoded_17941], **kwargs_17942)
                
                # SSA branch for the else part of an if statement (line 164)
                module_type_store.open_ssa_branch('else')
                
                # Call to _max_append(...): (line 168)
                # Processing the call arguments (line 168)
                # Getting the type of 'quoted' (line 168)
                quoted_17945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'quoted', False)
                str_17946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 32), 'str', '=%02X')
                
                # Call to ord(...): (line 168)
                # Processing the call arguments (line 168)
                # Getting the type of 'c' (line 168)
                c_17948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 46), 'c', False)
                # Processing the call keyword arguments (line 168)
                kwargs_17949 = {}
                # Getting the type of 'ord' (line 168)
                ord_17947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 42), 'ord', False)
                # Calling ord(args, kwargs) (line 168)
                ord_call_result_17950 = invoke(stypy.reporting.localization.Localization(__file__, 168, 42), ord_17947, *[c_17948], **kwargs_17949)
                
                # Applying the binary operator '%' (line 168)
                result_mod_17951 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 32), '%', str_17946, ord_call_result_17950)
                
                # Getting the type of 'max_encoded' (line 168)
                max_encoded_17952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 50), 'max_encoded', False)
                # Processing the call keyword arguments (line 168)
                kwargs_17953 = {}
                # Getting the type of '_max_append' (line 168)
                _max_append_17944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), '_max_append', False)
                # Calling _max_append(args, kwargs) (line 168)
                _max_append_call_result_17954 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), _max_append_17944, *[quoted_17945, result_mod_17951, max_encoded_17952], **kwargs_17953)
                
                # SSA join for if statement (line 164)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 161)
            if_condition_17924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), result_eq_17923)
            # Assigning a type to the variable 'if_condition_17924' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_17924', if_condition_17924)
            # SSA begins for if statement (line 161)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _max_append(...): (line 162)
            # Processing the call arguments (line 162)
            # Getting the type of 'quoted' (line 162)
            quoted_17926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'quoted', False)
            str_17927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 32), 'str', '_')
            # Getting the type of 'max_encoded' (line 162)
            max_encoded_17928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'max_encoded', False)
            # Processing the call keyword arguments (line 162)
            kwargs_17929 = {}
            # Getting the type of '_max_append' (line 162)
            _max_append_17925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), '_max_append', False)
            # Calling _max_append(args, kwargs) (line 162)
            _max_append_call_result_17930 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), _max_append_17925, *[quoted_17926, str_17927, max_encoded_17928], **kwargs_17929)
            
            # SSA branch for the else part of an if statement (line 161)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to match(...): (line 164)
            # Processing the call arguments (line 164)
            # Getting the type of 'c' (line 164)
            c_17933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'c', False)
            # Processing the call keyword arguments (line 164)
            kwargs_17934 = {}
            # Getting the type of 'hqre' (line 164)
            hqre_17931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), 'hqre', False)
            # Obtaining the member 'match' of a type (line 164)
            match_17932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 17), hqre_17931, 'match')
            # Calling match(args, kwargs) (line 164)
            match_call_result_17935 = invoke(stypy.reporting.localization.Localization(__file__, 164, 17), match_17932, *[c_17933], **kwargs_17934)
            
            # Applying the 'not' unary operator (line 164)
            result_not__17936 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 13), 'not', match_call_result_17935)
            
            # Testing if the type of an if condition is none (line 164)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 164, 13), result_not__17936):
                
                # Call to _max_append(...): (line 168)
                # Processing the call arguments (line 168)
                # Getting the type of 'quoted' (line 168)
                quoted_17945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'quoted', False)
                str_17946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 32), 'str', '=%02X')
                
                # Call to ord(...): (line 168)
                # Processing the call arguments (line 168)
                # Getting the type of 'c' (line 168)
                c_17948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 46), 'c', False)
                # Processing the call keyword arguments (line 168)
                kwargs_17949 = {}
                # Getting the type of 'ord' (line 168)
                ord_17947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 42), 'ord', False)
                # Calling ord(args, kwargs) (line 168)
                ord_call_result_17950 = invoke(stypy.reporting.localization.Localization(__file__, 168, 42), ord_17947, *[c_17948], **kwargs_17949)
                
                # Applying the binary operator '%' (line 168)
                result_mod_17951 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 32), '%', str_17946, ord_call_result_17950)
                
                # Getting the type of 'max_encoded' (line 168)
                max_encoded_17952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 50), 'max_encoded', False)
                # Processing the call keyword arguments (line 168)
                kwargs_17953 = {}
                # Getting the type of '_max_append' (line 168)
                _max_append_17944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), '_max_append', False)
                # Calling _max_append(args, kwargs) (line 168)
                _max_append_call_result_17954 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), _max_append_17944, *[quoted_17945, result_mod_17951, max_encoded_17952], **kwargs_17953)
                
            else:
                
                # Testing the type of an if condition (line 164)
                if_condition_17937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 13), result_not__17936)
                # Assigning a type to the variable 'if_condition_17937' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'if_condition_17937', if_condition_17937)
                # SSA begins for if statement (line 164)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to _max_append(...): (line 165)
                # Processing the call arguments (line 165)
                # Getting the type of 'quoted' (line 165)
                quoted_17939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'quoted', False)
                # Getting the type of 'c' (line 165)
                c_17940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'c', False)
                # Getting the type of 'max_encoded' (line 165)
                max_encoded_17941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 35), 'max_encoded', False)
                # Processing the call keyword arguments (line 165)
                kwargs_17942 = {}
                # Getting the type of '_max_append' (line 165)
                _max_append_17938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), '_max_append', False)
                # Calling _max_append(args, kwargs) (line 165)
                _max_append_call_result_17943 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), _max_append_17938, *[quoted_17939, c_17940, max_encoded_17941], **kwargs_17942)
                
                # SSA branch for the else part of an if statement (line 164)
                module_type_store.open_ssa_branch('else')
                
                # Call to _max_append(...): (line 168)
                # Processing the call arguments (line 168)
                # Getting the type of 'quoted' (line 168)
                quoted_17945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'quoted', False)
                str_17946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 32), 'str', '=%02X')
                
                # Call to ord(...): (line 168)
                # Processing the call arguments (line 168)
                # Getting the type of 'c' (line 168)
                c_17948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 46), 'c', False)
                # Processing the call keyword arguments (line 168)
                kwargs_17949 = {}
                # Getting the type of 'ord' (line 168)
                ord_17947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 42), 'ord', False)
                # Calling ord(args, kwargs) (line 168)
                ord_call_result_17950 = invoke(stypy.reporting.localization.Localization(__file__, 168, 42), ord_17947, *[c_17948], **kwargs_17949)
                
                # Applying the binary operator '%' (line 168)
                result_mod_17951 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 32), '%', str_17946, ord_call_result_17950)
                
                # Getting the type of 'max_encoded' (line 168)
                max_encoded_17952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 50), 'max_encoded', False)
                # Processing the call keyword arguments (line 168)
                kwargs_17953 = {}
                # Getting the type of '_max_append' (line 168)
                _max_append_17944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), '_max_append', False)
                # Calling _max_append(args, kwargs) (line 168)
                _max_append_call_result_17954 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), _max_append_17944, *[quoted_17945, result_mod_17951, max_encoded_17952], **kwargs_17953)
                
                # SSA join for if statement (line 164)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 161)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a BinOp to a Name (line 173):
    # Getting the type of 'eol' (line 173)
    eol_17955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'eol')
    str_17956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 19), 'str', ' ')
    # Applying the binary operator '+' (line 173)
    result_add_17957 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 13), '+', eol_17955, str_17956)
    
    # Assigning a type to the variable 'joiner' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'joiner', result_add_17957)
    
    # Call to join(...): (line 174)
    # Processing the call arguments (line 174)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'quoted' (line 174)
    quoted_17965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 68), 'quoted', False)
    comprehension_17966 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 24), quoted_17965)
    # Assigning a type to the variable 'line' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'line', comprehension_17966)
    str_17960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 24), 'str', '=?%s?q?%s?=')
    
    # Obtaining an instance of the builtin type 'tuple' (line 174)
    tuple_17961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 174)
    # Adding element type (line 174)
    # Getting the type of 'charset' (line 174)
    charset_17962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 41), 'charset', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 41), tuple_17961, charset_17962)
    # Adding element type (line 174)
    # Getting the type of 'line' (line 174)
    line_17963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 50), 'line', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 41), tuple_17961, line_17963)
    
    # Applying the binary operator '%' (line 174)
    result_mod_17964 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 24), '%', str_17960, tuple_17961)
    
    list_17967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 24), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 24), list_17967, result_mod_17964)
    # Processing the call keyword arguments (line 174)
    kwargs_17968 = {}
    # Getting the type of 'joiner' (line 174)
    joiner_17958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'joiner', False)
    # Obtaining the member 'join' of a type (line 174)
    join_17959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 11), joiner_17958, 'join')
    # Calling join(args, kwargs) (line 174)
    join_call_result_17969 = invoke(stypy.reporting.localization.Localization(__file__, 174, 11), join_17959, *[list_17967], **kwargs_17968)
    
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', join_call_result_17969)
    
    # ################# End of 'header_encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'header_encode' in the type store
    # Getting the type of 'stypy_return_type' (line 114)
    stypy_return_type_17970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17970)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'header_encode'
    return stypy_return_type_17970

# Assigning a type to the variable 'header_encode' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'header_encode', header_encode)

@norecursion
def encode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 178)
    False_17971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'False')
    int_17972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 42), 'int')
    # Getting the type of 'NL' (line 178)
    NL_17973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 50), 'NL')
    defaults = [False_17971, int_17972, NL_17973]
    # Create a new context for function 'encode'
    module_type_store = module_type_store.open_function_context('encode', 178, 0, False)
    
    # Passed parameters checking function
    encode.stypy_localization = localization
    encode.stypy_type_of_self = None
    encode.stypy_type_store = module_type_store
    encode.stypy_function_name = 'encode'
    encode.stypy_param_names_list = ['body', 'binary', 'maxlinelen', 'eol']
    encode.stypy_varargs_param_name = None
    encode.stypy_kwargs_param_name = None
    encode.stypy_call_defaults = defaults
    encode.stypy_call_varargs = varargs
    encode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode', ['body', 'binary', 'maxlinelen', 'eol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode', localization, ['body', 'binary', 'maxlinelen', 'eol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode(...)' code ##################

    str_17974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'str', 'Encode with quoted-printable, wrapping at maxlinelen characters.\n\n    If binary is False (the default), end-of-line characters will be converted\n    to the canonical email end-of-line sequence \\r\\n.  Otherwise they will\n    be left verbatim.\n\n    Each line of encoded text will end with eol, which defaults to "\\n".  Set\n    this to "\\r\\n" if you will be using the result of this function directly\n    in an email.\n\n    Each line will be wrapped at, at most, maxlinelen characters (defaults to\n    76 characters).  Long lines will have the `soft linefeed\' quoted-printable\n    character "=" appended to them, so the decoded text will be identical to\n    the original text.\n    ')
    
    # Getting the type of 'body' (line 194)
    body_17975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'body')
    # Applying the 'not' unary operator (line 194)
    result_not__17976 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 7), 'not', body_17975)
    
    # Testing if the type of an if condition is none (line 194)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 194, 4), result_not__17976):
        pass
    else:
        
        # Testing the type of an if condition (line 194)
        if_condition_17977 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 4), result_not__17976)
        # Assigning a type to the variable 'if_condition_17977' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'if_condition_17977', if_condition_17977)
        # SSA begins for if statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'body' (line 195)
        body_17978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'body')
        # Assigning a type to the variable 'stypy_return_type' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', body_17978)
        # SSA join for if statement (line 194)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'binary' (line 197)
    binary_17979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'binary')
    # Applying the 'not' unary operator (line 197)
    result_not__17980 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 7), 'not', binary_17979)
    
    # Testing if the type of an if condition is none (line 197)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 197, 4), result_not__17980):
        pass
    else:
        
        # Testing the type of an if condition (line 197)
        if_condition_17981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 4), result_not__17980)
        # Assigning a type to the variable 'if_condition_17981' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'if_condition_17981', if_condition_17981)
        # SSA begins for if statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 198):
        
        # Call to fix_eols(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'body' (line 198)
        body_17983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'body', False)
        # Processing the call keyword arguments (line 198)
        kwargs_17984 = {}
        # Getting the type of 'fix_eols' (line 198)
        fix_eols_17982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'fix_eols', False)
        # Calling fix_eols(args, kwargs) (line 198)
        fix_eols_call_result_17985 = invoke(stypy.reporting.localization.Localization(__file__, 198, 15), fix_eols_17982, *[body_17983], **kwargs_17984)
        
        # Assigning a type to the variable 'body' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'body', fix_eols_call_result_17985)
        # SSA join for if statement (line 197)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Str to a Name (line 203):
    str_17986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 19), 'str', '')
    # Assigning a type to the variable 'encoded_body' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'encoded_body', str_17986)
    
    # Assigning a Num to a Name (line 204):
    int_17987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 13), 'int')
    # Assigning a type to the variable 'lineno' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'lineno', int_17987)
    
    # Assigning a Call to a Name (line 207):
    
    # Call to splitlines(...): (line 207)
    # Processing the call arguments (line 207)
    int_17990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 28), 'int')
    # Processing the call keyword arguments (line 207)
    kwargs_17991 = {}
    # Getting the type of 'body' (line 207)
    body_17988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'body', False)
    # Obtaining the member 'splitlines' of a type (line 207)
    splitlines_17989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), body_17988, 'splitlines')
    # Calling splitlines(args, kwargs) (line 207)
    splitlines_call_result_17992 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), splitlines_17989, *[int_17990], **kwargs_17991)
    
    # Assigning a type to the variable 'lines' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'lines', splitlines_call_result_17992)
    
    # Getting the type of 'lines' (line 208)
    lines_17993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'lines')
    # Assigning a type to the variable 'lines_17993' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'lines_17993', lines_17993)
    # Testing if the for loop is going to be iterated (line 208)
    # Testing the type of a for loop iterable (line 208)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 208, 4), lines_17993)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 208, 4), lines_17993):
        # Getting the type of the for loop variable (line 208)
        for_loop_var_17994 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 208, 4), lines_17993)
        # Assigning a type to the variable 'line' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'line', for_loop_var_17994)
        # SSA begins for a for statement (line 208)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to endswith(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'CRLF' (line 210)
        CRLF_17997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 25), 'CRLF', False)
        # Processing the call keyword arguments (line 210)
        kwargs_17998 = {}
        # Getting the type of 'line' (line 210)
        line_17995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'line', False)
        # Obtaining the member 'endswith' of a type (line 210)
        endswith_17996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 11), line_17995, 'endswith')
        # Calling endswith(args, kwargs) (line 210)
        endswith_call_result_17999 = invoke(stypy.reporting.localization.Localization(__file__, 210, 11), endswith_17996, *[CRLF_17997], **kwargs_17998)
        
        # Testing if the type of an if condition is none (line 210)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 210, 8), endswith_call_result_17999):
            
            
            # Obtaining the type of the subscript
            int_18006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 18), 'int')
            # Getting the type of 'line' (line 212)
            line_18007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'line')
            # Obtaining the member '__getitem__' of a type (line 212)
            getitem___18008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 13), line_18007, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 212)
            subscript_call_result_18009 = invoke(stypy.reporting.localization.Localization(__file__, 212, 13), getitem___18008, int_18006)
            
            # Getting the type of 'CRLF' (line 212)
            CRLF_18010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 25), 'CRLF')
            # Applying the binary operator 'in' (line 212)
            result_contains_18011 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 13), 'in', subscript_call_result_18009, CRLF_18010)
            
            # Testing if the type of an if condition is none (line 212)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 212, 13), result_contains_18011):
                pass
            else:
                
                # Testing the type of an if condition (line 212)
                if_condition_18012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 13), result_contains_18011)
                # Assigning a type to the variable 'if_condition_18012' (line 212)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'if_condition_18012', if_condition_18012)
                # SSA begins for if statement (line 212)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 213):
                
                # Obtaining the type of the subscript
                int_18013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 25), 'int')
                slice_18014 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 213, 19), None, int_18013, None)
                # Getting the type of 'line' (line 213)
                line_18015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'line')
                # Obtaining the member '__getitem__' of a type (line 213)
                getitem___18016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 19), line_18015, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 213)
                subscript_call_result_18017 = invoke(stypy.reporting.localization.Localization(__file__, 213, 19), getitem___18016, slice_18014)
                
                # Assigning a type to the variable 'line' (line 213)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'line', subscript_call_result_18017)
                # SSA join for if statement (line 212)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 210)
            if_condition_18000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), endswith_call_result_17999)
            # Assigning a type to the variable 'if_condition_18000' (line 210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_18000', if_condition_18000)
            # SSA begins for if statement (line 210)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 211):
            
            # Obtaining the type of the subscript
            int_18001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 25), 'int')
            slice_18002 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 211, 19), None, int_18001, None)
            # Getting the type of 'line' (line 211)
            line_18003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 19), 'line')
            # Obtaining the member '__getitem__' of a type (line 211)
            getitem___18004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 19), line_18003, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 211)
            subscript_call_result_18005 = invoke(stypy.reporting.localization.Localization(__file__, 211, 19), getitem___18004, slice_18002)
            
            # Assigning a type to the variable 'line' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'line', subscript_call_result_18005)
            # SSA branch for the else part of an if statement (line 210)
            module_type_store.open_ssa_branch('else')
            
            
            # Obtaining the type of the subscript
            int_18006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 18), 'int')
            # Getting the type of 'line' (line 212)
            line_18007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'line')
            # Obtaining the member '__getitem__' of a type (line 212)
            getitem___18008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 13), line_18007, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 212)
            subscript_call_result_18009 = invoke(stypy.reporting.localization.Localization(__file__, 212, 13), getitem___18008, int_18006)
            
            # Getting the type of 'CRLF' (line 212)
            CRLF_18010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 25), 'CRLF')
            # Applying the binary operator 'in' (line 212)
            result_contains_18011 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 13), 'in', subscript_call_result_18009, CRLF_18010)
            
            # Testing if the type of an if condition is none (line 212)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 212, 13), result_contains_18011):
                pass
            else:
                
                # Testing the type of an if condition (line 212)
                if_condition_18012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 13), result_contains_18011)
                # Assigning a type to the variable 'if_condition_18012' (line 212)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'if_condition_18012', if_condition_18012)
                # SSA begins for if statement (line 212)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 213):
                
                # Obtaining the type of the subscript
                int_18013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 25), 'int')
                slice_18014 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 213, 19), None, int_18013, None)
                # Getting the type of 'line' (line 213)
                line_18015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'line')
                # Obtaining the member '__getitem__' of a type (line 213)
                getitem___18016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 19), line_18015, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 213)
                subscript_call_result_18017 = invoke(stypy.reporting.localization.Localization(__file__, 213, 19), getitem___18016, slice_18014)
                
                # Assigning a type to the variable 'line' (line 213)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'line', subscript_call_result_18017)
                # SSA join for if statement (line 212)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 210)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'lineno' (line 215)
        lineno_18018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'lineno')
        int_18019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 18), 'int')
        # Applying the binary operator '+=' (line 215)
        result_iadd_18020 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 8), '+=', lineno_18018, int_18019)
        # Assigning a type to the variable 'lineno' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'lineno', result_iadd_18020)
        
        
        # Assigning a Str to a Name (line 216):
        str_18021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 23), 'str', '')
        # Assigning a type to the variable 'encoded_line' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'encoded_line', str_18021)
        
        # Assigning a Name to a Name (line 217):
        # Getting the type of 'None' (line 217)
        None_18022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'None')
        # Assigning a type to the variable 'prev' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'prev', None_18022)
        
        # Assigning a Call to a Name (line 218):
        
        # Call to len(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'line' (line 218)
        line_18024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'line', False)
        # Processing the call keyword arguments (line 218)
        kwargs_18025 = {}
        # Getting the type of 'len' (line 218)
        len_18023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'len', False)
        # Calling len(args, kwargs) (line 218)
        len_call_result_18026 = invoke(stypy.reporting.localization.Localization(__file__, 218, 18), len_18023, *[line_18024], **kwargs_18025)
        
        # Assigning a type to the variable 'linelen' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'linelen', len_call_result_18026)
        
        
        # Call to range(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'linelen' (line 221)
        linelen_18028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 23), 'linelen', False)
        # Processing the call keyword arguments (line 221)
        kwargs_18029 = {}
        # Getting the type of 'range' (line 221)
        range_18027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 17), 'range', False)
        # Calling range(args, kwargs) (line 221)
        range_call_result_18030 = invoke(stypy.reporting.localization.Localization(__file__, 221, 17), range_18027, *[linelen_18028], **kwargs_18029)
        
        # Assigning a type to the variable 'range_call_result_18030' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'range_call_result_18030', range_call_result_18030)
        # Testing if the for loop is going to be iterated (line 221)
        # Testing the type of a for loop iterable (line 221)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 221, 8), range_call_result_18030)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 221, 8), range_call_result_18030):
            # Getting the type of the for loop variable (line 221)
            for_loop_var_18031 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 221, 8), range_call_result_18030)
            # Assigning a type to the variable 'j' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'j', for_loop_var_18031)
            # SSA begins for a for statement (line 221)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 222):
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 222)
            j_18032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 21), 'j')
            # Getting the type of 'line' (line 222)
            line_18033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'line')
            # Obtaining the member '__getitem__' of a type (line 222)
            getitem___18034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 16), line_18033, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 222)
            subscript_call_result_18035 = invoke(stypy.reporting.localization.Localization(__file__, 222, 16), getitem___18034, j_18032)
            
            # Assigning a type to the variable 'c' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'c', subscript_call_result_18035)
            
            # Assigning a Name to a Name (line 223):
            # Getting the type of 'c' (line 223)
            c_18036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 19), 'c')
            # Assigning a type to the variable 'prev' (line 223)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'prev', c_18036)
            
            # Call to match(...): (line 224)
            # Processing the call arguments (line 224)
            # Getting the type of 'c' (line 224)
            c_18039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 26), 'c', False)
            # Processing the call keyword arguments (line 224)
            kwargs_18040 = {}
            # Getting the type of 'bqre' (line 224)
            bqre_18037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'bqre', False)
            # Obtaining the member 'match' of a type (line 224)
            match_18038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 15), bqre_18037, 'match')
            # Calling match(args, kwargs) (line 224)
            match_call_result_18041 = invoke(stypy.reporting.localization.Localization(__file__, 224, 15), match_18038, *[c_18039], **kwargs_18040)
            
            # Testing if the type of an if condition is none (line 224)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 224, 12), match_call_result_18041):
                
                # Getting the type of 'j' (line 226)
                j_18047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'j')
                int_18048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 19), 'int')
                # Applying the binary operator '+' (line 226)
                result_add_18049 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 17), '+', j_18047, int_18048)
                
                # Getting the type of 'linelen' (line 226)
                linelen_18050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'linelen')
                # Applying the binary operator '==' (line 226)
                result_eq_18051 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 17), '==', result_add_18049, linelen_18050)
                
                # Testing if the type of an if condition is none (line 226)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 226, 17), result_eq_18051):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 226)
                    if_condition_18052 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 17), result_eq_18051)
                    # Assigning a type to the variable 'if_condition_18052' (line 226)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'if_condition_18052', if_condition_18052)
                    # SSA begins for if statement (line 226)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'c' (line 228)
                    c_18053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'c')
                    str_18054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 28), 'str', ' \t')
                    # Applying the binary operator 'notin' (line 228)
                    result_contains_18055 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 19), 'notin', c_18053, str_18054)
                    
                    # Testing if the type of an if condition is none (line 228)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 228, 16), result_contains_18055):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 228)
                        if_condition_18056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 16), result_contains_18055)
                        # Assigning a type to the variable 'if_condition_18056' (line 228)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'if_condition_18056', if_condition_18056)
                        # SSA begins for if statement (line 228)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'encoded_line' (line 229)
                        encoded_line_18057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'encoded_line')
                        # Getting the type of 'c' (line 229)
                        c_18058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 36), 'c')
                        # Applying the binary operator '+=' (line 229)
                        result_iadd_18059 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 20), '+=', encoded_line_18057, c_18058)
                        # Assigning a type to the variable 'encoded_line' (line 229)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'encoded_line', result_iadd_18059)
                        
                        # SSA join for if statement (line 228)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Assigning a Name to a Name (line 230):
                    # Getting the type of 'c' (line 230)
                    c_18060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'c')
                    # Assigning a type to the variable 'prev' (line 230)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'prev', c_18060)
                    # SSA join for if statement (line 226)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 224)
                if_condition_18042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 12), match_call_result_18041)
                # Assigning a type to the variable 'if_condition_18042' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'if_condition_18042', if_condition_18042)
                # SSA begins for if statement (line 224)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 225):
                
                # Call to quote(...): (line 225)
                # Processing the call arguments (line 225)
                # Getting the type of 'c' (line 225)
                c_18044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 26), 'c', False)
                # Processing the call keyword arguments (line 225)
                kwargs_18045 = {}
                # Getting the type of 'quote' (line 225)
                quote_18043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'quote', False)
                # Calling quote(args, kwargs) (line 225)
                quote_call_result_18046 = invoke(stypy.reporting.localization.Localization(__file__, 225, 20), quote_18043, *[c_18044], **kwargs_18045)
                
                # Assigning a type to the variable 'c' (line 225)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'c', quote_call_result_18046)
                # SSA branch for the else part of an if statement (line 224)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'j' (line 226)
                j_18047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'j')
                int_18048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 19), 'int')
                # Applying the binary operator '+' (line 226)
                result_add_18049 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 17), '+', j_18047, int_18048)
                
                # Getting the type of 'linelen' (line 226)
                linelen_18050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'linelen')
                # Applying the binary operator '==' (line 226)
                result_eq_18051 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 17), '==', result_add_18049, linelen_18050)
                
                # Testing if the type of an if condition is none (line 226)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 226, 17), result_eq_18051):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 226)
                    if_condition_18052 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 17), result_eq_18051)
                    # Assigning a type to the variable 'if_condition_18052' (line 226)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'if_condition_18052', if_condition_18052)
                    # SSA begins for if statement (line 226)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'c' (line 228)
                    c_18053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'c')
                    str_18054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 28), 'str', ' \t')
                    # Applying the binary operator 'notin' (line 228)
                    result_contains_18055 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 19), 'notin', c_18053, str_18054)
                    
                    # Testing if the type of an if condition is none (line 228)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 228, 16), result_contains_18055):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 228)
                        if_condition_18056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 16), result_contains_18055)
                        # Assigning a type to the variable 'if_condition_18056' (line 228)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'if_condition_18056', if_condition_18056)
                        # SSA begins for if statement (line 228)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'encoded_line' (line 229)
                        encoded_line_18057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'encoded_line')
                        # Getting the type of 'c' (line 229)
                        c_18058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 36), 'c')
                        # Applying the binary operator '+=' (line 229)
                        result_iadd_18059 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 20), '+=', encoded_line_18057, c_18058)
                        # Assigning a type to the variable 'encoded_line' (line 229)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'encoded_line', result_iadd_18059)
                        
                        # SSA join for if statement (line 228)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Assigning a Name to a Name (line 230):
                    # Getting the type of 'c' (line 230)
                    c_18060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'c')
                    # Assigning a type to the variable 'prev' (line 230)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'prev', c_18060)
                    # SSA join for if statement (line 226)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 224)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to len(...): (line 233)
            # Processing the call arguments (line 233)
            # Getting the type of 'encoded_line' (line 233)
            encoded_line_18062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 19), 'encoded_line', False)
            # Processing the call keyword arguments (line 233)
            kwargs_18063 = {}
            # Getting the type of 'len' (line 233)
            len_18061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'len', False)
            # Calling len(args, kwargs) (line 233)
            len_call_result_18064 = invoke(stypy.reporting.localization.Localization(__file__, 233, 15), len_18061, *[encoded_line_18062], **kwargs_18063)
            
            
            # Call to len(...): (line 233)
            # Processing the call arguments (line 233)
            # Getting the type of 'c' (line 233)
            c_18066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 39), 'c', False)
            # Processing the call keyword arguments (line 233)
            kwargs_18067 = {}
            # Getting the type of 'len' (line 233)
            len_18065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 35), 'len', False)
            # Calling len(args, kwargs) (line 233)
            len_call_result_18068 = invoke(stypy.reporting.localization.Localization(__file__, 233, 35), len_18065, *[c_18066], **kwargs_18067)
            
            # Applying the binary operator '+' (line 233)
            result_add_18069 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 15), '+', len_call_result_18064, len_call_result_18068)
            
            # Getting the type of 'maxlinelen' (line 233)
            maxlinelen_18070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 45), 'maxlinelen')
            # Applying the binary operator '>=' (line 233)
            result_ge_18071 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 15), '>=', result_add_18069, maxlinelen_18070)
            
            # Testing if the type of an if condition is none (line 233)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 233, 12), result_ge_18071):
                pass
            else:
                
                # Testing the type of an if condition (line 233)
                if_condition_18072 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 12), result_ge_18071)
                # Assigning a type to the variable 'if_condition_18072' (line 233)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'if_condition_18072', if_condition_18072)
                # SSA begins for if statement (line 233)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'encoded_body' (line 234)
                encoded_body_18073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'encoded_body')
                # Getting the type of 'encoded_line' (line 234)
                encoded_line_18074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 32), 'encoded_line')
                str_18075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 47), 'str', '=')
                # Applying the binary operator '+' (line 234)
                result_add_18076 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 32), '+', encoded_line_18074, str_18075)
                
                # Getting the type of 'eol' (line 234)
                eol_18077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 53), 'eol')
                # Applying the binary operator '+' (line 234)
                result_add_18078 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 51), '+', result_add_18076, eol_18077)
                
                # Applying the binary operator '+=' (line 234)
                result_iadd_18079 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 16), '+=', encoded_body_18073, result_add_18078)
                # Assigning a type to the variable 'encoded_body' (line 234)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'encoded_body', result_iadd_18079)
                
                
                # Assigning a Str to a Name (line 235):
                str_18080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 31), 'str', '')
                # Assigning a type to the variable 'encoded_line' (line 235)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'encoded_line', str_18080)
                # SSA join for if statement (line 233)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'encoded_line' (line 236)
            encoded_line_18081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'encoded_line')
            # Getting the type of 'c' (line 236)
            c_18082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 28), 'c')
            # Applying the binary operator '+=' (line 236)
            result_iadd_18083 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 12), '+=', encoded_line_18081, c_18082)
            # Assigning a type to the variable 'encoded_line' (line 236)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'encoded_line', result_iadd_18083)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Evaluating a boolean operation
        # Getting the type of 'prev' (line 238)
        prev_18084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'prev')
        
        # Getting the type of 'prev' (line 238)
        prev_18085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'prev')
        str_18086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 28), 'str', ' \t')
        # Applying the binary operator 'in' (line 238)
        result_contains_18087 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 20), 'in', prev_18085, str_18086)
        
        # Applying the binary operator 'and' (line 238)
        result_and_keyword_18088 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 11), 'and', prev_18084, result_contains_18087)
        
        # Testing if the type of an if condition is none (line 238)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 238, 8), result_and_keyword_18088):
            pass
        else:
            
            # Testing the type of an if condition (line 238)
            if_condition_18089 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), result_and_keyword_18088)
            # Assigning a type to the variable 'if_condition_18089' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_18089', if_condition_18089)
            # SSA begins for if statement (line 238)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'lineno' (line 240)
            lineno_18090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'lineno')
            int_18091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 24), 'int')
            # Applying the binary operator '+' (line 240)
            result_add_18092 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 15), '+', lineno_18090, int_18091)
            
            
            # Call to len(...): (line 240)
            # Processing the call arguments (line 240)
            # Getting the type of 'lines' (line 240)
            lines_18094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 33), 'lines', False)
            # Processing the call keyword arguments (line 240)
            kwargs_18095 = {}
            # Getting the type of 'len' (line 240)
            len_18093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 29), 'len', False)
            # Calling len(args, kwargs) (line 240)
            len_call_result_18096 = invoke(stypy.reporting.localization.Localization(__file__, 240, 29), len_18093, *[lines_18094], **kwargs_18095)
            
            # Applying the binary operator '==' (line 240)
            result_eq_18097 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 15), '==', result_add_18092, len_call_result_18096)
            
            # Testing if the type of an if condition is none (line 240)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 240, 12), result_eq_18097):
                
                # Getting the type of 'encoded_body' (line 248)
                encoded_body_18129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'encoded_body')
                # Getting the type of 'encoded_line' (line 248)
                encoded_line_18130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 32), 'encoded_line')
                # Getting the type of 'prev' (line 248)
                prev_18131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 47), 'prev')
                # Applying the binary operator '+' (line 248)
                result_add_18132 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 32), '+', encoded_line_18130, prev_18131)
                
                str_18133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 54), 'str', '=')
                # Applying the binary operator '+' (line 248)
                result_add_18134 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 52), '+', result_add_18132, str_18133)
                
                # Getting the type of 'eol' (line 248)
                eol_18135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 60), 'eol')
                # Applying the binary operator '+' (line 248)
                result_add_18136 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 58), '+', result_add_18134, eol_18135)
                
                # Applying the binary operator '+=' (line 248)
                result_iadd_18137 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 16), '+=', encoded_body_18129, result_add_18136)
                # Assigning a type to the variable 'encoded_body' (line 248)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'encoded_body', result_iadd_18137)
                
            else:
                
                # Testing the type of an if condition (line 240)
                if_condition_18098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 12), result_eq_18097)
                # Assigning a type to the variable 'if_condition_18098' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'if_condition_18098', if_condition_18098)
                # SSA begins for if statement (line 240)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 241):
                
                # Call to quote(...): (line 241)
                # Processing the call arguments (line 241)
                # Getting the type of 'prev' (line 241)
                prev_18100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 29), 'prev', False)
                # Processing the call keyword arguments (line 241)
                kwargs_18101 = {}
                # Getting the type of 'quote' (line 241)
                quote_18099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'quote', False)
                # Calling quote(args, kwargs) (line 241)
                quote_call_result_18102 = invoke(stypy.reporting.localization.Localization(__file__, 241, 23), quote_18099, *[prev_18100], **kwargs_18101)
                
                # Assigning a type to the variable 'prev' (line 241)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'prev', quote_call_result_18102)
                
                
                # Call to len(...): (line 242)
                # Processing the call arguments (line 242)
                # Getting the type of 'encoded_line' (line 242)
                encoded_line_18104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 23), 'encoded_line', False)
                # Processing the call keyword arguments (line 242)
                kwargs_18105 = {}
                # Getting the type of 'len' (line 242)
                len_18103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'len', False)
                # Calling len(args, kwargs) (line 242)
                len_call_result_18106 = invoke(stypy.reporting.localization.Localization(__file__, 242, 19), len_18103, *[encoded_line_18104], **kwargs_18105)
                
                
                # Call to len(...): (line 242)
                # Processing the call arguments (line 242)
                # Getting the type of 'prev' (line 242)
                prev_18108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 43), 'prev', False)
                # Processing the call keyword arguments (line 242)
                kwargs_18109 = {}
                # Getting the type of 'len' (line 242)
                len_18107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 39), 'len', False)
                # Calling len(args, kwargs) (line 242)
                len_call_result_18110 = invoke(stypy.reporting.localization.Localization(__file__, 242, 39), len_18107, *[prev_18108], **kwargs_18109)
                
                # Applying the binary operator '+' (line 242)
                result_add_18111 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 19), '+', len_call_result_18106, len_call_result_18110)
                
                # Getting the type of 'maxlinelen' (line 242)
                maxlinelen_18112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 51), 'maxlinelen')
                # Applying the binary operator '>' (line 242)
                result_gt_18113 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 19), '>', result_add_18111, maxlinelen_18112)
                
                # Testing if the type of an if condition is none (line 242)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 242, 16), result_gt_18113):
                    
                    # Getting the type of 'encoded_body' (line 245)
                    encoded_body_18124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'encoded_body')
                    # Getting the type of 'encoded_line' (line 245)
                    encoded_line_18125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 36), 'encoded_line')
                    # Getting the type of 'prev' (line 245)
                    prev_18126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 51), 'prev')
                    # Applying the binary operator '+' (line 245)
                    result_add_18127 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 36), '+', encoded_line_18125, prev_18126)
                    
                    # Applying the binary operator '+=' (line 245)
                    result_iadd_18128 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 20), '+=', encoded_body_18124, result_add_18127)
                    # Assigning a type to the variable 'encoded_body' (line 245)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'encoded_body', result_iadd_18128)
                    
                else:
                    
                    # Testing the type of an if condition (line 242)
                    if_condition_18114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 16), result_gt_18113)
                    # Assigning a type to the variable 'if_condition_18114' (line 242)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'if_condition_18114', if_condition_18114)
                    # SSA begins for if statement (line 242)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'encoded_body' (line 243)
                    encoded_body_18115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'encoded_body')
                    # Getting the type of 'encoded_line' (line 243)
                    encoded_line_18116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 36), 'encoded_line')
                    str_18117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 51), 'str', '=')
                    # Applying the binary operator '+' (line 243)
                    result_add_18118 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 36), '+', encoded_line_18116, str_18117)
                    
                    # Getting the type of 'eol' (line 243)
                    eol_18119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 57), 'eol')
                    # Applying the binary operator '+' (line 243)
                    result_add_18120 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 55), '+', result_add_18118, eol_18119)
                    
                    # Getting the type of 'prev' (line 243)
                    prev_18121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 63), 'prev')
                    # Applying the binary operator '+' (line 243)
                    result_add_18122 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 61), '+', result_add_18120, prev_18121)
                    
                    # Applying the binary operator '+=' (line 243)
                    result_iadd_18123 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 20), '+=', encoded_body_18115, result_add_18122)
                    # Assigning a type to the variable 'encoded_body' (line 243)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'encoded_body', result_iadd_18123)
                    
                    # SSA branch for the else part of an if statement (line 242)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'encoded_body' (line 245)
                    encoded_body_18124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'encoded_body')
                    # Getting the type of 'encoded_line' (line 245)
                    encoded_line_18125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 36), 'encoded_line')
                    # Getting the type of 'prev' (line 245)
                    prev_18126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 51), 'prev')
                    # Applying the binary operator '+' (line 245)
                    result_add_18127 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 36), '+', encoded_line_18125, prev_18126)
                    
                    # Applying the binary operator '+=' (line 245)
                    result_iadd_18128 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 20), '+=', encoded_body_18124, result_add_18127)
                    # Assigning a type to the variable 'encoded_body' (line 245)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'encoded_body', result_iadd_18128)
                    
                    # SSA join for if statement (line 242)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 240)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'encoded_body' (line 248)
                encoded_body_18129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'encoded_body')
                # Getting the type of 'encoded_line' (line 248)
                encoded_line_18130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 32), 'encoded_line')
                # Getting the type of 'prev' (line 248)
                prev_18131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 47), 'prev')
                # Applying the binary operator '+' (line 248)
                result_add_18132 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 32), '+', encoded_line_18130, prev_18131)
                
                str_18133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 54), 'str', '=')
                # Applying the binary operator '+' (line 248)
                result_add_18134 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 52), '+', result_add_18132, str_18133)
                
                # Getting the type of 'eol' (line 248)
                eol_18135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 60), 'eol')
                # Applying the binary operator '+' (line 248)
                result_add_18136 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 58), '+', result_add_18134, eol_18135)
                
                # Applying the binary operator '+=' (line 248)
                result_iadd_18137 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 16), '+=', encoded_body_18129, result_add_18136)
                # Assigning a type to the variable 'encoded_body' (line 248)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'encoded_body', result_iadd_18137)
                
                # SSA join for if statement (line 240)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Str to a Name (line 249):
            str_18138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 27), 'str', '')
            # Assigning a type to the variable 'encoded_line' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'encoded_line', str_18138)
            # SSA join for if statement (line 238)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Call to endswith(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'CRLF' (line 252)
        CRLF_18144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 34), 'CRLF', False)
        # Processing the call keyword arguments (line 252)
        kwargs_18145 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'lineno' (line 252)
        lineno_18139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 17), 'lineno', False)
        # Getting the type of 'lines' (line 252)
        lines_18140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 11), 'lines', False)
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___18141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 11), lines_18140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_18142 = invoke(stypy.reporting.localization.Localization(__file__, 252, 11), getitem___18141, lineno_18139)
        
        # Obtaining the member 'endswith' of a type (line 252)
        endswith_18143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 11), subscript_call_result_18142, 'endswith')
        # Calling endswith(args, kwargs) (line 252)
        endswith_call_result_18146 = invoke(stypy.reporting.localization.Localization(__file__, 252, 11), endswith_18143, *[CRLF_18144], **kwargs_18145)
        
        
        
        # Obtaining the type of the subscript
        int_18147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 57), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'lineno' (line 252)
        lineno_18148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 49), 'lineno')
        # Getting the type of 'lines' (line 252)
        lines_18149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 43), 'lines')
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___18150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 43), lines_18149, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_18151 = invoke(stypy.reporting.localization.Localization(__file__, 252, 43), getitem___18150, lineno_18148)
        
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___18152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 43), subscript_call_result_18151, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_18153 = invoke(stypy.reporting.localization.Localization(__file__, 252, 43), getitem___18152, int_18147)
        
        # Getting the type of 'CRLF' (line 252)
        CRLF_18154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 64), 'CRLF')
        # Applying the binary operator 'in' (line 252)
        result_contains_18155 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 43), 'in', subscript_call_result_18153, CRLF_18154)
        
        # Applying the binary operator 'or' (line 252)
        result_or_keyword_18156 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 11), 'or', endswith_call_result_18146, result_contains_18155)
        
        # Testing if the type of an if condition is none (line 252)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 252, 8), result_or_keyword_18156):
            
            # Getting the type of 'encoded_body' (line 255)
            encoded_body_18163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'encoded_body')
            # Getting the type of 'encoded_line' (line 255)
            encoded_line_18164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 28), 'encoded_line')
            # Applying the binary operator '+=' (line 255)
            result_iadd_18165 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 12), '+=', encoded_body_18163, encoded_line_18164)
            # Assigning a type to the variable 'encoded_body' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'encoded_body', result_iadd_18165)
            
        else:
            
            # Testing the type of an if condition (line 252)
            if_condition_18157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 8), result_or_keyword_18156)
            # Assigning a type to the variable 'if_condition_18157' (line 252)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'if_condition_18157', if_condition_18157)
            # SSA begins for if statement (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'encoded_body' (line 253)
            encoded_body_18158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'encoded_body')
            # Getting the type of 'encoded_line' (line 253)
            encoded_line_18159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'encoded_line')
            # Getting the type of 'eol' (line 253)
            eol_18160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 43), 'eol')
            # Applying the binary operator '+' (line 253)
            result_add_18161 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 28), '+', encoded_line_18159, eol_18160)
            
            # Applying the binary operator '+=' (line 253)
            result_iadd_18162 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 12), '+=', encoded_body_18158, result_add_18161)
            # Assigning a type to the variable 'encoded_body' (line 253)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'encoded_body', result_iadd_18162)
            
            # SSA branch for the else part of an if statement (line 252)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'encoded_body' (line 255)
            encoded_body_18163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'encoded_body')
            # Getting the type of 'encoded_line' (line 255)
            encoded_line_18164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 28), 'encoded_line')
            # Applying the binary operator '+=' (line 255)
            result_iadd_18165 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 12), '+=', encoded_body_18163, encoded_line_18164)
            # Assigning a type to the variable 'encoded_body' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'encoded_body', result_iadd_18165)
            
            # SSA join for if statement (line 252)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 256):
        str_18166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 23), 'str', '')
        # Assigning a type to the variable 'encoded_line' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'encoded_line', str_18166)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'encoded_body' (line 257)
    encoded_body_18167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'encoded_body')
    # Assigning a type to the variable 'stypy_return_type' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'stypy_return_type', encoded_body_18167)
    
    # ################# End of 'encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode' in the type store
    # Getting the type of 'stypy_return_type' (line 178)
    stypy_return_type_18168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18168)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode'
    return stypy_return_type_18168

# Assigning a type to the variable 'encode' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'encode', encode)

# Assigning a Name to a Name (line 261):
# Getting the type of 'encode' (line 261)
encode_18169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 14), 'encode')
# Assigning a type to the variable 'body_encode' (line 261)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 0), 'body_encode', encode_18169)

# Assigning a Name to a Name (line 262):
# Getting the type of 'encode' (line 262)
encode_18170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'encode')
# Assigning a type to the variable 'encodestring' (line 262)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'encodestring', encode_18170)

@norecursion
def decode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'NL' (line 268)
    NL_18171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 24), 'NL')
    defaults = [NL_18171]
    # Create a new context for function 'decode'
    module_type_store = module_type_store.open_function_context('decode', 268, 0, False)
    
    # Passed parameters checking function
    decode.stypy_localization = localization
    decode.stypy_type_of_self = None
    decode.stypy_type_store = module_type_store
    decode.stypy_function_name = 'decode'
    decode.stypy_param_names_list = ['encoded', 'eol']
    decode.stypy_varargs_param_name = None
    decode.stypy_kwargs_param_name = None
    decode.stypy_call_defaults = defaults
    decode.stypy_call_varargs = varargs
    decode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decode', ['encoded', 'eol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decode', localization, ['encoded', 'eol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decode(...)' code ##################

    str_18172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, (-1)), 'str', 'Decode a quoted-printable string.\n\n    Lines are separated with eol, which defaults to \\n.\n    ')
    
    # Getting the type of 'encoded' (line 273)
    encoded_18173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'encoded')
    # Applying the 'not' unary operator (line 273)
    result_not__18174 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 7), 'not', encoded_18173)
    
    # Testing if the type of an if condition is none (line 273)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 273, 4), result_not__18174):
        pass
    else:
        
        # Testing the type of an if condition (line 273)
        if_condition_18175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 4), result_not__18174)
        # Assigning a type to the variable 'if_condition_18175' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'if_condition_18175', if_condition_18175)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'encoded' (line 274)
        encoded_18176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'encoded')
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'stypy_return_type', encoded_18176)
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Str to a Name (line 278):
    str_18177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 14), 'str', '')
    # Assigning a type to the variable 'decoded' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'decoded', str_18177)
    
    
    # Call to splitlines(...): (line 280)
    # Processing the call keyword arguments (line 280)
    kwargs_18180 = {}
    # Getting the type of 'encoded' (line 280)
    encoded_18178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'encoded', False)
    # Obtaining the member 'splitlines' of a type (line 280)
    splitlines_18179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), encoded_18178, 'splitlines')
    # Calling splitlines(args, kwargs) (line 280)
    splitlines_call_result_18181 = invoke(stypy.reporting.localization.Localization(__file__, 280, 16), splitlines_18179, *[], **kwargs_18180)
    
    # Assigning a type to the variable 'splitlines_call_result_18181' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'splitlines_call_result_18181', splitlines_call_result_18181)
    # Testing if the for loop is going to be iterated (line 280)
    # Testing the type of a for loop iterable (line 280)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 280, 4), splitlines_call_result_18181)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 280, 4), splitlines_call_result_18181):
        # Getting the type of the for loop variable (line 280)
        for_loop_var_18182 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 280, 4), splitlines_call_result_18181)
        # Assigning a type to the variable 'line' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'line', for_loop_var_18182)
        # SSA begins for a for statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 281):
        
        # Call to rstrip(...): (line 281)
        # Processing the call keyword arguments (line 281)
        kwargs_18185 = {}
        # Getting the type of 'line' (line 281)
        line_18183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'line', False)
        # Obtaining the member 'rstrip' of a type (line 281)
        rstrip_18184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 15), line_18183, 'rstrip')
        # Calling rstrip(args, kwargs) (line 281)
        rstrip_call_result_18186 = invoke(stypy.reporting.localization.Localization(__file__, 281, 15), rstrip_18184, *[], **kwargs_18185)
        
        # Assigning a type to the variable 'line' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'line', rstrip_call_result_18186)
        
        # Getting the type of 'line' (line 282)
        line_18187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'line')
        # Applying the 'not' unary operator (line 282)
        result_not__18188 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 11), 'not', line_18187)
        
        # Testing if the type of an if condition is none (line 282)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 282, 8), result_not__18188):
            pass
        else:
            
            # Testing the type of an if condition (line 282)
            if_condition_18189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 8), result_not__18188)
            # Assigning a type to the variable 'if_condition_18189' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'if_condition_18189', if_condition_18189)
            # SSA begins for if statement (line 282)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'decoded' (line 283)
            decoded_18190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'decoded')
            # Getting the type of 'eol' (line 283)
            eol_18191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 23), 'eol')
            # Applying the binary operator '+=' (line 283)
            result_iadd_18192 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 12), '+=', decoded_18190, eol_18191)
            # Assigning a type to the variable 'decoded' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'decoded', result_iadd_18192)
            
            # SSA join for if statement (line 282)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Num to a Name (line 286):
        int_18193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 12), 'int')
        # Assigning a type to the variable 'i' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'i', int_18193)
        
        # Assigning a Call to a Name (line 287):
        
        # Call to len(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'line' (line 287)
        line_18195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'line', False)
        # Processing the call keyword arguments (line 287)
        kwargs_18196 = {}
        # Getting the type of 'len' (line 287)
        len_18194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'len', False)
        # Calling len(args, kwargs) (line 287)
        len_call_result_18197 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), len_18194, *[line_18195], **kwargs_18196)
        
        # Assigning a type to the variable 'n' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'n', len_call_result_18197)
        
        
        # Getting the type of 'i' (line 288)
        i_18198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 14), 'i')
        # Getting the type of 'n' (line 288)
        n_18199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'n')
        # Applying the binary operator '<' (line 288)
        result_lt_18200 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 14), '<', i_18198, n_18199)
        
        # Assigning a type to the variable 'result_lt_18200' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'result_lt_18200', result_lt_18200)
        # Testing if the while is going to be iterated (line 288)
        # Testing the type of an if condition (line 288)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 8), result_lt_18200)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 288, 8), result_lt_18200):
            # SSA begins for while statement (line 288)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Subscript to a Name (line 289):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 289)
            i_18201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 21), 'i')
            # Getting the type of 'line' (line 289)
            line_18202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'line')
            # Obtaining the member '__getitem__' of a type (line 289)
            getitem___18203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 16), line_18202, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 289)
            subscript_call_result_18204 = invoke(stypy.reporting.localization.Localization(__file__, 289, 16), getitem___18203, i_18201)
            
            # Assigning a type to the variable 'c' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'c', subscript_call_result_18204)
            
            # Getting the type of 'c' (line 290)
            c_18205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'c')
            str_18206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 20), 'str', '=')
            # Applying the binary operator '!=' (line 290)
            result_ne_18207 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 15), '!=', c_18205, str_18206)
            
            # Testing if the type of an if condition is none (line 290)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 290, 12), result_ne_18207):
                
                # Getting the type of 'i' (line 295)
                i_18215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'i')
                int_18216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 19), 'int')
                # Applying the binary operator '+' (line 295)
                result_add_18217 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 17), '+', i_18215, int_18216)
                
                # Getting the type of 'n' (line 295)
                n_18218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'n')
                # Applying the binary operator '==' (line 295)
                result_eq_18219 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 17), '==', result_add_18217, n_18218)
                
                # Testing if the type of an if condition is none (line 295)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 295, 17), result_eq_18219):
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'i' (line 299)
                    i_18224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'i')
                    int_18225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 19), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18226 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), '+', i_18224, int_18225)
                    
                    # Getting the type of 'n' (line 299)
                    n_18227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 23), 'n')
                    # Applying the binary operator '<' (line 299)
                    result_lt_18228 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), '<', result_add_18226, n_18227)
                    
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 299)
                    i_18229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'i')
                    int_18230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 36), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18231 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 34), '+', i_18229, int_18230)
                    
                    # Getting the type of 'line' (line 299)
                    line_18232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'line')
                    # Obtaining the member '__getitem__' of a type (line 299)
                    getitem___18233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 29), line_18232, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
                    subscript_call_result_18234 = invoke(stypy.reporting.localization.Localization(__file__, 299, 29), getitem___18233, result_add_18231)
                    
                    # Getting the type of 'hexdigits' (line 299)
                    hexdigits_18235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 42), 'hexdigits')
                    # Applying the binary operator 'in' (line 299)
                    result_contains_18236 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 29), 'in', subscript_call_result_18234, hexdigits_18235)
                    
                    # Applying the binary operator 'and' (line 299)
                    result_and_keyword_18237 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), 'and', result_lt_18228, result_contains_18236)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 299)
                    i_18238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 61), 'i')
                    int_18239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 63), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18240 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 61), '+', i_18238, int_18239)
                    
                    # Getting the type of 'line' (line 299)
                    line_18241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 56), 'line')
                    # Obtaining the member '__getitem__' of a type (line 299)
                    getitem___18242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 56), line_18241, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
                    subscript_call_result_18243 = invoke(stypy.reporting.localization.Localization(__file__, 299, 56), getitem___18242, result_add_18240)
                    
                    # Getting the type of 'hexdigits' (line 299)
                    hexdigits_18244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 69), 'hexdigits')
                    # Applying the binary operator 'in' (line 299)
                    result_contains_18245 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 56), 'in', subscript_call_result_18243, hexdigits_18244)
                    
                    # Applying the binary operator 'and' (line 299)
                    result_and_keyword_18246 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), 'and', result_and_keyword_18237, result_contains_18245)
                    
                    # Testing if the type of an if condition is none (line 299)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 299, 17), result_and_keyword_18246):
                        
                        # Getting the type of 'decoded' (line 304)
                        decoded_18264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded')
                        # Getting the type of 'c' (line 304)
                        c_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'c')
                        # Applying the binary operator '+=' (line 304)
                        result_iadd_18266 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 16), '+=', decoded_18264, c_18265)
                        # Assigning a type to the variable 'decoded' (line 304)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded', result_iadd_18266)
                        
                        
                        # Getting the type of 'i' (line 305)
                        i_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i')
                        int_18268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
                        # Applying the binary operator '+=' (line 305)
                        result_iadd_18269 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '+=', i_18267, int_18268)
                        # Assigning a type to the variable 'i' (line 305)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i', result_iadd_18269)
                        
                    else:
                        
                        # Testing the type of an if condition (line 299)
                        if_condition_18247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 17), result_and_keyword_18246)
                        # Assigning a type to the variable 'if_condition_18247' (line 299)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'if_condition_18247', if_condition_18247)
                        # SSA begins for if statement (line 299)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'decoded' (line 300)
                        decoded_18248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'decoded')
                        
                        # Call to unquote(...): (line 300)
                        # Processing the call arguments (line 300)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 300)
                        i_18250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 40), 'i', False)
                        # Getting the type of 'i' (line 300)
                        i_18251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 42), 'i', False)
                        int_18252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 44), 'int')
                        # Applying the binary operator '+' (line 300)
                        result_add_18253 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 42), '+', i_18251, int_18252)
                        
                        slice_18254 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 300, 35), i_18250, result_add_18253, None)
                        # Getting the type of 'line' (line 300)
                        line_18255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 35), 'line', False)
                        # Obtaining the member '__getitem__' of a type (line 300)
                        getitem___18256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 35), line_18255, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
                        subscript_call_result_18257 = invoke(stypy.reporting.localization.Localization(__file__, 300, 35), getitem___18256, slice_18254)
                        
                        # Processing the call keyword arguments (line 300)
                        kwargs_18258 = {}
                        # Getting the type of 'unquote' (line 300)
                        unquote_18249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'unquote', False)
                        # Calling unquote(args, kwargs) (line 300)
                        unquote_call_result_18259 = invoke(stypy.reporting.localization.Localization(__file__, 300, 27), unquote_18249, *[subscript_call_result_18257], **kwargs_18258)
                        
                        # Applying the binary operator '+=' (line 300)
                        result_iadd_18260 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 16), '+=', decoded_18248, unquote_call_result_18259)
                        # Assigning a type to the variable 'decoded' (line 300)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'decoded', result_iadd_18260)
                        
                        
                        # Getting the type of 'i' (line 301)
                        i_18261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'i')
                        int_18262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 21), 'int')
                        # Applying the binary operator '+=' (line 301)
                        result_iadd_18263 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 16), '+=', i_18261, int_18262)
                        # Assigning a type to the variable 'i' (line 301)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'i', result_iadd_18263)
                        
                        # SSA branch for the else part of an if statement (line 299)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'decoded' (line 304)
                        decoded_18264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded')
                        # Getting the type of 'c' (line 304)
                        c_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'c')
                        # Applying the binary operator '+=' (line 304)
                        result_iadd_18266 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 16), '+=', decoded_18264, c_18265)
                        # Assigning a type to the variable 'decoded' (line 304)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded', result_iadd_18266)
                        
                        
                        # Getting the type of 'i' (line 305)
                        i_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i')
                        int_18268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
                        # Applying the binary operator '+=' (line 305)
                        result_iadd_18269 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '+=', i_18267, int_18268)
                        # Assigning a type to the variable 'i' (line 305)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i', result_iadd_18269)
                        
                        # SSA join for if statement (line 299)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 295)
                    if_condition_18220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 17), result_eq_18219)
                    # Assigning a type to the variable 'if_condition_18220' (line 295)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'if_condition_18220', if_condition_18220)
                    # SSA begins for if statement (line 295)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'i' (line 296)
                    i_18221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'i')
                    int_18222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 21), 'int')
                    # Applying the binary operator '+=' (line 296)
                    result_iadd_18223 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 16), '+=', i_18221, int_18222)
                    # Assigning a type to the variable 'i' (line 296)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'i', result_iadd_18223)
                    
                    # SSA branch for the else part of an if statement (line 295)
                    module_type_store.open_ssa_branch('else')
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'i' (line 299)
                    i_18224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'i')
                    int_18225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 19), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18226 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), '+', i_18224, int_18225)
                    
                    # Getting the type of 'n' (line 299)
                    n_18227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 23), 'n')
                    # Applying the binary operator '<' (line 299)
                    result_lt_18228 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), '<', result_add_18226, n_18227)
                    
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 299)
                    i_18229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'i')
                    int_18230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 36), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18231 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 34), '+', i_18229, int_18230)
                    
                    # Getting the type of 'line' (line 299)
                    line_18232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'line')
                    # Obtaining the member '__getitem__' of a type (line 299)
                    getitem___18233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 29), line_18232, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
                    subscript_call_result_18234 = invoke(stypy.reporting.localization.Localization(__file__, 299, 29), getitem___18233, result_add_18231)
                    
                    # Getting the type of 'hexdigits' (line 299)
                    hexdigits_18235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 42), 'hexdigits')
                    # Applying the binary operator 'in' (line 299)
                    result_contains_18236 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 29), 'in', subscript_call_result_18234, hexdigits_18235)
                    
                    # Applying the binary operator 'and' (line 299)
                    result_and_keyword_18237 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), 'and', result_lt_18228, result_contains_18236)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 299)
                    i_18238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 61), 'i')
                    int_18239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 63), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18240 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 61), '+', i_18238, int_18239)
                    
                    # Getting the type of 'line' (line 299)
                    line_18241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 56), 'line')
                    # Obtaining the member '__getitem__' of a type (line 299)
                    getitem___18242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 56), line_18241, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
                    subscript_call_result_18243 = invoke(stypy.reporting.localization.Localization(__file__, 299, 56), getitem___18242, result_add_18240)
                    
                    # Getting the type of 'hexdigits' (line 299)
                    hexdigits_18244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 69), 'hexdigits')
                    # Applying the binary operator 'in' (line 299)
                    result_contains_18245 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 56), 'in', subscript_call_result_18243, hexdigits_18244)
                    
                    # Applying the binary operator 'and' (line 299)
                    result_and_keyword_18246 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), 'and', result_and_keyword_18237, result_contains_18245)
                    
                    # Testing if the type of an if condition is none (line 299)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 299, 17), result_and_keyword_18246):
                        
                        # Getting the type of 'decoded' (line 304)
                        decoded_18264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded')
                        # Getting the type of 'c' (line 304)
                        c_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'c')
                        # Applying the binary operator '+=' (line 304)
                        result_iadd_18266 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 16), '+=', decoded_18264, c_18265)
                        # Assigning a type to the variable 'decoded' (line 304)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded', result_iadd_18266)
                        
                        
                        # Getting the type of 'i' (line 305)
                        i_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i')
                        int_18268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
                        # Applying the binary operator '+=' (line 305)
                        result_iadd_18269 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '+=', i_18267, int_18268)
                        # Assigning a type to the variable 'i' (line 305)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i', result_iadd_18269)
                        
                    else:
                        
                        # Testing the type of an if condition (line 299)
                        if_condition_18247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 17), result_and_keyword_18246)
                        # Assigning a type to the variable 'if_condition_18247' (line 299)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'if_condition_18247', if_condition_18247)
                        # SSA begins for if statement (line 299)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'decoded' (line 300)
                        decoded_18248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'decoded')
                        
                        # Call to unquote(...): (line 300)
                        # Processing the call arguments (line 300)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 300)
                        i_18250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 40), 'i', False)
                        # Getting the type of 'i' (line 300)
                        i_18251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 42), 'i', False)
                        int_18252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 44), 'int')
                        # Applying the binary operator '+' (line 300)
                        result_add_18253 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 42), '+', i_18251, int_18252)
                        
                        slice_18254 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 300, 35), i_18250, result_add_18253, None)
                        # Getting the type of 'line' (line 300)
                        line_18255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 35), 'line', False)
                        # Obtaining the member '__getitem__' of a type (line 300)
                        getitem___18256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 35), line_18255, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
                        subscript_call_result_18257 = invoke(stypy.reporting.localization.Localization(__file__, 300, 35), getitem___18256, slice_18254)
                        
                        # Processing the call keyword arguments (line 300)
                        kwargs_18258 = {}
                        # Getting the type of 'unquote' (line 300)
                        unquote_18249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'unquote', False)
                        # Calling unquote(args, kwargs) (line 300)
                        unquote_call_result_18259 = invoke(stypy.reporting.localization.Localization(__file__, 300, 27), unquote_18249, *[subscript_call_result_18257], **kwargs_18258)
                        
                        # Applying the binary operator '+=' (line 300)
                        result_iadd_18260 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 16), '+=', decoded_18248, unquote_call_result_18259)
                        # Assigning a type to the variable 'decoded' (line 300)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'decoded', result_iadd_18260)
                        
                        
                        # Getting the type of 'i' (line 301)
                        i_18261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'i')
                        int_18262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 21), 'int')
                        # Applying the binary operator '+=' (line 301)
                        result_iadd_18263 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 16), '+=', i_18261, int_18262)
                        # Assigning a type to the variable 'i' (line 301)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'i', result_iadd_18263)
                        
                        # SSA branch for the else part of an if statement (line 299)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'decoded' (line 304)
                        decoded_18264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded')
                        # Getting the type of 'c' (line 304)
                        c_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'c')
                        # Applying the binary operator '+=' (line 304)
                        result_iadd_18266 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 16), '+=', decoded_18264, c_18265)
                        # Assigning a type to the variable 'decoded' (line 304)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded', result_iadd_18266)
                        
                        
                        # Getting the type of 'i' (line 305)
                        i_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i')
                        int_18268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
                        # Applying the binary operator '+=' (line 305)
                        result_iadd_18269 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '+=', i_18267, int_18268)
                        # Assigning a type to the variable 'i' (line 305)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i', result_iadd_18269)
                        
                        # SSA join for if statement (line 299)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 295)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 290)
                if_condition_18208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 12), result_ne_18207)
                # Assigning a type to the variable 'if_condition_18208' (line 290)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'if_condition_18208', if_condition_18208)
                # SSA begins for if statement (line 290)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'decoded' (line 291)
                decoded_18209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'decoded')
                # Getting the type of 'c' (line 291)
                c_18210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 27), 'c')
                # Applying the binary operator '+=' (line 291)
                result_iadd_18211 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 16), '+=', decoded_18209, c_18210)
                # Assigning a type to the variable 'decoded' (line 291)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'decoded', result_iadd_18211)
                
                
                # Getting the type of 'i' (line 292)
                i_18212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'i')
                int_18213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 21), 'int')
                # Applying the binary operator '+=' (line 292)
                result_iadd_18214 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 16), '+=', i_18212, int_18213)
                # Assigning a type to the variable 'i' (line 292)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'i', result_iadd_18214)
                
                # SSA branch for the else part of an if statement (line 290)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'i' (line 295)
                i_18215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'i')
                int_18216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 19), 'int')
                # Applying the binary operator '+' (line 295)
                result_add_18217 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 17), '+', i_18215, int_18216)
                
                # Getting the type of 'n' (line 295)
                n_18218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'n')
                # Applying the binary operator '==' (line 295)
                result_eq_18219 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 17), '==', result_add_18217, n_18218)
                
                # Testing if the type of an if condition is none (line 295)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 295, 17), result_eq_18219):
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'i' (line 299)
                    i_18224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'i')
                    int_18225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 19), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18226 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), '+', i_18224, int_18225)
                    
                    # Getting the type of 'n' (line 299)
                    n_18227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 23), 'n')
                    # Applying the binary operator '<' (line 299)
                    result_lt_18228 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), '<', result_add_18226, n_18227)
                    
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 299)
                    i_18229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'i')
                    int_18230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 36), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18231 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 34), '+', i_18229, int_18230)
                    
                    # Getting the type of 'line' (line 299)
                    line_18232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'line')
                    # Obtaining the member '__getitem__' of a type (line 299)
                    getitem___18233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 29), line_18232, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
                    subscript_call_result_18234 = invoke(stypy.reporting.localization.Localization(__file__, 299, 29), getitem___18233, result_add_18231)
                    
                    # Getting the type of 'hexdigits' (line 299)
                    hexdigits_18235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 42), 'hexdigits')
                    # Applying the binary operator 'in' (line 299)
                    result_contains_18236 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 29), 'in', subscript_call_result_18234, hexdigits_18235)
                    
                    # Applying the binary operator 'and' (line 299)
                    result_and_keyword_18237 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), 'and', result_lt_18228, result_contains_18236)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 299)
                    i_18238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 61), 'i')
                    int_18239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 63), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18240 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 61), '+', i_18238, int_18239)
                    
                    # Getting the type of 'line' (line 299)
                    line_18241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 56), 'line')
                    # Obtaining the member '__getitem__' of a type (line 299)
                    getitem___18242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 56), line_18241, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
                    subscript_call_result_18243 = invoke(stypy.reporting.localization.Localization(__file__, 299, 56), getitem___18242, result_add_18240)
                    
                    # Getting the type of 'hexdigits' (line 299)
                    hexdigits_18244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 69), 'hexdigits')
                    # Applying the binary operator 'in' (line 299)
                    result_contains_18245 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 56), 'in', subscript_call_result_18243, hexdigits_18244)
                    
                    # Applying the binary operator 'and' (line 299)
                    result_and_keyword_18246 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), 'and', result_and_keyword_18237, result_contains_18245)
                    
                    # Testing if the type of an if condition is none (line 299)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 299, 17), result_and_keyword_18246):
                        
                        # Getting the type of 'decoded' (line 304)
                        decoded_18264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded')
                        # Getting the type of 'c' (line 304)
                        c_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'c')
                        # Applying the binary operator '+=' (line 304)
                        result_iadd_18266 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 16), '+=', decoded_18264, c_18265)
                        # Assigning a type to the variable 'decoded' (line 304)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded', result_iadd_18266)
                        
                        
                        # Getting the type of 'i' (line 305)
                        i_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i')
                        int_18268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
                        # Applying the binary operator '+=' (line 305)
                        result_iadd_18269 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '+=', i_18267, int_18268)
                        # Assigning a type to the variable 'i' (line 305)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i', result_iadd_18269)
                        
                    else:
                        
                        # Testing the type of an if condition (line 299)
                        if_condition_18247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 17), result_and_keyword_18246)
                        # Assigning a type to the variable 'if_condition_18247' (line 299)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'if_condition_18247', if_condition_18247)
                        # SSA begins for if statement (line 299)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'decoded' (line 300)
                        decoded_18248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'decoded')
                        
                        # Call to unquote(...): (line 300)
                        # Processing the call arguments (line 300)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 300)
                        i_18250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 40), 'i', False)
                        # Getting the type of 'i' (line 300)
                        i_18251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 42), 'i', False)
                        int_18252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 44), 'int')
                        # Applying the binary operator '+' (line 300)
                        result_add_18253 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 42), '+', i_18251, int_18252)
                        
                        slice_18254 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 300, 35), i_18250, result_add_18253, None)
                        # Getting the type of 'line' (line 300)
                        line_18255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 35), 'line', False)
                        # Obtaining the member '__getitem__' of a type (line 300)
                        getitem___18256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 35), line_18255, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
                        subscript_call_result_18257 = invoke(stypy.reporting.localization.Localization(__file__, 300, 35), getitem___18256, slice_18254)
                        
                        # Processing the call keyword arguments (line 300)
                        kwargs_18258 = {}
                        # Getting the type of 'unquote' (line 300)
                        unquote_18249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'unquote', False)
                        # Calling unquote(args, kwargs) (line 300)
                        unquote_call_result_18259 = invoke(stypy.reporting.localization.Localization(__file__, 300, 27), unquote_18249, *[subscript_call_result_18257], **kwargs_18258)
                        
                        # Applying the binary operator '+=' (line 300)
                        result_iadd_18260 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 16), '+=', decoded_18248, unquote_call_result_18259)
                        # Assigning a type to the variable 'decoded' (line 300)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'decoded', result_iadd_18260)
                        
                        
                        # Getting the type of 'i' (line 301)
                        i_18261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'i')
                        int_18262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 21), 'int')
                        # Applying the binary operator '+=' (line 301)
                        result_iadd_18263 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 16), '+=', i_18261, int_18262)
                        # Assigning a type to the variable 'i' (line 301)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'i', result_iadd_18263)
                        
                        # SSA branch for the else part of an if statement (line 299)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'decoded' (line 304)
                        decoded_18264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded')
                        # Getting the type of 'c' (line 304)
                        c_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'c')
                        # Applying the binary operator '+=' (line 304)
                        result_iadd_18266 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 16), '+=', decoded_18264, c_18265)
                        # Assigning a type to the variable 'decoded' (line 304)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded', result_iadd_18266)
                        
                        
                        # Getting the type of 'i' (line 305)
                        i_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i')
                        int_18268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
                        # Applying the binary operator '+=' (line 305)
                        result_iadd_18269 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '+=', i_18267, int_18268)
                        # Assigning a type to the variable 'i' (line 305)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i', result_iadd_18269)
                        
                        # SSA join for if statement (line 299)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 295)
                    if_condition_18220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 17), result_eq_18219)
                    # Assigning a type to the variable 'if_condition_18220' (line 295)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'if_condition_18220', if_condition_18220)
                    # SSA begins for if statement (line 295)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'i' (line 296)
                    i_18221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'i')
                    int_18222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 21), 'int')
                    # Applying the binary operator '+=' (line 296)
                    result_iadd_18223 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 16), '+=', i_18221, int_18222)
                    # Assigning a type to the variable 'i' (line 296)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'i', result_iadd_18223)
                    
                    # SSA branch for the else part of an if statement (line 295)
                    module_type_store.open_ssa_branch('else')
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'i' (line 299)
                    i_18224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'i')
                    int_18225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 19), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18226 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), '+', i_18224, int_18225)
                    
                    # Getting the type of 'n' (line 299)
                    n_18227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 23), 'n')
                    # Applying the binary operator '<' (line 299)
                    result_lt_18228 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), '<', result_add_18226, n_18227)
                    
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 299)
                    i_18229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'i')
                    int_18230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 36), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18231 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 34), '+', i_18229, int_18230)
                    
                    # Getting the type of 'line' (line 299)
                    line_18232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'line')
                    # Obtaining the member '__getitem__' of a type (line 299)
                    getitem___18233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 29), line_18232, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
                    subscript_call_result_18234 = invoke(stypy.reporting.localization.Localization(__file__, 299, 29), getitem___18233, result_add_18231)
                    
                    # Getting the type of 'hexdigits' (line 299)
                    hexdigits_18235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 42), 'hexdigits')
                    # Applying the binary operator 'in' (line 299)
                    result_contains_18236 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 29), 'in', subscript_call_result_18234, hexdigits_18235)
                    
                    # Applying the binary operator 'and' (line 299)
                    result_and_keyword_18237 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), 'and', result_lt_18228, result_contains_18236)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 299)
                    i_18238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 61), 'i')
                    int_18239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 63), 'int')
                    # Applying the binary operator '+' (line 299)
                    result_add_18240 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 61), '+', i_18238, int_18239)
                    
                    # Getting the type of 'line' (line 299)
                    line_18241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 56), 'line')
                    # Obtaining the member '__getitem__' of a type (line 299)
                    getitem___18242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 56), line_18241, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
                    subscript_call_result_18243 = invoke(stypy.reporting.localization.Localization(__file__, 299, 56), getitem___18242, result_add_18240)
                    
                    # Getting the type of 'hexdigits' (line 299)
                    hexdigits_18244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 69), 'hexdigits')
                    # Applying the binary operator 'in' (line 299)
                    result_contains_18245 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 56), 'in', subscript_call_result_18243, hexdigits_18244)
                    
                    # Applying the binary operator 'and' (line 299)
                    result_and_keyword_18246 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), 'and', result_and_keyword_18237, result_contains_18245)
                    
                    # Testing if the type of an if condition is none (line 299)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 299, 17), result_and_keyword_18246):
                        
                        # Getting the type of 'decoded' (line 304)
                        decoded_18264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded')
                        # Getting the type of 'c' (line 304)
                        c_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'c')
                        # Applying the binary operator '+=' (line 304)
                        result_iadd_18266 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 16), '+=', decoded_18264, c_18265)
                        # Assigning a type to the variable 'decoded' (line 304)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded', result_iadd_18266)
                        
                        
                        # Getting the type of 'i' (line 305)
                        i_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i')
                        int_18268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
                        # Applying the binary operator '+=' (line 305)
                        result_iadd_18269 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '+=', i_18267, int_18268)
                        # Assigning a type to the variable 'i' (line 305)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i', result_iadd_18269)
                        
                    else:
                        
                        # Testing the type of an if condition (line 299)
                        if_condition_18247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 17), result_and_keyword_18246)
                        # Assigning a type to the variable 'if_condition_18247' (line 299)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'if_condition_18247', if_condition_18247)
                        # SSA begins for if statement (line 299)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'decoded' (line 300)
                        decoded_18248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'decoded')
                        
                        # Call to unquote(...): (line 300)
                        # Processing the call arguments (line 300)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'i' (line 300)
                        i_18250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 40), 'i', False)
                        # Getting the type of 'i' (line 300)
                        i_18251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 42), 'i', False)
                        int_18252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 44), 'int')
                        # Applying the binary operator '+' (line 300)
                        result_add_18253 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 42), '+', i_18251, int_18252)
                        
                        slice_18254 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 300, 35), i_18250, result_add_18253, None)
                        # Getting the type of 'line' (line 300)
                        line_18255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 35), 'line', False)
                        # Obtaining the member '__getitem__' of a type (line 300)
                        getitem___18256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 35), line_18255, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
                        subscript_call_result_18257 = invoke(stypy.reporting.localization.Localization(__file__, 300, 35), getitem___18256, slice_18254)
                        
                        # Processing the call keyword arguments (line 300)
                        kwargs_18258 = {}
                        # Getting the type of 'unquote' (line 300)
                        unquote_18249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'unquote', False)
                        # Calling unquote(args, kwargs) (line 300)
                        unquote_call_result_18259 = invoke(stypy.reporting.localization.Localization(__file__, 300, 27), unquote_18249, *[subscript_call_result_18257], **kwargs_18258)
                        
                        # Applying the binary operator '+=' (line 300)
                        result_iadd_18260 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 16), '+=', decoded_18248, unquote_call_result_18259)
                        # Assigning a type to the variable 'decoded' (line 300)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'decoded', result_iadd_18260)
                        
                        
                        # Getting the type of 'i' (line 301)
                        i_18261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'i')
                        int_18262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 21), 'int')
                        # Applying the binary operator '+=' (line 301)
                        result_iadd_18263 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 16), '+=', i_18261, int_18262)
                        # Assigning a type to the variable 'i' (line 301)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'i', result_iadd_18263)
                        
                        # SSA branch for the else part of an if statement (line 299)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'decoded' (line 304)
                        decoded_18264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded')
                        # Getting the type of 'c' (line 304)
                        c_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'c')
                        # Applying the binary operator '+=' (line 304)
                        result_iadd_18266 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 16), '+=', decoded_18264, c_18265)
                        # Assigning a type to the variable 'decoded' (line 304)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'decoded', result_iadd_18266)
                        
                        
                        # Getting the type of 'i' (line 305)
                        i_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i')
                        int_18268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 21), 'int')
                        # Applying the binary operator '+=' (line 305)
                        result_iadd_18269 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 16), '+=', i_18267, int_18268)
                        # Assigning a type to the variable 'i' (line 305)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'i', result_iadd_18269)
                        
                        # SSA join for if statement (line 299)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 295)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 290)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'i' (line 307)
            i_18270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'i')
            # Getting the type of 'n' (line 307)
            n_18271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'n')
            # Applying the binary operator '==' (line 307)
            result_eq_18272 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 15), '==', i_18270, n_18271)
            
            # Testing if the type of an if condition is none (line 307)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 307, 12), result_eq_18272):
                pass
            else:
                
                # Testing the type of an if condition (line 307)
                if_condition_18273 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 12), result_eq_18272)
                # Assigning a type to the variable 'if_condition_18273' (line 307)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'if_condition_18273', if_condition_18273)
                # SSA begins for if statement (line 307)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'decoded' (line 308)
                decoded_18274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'decoded')
                # Getting the type of 'eol' (line 308)
                eol_18275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 27), 'eol')
                # Applying the binary operator '+=' (line 308)
                result_iadd_18276 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 16), '+=', decoded_18274, eol_18275)
                # Assigning a type to the variable 'decoded' (line 308)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'decoded', result_iadd_18276)
                
                # SSA join for if statement (line 307)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for while statement (line 288)
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Evaluating a boolean operation
    
    
    # Call to endswith(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'eol' (line 310)
    eol_18279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 28), 'eol', False)
    # Processing the call keyword arguments (line 310)
    kwargs_18280 = {}
    # Getting the type of 'encoded' (line 310)
    encoded_18277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'encoded', False)
    # Obtaining the member 'endswith' of a type (line 310)
    endswith_18278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 11), encoded_18277, 'endswith')
    # Calling endswith(args, kwargs) (line 310)
    endswith_call_result_18281 = invoke(stypy.reporting.localization.Localization(__file__, 310, 11), endswith_18278, *[eol_18279], **kwargs_18280)
    
    # Applying the 'not' unary operator (line 310)
    result_not__18282 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 7), 'not', endswith_call_result_18281)
    
    
    # Call to endswith(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'eol' (line 310)
    eol_18285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 54), 'eol', False)
    # Processing the call keyword arguments (line 310)
    kwargs_18286 = {}
    # Getting the type of 'decoded' (line 310)
    decoded_18283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 37), 'decoded', False)
    # Obtaining the member 'endswith' of a type (line 310)
    endswith_18284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 37), decoded_18283, 'endswith')
    # Calling endswith(args, kwargs) (line 310)
    endswith_call_result_18287 = invoke(stypy.reporting.localization.Localization(__file__, 310, 37), endswith_18284, *[eol_18285], **kwargs_18286)
    
    # Applying the binary operator 'and' (line 310)
    result_and_keyword_18288 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 7), 'and', result_not__18282, endswith_call_result_18287)
    
    # Testing if the type of an if condition is none (line 310)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 310, 4), result_and_keyword_18288):
        pass
    else:
        
        # Testing the type of an if condition (line 310)
        if_condition_18289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 4), result_and_keyword_18288)
        # Assigning a type to the variable 'if_condition_18289' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'if_condition_18289', if_condition_18289)
        # SSA begins for if statement (line 310)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 311):
        
        # Obtaining the type of the subscript
        int_18290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 27), 'int')
        slice_18291 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 311, 18), None, int_18290, None)
        # Getting the type of 'decoded' (line 311)
        decoded_18292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 18), 'decoded')
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___18293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 18), decoded_18292, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_18294 = invoke(stypy.reporting.localization.Localization(__file__, 311, 18), getitem___18293, slice_18291)
        
        # Assigning a type to the variable 'decoded' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'decoded', subscript_call_result_18294)
        # SSA join for if statement (line 310)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'decoded' (line 312)
    decoded_18295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 'decoded')
    # Assigning a type to the variable 'stypy_return_type' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type', decoded_18295)
    
    # ################# End of 'decode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode' in the type store
    # Getting the type of 'stypy_return_type' (line 268)
    stypy_return_type_18296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18296)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode'
    return stypy_return_type_18296

# Assigning a type to the variable 'decode' (line 268)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 0), 'decode', decode)

# Assigning a Name to a Name (line 316):
# Getting the type of 'decode' (line 316)
decode_18297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 14), 'decode')
# Assigning a type to the variable 'body_decode' (line 316)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 0), 'body_decode', decode_18297)

# Assigning a Name to a Name (line 317):
# Getting the type of 'decode' (line 317)
decode_18298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'decode')
# Assigning a type to the variable 'decodestring' (line 317)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'decodestring', decode_18298)

@norecursion
def _unquote_match(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_unquote_match'
    module_type_store = module_type_store.open_function_context('_unquote_match', 321, 0, False)
    
    # Passed parameters checking function
    _unquote_match.stypy_localization = localization
    _unquote_match.stypy_type_of_self = None
    _unquote_match.stypy_type_store = module_type_store
    _unquote_match.stypy_function_name = '_unquote_match'
    _unquote_match.stypy_param_names_list = ['match']
    _unquote_match.stypy_varargs_param_name = None
    _unquote_match.stypy_kwargs_param_name = None
    _unquote_match.stypy_call_defaults = defaults
    _unquote_match.stypy_call_varargs = varargs
    _unquote_match.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_unquote_match', ['match'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_unquote_match', localization, ['match'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_unquote_match(...)' code ##################

    str_18299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 4), 'str', 'Turn a match in the form =AB to the ASCII character with value 0xab')
    
    # Assigning a Call to a Name (line 323):
    
    # Call to group(...): (line 323)
    # Processing the call arguments (line 323)
    int_18302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 20), 'int')
    # Processing the call keyword arguments (line 323)
    kwargs_18303 = {}
    # Getting the type of 'match' (line 323)
    match_18300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'match', False)
    # Obtaining the member 'group' of a type (line 323)
    group_18301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), match_18300, 'group')
    # Calling group(args, kwargs) (line 323)
    group_call_result_18304 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), group_18301, *[int_18302], **kwargs_18303)
    
    # Assigning a type to the variable 's' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 's', group_call_result_18304)
    
    # Call to unquote(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 's' (line 324)
    s_18306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 's', False)
    # Processing the call keyword arguments (line 324)
    kwargs_18307 = {}
    # Getting the type of 'unquote' (line 324)
    unquote_18305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 11), 'unquote', False)
    # Calling unquote(args, kwargs) (line 324)
    unquote_call_result_18308 = invoke(stypy.reporting.localization.Localization(__file__, 324, 11), unquote_18305, *[s_18306], **kwargs_18307)
    
    # Assigning a type to the variable 'stypy_return_type' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'stypy_return_type', unquote_call_result_18308)
    
    # ################# End of '_unquote_match(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_unquote_match' in the type store
    # Getting the type of 'stypy_return_type' (line 321)
    stypy_return_type_18309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18309)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_unquote_match'
    return stypy_return_type_18309

# Assigning a type to the variable '_unquote_match' (line 321)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), '_unquote_match', _unquote_match)

@norecursion
def header_decode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'header_decode'
    module_type_store = module_type_store.open_function_context('header_decode', 328, 0, False)
    
    # Passed parameters checking function
    header_decode.stypy_localization = localization
    header_decode.stypy_type_of_self = None
    header_decode.stypy_type_store = module_type_store
    header_decode.stypy_function_name = 'header_decode'
    header_decode.stypy_param_names_list = ['s']
    header_decode.stypy_varargs_param_name = None
    header_decode.stypy_kwargs_param_name = None
    header_decode.stypy_call_defaults = defaults
    header_decode.stypy_call_varargs = varargs
    header_decode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'header_decode', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'header_decode', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'header_decode(...)' code ##################

    str_18310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, (-1)), 'str', "Decode a string encoded with RFC 2045 MIME header `Q' encoding.\n\n    This function does not parse a full MIME header value encoded with\n    quoted-printable (like =?iso-8859-1?q?Hello_World?=) -- please use\n    the high level email.header class for that functionality.\n    ")
    
    # Assigning a Call to a Name (line 335):
    
    # Call to replace(...): (line 335)
    # Processing the call arguments (line 335)
    str_18313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 18), 'str', '_')
    str_18314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 23), 'str', ' ')
    # Processing the call keyword arguments (line 335)
    kwargs_18315 = {}
    # Getting the type of 's' (line 335)
    s_18311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 's', False)
    # Obtaining the member 'replace' of a type (line 335)
    replace_18312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), s_18311, 'replace')
    # Calling replace(args, kwargs) (line 335)
    replace_call_result_18316 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), replace_18312, *[str_18313, str_18314], **kwargs_18315)
    
    # Assigning a type to the variable 's' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 's', replace_call_result_18316)
    
    # Call to sub(...): (line 336)
    # Processing the call arguments (line 336)
    str_18319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 18), 'str', '=[a-fA-F0-9]{2}')
    # Getting the type of '_unquote_match' (line 336)
    _unquote_match_18320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 38), '_unquote_match', False)
    # Getting the type of 's' (line 336)
    s_18321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 54), 's', False)
    # Processing the call keyword arguments (line 336)
    kwargs_18322 = {}
    # Getting the type of 're' (line 336)
    re_18317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 're', False)
    # Obtaining the member 'sub' of a type (line 336)
    sub_18318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 11), re_18317, 'sub')
    # Calling sub(args, kwargs) (line 336)
    sub_call_result_18323 = invoke(stypy.reporting.localization.Localization(__file__, 336, 11), sub_18318, *[str_18319, _unquote_match_18320, s_18321], **kwargs_18322)
    
    # Assigning a type to the variable 'stypy_return_type' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type', sub_call_result_18323)
    
    # ################# End of 'header_decode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'header_decode' in the type store
    # Getting the type of 'stypy_return_type' (line 328)
    stypy_return_type_18324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18324)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'header_decode'
    return stypy_return_type_18324

# Assigning a type to the variable 'header_decode' (line 328)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 0), 'header_decode', header_decode)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
