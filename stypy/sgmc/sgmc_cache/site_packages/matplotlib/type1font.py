
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This module contains a class representing a Type 1 font.
3: 
4: This version reads pfa and pfb files and splits them for embedding in
5: pdf files. It also supports SlantFont and ExtendFont transformations,
6: similarly to pdfTeX and friends. There is no support yet for
7: subsetting.
8: 
9: Usage::
10: 
11:    >>> font = Type1Font(filename)
12:    >>> clear_part, encrypted_part, finale = font.parts
13:    >>> slanted_font = font.transform({'slant': 0.167})
14:    >>> extended_font = font.transform({'extend': 1.2})
15: 
16: Sources:
17: 
18: * Adobe Technical Note #5040, Supporting Downloadable PostScript
19:   Language Fonts.
20: 
21: * Adobe Type 1 Font Format, Adobe Systems Incorporated, third printing,
22:   v1.1, 1993. ISBN 0-201-57044-0.
23: '''
24: 
25: from __future__ import (absolute_import, division, print_function,
26:                         unicode_literals)
27: 
28: import six
29: from six import unichr
30: 
31: import binascii
32: import io
33: import itertools
34: import numpy as np
35: import re
36: import struct
37: import sys
38: 
39: if six.PY3:
40:     def ord(x):
41:         return x
42: 
43: 
44: class Type1Font(object):
45:     '''
46:     A class representing a Type-1 font, for use by backends.
47: 
48:     .. attribute:: parts
49: 
50:        A 3-tuple of the cleartext part, the encrypted part, and the
51:        finale of zeros.
52: 
53:     .. attribute:: prop
54: 
55:        A dictionary of font properties.
56:     '''
57:     __slots__ = ('parts', 'prop')
58: 
59:     def __init__(self, input):
60:         '''
61:         Initialize a Type-1 font. *input* can be either the file name of
62:         a pfb file or a 3-tuple of already-decoded Type-1 font parts.
63:         '''
64:         if isinstance(input, tuple) and len(input) == 3:
65:             self.parts = input
66:         else:
67:             with open(input, 'rb') as file:
68:                 data = self._read(file)
69:             self.parts = self._split(data)
70: 
71:         self._parse()
72: 
73:     def _read(self, file):
74:         '''
75:         Read the font from a file, decoding into usable parts.
76:         '''
77:         rawdata = file.read()
78:         if not rawdata.startswith(b'\x80'):
79:             return rawdata
80: 
81:         data = b''
82:         while len(rawdata) > 0:
83:             if not rawdata.startswith(b'\x80'):
84:                 raise RuntimeError('Broken pfb file (expected byte 128, '
85:                                    'got %d)' % ord(rawdata[0]))
86:             type = ord(rawdata[1])
87:             if type in (1, 2):
88:                 length, = struct.unpack(str('<i'), rawdata[2:6])
89:                 segment = rawdata[6:6 + length]
90:                 rawdata = rawdata[6 + length:]
91: 
92:             if type == 1:       # ASCII text: include verbatim
93:                 data += segment
94:             elif type == 2:     # binary data: encode in hexadecimal
95:                 data += binascii.hexlify(segment)
96:             elif type == 3:     # end of file
97:                 break
98:             else:
99:                 raise RuntimeError('Unknown segment type %d in pfb file' %
100:                                    type)
101: 
102:         return data
103: 
104:     def _split(self, data):
105:         '''
106:         Split the Type 1 font into its three main parts.
107: 
108:         The three parts are: (1) the cleartext part, which ends in a
109:         eexec operator; (2) the encrypted part; (3) the fixed part,
110:         which contains 512 ASCII zeros possibly divided on various
111:         lines, a cleartomark operator, and possibly something else.
112:         '''
113: 
114:         # Cleartext part: just find the eexec and skip whitespace
115:         idx = data.index(b'eexec')
116:         idx += len(b'eexec')
117:         while data[idx] in b' \t\r\n':
118:             idx += 1
119:         len1 = idx
120: 
121:         # Encrypted part: find the cleartomark operator and count
122:         # zeros backward
123:         idx = data.rindex(b'cleartomark') - 1
124:         zeros = 512
125:         while zeros and data[idx] in b'0' or data[idx] in b'\r\n':
126:             if data[idx] in b'0':
127:                 zeros -= 1
128:             idx -= 1
129:         if zeros:
130:             raise RuntimeError('Insufficiently many zeros in Type 1 font')
131: 
132:         # Convert encrypted part to binary (if we read a pfb file, we
133:         # may end up converting binary to hexadecimal to binary again;
134:         # but if we read a pfa file, this part is already in hex, and
135:         # I am not quite sure if even the pfb format guarantees that
136:         # it will be in binary).
137:         binary = binascii.unhexlify(data[len1:idx+1])
138: 
139:         return data[:len1], binary, data[idx+1:]
140: 
141:     _whitespace_re = re.compile(br'[\0\t\r\014\n ]+')
142:     _token_re = re.compile(br'/{0,2}[^]\0\t\r\v\n ()<>{}/%[]+')
143:     _comment_re = re.compile(br'%[^\r\n\v]*')
144:     _instring_re = re.compile(br'[()\\]')
145: 
146:     # token types, compared via object identity (poor man's enum)
147:     _whitespace = object()
148:     _name = object()
149:     _string = object()
150:     _delimiter = object()
151:     _number = object()
152: 
153:     @classmethod
154:     def _tokens(cls, text):
155:         '''
156:         A PostScript tokenizer. Yield (token, value) pairs such as
157:         (cls._whitespace, '   ') or (cls._name, '/Foobar').
158:         '''
159:         pos = 0
160:         while pos < len(text):
161:             match = (cls._comment_re.match(text[pos:]) or
162:                      cls._whitespace_re.match(text[pos:]))
163:             if match:
164:                 yield (cls._whitespace, match.group())
165:                 pos += match.end()
166:             elif text[pos] == b'(':
167:                 start = pos
168:                 pos += 1
169:                 depth = 1
170:                 while depth:
171:                     match = cls._instring_re.search(text[pos:])
172:                     if match is None:
173:                         return
174:                     pos += match.end()
175:                     if match.group() == b'(':
176:                         depth += 1
177:                     elif match.group() == b')':
178:                         depth -= 1
179:                     else:  # a backslash - skip the next character
180:                         pos += 1
181:                 yield (cls._string, text[start:pos])
182:             elif text[pos:pos + 2] in (b'<<', b'>>'):
183:                 yield (cls._delimiter, text[pos:pos + 2])
184:                 pos += 2
185:             elif text[pos] == b'<':
186:                 start = pos
187:                 pos += text[pos:].index(b'>')
188:                 yield (cls._string, text[start:pos])
189:             else:
190:                 match = cls._token_re.match(text[pos:])
191:                 if match:
192:                     try:
193:                         float(match.group())
194:                         yield (cls._number, match.group())
195:                     except ValueError:
196:                         yield (cls._name, match.group())
197:                     pos += match.end()
198:                 else:
199:                     yield (cls._delimiter, text[pos:pos + 1])
200:                     pos += 1
201: 
202:     def _parse(self):
203:         '''
204:         Find the values of various font properties. This limited kind
205:         of parsing is described in Chapter 10 "Adobe Type Manager
206:         Compatibility" of the Type-1 spec.
207:         '''
208:         # Start with reasonable defaults
209:         prop = {'weight': 'Regular', 'ItalicAngle': 0.0, 'isFixedPitch': False,
210:                 'UnderlinePosition': -100, 'UnderlineThickness': 50}
211:         filtered = ((token, value)
212:                     for token, value in self._tokens(self.parts[0])
213:                     if token is not self._whitespace)
214:         # The spec calls this an ASCII format; in Python 2.x we could
215:         # just treat the strings and names as opaque bytes but let's
216:         # turn them into proper Unicode, and be lenient in case of high bytes.
217:         convert = lambda x: x.decode('ascii', 'replace')
218:         for token, value in filtered:
219:             if token is self._name and value.startswith(b'/'):
220:                 key = convert(value[1:])
221:                 token, value = next(filtered)
222:                 if token is self._name:
223:                     if value in (b'true', b'false'):
224:                         value = value == b'true'
225:                     else:
226:                         value = convert(value.lstrip(b'/'))
227:                 elif token is self._string:
228:                     value = convert(value.lstrip(b'(').rstrip(b')'))
229:                 elif token is self._number:
230:                     if b'.' in value:
231:                         value = float(value)
232:                     else:
233:                         value = int(value)
234:                 else:  # more complicated value such as an array
235:                     value = None
236:                 if key != 'FontInfo' and value is not None:
237:                     prop[key] = value
238: 
239:         # Fill in the various *Name properties
240:         if 'FontName' not in prop:
241:             prop['FontName'] = (prop.get('FullName') or
242:                                 prop.get('FamilyName') or
243:                                 'Unknown')
244:         if 'FullName' not in prop:
245:             prop['FullName'] = prop['FontName']
246:         if 'FamilyName' not in prop:
247:             extras = r'(?i)([ -](regular|plain|italic|oblique|(semi)?bold|(ultra)?light|extra|condensed))+$'
248:             prop['FamilyName'] = re.sub(extras, '', prop['FullName'])
249: 
250:         self.prop = prop
251: 
252:     @classmethod
253:     def _transformer(cls, tokens, slant, extend):
254:         def fontname(name):
255:             result = name
256:             if slant:
257:                 result += b'_Slant_' + str(int(1000 * slant)).encode('latin-1')
258:             if extend != 1.0:
259:                 result += b'_Extend_' + str(int(1000 * extend)).encode('latin-1')
260:             return result
261: 
262:         def italicangle(angle):
263:             return str(float(angle) - np.arctan(slant) / np.pi * 180).encode('latin-1')
264: 
265:         def fontmatrix(array):
266:             array = array.lstrip(b'[').rstrip(b']').strip().split()
267:             array = [float(x) for x in array]
268:             oldmatrix = np.eye(3, 3)
269:             oldmatrix[0:3, 0] = array[::2]
270:             oldmatrix[0:3, 1] = array[1::2]
271:             modifier = np.array([[extend, 0, 0],
272:                                  [slant, 1, 0],
273:                                  [0, 0, 1]])
274:             newmatrix = np.dot(modifier, oldmatrix)
275:             array[::2] = newmatrix[0:3, 0]
276:             array[1::2] = newmatrix[0:3, 1]
277:             as_string = u'[' + u' '.join(str(x) for x in array) + u']'
278:             return as_string.encode('latin-1')
279: 
280:         def replace(fun):
281:             def replacer(tokens):
282:                 token, value = next(tokens)      # name, e.g., /FontMatrix
283:                 yield bytes(value)
284:                 token, value = next(tokens)      # possible whitespace
285:                 while token is cls._whitespace:
286:                     yield bytes(value)
287:                     token, value = next(tokens)
288:                 if value != b'[':                # name/number/etc.
289:                     yield bytes(fun(value))
290:                 else:                            # array, e.g., [1 2 3]
291:                     result = b''
292:                     while value != b']':
293:                         result += value
294:                         token, value = next(tokens)
295:                     result += value
296:                     yield fun(result)
297:             return replacer
298: 
299:         def suppress(tokens):
300:             for x in itertools.takewhile(lambda x: x[1] != b'def', tokens):
301:                 pass
302:             yield b''
303: 
304:         table = {b'/FontName': replace(fontname),
305:                  b'/ItalicAngle': replace(italicangle),
306:                  b'/FontMatrix': replace(fontmatrix),
307:                  b'/UniqueID': suppress}
308: 
309:         for token, value in tokens:
310:             if token is cls._name and value in table:
311:                 for value in table[value](itertools.chain([(token, value)],
312:                                                           tokens)):
313:                     yield value
314:             else:
315:                 yield value
316: 
317:     def transform(self, effects):
318:         '''
319:         Transform the font by slanting or extending. *effects* should
320:         be a dict where ``effects['slant']`` is the tangent of the
321:         angle that the font is to be slanted to the right (so negative
322:         values slant to the left) and ``effects['extend']`` is the
323:         multiplier by which the font is to be extended (so values less
324:         than 1.0 condense). Returns a new :class:`Type1Font` object.
325:         '''
326:         with io.BytesIO() as buffer:
327:             tokenizer = self._tokens(self.parts[0])
328:             transformed =  self._transformer(tokenizer,
329:                                              slant=effects.get('slant', 0.0),
330:                                              extend=effects.get('extend', 1.0))
331:             list(map(buffer.write, transformed))
332:             return Type1Font((buffer.getvalue(), self.parts[1], self.parts[2]))
333: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_160944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'unicode', u"\nThis module contains a class representing a Type 1 font.\n\nThis version reads pfa and pfb files and splits them for embedding in\npdf files. It also supports SlantFont and ExtendFont transformations,\nsimilarly to pdfTeX and friends. There is no support yet for\nsubsetting.\n\nUsage::\n\n   >>> font = Type1Font(filename)\n   >>> clear_part, encrypted_part, finale = font.parts\n   >>> slanted_font = font.transform({'slant': 0.167})\n   >>> extended_font = font.transform({'extend': 1.2})\n\nSources:\n\n* Adobe Technical Note #5040, Supporting Downloadable PostScript\n  Language Fonts.\n\n* Adobe Type 1 Font Format, Adobe Systems Incorporated, third printing,\n  v1.1, 1993. ISBN 0-201-57044-0.\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import six' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_160945 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'six')

if (type(import_160945) is not StypyTypeError):

    if (import_160945 != 'pyd_module'):
        __import__(import_160945)
        sys_modules_160946 = sys.modules[import_160945]
        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'six', sys_modules_160946.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'six', import_160945)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from six import unichr' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_160947 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'six')

if (type(import_160947) is not StypyTypeError):

    if (import_160947 != 'pyd_module'):
        __import__(import_160947)
        sys_modules_160948 = sys.modules[import_160947]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'six', sys_modules_160948.module_type_store, module_type_store, ['unichr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_160948, sys_modules_160948.module_type_store, module_type_store)
    else:
        from six import unichr

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'six', None, module_type_store, ['unichr'], [unichr])

else:
    # Assigning a type to the variable 'six' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'six', import_160947)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'import binascii' statement (line 31)
import binascii

import_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'binascii', binascii, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'import io' statement (line 32)
import io

import_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'io', io, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import itertools' statement (line 33)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'import numpy' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_160949 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy')

if (type(import_160949) is not StypyTypeError):

    if (import_160949 != 'pyd_module'):
        __import__(import_160949)
        sys_modules_160950 = sys.modules[import_160949]
        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'np', sys_modules_160950.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy', import_160949)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'import re' statement (line 35)
import re

import_module(stypy.reporting.localization.Localization(__file__, 35, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'import struct' statement (line 36)
import struct

import_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'struct', struct, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'import sys' statement (line 37)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'sys', sys, module_type_store)


# Getting the type of 'six' (line 39)
six_160951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 3), 'six')
# Obtaining the member 'PY3' of a type (line 39)
PY3_160952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 3), six_160951, 'PY3')
# Testing the type of an if condition (line 39)
if_condition_160953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 0), PY3_160952)
# Assigning a type to the variable 'if_condition_160953' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'if_condition_160953', if_condition_160953)
# SSA begins for if statement (line 39)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def ord(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ord'
    module_type_store = module_type_store.open_function_context('ord', 40, 4, False)
    
    # Passed parameters checking function
    ord.stypy_localization = localization
    ord.stypy_type_of_self = None
    ord.stypy_type_store = module_type_store
    ord.stypy_function_name = 'ord'
    ord.stypy_param_names_list = ['x']
    ord.stypy_varargs_param_name = None
    ord.stypy_kwargs_param_name = None
    ord.stypy_call_defaults = defaults
    ord.stypy_call_varargs = varargs
    ord.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ord', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ord', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ord(...)' code ##################

    # Getting the type of 'x' (line 41)
    x_160954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', x_160954)
    
    # ################# End of 'ord(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ord' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_160955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_160955)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ord'
    return stypy_return_type_160955

# Assigning a type to the variable 'ord' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'ord', ord)
# SSA join for if statement (line 39)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'Type1Font' class

class Type1Font(object, ):
    unicode_160956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'unicode', u'\n    A class representing a Type-1 font, for use by backends.\n\n    .. attribute:: parts\n\n       A 3-tuple of the cleartext part, the encrypted part, and the\n       finale of zeros.\n\n    .. attribute:: prop\n\n       A dictionary of font properties.\n    ')
    
    # Assigning a Tuple to a Name (line 57):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type1Font.__init__', ['input'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['input'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_160957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'unicode', u'\n        Initialize a Type-1 font. *input* can be either the file name of\n        a pfb file or a 3-tuple of already-decoded Type-1 font parts.\n        ')
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'input' (line 64)
        input_160959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'input', False)
        # Getting the type of 'tuple' (line 64)
        tuple_160960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'tuple', False)
        # Processing the call keyword arguments (line 64)
        kwargs_160961 = {}
        # Getting the type of 'isinstance' (line 64)
        isinstance_160958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 64)
        isinstance_call_result_160962 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), isinstance_160958, *[input_160959, tuple_160960], **kwargs_160961)
        
        
        
        # Call to len(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'input' (line 64)
        input_160964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 44), 'input', False)
        # Processing the call keyword arguments (line 64)
        kwargs_160965 = {}
        # Getting the type of 'len' (line 64)
        len_160963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'len', False)
        # Calling len(args, kwargs) (line 64)
        len_call_result_160966 = invoke(stypy.reporting.localization.Localization(__file__, 64, 40), len_160963, *[input_160964], **kwargs_160965)
        
        int_160967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 54), 'int')
        # Applying the binary operator '==' (line 64)
        result_eq_160968 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 40), '==', len_call_result_160966, int_160967)
        
        # Applying the binary operator 'and' (line 64)
        result_and_keyword_160969 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), 'and', isinstance_call_result_160962, result_eq_160968)
        
        # Testing the type of an if condition (line 64)
        if_condition_160970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_and_keyword_160969)
        # Assigning a type to the variable 'if_condition_160970' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_160970', if_condition_160970)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 65):
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'input' (line 65)
        input_160971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'input')
        # Getting the type of 'self' (line 65)
        self_160972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'self')
        # Setting the type of the member 'parts' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), self_160972, 'parts', input_160971)
        # SSA branch for the else part of an if statement (line 64)
        module_type_store.open_ssa_branch('else')
        
        # Call to open(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'input' (line 67)
        input_160974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'input', False)
        unicode_160975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'unicode', u'rb')
        # Processing the call keyword arguments (line 67)
        kwargs_160976 = {}
        # Getting the type of 'open' (line 67)
        open_160973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'open', False)
        # Calling open(args, kwargs) (line 67)
        open_call_result_160977 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), open_160973, *[input_160974, unicode_160975], **kwargs_160976)
        
        with_160978 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 67, 17), open_call_result_160977, 'with parameter', '__enter__', '__exit__')

        if with_160978:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 67)
            enter___160979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), open_call_result_160977, '__enter__')
            with_enter_160980 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), enter___160979)
            # Assigning a type to the variable 'file' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'file', with_enter_160980)
            
            # Assigning a Call to a Name (line 68):
            
            # Assigning a Call to a Name (line 68):
            
            # Call to _read(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'file' (line 68)
            file_160983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'file', False)
            # Processing the call keyword arguments (line 68)
            kwargs_160984 = {}
            # Getting the type of 'self' (line 68)
            self_160981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'self', False)
            # Obtaining the member '_read' of a type (line 68)
            _read_160982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 23), self_160981, '_read')
            # Calling _read(args, kwargs) (line 68)
            _read_call_result_160985 = invoke(stypy.reporting.localization.Localization(__file__, 68, 23), _read_160982, *[file_160983], **kwargs_160984)
            
            # Assigning a type to the variable 'data' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'data', _read_call_result_160985)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 67)
            exit___160986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), open_call_result_160977, '__exit__')
            with_exit_160987 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), exit___160986, None, None, None)

        
        # Assigning a Call to a Attribute (line 69):
        
        # Assigning a Call to a Attribute (line 69):
        
        # Call to _split(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'data' (line 69)
        data_160990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 37), 'data', False)
        # Processing the call keyword arguments (line 69)
        kwargs_160991 = {}
        # Getting the type of 'self' (line 69)
        self_160988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'self', False)
        # Obtaining the member '_split' of a type (line 69)
        _split_160989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), self_160988, '_split')
        # Calling _split(args, kwargs) (line 69)
        _split_call_result_160992 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), _split_160989, *[data_160990], **kwargs_160991)
        
        # Getting the type of 'self' (line 69)
        self_160993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'self')
        # Setting the type of the member 'parts' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), self_160993, 'parts', _split_call_result_160992)
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _parse(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_160996 = {}
        # Getting the type of 'self' (line 71)
        self_160994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self', False)
        # Obtaining the member '_parse' of a type (line 71)
        _parse_160995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_160994, '_parse')
        # Calling _parse(args, kwargs) (line 71)
        _parse_call_result_160997 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), _parse_160995, *[], **kwargs_160996)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read'
        module_type_store = module_type_store.open_function_context('_read', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type1Font._read.__dict__.__setitem__('stypy_localization', localization)
        Type1Font._read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type1Font._read.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type1Font._read.__dict__.__setitem__('stypy_function_name', 'Type1Font._read')
        Type1Font._read.__dict__.__setitem__('stypy_param_names_list', ['file'])
        Type1Font._read.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type1Font._read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type1Font._read.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type1Font._read.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type1Font._read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type1Font._read.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type1Font._read', ['file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read', localization, ['file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read(...)' code ##################

        unicode_160998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'unicode', u'\n        Read the font from a file, decoding into usable parts.\n        ')
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to read(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_161001 = {}
        # Getting the type of 'file' (line 77)
        file_160999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'file', False)
        # Obtaining the member 'read' of a type (line 77)
        read_161000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 18), file_160999, 'read')
        # Calling read(args, kwargs) (line 77)
        read_call_result_161002 = invoke(stypy.reporting.localization.Localization(__file__, 77, 18), read_161000, *[], **kwargs_161001)
        
        # Assigning a type to the variable 'rawdata' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'rawdata', read_call_result_161002)
        
        
        
        # Call to startswith(...): (line 78)
        # Processing the call arguments (line 78)
        str_161005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'str', '\x80')
        # Processing the call keyword arguments (line 78)
        kwargs_161006 = {}
        # Getting the type of 'rawdata' (line 78)
        rawdata_161003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'rawdata', False)
        # Obtaining the member 'startswith' of a type (line 78)
        startswith_161004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), rawdata_161003, 'startswith')
        # Calling startswith(args, kwargs) (line 78)
        startswith_call_result_161007 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), startswith_161004, *[str_161005], **kwargs_161006)
        
        # Applying the 'not' unary operator (line 78)
        result_not__161008 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), 'not', startswith_call_result_161007)
        
        # Testing the type of an if condition (line 78)
        if_condition_161009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_not__161008)
        # Assigning a type to the variable 'if_condition_161009' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_161009', if_condition_161009)
        # SSA begins for if statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'rawdata' (line 79)
        rawdata_161010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'rawdata')
        # Assigning a type to the variable 'stypy_return_type' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'stypy_return_type', rawdata_161010)
        # SSA join for if statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 81):
        
        # Assigning a Str to a Name (line 81):
        str_161011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 15), 'str', '')
        # Assigning a type to the variable 'data' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'data', str_161011)
        
        
        
        # Call to len(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'rawdata' (line 82)
        rawdata_161013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'rawdata', False)
        # Processing the call keyword arguments (line 82)
        kwargs_161014 = {}
        # Getting the type of 'len' (line 82)
        len_161012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'len', False)
        # Calling len(args, kwargs) (line 82)
        len_call_result_161015 = invoke(stypy.reporting.localization.Localization(__file__, 82, 14), len_161012, *[rawdata_161013], **kwargs_161014)
        
        int_161016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 29), 'int')
        # Applying the binary operator '>' (line 82)
        result_gt_161017 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 14), '>', len_call_result_161015, int_161016)
        
        # Testing the type of an if condition (line 82)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), result_gt_161017)
        # SSA begins for while statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        
        # Call to startswith(...): (line 83)
        # Processing the call arguments (line 83)
        str_161020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 38), 'str', '\x80')
        # Processing the call keyword arguments (line 83)
        kwargs_161021 = {}
        # Getting the type of 'rawdata' (line 83)
        rawdata_161018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'rawdata', False)
        # Obtaining the member 'startswith' of a type (line 83)
        startswith_161019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), rawdata_161018, 'startswith')
        # Calling startswith(args, kwargs) (line 83)
        startswith_call_result_161022 = invoke(stypy.reporting.localization.Localization(__file__, 83, 19), startswith_161019, *[str_161020], **kwargs_161021)
        
        # Applying the 'not' unary operator (line 83)
        result_not__161023 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 15), 'not', startswith_call_result_161022)
        
        # Testing the type of an if condition (line 83)
        if_condition_161024 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 12), result_not__161023)
        # Assigning a type to the variable 'if_condition_161024' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'if_condition_161024', if_condition_161024)
        # SSA begins for if statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 84)
        # Processing the call arguments (line 84)
        unicode_161026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 35), 'unicode', u'Broken pfb file (expected byte 128, got %d)')
        
        # Call to ord(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Obtaining the type of the subscript
        int_161028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 59), 'int')
        # Getting the type of 'rawdata' (line 85)
        rawdata_161029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 51), 'rawdata', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___161030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 51), rawdata_161029, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_161031 = invoke(stypy.reporting.localization.Localization(__file__, 85, 51), getitem___161030, int_161028)
        
        # Processing the call keyword arguments (line 85)
        kwargs_161032 = {}
        # Getting the type of 'ord' (line 85)
        ord_161027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 47), 'ord', False)
        # Calling ord(args, kwargs) (line 85)
        ord_call_result_161033 = invoke(stypy.reporting.localization.Localization(__file__, 85, 47), ord_161027, *[subscript_call_result_161031], **kwargs_161032)
        
        # Applying the binary operator '%' (line 84)
        result_mod_161034 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 35), '%', unicode_161026, ord_call_result_161033)
        
        # Processing the call keyword arguments (line 84)
        kwargs_161035 = {}
        # Getting the type of 'RuntimeError' (line 84)
        RuntimeError_161025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 84)
        RuntimeError_call_result_161036 = invoke(stypy.reporting.localization.Localization(__file__, 84, 22), RuntimeError_161025, *[result_mod_161034], **kwargs_161035)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 84, 16), RuntimeError_call_result_161036, 'raise parameter', BaseException)
        # SSA join for if statement (line 83)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to ord(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Obtaining the type of the subscript
        int_161038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'int')
        # Getting the type of 'rawdata' (line 86)
        rawdata_161039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'rawdata', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___161040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 23), rawdata_161039, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_161041 = invoke(stypy.reporting.localization.Localization(__file__, 86, 23), getitem___161040, int_161038)
        
        # Processing the call keyword arguments (line 86)
        kwargs_161042 = {}
        # Getting the type of 'ord' (line 86)
        ord_161037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'ord', False)
        # Calling ord(args, kwargs) (line 86)
        ord_call_result_161043 = invoke(stypy.reporting.localization.Localization(__file__, 86, 19), ord_161037, *[subscript_call_result_161041], **kwargs_161042)
        
        # Assigning a type to the variable 'type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'type', ord_call_result_161043)
        
        
        # Getting the type of 'type' (line 87)
        type_161044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 87)
        tuple_161045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 87)
        # Adding element type (line 87)
        int_161046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), tuple_161045, int_161046)
        # Adding element type (line 87)
        int_161047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), tuple_161045, int_161047)
        
        # Applying the binary operator 'in' (line 87)
        result_contains_161048 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), 'in', type_161044, tuple_161045)
        
        # Testing the type of an if condition (line 87)
        if_condition_161049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 12), result_contains_161048)
        # Assigning a type to the variable 'if_condition_161049' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'if_condition_161049', if_condition_161049)
        # SSA begins for if statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 88):
        
        # Assigning a Call to a Name:
        
        # Call to unpack(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to str(...): (line 88)
        # Processing the call arguments (line 88)
        unicode_161053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 44), 'unicode', u'<i')
        # Processing the call keyword arguments (line 88)
        kwargs_161054 = {}
        # Getting the type of 'str' (line 88)
        str_161052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 40), 'str', False)
        # Calling str(args, kwargs) (line 88)
        str_call_result_161055 = invoke(stypy.reporting.localization.Localization(__file__, 88, 40), str_161052, *[unicode_161053], **kwargs_161054)
        
        
        # Obtaining the type of the subscript
        int_161056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 59), 'int')
        int_161057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 61), 'int')
        slice_161058 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 88, 51), int_161056, int_161057, None)
        # Getting the type of 'rawdata' (line 88)
        rawdata_161059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 51), 'rawdata', False)
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___161060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 51), rawdata_161059, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_161061 = invoke(stypy.reporting.localization.Localization(__file__, 88, 51), getitem___161060, slice_161058)
        
        # Processing the call keyword arguments (line 88)
        kwargs_161062 = {}
        # Getting the type of 'struct' (line 88)
        struct_161050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'struct', False)
        # Obtaining the member 'unpack' of a type (line 88)
        unpack_161051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 26), struct_161050, 'unpack')
        # Calling unpack(args, kwargs) (line 88)
        unpack_call_result_161063 = invoke(stypy.reporting.localization.Localization(__file__, 88, 26), unpack_161051, *[str_call_result_161055, subscript_call_result_161061], **kwargs_161062)
        
        # Assigning a type to the variable 'call_assignment_160927' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'call_assignment_160927', unpack_call_result_161063)
        
        # Assigning a Call to a Name (line 88):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_161066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 16), 'int')
        # Processing the call keyword arguments
        kwargs_161067 = {}
        # Getting the type of 'call_assignment_160927' (line 88)
        call_assignment_160927_161064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'call_assignment_160927', False)
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___161065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 16), call_assignment_160927_161064, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_161068 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161065, *[int_161066], **kwargs_161067)
        
        # Assigning a type to the variable 'call_assignment_160928' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'call_assignment_160928', getitem___call_result_161068)
        
        # Assigning a Name to a Name (line 88):
        # Getting the type of 'call_assignment_160928' (line 88)
        call_assignment_160928_161069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'call_assignment_160928')
        # Assigning a type to the variable 'length' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'length', call_assignment_160928_161069)
        
        # Assigning a Subscript to a Name (line 89):
        
        # Assigning a Subscript to a Name (line 89):
        
        # Obtaining the type of the subscript
        int_161070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 34), 'int')
        int_161071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'int')
        # Getting the type of 'length' (line 89)
        length_161072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'length')
        # Applying the binary operator '+' (line 89)
        result_add_161073 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 36), '+', int_161071, length_161072)
        
        slice_161074 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 26), int_161070, result_add_161073, None)
        # Getting the type of 'rawdata' (line 89)
        rawdata_161075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'rawdata')
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___161076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 26), rawdata_161075, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_161077 = invoke(stypy.reporting.localization.Localization(__file__, 89, 26), getitem___161076, slice_161074)
        
        # Assigning a type to the variable 'segment' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'segment', subscript_call_result_161077)
        
        # Assigning a Subscript to a Name (line 90):
        
        # Assigning a Subscript to a Name (line 90):
        
        # Obtaining the type of the subscript
        int_161078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 34), 'int')
        # Getting the type of 'length' (line 90)
        length_161079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 38), 'length')
        # Applying the binary operator '+' (line 90)
        result_add_161080 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 34), '+', int_161078, length_161079)
        
        slice_161081 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 90, 26), result_add_161080, None, None)
        # Getting the type of 'rawdata' (line 90)
        rawdata_161082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'rawdata')
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___161083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 26), rawdata_161082, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_161084 = invoke(stypy.reporting.localization.Localization(__file__, 90, 26), getitem___161083, slice_161081)
        
        # Assigning a type to the variable 'rawdata' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'rawdata', subscript_call_result_161084)
        # SSA join for if statement (line 87)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'type' (line 92)
        type_161085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'type')
        int_161086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'int')
        # Applying the binary operator '==' (line 92)
        result_eq_161087 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 15), '==', type_161085, int_161086)
        
        # Testing the type of an if condition (line 92)
        if_condition_161088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 12), result_eq_161087)
        # Assigning a type to the variable 'if_condition_161088' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'if_condition_161088', if_condition_161088)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'data' (line 93)
        data_161089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'data')
        # Getting the type of 'segment' (line 93)
        segment_161090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'segment')
        # Applying the binary operator '+=' (line 93)
        result_iadd_161091 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 16), '+=', data_161089, segment_161090)
        # Assigning a type to the variable 'data' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'data', result_iadd_161091)
        
        # SSA branch for the else part of an if statement (line 92)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'type' (line 94)
        type_161092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'type')
        int_161093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'int')
        # Applying the binary operator '==' (line 94)
        result_eq_161094 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 17), '==', type_161092, int_161093)
        
        # Testing the type of an if condition (line 94)
        if_condition_161095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 17), result_eq_161094)
        # Assigning a type to the variable 'if_condition_161095' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'if_condition_161095', if_condition_161095)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'data' (line 95)
        data_161096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'data')
        
        # Call to hexlify(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'segment' (line 95)
        segment_161099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 41), 'segment', False)
        # Processing the call keyword arguments (line 95)
        kwargs_161100 = {}
        # Getting the type of 'binascii' (line 95)
        binascii_161097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'binascii', False)
        # Obtaining the member 'hexlify' of a type (line 95)
        hexlify_161098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 24), binascii_161097, 'hexlify')
        # Calling hexlify(args, kwargs) (line 95)
        hexlify_call_result_161101 = invoke(stypy.reporting.localization.Localization(__file__, 95, 24), hexlify_161098, *[segment_161099], **kwargs_161100)
        
        # Applying the binary operator '+=' (line 95)
        result_iadd_161102 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 16), '+=', data_161096, hexlify_call_result_161101)
        # Assigning a type to the variable 'data' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'data', result_iadd_161102)
        
        # SSA branch for the else part of an if statement (line 94)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'type' (line 96)
        type_161103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'type')
        int_161104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 25), 'int')
        # Applying the binary operator '==' (line 96)
        result_eq_161105 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 17), '==', type_161103, int_161104)
        
        # Testing the type of an if condition (line 96)
        if_condition_161106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 17), result_eq_161105)
        # Assigning a type to the variable 'if_condition_161106' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'if_condition_161106', if_condition_161106)
        # SSA begins for if statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA branch for the else part of an if statement (line 96)
        module_type_store.open_ssa_branch('else')
        
        # Call to RuntimeError(...): (line 99)
        # Processing the call arguments (line 99)
        unicode_161108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 35), 'unicode', u'Unknown segment type %d in pfb file')
        # Getting the type of 'type' (line 100)
        type_161109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 35), 'type', False)
        # Applying the binary operator '%' (line 99)
        result_mod_161110 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 35), '%', unicode_161108, type_161109)
        
        # Processing the call keyword arguments (line 99)
        kwargs_161111 = {}
        # Getting the type of 'RuntimeError' (line 99)
        RuntimeError_161107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 99)
        RuntimeError_call_result_161112 = invoke(stypy.reporting.localization.Localization(__file__, 99, 22), RuntimeError_161107, *[result_mod_161110], **kwargs_161111)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 99, 16), RuntimeError_call_result_161112, 'raise parameter', BaseException)
        # SSA join for if statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'data' (line 102)
        data_161113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'data')
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'stypy_return_type', data_161113)
        
        # ################# End of '_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_161114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161114)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read'
        return stypy_return_type_161114


    @norecursion
    def _split(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_split'
        module_type_store = module_type_store.open_function_context('_split', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type1Font._split.__dict__.__setitem__('stypy_localization', localization)
        Type1Font._split.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type1Font._split.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type1Font._split.__dict__.__setitem__('stypy_function_name', 'Type1Font._split')
        Type1Font._split.__dict__.__setitem__('stypy_param_names_list', ['data'])
        Type1Font._split.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type1Font._split.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type1Font._split.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type1Font._split.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type1Font._split.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type1Font._split.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type1Font._split', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_split', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_split(...)' code ##################

        unicode_161115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, (-1)), 'unicode', u'\n        Split the Type 1 font into its three main parts.\n\n        The three parts are: (1) the cleartext part, which ends in a\n        eexec operator; (2) the encrypted part; (3) the fixed part,\n        which contains 512 ASCII zeros possibly divided on various\n        lines, a cleartomark operator, and possibly something else.\n        ')
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to index(...): (line 115)
        # Processing the call arguments (line 115)
        str_161118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 25), 'str', 'eexec')
        # Processing the call keyword arguments (line 115)
        kwargs_161119 = {}
        # Getting the type of 'data' (line 115)
        data_161116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 14), 'data', False)
        # Obtaining the member 'index' of a type (line 115)
        index_161117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 14), data_161116, 'index')
        # Calling index(args, kwargs) (line 115)
        index_call_result_161120 = invoke(stypy.reporting.localization.Localization(__file__, 115, 14), index_161117, *[str_161118], **kwargs_161119)
        
        # Assigning a type to the variable 'idx' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'idx', index_call_result_161120)
        
        # Getting the type of 'idx' (line 116)
        idx_161121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'idx')
        
        # Call to len(...): (line 116)
        # Processing the call arguments (line 116)
        str_161123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'str', 'eexec')
        # Processing the call keyword arguments (line 116)
        kwargs_161124 = {}
        # Getting the type of 'len' (line 116)
        len_161122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'len', False)
        # Calling len(args, kwargs) (line 116)
        len_call_result_161125 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), len_161122, *[str_161123], **kwargs_161124)
        
        # Applying the binary operator '+=' (line 116)
        result_iadd_161126 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 8), '+=', idx_161121, len_call_result_161125)
        # Assigning a type to the variable 'idx' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'idx', result_iadd_161126)
        
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 117)
        idx_161127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 19), 'idx')
        # Getting the type of 'data' (line 117)
        data_161128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 14), 'data')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___161129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 14), data_161128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_161130 = invoke(stypy.reporting.localization.Localization(__file__, 117, 14), getitem___161129, idx_161127)
        
        str_161131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 27), 'str', ' \t\r\n')
        # Applying the binary operator 'in' (line 117)
        result_contains_161132 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 14), 'in', subscript_call_result_161130, str_161131)
        
        # Testing the type of an if condition (line 117)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 8), result_contains_161132)
        # SSA begins for while statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'idx' (line 118)
        idx_161133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'idx')
        int_161134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 19), 'int')
        # Applying the binary operator '+=' (line 118)
        result_iadd_161135 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 12), '+=', idx_161133, int_161134)
        # Assigning a type to the variable 'idx' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'idx', result_iadd_161135)
        
        # SSA join for while statement (line 117)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 119):
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'idx' (line 119)
        idx_161136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'idx')
        # Assigning a type to the variable 'len1' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'len1', idx_161136)
        
        # Assigning a BinOp to a Name (line 123):
        
        # Assigning a BinOp to a Name (line 123):
        
        # Call to rindex(...): (line 123)
        # Processing the call arguments (line 123)
        str_161139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 26), 'str', 'cleartomark')
        # Processing the call keyword arguments (line 123)
        kwargs_161140 = {}
        # Getting the type of 'data' (line 123)
        data_161137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'data', False)
        # Obtaining the member 'rindex' of a type (line 123)
        rindex_161138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 14), data_161137, 'rindex')
        # Calling rindex(args, kwargs) (line 123)
        rindex_call_result_161141 = invoke(stypy.reporting.localization.Localization(__file__, 123, 14), rindex_161138, *[str_161139], **kwargs_161140)
        
        int_161142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 44), 'int')
        # Applying the binary operator '-' (line 123)
        result_sub_161143 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 14), '-', rindex_call_result_161141, int_161142)
        
        # Assigning a type to the variable 'idx' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'idx', result_sub_161143)
        
        # Assigning a Num to a Name (line 124):
        
        # Assigning a Num to a Name (line 124):
        int_161144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 16), 'int')
        # Assigning a type to the variable 'zeros' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'zeros', int_161144)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'zeros' (line 125)
        zeros_161145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 14), 'zeros')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 125)
        idx_161146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'idx')
        # Getting the type of 'data' (line 125)
        data_161147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'data')
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___161148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 24), data_161147, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_161149 = invoke(stypy.reporting.localization.Localization(__file__, 125, 24), getitem___161148, idx_161146)
        
        str_161150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 37), 'str', '0')
        # Applying the binary operator 'in' (line 125)
        result_contains_161151 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 24), 'in', subscript_call_result_161149, str_161150)
        
        # Applying the binary operator 'and' (line 125)
        result_and_keyword_161152 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 14), 'and', zeros_161145, result_contains_161151)
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 125)
        idx_161153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 50), 'idx')
        # Getting the type of 'data' (line 125)
        data_161154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 45), 'data')
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___161155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 45), data_161154, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_161156 = invoke(stypy.reporting.localization.Localization(__file__, 125, 45), getitem___161155, idx_161153)
        
        str_161157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 58), 'str', '\r\n')
        # Applying the binary operator 'in' (line 125)
        result_contains_161158 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 45), 'in', subscript_call_result_161156, str_161157)
        
        # Applying the binary operator 'or' (line 125)
        result_or_keyword_161159 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 14), 'or', result_and_keyword_161152, result_contains_161158)
        
        # Testing the type of an if condition (line 125)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_or_keyword_161159)
        # SSA begins for while statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 126)
        idx_161160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'idx')
        # Getting the type of 'data' (line 126)
        data_161161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'data')
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___161162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 15), data_161161, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_161163 = invoke(stypy.reporting.localization.Localization(__file__, 126, 15), getitem___161162, idx_161160)
        
        str_161164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 28), 'str', '0')
        # Applying the binary operator 'in' (line 126)
        result_contains_161165 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 15), 'in', subscript_call_result_161163, str_161164)
        
        # Testing the type of an if condition (line 126)
        if_condition_161166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 12), result_contains_161165)
        # Assigning a type to the variable 'if_condition_161166' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'if_condition_161166', if_condition_161166)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'zeros' (line 127)
        zeros_161167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'zeros')
        int_161168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 25), 'int')
        # Applying the binary operator '-=' (line 127)
        result_isub_161169 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 16), '-=', zeros_161167, int_161168)
        # Assigning a type to the variable 'zeros' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'zeros', result_isub_161169)
        
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'idx' (line 128)
        idx_161170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'idx')
        int_161171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 19), 'int')
        # Applying the binary operator '-=' (line 128)
        result_isub_161172 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 12), '-=', idx_161170, int_161171)
        # Assigning a type to the variable 'idx' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'idx', result_isub_161172)
        
        # SSA join for while statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'zeros' (line 129)
        zeros_161173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'zeros')
        # Testing the type of an if condition (line 129)
        if_condition_161174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 8), zeros_161173)
        # Assigning a type to the variable 'if_condition_161174' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'if_condition_161174', if_condition_161174)
        # SSA begins for if statement (line 129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 130)
        # Processing the call arguments (line 130)
        unicode_161176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 31), 'unicode', u'Insufficiently many zeros in Type 1 font')
        # Processing the call keyword arguments (line 130)
        kwargs_161177 = {}
        # Getting the type of 'RuntimeError' (line 130)
        RuntimeError_161175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 130)
        RuntimeError_call_result_161178 = invoke(stypy.reporting.localization.Localization(__file__, 130, 18), RuntimeError_161175, *[unicode_161176], **kwargs_161177)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 130, 12), RuntimeError_call_result_161178, 'raise parameter', BaseException)
        # SSA join for if statement (line 129)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to unhexlify(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining the type of the subscript
        # Getting the type of 'len1' (line 137)
        len1_161181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 41), 'len1', False)
        # Getting the type of 'idx' (line 137)
        idx_161182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 46), 'idx', False)
        int_161183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 50), 'int')
        # Applying the binary operator '+' (line 137)
        result_add_161184 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 46), '+', idx_161182, int_161183)
        
        slice_161185 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 137, 36), len1_161181, result_add_161184, None)
        # Getting the type of 'data' (line 137)
        data_161186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___161187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 36), data_161186, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_161188 = invoke(stypy.reporting.localization.Localization(__file__, 137, 36), getitem___161187, slice_161185)
        
        # Processing the call keyword arguments (line 137)
        kwargs_161189 = {}
        # Getting the type of 'binascii' (line 137)
        binascii_161179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'binascii', False)
        # Obtaining the member 'unhexlify' of a type (line 137)
        unhexlify_161180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 17), binascii_161179, 'unhexlify')
        # Calling unhexlify(args, kwargs) (line 137)
        unhexlify_call_result_161190 = invoke(stypy.reporting.localization.Localization(__file__, 137, 17), unhexlify_161180, *[subscript_call_result_161188], **kwargs_161189)
        
        # Assigning a type to the variable 'binary' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'binary', unhexlify_call_result_161190)
        
        # Obtaining an instance of the builtin type 'tuple' (line 139)
        tuple_161191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 139)
        # Adding element type (line 139)
        
        # Obtaining the type of the subscript
        # Getting the type of 'len1' (line 139)
        len1_161192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'len1')
        slice_161193 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 139, 15), None, len1_161192, None)
        # Getting the type of 'data' (line 139)
        data_161194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'data')
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___161195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), data_161194, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_161196 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), getitem___161195, slice_161193)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 15), tuple_161191, subscript_call_result_161196)
        # Adding element type (line 139)
        # Getting the type of 'binary' (line 139)
        binary_161197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'binary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 15), tuple_161191, binary_161197)
        # Adding element type (line 139)
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 139)
        idx_161198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 41), 'idx')
        int_161199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 45), 'int')
        # Applying the binary operator '+' (line 139)
        result_add_161200 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 41), '+', idx_161198, int_161199)
        
        slice_161201 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 139, 36), result_add_161200, None, None)
        # Getting the type of 'data' (line 139)
        data_161202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 36), 'data')
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___161203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 36), data_161202, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_161204 = invoke(stypy.reporting.localization.Localization(__file__, 139, 36), getitem___161203, slice_161201)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 15), tuple_161191, subscript_call_result_161204)
        
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type', tuple_161191)
        
        # ################# End of '_split(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_split' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_161205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161205)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_split'
        return stypy_return_type_161205

    
    # Assigning a Call to a Name (line 141):
    
    # Assigning a Call to a Name (line 142):
    
    # Assigning a Call to a Name (line 143):
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 150):
    
    # Assigning a Call to a Name (line 151):

    @norecursion
    def _tokens(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_tokens'
        module_type_store = module_type_store.open_function_context('_tokens', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type1Font._tokens.__dict__.__setitem__('stypy_localization', localization)
        Type1Font._tokens.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type1Font._tokens.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type1Font._tokens.__dict__.__setitem__('stypy_function_name', 'Type1Font._tokens')
        Type1Font._tokens.__dict__.__setitem__('stypy_param_names_list', ['text'])
        Type1Font._tokens.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type1Font._tokens.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type1Font._tokens.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type1Font._tokens.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type1Font._tokens.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type1Font._tokens.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type1Font._tokens', ['text'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_tokens', localization, ['text'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_tokens(...)' code ##################

        unicode_161206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'unicode', u"\n        A PostScript tokenizer. Yield (token, value) pairs such as\n        (cls._whitespace, '   ') or (cls._name, '/Foobar').\n        ")
        
        # Assigning a Num to a Name (line 159):
        
        # Assigning a Num to a Name (line 159):
        int_161207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 14), 'int')
        # Assigning a type to the variable 'pos' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'pos', int_161207)
        
        
        # Getting the type of 'pos' (line 160)
        pos_161208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 14), 'pos')
        
        # Call to len(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'text' (line 160)
        text_161210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'text', False)
        # Processing the call keyword arguments (line 160)
        kwargs_161211 = {}
        # Getting the type of 'len' (line 160)
        len_161209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'len', False)
        # Calling len(args, kwargs) (line 160)
        len_call_result_161212 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), len_161209, *[text_161210], **kwargs_161211)
        
        # Applying the binary operator '<' (line 160)
        result_lt_161213 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 14), '<', pos_161208, len_call_result_161212)
        
        # Testing the type of an if condition (line 160)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_lt_161213)
        # SSA begins for while statement (line 160)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BoolOp to a Name (line 161):
        
        # Assigning a BoolOp to a Name (line 161):
        
        # Evaluating a boolean operation
        
        # Call to match(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 161)
        pos_161217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 48), 'pos', False)
        slice_161218 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 161, 43), pos_161217, None, None)
        # Getting the type of 'text' (line 161)
        text_161219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 43), 'text', False)
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___161220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 43), text_161219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_161221 = invoke(stypy.reporting.localization.Localization(__file__, 161, 43), getitem___161220, slice_161218)
        
        # Processing the call keyword arguments (line 161)
        kwargs_161222 = {}
        # Getting the type of 'cls' (line 161)
        cls_161214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 21), 'cls', False)
        # Obtaining the member '_comment_re' of a type (line 161)
        _comment_re_161215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 21), cls_161214, '_comment_re')
        # Obtaining the member 'match' of a type (line 161)
        match_161216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 21), _comment_re_161215, 'match')
        # Calling match(args, kwargs) (line 161)
        match_call_result_161223 = invoke(stypy.reporting.localization.Localization(__file__, 161, 21), match_161216, *[subscript_call_result_161221], **kwargs_161222)
        
        
        # Call to match(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 162)
        pos_161227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 51), 'pos', False)
        slice_161228 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 162, 46), pos_161227, None, None)
        # Getting the type of 'text' (line 162)
        text_161229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 46), 'text', False)
        # Obtaining the member '__getitem__' of a type (line 162)
        getitem___161230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 46), text_161229, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 162)
        subscript_call_result_161231 = invoke(stypy.reporting.localization.Localization(__file__, 162, 46), getitem___161230, slice_161228)
        
        # Processing the call keyword arguments (line 162)
        kwargs_161232 = {}
        # Getting the type of 'cls' (line 162)
        cls_161224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'cls', False)
        # Obtaining the member '_whitespace_re' of a type (line 162)
        _whitespace_re_161225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 21), cls_161224, '_whitespace_re')
        # Obtaining the member 'match' of a type (line 162)
        match_161226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 21), _whitespace_re_161225, 'match')
        # Calling match(args, kwargs) (line 162)
        match_call_result_161233 = invoke(stypy.reporting.localization.Localization(__file__, 162, 21), match_161226, *[subscript_call_result_161231], **kwargs_161232)
        
        # Applying the binary operator 'or' (line 161)
        result_or_keyword_161234 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 21), 'or', match_call_result_161223, match_call_result_161233)
        
        # Assigning a type to the variable 'match' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'match', result_or_keyword_161234)
        
        # Getting the type of 'match' (line 163)
        match_161235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'match')
        # Testing the type of an if condition (line 163)
        if_condition_161236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 12), match_161235)
        # Assigning a type to the variable 'if_condition_161236' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'if_condition_161236', if_condition_161236)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 164)
        tuple_161237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 164)
        # Adding element type (line 164)
        # Getting the type of 'cls' (line 164)
        cls_161238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'cls')
        # Obtaining the member '_whitespace' of a type (line 164)
        _whitespace_161239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 23), cls_161238, '_whitespace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 23), tuple_161237, _whitespace_161239)
        # Adding element type (line 164)
        
        # Call to group(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_161242 = {}
        # Getting the type of 'match' (line 164)
        match_161240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 40), 'match', False)
        # Obtaining the member 'group' of a type (line 164)
        group_161241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 40), match_161240, 'group')
        # Calling group(args, kwargs) (line 164)
        group_call_result_161243 = invoke(stypy.reporting.localization.Localization(__file__, 164, 40), group_161241, *[], **kwargs_161242)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 23), tuple_161237, group_call_result_161243)
        
        GeneratorType_161244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 16), GeneratorType_161244, tuple_161237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'stypy_return_type', GeneratorType_161244)
        
        # Getting the type of 'pos' (line 165)
        pos_161245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'pos')
        
        # Call to end(...): (line 165)
        # Processing the call keyword arguments (line 165)
        kwargs_161248 = {}
        # Getting the type of 'match' (line 165)
        match_161246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'match', False)
        # Obtaining the member 'end' of a type (line 165)
        end_161247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 23), match_161246, 'end')
        # Calling end(args, kwargs) (line 165)
        end_call_result_161249 = invoke(stypy.reporting.localization.Localization(__file__, 165, 23), end_161247, *[], **kwargs_161248)
        
        # Applying the binary operator '+=' (line 165)
        result_iadd_161250 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 16), '+=', pos_161245, end_call_result_161249)
        # Assigning a type to the variable 'pos' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'pos', result_iadd_161250)
        
        # SSA branch for the else part of an if statement (line 163)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 166)
        pos_161251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'pos')
        # Getting the type of 'text' (line 166)
        text_161252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'text')
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___161253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 17), text_161252, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_161254 = invoke(stypy.reporting.localization.Localization(__file__, 166, 17), getitem___161253, pos_161251)
        
        str_161255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'str', '(')
        # Applying the binary operator '==' (line 166)
        result_eq_161256 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 17), '==', subscript_call_result_161254, str_161255)
        
        # Testing the type of an if condition (line 166)
        if_condition_161257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 17), result_eq_161256)
        # Assigning a type to the variable 'if_condition_161257' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'if_condition_161257', if_condition_161257)
        # SSA begins for if statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 167):
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'pos' (line 167)
        pos_161258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'pos')
        # Assigning a type to the variable 'start' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'start', pos_161258)
        
        # Getting the type of 'pos' (line 168)
        pos_161259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'pos')
        int_161260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 23), 'int')
        # Applying the binary operator '+=' (line 168)
        result_iadd_161261 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 16), '+=', pos_161259, int_161260)
        # Assigning a type to the variable 'pos' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'pos', result_iadd_161261)
        
        
        # Assigning a Num to a Name (line 169):
        
        # Assigning a Num to a Name (line 169):
        int_161262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'int')
        # Assigning a type to the variable 'depth' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'depth', int_161262)
        
        # Getting the type of 'depth' (line 170)
        depth_161263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 22), 'depth')
        # Testing the type of an if condition (line 170)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 16), depth_161263)
        # SSA begins for while statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to search(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 171)
        pos_161267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 57), 'pos', False)
        slice_161268 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 171, 52), pos_161267, None, None)
        # Getting the type of 'text' (line 171)
        text_161269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 52), 'text', False)
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___161270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 52), text_161269, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_161271 = invoke(stypy.reporting.localization.Localization(__file__, 171, 52), getitem___161270, slice_161268)
        
        # Processing the call keyword arguments (line 171)
        kwargs_161272 = {}
        # Getting the type of 'cls' (line 171)
        cls_161264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 28), 'cls', False)
        # Obtaining the member '_instring_re' of a type (line 171)
        _instring_re_161265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 28), cls_161264, '_instring_re')
        # Obtaining the member 'search' of a type (line 171)
        search_161266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 28), _instring_re_161265, 'search')
        # Calling search(args, kwargs) (line 171)
        search_call_result_161273 = invoke(stypy.reporting.localization.Localization(__file__, 171, 28), search_161266, *[subscript_call_result_161271], **kwargs_161272)
        
        # Assigning a type to the variable 'match' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'match', search_call_result_161273)
        
        # Type idiom detected: calculating its left and rigth part (line 172)
        # Getting the type of 'match' (line 172)
        match_161274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'match')
        # Getting the type of 'None' (line 172)
        None_161275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'None')
        
        (may_be_161276, more_types_in_union_161277) = may_be_none(match_161274, None_161275)

        if may_be_161276:

            if more_types_in_union_161277:
                # Runtime conditional SSA (line 172)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'stypy_return_type', types.NoneType)

            if more_types_in_union_161277:
                # SSA join for if statement (line 172)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'pos' (line 174)
        pos_161278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'pos')
        
        # Call to end(...): (line 174)
        # Processing the call keyword arguments (line 174)
        kwargs_161281 = {}
        # Getting the type of 'match' (line 174)
        match_161279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'match', False)
        # Obtaining the member 'end' of a type (line 174)
        end_161280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 27), match_161279, 'end')
        # Calling end(args, kwargs) (line 174)
        end_call_result_161282 = invoke(stypy.reporting.localization.Localization(__file__, 174, 27), end_161280, *[], **kwargs_161281)
        
        # Applying the binary operator '+=' (line 174)
        result_iadd_161283 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 20), '+=', pos_161278, end_call_result_161282)
        # Assigning a type to the variable 'pos' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'pos', result_iadd_161283)
        
        
        
        
        # Call to group(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_161286 = {}
        # Getting the type of 'match' (line 175)
        match_161284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'match', False)
        # Obtaining the member 'group' of a type (line 175)
        group_161285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 23), match_161284, 'group')
        # Calling group(args, kwargs) (line 175)
        group_call_result_161287 = invoke(stypy.reporting.localization.Localization(__file__, 175, 23), group_161285, *[], **kwargs_161286)
        
        str_161288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 40), 'str', '(')
        # Applying the binary operator '==' (line 175)
        result_eq_161289 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 23), '==', group_call_result_161287, str_161288)
        
        # Testing the type of an if condition (line 175)
        if_condition_161290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 20), result_eq_161289)
        # Assigning a type to the variable 'if_condition_161290' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'if_condition_161290', if_condition_161290)
        # SSA begins for if statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'depth' (line 176)
        depth_161291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'depth')
        int_161292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 33), 'int')
        # Applying the binary operator '+=' (line 176)
        result_iadd_161293 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 24), '+=', depth_161291, int_161292)
        # Assigning a type to the variable 'depth' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'depth', result_iadd_161293)
        
        # SSA branch for the else part of an if statement (line 175)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to group(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_161296 = {}
        # Getting the type of 'match' (line 177)
        match_161294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'match', False)
        # Obtaining the member 'group' of a type (line 177)
        group_161295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 25), match_161294, 'group')
        # Calling group(args, kwargs) (line 177)
        group_call_result_161297 = invoke(stypy.reporting.localization.Localization(__file__, 177, 25), group_161295, *[], **kwargs_161296)
        
        str_161298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 42), 'str', ')')
        # Applying the binary operator '==' (line 177)
        result_eq_161299 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 25), '==', group_call_result_161297, str_161298)
        
        # Testing the type of an if condition (line 177)
        if_condition_161300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 25), result_eq_161299)
        # Assigning a type to the variable 'if_condition_161300' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'if_condition_161300', if_condition_161300)
        # SSA begins for if statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'depth' (line 178)
        depth_161301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'depth')
        int_161302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 33), 'int')
        # Applying the binary operator '-=' (line 178)
        result_isub_161303 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 24), '-=', depth_161301, int_161302)
        # Assigning a type to the variable 'depth' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'depth', result_isub_161303)
        
        # SSA branch for the else part of an if statement (line 177)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'pos' (line 180)
        pos_161304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'pos')
        int_161305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'int')
        # Applying the binary operator '+=' (line 180)
        result_iadd_161306 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 24), '+=', pos_161304, int_161305)
        # Assigning a type to the variable 'pos' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'pos', result_iadd_161306)
        
        # SSA join for if statement (line 177)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 175)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 181)
        tuple_161307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 181)
        # Adding element type (line 181)
        # Getting the type of 'cls' (line 181)
        cls_161308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 23), 'cls')
        # Obtaining the member '_string' of a type (line 181)
        _string_161309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 23), cls_161308, '_string')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 23), tuple_161307, _string_161309)
        # Adding element type (line 181)
        
        # Obtaining the type of the subscript
        # Getting the type of 'start' (line 181)
        start_161310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 41), 'start')
        # Getting the type of 'pos' (line 181)
        pos_161311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 47), 'pos')
        slice_161312 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 181, 36), start_161310, pos_161311, None)
        # Getting the type of 'text' (line 181)
        text_161313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'text')
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___161314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 36), text_161313, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_161315 = invoke(stypy.reporting.localization.Localization(__file__, 181, 36), getitem___161314, slice_161312)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 23), tuple_161307, subscript_call_result_161315)
        
        GeneratorType_161316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 16), GeneratorType_161316, tuple_161307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'stypy_return_type', GeneratorType_161316)
        # SSA branch for the else part of an if statement (line 166)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 182)
        pos_161317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'pos')
        # Getting the type of 'pos' (line 182)
        pos_161318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 'pos')
        int_161319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 32), 'int')
        # Applying the binary operator '+' (line 182)
        result_add_161320 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 26), '+', pos_161318, int_161319)
        
        slice_161321 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 182, 17), pos_161317, result_add_161320, None)
        # Getting the type of 'text' (line 182)
        text_161322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'text')
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___161323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 17), text_161322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_161324 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), getitem___161323, slice_161321)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 182)
        tuple_161325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 182)
        # Adding element type (line 182)
        str_161326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 39), 'str', '<<')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 39), tuple_161325, str_161326)
        # Adding element type (line 182)
        str_161327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 46), 'str', '>>')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 39), tuple_161325, str_161327)
        
        # Applying the binary operator 'in' (line 182)
        result_contains_161328 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 17), 'in', subscript_call_result_161324, tuple_161325)
        
        # Testing the type of an if condition (line 182)
        if_condition_161329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 17), result_contains_161328)
        # Assigning a type to the variable 'if_condition_161329' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'if_condition_161329', if_condition_161329)
        # SSA begins for if statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 183)
        tuple_161330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 183)
        # Adding element type (line 183)
        # Getting the type of 'cls' (line 183)
        cls_161331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 'cls')
        # Obtaining the member '_delimiter' of a type (line 183)
        _delimiter_161332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 23), cls_161331, '_delimiter')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 23), tuple_161330, _delimiter_161332)
        # Adding element type (line 183)
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 183)
        pos_161333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 44), 'pos')
        # Getting the type of 'pos' (line 183)
        pos_161334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 48), 'pos')
        int_161335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 54), 'int')
        # Applying the binary operator '+' (line 183)
        result_add_161336 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 48), '+', pos_161334, int_161335)
        
        slice_161337 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 183, 39), pos_161333, result_add_161336, None)
        # Getting the type of 'text' (line 183)
        text_161338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 39), 'text')
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___161339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 39), text_161338, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_161340 = invoke(stypy.reporting.localization.Localization(__file__, 183, 39), getitem___161339, slice_161337)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 23), tuple_161330, subscript_call_result_161340)
        
        GeneratorType_161341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 16), GeneratorType_161341, tuple_161330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'stypy_return_type', GeneratorType_161341)
        
        # Getting the type of 'pos' (line 184)
        pos_161342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'pos')
        int_161343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 23), 'int')
        # Applying the binary operator '+=' (line 184)
        result_iadd_161344 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 16), '+=', pos_161342, int_161343)
        # Assigning a type to the variable 'pos' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'pos', result_iadd_161344)
        
        # SSA branch for the else part of an if statement (line 182)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 185)
        pos_161345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 22), 'pos')
        # Getting the type of 'text' (line 185)
        text_161346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'text')
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___161347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 17), text_161346, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_161348 = invoke(stypy.reporting.localization.Localization(__file__, 185, 17), getitem___161347, pos_161345)
        
        str_161349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'str', '<')
        # Applying the binary operator '==' (line 185)
        result_eq_161350 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 17), '==', subscript_call_result_161348, str_161349)
        
        # Testing the type of an if condition (line 185)
        if_condition_161351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 17), result_eq_161350)
        # Assigning a type to the variable 'if_condition_161351' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'if_condition_161351', if_condition_161351)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 186):
        
        # Assigning a Name to a Name (line 186):
        # Getting the type of 'pos' (line 186)
        pos_161352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'pos')
        # Assigning a type to the variable 'start' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'start', pos_161352)
        
        # Getting the type of 'pos' (line 187)
        pos_161353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'pos')
        
        # Call to index(...): (line 187)
        # Processing the call arguments (line 187)
        str_161360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 40), 'str', '>')
        # Processing the call keyword arguments (line 187)
        kwargs_161361 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 187)
        pos_161354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'pos', False)
        slice_161355 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 23), pos_161354, None, None)
        # Getting the type of 'text' (line 187)
        text_161356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'text', False)
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___161357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 23), text_161356, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_161358 = invoke(stypy.reporting.localization.Localization(__file__, 187, 23), getitem___161357, slice_161355)
        
        # Obtaining the member 'index' of a type (line 187)
        index_161359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 23), subscript_call_result_161358, 'index')
        # Calling index(args, kwargs) (line 187)
        index_call_result_161362 = invoke(stypy.reporting.localization.Localization(__file__, 187, 23), index_161359, *[str_161360], **kwargs_161361)
        
        # Applying the binary operator '+=' (line 187)
        result_iadd_161363 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 16), '+=', pos_161353, index_call_result_161362)
        # Assigning a type to the variable 'pos' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'pos', result_iadd_161363)
        
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 188)
        tuple_161364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 188)
        # Adding element type (line 188)
        # Getting the type of 'cls' (line 188)
        cls_161365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'cls')
        # Obtaining the member '_string' of a type (line 188)
        _string_161366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 23), cls_161365, '_string')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 23), tuple_161364, _string_161366)
        # Adding element type (line 188)
        
        # Obtaining the type of the subscript
        # Getting the type of 'start' (line 188)
        start_161367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 41), 'start')
        # Getting the type of 'pos' (line 188)
        pos_161368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 47), 'pos')
        slice_161369 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 188, 36), start_161367, pos_161368, None)
        # Getting the type of 'text' (line 188)
        text_161370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 36), 'text')
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___161371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 36), text_161370, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_161372 = invoke(stypy.reporting.localization.Localization(__file__, 188, 36), getitem___161371, slice_161369)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 23), tuple_161364, subscript_call_result_161372)
        
        GeneratorType_161373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 16), GeneratorType_161373, tuple_161364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'stypy_return_type', GeneratorType_161373)
        # SSA branch for the else part of an if statement (line 185)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to match(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 190)
        pos_161377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 49), 'pos', False)
        slice_161378 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 190, 44), pos_161377, None, None)
        # Getting the type of 'text' (line 190)
        text_161379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 44), 'text', False)
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___161380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 44), text_161379, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_161381 = invoke(stypy.reporting.localization.Localization(__file__, 190, 44), getitem___161380, slice_161378)
        
        # Processing the call keyword arguments (line 190)
        kwargs_161382 = {}
        # Getting the type of 'cls' (line 190)
        cls_161374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 'cls', False)
        # Obtaining the member '_token_re' of a type (line 190)
        _token_re_161375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 24), cls_161374, '_token_re')
        # Obtaining the member 'match' of a type (line 190)
        match_161376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 24), _token_re_161375, 'match')
        # Calling match(args, kwargs) (line 190)
        match_call_result_161383 = invoke(stypy.reporting.localization.Localization(__file__, 190, 24), match_161376, *[subscript_call_result_161381], **kwargs_161382)
        
        # Assigning a type to the variable 'match' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'match', match_call_result_161383)
        
        # Getting the type of 'match' (line 191)
        match_161384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), 'match')
        # Testing the type of an if condition (line 191)
        if_condition_161385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 16), match_161384)
        # Assigning a type to the variable 'if_condition_161385' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'if_condition_161385', if_condition_161385)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to float(...): (line 193)
        # Processing the call arguments (line 193)
        
        # Call to group(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_161389 = {}
        # Getting the type of 'match' (line 193)
        match_161387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 30), 'match', False)
        # Obtaining the member 'group' of a type (line 193)
        group_161388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 30), match_161387, 'group')
        # Calling group(args, kwargs) (line 193)
        group_call_result_161390 = invoke(stypy.reporting.localization.Localization(__file__, 193, 30), group_161388, *[], **kwargs_161389)
        
        # Processing the call keyword arguments (line 193)
        kwargs_161391 = {}
        # Getting the type of 'float' (line 193)
        float_161386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'float', False)
        # Calling float(args, kwargs) (line 193)
        float_call_result_161392 = invoke(stypy.reporting.localization.Localization(__file__, 193, 24), float_161386, *[group_call_result_161390], **kwargs_161391)
        
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 194)
        tuple_161393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 194)
        # Adding element type (line 194)
        # Getting the type of 'cls' (line 194)
        cls_161394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 31), 'cls')
        # Obtaining the member '_number' of a type (line 194)
        _number_161395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 31), cls_161394, '_number')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 31), tuple_161393, _number_161395)
        # Adding element type (line 194)
        
        # Call to group(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_161398 = {}
        # Getting the type of 'match' (line 194)
        match_161396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 44), 'match', False)
        # Obtaining the member 'group' of a type (line 194)
        group_161397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 44), match_161396, 'group')
        # Calling group(args, kwargs) (line 194)
        group_call_result_161399 = invoke(stypy.reporting.localization.Localization(__file__, 194, 44), group_161397, *[], **kwargs_161398)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 31), tuple_161393, group_call_result_161399)
        
        GeneratorType_161400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 24), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 24), GeneratorType_161400, tuple_161393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 24), 'stypy_return_type', GeneratorType_161400)
        # SSA branch for the except part of a try statement (line 192)
        # SSA branch for the except 'ValueError' branch of a try statement (line 192)
        module_type_store.open_ssa_branch('except')
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 196)
        tuple_161401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 196)
        # Adding element type (line 196)
        # Getting the type of 'cls' (line 196)
        cls_161402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 31), 'cls')
        # Obtaining the member '_name' of a type (line 196)
        _name_161403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 31), cls_161402, '_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 31), tuple_161401, _name_161403)
        # Adding element type (line 196)
        
        # Call to group(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_161406 = {}
        # Getting the type of 'match' (line 196)
        match_161404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 42), 'match', False)
        # Obtaining the member 'group' of a type (line 196)
        group_161405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 42), match_161404, 'group')
        # Calling group(args, kwargs) (line 196)
        group_call_result_161407 = invoke(stypy.reporting.localization.Localization(__file__, 196, 42), group_161405, *[], **kwargs_161406)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 31), tuple_161401, group_call_result_161407)
        
        GeneratorType_161408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 24), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 24), GeneratorType_161408, tuple_161401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'stypy_return_type', GeneratorType_161408)
        # SSA join for try-except statement (line 192)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'pos' (line 197)
        pos_161409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'pos')
        
        # Call to end(...): (line 197)
        # Processing the call keyword arguments (line 197)
        kwargs_161412 = {}
        # Getting the type of 'match' (line 197)
        match_161410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 27), 'match', False)
        # Obtaining the member 'end' of a type (line 197)
        end_161411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 27), match_161410, 'end')
        # Calling end(args, kwargs) (line 197)
        end_call_result_161413 = invoke(stypy.reporting.localization.Localization(__file__, 197, 27), end_161411, *[], **kwargs_161412)
        
        # Applying the binary operator '+=' (line 197)
        result_iadd_161414 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 20), '+=', pos_161409, end_call_result_161413)
        # Assigning a type to the variable 'pos' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'pos', result_iadd_161414)
        
        # SSA branch for the else part of an if statement (line 191)
        module_type_store.open_ssa_branch('else')
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 199)
        tuple_161415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 199)
        # Adding element type (line 199)
        # Getting the type of 'cls' (line 199)
        cls_161416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'cls')
        # Obtaining the member '_delimiter' of a type (line 199)
        _delimiter_161417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 27), cls_161416, '_delimiter')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 27), tuple_161415, _delimiter_161417)
        # Adding element type (line 199)
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 199)
        pos_161418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 48), 'pos')
        # Getting the type of 'pos' (line 199)
        pos_161419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 52), 'pos')
        int_161420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 58), 'int')
        # Applying the binary operator '+' (line 199)
        result_add_161421 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 52), '+', pos_161419, int_161420)
        
        slice_161422 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 43), pos_161418, result_add_161421, None)
        # Getting the type of 'text' (line 199)
        text_161423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 43), 'text')
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___161424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 43), text_161423, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_161425 = invoke(stypy.reporting.localization.Localization(__file__, 199, 43), getitem___161424, slice_161422)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 27), tuple_161415, subscript_call_result_161425)
        
        GeneratorType_161426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 20), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 20), GeneratorType_161426, tuple_161415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'stypy_return_type', GeneratorType_161426)
        
        # Getting the type of 'pos' (line 200)
        pos_161427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'pos')
        int_161428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'int')
        # Applying the binary operator '+=' (line 200)
        result_iadd_161429 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 20), '+=', pos_161427, int_161428)
        # Assigning a type to the variable 'pos' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'pos', result_iadd_161429)
        
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 182)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 166)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 160)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_tokens(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_tokens' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_161430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161430)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_tokens'
        return stypy_return_type_161430


    @norecursion
    def _parse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parse'
        module_type_store = module_type_store.open_function_context('_parse', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type1Font._parse.__dict__.__setitem__('stypy_localization', localization)
        Type1Font._parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type1Font._parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type1Font._parse.__dict__.__setitem__('stypy_function_name', 'Type1Font._parse')
        Type1Font._parse.__dict__.__setitem__('stypy_param_names_list', [])
        Type1Font._parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type1Font._parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type1Font._parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type1Font._parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type1Font._parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type1Font._parse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type1Font._parse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse(...)' code ##################

        unicode_161431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, (-1)), 'unicode', u'\n        Find the values of various font properties. This limited kind\n        of parsing is described in Chapter 10 "Adobe Type Manager\n        Compatibility" of the Type-1 spec.\n        ')
        
        # Assigning a Dict to a Name (line 209):
        
        # Assigning a Dict to a Name (line 209):
        
        # Obtaining an instance of the builtin type 'dict' (line 209)
        dict_161432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 209)
        # Adding element type (key, value) (line 209)
        unicode_161433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 16), 'unicode', u'weight')
        unicode_161434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 26), 'unicode', u'Regular')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), dict_161432, (unicode_161433, unicode_161434))
        # Adding element type (key, value) (line 209)
        unicode_161435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 37), 'unicode', u'ItalicAngle')
        float_161436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 52), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), dict_161432, (unicode_161435, float_161436))
        # Adding element type (key, value) (line 209)
        unicode_161437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 57), 'unicode', u'isFixedPitch')
        # Getting the type of 'False' (line 209)
        False_161438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 73), 'False')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), dict_161432, (unicode_161437, False_161438))
        # Adding element type (key, value) (line 209)
        unicode_161439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 16), 'unicode', u'UnderlinePosition')
        int_161440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 37), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), dict_161432, (unicode_161439, int_161440))
        # Adding element type (key, value) (line 209)
        unicode_161441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 43), 'unicode', u'UnderlineThickness')
        int_161442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 65), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 15), dict_161432, (unicode_161441, int_161442))
        
        # Assigning a type to the variable 'prop' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'prop', dict_161432)
        
        # Assigning a GeneratorExp to a Name (line 211):
        
        # Assigning a GeneratorExp to a Name (line 211):
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 211, 20, True)
        # Calculating comprehension expression
        
        # Call to _tokens(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Obtaining the type of the subscript
        int_161452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 64), 'int')
        # Getting the type of 'self' (line 212)
        self_161453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 53), 'self', False)
        # Obtaining the member 'parts' of a type (line 212)
        parts_161454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 53), self_161453, 'parts')
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___161455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 53), parts_161454, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_161456 = invoke(stypy.reporting.localization.Localization(__file__, 212, 53), getitem___161455, int_161452)
        
        # Processing the call keyword arguments (line 212)
        kwargs_161457 = {}
        # Getting the type of 'self' (line 212)
        self_161450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 40), 'self', False)
        # Obtaining the member '_tokens' of a type (line 212)
        _tokens_161451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 40), self_161450, '_tokens')
        # Calling _tokens(args, kwargs) (line 212)
        _tokens_call_result_161458 = invoke(stypy.reporting.localization.Localization(__file__, 212, 40), _tokens_161451, *[subscript_call_result_161456], **kwargs_161457)
        
        comprehension_161459 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 20), _tokens_call_result_161458)
        # Assigning a type to the variable 'token' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'token', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 20), comprehension_161459))
        # Assigning a type to the variable 'value' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 20), comprehension_161459))
        
        # Getting the type of 'token' (line 213)
        token_161446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 23), 'token')
        # Getting the type of 'self' (line 213)
        self_161447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 36), 'self')
        # Obtaining the member '_whitespace' of a type (line 213)
        _whitespace_161448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 36), self_161447, '_whitespace')
        # Applying the binary operator 'isnot' (line 213)
        result_is_not_161449 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 23), 'isnot', token_161446, _whitespace_161448)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 211)
        tuple_161443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 211)
        # Adding element type (line 211)
        # Getting the type of 'token' (line 211)
        token_161444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'token')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 21), tuple_161443, token_161444)
        # Adding element type (line 211)
        # Getting the type of 'value' (line 211)
        value_161445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 28), 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 21), tuple_161443, value_161445)
        
        list_161460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 20), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 20), list_161460, tuple_161443)
        # Assigning a type to the variable 'filtered' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'filtered', list_161460)
        
        # Assigning a Lambda to a Name (line 217):
        
        # Assigning a Lambda to a Name (line 217):

        @norecursion
        def _stypy_temp_lambda_24(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_24'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_24', 217, 18, True)
            # Passed parameters checking function
            _stypy_temp_lambda_24.stypy_localization = localization
            _stypy_temp_lambda_24.stypy_type_of_self = None
            _stypy_temp_lambda_24.stypy_type_store = module_type_store
            _stypy_temp_lambda_24.stypy_function_name = '_stypy_temp_lambda_24'
            _stypy_temp_lambda_24.stypy_param_names_list = ['x']
            _stypy_temp_lambda_24.stypy_varargs_param_name = None
            _stypy_temp_lambda_24.stypy_kwargs_param_name = None
            _stypy_temp_lambda_24.stypy_call_defaults = defaults
            _stypy_temp_lambda_24.stypy_call_varargs = varargs
            _stypy_temp_lambda_24.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_24', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_24', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to decode(...): (line 217)
            # Processing the call arguments (line 217)
            unicode_161463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 37), 'unicode', u'ascii')
            unicode_161464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 46), 'unicode', u'replace')
            # Processing the call keyword arguments (line 217)
            kwargs_161465 = {}
            # Getting the type of 'x' (line 217)
            x_161461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'x', False)
            # Obtaining the member 'decode' of a type (line 217)
            decode_161462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 28), x_161461, 'decode')
            # Calling decode(args, kwargs) (line 217)
            decode_call_result_161466 = invoke(stypy.reporting.localization.Localization(__file__, 217, 28), decode_161462, *[unicode_161463, unicode_161464], **kwargs_161465)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'stypy_return_type', decode_call_result_161466)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_24' in the type store
            # Getting the type of 'stypy_return_type' (line 217)
            stypy_return_type_161467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_161467)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_24'
            return stypy_return_type_161467

        # Assigning a type to the variable '_stypy_temp_lambda_24' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), '_stypy_temp_lambda_24', _stypy_temp_lambda_24)
        # Getting the type of '_stypy_temp_lambda_24' (line 217)
        _stypy_temp_lambda_24_161468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), '_stypy_temp_lambda_24')
        # Assigning a type to the variable 'convert' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'convert', _stypy_temp_lambda_24_161468)
        
        # Getting the type of 'filtered' (line 218)
        filtered_161469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 28), 'filtered')
        # Testing the type of a for loop iterable (line 218)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 218, 8), filtered_161469)
        # Getting the type of the for loop variable (line 218)
        for_loop_var_161470 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 218, 8), filtered_161469)
        # Assigning a type to the variable 'token' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'token', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 8), for_loop_var_161470))
        # Assigning a type to the variable 'value' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 8), for_loop_var_161470))
        # SSA begins for a for statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'token' (line 219)
        token_161471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'token')
        # Getting the type of 'self' (line 219)
        self_161472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'self')
        # Obtaining the member '_name' of a type (line 219)
        _name_161473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 24), self_161472, '_name')
        # Applying the binary operator 'is' (line 219)
        result_is__161474 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 15), 'is', token_161471, _name_161473)
        
        
        # Call to startswith(...): (line 219)
        # Processing the call arguments (line 219)
        str_161477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 56), 'str', '/')
        # Processing the call keyword arguments (line 219)
        kwargs_161478 = {}
        # Getting the type of 'value' (line 219)
        value_161475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 39), 'value', False)
        # Obtaining the member 'startswith' of a type (line 219)
        startswith_161476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 39), value_161475, 'startswith')
        # Calling startswith(args, kwargs) (line 219)
        startswith_call_result_161479 = invoke(stypy.reporting.localization.Localization(__file__, 219, 39), startswith_161476, *[str_161477], **kwargs_161478)
        
        # Applying the binary operator 'and' (line 219)
        result_and_keyword_161480 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 15), 'and', result_is__161474, startswith_call_result_161479)
        
        # Testing the type of an if condition (line 219)
        if_condition_161481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 12), result_and_keyword_161480)
        # Assigning a type to the variable 'if_condition_161481' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'if_condition_161481', if_condition_161481)
        # SSA begins for if statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to convert(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Obtaining the type of the subscript
        int_161483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 36), 'int')
        slice_161484 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 220, 30), int_161483, None, None)
        # Getting the type of 'value' (line 220)
        value_161485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'value', False)
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___161486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 30), value_161485, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_161487 = invoke(stypy.reporting.localization.Localization(__file__, 220, 30), getitem___161486, slice_161484)
        
        # Processing the call keyword arguments (line 220)
        kwargs_161488 = {}
        # Getting the type of 'convert' (line 220)
        convert_161482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'convert', False)
        # Calling convert(args, kwargs) (line 220)
        convert_call_result_161489 = invoke(stypy.reporting.localization.Localization(__file__, 220, 22), convert_161482, *[subscript_call_result_161487], **kwargs_161488)
        
        # Assigning a type to the variable 'key' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'key', convert_call_result_161489)
        
        # Assigning a Call to a Tuple (line 221):
        
        # Assigning a Call to a Name:
        
        # Call to next(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'filtered' (line 221)
        filtered_161491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 36), 'filtered', False)
        # Processing the call keyword arguments (line 221)
        kwargs_161492 = {}
        # Getting the type of 'next' (line 221)
        next_161490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 31), 'next', False)
        # Calling next(args, kwargs) (line 221)
        next_call_result_161493 = invoke(stypy.reporting.localization.Localization(__file__, 221, 31), next_161490, *[filtered_161491], **kwargs_161492)
        
        # Assigning a type to the variable 'call_assignment_160929' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_160929', next_call_result_161493)
        
        # Assigning a Call to a Name (line 221):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_161496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 16), 'int')
        # Processing the call keyword arguments
        kwargs_161497 = {}
        # Getting the type of 'call_assignment_160929' (line 221)
        call_assignment_160929_161494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_160929', False)
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___161495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), call_assignment_160929_161494, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_161498 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161495, *[int_161496], **kwargs_161497)
        
        # Assigning a type to the variable 'call_assignment_160930' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_160930', getitem___call_result_161498)
        
        # Assigning a Name to a Name (line 221):
        # Getting the type of 'call_assignment_160930' (line 221)
        call_assignment_160930_161499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_160930')
        # Assigning a type to the variable 'token' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'token', call_assignment_160930_161499)
        
        # Assigning a Call to a Name (line 221):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_161502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 16), 'int')
        # Processing the call keyword arguments
        kwargs_161503 = {}
        # Getting the type of 'call_assignment_160929' (line 221)
        call_assignment_160929_161500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_160929', False)
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___161501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), call_assignment_160929_161500, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_161504 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161501, *[int_161502], **kwargs_161503)
        
        # Assigning a type to the variable 'call_assignment_160931' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_160931', getitem___call_result_161504)
        
        # Assigning a Name to a Name (line 221):
        # Getting the type of 'call_assignment_160931' (line 221)
        call_assignment_160931_161505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_160931')
        # Assigning a type to the variable 'value' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 23), 'value', call_assignment_160931_161505)
        
        
        # Getting the type of 'token' (line 222)
        token_161506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'token')
        # Getting the type of 'self' (line 222)
        self_161507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'self')
        # Obtaining the member '_name' of a type (line 222)
        _name_161508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 28), self_161507, '_name')
        # Applying the binary operator 'is' (line 222)
        result_is__161509 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 19), 'is', token_161506, _name_161508)
        
        # Testing the type of an if condition (line 222)
        if_condition_161510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 16), result_is__161509)
        # Assigning a type to the variable 'if_condition_161510' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'if_condition_161510', if_condition_161510)
        # SSA begins for if statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'value' (line 223)
        value_161511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'value')
        
        # Obtaining an instance of the builtin type 'tuple' (line 223)
        tuple_161512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 223)
        # Adding element type (line 223)
        str_161513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 33), 'str', 'true')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 33), tuple_161512, str_161513)
        # Adding element type (line 223)
        str_161514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 42), 'str', 'false')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 33), tuple_161512, str_161514)
        
        # Applying the binary operator 'in' (line 223)
        result_contains_161515 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 23), 'in', value_161511, tuple_161512)
        
        # Testing the type of an if condition (line 223)
        if_condition_161516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 20), result_contains_161515)
        # Assigning a type to the variable 'if_condition_161516' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'if_condition_161516', if_condition_161516)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Compare to a Name (line 224):
        
        # Assigning a Compare to a Name (line 224):
        
        # Getting the type of 'value' (line 224)
        value_161517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 32), 'value')
        str_161518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 41), 'str', 'true')
        # Applying the binary operator '==' (line 224)
        result_eq_161519 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 32), '==', value_161517, str_161518)
        
        # Assigning a type to the variable 'value' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 'value', result_eq_161519)
        # SSA branch for the else part of an if statement (line 223)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 226):
        
        # Assigning a Call to a Name (line 226):
        
        # Call to convert(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Call to lstrip(...): (line 226)
        # Processing the call arguments (line 226)
        str_161523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 53), 'str', '/')
        # Processing the call keyword arguments (line 226)
        kwargs_161524 = {}
        # Getting the type of 'value' (line 226)
        value_161521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'value', False)
        # Obtaining the member 'lstrip' of a type (line 226)
        lstrip_161522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 40), value_161521, 'lstrip')
        # Calling lstrip(args, kwargs) (line 226)
        lstrip_call_result_161525 = invoke(stypy.reporting.localization.Localization(__file__, 226, 40), lstrip_161522, *[str_161523], **kwargs_161524)
        
        # Processing the call keyword arguments (line 226)
        kwargs_161526 = {}
        # Getting the type of 'convert' (line 226)
        convert_161520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 32), 'convert', False)
        # Calling convert(args, kwargs) (line 226)
        convert_call_result_161527 = invoke(stypy.reporting.localization.Localization(__file__, 226, 32), convert_161520, *[lstrip_call_result_161525], **kwargs_161526)
        
        # Assigning a type to the variable 'value' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'value', convert_call_result_161527)
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 222)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'token' (line 227)
        token_161528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'token')
        # Getting the type of 'self' (line 227)
        self_161529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 30), 'self')
        # Obtaining the member '_string' of a type (line 227)
        _string_161530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 30), self_161529, '_string')
        # Applying the binary operator 'is' (line 227)
        result_is__161531 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 21), 'is', token_161528, _string_161530)
        
        # Testing the type of an if condition (line 227)
        if_condition_161532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 21), result_is__161531)
        # Assigning a type to the variable 'if_condition_161532' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'if_condition_161532', if_condition_161532)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 228):
        
        # Assigning a Call to a Name (line 228):
        
        # Call to convert(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Call to rstrip(...): (line 228)
        # Processing the call arguments (line 228)
        str_161540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 62), 'str', ')')
        # Processing the call keyword arguments (line 228)
        kwargs_161541 = {}
        
        # Call to lstrip(...): (line 228)
        # Processing the call arguments (line 228)
        str_161536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 49), 'str', '(')
        # Processing the call keyword arguments (line 228)
        kwargs_161537 = {}
        # Getting the type of 'value' (line 228)
        value_161534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 36), 'value', False)
        # Obtaining the member 'lstrip' of a type (line 228)
        lstrip_161535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 36), value_161534, 'lstrip')
        # Calling lstrip(args, kwargs) (line 228)
        lstrip_call_result_161538 = invoke(stypy.reporting.localization.Localization(__file__, 228, 36), lstrip_161535, *[str_161536], **kwargs_161537)
        
        # Obtaining the member 'rstrip' of a type (line 228)
        rstrip_161539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 36), lstrip_call_result_161538, 'rstrip')
        # Calling rstrip(args, kwargs) (line 228)
        rstrip_call_result_161542 = invoke(stypy.reporting.localization.Localization(__file__, 228, 36), rstrip_161539, *[str_161540], **kwargs_161541)
        
        # Processing the call keyword arguments (line 228)
        kwargs_161543 = {}
        # Getting the type of 'convert' (line 228)
        convert_161533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 28), 'convert', False)
        # Calling convert(args, kwargs) (line 228)
        convert_call_result_161544 = invoke(stypy.reporting.localization.Localization(__file__, 228, 28), convert_161533, *[rstrip_call_result_161542], **kwargs_161543)
        
        # Assigning a type to the variable 'value' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'value', convert_call_result_161544)
        # SSA branch for the else part of an if statement (line 227)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'token' (line 229)
        token_161545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 21), 'token')
        # Getting the type of 'self' (line 229)
        self_161546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 30), 'self')
        # Obtaining the member '_number' of a type (line 229)
        _number_161547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 30), self_161546, '_number')
        # Applying the binary operator 'is' (line 229)
        result_is__161548 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 21), 'is', token_161545, _number_161547)
        
        # Testing the type of an if condition (line 229)
        if_condition_161549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 21), result_is__161548)
        # Assigning a type to the variable 'if_condition_161549' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 21), 'if_condition_161549', if_condition_161549)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        str_161550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 23), 'str', '.')
        # Getting the type of 'value' (line 230)
        value_161551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 31), 'value')
        # Applying the binary operator 'in' (line 230)
        result_contains_161552 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 23), 'in', str_161550, value_161551)
        
        # Testing the type of an if condition (line 230)
        if_condition_161553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 20), result_contains_161552)
        # Assigning a type to the variable 'if_condition_161553' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'if_condition_161553', if_condition_161553)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to float(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'value' (line 231)
        value_161555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 38), 'value', False)
        # Processing the call keyword arguments (line 231)
        kwargs_161556 = {}
        # Getting the type of 'float' (line 231)
        float_161554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 32), 'float', False)
        # Calling float(args, kwargs) (line 231)
        float_call_result_161557 = invoke(stypy.reporting.localization.Localization(__file__, 231, 32), float_161554, *[value_161555], **kwargs_161556)
        
        # Assigning a type to the variable 'value' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'value', float_call_result_161557)
        # SSA branch for the else part of an if statement (line 230)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to int(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'value' (line 233)
        value_161559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 36), 'value', False)
        # Processing the call keyword arguments (line 233)
        kwargs_161560 = {}
        # Getting the type of 'int' (line 233)
        int_161558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 32), 'int', False)
        # Calling int(args, kwargs) (line 233)
        int_call_result_161561 = invoke(stypy.reporting.localization.Localization(__file__, 233, 32), int_161558, *[value_161559], **kwargs_161560)
        
        # Assigning a type to the variable 'value' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 24), 'value', int_call_result_161561)
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 229)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 235):
        
        # Assigning a Name to a Name (line 235):
        # Getting the type of 'None' (line 235)
        None_161562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'None')
        # Assigning a type to the variable 'value' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'value', None_161562)
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 222)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'key' (line 236)
        key_161563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 19), 'key')
        unicode_161564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 26), 'unicode', u'FontInfo')
        # Applying the binary operator '!=' (line 236)
        result_ne_161565 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 19), '!=', key_161563, unicode_161564)
        
        
        # Getting the type of 'value' (line 236)
        value_161566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 41), 'value')
        # Getting the type of 'None' (line 236)
        None_161567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 54), 'None')
        # Applying the binary operator 'isnot' (line 236)
        result_is_not_161568 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 41), 'isnot', value_161566, None_161567)
        
        # Applying the binary operator 'and' (line 236)
        result_and_keyword_161569 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 19), 'and', result_ne_161565, result_is_not_161568)
        
        # Testing the type of an if condition (line 236)
        if_condition_161570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 16), result_and_keyword_161569)
        # Assigning a type to the variable 'if_condition_161570' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'if_condition_161570', if_condition_161570)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 237):
        
        # Assigning a Name to a Subscript (line 237):
        # Getting the type of 'value' (line 237)
        value_161571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'value')
        # Getting the type of 'prop' (line 237)
        prop_161572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'prop')
        # Getting the type of 'key' (line 237)
        key_161573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 25), 'key')
        # Storing an element on a container (line 237)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 20), prop_161572, (key_161573, value_161571))
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 219)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        unicode_161574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 11), 'unicode', u'FontName')
        # Getting the type of 'prop' (line 240)
        prop_161575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 29), 'prop')
        # Applying the binary operator 'notin' (line 240)
        result_contains_161576 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), 'notin', unicode_161574, prop_161575)
        
        # Testing the type of an if condition (line 240)
        if_condition_161577 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_contains_161576)
        # Assigning a type to the variable 'if_condition_161577' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_161577', if_condition_161577)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BoolOp to a Subscript (line 241):
        
        # Assigning a BoolOp to a Subscript (line 241):
        
        # Evaluating a boolean operation
        
        # Call to get(...): (line 241)
        # Processing the call arguments (line 241)
        unicode_161580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 41), 'unicode', u'FullName')
        # Processing the call keyword arguments (line 241)
        kwargs_161581 = {}
        # Getting the type of 'prop' (line 241)
        prop_161578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 32), 'prop', False)
        # Obtaining the member 'get' of a type (line 241)
        get_161579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 32), prop_161578, 'get')
        # Calling get(args, kwargs) (line 241)
        get_call_result_161582 = invoke(stypy.reporting.localization.Localization(__file__, 241, 32), get_161579, *[unicode_161580], **kwargs_161581)
        
        
        # Call to get(...): (line 242)
        # Processing the call arguments (line 242)
        unicode_161585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 41), 'unicode', u'FamilyName')
        # Processing the call keyword arguments (line 242)
        kwargs_161586 = {}
        # Getting the type of 'prop' (line 242)
        prop_161583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 32), 'prop', False)
        # Obtaining the member 'get' of a type (line 242)
        get_161584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 32), prop_161583, 'get')
        # Calling get(args, kwargs) (line 242)
        get_call_result_161587 = invoke(stypy.reporting.localization.Localization(__file__, 242, 32), get_161584, *[unicode_161585], **kwargs_161586)
        
        # Applying the binary operator 'or' (line 241)
        result_or_keyword_161588 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 32), 'or', get_call_result_161582, get_call_result_161587)
        unicode_161589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 32), 'unicode', u'Unknown')
        # Applying the binary operator 'or' (line 241)
        result_or_keyword_161590 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 32), 'or', result_or_keyword_161588, unicode_161589)
        
        # Getting the type of 'prop' (line 241)
        prop_161591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'prop')
        unicode_161592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 17), 'unicode', u'FontName')
        # Storing an element on a container (line 241)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 12), prop_161591, (unicode_161592, result_or_keyword_161590))
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        unicode_161593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 11), 'unicode', u'FullName')
        # Getting the type of 'prop' (line 244)
        prop_161594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 29), 'prop')
        # Applying the binary operator 'notin' (line 244)
        result_contains_161595 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 11), 'notin', unicode_161593, prop_161594)
        
        # Testing the type of an if condition (line 244)
        if_condition_161596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 8), result_contains_161595)
        # Assigning a type to the variable 'if_condition_161596' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'if_condition_161596', if_condition_161596)
        # SSA begins for if statement (line 244)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 245):
        
        # Assigning a Subscript to a Subscript (line 245):
        
        # Obtaining the type of the subscript
        unicode_161597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 36), 'unicode', u'FontName')
        # Getting the type of 'prop' (line 245)
        prop_161598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 31), 'prop')
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___161599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 31), prop_161598, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_161600 = invoke(stypy.reporting.localization.Localization(__file__, 245, 31), getitem___161599, unicode_161597)
        
        # Getting the type of 'prop' (line 245)
        prop_161601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'prop')
        unicode_161602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 17), 'unicode', u'FullName')
        # Storing an element on a container (line 245)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 12), prop_161601, (unicode_161602, subscript_call_result_161600))
        # SSA join for if statement (line 244)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        unicode_161603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 11), 'unicode', u'FamilyName')
        # Getting the type of 'prop' (line 246)
        prop_161604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'prop')
        # Applying the binary operator 'notin' (line 246)
        result_contains_161605 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 11), 'notin', unicode_161603, prop_161604)
        
        # Testing the type of an if condition (line 246)
        if_condition_161606 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 8), result_contains_161605)
        # Assigning a type to the variable 'if_condition_161606' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'if_condition_161606', if_condition_161606)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 247):
        
        # Assigning a Str to a Name (line 247):
        unicode_161607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 21), 'unicode', u'(?i)([ -](regular|plain|italic|oblique|(semi)?bold|(ultra)?light|extra|condensed))+$')
        # Assigning a type to the variable 'extras' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'extras', unicode_161607)
        
        # Assigning a Call to a Subscript (line 248):
        
        # Assigning a Call to a Subscript (line 248):
        
        # Call to sub(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'extras' (line 248)
        extras_161610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 40), 'extras', False)
        unicode_161611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 48), 'unicode', u'')
        
        # Obtaining the type of the subscript
        unicode_161612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 57), 'unicode', u'FullName')
        # Getting the type of 'prop' (line 248)
        prop_161613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 52), 'prop', False)
        # Obtaining the member '__getitem__' of a type (line 248)
        getitem___161614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 52), prop_161613, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 248)
        subscript_call_result_161615 = invoke(stypy.reporting.localization.Localization(__file__, 248, 52), getitem___161614, unicode_161612)
        
        # Processing the call keyword arguments (line 248)
        kwargs_161616 = {}
        # Getting the type of 're' (line 248)
        re_161608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 33), 're', False)
        # Obtaining the member 'sub' of a type (line 248)
        sub_161609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 33), re_161608, 'sub')
        # Calling sub(args, kwargs) (line 248)
        sub_call_result_161617 = invoke(stypy.reporting.localization.Localization(__file__, 248, 33), sub_161609, *[extras_161610, unicode_161611, subscript_call_result_161615], **kwargs_161616)
        
        # Getting the type of 'prop' (line 248)
        prop_161618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'prop')
        unicode_161619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 17), 'unicode', u'FamilyName')
        # Storing an element on a container (line 248)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 12), prop_161618, (unicode_161619, sub_call_result_161617))
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 250):
        
        # Assigning a Name to a Attribute (line 250):
        # Getting the type of 'prop' (line 250)
        prop_161620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'prop')
        # Getting the type of 'self' (line 250)
        self_161621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member 'prop' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_161621, 'prop', prop_161620)
        
        # ################# End of '_parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_161622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse'
        return stypy_return_type_161622


    @norecursion
    def _transformer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_transformer'
        module_type_store = module_type_store.open_function_context('_transformer', 252, 4, False)
        # Assigning a type to the variable 'self' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type1Font._transformer.__dict__.__setitem__('stypy_localization', localization)
        Type1Font._transformer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type1Font._transformer.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type1Font._transformer.__dict__.__setitem__('stypy_function_name', 'Type1Font._transformer')
        Type1Font._transformer.__dict__.__setitem__('stypy_param_names_list', ['tokens', 'slant', 'extend'])
        Type1Font._transformer.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type1Font._transformer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type1Font._transformer.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type1Font._transformer.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type1Font._transformer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type1Font._transformer.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type1Font._transformer', ['tokens', 'slant', 'extend'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_transformer', localization, ['tokens', 'slant', 'extend'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_transformer(...)' code ##################


        @norecursion
        def fontname(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fontname'
            module_type_store = module_type_store.open_function_context('fontname', 254, 8, False)
            
            # Passed parameters checking function
            fontname.stypy_localization = localization
            fontname.stypy_type_of_self = None
            fontname.stypy_type_store = module_type_store
            fontname.stypy_function_name = 'fontname'
            fontname.stypy_param_names_list = ['name']
            fontname.stypy_varargs_param_name = None
            fontname.stypy_kwargs_param_name = None
            fontname.stypy_call_defaults = defaults
            fontname.stypy_call_varargs = varargs
            fontname.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fontname', ['name'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fontname', localization, ['name'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fontname(...)' code ##################

            
            # Assigning a Name to a Name (line 255):
            
            # Assigning a Name to a Name (line 255):
            # Getting the type of 'name' (line 255)
            name_161623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 21), 'name')
            # Assigning a type to the variable 'result' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'result', name_161623)
            
            # Getting the type of 'slant' (line 256)
            slant_161624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'slant')
            # Testing the type of an if condition (line 256)
            if_condition_161625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 12), slant_161624)
            # Assigning a type to the variable 'if_condition_161625' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'if_condition_161625', if_condition_161625)
            # SSA begins for if statement (line 256)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'result' (line 257)
            result_161626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'result')
            str_161627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 26), 'str', '_Slant_')
            
            # Call to encode(...): (line 257)
            # Processing the call arguments (line 257)
            unicode_161638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 69), 'unicode', u'latin-1')
            # Processing the call keyword arguments (line 257)
            kwargs_161639 = {}
            
            # Call to str(...): (line 257)
            # Processing the call arguments (line 257)
            
            # Call to int(...): (line 257)
            # Processing the call arguments (line 257)
            int_161630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 47), 'int')
            # Getting the type of 'slant' (line 257)
            slant_161631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 54), 'slant', False)
            # Applying the binary operator '*' (line 257)
            result_mul_161632 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 47), '*', int_161630, slant_161631)
            
            # Processing the call keyword arguments (line 257)
            kwargs_161633 = {}
            # Getting the type of 'int' (line 257)
            int_161629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 43), 'int', False)
            # Calling int(args, kwargs) (line 257)
            int_call_result_161634 = invoke(stypy.reporting.localization.Localization(__file__, 257, 43), int_161629, *[result_mul_161632], **kwargs_161633)
            
            # Processing the call keyword arguments (line 257)
            kwargs_161635 = {}
            # Getting the type of 'str' (line 257)
            str_161628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 39), 'str', False)
            # Calling str(args, kwargs) (line 257)
            str_call_result_161636 = invoke(stypy.reporting.localization.Localization(__file__, 257, 39), str_161628, *[int_call_result_161634], **kwargs_161635)
            
            # Obtaining the member 'encode' of a type (line 257)
            encode_161637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 39), str_call_result_161636, 'encode')
            # Calling encode(args, kwargs) (line 257)
            encode_call_result_161640 = invoke(stypy.reporting.localization.Localization(__file__, 257, 39), encode_161637, *[unicode_161638], **kwargs_161639)
            
            # Applying the binary operator '+' (line 257)
            result_add_161641 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 26), '+', str_161627, encode_call_result_161640)
            
            # Applying the binary operator '+=' (line 257)
            result_iadd_161642 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 16), '+=', result_161626, result_add_161641)
            # Assigning a type to the variable 'result' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'result', result_iadd_161642)
            
            # SSA join for if statement (line 256)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'extend' (line 258)
            extend_161643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'extend')
            float_161644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 25), 'float')
            # Applying the binary operator '!=' (line 258)
            result_ne_161645 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 15), '!=', extend_161643, float_161644)
            
            # Testing the type of an if condition (line 258)
            if_condition_161646 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 12), result_ne_161645)
            # Assigning a type to the variable 'if_condition_161646' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'if_condition_161646', if_condition_161646)
            # SSA begins for if statement (line 258)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'result' (line 259)
            result_161647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'result')
            str_161648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 26), 'str', '_Extend_')
            
            # Call to encode(...): (line 259)
            # Processing the call arguments (line 259)
            unicode_161659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 71), 'unicode', u'latin-1')
            # Processing the call keyword arguments (line 259)
            kwargs_161660 = {}
            
            # Call to str(...): (line 259)
            # Processing the call arguments (line 259)
            
            # Call to int(...): (line 259)
            # Processing the call arguments (line 259)
            int_161651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 48), 'int')
            # Getting the type of 'extend' (line 259)
            extend_161652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 55), 'extend', False)
            # Applying the binary operator '*' (line 259)
            result_mul_161653 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 48), '*', int_161651, extend_161652)
            
            # Processing the call keyword arguments (line 259)
            kwargs_161654 = {}
            # Getting the type of 'int' (line 259)
            int_161650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 44), 'int', False)
            # Calling int(args, kwargs) (line 259)
            int_call_result_161655 = invoke(stypy.reporting.localization.Localization(__file__, 259, 44), int_161650, *[result_mul_161653], **kwargs_161654)
            
            # Processing the call keyword arguments (line 259)
            kwargs_161656 = {}
            # Getting the type of 'str' (line 259)
            str_161649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 40), 'str', False)
            # Calling str(args, kwargs) (line 259)
            str_call_result_161657 = invoke(stypy.reporting.localization.Localization(__file__, 259, 40), str_161649, *[int_call_result_161655], **kwargs_161656)
            
            # Obtaining the member 'encode' of a type (line 259)
            encode_161658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 40), str_call_result_161657, 'encode')
            # Calling encode(args, kwargs) (line 259)
            encode_call_result_161661 = invoke(stypy.reporting.localization.Localization(__file__, 259, 40), encode_161658, *[unicode_161659], **kwargs_161660)
            
            # Applying the binary operator '+' (line 259)
            result_add_161662 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 26), '+', str_161648, encode_call_result_161661)
            
            # Applying the binary operator '+=' (line 259)
            result_iadd_161663 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 16), '+=', result_161647, result_add_161662)
            # Assigning a type to the variable 'result' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'result', result_iadd_161663)
            
            # SSA join for if statement (line 258)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'result' (line 260)
            result_161664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 19), 'result')
            # Assigning a type to the variable 'stypy_return_type' (line 260)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'stypy_return_type', result_161664)
            
            # ################# End of 'fontname(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fontname' in the type store
            # Getting the type of 'stypy_return_type' (line 254)
            stypy_return_type_161665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_161665)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fontname'
            return stypy_return_type_161665

        # Assigning a type to the variable 'fontname' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'fontname', fontname)

        @norecursion
        def italicangle(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'italicangle'
            module_type_store = module_type_store.open_function_context('italicangle', 262, 8, False)
            
            # Passed parameters checking function
            italicangle.stypy_localization = localization
            italicangle.stypy_type_of_self = None
            italicangle.stypy_type_store = module_type_store
            italicangle.stypy_function_name = 'italicangle'
            italicangle.stypy_param_names_list = ['angle']
            italicangle.stypy_varargs_param_name = None
            italicangle.stypy_kwargs_param_name = None
            italicangle.stypy_call_defaults = defaults
            italicangle.stypy_call_varargs = varargs
            italicangle.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'italicangle', ['angle'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'italicangle', localization, ['angle'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'italicangle(...)' code ##################

            
            # Call to encode(...): (line 263)
            # Processing the call arguments (line 263)
            unicode_161685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 77), 'unicode', u'latin-1')
            # Processing the call keyword arguments (line 263)
            kwargs_161686 = {}
            
            # Call to str(...): (line 263)
            # Processing the call arguments (line 263)
            
            # Call to float(...): (line 263)
            # Processing the call arguments (line 263)
            # Getting the type of 'angle' (line 263)
            angle_161668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 29), 'angle', False)
            # Processing the call keyword arguments (line 263)
            kwargs_161669 = {}
            # Getting the type of 'float' (line 263)
            float_161667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 23), 'float', False)
            # Calling float(args, kwargs) (line 263)
            float_call_result_161670 = invoke(stypy.reporting.localization.Localization(__file__, 263, 23), float_161667, *[angle_161668], **kwargs_161669)
            
            
            # Call to arctan(...): (line 263)
            # Processing the call arguments (line 263)
            # Getting the type of 'slant' (line 263)
            slant_161673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 48), 'slant', False)
            # Processing the call keyword arguments (line 263)
            kwargs_161674 = {}
            # Getting the type of 'np' (line 263)
            np_161671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 38), 'np', False)
            # Obtaining the member 'arctan' of a type (line 263)
            arctan_161672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 38), np_161671, 'arctan')
            # Calling arctan(args, kwargs) (line 263)
            arctan_call_result_161675 = invoke(stypy.reporting.localization.Localization(__file__, 263, 38), arctan_161672, *[slant_161673], **kwargs_161674)
            
            # Getting the type of 'np' (line 263)
            np_161676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 57), 'np', False)
            # Obtaining the member 'pi' of a type (line 263)
            pi_161677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 57), np_161676, 'pi')
            # Applying the binary operator 'div' (line 263)
            result_div_161678 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 38), 'div', arctan_call_result_161675, pi_161677)
            
            int_161679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 65), 'int')
            # Applying the binary operator '*' (line 263)
            result_mul_161680 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 63), '*', result_div_161678, int_161679)
            
            # Applying the binary operator '-' (line 263)
            result_sub_161681 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 23), '-', float_call_result_161670, result_mul_161680)
            
            # Processing the call keyword arguments (line 263)
            kwargs_161682 = {}
            # Getting the type of 'str' (line 263)
            str_161666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'str', False)
            # Calling str(args, kwargs) (line 263)
            str_call_result_161683 = invoke(stypy.reporting.localization.Localization(__file__, 263, 19), str_161666, *[result_sub_161681], **kwargs_161682)
            
            # Obtaining the member 'encode' of a type (line 263)
            encode_161684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 19), str_call_result_161683, 'encode')
            # Calling encode(args, kwargs) (line 263)
            encode_call_result_161687 = invoke(stypy.reporting.localization.Localization(__file__, 263, 19), encode_161684, *[unicode_161685], **kwargs_161686)
            
            # Assigning a type to the variable 'stypy_return_type' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'stypy_return_type', encode_call_result_161687)
            
            # ################# End of 'italicangle(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'italicangle' in the type store
            # Getting the type of 'stypy_return_type' (line 262)
            stypy_return_type_161688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_161688)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'italicangle'
            return stypy_return_type_161688

        # Assigning a type to the variable 'italicangle' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'italicangle', italicangle)

        @norecursion
        def fontmatrix(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fontmatrix'
            module_type_store = module_type_store.open_function_context('fontmatrix', 265, 8, False)
            
            # Passed parameters checking function
            fontmatrix.stypy_localization = localization
            fontmatrix.stypy_type_of_self = None
            fontmatrix.stypy_type_store = module_type_store
            fontmatrix.stypy_function_name = 'fontmatrix'
            fontmatrix.stypy_param_names_list = ['array']
            fontmatrix.stypy_varargs_param_name = None
            fontmatrix.stypy_kwargs_param_name = None
            fontmatrix.stypy_call_defaults = defaults
            fontmatrix.stypy_call_varargs = varargs
            fontmatrix.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fontmatrix', ['array'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fontmatrix', localization, ['array'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fontmatrix(...)' code ##################

            
            # Assigning a Call to a Name (line 266):
            
            # Assigning a Call to a Name (line 266):
            
            # Call to split(...): (line 266)
            # Processing the call keyword arguments (line 266)
            kwargs_161702 = {}
            
            # Call to strip(...): (line 266)
            # Processing the call keyword arguments (line 266)
            kwargs_161699 = {}
            
            # Call to rstrip(...): (line 266)
            # Processing the call arguments (line 266)
            str_161695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 46), 'str', ']')
            # Processing the call keyword arguments (line 266)
            kwargs_161696 = {}
            
            # Call to lstrip(...): (line 266)
            # Processing the call arguments (line 266)
            str_161691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 33), 'str', '[')
            # Processing the call keyword arguments (line 266)
            kwargs_161692 = {}
            # Getting the type of 'array' (line 266)
            array_161689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'array', False)
            # Obtaining the member 'lstrip' of a type (line 266)
            lstrip_161690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 20), array_161689, 'lstrip')
            # Calling lstrip(args, kwargs) (line 266)
            lstrip_call_result_161693 = invoke(stypy.reporting.localization.Localization(__file__, 266, 20), lstrip_161690, *[str_161691], **kwargs_161692)
            
            # Obtaining the member 'rstrip' of a type (line 266)
            rstrip_161694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 20), lstrip_call_result_161693, 'rstrip')
            # Calling rstrip(args, kwargs) (line 266)
            rstrip_call_result_161697 = invoke(stypy.reporting.localization.Localization(__file__, 266, 20), rstrip_161694, *[str_161695], **kwargs_161696)
            
            # Obtaining the member 'strip' of a type (line 266)
            strip_161698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 20), rstrip_call_result_161697, 'strip')
            # Calling strip(args, kwargs) (line 266)
            strip_call_result_161700 = invoke(stypy.reporting.localization.Localization(__file__, 266, 20), strip_161698, *[], **kwargs_161699)
            
            # Obtaining the member 'split' of a type (line 266)
            split_161701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 20), strip_call_result_161700, 'split')
            # Calling split(args, kwargs) (line 266)
            split_call_result_161703 = invoke(stypy.reporting.localization.Localization(__file__, 266, 20), split_161701, *[], **kwargs_161702)
            
            # Assigning a type to the variable 'array' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'array', split_call_result_161703)
            
            # Assigning a ListComp to a Name (line 267):
            
            # Assigning a ListComp to a Name (line 267):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'array' (line 267)
            array_161708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 39), 'array')
            comprehension_161709 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 21), array_161708)
            # Assigning a type to the variable 'x' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 21), 'x', comprehension_161709)
            
            # Call to float(...): (line 267)
            # Processing the call arguments (line 267)
            # Getting the type of 'x' (line 267)
            x_161705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 27), 'x', False)
            # Processing the call keyword arguments (line 267)
            kwargs_161706 = {}
            # Getting the type of 'float' (line 267)
            float_161704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 21), 'float', False)
            # Calling float(args, kwargs) (line 267)
            float_call_result_161707 = invoke(stypy.reporting.localization.Localization(__file__, 267, 21), float_161704, *[x_161705], **kwargs_161706)
            
            list_161710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 21), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 21), list_161710, float_call_result_161707)
            # Assigning a type to the variable 'array' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'array', list_161710)
            
            # Assigning a Call to a Name (line 268):
            
            # Assigning a Call to a Name (line 268):
            
            # Call to eye(...): (line 268)
            # Processing the call arguments (line 268)
            int_161713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 31), 'int')
            int_161714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 34), 'int')
            # Processing the call keyword arguments (line 268)
            kwargs_161715 = {}
            # Getting the type of 'np' (line 268)
            np_161711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 24), 'np', False)
            # Obtaining the member 'eye' of a type (line 268)
            eye_161712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 24), np_161711, 'eye')
            # Calling eye(args, kwargs) (line 268)
            eye_call_result_161716 = invoke(stypy.reporting.localization.Localization(__file__, 268, 24), eye_161712, *[int_161713, int_161714], **kwargs_161715)
            
            # Assigning a type to the variable 'oldmatrix' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'oldmatrix', eye_call_result_161716)
            
            # Assigning a Subscript to a Subscript (line 269):
            
            # Assigning a Subscript to a Subscript (line 269):
            
            # Obtaining the type of the subscript
            int_161717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 40), 'int')
            slice_161718 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 269, 32), None, None, int_161717)
            # Getting the type of 'array' (line 269)
            array_161719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 32), 'array')
            # Obtaining the member '__getitem__' of a type (line 269)
            getitem___161720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 32), array_161719, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 269)
            subscript_call_result_161721 = invoke(stypy.reporting.localization.Localization(__file__, 269, 32), getitem___161720, slice_161718)
            
            # Getting the type of 'oldmatrix' (line 269)
            oldmatrix_161722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'oldmatrix')
            int_161723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 22), 'int')
            int_161724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 24), 'int')
            slice_161725 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 269, 12), int_161723, int_161724, None)
            int_161726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 27), 'int')
            # Storing an element on a container (line 269)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 12), oldmatrix_161722, ((slice_161725, int_161726), subscript_call_result_161721))
            
            # Assigning a Subscript to a Subscript (line 270):
            
            # Assigning a Subscript to a Subscript (line 270):
            
            # Obtaining the type of the subscript
            int_161727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 38), 'int')
            int_161728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 41), 'int')
            slice_161729 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 270, 32), int_161727, None, int_161728)
            # Getting the type of 'array' (line 270)
            array_161730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 32), 'array')
            # Obtaining the member '__getitem__' of a type (line 270)
            getitem___161731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 32), array_161730, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 270)
            subscript_call_result_161732 = invoke(stypy.reporting.localization.Localization(__file__, 270, 32), getitem___161731, slice_161729)
            
            # Getting the type of 'oldmatrix' (line 270)
            oldmatrix_161733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'oldmatrix')
            int_161734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 22), 'int')
            int_161735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 24), 'int')
            slice_161736 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 270, 12), int_161734, int_161735, None)
            int_161737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 27), 'int')
            # Storing an element on a container (line 270)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 12), oldmatrix_161733, ((slice_161736, int_161737), subscript_call_result_161732))
            
            # Assigning a Call to a Name (line 271):
            
            # Assigning a Call to a Name (line 271):
            
            # Call to array(...): (line 271)
            # Processing the call arguments (line 271)
            
            # Obtaining an instance of the builtin type 'list' (line 271)
            list_161740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 32), 'list')
            # Adding type elements to the builtin type 'list' instance (line 271)
            # Adding element type (line 271)
            
            # Obtaining an instance of the builtin type 'list' (line 271)
            list_161741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 271)
            # Adding element type (line 271)
            # Getting the type of 'extend' (line 271)
            extend_161742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 34), 'extend', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), list_161741, extend_161742)
            # Adding element type (line 271)
            int_161743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 42), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), list_161741, int_161743)
            # Adding element type (line 271)
            int_161744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 45), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 33), list_161741, int_161744)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 32), list_161740, list_161741)
            # Adding element type (line 271)
            
            # Obtaining an instance of the builtin type 'list' (line 272)
            list_161745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 272)
            # Adding element type (line 272)
            # Getting the type of 'slant' (line 272)
            slant_161746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 34), 'slant', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 33), list_161745, slant_161746)
            # Adding element type (line 272)
            int_161747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 41), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 33), list_161745, int_161747)
            # Adding element type (line 272)
            int_161748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 44), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 33), list_161745, int_161748)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 32), list_161740, list_161745)
            # Adding element type (line 271)
            
            # Obtaining an instance of the builtin type 'list' (line 273)
            list_161749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 273)
            # Adding element type (line 273)
            int_161750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 34), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 33), list_161749, int_161750)
            # Adding element type (line 273)
            int_161751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 37), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 33), list_161749, int_161751)
            # Adding element type (line 273)
            int_161752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 40), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 33), list_161749, int_161752)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 32), list_161740, list_161749)
            
            # Processing the call keyword arguments (line 271)
            kwargs_161753 = {}
            # Getting the type of 'np' (line 271)
            np_161738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 23), 'np', False)
            # Obtaining the member 'array' of a type (line 271)
            array_161739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 23), np_161738, 'array')
            # Calling array(args, kwargs) (line 271)
            array_call_result_161754 = invoke(stypy.reporting.localization.Localization(__file__, 271, 23), array_161739, *[list_161740], **kwargs_161753)
            
            # Assigning a type to the variable 'modifier' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'modifier', array_call_result_161754)
            
            # Assigning a Call to a Name (line 274):
            
            # Assigning a Call to a Name (line 274):
            
            # Call to dot(...): (line 274)
            # Processing the call arguments (line 274)
            # Getting the type of 'modifier' (line 274)
            modifier_161757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 31), 'modifier', False)
            # Getting the type of 'oldmatrix' (line 274)
            oldmatrix_161758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 41), 'oldmatrix', False)
            # Processing the call keyword arguments (line 274)
            kwargs_161759 = {}
            # Getting the type of 'np' (line 274)
            np_161755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'np', False)
            # Obtaining the member 'dot' of a type (line 274)
            dot_161756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 24), np_161755, 'dot')
            # Calling dot(args, kwargs) (line 274)
            dot_call_result_161760 = invoke(stypy.reporting.localization.Localization(__file__, 274, 24), dot_161756, *[modifier_161757, oldmatrix_161758], **kwargs_161759)
            
            # Assigning a type to the variable 'newmatrix' (line 274)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'newmatrix', dot_call_result_161760)
            
            # Assigning a Subscript to a Subscript (line 275):
            
            # Assigning a Subscript to a Subscript (line 275):
            
            # Obtaining the type of the subscript
            int_161761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 35), 'int')
            int_161762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 37), 'int')
            slice_161763 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 275, 25), int_161761, int_161762, None)
            int_161764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 40), 'int')
            # Getting the type of 'newmatrix' (line 275)
            newmatrix_161765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 25), 'newmatrix')
            # Obtaining the member '__getitem__' of a type (line 275)
            getitem___161766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 25), newmatrix_161765, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 275)
            subscript_call_result_161767 = invoke(stypy.reporting.localization.Localization(__file__, 275, 25), getitem___161766, (slice_161763, int_161764))
            
            # Getting the type of 'array' (line 275)
            array_161768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'array')
            int_161769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 20), 'int')
            slice_161770 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 275, 12), None, None, int_161769)
            # Storing an element on a container (line 275)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 12), array_161768, (slice_161770, subscript_call_result_161767))
            
            # Assigning a Subscript to a Subscript (line 276):
            
            # Assigning a Subscript to a Subscript (line 276):
            
            # Obtaining the type of the subscript
            int_161771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 36), 'int')
            int_161772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 38), 'int')
            slice_161773 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 276, 26), int_161771, int_161772, None)
            int_161774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 41), 'int')
            # Getting the type of 'newmatrix' (line 276)
            newmatrix_161775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 26), 'newmatrix')
            # Obtaining the member '__getitem__' of a type (line 276)
            getitem___161776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 26), newmatrix_161775, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 276)
            subscript_call_result_161777 = invoke(stypy.reporting.localization.Localization(__file__, 276, 26), getitem___161776, (slice_161773, int_161774))
            
            # Getting the type of 'array' (line 276)
            array_161778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'array')
            int_161779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 18), 'int')
            int_161780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 21), 'int')
            slice_161781 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 276, 12), int_161779, None, int_161780)
            # Storing an element on a container (line 276)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 12), array_161778, (slice_161781, subscript_call_result_161777))
            
            # Assigning a BinOp to a Name (line 277):
            
            # Assigning a BinOp to a Name (line 277):
            unicode_161782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 24), 'unicode', u'[')
            
            # Call to join(...): (line 277)
            # Processing the call arguments (line 277)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 277, 41, True)
            # Calculating comprehension expression
            # Getting the type of 'array' (line 277)
            array_161789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 57), 'array', False)
            comprehension_161790 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 41), array_161789)
            # Assigning a type to the variable 'x' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 41), 'x', comprehension_161790)
            
            # Call to str(...): (line 277)
            # Processing the call arguments (line 277)
            # Getting the type of 'x' (line 277)
            x_161786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 45), 'x', False)
            # Processing the call keyword arguments (line 277)
            kwargs_161787 = {}
            # Getting the type of 'str' (line 277)
            str_161785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 41), 'str', False)
            # Calling str(args, kwargs) (line 277)
            str_call_result_161788 = invoke(stypy.reporting.localization.Localization(__file__, 277, 41), str_161785, *[x_161786], **kwargs_161787)
            
            list_161791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 41), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 41), list_161791, str_call_result_161788)
            # Processing the call keyword arguments (line 277)
            kwargs_161792 = {}
            unicode_161783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 31), 'unicode', u' ')
            # Obtaining the member 'join' of a type (line 277)
            join_161784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 31), unicode_161783, 'join')
            # Calling join(args, kwargs) (line 277)
            join_call_result_161793 = invoke(stypy.reporting.localization.Localization(__file__, 277, 31), join_161784, *[list_161791], **kwargs_161792)
            
            # Applying the binary operator '+' (line 277)
            result_add_161794 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 24), '+', unicode_161782, join_call_result_161793)
            
            unicode_161795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 66), 'unicode', u']')
            # Applying the binary operator '+' (line 277)
            result_add_161796 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 64), '+', result_add_161794, unicode_161795)
            
            # Assigning a type to the variable 'as_string' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'as_string', result_add_161796)
            
            # Call to encode(...): (line 278)
            # Processing the call arguments (line 278)
            unicode_161799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 36), 'unicode', u'latin-1')
            # Processing the call keyword arguments (line 278)
            kwargs_161800 = {}
            # Getting the type of 'as_string' (line 278)
            as_string_161797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'as_string', False)
            # Obtaining the member 'encode' of a type (line 278)
            encode_161798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 19), as_string_161797, 'encode')
            # Calling encode(args, kwargs) (line 278)
            encode_call_result_161801 = invoke(stypy.reporting.localization.Localization(__file__, 278, 19), encode_161798, *[unicode_161799], **kwargs_161800)
            
            # Assigning a type to the variable 'stypy_return_type' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'stypy_return_type', encode_call_result_161801)
            
            # ################# End of 'fontmatrix(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fontmatrix' in the type store
            # Getting the type of 'stypy_return_type' (line 265)
            stypy_return_type_161802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_161802)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fontmatrix'
            return stypy_return_type_161802

        # Assigning a type to the variable 'fontmatrix' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'fontmatrix', fontmatrix)

        @norecursion
        def replace(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'replace'
            module_type_store = module_type_store.open_function_context('replace', 280, 8, False)
            
            # Passed parameters checking function
            replace.stypy_localization = localization
            replace.stypy_type_of_self = None
            replace.stypy_type_store = module_type_store
            replace.stypy_function_name = 'replace'
            replace.stypy_param_names_list = ['fun']
            replace.stypy_varargs_param_name = None
            replace.stypy_kwargs_param_name = None
            replace.stypy_call_defaults = defaults
            replace.stypy_call_varargs = varargs
            replace.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'replace', ['fun'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'replace', localization, ['fun'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'replace(...)' code ##################


            @norecursion
            def replacer(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'replacer'
                module_type_store = module_type_store.open_function_context('replacer', 281, 12, False)
                
                # Passed parameters checking function
                replacer.stypy_localization = localization
                replacer.stypy_type_of_self = None
                replacer.stypy_type_store = module_type_store
                replacer.stypy_function_name = 'replacer'
                replacer.stypy_param_names_list = ['tokens']
                replacer.stypy_varargs_param_name = None
                replacer.stypy_kwargs_param_name = None
                replacer.stypy_call_defaults = defaults
                replacer.stypy_call_varargs = varargs
                replacer.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'replacer', ['tokens'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'replacer', localization, ['tokens'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'replacer(...)' code ##################

                
                # Assigning a Call to a Tuple (line 282):
                
                # Assigning a Call to a Name:
                
                # Call to next(...): (line 282)
                # Processing the call arguments (line 282)
                # Getting the type of 'tokens' (line 282)
                tokens_161804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 36), 'tokens', False)
                # Processing the call keyword arguments (line 282)
                kwargs_161805 = {}
                # Getting the type of 'next' (line 282)
                next_161803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 31), 'next', False)
                # Calling next(args, kwargs) (line 282)
                next_call_result_161806 = invoke(stypy.reporting.localization.Localization(__file__, 282, 31), next_161803, *[tokens_161804], **kwargs_161805)
                
                # Assigning a type to the variable 'call_assignment_160932' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'call_assignment_160932', next_call_result_161806)
                
                # Assigning a Call to a Name (line 282):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_161809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 16), 'int')
                # Processing the call keyword arguments
                kwargs_161810 = {}
                # Getting the type of 'call_assignment_160932' (line 282)
                call_assignment_160932_161807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'call_assignment_160932', False)
                # Obtaining the member '__getitem__' of a type (line 282)
                getitem___161808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 16), call_assignment_160932_161807, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_161811 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161808, *[int_161809], **kwargs_161810)
                
                # Assigning a type to the variable 'call_assignment_160933' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'call_assignment_160933', getitem___call_result_161811)
                
                # Assigning a Name to a Name (line 282):
                # Getting the type of 'call_assignment_160933' (line 282)
                call_assignment_160933_161812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'call_assignment_160933')
                # Assigning a type to the variable 'token' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'token', call_assignment_160933_161812)
                
                # Assigning a Call to a Name (line 282):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_161815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 16), 'int')
                # Processing the call keyword arguments
                kwargs_161816 = {}
                # Getting the type of 'call_assignment_160932' (line 282)
                call_assignment_160932_161813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'call_assignment_160932', False)
                # Obtaining the member '__getitem__' of a type (line 282)
                getitem___161814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 16), call_assignment_160932_161813, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_161817 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161814, *[int_161815], **kwargs_161816)
                
                # Assigning a type to the variable 'call_assignment_160934' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'call_assignment_160934', getitem___call_result_161817)
                
                # Assigning a Name to a Name (line 282):
                # Getting the type of 'call_assignment_160934' (line 282)
                call_assignment_160934_161818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'call_assignment_160934')
                # Assigning a type to the variable 'value' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 23), 'value', call_assignment_160934_161818)
                # Creating a generator
                
                # Call to bytes(...): (line 283)
                # Processing the call arguments (line 283)
                # Getting the type of 'value' (line 283)
                value_161820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 28), 'value', False)
                # Processing the call keyword arguments (line 283)
                kwargs_161821 = {}
                # Getting the type of 'bytes' (line 283)
                bytes_161819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 22), 'bytes', False)
                # Calling bytes(args, kwargs) (line 283)
                bytes_call_result_161822 = invoke(stypy.reporting.localization.Localization(__file__, 283, 22), bytes_161819, *[value_161820], **kwargs_161821)
                
                GeneratorType_161823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 16), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 16), GeneratorType_161823, bytes_call_result_161822)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'stypy_return_type', GeneratorType_161823)
                
                # Assigning a Call to a Tuple (line 284):
                
                # Assigning a Call to a Name:
                
                # Call to next(...): (line 284)
                # Processing the call arguments (line 284)
                # Getting the type of 'tokens' (line 284)
                tokens_161825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 36), 'tokens', False)
                # Processing the call keyword arguments (line 284)
                kwargs_161826 = {}
                # Getting the type of 'next' (line 284)
                next_161824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 31), 'next', False)
                # Calling next(args, kwargs) (line 284)
                next_call_result_161827 = invoke(stypy.reporting.localization.Localization(__file__, 284, 31), next_161824, *[tokens_161825], **kwargs_161826)
                
                # Assigning a type to the variable 'call_assignment_160935' (line 284)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'call_assignment_160935', next_call_result_161827)
                
                # Assigning a Call to a Name (line 284):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_161830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 16), 'int')
                # Processing the call keyword arguments
                kwargs_161831 = {}
                # Getting the type of 'call_assignment_160935' (line 284)
                call_assignment_160935_161828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'call_assignment_160935', False)
                # Obtaining the member '__getitem__' of a type (line 284)
                getitem___161829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), call_assignment_160935_161828, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_161832 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161829, *[int_161830], **kwargs_161831)
                
                # Assigning a type to the variable 'call_assignment_160936' (line 284)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'call_assignment_160936', getitem___call_result_161832)
                
                # Assigning a Name to a Name (line 284):
                # Getting the type of 'call_assignment_160936' (line 284)
                call_assignment_160936_161833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'call_assignment_160936')
                # Assigning a type to the variable 'token' (line 284)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'token', call_assignment_160936_161833)
                
                # Assigning a Call to a Name (line 284):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_161836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 16), 'int')
                # Processing the call keyword arguments
                kwargs_161837 = {}
                # Getting the type of 'call_assignment_160935' (line 284)
                call_assignment_160935_161834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'call_assignment_160935', False)
                # Obtaining the member '__getitem__' of a type (line 284)
                getitem___161835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), call_assignment_160935_161834, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_161838 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161835, *[int_161836], **kwargs_161837)
                
                # Assigning a type to the variable 'call_assignment_160937' (line 284)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'call_assignment_160937', getitem___call_result_161838)
                
                # Assigning a Name to a Name (line 284):
                # Getting the type of 'call_assignment_160937' (line 284)
                call_assignment_160937_161839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'call_assignment_160937')
                # Assigning a type to the variable 'value' (line 284)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'value', call_assignment_160937_161839)
                
                
                # Getting the type of 'token' (line 285)
                token_161840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 22), 'token')
                # Getting the type of 'cls' (line 285)
                cls_161841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 31), 'cls')
                # Obtaining the member '_whitespace' of a type (line 285)
                _whitespace_161842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 31), cls_161841, '_whitespace')
                # Applying the binary operator 'is' (line 285)
                result_is__161843 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 22), 'is', token_161840, _whitespace_161842)
                
                # Testing the type of an if condition (line 285)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 16), result_is__161843)
                # SSA begins for while statement (line 285)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                # Creating a generator
                
                # Call to bytes(...): (line 286)
                # Processing the call arguments (line 286)
                # Getting the type of 'value' (line 286)
                value_161845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 32), 'value', False)
                # Processing the call keyword arguments (line 286)
                kwargs_161846 = {}
                # Getting the type of 'bytes' (line 286)
                bytes_161844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'bytes', False)
                # Calling bytes(args, kwargs) (line 286)
                bytes_call_result_161847 = invoke(stypy.reporting.localization.Localization(__file__, 286, 26), bytes_161844, *[value_161845], **kwargs_161846)
                
                GeneratorType_161848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 20), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 20), GeneratorType_161848, bytes_call_result_161847)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'stypy_return_type', GeneratorType_161848)
                
                # Assigning a Call to a Tuple (line 287):
                
                # Assigning a Call to a Name:
                
                # Call to next(...): (line 287)
                # Processing the call arguments (line 287)
                # Getting the type of 'tokens' (line 287)
                tokens_161850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 40), 'tokens', False)
                # Processing the call keyword arguments (line 287)
                kwargs_161851 = {}
                # Getting the type of 'next' (line 287)
                next_161849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 35), 'next', False)
                # Calling next(args, kwargs) (line 287)
                next_call_result_161852 = invoke(stypy.reporting.localization.Localization(__file__, 287, 35), next_161849, *[tokens_161850], **kwargs_161851)
                
                # Assigning a type to the variable 'call_assignment_160938' (line 287)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'call_assignment_160938', next_call_result_161852)
                
                # Assigning a Call to a Name (line 287):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_161855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 20), 'int')
                # Processing the call keyword arguments
                kwargs_161856 = {}
                # Getting the type of 'call_assignment_160938' (line 287)
                call_assignment_160938_161853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'call_assignment_160938', False)
                # Obtaining the member '__getitem__' of a type (line 287)
                getitem___161854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 20), call_assignment_160938_161853, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_161857 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161854, *[int_161855], **kwargs_161856)
                
                # Assigning a type to the variable 'call_assignment_160939' (line 287)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'call_assignment_160939', getitem___call_result_161857)
                
                # Assigning a Name to a Name (line 287):
                # Getting the type of 'call_assignment_160939' (line 287)
                call_assignment_160939_161858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'call_assignment_160939')
                # Assigning a type to the variable 'token' (line 287)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'token', call_assignment_160939_161858)
                
                # Assigning a Call to a Name (line 287):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_161861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 20), 'int')
                # Processing the call keyword arguments
                kwargs_161862 = {}
                # Getting the type of 'call_assignment_160938' (line 287)
                call_assignment_160938_161859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'call_assignment_160938', False)
                # Obtaining the member '__getitem__' of a type (line 287)
                getitem___161860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 20), call_assignment_160938_161859, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_161863 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161860, *[int_161861], **kwargs_161862)
                
                # Assigning a type to the variable 'call_assignment_160940' (line 287)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'call_assignment_160940', getitem___call_result_161863)
                
                # Assigning a Name to a Name (line 287):
                # Getting the type of 'call_assignment_160940' (line 287)
                call_assignment_160940_161864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'call_assignment_160940')
                # Assigning a type to the variable 'value' (line 287)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 27), 'value', call_assignment_160940_161864)
                # SSA join for while statement (line 285)
                module_type_store = module_type_store.join_ssa_context()
                
                
                
                # Getting the type of 'value' (line 288)
                value_161865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'value')
                str_161866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 28), 'str', '[')
                # Applying the binary operator '!=' (line 288)
                result_ne_161867 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 19), '!=', value_161865, str_161866)
                
                # Testing the type of an if condition (line 288)
                if_condition_161868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 16), result_ne_161867)
                # Assigning a type to the variable 'if_condition_161868' (line 288)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'if_condition_161868', if_condition_161868)
                # SSA begins for if statement (line 288)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Creating a generator
                
                # Call to bytes(...): (line 289)
                # Processing the call arguments (line 289)
                
                # Call to fun(...): (line 289)
                # Processing the call arguments (line 289)
                # Getting the type of 'value' (line 289)
                value_161871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 36), 'value', False)
                # Processing the call keyword arguments (line 289)
                kwargs_161872 = {}
                # Getting the type of 'fun' (line 289)
                fun_161870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 32), 'fun', False)
                # Calling fun(args, kwargs) (line 289)
                fun_call_result_161873 = invoke(stypy.reporting.localization.Localization(__file__, 289, 32), fun_161870, *[value_161871], **kwargs_161872)
                
                # Processing the call keyword arguments (line 289)
                kwargs_161874 = {}
                # Getting the type of 'bytes' (line 289)
                bytes_161869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'bytes', False)
                # Calling bytes(args, kwargs) (line 289)
                bytes_call_result_161875 = invoke(stypy.reporting.localization.Localization(__file__, 289, 26), bytes_161869, *[fun_call_result_161873], **kwargs_161874)
                
                GeneratorType_161876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 20), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 20), GeneratorType_161876, bytes_call_result_161875)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'stypy_return_type', GeneratorType_161876)
                # SSA branch for the else part of an if statement (line 288)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Str to a Name (line 291):
                
                # Assigning a Str to a Name (line 291):
                str_161877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 29), 'str', '')
                # Assigning a type to the variable 'result' (line 291)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'result', str_161877)
                
                
                # Getting the type of 'value' (line 292)
                value_161878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'value')
                str_161879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 35), 'str', ']')
                # Applying the binary operator '!=' (line 292)
                result_ne_161880 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 26), '!=', value_161878, str_161879)
                
                # Testing the type of an if condition (line 292)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 20), result_ne_161880)
                # SSA begins for while statement (line 292)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Getting the type of 'result' (line 293)
                result_161881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'result')
                # Getting the type of 'value' (line 293)
                value_161882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 34), 'value')
                # Applying the binary operator '+=' (line 293)
                result_iadd_161883 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 24), '+=', result_161881, value_161882)
                # Assigning a type to the variable 'result' (line 293)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'result', result_iadd_161883)
                
                
                # Assigning a Call to a Tuple (line 294):
                
                # Assigning a Call to a Name:
                
                # Call to next(...): (line 294)
                # Processing the call arguments (line 294)
                # Getting the type of 'tokens' (line 294)
                tokens_161885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 44), 'tokens', False)
                # Processing the call keyword arguments (line 294)
                kwargs_161886 = {}
                # Getting the type of 'next' (line 294)
                next_161884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 39), 'next', False)
                # Calling next(args, kwargs) (line 294)
                next_call_result_161887 = invoke(stypy.reporting.localization.Localization(__file__, 294, 39), next_161884, *[tokens_161885], **kwargs_161886)
                
                # Assigning a type to the variable 'call_assignment_160941' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'call_assignment_160941', next_call_result_161887)
                
                # Assigning a Call to a Name (line 294):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_161890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 24), 'int')
                # Processing the call keyword arguments
                kwargs_161891 = {}
                # Getting the type of 'call_assignment_160941' (line 294)
                call_assignment_160941_161888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'call_assignment_160941', False)
                # Obtaining the member '__getitem__' of a type (line 294)
                getitem___161889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 24), call_assignment_160941_161888, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_161892 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161889, *[int_161890], **kwargs_161891)
                
                # Assigning a type to the variable 'call_assignment_160942' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'call_assignment_160942', getitem___call_result_161892)
                
                # Assigning a Name to a Name (line 294):
                # Getting the type of 'call_assignment_160942' (line 294)
                call_assignment_160942_161893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'call_assignment_160942')
                # Assigning a type to the variable 'token' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'token', call_assignment_160942_161893)
                
                # Assigning a Call to a Name (line 294):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_161896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 24), 'int')
                # Processing the call keyword arguments
                kwargs_161897 = {}
                # Getting the type of 'call_assignment_160941' (line 294)
                call_assignment_160941_161894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'call_assignment_160941', False)
                # Obtaining the member '__getitem__' of a type (line 294)
                getitem___161895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 24), call_assignment_160941_161894, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_161898 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___161895, *[int_161896], **kwargs_161897)
                
                # Assigning a type to the variable 'call_assignment_160943' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'call_assignment_160943', getitem___call_result_161898)
                
                # Assigning a Name to a Name (line 294):
                # Getting the type of 'call_assignment_160943' (line 294)
                call_assignment_160943_161899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'call_assignment_160943')
                # Assigning a type to the variable 'value' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 31), 'value', call_assignment_160943_161899)
                # SSA join for while statement (line 292)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Getting the type of 'result' (line 295)
                result_161900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'result')
                # Getting the type of 'value' (line 295)
                value_161901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 30), 'value')
                # Applying the binary operator '+=' (line 295)
                result_iadd_161902 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 20), '+=', result_161900, value_161901)
                # Assigning a type to the variable 'result' (line 295)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'result', result_iadd_161902)
                
                # Creating a generator
                
                # Call to fun(...): (line 296)
                # Processing the call arguments (line 296)
                # Getting the type of 'result' (line 296)
                result_161904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 30), 'result', False)
                # Processing the call keyword arguments (line 296)
                kwargs_161905 = {}
                # Getting the type of 'fun' (line 296)
                fun_161903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 26), 'fun', False)
                # Calling fun(args, kwargs) (line 296)
                fun_call_result_161906 = invoke(stypy.reporting.localization.Localization(__file__, 296, 26), fun_161903, *[result_161904], **kwargs_161905)
                
                GeneratorType_161907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 20), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 20), GeneratorType_161907, fun_call_result_161906)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'stypy_return_type', GeneratorType_161907)
                # SSA join for if statement (line 288)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # ################# End of 'replacer(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'replacer' in the type store
                # Getting the type of 'stypy_return_type' (line 281)
                stypy_return_type_161908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_161908)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'replacer'
                return stypy_return_type_161908

            # Assigning a type to the variable 'replacer' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'replacer', replacer)
            # Getting the type of 'replacer' (line 297)
            replacer_161909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'replacer')
            # Assigning a type to the variable 'stypy_return_type' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'stypy_return_type', replacer_161909)
            
            # ################# End of 'replace(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'replace' in the type store
            # Getting the type of 'stypy_return_type' (line 280)
            stypy_return_type_161910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_161910)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'replace'
            return stypy_return_type_161910

        # Assigning a type to the variable 'replace' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'replace', replace)

        @norecursion
        def suppress(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'suppress'
            module_type_store = module_type_store.open_function_context('suppress', 299, 8, False)
            
            # Passed parameters checking function
            suppress.stypy_localization = localization
            suppress.stypy_type_of_self = None
            suppress.stypy_type_store = module_type_store
            suppress.stypy_function_name = 'suppress'
            suppress.stypy_param_names_list = ['tokens']
            suppress.stypy_varargs_param_name = None
            suppress.stypy_kwargs_param_name = None
            suppress.stypy_call_defaults = defaults
            suppress.stypy_call_varargs = varargs
            suppress.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'suppress', ['tokens'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'suppress', localization, ['tokens'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'suppress(...)' code ##################

            
            
            # Call to takewhile(...): (line 300)
            # Processing the call arguments (line 300)

            @norecursion
            def _stypy_temp_lambda_25(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_25'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_25', 300, 41, True)
                # Passed parameters checking function
                _stypy_temp_lambda_25.stypy_localization = localization
                _stypy_temp_lambda_25.stypy_type_of_self = None
                _stypy_temp_lambda_25.stypy_type_store = module_type_store
                _stypy_temp_lambda_25.stypy_function_name = '_stypy_temp_lambda_25'
                _stypy_temp_lambda_25.stypy_param_names_list = ['x']
                _stypy_temp_lambda_25.stypy_varargs_param_name = None
                _stypy_temp_lambda_25.stypy_kwargs_param_name = None
                _stypy_temp_lambda_25.stypy_call_defaults = defaults
                _stypy_temp_lambda_25.stypy_call_varargs = varargs
                _stypy_temp_lambda_25.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_25', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_25', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                
                # Obtaining the type of the subscript
                int_161913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 53), 'int')
                # Getting the type of 'x' (line 300)
                x_161914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 51), 'x', False)
                # Obtaining the member '__getitem__' of a type (line 300)
                getitem___161915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 51), x_161914, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 300)
                subscript_call_result_161916 = invoke(stypy.reporting.localization.Localization(__file__, 300, 51), getitem___161915, int_161913)
                
                str_161917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 59), 'str', 'def')
                # Applying the binary operator '!=' (line 300)
                result_ne_161918 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 51), '!=', subscript_call_result_161916, str_161917)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 300)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 41), 'stypy_return_type', result_ne_161918)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_25' in the type store
                # Getting the type of 'stypy_return_type' (line 300)
                stypy_return_type_161919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 41), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_161919)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_25'
                return stypy_return_type_161919

            # Assigning a type to the variable '_stypy_temp_lambda_25' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 41), '_stypy_temp_lambda_25', _stypy_temp_lambda_25)
            # Getting the type of '_stypy_temp_lambda_25' (line 300)
            _stypy_temp_lambda_25_161920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 41), '_stypy_temp_lambda_25')
            # Getting the type of 'tokens' (line 300)
            tokens_161921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 67), 'tokens', False)
            # Processing the call keyword arguments (line 300)
            kwargs_161922 = {}
            # Getting the type of 'itertools' (line 300)
            itertools_161911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 21), 'itertools', False)
            # Obtaining the member 'takewhile' of a type (line 300)
            takewhile_161912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 21), itertools_161911, 'takewhile')
            # Calling takewhile(args, kwargs) (line 300)
            takewhile_call_result_161923 = invoke(stypy.reporting.localization.Localization(__file__, 300, 21), takewhile_161912, *[_stypy_temp_lambda_25_161920, tokens_161921], **kwargs_161922)
            
            # Testing the type of a for loop iterable (line 300)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 300, 12), takewhile_call_result_161923)
            # Getting the type of the for loop variable (line 300)
            for_loop_var_161924 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 300, 12), takewhile_call_result_161923)
            # Assigning a type to the variable 'x' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'x', for_loop_var_161924)
            # SSA begins for a for statement (line 300)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            pass
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Creating a generator
            str_161925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 18), 'str', '')
            GeneratorType_161926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 12), GeneratorType_161926, str_161925)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'stypy_return_type', GeneratorType_161926)
            
            # ################# End of 'suppress(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'suppress' in the type store
            # Getting the type of 'stypy_return_type' (line 299)
            stypy_return_type_161927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_161927)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'suppress'
            return stypy_return_type_161927

        # Assigning a type to the variable 'suppress' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'suppress', suppress)
        
        # Assigning a Dict to a Name (line 304):
        
        # Assigning a Dict to a Name (line 304):
        
        # Obtaining an instance of the builtin type 'dict' (line 304)
        dict_161928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 304)
        # Adding element type (key, value) (line 304)
        str_161929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 17), 'str', '/FontName')
        
        # Call to replace(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'fontname' (line 304)
        fontname_161931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 39), 'fontname', False)
        # Processing the call keyword arguments (line 304)
        kwargs_161932 = {}
        # Getting the type of 'replace' (line 304)
        replace_161930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 31), 'replace', False)
        # Calling replace(args, kwargs) (line 304)
        replace_call_result_161933 = invoke(stypy.reporting.localization.Localization(__file__, 304, 31), replace_161930, *[fontname_161931], **kwargs_161932)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 16), dict_161928, (str_161929, replace_call_result_161933))
        # Adding element type (key, value) (line 304)
        str_161934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 17), 'str', '/ItalicAngle')
        
        # Call to replace(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'italicangle' (line 305)
        italicangle_161936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 42), 'italicangle', False)
        # Processing the call keyword arguments (line 305)
        kwargs_161937 = {}
        # Getting the type of 'replace' (line 305)
        replace_161935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 34), 'replace', False)
        # Calling replace(args, kwargs) (line 305)
        replace_call_result_161938 = invoke(stypy.reporting.localization.Localization(__file__, 305, 34), replace_161935, *[italicangle_161936], **kwargs_161937)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 16), dict_161928, (str_161934, replace_call_result_161938))
        # Adding element type (key, value) (line 304)
        str_161939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 17), 'str', '/FontMatrix')
        
        # Call to replace(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'fontmatrix' (line 306)
        fontmatrix_161941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 41), 'fontmatrix', False)
        # Processing the call keyword arguments (line 306)
        kwargs_161942 = {}
        # Getting the type of 'replace' (line 306)
        replace_161940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 33), 'replace', False)
        # Calling replace(args, kwargs) (line 306)
        replace_call_result_161943 = invoke(stypy.reporting.localization.Localization(__file__, 306, 33), replace_161940, *[fontmatrix_161941], **kwargs_161942)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 16), dict_161928, (str_161939, replace_call_result_161943))
        # Adding element type (key, value) (line 304)
        str_161944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 17), 'str', '/UniqueID')
        # Getting the type of 'suppress' (line 307)
        suppress_161945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 31), 'suppress')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 16), dict_161928, (str_161944, suppress_161945))
        
        # Assigning a type to the variable 'table' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'table', dict_161928)
        
        # Getting the type of 'tokens' (line 309)
        tokens_161946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'tokens')
        # Testing the type of a for loop iterable (line 309)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 309, 8), tokens_161946)
        # Getting the type of the for loop variable (line 309)
        for_loop_var_161947 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 309, 8), tokens_161946)
        # Assigning a type to the variable 'token' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'token', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 8), for_loop_var_161947))
        # Assigning a type to the variable 'value' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 8), for_loop_var_161947))
        # SSA begins for a for statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'token' (line 310)
        token_161948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'token')
        # Getting the type of 'cls' (line 310)
        cls_161949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 24), 'cls')
        # Obtaining the member '_name' of a type (line 310)
        _name_161950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 24), cls_161949, '_name')
        # Applying the binary operator 'is' (line 310)
        result_is__161951 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 15), 'is', token_161948, _name_161950)
        
        
        # Getting the type of 'value' (line 310)
        value_161952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 38), 'value')
        # Getting the type of 'table' (line 310)
        table_161953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 47), 'table')
        # Applying the binary operator 'in' (line 310)
        result_contains_161954 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 38), 'in', value_161952, table_161953)
        
        # Applying the binary operator 'and' (line 310)
        result_and_keyword_161955 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 15), 'and', result_is__161951, result_contains_161954)
        
        # Testing the type of an if condition (line 310)
        if_condition_161956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 12), result_and_keyword_161955)
        # Assigning a type to the variable 'if_condition_161956' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'if_condition_161956', if_condition_161956)
        # SSA begins for if statement (line 310)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to (...): (line 311)
        # Processing the call arguments (line 311)
        
        # Call to chain(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_161963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        
        # Obtaining an instance of the builtin type 'tuple' (line 311)
        tuple_161964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 60), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 311)
        # Adding element type (line 311)
        # Getting the type of 'token' (line 311)
        token_161965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 60), 'token', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 60), tuple_161964, token_161965)
        # Adding element type (line 311)
        # Getting the type of 'value' (line 311)
        value_161966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 67), 'value', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 60), tuple_161964, value_161966)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 58), list_161963, tuple_161964)
        
        # Getting the type of 'tokens' (line 312)
        tokens_161967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 58), 'tokens', False)
        # Processing the call keyword arguments (line 311)
        kwargs_161968 = {}
        # Getting the type of 'itertools' (line 311)
        itertools_161961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 42), 'itertools', False)
        # Obtaining the member 'chain' of a type (line 311)
        chain_161962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 42), itertools_161961, 'chain')
        # Calling chain(args, kwargs) (line 311)
        chain_call_result_161969 = invoke(stypy.reporting.localization.Localization(__file__, 311, 42), chain_161962, *[list_161963, tokens_161967], **kwargs_161968)
        
        # Processing the call keyword arguments (line 311)
        kwargs_161970 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'value' (line 311)
        value_161957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 35), 'value', False)
        # Getting the type of 'table' (line 311)
        table_161958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 29), 'table', False)
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___161959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 29), table_161958, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_161960 = invoke(stypy.reporting.localization.Localization(__file__, 311, 29), getitem___161959, value_161957)
        
        # Calling (args, kwargs) (line 311)
        _call_result_161971 = invoke(stypy.reporting.localization.Localization(__file__, 311, 29), subscript_call_result_161960, *[chain_call_result_161969], **kwargs_161970)
        
        # Testing the type of a for loop iterable (line 311)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 311, 16), _call_result_161971)
        # Getting the type of the for loop variable (line 311)
        for_loop_var_161972 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 311, 16), _call_result_161971)
        # Assigning a type to the variable 'value' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'value', for_loop_var_161972)
        # SSA begins for a for statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Creating a generator
        # Getting the type of 'value' (line 313)
        value_161973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 26), 'value')
        GeneratorType_161974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 20), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 20), GeneratorType_161974, value_161973)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 20), 'stypy_return_type', GeneratorType_161974)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 310)
        module_type_store.open_ssa_branch('else')
        # Creating a generator
        # Getting the type of 'value' (line 315)
        value_161975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 22), 'value')
        GeneratorType_161976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 16), GeneratorType_161976, value_161975)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'stypy_return_type', GeneratorType_161976)
        # SSA join for if statement (line 310)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_transformer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_transformer' in the type store
        # Getting the type of 'stypy_return_type' (line 252)
        stypy_return_type_161977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_161977)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_transformer'
        return stypy_return_type_161977


    @norecursion
    def transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'transform'
        module_type_store = module_type_store.open_function_context('transform', 317, 4, False)
        # Assigning a type to the variable 'self' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type1Font.transform.__dict__.__setitem__('stypy_localization', localization)
        Type1Font.transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type1Font.transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type1Font.transform.__dict__.__setitem__('stypy_function_name', 'Type1Font.transform')
        Type1Font.transform.__dict__.__setitem__('stypy_param_names_list', ['effects'])
        Type1Font.transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type1Font.transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type1Font.transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type1Font.transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type1Font.transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type1Font.transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type1Font.transform', ['effects'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transform', localization, ['effects'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transform(...)' code ##################

        unicode_161978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, (-1)), 'unicode', u"\n        Transform the font by slanting or extending. *effects* should\n        be a dict where ``effects['slant']`` is the tangent of the\n        angle that the font is to be slanted to the right (so negative\n        values slant to the left) and ``effects['extend']`` is the\n        multiplier by which the font is to be extended (so values less\n        than 1.0 condense). Returns a new :class:`Type1Font` object.\n        ")
        
        # Call to BytesIO(...): (line 326)
        # Processing the call keyword arguments (line 326)
        kwargs_161981 = {}
        # Getting the type of 'io' (line 326)
        io_161979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 13), 'io', False)
        # Obtaining the member 'BytesIO' of a type (line 326)
        BytesIO_161980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 13), io_161979, 'BytesIO')
        # Calling BytesIO(args, kwargs) (line 326)
        BytesIO_call_result_161982 = invoke(stypy.reporting.localization.Localization(__file__, 326, 13), BytesIO_161980, *[], **kwargs_161981)
        
        with_161983 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 326, 13), BytesIO_call_result_161982, 'with parameter', '__enter__', '__exit__')

        if with_161983:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 326)
            enter___161984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 13), BytesIO_call_result_161982, '__enter__')
            with_enter_161985 = invoke(stypy.reporting.localization.Localization(__file__, 326, 13), enter___161984)
            # Assigning a type to the variable 'buffer' (line 326)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 13), 'buffer', with_enter_161985)
            
            # Assigning a Call to a Name (line 327):
            
            # Assigning a Call to a Name (line 327):
            
            # Call to _tokens(...): (line 327)
            # Processing the call arguments (line 327)
            
            # Obtaining the type of the subscript
            int_161988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 48), 'int')
            # Getting the type of 'self' (line 327)
            self_161989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 37), 'self', False)
            # Obtaining the member 'parts' of a type (line 327)
            parts_161990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 37), self_161989, 'parts')
            # Obtaining the member '__getitem__' of a type (line 327)
            getitem___161991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 37), parts_161990, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 327)
            subscript_call_result_161992 = invoke(stypy.reporting.localization.Localization(__file__, 327, 37), getitem___161991, int_161988)
            
            # Processing the call keyword arguments (line 327)
            kwargs_161993 = {}
            # Getting the type of 'self' (line 327)
            self_161986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 24), 'self', False)
            # Obtaining the member '_tokens' of a type (line 327)
            _tokens_161987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 24), self_161986, '_tokens')
            # Calling _tokens(args, kwargs) (line 327)
            _tokens_call_result_161994 = invoke(stypy.reporting.localization.Localization(__file__, 327, 24), _tokens_161987, *[subscript_call_result_161992], **kwargs_161993)
            
            # Assigning a type to the variable 'tokenizer' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'tokenizer', _tokens_call_result_161994)
            
            # Assigning a Call to a Name (line 328):
            
            # Assigning a Call to a Name (line 328):
            
            # Call to _transformer(...): (line 328)
            # Processing the call arguments (line 328)
            # Getting the type of 'tokenizer' (line 328)
            tokenizer_161997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 45), 'tokenizer', False)
            # Processing the call keyword arguments (line 328)
            
            # Call to get(...): (line 329)
            # Processing the call arguments (line 329)
            unicode_162000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 63), 'unicode', u'slant')
            float_162001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 72), 'float')
            # Processing the call keyword arguments (line 329)
            kwargs_162002 = {}
            # Getting the type of 'effects' (line 329)
            effects_161998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 51), 'effects', False)
            # Obtaining the member 'get' of a type (line 329)
            get_161999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 51), effects_161998, 'get')
            # Calling get(args, kwargs) (line 329)
            get_call_result_162003 = invoke(stypy.reporting.localization.Localization(__file__, 329, 51), get_161999, *[unicode_162000, float_162001], **kwargs_162002)
            
            keyword_162004 = get_call_result_162003
            
            # Call to get(...): (line 330)
            # Processing the call arguments (line 330)
            unicode_162007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 64), 'unicode', u'extend')
            float_162008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 74), 'float')
            # Processing the call keyword arguments (line 330)
            kwargs_162009 = {}
            # Getting the type of 'effects' (line 330)
            effects_162005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 52), 'effects', False)
            # Obtaining the member 'get' of a type (line 330)
            get_162006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 52), effects_162005, 'get')
            # Calling get(args, kwargs) (line 330)
            get_call_result_162010 = invoke(stypy.reporting.localization.Localization(__file__, 330, 52), get_162006, *[unicode_162007, float_162008], **kwargs_162009)
            
            keyword_162011 = get_call_result_162010
            kwargs_162012 = {'slant': keyword_162004, 'extend': keyword_162011}
            # Getting the type of 'self' (line 328)
            self_161995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 27), 'self', False)
            # Obtaining the member '_transformer' of a type (line 328)
            _transformer_161996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 27), self_161995, '_transformer')
            # Calling _transformer(args, kwargs) (line 328)
            _transformer_call_result_162013 = invoke(stypy.reporting.localization.Localization(__file__, 328, 27), _transformer_161996, *[tokenizer_161997], **kwargs_162012)
            
            # Assigning a type to the variable 'transformed' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'transformed', _transformer_call_result_162013)
            
            # Call to list(...): (line 331)
            # Processing the call arguments (line 331)
            
            # Call to map(...): (line 331)
            # Processing the call arguments (line 331)
            # Getting the type of 'buffer' (line 331)
            buffer_162016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'buffer', False)
            # Obtaining the member 'write' of a type (line 331)
            write_162017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 21), buffer_162016, 'write')
            # Getting the type of 'transformed' (line 331)
            transformed_162018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 35), 'transformed', False)
            # Processing the call keyword arguments (line 331)
            kwargs_162019 = {}
            # Getting the type of 'map' (line 331)
            map_162015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 17), 'map', False)
            # Calling map(args, kwargs) (line 331)
            map_call_result_162020 = invoke(stypy.reporting.localization.Localization(__file__, 331, 17), map_162015, *[write_162017, transformed_162018], **kwargs_162019)
            
            # Processing the call keyword arguments (line 331)
            kwargs_162021 = {}
            # Getting the type of 'list' (line 331)
            list_162014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'list', False)
            # Calling list(args, kwargs) (line 331)
            list_call_result_162022 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), list_162014, *[map_call_result_162020], **kwargs_162021)
            
            
            # Call to Type1Font(...): (line 332)
            # Processing the call arguments (line 332)
            
            # Obtaining an instance of the builtin type 'tuple' (line 332)
            tuple_162024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 332)
            # Adding element type (line 332)
            
            # Call to getvalue(...): (line 332)
            # Processing the call keyword arguments (line 332)
            kwargs_162027 = {}
            # Getting the type of 'buffer' (line 332)
            buffer_162025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 30), 'buffer', False)
            # Obtaining the member 'getvalue' of a type (line 332)
            getvalue_162026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 30), buffer_162025, 'getvalue')
            # Calling getvalue(args, kwargs) (line 332)
            getvalue_call_result_162028 = invoke(stypy.reporting.localization.Localization(__file__, 332, 30), getvalue_162026, *[], **kwargs_162027)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 30), tuple_162024, getvalue_call_result_162028)
            # Adding element type (line 332)
            
            # Obtaining the type of the subscript
            int_162029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 60), 'int')
            # Getting the type of 'self' (line 332)
            self_162030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 49), 'self', False)
            # Obtaining the member 'parts' of a type (line 332)
            parts_162031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 49), self_162030, 'parts')
            # Obtaining the member '__getitem__' of a type (line 332)
            getitem___162032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 49), parts_162031, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 332)
            subscript_call_result_162033 = invoke(stypy.reporting.localization.Localization(__file__, 332, 49), getitem___162032, int_162029)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 30), tuple_162024, subscript_call_result_162033)
            # Adding element type (line 332)
            
            # Obtaining the type of the subscript
            int_162034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 75), 'int')
            # Getting the type of 'self' (line 332)
            self_162035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 64), 'self', False)
            # Obtaining the member 'parts' of a type (line 332)
            parts_162036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 64), self_162035, 'parts')
            # Obtaining the member '__getitem__' of a type (line 332)
            getitem___162037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 64), parts_162036, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 332)
            subscript_call_result_162038 = invoke(stypy.reporting.localization.Localization(__file__, 332, 64), getitem___162037, int_162034)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 30), tuple_162024, subscript_call_result_162038)
            
            # Processing the call keyword arguments (line 332)
            kwargs_162039 = {}
            # Getting the type of 'Type1Font' (line 332)
            Type1Font_162023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'Type1Font', False)
            # Calling Type1Font(args, kwargs) (line 332)
            Type1Font_call_result_162040 = invoke(stypy.reporting.localization.Localization(__file__, 332, 19), Type1Font_162023, *[tuple_162024], **kwargs_162039)
            
            # Assigning a type to the variable 'stypy_return_type' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'stypy_return_type', Type1Font_call_result_162040)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 326)
            exit___162041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 13), BytesIO_call_result_161982, '__exit__')
            with_exit_162042 = invoke(stypy.reporting.localization.Localization(__file__, 326, 13), exit___162041, None, None, None)

        
        # ################# End of 'transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transform' in the type store
        # Getting the type of 'stypy_return_type' (line 317)
        stypy_return_type_162043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_162043)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transform'
        return stypy_return_type_162043


# Assigning a type to the variable 'Type1Font' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'Type1Font', Type1Font)

# Assigning a Tuple to a Name (line 57):

# Obtaining an instance of the builtin type 'tuple' (line 57)
tuple_162044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 57)
# Adding element type (line 57)
unicode_162045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 17), 'unicode', u'parts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 17), tuple_162044, unicode_162045)
# Adding element type (line 57)
unicode_162046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 26), 'unicode', u'prop')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 17), tuple_162044, unicode_162046)

# Getting the type of 'Type1Font'
Type1Font_162047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type1Font')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type1Font_162047, '__slots__', tuple_162044)

# Assigning a Call to a Name (line 141):

# Call to compile(...): (line 141)
# Processing the call arguments (line 141)
str_162050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 32), 'str', '[\\0\\t\\r\\014\\n ]+')
# Processing the call keyword arguments (line 141)
kwargs_162051 = {}
# Getting the type of 're' (line 141)
re_162048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 're', False)
# Obtaining the member 'compile' of a type (line 141)
compile_162049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 21), re_162048, 'compile')
# Calling compile(args, kwargs) (line 141)
compile_call_result_162052 = invoke(stypy.reporting.localization.Localization(__file__, 141, 21), compile_162049, *[str_162050], **kwargs_162051)

# Getting the type of 'Type1Font'
Type1Font_162053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type1Font')
# Setting the type of the member '_whitespace_re' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type1Font_162053, '_whitespace_re', compile_call_result_162052)

# Assigning a Call to a Name (line 142):

# Call to compile(...): (line 142)
# Processing the call arguments (line 142)
str_162056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 27), 'str', '/{0,2}[^]\\0\\t\\r\\v\\n ()<>{}/%[]+')
# Processing the call keyword arguments (line 142)
kwargs_162057 = {}
# Getting the type of 're' (line 142)
re_162054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 're', False)
# Obtaining the member 'compile' of a type (line 142)
compile_162055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 16), re_162054, 'compile')
# Calling compile(args, kwargs) (line 142)
compile_call_result_162058 = invoke(stypy.reporting.localization.Localization(__file__, 142, 16), compile_162055, *[str_162056], **kwargs_162057)

# Getting the type of 'Type1Font'
Type1Font_162059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type1Font')
# Setting the type of the member '_token_re' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type1Font_162059, '_token_re', compile_call_result_162058)

# Assigning a Call to a Name (line 143):

# Call to compile(...): (line 143)
# Processing the call arguments (line 143)
str_162062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'str', '%[^\\r\\n\\v]*')
# Processing the call keyword arguments (line 143)
kwargs_162063 = {}
# Getting the type of 're' (line 143)
re_162060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 're', False)
# Obtaining the member 'compile' of a type (line 143)
compile_162061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 18), re_162060, 'compile')
# Calling compile(args, kwargs) (line 143)
compile_call_result_162064 = invoke(stypy.reporting.localization.Localization(__file__, 143, 18), compile_162061, *[str_162062], **kwargs_162063)

# Getting the type of 'Type1Font'
Type1Font_162065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type1Font')
# Setting the type of the member '_comment_re' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type1Font_162065, '_comment_re', compile_call_result_162064)

# Assigning a Call to a Name (line 144):

# Call to compile(...): (line 144)
# Processing the call arguments (line 144)
str_162068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 30), 'str', '[()\\\\]')
# Processing the call keyword arguments (line 144)
kwargs_162069 = {}
# Getting the type of 're' (line 144)
re_162066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 're', False)
# Obtaining the member 'compile' of a type (line 144)
compile_162067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 19), re_162066, 'compile')
# Calling compile(args, kwargs) (line 144)
compile_call_result_162070 = invoke(stypy.reporting.localization.Localization(__file__, 144, 19), compile_162067, *[str_162068], **kwargs_162069)

# Getting the type of 'Type1Font'
Type1Font_162071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type1Font')
# Setting the type of the member '_instring_re' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type1Font_162071, '_instring_re', compile_call_result_162070)

# Assigning a Call to a Name (line 147):

# Call to object(...): (line 147)
# Processing the call keyword arguments (line 147)
kwargs_162073 = {}
# Getting the type of 'object' (line 147)
object_162072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 18), 'object', False)
# Calling object(args, kwargs) (line 147)
object_call_result_162074 = invoke(stypy.reporting.localization.Localization(__file__, 147, 18), object_162072, *[], **kwargs_162073)

# Getting the type of 'Type1Font'
Type1Font_162075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type1Font')
# Setting the type of the member '_whitespace' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type1Font_162075, '_whitespace', object_call_result_162074)

# Assigning a Call to a Name (line 148):

# Call to object(...): (line 148)
# Processing the call keyword arguments (line 148)
kwargs_162077 = {}
# Getting the type of 'object' (line 148)
object_162076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'object', False)
# Calling object(args, kwargs) (line 148)
object_call_result_162078 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), object_162076, *[], **kwargs_162077)

# Getting the type of 'Type1Font'
Type1Font_162079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type1Font')
# Setting the type of the member '_name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type1Font_162079, '_name', object_call_result_162078)

# Assigning a Call to a Name (line 149):

# Call to object(...): (line 149)
# Processing the call keyword arguments (line 149)
kwargs_162081 = {}
# Getting the type of 'object' (line 149)
object_162080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 14), 'object', False)
# Calling object(args, kwargs) (line 149)
object_call_result_162082 = invoke(stypy.reporting.localization.Localization(__file__, 149, 14), object_162080, *[], **kwargs_162081)

# Getting the type of 'Type1Font'
Type1Font_162083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type1Font')
# Setting the type of the member '_string' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type1Font_162083, '_string', object_call_result_162082)

# Assigning a Call to a Name (line 150):

# Call to object(...): (line 150)
# Processing the call keyword arguments (line 150)
kwargs_162085 = {}
# Getting the type of 'object' (line 150)
object_162084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'object', False)
# Calling object(args, kwargs) (line 150)
object_call_result_162086 = invoke(stypy.reporting.localization.Localization(__file__, 150, 17), object_162084, *[], **kwargs_162085)

# Getting the type of 'Type1Font'
Type1Font_162087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type1Font')
# Setting the type of the member '_delimiter' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type1Font_162087, '_delimiter', object_call_result_162086)

# Assigning a Call to a Name (line 151):

# Call to object(...): (line 151)
# Processing the call keyword arguments (line 151)
kwargs_162089 = {}
# Getting the type of 'object' (line 151)
object_162088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'object', False)
# Calling object(args, kwargs) (line 151)
object_call_result_162090 = invoke(stypy.reporting.localization.Localization(__file__, 151, 14), object_162088, *[], **kwargs_162089)

# Getting the type of 'Type1Font'
Type1Font_162091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type1Font')
# Setting the type of the member '_number' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type1Font_162091, '_number', object_call_result_162090)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
