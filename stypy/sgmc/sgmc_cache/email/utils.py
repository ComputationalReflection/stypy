
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2010 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Miscellaneous utilities.'''
6: 
7: __all__ = [
8:     'collapse_rfc2231_value',
9:     'decode_params',
10:     'decode_rfc2231',
11:     'encode_rfc2231',
12:     'formataddr',
13:     'formatdate',
14:     'getaddresses',
15:     'make_msgid',
16:     'mktime_tz',
17:     'parseaddr',
18:     'parsedate',
19:     'parsedate_tz',
20:     'unquote',
21:     ]
22: 
23: import os
24: import re
25: import time
26: import base64
27: import random
28: import socket
29: import urllib
30: import warnings
31: 
32: from email._parseaddr import quote
33: from email._parseaddr import AddressList as _AddressList
34: from email._parseaddr import mktime_tz
35: 
36: # We need wormarounds for bugs in these methods in older Pythons (see below)
37: from email._parseaddr import parsedate as _parsedate
38: from email._parseaddr import parsedate_tz as _parsedate_tz
39: 
40: from quopri import decodestring as _qdecode
41: 
42: # Intrapackage imports
43: from email.encoders import _bencode, _qencode
44: 
45: COMMASPACE = ', '
46: EMPTYSTRING = ''
47: UEMPTYSTRING = u''
48: CRLF = '\r\n'
49: TICK = "'"
50: 
51: specialsre = re.compile(r'[][\\()<>@,:;".]')
52: escapesre = re.compile(r'[][\\()"]')
53: 
54: 
55: 
56: # Helpers
57: 
58: def _identity(s):
59:     return s
60: 
61: 
62: def _bdecode(s):
63:     '''Decodes a base64 string.
64: 
65:     This function is equivalent to base64.decodestring and it's retained only
66:     for backward compatibility. It used to remove the last \\n of the decoded
67:     string, if it had any (see issue 7143).
68:     '''
69:     if not s:
70:         return s
71:     return base64.decodestring(s)
72: 
73: 
74: 
75: def fix_eols(s):
76:     '''Replace all line-ending characters with \\r\\n.'''
77:     # Fix newlines with no preceding carriage return
78:     s = re.sub(r'(?<!\r)\n', CRLF, s)
79:     # Fix carriage returns with no following newline
80:     s = re.sub(r'\r(?!\n)', CRLF, s)
81:     return s
82: 
83: 
84: 
85: def formataddr(pair):
86:     '''The inverse of parseaddr(), this takes a 2-tuple of the form
87:     (realname, email_address) and returns the string value suitable
88:     for an RFC 2822 From, To or Cc header.
89: 
90:     If the first element of pair is false, then the second element is
91:     returned unmodified.
92:     '''
93:     name, address = pair
94:     if name:
95:         quotes = ''
96:         if specialsre.search(name):
97:             quotes = '"'
98:         name = escapesre.sub(r'\\\g<0>', name)
99:         return '%s%s%s <%s>' % (quotes, name, quotes, address)
100:     return address
101: 
102: 
103: 
104: def getaddresses(fieldvalues):
105:     '''Return a list of (REALNAME, EMAIL) for each fieldvalue.'''
106:     all = COMMASPACE.join(fieldvalues)
107:     a = _AddressList(all)
108:     return a.addresslist
109: 
110: 
111: 
112: ecre = re.compile(r'''
113:   =\?                   # literal =?
114:   (?P<charset>[^?]*?)   # non-greedy up to the next ? is the charset
115:   \?                    # literal ?
116:   (?P<encoding>[qb])    # either a "q" or a "b", case insensitive
117:   \?                    # literal ?
118:   (?P<atom>.*?)         # non-greedy up to the next ?= is the atom
119:   \?=                   # literal ?=
120:   ''', re.VERBOSE | re.IGNORECASE)
121: 
122: 
123: 
124: def formatdate(timeval=None, localtime=False, usegmt=False):
125:     '''Returns a date string as specified by RFC 2822, e.g.:
126: 
127:     Fri, 09 Nov 2001 01:08:47 -0000
128: 
129:     Optional timeval if given is a floating point time value as accepted by
130:     gmtime() and localtime(), otherwise the current time is used.
131: 
132:     Optional localtime is a flag that when True, interprets timeval, and
133:     returns a date relative to the local timezone instead of UTC, properly
134:     taking daylight savings time into account.
135: 
136:     Optional argument usegmt means that the timezone is written out as
137:     an ascii string, not numeric one (so "GMT" instead of "+0000"). This
138:     is needed for HTTP, and is only used when localtime==False.
139:     '''
140:     # Note: we cannot use strftime() because that honors the locale and RFC
141:     # 2822 requires that day and month names be the English abbreviations.
142:     if timeval is None:
143:         timeval = time.time()
144:     if localtime:
145:         now = time.localtime(timeval)
146:         # Calculate timezone offset, based on whether the local zone has
147:         # daylight savings time, and whether DST is in effect.
148:         if time.daylight and now[-1]:
149:             offset = time.altzone
150:         else:
151:             offset = time.timezone
152:         hours, minutes = divmod(abs(offset), 3600)
153:         # Remember offset is in seconds west of UTC, but the timezone is in
154:         # minutes east of UTC, so the signs differ.
155:         if offset > 0:
156:             sign = '-'
157:         else:
158:             sign = '+'
159:         zone = '%s%02d%02d' % (sign, hours, minutes // 60)
160:     else:
161:         now = time.gmtime(timeval)
162:         # Timezone offset is always -0000
163:         if usegmt:
164:             zone = 'GMT'
165:         else:
166:             zone = '-0000'
167:     return '%s, %02d %s %04d %02d:%02d:%02d %s' % (
168:         ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][now[6]],
169:         now[2],
170:         ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
171:          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][now[1] - 1],
172:         now[0], now[3], now[4], now[5],
173:         zone)
174: 
175: 
176: 
177: def make_msgid(idstring=None):
178:     '''Returns a string suitable for RFC 2822 compliant Message-ID, e.g:
179: 
180:     <142480216486.20800.16526388040877946887@nightshade.la.mastaler.com>
181: 
182:     Optional idstring if given is a string used to strengthen the
183:     uniqueness of the message id.
184:     '''
185:     timeval = int(time.time()*100)
186:     pid = os.getpid()
187:     randint = random.getrandbits(64)
188:     if idstring is None:
189:         idstring = ''
190:     else:
191:         idstring = '.' + idstring
192:     idhost = socket.getfqdn()
193:     msgid = '<%d.%d.%d%s@%s>' % (timeval, pid, randint, idstring, idhost)
194:     return msgid
195: 
196: 
197: 
198: # These functions are in the standalone mimelib version only because they've
199: # subsequently been fixed in the latest Python versions.  We use this to worm
200: # around broken older Pythons.
201: def parsedate(data):
202:     if not data:
203:         return None
204:     return _parsedate(data)
205: 
206: 
207: def parsedate_tz(data):
208:     if not data:
209:         return None
210:     return _parsedate_tz(data)
211: 
212: 
213: def parseaddr(addr):
214:     addrs = _AddressList(addr).addresslist
215:     if not addrs:
216:         return '', ''
217:     return addrs[0]
218: 
219: 
220: # rfc822.unquote() doesn't properly de-backslash-ify in Python pre-2.3.
221: def unquote(str):
222:     '''Remove quotes from a string.'''
223:     if len(str) > 1:
224:         if str.startswith('"') and str.endswith('"'):
225:             return str[1:-1].replace('\\\\', '\\').replace('\\"', '"')
226:         if str.startswith('<') and str.endswith('>'):
227:             return str[1:-1]
228:     return str
229: 
230: 
231: 
232: # RFC2231-related functions - parameter encoding and decoding
233: def decode_rfc2231(s):
234:     '''Decode string according to RFC 2231'''
235:     parts = s.split(TICK, 2)
236:     if len(parts) <= 2:
237:         return None, None, s
238:     return parts
239: 
240: 
241: def encode_rfc2231(s, charset=None, language=None):
242:     '''Encode string according to RFC 2231.
243: 
244:     If neither charset nor language is given, then s is returned as-is.  If
245:     charset is given but not language, the string is encoded using the empty
246:     string for language.
247:     '''
248:     import urllib
249:     s = urllib.quote(s, safe='')
250:     if charset is None and language is None:
251:         return s
252:     if language is None:
253:         language = ''
254:     return "%s'%s'%s" % (charset, language, s)
255: 
256: 
257: rfc2231_continuation = re.compile(r'^(?P<name>\w+)\*((?P<num>[0-9]+)\*?)?$')
258: 
259: def decode_params(params):
260:     '''Decode parameters list according to RFC 2231.
261: 
262:     params is a sequence of 2-tuples containing (param name, string value).
263:     '''
264:     # Copy params so we don't mess with the original
265:     params = params[:]
266:     new_params = []
267:     # Map parameter's name to a list of continuations.  The values are a
268:     # 3-tuple of the continuation number, the string value, and a flag
269:     # specifying whether a particular segment is %-encoded.
270:     rfc2231_params = {}
271:     name, value = params.pop(0)
272:     new_params.append((name, value))
273:     while params:
274:         name, value = params.pop(0)
275:         if name.endswith('*'):
276:             encoded = True
277:         else:
278:             encoded = False
279:         value = unquote(value)
280:         mo = rfc2231_continuation.match(name)
281:         if mo:
282:             name, num = mo.group('name', 'num')
283:             if num is not None:
284:                 num = int(num)
285:             rfc2231_params.setdefault(name, []).append((num, value, encoded))
286:         else:
287:             new_params.append((name, '"%s"' % quote(value)))
288:     if rfc2231_params:
289:         for name, continuations in rfc2231_params.items():
290:             value = []
291:             extended = False
292:             # Sort by number
293:             continuations.sort()
294:             # And now append all values in numerical order, converting
295:             # %-encodings for the encoded segments.  If any of the
296:             # continuation names ends in a *, then the entire string, after
297:             # decoding segments and concatenating, must have the charset and
298:             # language specifiers at the beginning of the string.
299:             for num, s, encoded in continuations:
300:                 if encoded:
301:                     s = urllib.unquote(s)
302:                     extended = True
303:                 value.append(s)
304:             value = quote(EMPTYSTRING.join(value))
305:             if extended:
306:                 charset, language, value = decode_rfc2231(value)
307:                 new_params.append((name, (charset, language, '"%s"' % value)))
308:             else:
309:                 new_params.append((name, '"%s"' % value))
310:     return new_params
311: 
312: def collapse_rfc2231_value(value, errors='replace',
313:                            fallback_charset='us-ascii'):
314:     if isinstance(value, tuple):
315:         rawval = unquote(value[2])
316:         charset = value[0] or 'us-ascii'
317:         try:
318:             return unicode(rawval, charset, errors)
319:         except LookupError:
320:             # XXX charset is unknown to Python.
321:             return unicode(rawval, fallback_charset, errors)
322:     else:
323:         return unquote(value)
324: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_18343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Miscellaneous utilities.')

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['collapse_rfc2231_value', 'decode_params', 'decode_rfc2231', 'encode_rfc2231', 'formataddr', 'formatdate', 'getaddresses', 'make_msgid', 'mktime_tz', 'parseaddr', 'parsedate', 'parsedate_tz', 'unquote']
module_type_store.set_exportable_members(['collapse_rfc2231_value', 'decode_params', 'decode_rfc2231', 'encode_rfc2231', 'formataddr', 'formatdate', 'getaddresses', 'make_msgid', 'mktime_tz', 'parseaddr', 'parsedate', 'parsedate_tz', 'unquote'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_18344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_18345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'str', 'collapse_rfc2231_value')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18345)
# Adding element type (line 7)
str_18346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'str', 'decode_params')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18346)
# Adding element type (line 7)
str_18347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'str', 'decode_rfc2231')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18347)
# Adding element type (line 7)
str_18348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'str', 'encode_rfc2231')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18348)
# Adding element type (line 7)
str_18349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'str', 'formataddr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18349)
# Adding element type (line 7)
str_18350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'str', 'formatdate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18350)
# Adding element type (line 7)
str_18351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'getaddresses')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18351)
# Adding element type (line 7)
str_18352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', 'make_msgid')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18352)
# Adding element type (line 7)
str_18353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'str', 'mktime_tz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18353)
# Adding element type (line 7)
str_18354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'parseaddr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18354)
# Adding element type (line 7)
str_18355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'str', 'parsedate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18355)
# Adding element type (line 7)
str_18356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'str', 'parsedate_tz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18356)
# Adding element type (line 7)
str_18357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'str', 'unquote')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_18344, str_18357)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_18344)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import os' statement (line 23)
import os

import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import re' statement (line 24)
import re

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import time' statement (line 25)
import time

import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'import base64' statement (line 26)
import base64

import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'base64', base64, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'import random' statement (line 27)
import random

import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import socket' statement (line 28)
import socket

import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'socket', socket, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'import urllib' statement (line 29)
import urllib

import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'urllib', urllib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'import warnings' statement (line 30)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from email._parseaddr import quote' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_18358 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'email._parseaddr')

if (type(import_18358) is not StypyTypeError):

    if (import_18358 != 'pyd_module'):
        __import__(import_18358)
        sys_modules_18359 = sys.modules[import_18358]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'email._parseaddr', sys_modules_18359.module_type_store, module_type_store, ['quote'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_18359, sys_modules_18359.module_type_store, module_type_store)
    else:
        from email._parseaddr import quote

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'email._parseaddr', None, module_type_store, ['quote'], [quote])

else:
    # Assigning a type to the variable 'email._parseaddr' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'email._parseaddr', import_18358)

remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from email._parseaddr import _AddressList' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_18360 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'email._parseaddr')

if (type(import_18360) is not StypyTypeError):

    if (import_18360 != 'pyd_module'):
        __import__(import_18360)
        sys_modules_18361 = sys.modules[import_18360]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'email._parseaddr', sys_modules_18361.module_type_store, module_type_store, ['AddressList'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_18361, sys_modules_18361.module_type_store, module_type_store)
    else:
        from email._parseaddr import AddressList as _AddressList

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'email._parseaddr', None, module_type_store, ['AddressList'], [_AddressList])

else:
    # Assigning a type to the variable 'email._parseaddr' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'email._parseaddr', import_18360)

# Adding an alias
module_type_store.add_alias('_AddressList', 'AddressList')
remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from email._parseaddr import mktime_tz' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_18362 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'email._parseaddr')

if (type(import_18362) is not StypyTypeError):

    if (import_18362 != 'pyd_module'):
        __import__(import_18362)
        sys_modules_18363 = sys.modules[import_18362]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'email._parseaddr', sys_modules_18363.module_type_store, module_type_store, ['mktime_tz'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_18363, sys_modules_18363.module_type_store, module_type_store)
    else:
        from email._parseaddr import mktime_tz

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'email._parseaddr', None, module_type_store, ['mktime_tz'], [mktime_tz])

else:
    # Assigning a type to the variable 'email._parseaddr' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'email._parseaddr', import_18362)

remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from email._parseaddr import _parsedate' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_18364 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'email._parseaddr')

if (type(import_18364) is not StypyTypeError):

    if (import_18364 != 'pyd_module'):
        __import__(import_18364)
        sys_modules_18365 = sys.modules[import_18364]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'email._parseaddr', sys_modules_18365.module_type_store, module_type_store, ['parsedate'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_18365, sys_modules_18365.module_type_store, module_type_store)
    else:
        from email._parseaddr import parsedate as _parsedate

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'email._parseaddr', None, module_type_store, ['parsedate'], [_parsedate])

else:
    # Assigning a type to the variable 'email._parseaddr' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'email._parseaddr', import_18364)

# Adding an alias
module_type_store.add_alias('_parsedate', 'parsedate')
remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from email._parseaddr import _parsedate_tz' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_18366 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'email._parseaddr')

if (type(import_18366) is not StypyTypeError):

    if (import_18366 != 'pyd_module'):
        __import__(import_18366)
        sys_modules_18367 = sys.modules[import_18366]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'email._parseaddr', sys_modules_18367.module_type_store, module_type_store, ['parsedate_tz'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_18367, sys_modules_18367.module_type_store, module_type_store)
    else:
        from email._parseaddr import parsedate_tz as _parsedate_tz

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'email._parseaddr', None, module_type_store, ['parsedate_tz'], [_parsedate_tz])

else:
    # Assigning a type to the variable 'email._parseaddr' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'email._parseaddr', import_18366)

# Adding an alias
module_type_store.add_alias('_parsedate_tz', 'parsedate_tz')
remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'from quopri import _qdecode' statement (line 40)
try:
    from quopri import decodestring as _qdecode

except:
    _qdecode = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'quopri', None, module_type_store, ['decodestring'], [_qdecode])
# Adding an alias
module_type_store.add_alias('_qdecode', 'decodestring')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 43, 0))

# 'from email.encoders import _bencode, _qencode' statement (line 43)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_18368 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'email.encoders')

if (type(import_18368) is not StypyTypeError):

    if (import_18368 != 'pyd_module'):
        __import__(import_18368)
        sys_modules_18369 = sys.modules[import_18368]
        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'email.encoders', sys_modules_18369.module_type_store, module_type_store, ['_bencode', '_qencode'])
        nest_module(stypy.reporting.localization.Localization(__file__, 43, 0), __file__, sys_modules_18369, sys_modules_18369.module_type_store, module_type_store)
    else:
        from email.encoders import _bencode, _qencode

        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'email.encoders', None, module_type_store, ['_bencode', '_qencode'], [_bencode, _qencode])

else:
    # Assigning a type to the variable 'email.encoders' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'email.encoders', import_18368)

remove_current_file_folder_from_path('C:/Python27/lib/email/')


# Assigning a Str to a Name (line 45):

# Assigning a Str to a Name (line 45):
str_18370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 13), 'str', ', ')
# Assigning a type to the variable 'COMMASPACE' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'COMMASPACE', str_18370)

# Assigning a Str to a Name (line 46):

# Assigning a Str to a Name (line 46):
str_18371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 14), 'str', '')
# Assigning a type to the variable 'EMPTYSTRING' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'EMPTYSTRING', str_18371)

# Assigning a Str to a Name (line 47):

# Assigning a Str to a Name (line 47):
unicode_18372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 15), 'unicode', u'')
# Assigning a type to the variable 'UEMPTYSTRING' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'UEMPTYSTRING', unicode_18372)

# Assigning a Str to a Name (line 48):

# Assigning a Str to a Name (line 48):
str_18373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 7), 'str', '\r\n')
# Assigning a type to the variable 'CRLF' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'CRLF', str_18373)

# Assigning a Str to a Name (line 49):

# Assigning a Str to a Name (line 49):
str_18374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 7), 'str', "'")
# Assigning a type to the variable 'TICK' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'TICK', str_18374)

# Assigning a Call to a Name (line 51):

# Assigning a Call to a Name (line 51):

# Call to compile(...): (line 51)
# Processing the call arguments (line 51)
str_18377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 24), 'str', '[][\\\\()<>@,:;".]')
# Processing the call keyword arguments (line 51)
kwargs_18378 = {}
# Getting the type of 're' (line 51)
re_18375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 're', False)
# Obtaining the member 'compile' of a type (line 51)
compile_18376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 13), re_18375, 'compile')
# Calling compile(args, kwargs) (line 51)
compile_call_result_18379 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), compile_18376, *[str_18377], **kwargs_18378)

# Assigning a type to the variable 'specialsre' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'specialsre', compile_call_result_18379)

# Assigning a Call to a Name (line 52):

# Assigning a Call to a Name (line 52):

# Call to compile(...): (line 52)
# Processing the call arguments (line 52)
str_18382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'str', '[][\\\\()"]')
# Processing the call keyword arguments (line 52)
kwargs_18383 = {}
# Getting the type of 're' (line 52)
re_18380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 're', False)
# Obtaining the member 'compile' of a type (line 52)
compile_18381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), re_18380, 'compile')
# Calling compile(args, kwargs) (line 52)
compile_call_result_18384 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), compile_18381, *[str_18382], **kwargs_18383)

# Assigning a type to the variable 'escapesre' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'escapesre', compile_call_result_18384)

@norecursion
def _identity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_identity'
    module_type_store = module_type_store.open_function_context('_identity', 58, 0, False)
    
    # Passed parameters checking function
    _identity.stypy_localization = localization
    _identity.stypy_type_of_self = None
    _identity.stypy_type_store = module_type_store
    _identity.stypy_function_name = '_identity'
    _identity.stypy_param_names_list = ['s']
    _identity.stypy_varargs_param_name = None
    _identity.stypy_kwargs_param_name = None
    _identity.stypy_call_defaults = defaults
    _identity.stypy_call_varargs = varargs
    _identity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_identity', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_identity', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_identity(...)' code ##################

    # Getting the type of 's' (line 59)
    s_18385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type', s_18385)
    
    # ################# End of '_identity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_identity' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_18386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18386)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_identity'
    return stypy_return_type_18386

# Assigning a type to the variable '_identity' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), '_identity', _identity)

@norecursion
def _bdecode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_bdecode'
    module_type_store = module_type_store.open_function_context('_bdecode', 62, 0, False)
    
    # Passed parameters checking function
    _bdecode.stypy_localization = localization
    _bdecode.stypy_type_of_self = None
    _bdecode.stypy_type_store = module_type_store
    _bdecode.stypy_function_name = '_bdecode'
    _bdecode.stypy_param_names_list = ['s']
    _bdecode.stypy_varargs_param_name = None
    _bdecode.stypy_kwargs_param_name = None
    _bdecode.stypy_call_defaults = defaults
    _bdecode.stypy_call_varargs = varargs
    _bdecode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_bdecode', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_bdecode', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_bdecode(...)' code ##################

    str_18387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', "Decodes a base64 string.\n\n    This function is equivalent to base64.decodestring and it's retained only\n    for backward compatibility. It used to remove the last \\n of the decoded\n    string, if it had any (see issue 7143).\n    ")
    
    # Getting the type of 's' (line 69)
    s_18388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 's')
    # Applying the 'not' unary operator (line 69)
    result_not__18389 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 7), 'not', s_18388)
    
    # Testing if the type of an if condition is none (line 69)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 4), result_not__18389):
        pass
    else:
        
        # Testing the type of an if condition (line 69)
        if_condition_18390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 4), result_not__18389)
        # Assigning a type to the variable 'if_condition_18390' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'if_condition_18390', if_condition_18390)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 's' (line 70)
        s_18391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', s_18391)
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to decodestring(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 's' (line 71)
    s_18394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 31), 's', False)
    # Processing the call keyword arguments (line 71)
    kwargs_18395 = {}
    # Getting the type of 'base64' (line 71)
    base64_18392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'base64', False)
    # Obtaining the member 'decodestring' of a type (line 71)
    decodestring_18393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), base64_18392, 'decodestring')
    # Calling decodestring(args, kwargs) (line 71)
    decodestring_call_result_18396 = invoke(stypy.reporting.localization.Localization(__file__, 71, 11), decodestring_18393, *[s_18394], **kwargs_18395)
    
    # Assigning a type to the variable 'stypy_return_type' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type', decodestring_call_result_18396)
    
    # ################# End of '_bdecode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_bdecode' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_18397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18397)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_bdecode'
    return stypy_return_type_18397

# Assigning a type to the variable '_bdecode' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), '_bdecode', _bdecode)

@norecursion
def fix_eols(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fix_eols'
    module_type_store = module_type_store.open_function_context('fix_eols', 75, 0, False)
    
    # Passed parameters checking function
    fix_eols.stypy_localization = localization
    fix_eols.stypy_type_of_self = None
    fix_eols.stypy_type_store = module_type_store
    fix_eols.stypy_function_name = 'fix_eols'
    fix_eols.stypy_param_names_list = ['s']
    fix_eols.stypy_varargs_param_name = None
    fix_eols.stypy_kwargs_param_name = None
    fix_eols.stypy_call_defaults = defaults
    fix_eols.stypy_call_varargs = varargs
    fix_eols.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fix_eols', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fix_eols', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fix_eols(...)' code ##################

    str_18398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'str', 'Replace all line-ending characters with \\r\\n.')
    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to sub(...): (line 78)
    # Processing the call arguments (line 78)
    str_18401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'str', '(?<!\\r)\\n')
    # Getting the type of 'CRLF' (line 78)
    CRLF_18402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 29), 'CRLF', False)
    # Getting the type of 's' (line 78)
    s_18403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 35), 's', False)
    # Processing the call keyword arguments (line 78)
    kwargs_18404 = {}
    # Getting the type of 're' (line 78)
    re_18399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 're', False)
    # Obtaining the member 'sub' of a type (line 78)
    sub_18400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), re_18399, 'sub')
    # Calling sub(args, kwargs) (line 78)
    sub_call_result_18405 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), sub_18400, *[str_18401, CRLF_18402, s_18403], **kwargs_18404)
    
    # Assigning a type to the variable 's' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 's', sub_call_result_18405)
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to sub(...): (line 80)
    # Processing the call arguments (line 80)
    str_18408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 15), 'str', '\\r(?!\\n)')
    # Getting the type of 'CRLF' (line 80)
    CRLF_18409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'CRLF', False)
    # Getting the type of 's' (line 80)
    s_18410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 's', False)
    # Processing the call keyword arguments (line 80)
    kwargs_18411 = {}
    # Getting the type of 're' (line 80)
    re_18406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 're', False)
    # Obtaining the member 'sub' of a type (line 80)
    sub_18407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), re_18406, 'sub')
    # Calling sub(args, kwargs) (line 80)
    sub_call_result_18412 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), sub_18407, *[str_18408, CRLF_18409, s_18410], **kwargs_18411)
    
    # Assigning a type to the variable 's' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 's', sub_call_result_18412)
    # Getting the type of 's' (line 81)
    s_18413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', s_18413)
    
    # ################# End of 'fix_eols(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fix_eols' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_18414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18414)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fix_eols'
    return stypy_return_type_18414

# Assigning a type to the variable 'fix_eols' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'fix_eols', fix_eols)

@norecursion
def formataddr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'formataddr'
    module_type_store = module_type_store.open_function_context('formataddr', 85, 0, False)
    
    # Passed parameters checking function
    formataddr.stypy_localization = localization
    formataddr.stypy_type_of_self = None
    formataddr.stypy_type_store = module_type_store
    formataddr.stypy_function_name = 'formataddr'
    formataddr.stypy_param_names_list = ['pair']
    formataddr.stypy_varargs_param_name = None
    formataddr.stypy_kwargs_param_name = None
    formataddr.stypy_call_defaults = defaults
    formataddr.stypy_call_varargs = varargs
    formataddr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'formataddr', ['pair'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'formataddr', localization, ['pair'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'formataddr(...)' code ##################

    str_18415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', 'The inverse of parseaddr(), this takes a 2-tuple of the form\n    (realname, email_address) and returns the string value suitable\n    for an RFC 2822 From, To or Cc header.\n\n    If the first element of pair is false, then the second element is\n    returned unmodified.\n    ')
    
    # Assigning a Name to a Tuple (line 93):
    
    # Assigning a Subscript to a Name (line 93):
    
    # Obtaining the type of the subscript
    int_18416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'int')
    # Getting the type of 'pair' (line 93)
    pair_18417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'pair')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___18418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), pair_18417, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_18419 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), getitem___18418, int_18416)
    
    # Assigning a type to the variable 'tuple_var_assignment_18325' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_18325', subscript_call_result_18419)
    
    # Assigning a Subscript to a Name (line 93):
    
    # Obtaining the type of the subscript
    int_18420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'int')
    # Getting the type of 'pair' (line 93)
    pair_18421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'pair')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___18422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), pair_18421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_18423 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), getitem___18422, int_18420)
    
    # Assigning a type to the variable 'tuple_var_assignment_18326' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_18326', subscript_call_result_18423)
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'tuple_var_assignment_18325' (line 93)
    tuple_var_assignment_18325_18424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_18325')
    # Assigning a type to the variable 'name' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'name', tuple_var_assignment_18325_18424)
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'tuple_var_assignment_18326' (line 93)
    tuple_var_assignment_18326_18425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_18326')
    # Assigning a type to the variable 'address' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 10), 'address', tuple_var_assignment_18326_18425)
    # Getting the type of 'name' (line 94)
    name_18426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'name')
    # Testing if the type of an if condition is none (line 94)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 94, 4), name_18426):
        pass
    else:
        
        # Testing the type of an if condition (line 94)
        if_condition_18427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), name_18426)
        # Assigning a type to the variable 'if_condition_18427' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_18427', if_condition_18427)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 95):
        
        # Assigning a Str to a Name (line 95):
        str_18428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 17), 'str', '')
        # Assigning a type to the variable 'quotes' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'quotes', str_18428)
        
        # Call to search(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'name' (line 96)
        name_18431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'name', False)
        # Processing the call keyword arguments (line 96)
        kwargs_18432 = {}
        # Getting the type of 'specialsre' (line 96)
        specialsre_18429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'specialsre', False)
        # Obtaining the member 'search' of a type (line 96)
        search_18430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), specialsre_18429, 'search')
        # Calling search(args, kwargs) (line 96)
        search_call_result_18433 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), search_18430, *[name_18431], **kwargs_18432)
        
        # Testing if the type of an if condition is none (line 96)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 8), search_call_result_18433):
            pass
        else:
            
            # Testing the type of an if condition (line 96)
            if_condition_18434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 8), search_call_result_18433)
            # Assigning a type to the variable 'if_condition_18434' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'if_condition_18434', if_condition_18434)
            # SSA begins for if statement (line 96)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 97):
            
            # Assigning a Str to a Name (line 97):
            str_18435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 21), 'str', '"')
            # Assigning a type to the variable 'quotes' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'quotes', str_18435)
            # SSA join for if statement (line 96)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to sub(...): (line 98)
        # Processing the call arguments (line 98)
        str_18438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 29), 'str', '\\\\\\g<0>')
        # Getting the type of 'name' (line 98)
        name_18439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 41), 'name', False)
        # Processing the call keyword arguments (line 98)
        kwargs_18440 = {}
        # Getting the type of 'escapesre' (line 98)
        escapesre_18436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'escapesre', False)
        # Obtaining the member 'sub' of a type (line 98)
        sub_18437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), escapesre_18436, 'sub')
        # Calling sub(args, kwargs) (line 98)
        sub_call_result_18441 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), sub_18437, *[str_18438, name_18439], **kwargs_18440)
        
        # Assigning a type to the variable 'name' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'name', sub_call_result_18441)
        str_18442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 15), 'str', '%s%s%s <%s>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_18443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        # Getting the type of 'quotes' (line 99)
        quotes_18444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 32), 'quotes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 32), tuple_18443, quotes_18444)
        # Adding element type (line 99)
        # Getting the type of 'name' (line 99)
        name_18445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 40), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 32), tuple_18443, name_18445)
        # Adding element type (line 99)
        # Getting the type of 'quotes' (line 99)
        quotes_18446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 46), 'quotes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 32), tuple_18443, quotes_18446)
        # Adding element type (line 99)
        # Getting the type of 'address' (line 99)
        address_18447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 54), 'address')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 32), tuple_18443, address_18447)
        
        # Applying the binary operator '%' (line 99)
        result_mod_18448 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 15), '%', str_18442, tuple_18443)
        
        # Assigning a type to the variable 'stypy_return_type' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'stypy_return_type', result_mod_18448)
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'address' (line 100)
    address_18449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'address')
    # Assigning a type to the variable 'stypy_return_type' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type', address_18449)
    
    # ################# End of 'formataddr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'formataddr' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_18450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18450)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'formataddr'
    return stypy_return_type_18450

# Assigning a type to the variable 'formataddr' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'formataddr', formataddr)

@norecursion
def getaddresses(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getaddresses'
    module_type_store = module_type_store.open_function_context('getaddresses', 104, 0, False)
    
    # Passed parameters checking function
    getaddresses.stypy_localization = localization
    getaddresses.stypy_type_of_self = None
    getaddresses.stypy_type_store = module_type_store
    getaddresses.stypy_function_name = 'getaddresses'
    getaddresses.stypy_param_names_list = ['fieldvalues']
    getaddresses.stypy_varargs_param_name = None
    getaddresses.stypy_kwargs_param_name = None
    getaddresses.stypy_call_defaults = defaults
    getaddresses.stypy_call_varargs = varargs
    getaddresses.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getaddresses', ['fieldvalues'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getaddresses', localization, ['fieldvalues'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getaddresses(...)' code ##################

    str_18451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 4), 'str', 'Return a list of (REALNAME, EMAIL) for each fieldvalue.')
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to join(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'fieldvalues' (line 106)
    fieldvalues_18454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'fieldvalues', False)
    # Processing the call keyword arguments (line 106)
    kwargs_18455 = {}
    # Getting the type of 'COMMASPACE' (line 106)
    COMMASPACE_18452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 10), 'COMMASPACE', False)
    # Obtaining the member 'join' of a type (line 106)
    join_18453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 10), COMMASPACE_18452, 'join')
    # Calling join(args, kwargs) (line 106)
    join_call_result_18456 = invoke(stypy.reporting.localization.Localization(__file__, 106, 10), join_18453, *[fieldvalues_18454], **kwargs_18455)
    
    # Assigning a type to the variable 'all' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'all', join_call_result_18456)
    
    # Assigning a Call to a Name (line 107):
    
    # Assigning a Call to a Name (line 107):
    
    # Call to _AddressList(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'all' (line 107)
    all_18458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'all', False)
    # Processing the call keyword arguments (line 107)
    kwargs_18459 = {}
    # Getting the type of '_AddressList' (line 107)
    _AddressList_18457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), '_AddressList', False)
    # Calling _AddressList(args, kwargs) (line 107)
    _AddressList_call_result_18460 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), _AddressList_18457, *[all_18458], **kwargs_18459)
    
    # Assigning a type to the variable 'a' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'a', _AddressList_call_result_18460)
    # Getting the type of 'a' (line 108)
    a_18461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'a')
    # Obtaining the member 'addresslist' of a type (line 108)
    addresslist_18462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 11), a_18461, 'addresslist')
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type', addresslist_18462)
    
    # ################# End of 'getaddresses(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getaddresses' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_18463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18463)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getaddresses'
    return stypy_return_type_18463

# Assigning a type to the variable 'getaddresses' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'getaddresses', getaddresses)

# Assigning a Call to a Name (line 112):

# Assigning a Call to a Name (line 112):

# Call to compile(...): (line 112)
# Processing the call arguments (line 112)
str_18466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, (-1)), 'str', '\n  =\\?                   # literal =?\n  (?P<charset>[^?]*?)   # non-greedy up to the next ? is the charset\n  \\?                    # literal ?\n  (?P<encoding>[qb])    # either a "q" or a "b", case insensitive\n  \\?                    # literal ?\n  (?P<atom>.*?)         # non-greedy up to the next ?= is the atom\n  \\?=                   # literal ?=\n  ')
# Getting the type of 're' (line 120)
re_18467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 7), 're', False)
# Obtaining the member 'VERBOSE' of a type (line 120)
VERBOSE_18468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 7), re_18467, 'VERBOSE')
# Getting the type of 're' (line 120)
re_18469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 're', False)
# Obtaining the member 'IGNORECASE' of a type (line 120)
IGNORECASE_18470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 20), re_18469, 'IGNORECASE')
# Applying the binary operator '|' (line 120)
result_or__18471 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 7), '|', VERBOSE_18468, IGNORECASE_18470)

# Processing the call keyword arguments (line 112)
kwargs_18472 = {}
# Getting the type of 're' (line 112)
re_18464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 're', False)
# Obtaining the member 'compile' of a type (line 112)
compile_18465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 7), re_18464, 'compile')
# Calling compile(args, kwargs) (line 112)
compile_call_result_18473 = invoke(stypy.reporting.localization.Localization(__file__, 112, 7), compile_18465, *[str_18466, result_or__18471], **kwargs_18472)

# Assigning a type to the variable 'ecre' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'ecre', compile_call_result_18473)

@norecursion
def formatdate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 124)
    None_18474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'None')
    # Getting the type of 'False' (line 124)
    False_18475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 39), 'False')
    # Getting the type of 'False' (line 124)
    False_18476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 53), 'False')
    defaults = [None_18474, False_18475, False_18476]
    # Create a new context for function 'formatdate'
    module_type_store = module_type_store.open_function_context('formatdate', 124, 0, False)
    
    # Passed parameters checking function
    formatdate.stypy_localization = localization
    formatdate.stypy_type_of_self = None
    formatdate.stypy_type_store = module_type_store
    formatdate.stypy_function_name = 'formatdate'
    formatdate.stypy_param_names_list = ['timeval', 'localtime', 'usegmt']
    formatdate.stypy_varargs_param_name = None
    formatdate.stypy_kwargs_param_name = None
    formatdate.stypy_call_defaults = defaults
    formatdate.stypy_call_varargs = varargs
    formatdate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'formatdate', ['timeval', 'localtime', 'usegmt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'formatdate', localization, ['timeval', 'localtime', 'usegmt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'formatdate(...)' code ##################

    str_18477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, (-1)), 'str', 'Returns a date string as specified by RFC 2822, e.g.:\n\n    Fri, 09 Nov 2001 01:08:47 -0000\n\n    Optional timeval if given is a floating point time value as accepted by\n    gmtime() and localtime(), otherwise the current time is used.\n\n    Optional localtime is a flag that when True, interprets timeval, and\n    returns a date relative to the local timezone instead of UTC, properly\n    taking daylight savings time into account.\n\n    Optional argument usegmt means that the timezone is written out as\n    an ascii string, not numeric one (so "GMT" instead of "+0000"). This\n    is needed for HTTP, and is only used when localtime==False.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 142)
    # Getting the type of 'timeval' (line 142)
    timeval_18478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 7), 'timeval')
    # Getting the type of 'None' (line 142)
    None_18479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'None')
    
    (may_be_18480, more_types_in_union_18481) = may_be_none(timeval_18478, None_18479)

    if may_be_18480:

        if more_types_in_union_18481:
            # Runtime conditional SSA (line 142)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 143):
        
        # Assigning a Call to a Name (line 143):
        
        # Call to time(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_18484 = {}
        # Getting the type of 'time' (line 143)
        time_18482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'time', False)
        # Obtaining the member 'time' of a type (line 143)
        time_18483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 18), time_18482, 'time')
        # Calling time(args, kwargs) (line 143)
        time_call_result_18485 = invoke(stypy.reporting.localization.Localization(__file__, 143, 18), time_18483, *[], **kwargs_18484)
        
        # Assigning a type to the variable 'timeval' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'timeval', time_call_result_18485)

        if more_types_in_union_18481:
            # SSA join for if statement (line 142)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'localtime' (line 144)
    localtime_18486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 7), 'localtime')
    # Testing if the type of an if condition is none (line 144)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 144, 4), localtime_18486):
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to gmtime(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'timeval' (line 161)
        timeval_18535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'timeval', False)
        # Processing the call keyword arguments (line 161)
        kwargs_18536 = {}
        # Getting the type of 'time' (line 161)
        time_18533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 14), 'time', False)
        # Obtaining the member 'gmtime' of a type (line 161)
        gmtime_18534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 14), time_18533, 'gmtime')
        # Calling gmtime(args, kwargs) (line 161)
        gmtime_call_result_18537 = invoke(stypy.reporting.localization.Localization(__file__, 161, 14), gmtime_18534, *[timeval_18535], **kwargs_18536)
        
        # Assigning a type to the variable 'now' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'now', gmtime_call_result_18537)
        # Getting the type of 'usegmt' (line 163)
        usegmt_18538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'usegmt')
        # Testing if the type of an if condition is none (line 163)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 163, 8), usegmt_18538):
            
            # Assigning a Str to a Name (line 166):
            
            # Assigning a Str to a Name (line 166):
            str_18541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 19), 'str', '-0000')
            # Assigning a type to the variable 'zone' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'zone', str_18541)
        else:
            
            # Testing the type of an if condition (line 163)
            if_condition_18539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), usegmt_18538)
            # Assigning a type to the variable 'if_condition_18539' (line 163)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_18539', if_condition_18539)
            # SSA begins for if statement (line 163)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 164):
            
            # Assigning a Str to a Name (line 164):
            str_18540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 19), 'str', 'GMT')
            # Assigning a type to the variable 'zone' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'zone', str_18540)
            # SSA branch for the else part of an if statement (line 163)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 166):
            
            # Assigning a Str to a Name (line 166):
            str_18541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 19), 'str', '-0000')
            # Assigning a type to the variable 'zone' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'zone', str_18541)
            # SSA join for if statement (line 163)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 144)
        if_condition_18487 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 4), localtime_18486)
        # Assigning a type to the variable 'if_condition_18487' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'if_condition_18487', if_condition_18487)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to localtime(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'timeval' (line 145)
        timeval_18490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 29), 'timeval', False)
        # Processing the call keyword arguments (line 145)
        kwargs_18491 = {}
        # Getting the type of 'time' (line 145)
        time_18488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 14), 'time', False)
        # Obtaining the member 'localtime' of a type (line 145)
        localtime_18489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 14), time_18488, 'localtime')
        # Calling localtime(args, kwargs) (line 145)
        localtime_call_result_18492 = invoke(stypy.reporting.localization.Localization(__file__, 145, 14), localtime_18489, *[timeval_18490], **kwargs_18491)
        
        # Assigning a type to the variable 'now' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'now', localtime_call_result_18492)
        
        # Evaluating a boolean operation
        # Getting the type of 'time' (line 148)
        time_18493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'time')
        # Obtaining the member 'daylight' of a type (line 148)
        daylight_18494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 11), time_18493, 'daylight')
        
        # Obtaining the type of the subscript
        int_18495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 33), 'int')
        # Getting the type of 'now' (line 148)
        now_18496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'now')
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___18497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 29), now_18496, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_18498 = invoke(stypy.reporting.localization.Localization(__file__, 148, 29), getitem___18497, int_18495)
        
        # Applying the binary operator 'and' (line 148)
        result_and_keyword_18499 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 11), 'and', daylight_18494, subscript_call_result_18498)
        
        # Testing if the type of an if condition is none (line 148)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 148, 8), result_and_keyword_18499):
            
            # Assigning a Attribute to a Name (line 151):
            
            # Assigning a Attribute to a Name (line 151):
            # Getting the type of 'time' (line 151)
            time_18503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'time')
            # Obtaining the member 'timezone' of a type (line 151)
            timezone_18504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 21), time_18503, 'timezone')
            # Assigning a type to the variable 'offset' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'offset', timezone_18504)
        else:
            
            # Testing the type of an if condition (line 148)
            if_condition_18500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 8), result_and_keyword_18499)
            # Assigning a type to the variable 'if_condition_18500' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'if_condition_18500', if_condition_18500)
            # SSA begins for if statement (line 148)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 149):
            
            # Assigning a Attribute to a Name (line 149):
            # Getting the type of 'time' (line 149)
            time_18501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'time')
            # Obtaining the member 'altzone' of a type (line 149)
            altzone_18502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 21), time_18501, 'altzone')
            # Assigning a type to the variable 'offset' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'offset', altzone_18502)
            # SSA branch for the else part of an if statement (line 148)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 151):
            
            # Assigning a Attribute to a Name (line 151):
            # Getting the type of 'time' (line 151)
            time_18503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'time')
            # Obtaining the member 'timezone' of a type (line 151)
            timezone_18504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 21), time_18503, 'timezone')
            # Assigning a type to the variable 'offset' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'offset', timezone_18504)
            # SSA join for if statement (line 148)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Tuple (line 152):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Call to abs(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'offset' (line 152)
        offset_18507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 36), 'offset', False)
        # Processing the call keyword arguments (line 152)
        kwargs_18508 = {}
        # Getting the type of 'abs' (line 152)
        abs_18506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'abs', False)
        # Calling abs(args, kwargs) (line 152)
        abs_call_result_18509 = invoke(stypy.reporting.localization.Localization(__file__, 152, 32), abs_18506, *[offset_18507], **kwargs_18508)
        
        int_18510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 45), 'int')
        # Processing the call keyword arguments (line 152)
        kwargs_18511 = {}
        # Getting the type of 'divmod' (line 152)
        divmod_18505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'divmod', False)
        # Calling divmod(args, kwargs) (line 152)
        divmod_call_result_18512 = invoke(stypy.reporting.localization.Localization(__file__, 152, 25), divmod_18505, *[abs_call_result_18509, int_18510], **kwargs_18511)
        
        # Assigning a type to the variable 'call_assignment_18327' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'call_assignment_18327', divmod_call_result_18512)
        
        # Assigning a Call to a Name (line 152):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_18327' (line 152)
        call_assignment_18327_18513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'call_assignment_18327', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_18514 = stypy_get_value_from_tuple(call_assignment_18327_18513, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_18328' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'call_assignment_18328', stypy_get_value_from_tuple_call_result_18514)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'call_assignment_18328' (line 152)
        call_assignment_18328_18515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'call_assignment_18328')
        # Assigning a type to the variable 'hours' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'hours', call_assignment_18328_18515)
        
        # Assigning a Call to a Name (line 152):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_18327' (line 152)
        call_assignment_18327_18516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'call_assignment_18327', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_18517 = stypy_get_value_from_tuple(call_assignment_18327_18516, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_18329' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'call_assignment_18329', stypy_get_value_from_tuple_call_result_18517)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'call_assignment_18329' (line 152)
        call_assignment_18329_18518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'call_assignment_18329')
        # Assigning a type to the variable 'minutes' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'minutes', call_assignment_18329_18518)
        
        # Getting the type of 'offset' (line 155)
        offset_18519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'offset')
        int_18520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'int')
        # Applying the binary operator '>' (line 155)
        result_gt_18521 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), '>', offset_18519, int_18520)
        
        # Testing if the type of an if condition is none (line 155)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 155, 8), result_gt_18521):
            
            # Assigning a Str to a Name (line 158):
            
            # Assigning a Str to a Name (line 158):
            str_18524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 19), 'str', '+')
            # Assigning a type to the variable 'sign' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'sign', str_18524)
        else:
            
            # Testing the type of an if condition (line 155)
            if_condition_18522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), result_gt_18521)
            # Assigning a type to the variable 'if_condition_18522' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_18522', if_condition_18522)
            # SSA begins for if statement (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 156):
            
            # Assigning a Str to a Name (line 156):
            str_18523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 19), 'str', '-')
            # Assigning a type to the variable 'sign' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'sign', str_18523)
            # SSA branch for the else part of an if statement (line 155)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 158):
            
            # Assigning a Str to a Name (line 158):
            str_18524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 19), 'str', '+')
            # Assigning a type to the variable 'sign' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'sign', str_18524)
            # SSA join for if statement (line 155)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 159):
        
        # Assigning a BinOp to a Name (line 159):
        str_18525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 15), 'str', '%s%02d%02d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 159)
        tuple_18526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 159)
        # Adding element type (line 159)
        # Getting the type of 'sign' (line 159)
        sign_18527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'sign')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 31), tuple_18526, sign_18527)
        # Adding element type (line 159)
        # Getting the type of 'hours' (line 159)
        hours_18528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 37), 'hours')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 31), tuple_18526, hours_18528)
        # Adding element type (line 159)
        # Getting the type of 'minutes' (line 159)
        minutes_18529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 44), 'minutes')
        int_18530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 55), 'int')
        # Applying the binary operator '//' (line 159)
        result_floordiv_18531 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 44), '//', minutes_18529, int_18530)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 31), tuple_18526, result_floordiv_18531)
        
        # Applying the binary operator '%' (line 159)
        result_mod_18532 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 15), '%', str_18525, tuple_18526)
        
        # Assigning a type to the variable 'zone' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'zone', result_mod_18532)
        # SSA branch for the else part of an if statement (line 144)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to gmtime(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'timeval' (line 161)
        timeval_18535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'timeval', False)
        # Processing the call keyword arguments (line 161)
        kwargs_18536 = {}
        # Getting the type of 'time' (line 161)
        time_18533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 14), 'time', False)
        # Obtaining the member 'gmtime' of a type (line 161)
        gmtime_18534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 14), time_18533, 'gmtime')
        # Calling gmtime(args, kwargs) (line 161)
        gmtime_call_result_18537 = invoke(stypy.reporting.localization.Localization(__file__, 161, 14), gmtime_18534, *[timeval_18535], **kwargs_18536)
        
        # Assigning a type to the variable 'now' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'now', gmtime_call_result_18537)
        # Getting the type of 'usegmt' (line 163)
        usegmt_18538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'usegmt')
        # Testing if the type of an if condition is none (line 163)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 163, 8), usegmt_18538):
            
            # Assigning a Str to a Name (line 166):
            
            # Assigning a Str to a Name (line 166):
            str_18541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 19), 'str', '-0000')
            # Assigning a type to the variable 'zone' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'zone', str_18541)
        else:
            
            # Testing the type of an if condition (line 163)
            if_condition_18539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), usegmt_18538)
            # Assigning a type to the variable 'if_condition_18539' (line 163)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_18539', if_condition_18539)
            # SSA begins for if statement (line 163)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 164):
            
            # Assigning a Str to a Name (line 164):
            str_18540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 19), 'str', 'GMT')
            # Assigning a type to the variable 'zone' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'zone', str_18540)
            # SSA branch for the else part of an if statement (line 163)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 166):
            
            # Assigning a Str to a Name (line 166):
            str_18541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 19), 'str', '-0000')
            # Assigning a type to the variable 'zone' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'zone', str_18541)
            # SSA join for if statement (line 163)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        

    str_18542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 11), 'str', '%s, %02d %s %04d %02d:%02d:%02d %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 168)
    tuple_18543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 168)
    # Adding element type (line 168)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_18544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 62), 'int')
    # Getting the type of 'now' (line 168)
    now_18545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 58), 'now')
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___18546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 58), now_18545, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_18547 = invoke(stypy.reporting.localization.Localization(__file__, 168, 58), getitem___18546, int_18544)
    
    
    # Obtaining an instance of the builtin type 'list' (line 168)
    list_18548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 168)
    # Adding element type (line 168)
    str_18549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 9), 'str', 'Mon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), list_18548, str_18549)
    # Adding element type (line 168)
    str_18550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 16), 'str', 'Tue')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), list_18548, str_18550)
    # Adding element type (line 168)
    str_18551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 23), 'str', 'Wed')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), list_18548, str_18551)
    # Adding element type (line 168)
    str_18552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 30), 'str', 'Thu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), list_18548, str_18552)
    # Adding element type (line 168)
    str_18553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 37), 'str', 'Fri')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), list_18548, str_18553)
    # Adding element type (line 168)
    str_18554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 44), 'str', 'Sat')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), list_18548, str_18554)
    # Adding element type (line 168)
    str_18555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 51), 'str', 'Sun')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), list_18548, str_18555)
    
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___18556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), list_18548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_18557 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), getitem___18556, subscript_call_result_18547)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), tuple_18543, subscript_call_result_18557)
    # Adding element type (line 168)
    
    # Obtaining the type of the subscript
    int_18558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    # Getting the type of 'now' (line 169)
    now_18559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'now')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___18560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), now_18559, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_18561 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___18560, int_18558)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), tuple_18543, subscript_call_result_18561)
    # Adding element type (line 168)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_18562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 55), 'int')
    # Getting the type of 'now' (line 171)
    now_18563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 51), 'now')
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___18564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 51), now_18563, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_18565 = invoke(stypy.reporting.localization.Localization(__file__, 171, 51), getitem___18564, int_18562)
    
    int_18566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 60), 'int')
    # Applying the binary operator '-' (line 171)
    result_sub_18567 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 51), '-', subscript_call_result_18565, int_18566)
    
    
    # Obtaining an instance of the builtin type 'list' (line 170)
    list_18568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 170)
    # Adding element type (line 170)
    str_18569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 9), 'str', 'Jan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18569)
    # Adding element type (line 170)
    str_18570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 16), 'str', 'Feb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18570)
    # Adding element type (line 170)
    str_18571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'str', 'Mar')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18571)
    # Adding element type (line 170)
    str_18572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 30), 'str', 'Apr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18572)
    # Adding element type (line 170)
    str_18573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 37), 'str', 'May')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18573)
    # Adding element type (line 170)
    str_18574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 44), 'str', 'Jun')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18574)
    # Adding element type (line 170)
    str_18575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 9), 'str', 'Jul')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18575)
    # Adding element type (line 170)
    str_18576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 16), 'str', 'Aug')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18576)
    # Adding element type (line 170)
    str_18577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 23), 'str', 'Sep')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18577)
    # Adding element type (line 170)
    str_18578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'str', 'Oct')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18578)
    # Adding element type (line 170)
    str_18579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 37), 'str', 'Nov')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18579)
    # Adding element type (line 170)
    str_18580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 44), 'str', 'Dec')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, str_18580)
    
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___18581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), list_18568, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_18582 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), getitem___18581, result_sub_18567)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), tuple_18543, subscript_call_result_18582)
    # Adding element type (line 168)
    
    # Obtaining the type of the subscript
    int_18583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 12), 'int')
    # Getting the type of 'now' (line 172)
    now_18584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'now')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___18585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), now_18584, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_18586 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), getitem___18585, int_18583)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), tuple_18543, subscript_call_result_18586)
    # Adding element type (line 168)
    
    # Obtaining the type of the subscript
    int_18587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 20), 'int')
    # Getting the type of 'now' (line 172)
    now_18588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'now')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___18589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), now_18588, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_18590 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), getitem___18589, int_18587)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), tuple_18543, subscript_call_result_18590)
    # Adding element type (line 168)
    
    # Obtaining the type of the subscript
    int_18591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 28), 'int')
    # Getting the type of 'now' (line 172)
    now_18592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 24), 'now')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___18593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 24), now_18592, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_18594 = invoke(stypy.reporting.localization.Localization(__file__, 172, 24), getitem___18593, int_18591)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), tuple_18543, subscript_call_result_18594)
    # Adding element type (line 168)
    
    # Obtaining the type of the subscript
    int_18595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 36), 'int')
    # Getting the type of 'now' (line 172)
    now_18596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'now')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___18597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 32), now_18596, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_18598 = invoke(stypy.reporting.localization.Localization(__file__, 172, 32), getitem___18597, int_18595)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), tuple_18543, subscript_call_result_18598)
    # Adding element type (line 168)
    # Getting the type of 'zone' (line 173)
    zone_18599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'zone')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), tuple_18543, zone_18599)
    
    # Applying the binary operator '%' (line 167)
    result_mod_18600 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 11), '%', str_18542, tuple_18543)
    
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type', result_mod_18600)
    
    # ################# End of 'formatdate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'formatdate' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_18601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18601)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'formatdate'
    return stypy_return_type_18601

# Assigning a type to the variable 'formatdate' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'formatdate', formatdate)

@norecursion
def make_msgid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 177)
    None_18602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'None')
    defaults = [None_18602]
    # Create a new context for function 'make_msgid'
    module_type_store = module_type_store.open_function_context('make_msgid', 177, 0, False)
    
    # Passed parameters checking function
    make_msgid.stypy_localization = localization
    make_msgid.stypy_type_of_self = None
    make_msgid.stypy_type_store = module_type_store
    make_msgid.stypy_function_name = 'make_msgid'
    make_msgid.stypy_param_names_list = ['idstring']
    make_msgid.stypy_varargs_param_name = None
    make_msgid.stypy_kwargs_param_name = None
    make_msgid.stypy_call_defaults = defaults
    make_msgid.stypy_call_varargs = varargs
    make_msgid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_msgid', ['idstring'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_msgid', localization, ['idstring'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_msgid(...)' code ##################

    str_18603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, (-1)), 'str', 'Returns a string suitable for RFC 2822 compliant Message-ID, e.g:\n\n    <142480216486.20800.16526388040877946887@nightshade.la.mastaler.com>\n\n    Optional idstring if given is a string used to strengthen the\n    uniqueness of the message id.\n    ')
    
    # Assigning a Call to a Name (line 185):
    
    # Assigning a Call to a Name (line 185):
    
    # Call to int(...): (line 185)
    # Processing the call arguments (line 185)
    
    # Call to time(...): (line 185)
    # Processing the call keyword arguments (line 185)
    kwargs_18607 = {}
    # Getting the type of 'time' (line 185)
    time_18605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 18), 'time', False)
    # Obtaining the member 'time' of a type (line 185)
    time_18606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 18), time_18605, 'time')
    # Calling time(args, kwargs) (line 185)
    time_call_result_18608 = invoke(stypy.reporting.localization.Localization(__file__, 185, 18), time_18606, *[], **kwargs_18607)
    
    int_18609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'int')
    # Applying the binary operator '*' (line 185)
    result_mul_18610 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 18), '*', time_call_result_18608, int_18609)
    
    # Processing the call keyword arguments (line 185)
    kwargs_18611 = {}
    # Getting the type of 'int' (line 185)
    int_18604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 14), 'int', False)
    # Calling int(args, kwargs) (line 185)
    int_call_result_18612 = invoke(stypy.reporting.localization.Localization(__file__, 185, 14), int_18604, *[result_mul_18610], **kwargs_18611)
    
    # Assigning a type to the variable 'timeval' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'timeval', int_call_result_18612)
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to getpid(...): (line 186)
    # Processing the call keyword arguments (line 186)
    kwargs_18615 = {}
    # Getting the type of 'os' (line 186)
    os_18613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 10), 'os', False)
    # Obtaining the member 'getpid' of a type (line 186)
    getpid_18614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 10), os_18613, 'getpid')
    # Calling getpid(args, kwargs) (line 186)
    getpid_call_result_18616 = invoke(stypy.reporting.localization.Localization(__file__, 186, 10), getpid_18614, *[], **kwargs_18615)
    
    # Assigning a type to the variable 'pid' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'pid', getpid_call_result_18616)
    
    # Assigning a Call to a Name (line 187):
    
    # Assigning a Call to a Name (line 187):
    
    # Call to getrandbits(...): (line 187)
    # Processing the call arguments (line 187)
    int_18619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 33), 'int')
    # Processing the call keyword arguments (line 187)
    kwargs_18620 = {}
    # Getting the type of 'random' (line 187)
    random_18617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'random', False)
    # Obtaining the member 'getrandbits' of a type (line 187)
    getrandbits_18618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 14), random_18617, 'getrandbits')
    # Calling getrandbits(args, kwargs) (line 187)
    getrandbits_call_result_18621 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), getrandbits_18618, *[int_18619], **kwargs_18620)
    
    # Assigning a type to the variable 'randint' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'randint', getrandbits_call_result_18621)
    
    # Type idiom detected: calculating its left and rigth part (line 188)
    # Getting the type of 'idstring' (line 188)
    idstring_18622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 7), 'idstring')
    # Getting the type of 'None' (line 188)
    None_18623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'None')
    
    (may_be_18624, more_types_in_union_18625) = may_be_none(idstring_18622, None_18623)

    if may_be_18624:

        if more_types_in_union_18625:
            # Runtime conditional SSA (line 188)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 189):
        
        # Assigning a Str to a Name (line 189):
        str_18626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 19), 'str', '')
        # Assigning a type to the variable 'idstring' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'idstring', str_18626)

        if more_types_in_union_18625:
            # Runtime conditional SSA for else branch (line 188)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_18624) or more_types_in_union_18625):
        
        # Assigning a BinOp to a Name (line 191):
        
        # Assigning a BinOp to a Name (line 191):
        str_18627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 19), 'str', '.')
        # Getting the type of 'idstring' (line 191)
        idstring_18628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'idstring')
        # Applying the binary operator '+' (line 191)
        result_add_18629 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 19), '+', str_18627, idstring_18628)
        
        # Assigning a type to the variable 'idstring' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'idstring', result_add_18629)

        if (may_be_18624 and more_types_in_union_18625):
            # SSA join for if statement (line 188)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to getfqdn(...): (line 192)
    # Processing the call keyword arguments (line 192)
    kwargs_18632 = {}
    # Getting the type of 'socket' (line 192)
    socket_18630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 13), 'socket', False)
    # Obtaining the member 'getfqdn' of a type (line 192)
    getfqdn_18631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 13), socket_18630, 'getfqdn')
    # Calling getfqdn(args, kwargs) (line 192)
    getfqdn_call_result_18633 = invoke(stypy.reporting.localization.Localization(__file__, 192, 13), getfqdn_18631, *[], **kwargs_18632)
    
    # Assigning a type to the variable 'idhost' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'idhost', getfqdn_call_result_18633)
    
    # Assigning a BinOp to a Name (line 193):
    
    # Assigning a BinOp to a Name (line 193):
    str_18634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 12), 'str', '<%d.%d.%d%s@%s>')
    
    # Obtaining an instance of the builtin type 'tuple' (line 193)
    tuple_18635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 193)
    # Adding element type (line 193)
    # Getting the type of 'timeval' (line 193)
    timeval_18636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 33), 'timeval')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 33), tuple_18635, timeval_18636)
    # Adding element type (line 193)
    # Getting the type of 'pid' (line 193)
    pid_18637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 42), 'pid')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 33), tuple_18635, pid_18637)
    # Adding element type (line 193)
    # Getting the type of 'randint' (line 193)
    randint_18638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 47), 'randint')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 33), tuple_18635, randint_18638)
    # Adding element type (line 193)
    # Getting the type of 'idstring' (line 193)
    idstring_18639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 56), 'idstring')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 33), tuple_18635, idstring_18639)
    # Adding element type (line 193)
    # Getting the type of 'idhost' (line 193)
    idhost_18640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 66), 'idhost')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 33), tuple_18635, idhost_18640)
    
    # Applying the binary operator '%' (line 193)
    result_mod_18641 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 12), '%', str_18634, tuple_18635)
    
    # Assigning a type to the variable 'msgid' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'msgid', result_mod_18641)
    # Getting the type of 'msgid' (line 194)
    msgid_18642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'msgid')
    # Assigning a type to the variable 'stypy_return_type' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type', msgid_18642)
    
    # ################# End of 'make_msgid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_msgid' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_18643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18643)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_msgid'
    return stypy_return_type_18643

# Assigning a type to the variable 'make_msgid' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'make_msgid', make_msgid)

@norecursion
def parsedate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parsedate'
    module_type_store = module_type_store.open_function_context('parsedate', 201, 0, False)
    
    # Passed parameters checking function
    parsedate.stypy_localization = localization
    parsedate.stypy_type_of_self = None
    parsedate.stypy_type_store = module_type_store
    parsedate.stypy_function_name = 'parsedate'
    parsedate.stypy_param_names_list = ['data']
    parsedate.stypy_varargs_param_name = None
    parsedate.stypy_kwargs_param_name = None
    parsedate.stypy_call_defaults = defaults
    parsedate.stypy_call_varargs = varargs
    parsedate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parsedate', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parsedate', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parsedate(...)' code ##################

    
    # Getting the type of 'data' (line 202)
    data_18644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'data')
    # Applying the 'not' unary operator (line 202)
    result_not__18645 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 7), 'not', data_18644)
    
    # Testing if the type of an if condition is none (line 202)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 202, 4), result_not__18645):
        pass
    else:
        
        # Testing the type of an if condition (line 202)
        if_condition_18646 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 4), result_not__18645)
        # Assigning a type to the variable 'if_condition_18646' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'if_condition_18646', if_condition_18646)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 203)
        None_18647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', None_18647)
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to _parsedate(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'data' (line 204)
    data_18649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'data', False)
    # Processing the call keyword arguments (line 204)
    kwargs_18650 = {}
    # Getting the type of '_parsedate' (line 204)
    _parsedate_18648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), '_parsedate', False)
    # Calling _parsedate(args, kwargs) (line 204)
    _parsedate_call_result_18651 = invoke(stypy.reporting.localization.Localization(__file__, 204, 11), _parsedate_18648, *[data_18649], **kwargs_18650)
    
    # Assigning a type to the variable 'stypy_return_type' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type', _parsedate_call_result_18651)
    
    # ################# End of 'parsedate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parsedate' in the type store
    # Getting the type of 'stypy_return_type' (line 201)
    stypy_return_type_18652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18652)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parsedate'
    return stypy_return_type_18652

# Assigning a type to the variable 'parsedate' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'parsedate', parsedate)

@norecursion
def parsedate_tz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parsedate_tz'
    module_type_store = module_type_store.open_function_context('parsedate_tz', 207, 0, False)
    
    # Passed parameters checking function
    parsedate_tz.stypy_localization = localization
    parsedate_tz.stypy_type_of_self = None
    parsedate_tz.stypy_type_store = module_type_store
    parsedate_tz.stypy_function_name = 'parsedate_tz'
    parsedate_tz.stypy_param_names_list = ['data']
    parsedate_tz.stypy_varargs_param_name = None
    parsedate_tz.stypy_kwargs_param_name = None
    parsedate_tz.stypy_call_defaults = defaults
    parsedate_tz.stypy_call_varargs = varargs
    parsedate_tz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parsedate_tz', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parsedate_tz', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parsedate_tz(...)' code ##################

    
    # Getting the type of 'data' (line 208)
    data_18653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'data')
    # Applying the 'not' unary operator (line 208)
    result_not__18654 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 7), 'not', data_18653)
    
    # Testing if the type of an if condition is none (line 208)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 208, 4), result_not__18654):
        pass
    else:
        
        # Testing the type of an if condition (line 208)
        if_condition_18655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 4), result_not__18654)
        # Assigning a type to the variable 'if_condition_18655' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'if_condition_18655', if_condition_18655)
        # SSA begins for if statement (line 208)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 209)
        None_18656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'stypy_return_type', None_18656)
        # SSA join for if statement (line 208)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to _parsedate_tz(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'data' (line 210)
    data_18658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 25), 'data', False)
    # Processing the call keyword arguments (line 210)
    kwargs_18659 = {}
    # Getting the type of '_parsedate_tz' (line 210)
    _parsedate_tz_18657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), '_parsedate_tz', False)
    # Calling _parsedate_tz(args, kwargs) (line 210)
    _parsedate_tz_call_result_18660 = invoke(stypy.reporting.localization.Localization(__file__, 210, 11), _parsedate_tz_18657, *[data_18658], **kwargs_18659)
    
    # Assigning a type to the variable 'stypy_return_type' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type', _parsedate_tz_call_result_18660)
    
    # ################# End of 'parsedate_tz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parsedate_tz' in the type store
    # Getting the type of 'stypy_return_type' (line 207)
    stypy_return_type_18661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18661)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parsedate_tz'
    return stypy_return_type_18661

# Assigning a type to the variable 'parsedate_tz' (line 207)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'parsedate_tz', parsedate_tz)

@norecursion
def parseaddr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parseaddr'
    module_type_store = module_type_store.open_function_context('parseaddr', 213, 0, False)
    
    # Passed parameters checking function
    parseaddr.stypy_localization = localization
    parseaddr.stypy_type_of_self = None
    parseaddr.stypy_type_store = module_type_store
    parseaddr.stypy_function_name = 'parseaddr'
    parseaddr.stypy_param_names_list = ['addr']
    parseaddr.stypy_varargs_param_name = None
    parseaddr.stypy_kwargs_param_name = None
    parseaddr.stypy_call_defaults = defaults
    parseaddr.stypy_call_varargs = varargs
    parseaddr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parseaddr', ['addr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parseaddr', localization, ['addr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parseaddr(...)' code ##################

    
    # Assigning a Attribute to a Name (line 214):
    
    # Assigning a Attribute to a Name (line 214):
    
    # Call to _AddressList(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'addr' (line 214)
    addr_18663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 25), 'addr', False)
    # Processing the call keyword arguments (line 214)
    kwargs_18664 = {}
    # Getting the type of '_AddressList' (line 214)
    _AddressList_18662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), '_AddressList', False)
    # Calling _AddressList(args, kwargs) (line 214)
    _AddressList_call_result_18665 = invoke(stypy.reporting.localization.Localization(__file__, 214, 12), _AddressList_18662, *[addr_18663], **kwargs_18664)
    
    # Obtaining the member 'addresslist' of a type (line 214)
    addresslist_18666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), _AddressList_call_result_18665, 'addresslist')
    # Assigning a type to the variable 'addrs' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'addrs', addresslist_18666)
    
    # Getting the type of 'addrs' (line 215)
    addrs_18667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'addrs')
    # Applying the 'not' unary operator (line 215)
    result_not__18668 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 7), 'not', addrs_18667)
    
    # Testing if the type of an if condition is none (line 215)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 215, 4), result_not__18668):
        pass
    else:
        
        # Testing the type of an if condition (line 215)
        if_condition_18669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 4), result_not__18668)
        # Assigning a type to the variable 'if_condition_18669' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'if_condition_18669', if_condition_18669)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 216)
        tuple_18670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 216)
        # Adding element type (line 216)
        str_18671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 15), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 15), tuple_18670, str_18671)
        # Adding element type (line 216)
        str_18672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 19), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 15), tuple_18670, str_18672)
        
        # Assigning a type to the variable 'stypy_return_type' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', tuple_18670)
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining the type of the subscript
    int_18673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 17), 'int')
    # Getting the type of 'addrs' (line 217)
    addrs_18674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'addrs')
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___18675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 11), addrs_18674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_18676 = invoke(stypy.reporting.localization.Localization(__file__, 217, 11), getitem___18675, int_18673)
    
    # Assigning a type to the variable 'stypy_return_type' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type', subscript_call_result_18676)
    
    # ################# End of 'parseaddr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parseaddr' in the type store
    # Getting the type of 'stypy_return_type' (line 213)
    stypy_return_type_18677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18677)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parseaddr'
    return stypy_return_type_18677

# Assigning a type to the variable 'parseaddr' (line 213)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'parseaddr', parseaddr)

@norecursion
def unquote(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unquote'
    module_type_store = module_type_store.open_function_context('unquote', 221, 0, False)
    
    # Passed parameters checking function
    unquote.stypy_localization = localization
    unquote.stypy_type_of_self = None
    unquote.stypy_type_store = module_type_store
    unquote.stypy_function_name = 'unquote'
    unquote.stypy_param_names_list = ['str']
    unquote.stypy_varargs_param_name = None
    unquote.stypy_kwargs_param_name = None
    unquote.stypy_call_defaults = defaults
    unquote.stypy_call_varargs = varargs
    unquote.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unquote', ['str'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unquote', localization, ['str'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unquote(...)' code ##################

    str_18678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 4), 'str', 'Remove quotes from a string.')
    
    
    # Call to len(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'str' (line 223)
    str_18680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'str', False)
    # Processing the call keyword arguments (line 223)
    kwargs_18681 = {}
    # Getting the type of 'len' (line 223)
    len_18679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 7), 'len', False)
    # Calling len(args, kwargs) (line 223)
    len_call_result_18682 = invoke(stypy.reporting.localization.Localization(__file__, 223, 7), len_18679, *[str_18680], **kwargs_18681)
    
    int_18683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 18), 'int')
    # Applying the binary operator '>' (line 223)
    result_gt_18684 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 7), '>', len_call_result_18682, int_18683)
    
    # Testing if the type of an if condition is none (line 223)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 223, 4), result_gt_18684):
        pass
    else:
        
        # Testing the type of an if condition (line 223)
        if_condition_18685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 4), result_gt_18684)
        # Assigning a type to the variable 'if_condition_18685' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'if_condition_18685', if_condition_18685)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Evaluating a boolean operation
        
        # Call to startswith(...): (line 224)
        # Processing the call arguments (line 224)
        str_18688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 26), 'str', '"')
        # Processing the call keyword arguments (line 224)
        kwargs_18689 = {}
        # Getting the type of 'str' (line 224)
        str_18686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 11), 'str', False)
        # Obtaining the member 'startswith' of a type (line 224)
        startswith_18687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 11), str_18686, 'startswith')
        # Calling startswith(args, kwargs) (line 224)
        startswith_call_result_18690 = invoke(stypy.reporting.localization.Localization(__file__, 224, 11), startswith_18687, *[str_18688], **kwargs_18689)
        
        
        # Call to endswith(...): (line 224)
        # Processing the call arguments (line 224)
        str_18693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 48), 'str', '"')
        # Processing the call keyword arguments (line 224)
        kwargs_18694 = {}
        # Getting the type of 'str' (line 224)
        str_18691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 35), 'str', False)
        # Obtaining the member 'endswith' of a type (line 224)
        endswith_18692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 35), str_18691, 'endswith')
        # Calling endswith(args, kwargs) (line 224)
        endswith_call_result_18695 = invoke(stypy.reporting.localization.Localization(__file__, 224, 35), endswith_18692, *[str_18693], **kwargs_18694)
        
        # Applying the binary operator 'and' (line 224)
        result_and_keyword_18696 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 11), 'and', startswith_call_result_18690, endswith_call_result_18695)
        
        # Testing if the type of an if condition is none (line 224)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 224, 8), result_and_keyword_18696):
            pass
        else:
            
            # Testing the type of an if condition (line 224)
            if_condition_18697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 8), result_and_keyword_18696)
            # Assigning a type to the variable 'if_condition_18697' (line 224)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'if_condition_18697', if_condition_18697)
            # SSA begins for if statement (line 224)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to replace(...): (line 225)
            # Processing the call arguments (line 225)
            str_18710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 59), 'str', '\\"')
            str_18711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 66), 'str', '"')
            # Processing the call keyword arguments (line 225)
            kwargs_18712 = {}
            
            # Call to replace(...): (line 225)
            # Processing the call arguments (line 225)
            str_18705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 37), 'str', '\\\\')
            str_18706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 45), 'str', '\\')
            # Processing the call keyword arguments (line 225)
            kwargs_18707 = {}
            
            # Obtaining the type of the subscript
            int_18698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 23), 'int')
            int_18699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 25), 'int')
            slice_18700 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 225, 19), int_18698, int_18699, None)
            # Getting the type of 'str' (line 225)
            str_18701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 19), 'str', False)
            # Obtaining the member '__getitem__' of a type (line 225)
            getitem___18702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 19), str_18701, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 225)
            subscript_call_result_18703 = invoke(stypy.reporting.localization.Localization(__file__, 225, 19), getitem___18702, slice_18700)
            
            # Obtaining the member 'replace' of a type (line 225)
            replace_18704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 19), subscript_call_result_18703, 'replace')
            # Calling replace(args, kwargs) (line 225)
            replace_call_result_18708 = invoke(stypy.reporting.localization.Localization(__file__, 225, 19), replace_18704, *[str_18705, str_18706], **kwargs_18707)
            
            # Obtaining the member 'replace' of a type (line 225)
            replace_18709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 19), replace_call_result_18708, 'replace')
            # Calling replace(args, kwargs) (line 225)
            replace_call_result_18713 = invoke(stypy.reporting.localization.Localization(__file__, 225, 19), replace_18709, *[str_18710, str_18711], **kwargs_18712)
            
            # Assigning a type to the variable 'stypy_return_type' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'stypy_return_type', replace_call_result_18713)
            # SSA join for if statement (line 224)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Call to startswith(...): (line 226)
        # Processing the call arguments (line 226)
        str_18716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 26), 'str', '<')
        # Processing the call keyword arguments (line 226)
        kwargs_18717 = {}
        # Getting the type of 'str' (line 226)
        str_18714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), 'str', False)
        # Obtaining the member 'startswith' of a type (line 226)
        startswith_18715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 11), str_18714, 'startswith')
        # Calling startswith(args, kwargs) (line 226)
        startswith_call_result_18718 = invoke(stypy.reporting.localization.Localization(__file__, 226, 11), startswith_18715, *[str_18716], **kwargs_18717)
        
        
        # Call to endswith(...): (line 226)
        # Processing the call arguments (line 226)
        str_18721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 48), 'str', '>')
        # Processing the call keyword arguments (line 226)
        kwargs_18722 = {}
        # Getting the type of 'str' (line 226)
        str_18719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 35), 'str', False)
        # Obtaining the member 'endswith' of a type (line 226)
        endswith_18720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 35), str_18719, 'endswith')
        # Calling endswith(args, kwargs) (line 226)
        endswith_call_result_18723 = invoke(stypy.reporting.localization.Localization(__file__, 226, 35), endswith_18720, *[str_18721], **kwargs_18722)
        
        # Applying the binary operator 'and' (line 226)
        result_and_keyword_18724 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 11), 'and', startswith_call_result_18718, endswith_call_result_18723)
        
        # Testing if the type of an if condition is none (line 226)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 226, 8), result_and_keyword_18724):
            pass
        else:
            
            # Testing the type of an if condition (line 226)
            if_condition_18725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 8), result_and_keyword_18724)
            # Assigning a type to the variable 'if_condition_18725' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'if_condition_18725', if_condition_18725)
            # SSA begins for if statement (line 226)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_18726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 23), 'int')
            int_18727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 25), 'int')
            slice_18728 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 227, 19), int_18726, int_18727, None)
            # Getting the type of 'str' (line 227)
            str_18729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 19), 'str')
            # Obtaining the member '__getitem__' of a type (line 227)
            getitem___18730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 19), str_18729, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 227)
            subscript_call_result_18731 = invoke(stypy.reporting.localization.Localization(__file__, 227, 19), getitem___18730, slice_18728)
            
            # Assigning a type to the variable 'stypy_return_type' (line 227)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'stypy_return_type', subscript_call_result_18731)
            # SSA join for if statement (line 226)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'str' (line 228)
    str_18732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'str')
    # Assigning a type to the variable 'stypy_return_type' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type', str_18732)
    
    # ################# End of 'unquote(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unquote' in the type store
    # Getting the type of 'stypy_return_type' (line 221)
    stypy_return_type_18733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18733)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unquote'
    return stypy_return_type_18733

# Assigning a type to the variable 'unquote' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'unquote', unquote)

@norecursion
def decode_rfc2231(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'decode_rfc2231'
    module_type_store = module_type_store.open_function_context('decode_rfc2231', 233, 0, False)
    
    # Passed parameters checking function
    decode_rfc2231.stypy_localization = localization
    decode_rfc2231.stypy_type_of_self = None
    decode_rfc2231.stypy_type_store = module_type_store
    decode_rfc2231.stypy_function_name = 'decode_rfc2231'
    decode_rfc2231.stypy_param_names_list = ['s']
    decode_rfc2231.stypy_varargs_param_name = None
    decode_rfc2231.stypy_kwargs_param_name = None
    decode_rfc2231.stypy_call_defaults = defaults
    decode_rfc2231.stypy_call_varargs = varargs
    decode_rfc2231.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decode_rfc2231', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decode_rfc2231', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decode_rfc2231(...)' code ##################

    str_18734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 4), 'str', 'Decode string according to RFC 2231')
    
    # Assigning a Call to a Name (line 235):
    
    # Assigning a Call to a Name (line 235):
    
    # Call to split(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'TICK' (line 235)
    TICK_18737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'TICK', False)
    int_18738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 26), 'int')
    # Processing the call keyword arguments (line 235)
    kwargs_18739 = {}
    # Getting the type of 's' (line 235)
    s_18735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 's', False)
    # Obtaining the member 'split' of a type (line 235)
    split_18736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), s_18735, 'split')
    # Calling split(args, kwargs) (line 235)
    split_call_result_18740 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), split_18736, *[TICK_18737, int_18738], **kwargs_18739)
    
    # Assigning a type to the variable 'parts' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'parts', split_call_result_18740)
    
    
    # Call to len(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'parts' (line 236)
    parts_18742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'parts', False)
    # Processing the call keyword arguments (line 236)
    kwargs_18743 = {}
    # Getting the type of 'len' (line 236)
    len_18741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 7), 'len', False)
    # Calling len(args, kwargs) (line 236)
    len_call_result_18744 = invoke(stypy.reporting.localization.Localization(__file__, 236, 7), len_18741, *[parts_18742], **kwargs_18743)
    
    int_18745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 21), 'int')
    # Applying the binary operator '<=' (line 236)
    result_le_18746 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 7), '<=', len_call_result_18744, int_18745)
    
    # Testing if the type of an if condition is none (line 236)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 236, 4), result_le_18746):
        pass
    else:
        
        # Testing the type of an if condition (line 236)
        if_condition_18747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 4), result_le_18746)
        # Assigning a type to the variable 'if_condition_18747' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'if_condition_18747', if_condition_18747)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 237)
        tuple_18748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 237)
        # Adding element type (line 237)
        # Getting the type of 'None' (line 237)
        None_18749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), tuple_18748, None_18749)
        # Adding element type (line 237)
        # Getting the type of 'None' (line 237)
        None_18750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), tuple_18748, None_18750)
        # Adding element type (line 237)
        # Getting the type of 's' (line 237)
        s_18751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 's')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), tuple_18748, s_18751)
        
        # Assigning a type to the variable 'stypy_return_type' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type', tuple_18748)
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'parts' (line 238)
    parts_18752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'parts')
    # Assigning a type to the variable 'stypy_return_type' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type', parts_18752)
    
    # ################# End of 'decode_rfc2231(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode_rfc2231' in the type store
    # Getting the type of 'stypy_return_type' (line 233)
    stypy_return_type_18753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18753)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode_rfc2231'
    return stypy_return_type_18753

# Assigning a type to the variable 'decode_rfc2231' (line 233)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'decode_rfc2231', decode_rfc2231)

@norecursion
def encode_rfc2231(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 241)
    None_18754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 30), 'None')
    # Getting the type of 'None' (line 241)
    None_18755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 45), 'None')
    defaults = [None_18754, None_18755]
    # Create a new context for function 'encode_rfc2231'
    module_type_store = module_type_store.open_function_context('encode_rfc2231', 241, 0, False)
    
    # Passed parameters checking function
    encode_rfc2231.stypy_localization = localization
    encode_rfc2231.stypy_type_of_self = None
    encode_rfc2231.stypy_type_store = module_type_store
    encode_rfc2231.stypy_function_name = 'encode_rfc2231'
    encode_rfc2231.stypy_param_names_list = ['s', 'charset', 'language']
    encode_rfc2231.stypy_varargs_param_name = None
    encode_rfc2231.stypy_kwargs_param_name = None
    encode_rfc2231.stypy_call_defaults = defaults
    encode_rfc2231.stypy_call_varargs = varargs
    encode_rfc2231.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode_rfc2231', ['s', 'charset', 'language'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode_rfc2231', localization, ['s', 'charset', 'language'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode_rfc2231(...)' code ##################

    str_18756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, (-1)), 'str', 'Encode string according to RFC 2231.\n\n    If neither charset nor language is given, then s is returned as-is.  If\n    charset is given but not language, the string is encoded using the empty\n    string for language.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 248, 4))
    
    # 'import urllib' statement (line 248)
    import urllib

    import_module(stypy.reporting.localization.Localization(__file__, 248, 4), 'urllib', urllib, module_type_store)
    
    
    # Assigning a Call to a Name (line 249):
    
    # Assigning a Call to a Name (line 249):
    
    # Call to quote(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 's' (line 249)
    s_18759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 's', False)
    # Processing the call keyword arguments (line 249)
    str_18760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 29), 'str', '')
    keyword_18761 = str_18760
    kwargs_18762 = {'safe': keyword_18761}
    # Getting the type of 'urllib' (line 249)
    urllib_18757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'urllib', False)
    # Obtaining the member 'quote' of a type (line 249)
    quote_18758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), urllib_18757, 'quote')
    # Calling quote(args, kwargs) (line 249)
    quote_call_result_18763 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), quote_18758, *[s_18759], **kwargs_18762)
    
    # Assigning a type to the variable 's' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 's', quote_call_result_18763)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'charset' (line 250)
    charset_18764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 7), 'charset')
    # Getting the type of 'None' (line 250)
    None_18765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 18), 'None')
    # Applying the binary operator 'is' (line 250)
    result_is__18766 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 7), 'is', charset_18764, None_18765)
    
    
    # Getting the type of 'language' (line 250)
    language_18767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'language')
    # Getting the type of 'None' (line 250)
    None_18768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 39), 'None')
    # Applying the binary operator 'is' (line 250)
    result_is__18769 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 27), 'is', language_18767, None_18768)
    
    # Applying the binary operator 'and' (line 250)
    result_and_keyword_18770 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 7), 'and', result_is__18766, result_is__18769)
    
    # Testing if the type of an if condition is none (line 250)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 250, 4), result_and_keyword_18770):
        pass
    else:
        
        # Testing the type of an if condition (line 250)
        if_condition_18771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 4), result_and_keyword_18770)
        # Assigning a type to the variable 'if_condition_18771' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'if_condition_18771', if_condition_18771)
        # SSA begins for if statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 's' (line 251)
        s_18772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'stypy_return_type', s_18772)
        # SSA join for if statement (line 250)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Type idiom detected: calculating its left and rigth part (line 252)
    # Getting the type of 'language' (line 252)
    language_18773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 7), 'language')
    # Getting the type of 'None' (line 252)
    None_18774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'None')
    
    (may_be_18775, more_types_in_union_18776) = may_be_none(language_18773, None_18774)

    if may_be_18775:

        if more_types_in_union_18776:
            # Runtime conditional SSA (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 253):
        
        # Assigning a Str to a Name (line 253):
        str_18777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 19), 'str', '')
        # Assigning a type to the variable 'language' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'language', str_18777)

        if more_types_in_union_18776:
            # SSA join for if statement (line 252)
            module_type_store = module_type_store.join_ssa_context()


    
    str_18778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 11), 'str', "%s'%s'%s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 254)
    tuple_18779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 254)
    # Adding element type (line 254)
    # Getting the type of 'charset' (line 254)
    charset_18780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 25), 'charset')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 25), tuple_18779, charset_18780)
    # Adding element type (line 254)
    # Getting the type of 'language' (line 254)
    language_18781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'language')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 25), tuple_18779, language_18781)
    # Adding element type (line 254)
    # Getting the type of 's' (line 254)
    s_18782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 44), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 25), tuple_18779, s_18782)
    
    # Applying the binary operator '%' (line 254)
    result_mod_18783 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 11), '%', str_18778, tuple_18779)
    
    # Assigning a type to the variable 'stypy_return_type' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type', result_mod_18783)
    
    # ################# End of 'encode_rfc2231(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode_rfc2231' in the type store
    # Getting the type of 'stypy_return_type' (line 241)
    stypy_return_type_18784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18784)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode_rfc2231'
    return stypy_return_type_18784

# Assigning a type to the variable 'encode_rfc2231' (line 241)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'encode_rfc2231', encode_rfc2231)

# Assigning a Call to a Name (line 257):

# Assigning a Call to a Name (line 257):

# Call to compile(...): (line 257)
# Processing the call arguments (line 257)
str_18787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 34), 'str', '^(?P<name>\\w+)\\*((?P<num>[0-9]+)\\*?)?$')
# Processing the call keyword arguments (line 257)
kwargs_18788 = {}
# Getting the type of 're' (line 257)
re_18785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 23), 're', False)
# Obtaining the member 'compile' of a type (line 257)
compile_18786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 23), re_18785, 'compile')
# Calling compile(args, kwargs) (line 257)
compile_call_result_18789 = invoke(stypy.reporting.localization.Localization(__file__, 257, 23), compile_18786, *[str_18787], **kwargs_18788)

# Assigning a type to the variable 'rfc2231_continuation' (line 257)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'rfc2231_continuation', compile_call_result_18789)

@norecursion
def decode_params(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'decode_params'
    module_type_store = module_type_store.open_function_context('decode_params', 259, 0, False)
    
    # Passed parameters checking function
    decode_params.stypy_localization = localization
    decode_params.stypy_type_of_self = None
    decode_params.stypy_type_store = module_type_store
    decode_params.stypy_function_name = 'decode_params'
    decode_params.stypy_param_names_list = ['params']
    decode_params.stypy_varargs_param_name = None
    decode_params.stypy_kwargs_param_name = None
    decode_params.stypy_call_defaults = defaults
    decode_params.stypy_call_varargs = varargs
    decode_params.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decode_params', ['params'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decode_params', localization, ['params'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decode_params(...)' code ##################

    str_18790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, (-1)), 'str', 'Decode parameters list according to RFC 2231.\n\n    params is a sequence of 2-tuples containing (param name, string value).\n    ')
    
    # Assigning a Subscript to a Name (line 265):
    
    # Assigning a Subscript to a Name (line 265):
    
    # Obtaining the type of the subscript
    slice_18791 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 265, 13), None, None, None)
    # Getting the type of 'params' (line 265)
    params_18792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'params')
    # Obtaining the member '__getitem__' of a type (line 265)
    getitem___18793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 13), params_18792, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 265)
    subscript_call_result_18794 = invoke(stypy.reporting.localization.Localization(__file__, 265, 13), getitem___18793, slice_18791)
    
    # Assigning a type to the variable 'params' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'params', subscript_call_result_18794)
    
    # Assigning a List to a Name (line 266):
    
    # Assigning a List to a Name (line 266):
    
    # Obtaining an instance of the builtin type 'list' (line 266)
    list_18795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 266)
    
    # Assigning a type to the variable 'new_params' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'new_params', list_18795)
    
    # Assigning a Dict to a Name (line 270):
    
    # Assigning a Dict to a Name (line 270):
    
    # Obtaining an instance of the builtin type 'dict' (line 270)
    dict_18796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 21), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 270)
    
    # Assigning a type to the variable 'rfc2231_params' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'rfc2231_params', dict_18796)
    
    # Assigning a Call to a Tuple (line 271):
    
    # Assigning a Call to a Name:
    
    # Call to pop(...): (line 271)
    # Processing the call arguments (line 271)
    int_18799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 29), 'int')
    # Processing the call keyword arguments (line 271)
    kwargs_18800 = {}
    # Getting the type of 'params' (line 271)
    params_18797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 18), 'params', False)
    # Obtaining the member 'pop' of a type (line 271)
    pop_18798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 18), params_18797, 'pop')
    # Calling pop(args, kwargs) (line 271)
    pop_call_result_18801 = invoke(stypy.reporting.localization.Localization(__file__, 271, 18), pop_18798, *[int_18799], **kwargs_18800)
    
    # Assigning a type to the variable 'call_assignment_18330' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'call_assignment_18330', pop_call_result_18801)
    
    # Assigning a Call to a Name (line 271):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_18330' (line 271)
    call_assignment_18330_18802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'call_assignment_18330', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_18803 = stypy_get_value_from_tuple(call_assignment_18330_18802, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_18331' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'call_assignment_18331', stypy_get_value_from_tuple_call_result_18803)
    
    # Assigning a Name to a Name (line 271):
    # Getting the type of 'call_assignment_18331' (line 271)
    call_assignment_18331_18804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'call_assignment_18331')
    # Assigning a type to the variable 'name' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'name', call_assignment_18331_18804)
    
    # Assigning a Call to a Name (line 271):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_18330' (line 271)
    call_assignment_18330_18805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'call_assignment_18330', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_18806 = stypy_get_value_from_tuple(call_assignment_18330_18805, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_18332' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'call_assignment_18332', stypy_get_value_from_tuple_call_result_18806)
    
    # Assigning a Name to a Name (line 271):
    # Getting the type of 'call_assignment_18332' (line 271)
    call_assignment_18332_18807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'call_assignment_18332')
    # Assigning a type to the variable 'value' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 10), 'value', call_assignment_18332_18807)
    
    # Call to append(...): (line 272)
    # Processing the call arguments (line 272)
    
    # Obtaining an instance of the builtin type 'tuple' (line 272)
    tuple_18810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 272)
    # Adding element type (line 272)
    # Getting the type of 'name' (line 272)
    name_18811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 23), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 23), tuple_18810, name_18811)
    # Adding element type (line 272)
    # Getting the type of 'value' (line 272)
    value_18812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 29), 'value', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 23), tuple_18810, value_18812)
    
    # Processing the call keyword arguments (line 272)
    kwargs_18813 = {}
    # Getting the type of 'new_params' (line 272)
    new_params_18808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'new_params', False)
    # Obtaining the member 'append' of a type (line 272)
    append_18809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 4), new_params_18808, 'append')
    # Calling append(args, kwargs) (line 272)
    append_call_result_18814 = invoke(stypy.reporting.localization.Localization(__file__, 272, 4), append_18809, *[tuple_18810], **kwargs_18813)
    
    
    # Getting the type of 'params' (line 273)
    params_18815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 10), 'params')
    # Assigning a type to the variable 'params_18815' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'params_18815', params_18815)
    # Testing if the while is going to be iterated (line 273)
    # Testing the type of an if condition (line 273)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 4), params_18815)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 273, 4), params_18815):
        # SSA begins for while statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Tuple (line 274):
        
        # Assigning a Call to a Name:
        
        # Call to pop(...): (line 274)
        # Processing the call arguments (line 274)
        int_18818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 33), 'int')
        # Processing the call keyword arguments (line 274)
        kwargs_18819 = {}
        # Getting the type of 'params' (line 274)
        params_18816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'params', False)
        # Obtaining the member 'pop' of a type (line 274)
        pop_18817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 22), params_18816, 'pop')
        # Calling pop(args, kwargs) (line 274)
        pop_call_result_18820 = invoke(stypy.reporting.localization.Localization(__file__, 274, 22), pop_18817, *[int_18818], **kwargs_18819)
        
        # Assigning a type to the variable 'call_assignment_18333' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_18333', pop_call_result_18820)
        
        # Assigning a Call to a Name (line 274):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_18333' (line 274)
        call_assignment_18333_18821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_18333', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_18822 = stypy_get_value_from_tuple(call_assignment_18333_18821, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_18334' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_18334', stypy_get_value_from_tuple_call_result_18822)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'call_assignment_18334' (line 274)
        call_assignment_18334_18823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_18334')
        # Assigning a type to the variable 'name' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'name', call_assignment_18334_18823)
        
        # Assigning a Call to a Name (line 274):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_18333' (line 274)
        call_assignment_18333_18824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_18333', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_18825 = stypy_get_value_from_tuple(call_assignment_18333_18824, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_18335' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_18335', stypy_get_value_from_tuple_call_result_18825)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'call_assignment_18335' (line 274)
        call_assignment_18335_18826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'call_assignment_18335')
        # Assigning a type to the variable 'value' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 14), 'value', call_assignment_18335_18826)
        
        # Call to endswith(...): (line 275)
        # Processing the call arguments (line 275)
        str_18829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 25), 'str', '*')
        # Processing the call keyword arguments (line 275)
        kwargs_18830 = {}
        # Getting the type of 'name' (line 275)
        name_18827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'name', False)
        # Obtaining the member 'endswith' of a type (line 275)
        endswith_18828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 11), name_18827, 'endswith')
        # Calling endswith(args, kwargs) (line 275)
        endswith_call_result_18831 = invoke(stypy.reporting.localization.Localization(__file__, 275, 11), endswith_18828, *[str_18829], **kwargs_18830)
        
        # Testing if the type of an if condition is none (line 275)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 275, 8), endswith_call_result_18831):
            
            # Assigning a Name to a Name (line 278):
            
            # Assigning a Name to a Name (line 278):
            # Getting the type of 'False' (line 278)
            False_18834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'False')
            # Assigning a type to the variable 'encoded' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'encoded', False_18834)
        else:
            
            # Testing the type of an if condition (line 275)
            if_condition_18832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 8), endswith_call_result_18831)
            # Assigning a type to the variable 'if_condition_18832' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'if_condition_18832', if_condition_18832)
            # SSA begins for if statement (line 275)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 276):
            
            # Assigning a Name to a Name (line 276):
            # Getting the type of 'True' (line 276)
            True_18833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'True')
            # Assigning a type to the variable 'encoded' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'encoded', True_18833)
            # SSA branch for the else part of an if statement (line 275)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 278):
            
            # Assigning a Name to a Name (line 278):
            # Getting the type of 'False' (line 278)
            False_18834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'False')
            # Assigning a type to the variable 'encoded' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'encoded', False_18834)
            # SSA join for if statement (line 275)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 279):
        
        # Assigning a Call to a Name (line 279):
        
        # Call to unquote(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'value' (line 279)
        value_18836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'value', False)
        # Processing the call keyword arguments (line 279)
        kwargs_18837 = {}
        # Getting the type of 'unquote' (line 279)
        unquote_18835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'unquote', False)
        # Calling unquote(args, kwargs) (line 279)
        unquote_call_result_18838 = invoke(stypy.reporting.localization.Localization(__file__, 279, 16), unquote_18835, *[value_18836], **kwargs_18837)
        
        # Assigning a type to the variable 'value' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'value', unquote_call_result_18838)
        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to match(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'name' (line 280)
        name_18841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 40), 'name', False)
        # Processing the call keyword arguments (line 280)
        kwargs_18842 = {}
        # Getting the type of 'rfc2231_continuation' (line 280)
        rfc2231_continuation_18839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 13), 'rfc2231_continuation', False)
        # Obtaining the member 'match' of a type (line 280)
        match_18840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 13), rfc2231_continuation_18839, 'match')
        # Calling match(args, kwargs) (line 280)
        match_call_result_18843 = invoke(stypy.reporting.localization.Localization(__file__, 280, 13), match_18840, *[name_18841], **kwargs_18842)
        
        # Assigning a type to the variable 'mo' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'mo', match_call_result_18843)
        # Getting the type of 'mo' (line 281)
        mo_18844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 'mo')
        # Testing if the type of an if condition is none (line 281)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 281, 8), mo_18844):
            
            # Call to append(...): (line 287)
            # Processing the call arguments (line 287)
            
            # Obtaining an instance of the builtin type 'tuple' (line 287)
            tuple_18881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 287)
            # Adding element type (line 287)
            # Getting the type of 'name' (line 287)
            name_18882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'name', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 31), tuple_18881, name_18882)
            # Adding element type (line 287)
            str_18883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 37), 'str', '"%s"')
            
            # Call to quote(...): (line 287)
            # Processing the call arguments (line 287)
            # Getting the type of 'value' (line 287)
            value_18885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 52), 'value', False)
            # Processing the call keyword arguments (line 287)
            kwargs_18886 = {}
            # Getting the type of 'quote' (line 287)
            quote_18884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 46), 'quote', False)
            # Calling quote(args, kwargs) (line 287)
            quote_call_result_18887 = invoke(stypy.reporting.localization.Localization(__file__, 287, 46), quote_18884, *[value_18885], **kwargs_18886)
            
            # Applying the binary operator '%' (line 287)
            result_mod_18888 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 37), '%', str_18883, quote_call_result_18887)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 31), tuple_18881, result_mod_18888)
            
            # Processing the call keyword arguments (line 287)
            kwargs_18889 = {}
            # Getting the type of 'new_params' (line 287)
            new_params_18879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'new_params', False)
            # Obtaining the member 'append' of a type (line 287)
            append_18880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), new_params_18879, 'append')
            # Calling append(args, kwargs) (line 287)
            append_call_result_18890 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), append_18880, *[tuple_18881], **kwargs_18889)
            
        else:
            
            # Testing the type of an if condition (line 281)
            if_condition_18845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 8), mo_18844)
            # Assigning a type to the variable 'if_condition_18845' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'if_condition_18845', if_condition_18845)
            # SSA begins for if statement (line 281)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 282):
            
            # Assigning a Call to a Name:
            
            # Call to group(...): (line 282)
            # Processing the call arguments (line 282)
            str_18848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 33), 'str', 'name')
            str_18849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 41), 'str', 'num')
            # Processing the call keyword arguments (line 282)
            kwargs_18850 = {}
            # Getting the type of 'mo' (line 282)
            mo_18846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'mo', False)
            # Obtaining the member 'group' of a type (line 282)
            group_18847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), mo_18846, 'group')
            # Calling group(args, kwargs) (line 282)
            group_call_result_18851 = invoke(stypy.reporting.localization.Localization(__file__, 282, 24), group_18847, *[str_18848, str_18849], **kwargs_18850)
            
            # Assigning a type to the variable 'call_assignment_18336' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'call_assignment_18336', group_call_result_18851)
            
            # Assigning a Call to a Name (line 282):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_18336' (line 282)
            call_assignment_18336_18852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'call_assignment_18336', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_18853 = stypy_get_value_from_tuple(call_assignment_18336_18852, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_18337' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'call_assignment_18337', stypy_get_value_from_tuple_call_result_18853)
            
            # Assigning a Name to a Name (line 282):
            # Getting the type of 'call_assignment_18337' (line 282)
            call_assignment_18337_18854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'call_assignment_18337')
            # Assigning a type to the variable 'name' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'name', call_assignment_18337_18854)
            
            # Assigning a Call to a Name (line 282):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_18336' (line 282)
            call_assignment_18336_18855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'call_assignment_18336', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_18856 = stypy_get_value_from_tuple(call_assignment_18336_18855, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_18338' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'call_assignment_18338', stypy_get_value_from_tuple_call_result_18856)
            
            # Assigning a Name to a Name (line 282):
            # Getting the type of 'call_assignment_18338' (line 282)
            call_assignment_18338_18857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'call_assignment_18338')
            # Assigning a type to the variable 'num' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'num', call_assignment_18338_18857)
            
            # Type idiom detected: calculating its left and rigth part (line 283)
            # Getting the type of 'num' (line 283)
            num_18858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'num')
            # Getting the type of 'None' (line 283)
            None_18859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 26), 'None')
            
            (may_be_18860, more_types_in_union_18861) = may_not_be_none(num_18858, None_18859)

            if may_be_18860:

                if more_types_in_union_18861:
                    # Runtime conditional SSA (line 283)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Name (line 284):
                
                # Assigning a Call to a Name (line 284):
                
                # Call to int(...): (line 284)
                # Processing the call arguments (line 284)
                # Getting the type of 'num' (line 284)
                num_18863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 26), 'num', False)
                # Processing the call keyword arguments (line 284)
                kwargs_18864 = {}
                # Getting the type of 'int' (line 284)
                int_18862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 22), 'int', False)
                # Calling int(args, kwargs) (line 284)
                int_call_result_18865 = invoke(stypy.reporting.localization.Localization(__file__, 284, 22), int_18862, *[num_18863], **kwargs_18864)
                
                # Assigning a type to the variable 'num' (line 284)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'num', int_call_result_18865)

                if more_types_in_union_18861:
                    # SSA join for if statement (line 283)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to append(...): (line 285)
            # Processing the call arguments (line 285)
            
            # Obtaining an instance of the builtin type 'tuple' (line 285)
            tuple_18873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 56), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 285)
            # Adding element type (line 285)
            # Getting the type of 'num' (line 285)
            num_18874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 56), 'num', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 56), tuple_18873, num_18874)
            # Adding element type (line 285)
            # Getting the type of 'value' (line 285)
            value_18875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 61), 'value', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 56), tuple_18873, value_18875)
            # Adding element type (line 285)
            # Getting the type of 'encoded' (line 285)
            encoded_18876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 68), 'encoded', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 56), tuple_18873, encoded_18876)
            
            # Processing the call keyword arguments (line 285)
            kwargs_18877 = {}
            
            # Call to setdefault(...): (line 285)
            # Processing the call arguments (line 285)
            # Getting the type of 'name' (line 285)
            name_18868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 38), 'name', False)
            
            # Obtaining an instance of the builtin type 'list' (line 285)
            list_18869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 44), 'list')
            # Adding type elements to the builtin type 'list' instance (line 285)
            
            # Processing the call keyword arguments (line 285)
            kwargs_18870 = {}
            # Getting the type of 'rfc2231_params' (line 285)
            rfc2231_params_18866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'rfc2231_params', False)
            # Obtaining the member 'setdefault' of a type (line 285)
            setdefault_18867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), rfc2231_params_18866, 'setdefault')
            # Calling setdefault(args, kwargs) (line 285)
            setdefault_call_result_18871 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), setdefault_18867, *[name_18868, list_18869], **kwargs_18870)
            
            # Obtaining the member 'append' of a type (line 285)
            append_18872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), setdefault_call_result_18871, 'append')
            # Calling append(args, kwargs) (line 285)
            append_call_result_18878 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), append_18872, *[tuple_18873], **kwargs_18877)
            
            # SSA branch for the else part of an if statement (line 281)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 287)
            # Processing the call arguments (line 287)
            
            # Obtaining an instance of the builtin type 'tuple' (line 287)
            tuple_18881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 287)
            # Adding element type (line 287)
            # Getting the type of 'name' (line 287)
            name_18882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'name', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 31), tuple_18881, name_18882)
            # Adding element type (line 287)
            str_18883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 37), 'str', '"%s"')
            
            # Call to quote(...): (line 287)
            # Processing the call arguments (line 287)
            # Getting the type of 'value' (line 287)
            value_18885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 52), 'value', False)
            # Processing the call keyword arguments (line 287)
            kwargs_18886 = {}
            # Getting the type of 'quote' (line 287)
            quote_18884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 46), 'quote', False)
            # Calling quote(args, kwargs) (line 287)
            quote_call_result_18887 = invoke(stypy.reporting.localization.Localization(__file__, 287, 46), quote_18884, *[value_18885], **kwargs_18886)
            
            # Applying the binary operator '%' (line 287)
            result_mod_18888 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 37), '%', str_18883, quote_call_result_18887)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 31), tuple_18881, result_mod_18888)
            
            # Processing the call keyword arguments (line 287)
            kwargs_18889 = {}
            # Getting the type of 'new_params' (line 287)
            new_params_18879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'new_params', False)
            # Obtaining the member 'append' of a type (line 287)
            append_18880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), new_params_18879, 'append')
            # Calling append(args, kwargs) (line 287)
            append_call_result_18890 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), append_18880, *[tuple_18881], **kwargs_18889)
            
            # SSA join for if statement (line 281)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for while statement (line 273)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'rfc2231_params' (line 288)
    rfc2231_params_18891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 7), 'rfc2231_params')
    # Testing if the type of an if condition is none (line 288)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 288, 4), rfc2231_params_18891):
        pass
    else:
        
        # Testing the type of an if condition (line 288)
        if_condition_18892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 4), rfc2231_params_18891)
        # Assigning a type to the variable 'if_condition_18892' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'if_condition_18892', if_condition_18892)
        # SSA begins for if statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to items(...): (line 289)
        # Processing the call keyword arguments (line 289)
        kwargs_18895 = {}
        # Getting the type of 'rfc2231_params' (line 289)
        rfc2231_params_18893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 35), 'rfc2231_params', False)
        # Obtaining the member 'items' of a type (line 289)
        items_18894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 35), rfc2231_params_18893, 'items')
        # Calling items(args, kwargs) (line 289)
        items_call_result_18896 = invoke(stypy.reporting.localization.Localization(__file__, 289, 35), items_18894, *[], **kwargs_18895)
        
        # Assigning a type to the variable 'items_call_result_18896' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'items_call_result_18896', items_call_result_18896)
        # Testing if the for loop is going to be iterated (line 289)
        # Testing the type of a for loop iterable (line 289)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 289, 8), items_call_result_18896)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 289, 8), items_call_result_18896):
            # Getting the type of the for loop variable (line 289)
            for_loop_var_18897 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 289, 8), items_call_result_18896)
            # Assigning a type to the variable 'name' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 8), for_loop_var_18897, 2, 0))
            # Assigning a type to the variable 'continuations' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'continuations', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 8), for_loop_var_18897, 2, 1))
            # SSA begins for a for statement (line 289)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a List to a Name (line 290):
            
            # Assigning a List to a Name (line 290):
            
            # Obtaining an instance of the builtin type 'list' (line 290)
            list_18898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 290)
            
            # Assigning a type to the variable 'value' (line 290)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'value', list_18898)
            
            # Assigning a Name to a Name (line 291):
            
            # Assigning a Name to a Name (line 291):
            # Getting the type of 'False' (line 291)
            False_18899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 23), 'False')
            # Assigning a type to the variable 'extended' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'extended', False_18899)
            
            # Call to sort(...): (line 293)
            # Processing the call keyword arguments (line 293)
            kwargs_18902 = {}
            # Getting the type of 'continuations' (line 293)
            continuations_18900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'continuations', False)
            # Obtaining the member 'sort' of a type (line 293)
            sort_18901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), continuations_18900, 'sort')
            # Calling sort(args, kwargs) (line 293)
            sort_call_result_18903 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), sort_18901, *[], **kwargs_18902)
            
            
            # Getting the type of 'continuations' (line 299)
            continuations_18904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 35), 'continuations')
            # Assigning a type to the variable 'continuations_18904' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'continuations_18904', continuations_18904)
            # Testing if the for loop is going to be iterated (line 299)
            # Testing the type of a for loop iterable (line 299)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 299, 12), continuations_18904)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 299, 12), continuations_18904):
                # Getting the type of the for loop variable (line 299)
                for_loop_var_18905 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 299, 12), continuations_18904)
                # Assigning a type to the variable 'num' (line 299)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'num', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 12), for_loop_var_18905, 3, 0))
                # Assigning a type to the variable 's' (line 299)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 's', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 12), for_loop_var_18905, 3, 1))
                # Assigning a type to the variable 'encoded' (line 299)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'encoded', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 12), for_loop_var_18905, 3, 2))
                # SSA begins for a for statement (line 299)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                # Getting the type of 'encoded' (line 300)
                encoded_18906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 19), 'encoded')
                # Testing if the type of an if condition is none (line 300)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 300, 16), encoded_18906):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 300)
                    if_condition_18907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 16), encoded_18906)
                    # Assigning a type to the variable 'if_condition_18907' (line 300)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'if_condition_18907', if_condition_18907)
                    # SSA begins for if statement (line 300)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 301):
                    
                    # Assigning a Call to a Name (line 301):
                    
                    # Call to unquote(...): (line 301)
                    # Processing the call arguments (line 301)
                    # Getting the type of 's' (line 301)
                    s_18910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 39), 's', False)
                    # Processing the call keyword arguments (line 301)
                    kwargs_18911 = {}
                    # Getting the type of 'urllib' (line 301)
                    urllib_18908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'urllib', False)
                    # Obtaining the member 'unquote' of a type (line 301)
                    unquote_18909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 24), urllib_18908, 'unquote')
                    # Calling unquote(args, kwargs) (line 301)
                    unquote_call_result_18912 = invoke(stypy.reporting.localization.Localization(__file__, 301, 24), unquote_18909, *[s_18910], **kwargs_18911)
                    
                    # Assigning a type to the variable 's' (line 301)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 20), 's', unquote_call_result_18912)
                    
                    # Assigning a Name to a Name (line 302):
                    
                    # Assigning a Name to a Name (line 302):
                    # Getting the type of 'True' (line 302)
                    True_18913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 31), 'True')
                    # Assigning a type to the variable 'extended' (line 302)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 20), 'extended', True_18913)
                    # SSA join for if statement (line 300)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to append(...): (line 303)
                # Processing the call arguments (line 303)
                # Getting the type of 's' (line 303)
                s_18916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 29), 's', False)
                # Processing the call keyword arguments (line 303)
                kwargs_18917 = {}
                # Getting the type of 'value' (line 303)
                value_18914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'value', False)
                # Obtaining the member 'append' of a type (line 303)
                append_18915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), value_18914, 'append')
                # Calling append(args, kwargs) (line 303)
                append_call_result_18918 = invoke(stypy.reporting.localization.Localization(__file__, 303, 16), append_18915, *[s_18916], **kwargs_18917)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Name (line 304):
            
            # Assigning a Call to a Name (line 304):
            
            # Call to quote(...): (line 304)
            # Processing the call arguments (line 304)
            
            # Call to join(...): (line 304)
            # Processing the call arguments (line 304)
            # Getting the type of 'value' (line 304)
            value_18922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'value', False)
            # Processing the call keyword arguments (line 304)
            kwargs_18923 = {}
            # Getting the type of 'EMPTYSTRING' (line 304)
            EMPTYSTRING_18920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 26), 'EMPTYSTRING', False)
            # Obtaining the member 'join' of a type (line 304)
            join_18921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 26), EMPTYSTRING_18920, 'join')
            # Calling join(args, kwargs) (line 304)
            join_call_result_18924 = invoke(stypy.reporting.localization.Localization(__file__, 304, 26), join_18921, *[value_18922], **kwargs_18923)
            
            # Processing the call keyword arguments (line 304)
            kwargs_18925 = {}
            # Getting the type of 'quote' (line 304)
            quote_18919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'quote', False)
            # Calling quote(args, kwargs) (line 304)
            quote_call_result_18926 = invoke(stypy.reporting.localization.Localization(__file__, 304, 20), quote_18919, *[join_call_result_18924], **kwargs_18925)
            
            # Assigning a type to the variable 'value' (line 304)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'value', quote_call_result_18926)
            # Getting the type of 'extended' (line 305)
            extended_18927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'extended')
            # Testing if the type of an if condition is none (line 305)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 305, 12), extended_18927):
                
                # Call to append(...): (line 309)
                # Processing the call arguments (line 309)
                
                # Obtaining an instance of the builtin type 'tuple' (line 309)
                tuple_18956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 35), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 309)
                # Adding element type (line 309)
                # Getting the type of 'name' (line 309)
                name_18957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 35), 'name', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 35), tuple_18956, name_18957)
                # Adding element type (line 309)
                str_18958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 41), 'str', '"%s"')
                # Getting the type of 'value' (line 309)
                value_18959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 50), 'value', False)
                # Applying the binary operator '%' (line 309)
                result_mod_18960 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 41), '%', str_18958, value_18959)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 35), tuple_18956, result_mod_18960)
                
                # Processing the call keyword arguments (line 309)
                kwargs_18961 = {}
                # Getting the type of 'new_params' (line 309)
                new_params_18954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'new_params', False)
                # Obtaining the member 'append' of a type (line 309)
                append_18955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 16), new_params_18954, 'append')
                # Calling append(args, kwargs) (line 309)
                append_call_result_18962 = invoke(stypy.reporting.localization.Localization(__file__, 309, 16), append_18955, *[tuple_18956], **kwargs_18961)
                
            else:
                
                # Testing the type of an if condition (line 305)
                if_condition_18928 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 12), extended_18927)
                # Assigning a type to the variable 'if_condition_18928' (line 305)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'if_condition_18928', if_condition_18928)
                # SSA begins for if statement (line 305)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 306):
                
                # Assigning a Call to a Name:
                
                # Call to decode_rfc2231(...): (line 306)
                # Processing the call arguments (line 306)
                # Getting the type of 'value' (line 306)
                value_18930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 58), 'value', False)
                # Processing the call keyword arguments (line 306)
                kwargs_18931 = {}
                # Getting the type of 'decode_rfc2231' (line 306)
                decode_rfc2231_18929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 43), 'decode_rfc2231', False)
                # Calling decode_rfc2231(args, kwargs) (line 306)
                decode_rfc2231_call_result_18932 = invoke(stypy.reporting.localization.Localization(__file__, 306, 43), decode_rfc2231_18929, *[value_18930], **kwargs_18931)
                
                # Assigning a type to the variable 'call_assignment_18339' (line 306)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'call_assignment_18339', decode_rfc2231_call_result_18932)
                
                # Assigning a Call to a Name (line 306):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_18339' (line 306)
                call_assignment_18339_18933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'call_assignment_18339', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18934 = stypy_get_value_from_tuple(call_assignment_18339_18933, 3, 0)
                
                # Assigning a type to the variable 'call_assignment_18340' (line 306)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'call_assignment_18340', stypy_get_value_from_tuple_call_result_18934)
                
                # Assigning a Name to a Name (line 306):
                # Getting the type of 'call_assignment_18340' (line 306)
                call_assignment_18340_18935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'call_assignment_18340')
                # Assigning a type to the variable 'charset' (line 306)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'charset', call_assignment_18340_18935)
                
                # Assigning a Call to a Name (line 306):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_18339' (line 306)
                call_assignment_18339_18936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'call_assignment_18339', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18937 = stypy_get_value_from_tuple(call_assignment_18339_18936, 3, 1)
                
                # Assigning a type to the variable 'call_assignment_18341' (line 306)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'call_assignment_18341', stypy_get_value_from_tuple_call_result_18937)
                
                # Assigning a Name to a Name (line 306):
                # Getting the type of 'call_assignment_18341' (line 306)
                call_assignment_18341_18938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'call_assignment_18341')
                # Assigning a type to the variable 'language' (line 306)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 25), 'language', call_assignment_18341_18938)
                
                # Assigning a Call to a Name (line 306):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_18339' (line 306)
                call_assignment_18339_18939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'call_assignment_18339', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18940 = stypy_get_value_from_tuple(call_assignment_18339_18939, 3, 2)
                
                # Assigning a type to the variable 'call_assignment_18342' (line 306)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'call_assignment_18342', stypy_get_value_from_tuple_call_result_18940)
                
                # Assigning a Name to a Name (line 306):
                # Getting the type of 'call_assignment_18342' (line 306)
                call_assignment_18342_18941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'call_assignment_18342')
                # Assigning a type to the variable 'value' (line 306)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 35), 'value', call_assignment_18342_18941)
                
                # Call to append(...): (line 307)
                # Processing the call arguments (line 307)
                
                # Obtaining an instance of the builtin type 'tuple' (line 307)
                tuple_18944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 35), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 307)
                # Adding element type (line 307)
                # Getting the type of 'name' (line 307)
                name_18945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 35), 'name', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 35), tuple_18944, name_18945)
                # Adding element type (line 307)
                
                # Obtaining an instance of the builtin type 'tuple' (line 307)
                tuple_18946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 42), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 307)
                # Adding element type (line 307)
                # Getting the type of 'charset' (line 307)
                charset_18947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 42), 'charset', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 42), tuple_18946, charset_18947)
                # Adding element type (line 307)
                # Getting the type of 'language' (line 307)
                language_18948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 51), 'language', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 42), tuple_18946, language_18948)
                # Adding element type (line 307)
                str_18949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 61), 'str', '"%s"')
                # Getting the type of 'value' (line 307)
                value_18950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 70), 'value', False)
                # Applying the binary operator '%' (line 307)
                result_mod_18951 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 61), '%', str_18949, value_18950)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 42), tuple_18946, result_mod_18951)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 35), tuple_18944, tuple_18946)
                
                # Processing the call keyword arguments (line 307)
                kwargs_18952 = {}
                # Getting the type of 'new_params' (line 307)
                new_params_18942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'new_params', False)
                # Obtaining the member 'append' of a type (line 307)
                append_18943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 16), new_params_18942, 'append')
                # Calling append(args, kwargs) (line 307)
                append_call_result_18953 = invoke(stypy.reporting.localization.Localization(__file__, 307, 16), append_18943, *[tuple_18944], **kwargs_18952)
                
                # SSA branch for the else part of an if statement (line 305)
                module_type_store.open_ssa_branch('else')
                
                # Call to append(...): (line 309)
                # Processing the call arguments (line 309)
                
                # Obtaining an instance of the builtin type 'tuple' (line 309)
                tuple_18956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 35), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 309)
                # Adding element type (line 309)
                # Getting the type of 'name' (line 309)
                name_18957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 35), 'name', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 35), tuple_18956, name_18957)
                # Adding element type (line 309)
                str_18958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 41), 'str', '"%s"')
                # Getting the type of 'value' (line 309)
                value_18959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 50), 'value', False)
                # Applying the binary operator '%' (line 309)
                result_mod_18960 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 41), '%', str_18958, value_18959)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 35), tuple_18956, result_mod_18960)
                
                # Processing the call keyword arguments (line 309)
                kwargs_18961 = {}
                # Getting the type of 'new_params' (line 309)
                new_params_18954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'new_params', False)
                # Obtaining the member 'append' of a type (line 309)
                append_18955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 16), new_params_18954, 'append')
                # Calling append(args, kwargs) (line 309)
                append_call_result_18962 = invoke(stypy.reporting.localization.Localization(__file__, 309, 16), append_18955, *[tuple_18956], **kwargs_18961)
                
                # SSA join for if statement (line 305)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 288)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'new_params' (line 310)
    new_params_18963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'new_params')
    # Assigning a type to the variable 'stypy_return_type' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type', new_params_18963)
    
    # ################# End of 'decode_params(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode_params' in the type store
    # Getting the type of 'stypy_return_type' (line 259)
    stypy_return_type_18964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18964)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode_params'
    return stypy_return_type_18964

# Assigning a type to the variable 'decode_params' (line 259)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'decode_params', decode_params)

@norecursion
def collapse_rfc2231_value(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_18965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 41), 'str', 'replace')
    str_18966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 44), 'str', 'us-ascii')
    defaults = [str_18965, str_18966]
    # Create a new context for function 'collapse_rfc2231_value'
    module_type_store = module_type_store.open_function_context('collapse_rfc2231_value', 312, 0, False)
    
    # Passed parameters checking function
    collapse_rfc2231_value.stypy_localization = localization
    collapse_rfc2231_value.stypy_type_of_self = None
    collapse_rfc2231_value.stypy_type_store = module_type_store
    collapse_rfc2231_value.stypy_function_name = 'collapse_rfc2231_value'
    collapse_rfc2231_value.stypy_param_names_list = ['value', 'errors', 'fallback_charset']
    collapse_rfc2231_value.stypy_varargs_param_name = None
    collapse_rfc2231_value.stypy_kwargs_param_name = None
    collapse_rfc2231_value.stypy_call_defaults = defaults
    collapse_rfc2231_value.stypy_call_varargs = varargs
    collapse_rfc2231_value.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'collapse_rfc2231_value', ['value', 'errors', 'fallback_charset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'collapse_rfc2231_value', localization, ['value', 'errors', 'fallback_charset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'collapse_rfc2231_value(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 314)
    # Getting the type of 'tuple' (line 314)
    tuple_18967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 25), 'tuple')
    # Getting the type of 'value' (line 314)
    value_18968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 18), 'value')
    
    (may_be_18969, more_types_in_union_18970) = may_be_subtype(tuple_18967, value_18968)

    if may_be_18969:

        if more_types_in_union_18970:
            # Runtime conditional SSA (line 314)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'value' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'value', remove_not_subtype_from_union(value_18968, tuple))
        
        # Assigning a Call to a Name (line 315):
        
        # Assigning a Call to a Name (line 315):
        
        # Call to unquote(...): (line 315)
        # Processing the call arguments (line 315)
        
        # Obtaining the type of the subscript
        int_18972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 31), 'int')
        # Getting the type of 'value' (line 315)
        value_18973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 25), 'value', False)
        # Obtaining the member '__getitem__' of a type (line 315)
        getitem___18974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 25), value_18973, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 315)
        subscript_call_result_18975 = invoke(stypy.reporting.localization.Localization(__file__, 315, 25), getitem___18974, int_18972)
        
        # Processing the call keyword arguments (line 315)
        kwargs_18976 = {}
        # Getting the type of 'unquote' (line 315)
        unquote_18971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 17), 'unquote', False)
        # Calling unquote(args, kwargs) (line 315)
        unquote_call_result_18977 = invoke(stypy.reporting.localization.Localization(__file__, 315, 17), unquote_18971, *[subscript_call_result_18975], **kwargs_18976)
        
        # Assigning a type to the variable 'rawval' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'rawval', unquote_call_result_18977)
        
        # Assigning a BoolOp to a Name (line 316):
        
        # Assigning a BoolOp to a Name (line 316):
        
        # Evaluating a boolean operation
        
        # Obtaining the type of the subscript
        int_18978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 24), 'int')
        # Getting the type of 'value' (line 316)
        value_18979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 18), 'value')
        # Obtaining the member '__getitem__' of a type (line 316)
        getitem___18980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 18), value_18979, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 316)
        subscript_call_result_18981 = invoke(stypy.reporting.localization.Localization(__file__, 316, 18), getitem___18980, int_18978)
        
        str_18982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 30), 'str', 'us-ascii')
        # Applying the binary operator 'or' (line 316)
        result_or_keyword_18983 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 18), 'or', subscript_call_result_18981, str_18982)
        
        # Assigning a type to the variable 'charset' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'charset', result_or_keyword_18983)
        
        
        # SSA begins for try-except statement (line 317)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to unicode(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'rawval' (line 318)
        rawval_18985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'rawval', False)
        # Getting the type of 'charset' (line 318)
        charset_18986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 35), 'charset', False)
        # Getting the type of 'errors' (line 318)
        errors_18987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 44), 'errors', False)
        # Processing the call keyword arguments (line 318)
        kwargs_18988 = {}
        # Getting the type of 'unicode' (line 318)
        unicode_18984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 19), 'unicode', False)
        # Calling unicode(args, kwargs) (line 318)
        unicode_call_result_18989 = invoke(stypy.reporting.localization.Localization(__file__, 318, 19), unicode_18984, *[rawval_18985, charset_18986, errors_18987], **kwargs_18988)
        
        # Assigning a type to the variable 'stypy_return_type' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'stypy_return_type', unicode_call_result_18989)
        # SSA branch for the except part of a try statement (line 317)
        # SSA branch for the except 'LookupError' branch of a try statement (line 317)
        module_type_store.open_ssa_branch('except')
        
        # Call to unicode(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'rawval' (line 321)
        rawval_18991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 27), 'rawval', False)
        # Getting the type of 'fallback_charset' (line 321)
        fallback_charset_18992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 35), 'fallback_charset', False)
        # Getting the type of 'errors' (line 321)
        errors_18993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 53), 'errors', False)
        # Processing the call keyword arguments (line 321)
        kwargs_18994 = {}
        # Getting the type of 'unicode' (line 321)
        unicode_18990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'unicode', False)
        # Calling unicode(args, kwargs) (line 321)
        unicode_call_result_18995 = invoke(stypy.reporting.localization.Localization(__file__, 321, 19), unicode_18990, *[rawval_18991, fallback_charset_18992, errors_18993], **kwargs_18994)
        
        # Assigning a type to the variable 'stypy_return_type' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'stypy_return_type', unicode_call_result_18995)
        # SSA join for try-except statement (line 317)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_18970:
            # Runtime conditional SSA for else branch (line 314)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_18969) or more_types_in_union_18970):
        # Assigning a type to the variable 'value' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'value', remove_subtype_from_union(value_18968, tuple))
        
        # Call to unquote(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'value' (line 323)
        value_18997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 23), 'value', False)
        # Processing the call keyword arguments (line 323)
        kwargs_18998 = {}
        # Getting the type of 'unquote' (line 323)
        unquote_18996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'unquote', False)
        # Calling unquote(args, kwargs) (line 323)
        unquote_call_result_18999 = invoke(stypy.reporting.localization.Localization(__file__, 323, 15), unquote_18996, *[value_18997], **kwargs_18998)
        
        # Assigning a type to the variable 'stypy_return_type' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'stypy_return_type', unquote_call_result_18999)

        if (may_be_18969 and more_types_in_union_18970):
            # SSA join for if statement (line 314)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'collapse_rfc2231_value(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'collapse_rfc2231_value' in the type store
    # Getting the type of 'stypy_return_type' (line 312)
    stypy_return_type_19000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19000)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'collapse_rfc2231_value'
    return stypy_return_type_19000

# Assigning a type to the variable 'collapse_rfc2231_value' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'collapse_rfc2231_value', collapse_rfc2231_value)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
