
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Ben Gertzfield, Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: __all__ = [
6:     'Charset',
7:     'add_alias',
8:     'add_charset',
9:     'add_codec',
10:     ]
11: 
12: import codecs
13: import email.base64mime
14: import email.quoprimime
15: 
16: from email import errors
17: from email.encoders import encode_7or8bit
18: 
19: 
20: 
21: # Flags for types of header encodings
22: QP          = 1 # Quoted-Printable
23: BASE64      = 2 # Base64
24: SHORTEST    = 3 # the shorter of QP and base64, but only for headers
25: 
26: # In "=?charset?q?hello_world?=", the =?, ?q?, and ?= add up to 7
27: MISC_LEN = 7
28: 
29: DEFAULT_CHARSET = 'us-ascii'
30: 
31: 
32: 
33: # Defaults
34: CHARSETS = {
35:     # input        header enc  body enc output conv
36:     'iso-8859-1':  (QP,        QP,      None),
37:     'iso-8859-2':  (QP,        QP,      None),
38:     'iso-8859-3':  (QP,        QP,      None),
39:     'iso-8859-4':  (QP,        QP,      None),
40:     # iso-8859-5 is Cyrillic, and not especially used
41:     # iso-8859-6 is Arabic, also not particularly used
42:     # iso-8859-7 is Greek, QP will not make it readable
43:     # iso-8859-8 is Hebrew, QP will not make it readable
44:     'iso-8859-9':  (QP,        QP,      None),
45:     'iso-8859-10': (QP,        QP,      None),
46:     # iso-8859-11 is Thai, QP will not make it readable
47:     'iso-8859-13': (QP,        QP,      None),
48:     'iso-8859-14': (QP,        QP,      None),
49:     'iso-8859-15': (QP,        QP,      None),
50:     'iso-8859-16': (QP,        QP,      None),
51:     'windows-1252':(QP,        QP,      None),
52:     'viscii':      (QP,        QP,      None),
53:     'us-ascii':    (None,      None,    None),
54:     'big5':        (BASE64,    BASE64,  None),
55:     'gb2312':      (BASE64,    BASE64,  None),
56:     'euc-jp':      (BASE64,    None,    'iso-2022-jp'),
57:     'shift_jis':   (BASE64,    None,    'iso-2022-jp'),
58:     'iso-2022-jp': (BASE64,    None,    None),
59:     'koi8-r':      (BASE64,    BASE64,  None),
60:     'utf-8':       (SHORTEST,  BASE64, 'utf-8'),
61:     # We're making this one up to represent raw unencoded 8-bit
62:     '8bit':        (None,      BASE64, 'utf-8'),
63:     }
64: 
65: # Aliases for other commonly-used names for character sets.  Map
66: # them to the real ones used in email.
67: ALIASES = {
68:     'latin_1': 'iso-8859-1',
69:     'latin-1': 'iso-8859-1',
70:     'latin_2': 'iso-8859-2',
71:     'latin-2': 'iso-8859-2',
72:     'latin_3': 'iso-8859-3',
73:     'latin-3': 'iso-8859-3',
74:     'latin_4': 'iso-8859-4',
75:     'latin-4': 'iso-8859-4',
76:     'latin_5': 'iso-8859-9',
77:     'latin-5': 'iso-8859-9',
78:     'latin_6': 'iso-8859-10',
79:     'latin-6': 'iso-8859-10',
80:     'latin_7': 'iso-8859-13',
81:     'latin-7': 'iso-8859-13',
82:     'latin_8': 'iso-8859-14',
83:     'latin-8': 'iso-8859-14',
84:     'latin_9': 'iso-8859-15',
85:     'latin-9': 'iso-8859-15',
86:     'latin_10':'iso-8859-16',
87:     'latin-10':'iso-8859-16',
88:     'cp949':   'ks_c_5601-1987',
89:     'euc_jp':  'euc-jp',
90:     'euc_kr':  'euc-kr',
91:     'ascii':   'us-ascii',
92:     }
93: 
94: 
95: # Map charsets to their Unicode codec strings.
96: CODEC_MAP = {
97:     'gb2312':      'eucgb2312_cn',
98:     'big5':        'big5_tw',
99:     # Hack: We don't want *any* conversion for stuff marked us-ascii, as all
100:     # sorts of garbage might be sent to us in the guise of 7-bit us-ascii.
101:     # Let that stuff pass through without conversion to/from Unicode.
102:     'us-ascii':    None,
103:     }
104: 
105: 
106: 
107: # Convenience functions for extending the above mappings
108: def add_charset(charset, header_enc=None, body_enc=None, output_charset=None):
109:     '''Add character set properties to the global registry.
110: 
111:     charset is the input character set, and must be the canonical name of a
112:     character set.
113: 
114:     Optional header_enc and body_enc is either Charset.QP for
115:     quoted-printable, Charset.BASE64 for base64 encoding, Charset.SHORTEST for
116:     the shortest of qp or base64 encoding, or None for no encoding.  SHORTEST
117:     is only valid for header_enc.  It describes how message headers and
118:     message bodies in the input charset are to be encoded.  Default is no
119:     encoding.
120: 
121:     Optional output_charset is the character set that the output should be
122:     in.  Conversions will proceed from input charset, to Unicode, to the
123:     output charset when the method Charset.convert() is called.  The default
124:     is to output in the same character set as the input.
125: 
126:     Both input_charset and output_charset must have Unicode codec entries in
127:     the module's charset-to-codec mapping; use add_codec(charset, codecname)
128:     to add codecs the module does not know about.  See the codecs module's
129:     documentation for more information.
130:     '''
131:     if body_enc == SHORTEST:
132:         raise ValueError('SHORTEST not allowed for body_enc')
133:     CHARSETS[charset] = (header_enc, body_enc, output_charset)
134: 
135: 
136: def add_alias(alias, canonical):
137:     '''Add a character set alias.
138: 
139:     alias is the alias name, e.g. latin-1
140:     canonical is the character set's canonical name, e.g. iso-8859-1
141:     '''
142:     ALIASES[alias] = canonical
143: 
144: 
145: def add_codec(charset, codecname):
146:     '''Add a codec that map characters in the given charset to/from Unicode.
147: 
148:     charset is the canonical name of a character set.  codecname is the name
149:     of a Python codec, as appropriate for the second argument to the unicode()
150:     built-in, or to the encode() method of a Unicode string.
151:     '''
152:     CODEC_MAP[charset] = codecname
153: 
154: 
155: 
156: class Charset:
157:     '''Map character sets to their email properties.
158: 
159:     This class provides information about the requirements imposed on email
160:     for a specific character set.  It also provides convenience routines for
161:     converting between character sets, given the availability of the
162:     applicable codecs.  Given a character set, it will do its best to provide
163:     information on how to use that character set in an email in an
164:     RFC-compliant way.
165: 
166:     Certain character sets must be encoded with quoted-printable or base64
167:     when used in email headers or bodies.  Certain character sets must be
168:     converted outright, and are not allowed in email.  Instances of this
169:     module expose the following information about a character set:
170: 
171:     input_charset: The initial character set specified.  Common aliases
172:                    are converted to their `official' email names (e.g. latin_1
173:                    is converted to iso-8859-1).  Defaults to 7-bit us-ascii.
174: 
175:     header_encoding: If the character set must be encoded before it can be
176:                      used in an email header, this attribute will be set to
177:                      Charset.QP (for quoted-printable), Charset.BASE64 (for
178:                      base64 encoding), or Charset.SHORTEST for the shortest of
179:                      QP or BASE64 encoding.  Otherwise, it will be None.
180: 
181:     body_encoding: Same as header_encoding, but describes the encoding for the
182:                    mail message's body, which indeed may be different than the
183:                    header encoding.  Charset.SHORTEST is not allowed for
184:                    body_encoding.
185: 
186:     output_charset: Some character sets must be converted before they can be
187:                     used in email headers or bodies.  If the input_charset is
188:                     one of them, this attribute will contain the name of the
189:                     charset output will be converted to.  Otherwise, it will
190:                     be None.
191: 
192:     input_codec: The name of the Python codec used to convert the
193:                  input_charset to Unicode.  If no conversion codec is
194:                  necessary, this attribute will be None.
195: 
196:     output_codec: The name of the Python codec used to convert Unicode
197:                   to the output_charset.  If no conversion codec is necessary,
198:                   this attribute will have the same value as the input_codec.
199:     '''
200:     def __init__(self, input_charset=DEFAULT_CHARSET):
201:         # RFC 2046, $4.1.2 says charsets are not case sensitive.  We coerce to
202:         # unicode because its .lower() is locale insensitive.  If the argument
203:         # is already a unicode, we leave it at that, but ensure that the
204:         # charset is ASCII, as the standard (RFC XXX) requires.
205:         try:
206:             if isinstance(input_charset, unicode):
207:                 input_charset.encode('ascii')
208:             else:
209:                 input_charset = unicode(input_charset, 'ascii')
210:         except UnicodeError:
211:             raise errors.CharsetError(input_charset)
212:         input_charset = input_charset.lower().encode('ascii')
213:         # Set the input charset after filtering through the aliases and/or codecs
214:         if not (input_charset in ALIASES or input_charset in CHARSETS):
215:             try:
216:                 input_charset = codecs.lookup(input_charset).name
217:             except LookupError:
218:                 pass
219:         self.input_charset = ALIASES.get(input_charset, input_charset)
220:         # We can try to guess which encoding and conversion to use by the
221:         # charset_map dictionary.  Try that first, but let the user override
222:         # it.
223:         henc, benc, conv = CHARSETS.get(self.input_charset,
224:                                         (SHORTEST, BASE64, None))
225:         if not conv:
226:             conv = self.input_charset
227:         # Set the attributes, allowing the arguments to override the default.
228:         self.header_encoding = henc
229:         self.body_encoding = benc
230:         self.output_charset = ALIASES.get(conv, conv)
231:         # Now set the codecs.  If one isn't defined for input_charset,
232:         # guess and try a Unicode codec with the same name as input_codec.
233:         self.input_codec = CODEC_MAP.get(self.input_charset,
234:                                          self.input_charset)
235:         self.output_codec = CODEC_MAP.get(self.output_charset,
236:                                           self.output_charset)
237: 
238:     def __str__(self):
239:         return self.input_charset.lower()
240: 
241:     __repr__ = __str__
242: 
243:     def __eq__(self, other):
244:         return str(self) == str(other).lower()
245: 
246:     def __ne__(self, other):
247:         return not self.__eq__(other)
248: 
249:     def get_body_encoding(self):
250:         '''Return the content-transfer-encoding used for body encoding.
251: 
252:         This is either the string `quoted-printable' or `base64' depending on
253:         the encoding used, or it is a function in which case you should call
254:         the function with a single argument, the Message object being
255:         encoded.  The function should then set the Content-Transfer-Encoding
256:         header itself to whatever is appropriate.
257: 
258:         Returns "quoted-printable" if self.body_encoding is QP.
259:         Returns "base64" if self.body_encoding is BASE64.
260:         Returns "7bit" otherwise.
261:         '''
262:         assert self.body_encoding != SHORTEST
263:         if self.body_encoding == QP:
264:             return 'quoted-printable'
265:         elif self.body_encoding == BASE64:
266:             return 'base64'
267:         else:
268:             return encode_7or8bit
269: 
270:     def convert(self, s):
271:         '''Convert a string from the input_codec to the output_codec.'''
272:         if self.input_codec != self.output_codec:
273:             return unicode(s, self.input_codec).encode(self.output_codec)
274:         else:
275:             return s
276: 
277:     def to_splittable(self, s):
278:         '''Convert a possibly multibyte string to a safely splittable format.
279: 
280:         Uses the input_codec to try and convert the string to Unicode, so it
281:         can be safely split on character boundaries (even for multibyte
282:         characters).
283: 
284:         Returns the string as-is if it isn't known how to convert it to
285:         Unicode with the input_charset.
286: 
287:         Characters that could not be converted to Unicode will be replaced
288:         with the Unicode replacement character U+FFFD.
289:         '''
290:         if isinstance(s, unicode) or self.input_codec is None:
291:             return s
292:         try:
293:             return unicode(s, self.input_codec, 'replace')
294:         except LookupError:
295:             # Input codec not installed on system, so return the original
296:             # string unchanged.
297:             return s
298: 
299:     def from_splittable(self, ustr, to_output=True):
300:         '''Convert a splittable string back into an encoded string.
301: 
302:         Uses the proper codec to try and convert the string from Unicode back
303:         into an encoded format.  Return the string as-is if it is not Unicode,
304:         or if it could not be converted from Unicode.
305: 
306:         Characters that could not be converted from Unicode will be replaced
307:         with an appropriate character (usually '?').
308: 
309:         If to_output is True (the default), uses output_codec to convert to an
310:         encoded format.  If to_output is False, uses input_codec.
311:         '''
312:         if to_output:
313:             codec = self.output_codec
314:         else:
315:             codec = self.input_codec
316:         if not isinstance(ustr, unicode) or codec is None:
317:             return ustr
318:         try:
319:             return ustr.encode(codec, 'replace')
320:         except LookupError:
321:             # Output codec not installed
322:             return ustr
323: 
324:     def get_output_charset(self):
325:         '''Return the output character set.
326: 
327:         This is self.output_charset if that is not None, otherwise it is
328:         self.input_charset.
329:         '''
330:         return self.output_charset or self.input_charset
331: 
332:     def encoded_header_len(self, s):
333:         '''Return the length of the encoded header string.'''
334:         cset = self.get_output_charset()
335:         # The len(s) of a 7bit encoding is len(s)
336:         if self.header_encoding == BASE64:
337:             return email.base64mime.base64_len(s) + len(cset) + MISC_LEN
338:         elif self.header_encoding == QP:
339:             return email.quoprimime.header_quopri_len(s) + len(cset) + MISC_LEN
340:         elif self.header_encoding == SHORTEST:
341:             lenb64 = email.base64mime.base64_len(s)
342:             lenqp = email.quoprimime.header_quopri_len(s)
343:             return min(lenb64, lenqp) + len(cset) + MISC_LEN
344:         else:
345:             return len(s)
346: 
347:     def header_encode(self, s, convert=False):
348:         '''Header-encode a string, optionally converting it to output_charset.
349: 
350:         If convert is True, the string will be converted from the input
351:         charset to the output charset automatically.  This is not useful for
352:         multibyte character sets, which have line length issues (multibyte
353:         characters must be split on a character, not a byte boundary); use the
354:         high-level Header class to deal with these issues.  convert defaults
355:         to False.
356: 
357:         The type of encoding (base64 or quoted-printable) will be based on
358:         self.header_encoding.
359:         '''
360:         cset = self.get_output_charset()
361:         if convert:
362:             s = self.convert(s)
363:         # 7bit/8bit encodings return the string unchanged (modulo conversions)
364:         if self.header_encoding == BASE64:
365:             return email.base64mime.header_encode(s, cset)
366:         elif self.header_encoding == QP:
367:             return email.quoprimime.header_encode(s, cset, maxlinelen=None)
368:         elif self.header_encoding == SHORTEST:
369:             lenb64 = email.base64mime.base64_len(s)
370:             lenqp = email.quoprimime.header_quopri_len(s)
371:             if lenb64 < lenqp:
372:                 return email.base64mime.header_encode(s, cset)
373:             else:
374:                 return email.quoprimime.header_encode(s, cset, maxlinelen=None)
375:         else:
376:             return s
377: 
378:     def body_encode(self, s, convert=True):
379:         '''Body-encode a string and convert it to output_charset.
380: 
381:         If convert is True (the default), the string will be converted from
382:         the input charset to output charset automatically.  Unlike
383:         header_encode(), there are no issues with byte boundaries and
384:         multibyte charsets in email bodies, so this is usually pretty safe.
385: 
386:         The type of encoding (base64 or quoted-printable) will be based on
387:         self.body_encoding.
388:         '''
389:         if convert:
390:             s = self.convert(s)
391:         # 7bit/8bit encodings return the string unchanged (module conversions)
392:         if self.body_encoding is BASE64:
393:             return email.base64mime.body_encode(s)
394:         elif self.body_encoding is QP:
395:             return email.quoprimime.body_encode(s)
396:         else:
397:             return s
398: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 5):

# Assigning a List to a Name (line 5):
__all__ = ['Charset', 'add_alias', 'add_charset', 'add_codec']
module_type_store.set_exportable_members(['Charset', 'add_alias', 'add_charset', 'add_codec'])

# Obtaining an instance of the builtin type 'list' (line 5)
list_12178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
str_12179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'str', 'Charset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 10), list_12178, str_12179)
# Adding element type (line 5)
str_12180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 4), 'str', 'add_alias')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 10), list_12178, str_12180)
# Adding element type (line 5)
str_12181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'str', 'add_charset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 10), list_12178, str_12181)
# Adding element type (line 5)
str_12182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'str', 'add_codec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 10), list_12178, str_12182)

# Assigning a type to the variable '__all__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__all__', list_12178)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import codecs' statement (line 12)
import codecs

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'codecs', codecs, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import email.base64mime' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_12183 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.base64mime')

if (type(import_12183) is not StypyTypeError):

    if (import_12183 != 'pyd_module'):
        __import__(import_12183)
        sys_modules_12184 = sys.modules[import_12183]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.base64mime', sys_modules_12184.module_type_store, module_type_store)
    else:
        import email.base64mime

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.base64mime', email.base64mime, module_type_store)

else:
    # Assigning a type to the variable 'email.base64mime' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.base64mime', import_12183)

remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import email.quoprimime' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_12185 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'email.quoprimime')

if (type(import_12185) is not StypyTypeError):

    if (import_12185 != 'pyd_module'):
        __import__(import_12185)
        sys_modules_12186 = sys.modules[import_12185]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'email.quoprimime', sys_modules_12186.module_type_store, module_type_store)
    else:
        import email.quoprimime

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'email.quoprimime', email.quoprimime, module_type_store)

else:
    # Assigning a type to the variable 'email.quoprimime' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'email.quoprimime', import_12185)

remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from email import errors' statement (line 16)
try:
    from email import errors

except:
    errors = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'email', None, module_type_store, ['errors'], [errors])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from email.encoders import encode_7or8bit' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_12187 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'email.encoders')

if (type(import_12187) is not StypyTypeError):

    if (import_12187 != 'pyd_module'):
        __import__(import_12187)
        sys_modules_12188 = sys.modules[import_12187]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'email.encoders', sys_modules_12188.module_type_store, module_type_store, ['encode_7or8bit'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_12188, sys_modules_12188.module_type_store, module_type_store)
    else:
        from email.encoders import encode_7or8bit

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'email.encoders', None, module_type_store, ['encode_7or8bit'], [encode_7or8bit])

else:
    # Assigning a type to the variable 'email.encoders' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'email.encoders', import_12187)

remove_current_file_folder_from_path('C:/Python27/lib/email/')


# Assigning a Num to a Name (line 22):

# Assigning a Num to a Name (line 22):
int_12189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'int')
# Assigning a type to the variable 'QP' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'QP', int_12189)

# Assigning a Num to a Name (line 23):

# Assigning a Num to a Name (line 23):
int_12190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'int')
# Assigning a type to the variable 'BASE64' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'BASE64', int_12190)

# Assigning a Num to a Name (line 24):

# Assigning a Num to a Name (line 24):
int_12191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'int')
# Assigning a type to the variable 'SHORTEST' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'SHORTEST', int_12191)

# Assigning a Num to a Name (line 27):

# Assigning a Num to a Name (line 27):
int_12192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'int')
# Assigning a type to the variable 'MISC_LEN' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'MISC_LEN', int_12192)

# Assigning a Str to a Name (line 29):

# Assigning a Str to a Name (line 29):
str_12193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'str', 'us-ascii')
# Assigning a type to the variable 'DEFAULT_CHARSET' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'DEFAULT_CHARSET', str_12193)

# Assigning a Dict to a Name (line 34):

# Assigning a Dict to a Name (line 34):

# Obtaining an instance of the builtin type 'dict' (line 34)
dict_12194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 34)
# Adding element type (key, value) (line 34)
str_12195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'iso-8859-1')

# Obtaining an instance of the builtin type 'tuple' (line 36)
tuple_12196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 36)
# Adding element type (line 36)
# Getting the type of 'QP' (line 36)
QP_12197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 20), tuple_12196, QP_12197)
# Adding element type (line 36)
# Getting the type of 'QP' (line 36)
QP_12198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 20), tuple_12196, QP_12198)
# Adding element type (line 36)
# Getting the type of 'None' (line 36)
None_12199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 20), tuple_12196, None_12199)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12195, tuple_12196))
# Adding element type (key, value) (line 34)
str_12200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'str', 'iso-8859-2')

# Obtaining an instance of the builtin type 'tuple' (line 37)
tuple_12201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 37)
# Adding element type (line 37)
# Getting the type of 'QP' (line 37)
QP_12202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 20), tuple_12201, QP_12202)
# Adding element type (line 37)
# Getting the type of 'QP' (line 37)
QP_12203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 20), tuple_12201, QP_12203)
# Adding element type (line 37)
# Getting the type of 'None' (line 37)
None_12204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 20), tuple_12201, None_12204)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12200, tuple_12201))
# Adding element type (key, value) (line 34)
str_12205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'str', 'iso-8859-3')

# Obtaining an instance of the builtin type 'tuple' (line 38)
tuple_12206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 38)
# Adding element type (line 38)
# Getting the type of 'QP' (line 38)
QP_12207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 20), tuple_12206, QP_12207)
# Adding element type (line 38)
# Getting the type of 'QP' (line 38)
QP_12208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 20), tuple_12206, QP_12208)
# Adding element type (line 38)
# Getting the type of 'None' (line 38)
None_12209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 20), tuple_12206, None_12209)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12205, tuple_12206))
# Adding element type (key, value) (line 34)
str_12210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 4), 'str', 'iso-8859-4')

# Obtaining an instance of the builtin type 'tuple' (line 39)
tuple_12211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 39)
# Adding element type (line 39)
# Getting the type of 'QP' (line 39)
QP_12212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), tuple_12211, QP_12212)
# Adding element type (line 39)
# Getting the type of 'QP' (line 39)
QP_12213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), tuple_12211, QP_12213)
# Adding element type (line 39)
# Getting the type of 'None' (line 39)
None_12214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), tuple_12211, None_12214)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12210, tuple_12211))
# Adding element type (key, value) (line 34)
str_12215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 4), 'str', 'iso-8859-9')

# Obtaining an instance of the builtin type 'tuple' (line 44)
tuple_12216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 44)
# Adding element type (line 44)
# Getting the type of 'QP' (line 44)
QP_12217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 20), tuple_12216, QP_12217)
# Adding element type (line 44)
# Getting the type of 'QP' (line 44)
QP_12218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 20), tuple_12216, QP_12218)
# Adding element type (line 44)
# Getting the type of 'None' (line 44)
None_12219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 20), tuple_12216, None_12219)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12215, tuple_12216))
# Adding element type (key, value) (line 34)
str_12220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'str', 'iso-8859-10')

# Obtaining an instance of the builtin type 'tuple' (line 45)
tuple_12221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 45)
# Adding element type (line 45)
# Getting the type of 'QP' (line 45)
QP_12222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 20), tuple_12221, QP_12222)
# Adding element type (line 45)
# Getting the type of 'QP' (line 45)
QP_12223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 20), tuple_12221, QP_12223)
# Adding element type (line 45)
# Getting the type of 'None' (line 45)
None_12224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 20), tuple_12221, None_12224)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12220, tuple_12221))
# Adding element type (key, value) (line 34)
str_12225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'str', 'iso-8859-13')

# Obtaining an instance of the builtin type 'tuple' (line 47)
tuple_12226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 47)
# Adding element type (line 47)
# Getting the type of 'QP' (line 47)
QP_12227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 20), tuple_12226, QP_12227)
# Adding element type (line 47)
# Getting the type of 'QP' (line 47)
QP_12228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 20), tuple_12226, QP_12228)
# Adding element type (line 47)
# Getting the type of 'None' (line 47)
None_12229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 20), tuple_12226, None_12229)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12225, tuple_12226))
# Adding element type (key, value) (line 34)
str_12230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'str', 'iso-8859-14')

# Obtaining an instance of the builtin type 'tuple' (line 48)
tuple_12231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 48)
# Adding element type (line 48)
# Getting the type of 'QP' (line 48)
QP_12232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), tuple_12231, QP_12232)
# Adding element type (line 48)
# Getting the type of 'QP' (line 48)
QP_12233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), tuple_12231, QP_12233)
# Adding element type (line 48)
# Getting the type of 'None' (line 48)
None_12234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), tuple_12231, None_12234)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12230, tuple_12231))
# Adding element type (key, value) (line 34)
str_12235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'str', 'iso-8859-15')

# Obtaining an instance of the builtin type 'tuple' (line 49)
tuple_12236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 49)
# Adding element type (line 49)
# Getting the type of 'QP' (line 49)
QP_12237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 20), tuple_12236, QP_12237)
# Adding element type (line 49)
# Getting the type of 'QP' (line 49)
QP_12238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 20), tuple_12236, QP_12238)
# Adding element type (line 49)
# Getting the type of 'None' (line 49)
None_12239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 20), tuple_12236, None_12239)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12235, tuple_12236))
# Adding element type (key, value) (line 34)
str_12240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 4), 'str', 'iso-8859-16')

# Obtaining an instance of the builtin type 'tuple' (line 50)
tuple_12241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 50)
# Adding element type (line 50)
# Getting the type of 'QP' (line 50)
QP_12242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_12241, QP_12242)
# Adding element type (line 50)
# Getting the type of 'QP' (line 50)
QP_12243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_12241, QP_12243)
# Adding element type (line 50)
# Getting the type of 'None' (line 50)
None_12244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), tuple_12241, None_12244)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12240, tuple_12241))
# Adding element type (key, value) (line 34)
str_12245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'str', 'windows-1252')

# Obtaining an instance of the builtin type 'tuple' (line 51)
tuple_12246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 51)
# Adding element type (line 51)
# Getting the type of 'QP' (line 51)
QP_12247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_12246, QP_12247)
# Adding element type (line 51)
# Getting the type of 'QP' (line 51)
QP_12248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_12246, QP_12248)
# Adding element type (line 51)
# Getting the type of 'None' (line 51)
None_12249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_12246, None_12249)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12245, tuple_12246))
# Adding element type (key, value) (line 34)
str_12250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'str', 'viscii')

# Obtaining an instance of the builtin type 'tuple' (line 52)
tuple_12251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 52)
# Adding element type (line 52)
# Getting the type of 'QP' (line 52)
QP_12252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 20), tuple_12251, QP_12252)
# Adding element type (line 52)
# Getting the type of 'QP' (line 52)
QP_12253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'QP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 20), tuple_12251, QP_12253)
# Adding element type (line 52)
# Getting the type of 'None' (line 52)
None_12254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 20), tuple_12251, None_12254)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12250, tuple_12251))
# Adding element type (key, value) (line 34)
str_12255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 4), 'str', 'us-ascii')

# Obtaining an instance of the builtin type 'tuple' (line 53)
tuple_12256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 53)
# Adding element type (line 53)
# Getting the type of 'None' (line 53)
None_12257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 20), tuple_12256, None_12257)
# Adding element type (line 53)
# Getting the type of 'None' (line 53)
None_12258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 20), tuple_12256, None_12258)
# Adding element type (line 53)
# Getting the type of 'None' (line 53)
None_12259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 20), tuple_12256, None_12259)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12255, tuple_12256))
# Adding element type (key, value) (line 34)
str_12260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 4), 'str', 'big5')

# Obtaining an instance of the builtin type 'tuple' (line 54)
tuple_12261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 54)
# Adding element type (line 54)
# Getting the type of 'BASE64' (line 54)
BASE64_12262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), tuple_12261, BASE64_12262)
# Adding element type (line 54)
# Getting the type of 'BASE64' (line 54)
BASE64_12263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 31), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), tuple_12261, BASE64_12263)
# Adding element type (line 54)
# Getting the type of 'None' (line 54)
None_12264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), tuple_12261, None_12264)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12260, tuple_12261))
# Adding element type (key, value) (line 34)
str_12265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'str', 'gb2312')

# Obtaining an instance of the builtin type 'tuple' (line 55)
tuple_12266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 55)
# Adding element type (line 55)
# Getting the type of 'BASE64' (line 55)
BASE64_12267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), tuple_12266, BASE64_12267)
# Adding element type (line 55)
# Getting the type of 'BASE64' (line 55)
BASE64_12268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 31), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), tuple_12266, BASE64_12268)
# Adding element type (line 55)
# Getting the type of 'None' (line 55)
None_12269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), tuple_12266, None_12269)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12265, tuple_12266))
# Adding element type (key, value) (line 34)
str_12270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'str', 'euc-jp')

# Obtaining an instance of the builtin type 'tuple' (line 56)
tuple_12271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 56)
# Adding element type (line 56)
# Getting the type of 'BASE64' (line 56)
BASE64_12272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 20), tuple_12271, BASE64_12272)
# Adding element type (line 56)
# Getting the type of 'None' (line 56)
None_12273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 31), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 20), tuple_12271, None_12273)
# Adding element type (line 56)
str_12274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 40), 'str', 'iso-2022-jp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 20), tuple_12271, str_12274)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12270, tuple_12271))
# Adding element type (key, value) (line 34)
str_12275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 4), 'str', 'shift_jis')

# Obtaining an instance of the builtin type 'tuple' (line 57)
tuple_12276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 57)
# Adding element type (line 57)
# Getting the type of 'BASE64' (line 57)
BASE64_12277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 20), tuple_12276, BASE64_12277)
# Adding element type (line 57)
# Getting the type of 'None' (line 57)
None_12278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 31), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 20), tuple_12276, None_12278)
# Adding element type (line 57)
str_12279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 40), 'str', 'iso-2022-jp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 20), tuple_12276, str_12279)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12275, tuple_12276))
# Adding element type (key, value) (line 34)
str_12280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'str', 'iso-2022-jp')

# Obtaining an instance of the builtin type 'tuple' (line 58)
tuple_12281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 58)
# Adding element type (line 58)
# Getting the type of 'BASE64' (line 58)
BASE64_12282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 20), tuple_12281, BASE64_12282)
# Adding element type (line 58)
# Getting the type of 'None' (line 58)
None_12283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 31), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 20), tuple_12281, None_12283)
# Adding element type (line 58)
# Getting the type of 'None' (line 58)
None_12284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 20), tuple_12281, None_12284)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12280, tuple_12281))
# Adding element type (key, value) (line 34)
str_12285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'str', 'koi8-r')

# Obtaining an instance of the builtin type 'tuple' (line 59)
tuple_12286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 59)
# Adding element type (line 59)
# Getting the type of 'BASE64' (line 59)
BASE64_12287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 20), tuple_12286, BASE64_12287)
# Adding element type (line 59)
# Getting the type of 'BASE64' (line 59)
BASE64_12288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 31), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 20), tuple_12286, BASE64_12288)
# Adding element type (line 59)
# Getting the type of 'None' (line 59)
None_12289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 20), tuple_12286, None_12289)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12285, tuple_12286))
# Adding element type (key, value) (line 34)
str_12290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'str', 'utf-8')

# Obtaining an instance of the builtin type 'tuple' (line 60)
tuple_12291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 60)
# Adding element type (line 60)
# Getting the type of 'SHORTEST' (line 60)
SHORTEST_12292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'SHORTEST')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 20), tuple_12291, SHORTEST_12292)
# Adding element type (line 60)
# Getting the type of 'BASE64' (line 60)
BASE64_12293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 20), tuple_12291, BASE64_12293)
# Adding element type (line 60)
str_12294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'str', 'utf-8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 20), tuple_12291, str_12294)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12290, tuple_12291))
# Adding element type (key, value) (line 34)
str_12295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'str', '8bit')

# Obtaining an instance of the builtin type 'tuple' (line 62)
tuple_12296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 62)
# Adding element type (line 62)
# Getting the type of 'None' (line 62)
None_12297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), tuple_12296, None_12297)
# Adding element type (line 62)
# Getting the type of 'BASE64' (line 62)
BASE64_12298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'BASE64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), tuple_12296, BASE64_12298)
# Adding element type (line 62)
str_12299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 39), 'str', 'utf-8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), tuple_12296, str_12299)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), dict_12194, (str_12295, tuple_12296))

# Assigning a type to the variable 'CHARSETS' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'CHARSETS', dict_12194)

# Assigning a Dict to a Name (line 67):

# Assigning a Dict to a Name (line 67):

# Obtaining an instance of the builtin type 'dict' (line 67)
dict_12300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 67)
# Adding element type (key, value) (line 67)
str_12301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'str', 'latin_1')
str_12302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 15), 'str', 'iso-8859-1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12301, str_12302))
# Adding element type (key, value) (line 67)
str_12303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'str', 'latin-1')
str_12304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 15), 'str', 'iso-8859-1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12303, str_12304))
# Adding element type (key, value) (line 67)
str_12305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'str', 'latin_2')
str_12306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 15), 'str', 'iso-8859-2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12305, str_12306))
# Adding element type (key, value) (line 67)
str_12307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'str', 'latin-2')
str_12308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'str', 'iso-8859-2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12307, str_12308))
# Adding element type (key, value) (line 67)
str_12309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'str', 'latin_3')
str_12310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 15), 'str', 'iso-8859-3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12309, str_12310))
# Adding element type (key, value) (line 67)
str_12311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'str', 'latin-3')
str_12312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 15), 'str', 'iso-8859-3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12311, str_12312))
# Adding element type (key, value) (line 67)
str_12313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'str', 'latin_4')
str_12314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 15), 'str', 'iso-8859-4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12313, str_12314))
# Adding element type (key, value) (line 67)
str_12315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'str', 'latin-4')
str_12316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 15), 'str', 'iso-8859-4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12315, str_12316))
# Adding element type (key, value) (line 67)
str_12317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'str', 'latin_5')
str_12318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'str', 'iso-8859-9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12317, str_12318))
# Adding element type (key, value) (line 67)
str_12319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'str', 'latin-5')
str_12320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 15), 'str', 'iso-8859-9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12319, str_12320))
# Adding element type (key, value) (line 67)
str_12321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 4), 'str', 'latin_6')
str_12322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'str', 'iso-8859-10')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12321, str_12322))
# Adding element type (key, value) (line 67)
str_12323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'str', 'latin-6')
str_12324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 15), 'str', 'iso-8859-10')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12323, str_12324))
# Adding element type (key, value) (line 67)
str_12325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 4), 'str', 'latin_7')
str_12326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 15), 'str', 'iso-8859-13')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12325, str_12326))
# Adding element type (key, value) (line 67)
str_12327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 4), 'str', 'latin-7')
str_12328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 15), 'str', 'iso-8859-13')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12327, str_12328))
# Adding element type (key, value) (line 67)
str_12329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'str', 'latin_8')
str_12330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'str', 'iso-8859-14')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12329, str_12330))
# Adding element type (key, value) (line 67)
str_12331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 4), 'str', 'latin-8')
str_12332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'str', 'iso-8859-14')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12331, str_12332))
# Adding element type (key, value) (line 67)
str_12333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 4), 'str', 'latin_9')
str_12334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 15), 'str', 'iso-8859-15')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12333, str_12334))
# Adding element type (key, value) (line 67)
str_12335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'str', 'latin-9')
str_12336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 15), 'str', 'iso-8859-15')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12335, str_12336))
# Adding element type (key, value) (line 67)
str_12337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'str', 'latin_10')
str_12338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 15), 'str', 'iso-8859-16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12337, str_12338))
# Adding element type (key, value) (line 67)
str_12339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 4), 'str', 'latin-10')
str_12340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'str', 'iso-8859-16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12339, str_12340))
# Adding element type (key, value) (line 67)
str_12341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'str', 'cp949')
str_12342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'str', 'ks_c_5601-1987')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12341, str_12342))
# Adding element type (key, value) (line 67)
str_12343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'str', 'euc_jp')
str_12344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 15), 'str', 'euc-jp')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12343, str_12344))
# Adding element type (key, value) (line 67)
str_12345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'str', 'euc_kr')
str_12346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 15), 'str', 'euc-kr')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12345, str_12346))
# Adding element type (key, value) (line 67)
str_12347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'str', 'ascii')
str_12348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 15), 'str', 'us-ascii')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_12300, (str_12347, str_12348))

# Assigning a type to the variable 'ALIASES' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'ALIASES', dict_12300)

# Assigning a Dict to a Name (line 96):

# Assigning a Dict to a Name (line 96):

# Obtaining an instance of the builtin type 'dict' (line 96)
dict_12349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 96)
# Adding element type (key, value) (line 96)
str_12350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'str', 'gb2312')
str_12351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 19), 'str', 'eucgb2312_cn')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 12), dict_12349, (str_12350, str_12351))
# Adding element type (key, value) (line 96)
str_12352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'str', 'big5')
str_12353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'str', 'big5_tw')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 12), dict_12349, (str_12352, str_12353))
# Adding element type (key, value) (line 96)
str_12354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 4), 'str', 'us-ascii')
# Getting the type of 'None' (line 102)
None_12355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 12), dict_12349, (str_12354, None_12355))

# Assigning a type to the variable 'CODEC_MAP' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'CODEC_MAP', dict_12349)

@norecursion
def add_charset(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 108)
    None_12356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 36), 'None')
    # Getting the type of 'None' (line 108)
    None_12357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 51), 'None')
    # Getting the type of 'None' (line 108)
    None_12358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 72), 'None')
    defaults = [None_12356, None_12357, None_12358]
    # Create a new context for function 'add_charset'
    module_type_store = module_type_store.open_function_context('add_charset', 108, 0, False)
    
    # Passed parameters checking function
    add_charset.stypy_localization = localization
    add_charset.stypy_type_of_self = None
    add_charset.stypy_type_store = module_type_store
    add_charset.stypy_function_name = 'add_charset'
    add_charset.stypy_param_names_list = ['charset', 'header_enc', 'body_enc', 'output_charset']
    add_charset.stypy_varargs_param_name = None
    add_charset.stypy_kwargs_param_name = None
    add_charset.stypy_call_defaults = defaults
    add_charset.stypy_call_varargs = varargs
    add_charset.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_charset', ['charset', 'header_enc', 'body_enc', 'output_charset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_charset', localization, ['charset', 'header_enc', 'body_enc', 'output_charset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_charset(...)' code ##################

    str_12359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, (-1)), 'str', "Add character set properties to the global registry.\n\n    charset is the input character set, and must be the canonical name of a\n    character set.\n\n    Optional header_enc and body_enc is either Charset.QP for\n    quoted-printable, Charset.BASE64 for base64 encoding, Charset.SHORTEST for\n    the shortest of qp or base64 encoding, or None for no encoding.  SHORTEST\n    is only valid for header_enc.  It describes how message headers and\n    message bodies in the input charset are to be encoded.  Default is no\n    encoding.\n\n    Optional output_charset is the character set that the output should be\n    in.  Conversions will proceed from input charset, to Unicode, to the\n    output charset when the method Charset.convert() is called.  The default\n    is to output in the same character set as the input.\n\n    Both input_charset and output_charset must have Unicode codec entries in\n    the module's charset-to-codec mapping; use add_codec(charset, codecname)\n    to add codecs the module does not know about.  See the codecs module's\n    documentation for more information.\n    ")
    
    # Getting the type of 'body_enc' (line 131)
    body_enc_12360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 7), 'body_enc')
    # Getting the type of 'SHORTEST' (line 131)
    SHORTEST_12361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'SHORTEST')
    # Applying the binary operator '==' (line 131)
    result_eq_12362 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 7), '==', body_enc_12360, SHORTEST_12361)
    
    # Testing if the type of an if condition is none (line 131)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 131, 4), result_eq_12362):
        pass
    else:
        
        # Testing the type of an if condition (line 131)
        if_condition_12363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 4), result_eq_12362)
        # Assigning a type to the variable 'if_condition_12363' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'if_condition_12363', if_condition_12363)
        # SSA begins for if statement (line 131)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 132)
        # Processing the call arguments (line 132)
        str_12365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 25), 'str', 'SHORTEST not allowed for body_enc')
        # Processing the call keyword arguments (line 132)
        kwargs_12366 = {}
        # Getting the type of 'ValueError' (line 132)
        ValueError_12364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 132)
        ValueError_call_result_12367 = invoke(stypy.reporting.localization.Localization(__file__, 132, 14), ValueError_12364, *[str_12365], **kwargs_12366)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 132, 8), ValueError_call_result_12367, 'raise parameter', BaseException)
        # SSA join for if statement (line 131)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Tuple to a Subscript (line 133):
    
    # Assigning a Tuple to a Subscript (line 133):
    
    # Obtaining an instance of the builtin type 'tuple' (line 133)
    tuple_12368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 133)
    # Adding element type (line 133)
    # Getting the type of 'header_enc' (line 133)
    header_enc_12369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 25), 'header_enc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 25), tuple_12368, header_enc_12369)
    # Adding element type (line 133)
    # Getting the type of 'body_enc' (line 133)
    body_enc_12370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 37), 'body_enc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 25), tuple_12368, body_enc_12370)
    # Adding element type (line 133)
    # Getting the type of 'output_charset' (line 133)
    output_charset_12371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 47), 'output_charset')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 25), tuple_12368, output_charset_12371)
    
    # Getting the type of 'CHARSETS' (line 133)
    CHARSETS_12372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'CHARSETS')
    # Getting the type of 'charset' (line 133)
    charset_12373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'charset')
    # Storing an element on a container (line 133)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), CHARSETS_12372, (charset_12373, tuple_12368))
    
    # ################# End of 'add_charset(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_charset' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_12374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12374)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_charset'
    return stypy_return_type_12374

# Assigning a type to the variable 'add_charset' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'add_charset', add_charset)

@norecursion
def add_alias(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'add_alias'
    module_type_store = module_type_store.open_function_context('add_alias', 136, 0, False)
    
    # Passed parameters checking function
    add_alias.stypy_localization = localization
    add_alias.stypy_type_of_self = None
    add_alias.stypy_type_store = module_type_store
    add_alias.stypy_function_name = 'add_alias'
    add_alias.stypy_param_names_list = ['alias', 'canonical']
    add_alias.stypy_varargs_param_name = None
    add_alias.stypy_kwargs_param_name = None
    add_alias.stypy_call_defaults = defaults
    add_alias.stypy_call_varargs = varargs
    add_alias.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_alias', ['alias', 'canonical'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_alias', localization, ['alias', 'canonical'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_alias(...)' code ##################

    str_12375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, (-1)), 'str', "Add a character set alias.\n\n    alias is the alias name, e.g. latin-1\n    canonical is the character set's canonical name, e.g. iso-8859-1\n    ")
    
    # Assigning a Name to a Subscript (line 142):
    
    # Assigning a Name to a Subscript (line 142):
    # Getting the type of 'canonical' (line 142)
    canonical_12376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'canonical')
    # Getting the type of 'ALIASES' (line 142)
    ALIASES_12377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'ALIASES')
    # Getting the type of 'alias' (line 142)
    alias_12378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'alias')
    # Storing an element on a container (line 142)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 4), ALIASES_12377, (alias_12378, canonical_12376))
    
    # ################# End of 'add_alias(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_alias' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12379)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_alias'
    return stypy_return_type_12379

# Assigning a type to the variable 'add_alias' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'add_alias', add_alias)

@norecursion
def add_codec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'add_codec'
    module_type_store = module_type_store.open_function_context('add_codec', 145, 0, False)
    
    # Passed parameters checking function
    add_codec.stypy_localization = localization
    add_codec.stypy_type_of_self = None
    add_codec.stypy_type_store = module_type_store
    add_codec.stypy_function_name = 'add_codec'
    add_codec.stypy_param_names_list = ['charset', 'codecname']
    add_codec.stypy_varargs_param_name = None
    add_codec.stypy_kwargs_param_name = None
    add_codec.stypy_call_defaults = defaults
    add_codec.stypy_call_varargs = varargs
    add_codec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_codec', ['charset', 'codecname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_codec', localization, ['charset', 'codecname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_codec(...)' code ##################

    str_12380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, (-1)), 'str', 'Add a codec that map characters in the given charset to/from Unicode.\n\n    charset is the canonical name of a character set.  codecname is the name\n    of a Python codec, as appropriate for the second argument to the unicode()\n    built-in, or to the encode() method of a Unicode string.\n    ')
    
    # Assigning a Name to a Subscript (line 152):
    
    # Assigning a Name to a Subscript (line 152):
    # Getting the type of 'codecname' (line 152)
    codecname_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'codecname')
    # Getting the type of 'CODEC_MAP' (line 152)
    CODEC_MAP_12382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'CODEC_MAP')
    # Getting the type of 'charset' (line 152)
    charset_12383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), 'charset')
    # Storing an element on a container (line 152)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 4), CODEC_MAP_12382, (charset_12383, codecname_12381))
    
    # ################# End of 'add_codec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_codec' in the type store
    # Getting the type of 'stypy_return_type' (line 145)
    stypy_return_type_12384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12384)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_codec'
    return stypy_return_type_12384

# Assigning a type to the variable 'add_codec' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'add_codec', add_codec)
# Declaration of the 'Charset' class

class Charset:
    str_12385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, (-1)), 'str', "Map character sets to their email properties.\n\n    This class provides information about the requirements imposed on email\n    for a specific character set.  It also provides convenience routines for\n    converting between character sets, given the availability of the\n    applicable codecs.  Given a character set, it will do its best to provide\n    information on how to use that character set in an email in an\n    RFC-compliant way.\n\n    Certain character sets must be encoded with quoted-printable or base64\n    when used in email headers or bodies.  Certain character sets must be\n    converted outright, and are not allowed in email.  Instances of this\n    module expose the following information about a character set:\n\n    input_charset: The initial character set specified.  Common aliases\n                   are converted to their `official' email names (e.g. latin_1\n                   is converted to iso-8859-1).  Defaults to 7-bit us-ascii.\n\n    header_encoding: If the character set must be encoded before it can be\n                     used in an email header, this attribute will be set to\n                     Charset.QP (for quoted-printable), Charset.BASE64 (for\n                     base64 encoding), or Charset.SHORTEST for the shortest of\n                     QP or BASE64 encoding.  Otherwise, it will be None.\n\n    body_encoding: Same as header_encoding, but describes the encoding for the\n                   mail message's body, which indeed may be different than the\n                   header encoding.  Charset.SHORTEST is not allowed for\n                   body_encoding.\n\n    output_charset: Some character sets must be converted before they can be\n                    used in email headers or bodies.  If the input_charset is\n                    one of them, this attribute will contain the name of the\n                    charset output will be converted to.  Otherwise, it will\n                    be None.\n\n    input_codec: The name of the Python codec used to convert the\n                 input_charset to Unicode.  If no conversion codec is\n                 necessary, this attribute will be None.\n\n    output_codec: The name of the Python codec used to convert Unicode\n                  to the output_charset.  If no conversion codec is necessary,\n                  this attribute will have the same value as the input_codec.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'DEFAULT_CHARSET' (line 200)
        DEFAULT_CHARSET_12386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 37), 'DEFAULT_CHARSET')
        defaults = [DEFAULT_CHARSET_12386]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 200, 4, False)
        # Assigning a type to the variable 'self' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.__init__', ['input_charset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['input_charset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Type idiom detected: calculating its left and rigth part (line 206)
        # Getting the type of 'unicode' (line 206)
        unicode_12387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 41), 'unicode')
        # Getting the type of 'input_charset' (line 206)
        input_charset_12388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'input_charset')
        
        (may_be_12389, more_types_in_union_12390) = may_be_subtype(unicode_12387, input_charset_12388)

        if may_be_12389:

            if more_types_in_union_12390:
                # Runtime conditional SSA (line 206)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'input_charset' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'input_charset', remove_not_subtype_from_union(input_charset_12388, unicode))
            
            # Call to encode(...): (line 207)
            # Processing the call arguments (line 207)
            str_12393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 37), 'str', 'ascii')
            # Processing the call keyword arguments (line 207)
            kwargs_12394 = {}
            # Getting the type of 'input_charset' (line 207)
            input_charset_12391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'input_charset', False)
            # Obtaining the member 'encode' of a type (line 207)
            encode_12392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), input_charset_12391, 'encode')
            # Calling encode(args, kwargs) (line 207)
            encode_call_result_12395 = invoke(stypy.reporting.localization.Localization(__file__, 207, 16), encode_12392, *[str_12393], **kwargs_12394)
            

            if more_types_in_union_12390:
                # Runtime conditional SSA for else branch (line 206)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_12389) or more_types_in_union_12390):
            # Assigning a type to the variable 'input_charset' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'input_charset', remove_subtype_from_union(input_charset_12388, unicode))
            
            # Assigning a Call to a Name (line 209):
            
            # Assigning a Call to a Name (line 209):
            
            # Call to unicode(...): (line 209)
            # Processing the call arguments (line 209)
            # Getting the type of 'input_charset' (line 209)
            input_charset_12397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 40), 'input_charset', False)
            str_12398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 55), 'str', 'ascii')
            # Processing the call keyword arguments (line 209)
            kwargs_12399 = {}
            # Getting the type of 'unicode' (line 209)
            unicode_12396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 32), 'unicode', False)
            # Calling unicode(args, kwargs) (line 209)
            unicode_call_result_12400 = invoke(stypy.reporting.localization.Localization(__file__, 209, 32), unicode_12396, *[input_charset_12397, str_12398], **kwargs_12399)
            
            # Assigning a type to the variable 'input_charset' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'input_charset', unicode_call_result_12400)

            if (may_be_12389 and more_types_in_union_12390):
                # SSA join for if statement (line 206)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the except part of a try statement (line 205)
        # SSA branch for the except 'UnicodeError' branch of a try statement (line 205)
        module_type_store.open_ssa_branch('except')
        
        # Call to CharsetError(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'input_charset' (line 211)
        input_charset_12403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'input_charset', False)
        # Processing the call keyword arguments (line 211)
        kwargs_12404 = {}
        # Getting the type of 'errors' (line 211)
        errors_12401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'errors', False)
        # Obtaining the member 'CharsetError' of a type (line 211)
        CharsetError_12402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 18), errors_12401, 'CharsetError')
        # Calling CharsetError(args, kwargs) (line 211)
        CharsetError_call_result_12405 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), CharsetError_12402, *[input_charset_12403], **kwargs_12404)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 211, 12), CharsetError_call_result_12405, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to encode(...): (line 212)
        # Processing the call arguments (line 212)
        str_12411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 53), 'str', 'ascii')
        # Processing the call keyword arguments (line 212)
        kwargs_12412 = {}
        
        # Call to lower(...): (line 212)
        # Processing the call keyword arguments (line 212)
        kwargs_12408 = {}
        # Getting the type of 'input_charset' (line 212)
        input_charset_12406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'input_charset', False)
        # Obtaining the member 'lower' of a type (line 212)
        lower_12407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 24), input_charset_12406, 'lower')
        # Calling lower(args, kwargs) (line 212)
        lower_call_result_12409 = invoke(stypy.reporting.localization.Localization(__file__, 212, 24), lower_12407, *[], **kwargs_12408)
        
        # Obtaining the member 'encode' of a type (line 212)
        encode_12410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 24), lower_call_result_12409, 'encode')
        # Calling encode(args, kwargs) (line 212)
        encode_call_result_12413 = invoke(stypy.reporting.localization.Localization(__file__, 212, 24), encode_12410, *[str_12411], **kwargs_12412)
        
        # Assigning a type to the variable 'input_charset' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'input_charset', encode_call_result_12413)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'input_charset' (line 214)
        input_charset_12414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'input_charset')
        # Getting the type of 'ALIASES' (line 214)
        ALIASES_12415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 33), 'ALIASES')
        # Applying the binary operator 'in' (line 214)
        result_contains_12416 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 16), 'in', input_charset_12414, ALIASES_12415)
        
        
        # Getting the type of 'input_charset' (line 214)
        input_charset_12417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 44), 'input_charset')
        # Getting the type of 'CHARSETS' (line 214)
        CHARSETS_12418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 61), 'CHARSETS')
        # Applying the binary operator 'in' (line 214)
        result_contains_12419 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 44), 'in', input_charset_12417, CHARSETS_12418)
        
        # Applying the binary operator 'or' (line 214)
        result_or_keyword_12420 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 16), 'or', result_contains_12416, result_contains_12419)
        
        # Applying the 'not' unary operator (line 214)
        result_not__12421 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), 'not', result_or_keyword_12420)
        
        # Testing if the type of an if condition is none (line 214)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 214, 8), result_not__12421):
            pass
        else:
            
            # Testing the type of an if condition (line 214)
            if_condition_12422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 8), result_not__12421)
            # Assigning a type to the variable 'if_condition_12422' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'if_condition_12422', if_condition_12422)
            # SSA begins for if statement (line 214)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # SSA begins for try-except statement (line 215)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Attribute to a Name (line 216):
            
            # Assigning a Attribute to a Name (line 216):
            
            # Call to lookup(...): (line 216)
            # Processing the call arguments (line 216)
            # Getting the type of 'input_charset' (line 216)
            input_charset_12425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 46), 'input_charset', False)
            # Processing the call keyword arguments (line 216)
            kwargs_12426 = {}
            # Getting the type of 'codecs' (line 216)
            codecs_12423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 32), 'codecs', False)
            # Obtaining the member 'lookup' of a type (line 216)
            lookup_12424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 32), codecs_12423, 'lookup')
            # Calling lookup(args, kwargs) (line 216)
            lookup_call_result_12427 = invoke(stypy.reporting.localization.Localization(__file__, 216, 32), lookup_12424, *[input_charset_12425], **kwargs_12426)
            
            # Obtaining the member 'name' of a type (line 216)
            name_12428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 32), lookup_call_result_12427, 'name')
            # Assigning a type to the variable 'input_charset' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'input_charset', name_12428)
            # SSA branch for the except part of a try statement (line 215)
            # SSA branch for the except 'LookupError' branch of a try statement (line 215)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA join for try-except statement (line 215)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 214)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Attribute (line 219):
        
        # Assigning a Call to a Attribute (line 219):
        
        # Call to get(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'input_charset' (line 219)
        input_charset_12431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 41), 'input_charset', False)
        # Getting the type of 'input_charset' (line 219)
        input_charset_12432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 56), 'input_charset', False)
        # Processing the call keyword arguments (line 219)
        kwargs_12433 = {}
        # Getting the type of 'ALIASES' (line 219)
        ALIASES_12429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 29), 'ALIASES', False)
        # Obtaining the member 'get' of a type (line 219)
        get_12430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 29), ALIASES_12429, 'get')
        # Calling get(args, kwargs) (line 219)
        get_call_result_12434 = invoke(stypy.reporting.localization.Localization(__file__, 219, 29), get_12430, *[input_charset_12431, input_charset_12432], **kwargs_12433)
        
        # Getting the type of 'self' (line 219)
        self_12435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self')
        # Setting the type of the member 'input_charset' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_12435, 'input_charset', get_call_result_12434)
        
        # Assigning a Call to a Tuple (line 223):
        
        # Assigning a Call to a Name:
        
        # Call to get(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'self' (line 223)
        self_12438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 40), 'self', False)
        # Obtaining the member 'input_charset' of a type (line 223)
        input_charset_12439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 40), self_12438, 'input_charset')
        
        # Obtaining an instance of the builtin type 'tuple' (line 224)
        tuple_12440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 224)
        # Adding element type (line 224)
        # Getting the type of 'SHORTEST' (line 224)
        SHORTEST_12441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 41), 'SHORTEST', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 41), tuple_12440, SHORTEST_12441)
        # Adding element type (line 224)
        # Getting the type of 'BASE64' (line 224)
        BASE64_12442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 51), 'BASE64', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 41), tuple_12440, BASE64_12442)
        # Adding element type (line 224)
        # Getting the type of 'None' (line 224)
        None_12443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 59), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 41), tuple_12440, None_12443)
        
        # Processing the call keyword arguments (line 223)
        kwargs_12444 = {}
        # Getting the type of 'CHARSETS' (line 223)
        CHARSETS_12436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 27), 'CHARSETS', False)
        # Obtaining the member 'get' of a type (line 223)
        get_12437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 27), CHARSETS_12436, 'get')
        # Calling get(args, kwargs) (line 223)
        get_call_result_12445 = invoke(stypy.reporting.localization.Localization(__file__, 223, 27), get_12437, *[input_charset_12439, tuple_12440], **kwargs_12444)
        
        # Assigning a type to the variable 'call_assignment_12174' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'call_assignment_12174', get_call_result_12445)
        
        # Assigning a Call to a Name (line 223):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_12174' (line 223)
        call_assignment_12174_12446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'call_assignment_12174', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_12447 = stypy_get_value_from_tuple(call_assignment_12174_12446, 3, 0)
        
        # Assigning a type to the variable 'call_assignment_12175' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'call_assignment_12175', stypy_get_value_from_tuple_call_result_12447)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'call_assignment_12175' (line 223)
        call_assignment_12175_12448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'call_assignment_12175')
        # Assigning a type to the variable 'henc' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'henc', call_assignment_12175_12448)
        
        # Assigning a Call to a Name (line 223):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_12174' (line 223)
        call_assignment_12174_12449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'call_assignment_12174', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_12450 = stypy_get_value_from_tuple(call_assignment_12174_12449, 3, 1)
        
        # Assigning a type to the variable 'call_assignment_12176' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'call_assignment_12176', stypy_get_value_from_tuple_call_result_12450)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'call_assignment_12176' (line 223)
        call_assignment_12176_12451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'call_assignment_12176')
        # Assigning a type to the variable 'benc' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 14), 'benc', call_assignment_12176_12451)
        
        # Assigning a Call to a Name (line 223):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_12174' (line 223)
        call_assignment_12174_12452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'call_assignment_12174', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_12453 = stypy_get_value_from_tuple(call_assignment_12174_12452, 3, 2)
        
        # Assigning a type to the variable 'call_assignment_12177' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'call_assignment_12177', stypy_get_value_from_tuple_call_result_12453)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'call_assignment_12177' (line 223)
        call_assignment_12177_12454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'call_assignment_12177')
        # Assigning a type to the variable 'conv' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'conv', call_assignment_12177_12454)
        
        # Getting the type of 'conv' (line 225)
        conv_12455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'conv')
        # Applying the 'not' unary operator (line 225)
        result_not__12456 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 11), 'not', conv_12455)
        
        # Testing if the type of an if condition is none (line 225)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 225, 8), result_not__12456):
            pass
        else:
            
            # Testing the type of an if condition (line 225)
            if_condition_12457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 8), result_not__12456)
            # Assigning a type to the variable 'if_condition_12457' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'if_condition_12457', if_condition_12457)
            # SSA begins for if statement (line 225)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 226):
            
            # Assigning a Attribute to a Name (line 226):
            # Getting the type of 'self' (line 226)
            self_12458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'self')
            # Obtaining the member 'input_charset' of a type (line 226)
            input_charset_12459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 19), self_12458, 'input_charset')
            # Assigning a type to the variable 'conv' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'conv', input_charset_12459)
            # SSA join for if statement (line 225)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Attribute (line 228):
        
        # Assigning a Name to a Attribute (line 228):
        # Getting the type of 'henc' (line 228)
        henc_12460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 31), 'henc')
        # Getting the type of 'self' (line 228)
        self_12461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self')
        # Setting the type of the member 'header_encoding' of a type (line 228)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_12461, 'header_encoding', henc_12460)
        
        # Assigning a Name to a Attribute (line 229):
        
        # Assigning a Name to a Attribute (line 229):
        # Getting the type of 'benc' (line 229)
        benc_12462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 29), 'benc')
        # Getting the type of 'self' (line 229)
        self_12463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'self')
        # Setting the type of the member 'body_encoding' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), self_12463, 'body_encoding', benc_12462)
        
        # Assigning a Call to a Attribute (line 230):
        
        # Assigning a Call to a Attribute (line 230):
        
        # Call to get(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'conv' (line 230)
        conv_12466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 42), 'conv', False)
        # Getting the type of 'conv' (line 230)
        conv_12467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 48), 'conv', False)
        # Processing the call keyword arguments (line 230)
        kwargs_12468 = {}
        # Getting the type of 'ALIASES' (line 230)
        ALIASES_12464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 30), 'ALIASES', False)
        # Obtaining the member 'get' of a type (line 230)
        get_12465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 30), ALIASES_12464, 'get')
        # Calling get(args, kwargs) (line 230)
        get_call_result_12469 = invoke(stypy.reporting.localization.Localization(__file__, 230, 30), get_12465, *[conv_12466, conv_12467], **kwargs_12468)
        
        # Getting the type of 'self' (line 230)
        self_12470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self')
        # Setting the type of the member 'output_charset' of a type (line 230)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_12470, 'output_charset', get_call_result_12469)
        
        # Assigning a Call to a Attribute (line 233):
        
        # Assigning a Call to a Attribute (line 233):
        
        # Call to get(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'self' (line 233)
        self_12473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 41), 'self', False)
        # Obtaining the member 'input_charset' of a type (line 233)
        input_charset_12474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 41), self_12473, 'input_charset')
        # Getting the type of 'self' (line 234)
        self_12475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 41), 'self', False)
        # Obtaining the member 'input_charset' of a type (line 234)
        input_charset_12476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 41), self_12475, 'input_charset')
        # Processing the call keyword arguments (line 233)
        kwargs_12477 = {}
        # Getting the type of 'CODEC_MAP' (line 233)
        CODEC_MAP_12471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'CODEC_MAP', False)
        # Obtaining the member 'get' of a type (line 233)
        get_12472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 27), CODEC_MAP_12471, 'get')
        # Calling get(args, kwargs) (line 233)
        get_call_result_12478 = invoke(stypy.reporting.localization.Localization(__file__, 233, 27), get_12472, *[input_charset_12474, input_charset_12476], **kwargs_12477)
        
        # Getting the type of 'self' (line 233)
        self_12479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'self')
        # Setting the type of the member 'input_codec' of a type (line 233)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), self_12479, 'input_codec', get_call_result_12478)
        
        # Assigning a Call to a Attribute (line 235):
        
        # Assigning a Call to a Attribute (line 235):
        
        # Call to get(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'self' (line 235)
        self_12482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 42), 'self', False)
        # Obtaining the member 'output_charset' of a type (line 235)
        output_charset_12483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 42), self_12482, 'output_charset')
        # Getting the type of 'self' (line 236)
        self_12484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 42), 'self', False)
        # Obtaining the member 'output_charset' of a type (line 236)
        output_charset_12485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 42), self_12484, 'output_charset')
        # Processing the call keyword arguments (line 235)
        kwargs_12486 = {}
        # Getting the type of 'CODEC_MAP' (line 235)
        CODEC_MAP_12480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'CODEC_MAP', False)
        # Obtaining the member 'get' of a type (line 235)
        get_12481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 28), CODEC_MAP_12480, 'get')
        # Calling get(args, kwargs) (line 235)
        get_call_result_12487 = invoke(stypy.reporting.localization.Localization(__file__, 235, 28), get_12481, *[output_charset_12483, output_charset_12485], **kwargs_12486)
        
        # Getting the type of 'self' (line 235)
        self_12488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self')
        # Setting the type of the member 'output_codec' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_12488, 'output_codec', get_call_result_12487)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 238, 4, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Charset.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Charset.stypy__str__')
        Charset.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Charset.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Call to lower(...): (line 239)
        # Processing the call keyword arguments (line 239)
        kwargs_12492 = {}
        # Getting the type of 'self' (line 239)
        self_12489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'self', False)
        # Obtaining the member 'input_charset' of a type (line 239)
        input_charset_12490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 15), self_12489, 'input_charset')
        # Obtaining the member 'lower' of a type (line 239)
        lower_12491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 15), input_charset_12490, 'lower')
        # Calling lower(args, kwargs) (line 239)
        lower_call_result_12493 = invoke(stypy.reporting.localization.Localization(__file__, 239, 15), lower_12491, *[], **kwargs_12492)
        
        # Assigning a type to the variable 'stypy_return_type' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'stypy_return_type', lower_call_result_12493)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 238)
        stypy_return_type_12494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12494)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_12494

    
    # Assigning a Name to a Name (line 241):

    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 243, 4, False)
        # Assigning a type to the variable 'self' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        Charset.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'Charset.stypy__eq__')
        Charset.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Charset.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        
        # Call to str(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'self' (line 244)
        self_12496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'self', False)
        # Processing the call keyword arguments (line 244)
        kwargs_12497 = {}
        # Getting the type of 'str' (line 244)
        str_12495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'str', False)
        # Calling str(args, kwargs) (line 244)
        str_call_result_12498 = invoke(stypy.reporting.localization.Localization(__file__, 244, 15), str_12495, *[self_12496], **kwargs_12497)
        
        
        # Call to lower(...): (line 244)
        # Processing the call keyword arguments (line 244)
        kwargs_12504 = {}
        
        # Call to str(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'other' (line 244)
        other_12500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 32), 'other', False)
        # Processing the call keyword arguments (line 244)
        kwargs_12501 = {}
        # Getting the type of 'str' (line 244)
        str_12499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 28), 'str', False)
        # Calling str(args, kwargs) (line 244)
        str_call_result_12502 = invoke(stypy.reporting.localization.Localization(__file__, 244, 28), str_12499, *[other_12500], **kwargs_12501)
        
        # Obtaining the member 'lower' of a type (line 244)
        lower_12503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 28), str_call_result_12502, 'lower')
        # Calling lower(args, kwargs) (line 244)
        lower_call_result_12505 = invoke(stypy.reporting.localization.Localization(__file__, 244, 28), lower_12503, *[], **kwargs_12504)
        
        # Applying the binary operator '==' (line 244)
        result_eq_12506 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 15), '==', str_call_result_12498, lower_call_result_12505)
        
        # Assigning a type to the variable 'stypy_return_type' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'stypy_return_type', result_eq_12506)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_12507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12507)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_12507


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.__ne__.__dict__.__setitem__('stypy_localization', localization)
        Charset.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.__ne__.__dict__.__setitem__('stypy_function_name', 'Charset.__ne__')
        Charset.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Charset.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.__ne__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ne__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ne__(...)' code ##################

        
        
        # Call to __eq__(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'other' (line 247)
        other_12510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 31), 'other', False)
        # Processing the call keyword arguments (line 247)
        kwargs_12511 = {}
        # Getting the type of 'self' (line 247)
        self_12508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 19), 'self', False)
        # Obtaining the member '__eq__' of a type (line 247)
        eq___12509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 19), self_12508, '__eq__')
        # Calling __eq__(args, kwargs) (line 247)
        eq___call_result_12512 = invoke(stypy.reporting.localization.Localization(__file__, 247, 19), eq___12509, *[other_12510], **kwargs_12511)
        
        # Applying the 'not' unary operator (line 247)
        result_not__12513 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 15), 'not', eq___call_result_12512)
        
        # Assigning a type to the variable 'stypy_return_type' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'stypy_return_type', result_not__12513)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_12514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12514)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_12514


    @norecursion
    def get_body_encoding(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_body_encoding'
        module_type_store = module_type_store.open_function_context('get_body_encoding', 249, 4, False)
        # Assigning a type to the variable 'self' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.get_body_encoding.__dict__.__setitem__('stypy_localization', localization)
        Charset.get_body_encoding.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.get_body_encoding.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.get_body_encoding.__dict__.__setitem__('stypy_function_name', 'Charset.get_body_encoding')
        Charset.get_body_encoding.__dict__.__setitem__('stypy_param_names_list', [])
        Charset.get_body_encoding.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.get_body_encoding.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.get_body_encoding.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.get_body_encoding.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.get_body_encoding.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.get_body_encoding.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.get_body_encoding', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_body_encoding', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_body_encoding(...)' code ##################

        str_12515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, (-1)), 'str', 'Return the content-transfer-encoding used for body encoding.\n\n        This is either the string `quoted-printable\' or `base64\' depending on\n        the encoding used, or it is a function in which case you should call\n        the function with a single argument, the Message object being\n        encoded.  The function should then set the Content-Transfer-Encoding\n        header itself to whatever is appropriate.\n\n        Returns "quoted-printable" if self.body_encoding is QP.\n        Returns "base64" if self.body_encoding is BASE64.\n        Returns "7bit" otherwise.\n        ')
        # Evaluating assert statement condition
        
        # Getting the type of 'self' (line 262)
        self_12516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'self')
        # Obtaining the member 'body_encoding' of a type (line 262)
        body_encoding_12517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 15), self_12516, 'body_encoding')
        # Getting the type of 'SHORTEST' (line 262)
        SHORTEST_12518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 37), 'SHORTEST')
        # Applying the binary operator '!=' (line 262)
        result_ne_12519 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 15), '!=', body_encoding_12517, SHORTEST_12518)
        
        assert_12520 = result_ne_12519
        # Assigning a type to the variable 'assert_12520' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'assert_12520', result_ne_12519)
        
        # Getting the type of 'self' (line 263)
        self_12521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 11), 'self')
        # Obtaining the member 'body_encoding' of a type (line 263)
        body_encoding_12522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 11), self_12521, 'body_encoding')
        # Getting the type of 'QP' (line 263)
        QP_12523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'QP')
        # Applying the binary operator '==' (line 263)
        result_eq_12524 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 11), '==', body_encoding_12522, QP_12523)
        
        # Testing if the type of an if condition is none (line 263)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 263, 8), result_eq_12524):
            
            # Getting the type of 'self' (line 265)
            self_12527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'self')
            # Obtaining the member 'body_encoding' of a type (line 265)
            body_encoding_12528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 13), self_12527, 'body_encoding')
            # Getting the type of 'BASE64' (line 265)
            BASE64_12529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 35), 'BASE64')
            # Applying the binary operator '==' (line 265)
            result_eq_12530 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 13), '==', body_encoding_12528, BASE64_12529)
            
            # Testing if the type of an if condition is none (line 265)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 265, 13), result_eq_12530):
                # Getting the type of 'encode_7or8bit' (line 268)
                encode_7or8bit_12533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'encode_7or8bit')
                # Assigning a type to the variable 'stypy_return_type' (line 268)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'stypy_return_type', encode_7or8bit_12533)
            else:
                
                # Testing the type of an if condition (line 265)
                if_condition_12531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 13), result_eq_12530)
                # Assigning a type to the variable 'if_condition_12531' (line 265)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'if_condition_12531', if_condition_12531)
                # SSA begins for if statement (line 265)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                str_12532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 19), 'str', 'base64')
                # Assigning a type to the variable 'stypy_return_type' (line 266)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'stypy_return_type', str_12532)
                # SSA branch for the else part of an if statement (line 265)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'encode_7or8bit' (line 268)
                encode_7or8bit_12533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'encode_7or8bit')
                # Assigning a type to the variable 'stypy_return_type' (line 268)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'stypy_return_type', encode_7or8bit_12533)
                # SSA join for if statement (line 265)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 263)
            if_condition_12525 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 8), result_eq_12524)
            # Assigning a type to the variable 'if_condition_12525' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'if_condition_12525', if_condition_12525)
            # SSA begins for if statement (line 263)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_12526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 19), 'str', 'quoted-printable')
            # Assigning a type to the variable 'stypy_return_type' (line 264)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'stypy_return_type', str_12526)
            # SSA branch for the else part of an if statement (line 263)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'self' (line 265)
            self_12527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'self')
            # Obtaining the member 'body_encoding' of a type (line 265)
            body_encoding_12528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 13), self_12527, 'body_encoding')
            # Getting the type of 'BASE64' (line 265)
            BASE64_12529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 35), 'BASE64')
            # Applying the binary operator '==' (line 265)
            result_eq_12530 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 13), '==', body_encoding_12528, BASE64_12529)
            
            # Testing if the type of an if condition is none (line 265)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 265, 13), result_eq_12530):
                # Getting the type of 'encode_7or8bit' (line 268)
                encode_7or8bit_12533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'encode_7or8bit')
                # Assigning a type to the variable 'stypy_return_type' (line 268)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'stypy_return_type', encode_7or8bit_12533)
            else:
                
                # Testing the type of an if condition (line 265)
                if_condition_12531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 13), result_eq_12530)
                # Assigning a type to the variable 'if_condition_12531' (line 265)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'if_condition_12531', if_condition_12531)
                # SSA begins for if statement (line 265)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                str_12532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 19), 'str', 'base64')
                # Assigning a type to the variable 'stypy_return_type' (line 266)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'stypy_return_type', str_12532)
                # SSA branch for the else part of an if statement (line 265)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'encode_7or8bit' (line 268)
                encode_7or8bit_12533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'encode_7or8bit')
                # Assigning a type to the variable 'stypy_return_type' (line 268)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'stypy_return_type', encode_7or8bit_12533)
                # SSA join for if statement (line 265)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 263)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'get_body_encoding(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_body_encoding' in the type store
        # Getting the type of 'stypy_return_type' (line 249)
        stypy_return_type_12534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12534)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_body_encoding'
        return stypy_return_type_12534


    @norecursion
    def convert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert'
        module_type_store = module_type_store.open_function_context('convert', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.convert.__dict__.__setitem__('stypy_localization', localization)
        Charset.convert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.convert.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.convert.__dict__.__setitem__('stypy_function_name', 'Charset.convert')
        Charset.convert.__dict__.__setitem__('stypy_param_names_list', ['s'])
        Charset.convert.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.convert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.convert.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.convert.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.convert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.convert.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.convert', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert(...)' code ##################

        str_12535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 8), 'str', 'Convert a string from the input_codec to the output_codec.')
        
        # Getting the type of 'self' (line 272)
        self_12536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'self')
        # Obtaining the member 'input_codec' of a type (line 272)
        input_codec_12537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 11), self_12536, 'input_codec')
        # Getting the type of 'self' (line 272)
        self_12538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 31), 'self')
        # Obtaining the member 'output_codec' of a type (line 272)
        output_codec_12539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 31), self_12538, 'output_codec')
        # Applying the binary operator '!=' (line 272)
        result_ne_12540 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 11), '!=', input_codec_12537, output_codec_12539)
        
        # Testing if the type of an if condition is none (line 272)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 272, 8), result_ne_12540):
            # Getting the type of 's' (line 275)
            s_12553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 's')
            # Assigning a type to the variable 'stypy_return_type' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'stypy_return_type', s_12553)
        else:
            
            # Testing the type of an if condition (line 272)
            if_condition_12541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 8), result_ne_12540)
            # Assigning a type to the variable 'if_condition_12541' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'if_condition_12541', if_condition_12541)
            # SSA begins for if statement (line 272)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to encode(...): (line 273)
            # Processing the call arguments (line 273)
            # Getting the type of 'self' (line 273)
            self_12549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 55), 'self', False)
            # Obtaining the member 'output_codec' of a type (line 273)
            output_codec_12550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 55), self_12549, 'output_codec')
            # Processing the call keyword arguments (line 273)
            kwargs_12551 = {}
            
            # Call to unicode(...): (line 273)
            # Processing the call arguments (line 273)
            # Getting the type of 's' (line 273)
            s_12543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 27), 's', False)
            # Getting the type of 'self' (line 273)
            self_12544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 30), 'self', False)
            # Obtaining the member 'input_codec' of a type (line 273)
            input_codec_12545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 30), self_12544, 'input_codec')
            # Processing the call keyword arguments (line 273)
            kwargs_12546 = {}
            # Getting the type of 'unicode' (line 273)
            unicode_12542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 19), 'unicode', False)
            # Calling unicode(args, kwargs) (line 273)
            unicode_call_result_12547 = invoke(stypy.reporting.localization.Localization(__file__, 273, 19), unicode_12542, *[s_12543, input_codec_12545], **kwargs_12546)
            
            # Obtaining the member 'encode' of a type (line 273)
            encode_12548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 19), unicode_call_result_12547, 'encode')
            # Calling encode(args, kwargs) (line 273)
            encode_call_result_12552 = invoke(stypy.reporting.localization.Localization(__file__, 273, 19), encode_12548, *[output_codec_12550], **kwargs_12551)
            
            # Assigning a type to the variable 'stypy_return_type' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'stypy_return_type', encode_call_result_12552)
            # SSA branch for the else part of an if statement (line 272)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 's' (line 275)
            s_12553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 's')
            # Assigning a type to the variable 'stypy_return_type' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'stypy_return_type', s_12553)
            # SSA join for if statement (line 272)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_12554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12554)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_12554


    @norecursion
    def to_splittable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'to_splittable'
        module_type_store = module_type_store.open_function_context('to_splittable', 277, 4, False)
        # Assigning a type to the variable 'self' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.to_splittable.__dict__.__setitem__('stypy_localization', localization)
        Charset.to_splittable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.to_splittable.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.to_splittable.__dict__.__setitem__('stypy_function_name', 'Charset.to_splittable')
        Charset.to_splittable.__dict__.__setitem__('stypy_param_names_list', ['s'])
        Charset.to_splittable.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.to_splittable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.to_splittable.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.to_splittable.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.to_splittable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.to_splittable.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.to_splittable', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'to_splittable', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'to_splittable(...)' code ##################

        str_12555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, (-1)), 'str', "Convert a possibly multibyte string to a safely splittable format.\n\n        Uses the input_codec to try and convert the string to Unicode, so it\n        can be safely split on character boundaries (even for multibyte\n        characters).\n\n        Returns the string as-is if it isn't known how to convert it to\n        Unicode with the input_charset.\n\n        Characters that could not be converted to Unicode will be replaced\n        with the Unicode replacement character U+FFFD.\n        ")
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 's' (line 290)
        s_12557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 's', False)
        # Getting the type of 'unicode' (line 290)
        unicode_12558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 25), 'unicode', False)
        # Processing the call keyword arguments (line 290)
        kwargs_12559 = {}
        # Getting the type of 'isinstance' (line 290)
        isinstance_12556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 290)
        isinstance_call_result_12560 = invoke(stypy.reporting.localization.Localization(__file__, 290, 11), isinstance_12556, *[s_12557, unicode_12558], **kwargs_12559)
        
        
        # Getting the type of 'self' (line 290)
        self_12561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 37), 'self')
        # Obtaining the member 'input_codec' of a type (line 290)
        input_codec_12562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 37), self_12561, 'input_codec')
        # Getting the type of 'None' (line 290)
        None_12563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 57), 'None')
        # Applying the binary operator 'is' (line 290)
        result_is__12564 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 37), 'is', input_codec_12562, None_12563)
        
        # Applying the binary operator 'or' (line 290)
        result_or_keyword_12565 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 11), 'or', isinstance_call_result_12560, result_is__12564)
        
        # Testing if the type of an if condition is none (line 290)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 290, 8), result_or_keyword_12565):
            pass
        else:
            
            # Testing the type of an if condition (line 290)
            if_condition_12566 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), result_or_keyword_12565)
            # Assigning a type to the variable 'if_condition_12566' (line 290)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_12566', if_condition_12566)
            # SSA begins for if statement (line 290)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 's' (line 291)
            s_12567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 19), 's')
            # Assigning a type to the variable 'stypy_return_type' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'stypy_return_type', s_12567)
            # SSA join for if statement (line 290)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # SSA begins for try-except statement (line 292)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to unicode(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 's' (line 293)
        s_12569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 27), 's', False)
        # Getting the type of 'self' (line 293)
        self_12570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'self', False)
        # Obtaining the member 'input_codec' of a type (line 293)
        input_codec_12571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 30), self_12570, 'input_codec')
        str_12572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 48), 'str', 'replace')
        # Processing the call keyword arguments (line 293)
        kwargs_12573 = {}
        # Getting the type of 'unicode' (line 293)
        unicode_12568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'unicode', False)
        # Calling unicode(args, kwargs) (line 293)
        unicode_call_result_12574 = invoke(stypy.reporting.localization.Localization(__file__, 293, 19), unicode_12568, *[s_12569, input_codec_12571, str_12572], **kwargs_12573)
        
        # Assigning a type to the variable 'stypy_return_type' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'stypy_return_type', unicode_call_result_12574)
        # SSA branch for the except part of a try statement (line 292)
        # SSA branch for the except 'LookupError' branch of a try statement (line 292)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 's' (line 297)
        s_12575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'stypy_return_type', s_12575)
        # SSA join for try-except statement (line 292)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'to_splittable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'to_splittable' in the type store
        # Getting the type of 'stypy_return_type' (line 277)
        stypy_return_type_12576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12576)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'to_splittable'
        return stypy_return_type_12576


    @norecursion
    def from_splittable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 299)
        True_12577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 46), 'True')
        defaults = [True_12577]
        # Create a new context for function 'from_splittable'
        module_type_store = module_type_store.open_function_context('from_splittable', 299, 4, False)
        # Assigning a type to the variable 'self' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.from_splittable.__dict__.__setitem__('stypy_localization', localization)
        Charset.from_splittable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.from_splittable.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.from_splittable.__dict__.__setitem__('stypy_function_name', 'Charset.from_splittable')
        Charset.from_splittable.__dict__.__setitem__('stypy_param_names_list', ['ustr', 'to_output'])
        Charset.from_splittable.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.from_splittable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.from_splittable.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.from_splittable.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.from_splittable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.from_splittable.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.from_splittable', ['ustr', 'to_output'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_splittable', localization, ['ustr', 'to_output'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_splittable(...)' code ##################

        str_12578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, (-1)), 'str', "Convert a splittable string back into an encoded string.\n\n        Uses the proper codec to try and convert the string from Unicode back\n        into an encoded format.  Return the string as-is if it is not Unicode,\n        or if it could not be converted from Unicode.\n\n        Characters that could not be converted from Unicode will be replaced\n        with an appropriate character (usually '?').\n\n        If to_output is True (the default), uses output_codec to convert to an\n        encoded format.  If to_output is False, uses input_codec.\n        ")
        # Getting the type of 'to_output' (line 312)
        to_output_12579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 'to_output')
        # Testing if the type of an if condition is none (line 312)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 312, 8), to_output_12579):
            
            # Assigning a Attribute to a Name (line 315):
            
            # Assigning a Attribute to a Name (line 315):
            # Getting the type of 'self' (line 315)
            self_12583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 20), 'self')
            # Obtaining the member 'input_codec' of a type (line 315)
            input_codec_12584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 20), self_12583, 'input_codec')
            # Assigning a type to the variable 'codec' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'codec', input_codec_12584)
        else:
            
            # Testing the type of an if condition (line 312)
            if_condition_12580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 8), to_output_12579)
            # Assigning a type to the variable 'if_condition_12580' (line 312)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'if_condition_12580', if_condition_12580)
            # SSA begins for if statement (line 312)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 313):
            
            # Assigning a Attribute to a Name (line 313):
            # Getting the type of 'self' (line 313)
            self_12581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 20), 'self')
            # Obtaining the member 'output_codec' of a type (line 313)
            output_codec_12582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 20), self_12581, 'output_codec')
            # Assigning a type to the variable 'codec' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'codec', output_codec_12582)
            # SSA branch for the else part of an if statement (line 312)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 315):
            
            # Assigning a Attribute to a Name (line 315):
            # Getting the type of 'self' (line 315)
            self_12583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 20), 'self')
            # Obtaining the member 'input_codec' of a type (line 315)
            input_codec_12584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 20), self_12583, 'input_codec')
            # Assigning a type to the variable 'codec' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'codec', input_codec_12584)
            # SSA join for if statement (line 312)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        
        # Call to isinstance(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'ustr' (line 316)
        ustr_12586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 26), 'ustr', False)
        # Getting the type of 'unicode' (line 316)
        unicode_12587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 32), 'unicode', False)
        # Processing the call keyword arguments (line 316)
        kwargs_12588 = {}
        # Getting the type of 'isinstance' (line 316)
        isinstance_12585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 316)
        isinstance_call_result_12589 = invoke(stypy.reporting.localization.Localization(__file__, 316, 15), isinstance_12585, *[ustr_12586, unicode_12587], **kwargs_12588)
        
        # Applying the 'not' unary operator (line 316)
        result_not__12590 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 11), 'not', isinstance_call_result_12589)
        
        
        # Getting the type of 'codec' (line 316)
        codec_12591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 44), 'codec')
        # Getting the type of 'None' (line 316)
        None_12592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 53), 'None')
        # Applying the binary operator 'is' (line 316)
        result_is__12593 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 44), 'is', codec_12591, None_12592)
        
        # Applying the binary operator 'or' (line 316)
        result_or_keyword_12594 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 11), 'or', result_not__12590, result_is__12593)
        
        # Testing if the type of an if condition is none (line 316)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 316, 8), result_or_keyword_12594):
            pass
        else:
            
            # Testing the type of an if condition (line 316)
            if_condition_12595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 8), result_or_keyword_12594)
            # Assigning a type to the variable 'if_condition_12595' (line 316)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'if_condition_12595', if_condition_12595)
            # SSA begins for if statement (line 316)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'ustr' (line 317)
            ustr_12596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'ustr')
            # Assigning a type to the variable 'stypy_return_type' (line 317)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'stypy_return_type', ustr_12596)
            # SSA join for if statement (line 316)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # SSA begins for try-except statement (line 318)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to encode(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'codec' (line 319)
        codec_12599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 31), 'codec', False)
        str_12600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 38), 'str', 'replace')
        # Processing the call keyword arguments (line 319)
        kwargs_12601 = {}
        # Getting the type of 'ustr' (line 319)
        ustr_12597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'ustr', False)
        # Obtaining the member 'encode' of a type (line 319)
        encode_12598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 19), ustr_12597, 'encode')
        # Calling encode(args, kwargs) (line 319)
        encode_call_result_12602 = invoke(stypy.reporting.localization.Localization(__file__, 319, 19), encode_12598, *[codec_12599, str_12600], **kwargs_12601)
        
        # Assigning a type to the variable 'stypy_return_type' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'stypy_return_type', encode_call_result_12602)
        # SSA branch for the except part of a try statement (line 318)
        # SSA branch for the except 'LookupError' branch of a try statement (line 318)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'ustr' (line 322)
        ustr_12603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 19), 'ustr')
        # Assigning a type to the variable 'stypy_return_type' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'stypy_return_type', ustr_12603)
        # SSA join for try-except statement (line 318)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'from_splittable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_splittable' in the type store
        # Getting the type of 'stypy_return_type' (line 299)
        stypy_return_type_12604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12604)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_splittable'
        return stypy_return_type_12604


    @norecursion
    def get_output_charset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_output_charset'
        module_type_store = module_type_store.open_function_context('get_output_charset', 324, 4, False)
        # Assigning a type to the variable 'self' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.get_output_charset.__dict__.__setitem__('stypy_localization', localization)
        Charset.get_output_charset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.get_output_charset.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.get_output_charset.__dict__.__setitem__('stypy_function_name', 'Charset.get_output_charset')
        Charset.get_output_charset.__dict__.__setitem__('stypy_param_names_list', [])
        Charset.get_output_charset.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.get_output_charset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.get_output_charset.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.get_output_charset.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.get_output_charset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.get_output_charset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.get_output_charset', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_output_charset', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_output_charset(...)' code ##################

        str_12605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, (-1)), 'str', 'Return the output character set.\n\n        This is self.output_charset if that is not None, otherwise it is\n        self.input_charset.\n        ')
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 330)
        self_12606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 15), 'self')
        # Obtaining the member 'output_charset' of a type (line 330)
        output_charset_12607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 15), self_12606, 'output_charset')
        # Getting the type of 'self' (line 330)
        self_12608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 38), 'self')
        # Obtaining the member 'input_charset' of a type (line 330)
        input_charset_12609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 38), self_12608, 'input_charset')
        # Applying the binary operator 'or' (line 330)
        result_or_keyword_12610 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 15), 'or', output_charset_12607, input_charset_12609)
        
        # Assigning a type to the variable 'stypy_return_type' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'stypy_return_type', result_or_keyword_12610)
        
        # ################# End of 'get_output_charset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_output_charset' in the type store
        # Getting the type of 'stypy_return_type' (line 324)
        stypy_return_type_12611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_output_charset'
        return stypy_return_type_12611


    @norecursion
    def encoded_header_len(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'encoded_header_len'
        module_type_store = module_type_store.open_function_context('encoded_header_len', 332, 4, False)
        # Assigning a type to the variable 'self' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.encoded_header_len.__dict__.__setitem__('stypy_localization', localization)
        Charset.encoded_header_len.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.encoded_header_len.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.encoded_header_len.__dict__.__setitem__('stypy_function_name', 'Charset.encoded_header_len')
        Charset.encoded_header_len.__dict__.__setitem__('stypy_param_names_list', ['s'])
        Charset.encoded_header_len.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.encoded_header_len.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.encoded_header_len.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.encoded_header_len.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.encoded_header_len.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.encoded_header_len.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.encoded_header_len', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'encoded_header_len', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'encoded_header_len(...)' code ##################

        str_12612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 8), 'str', 'Return the length of the encoded header string.')
        
        # Assigning a Call to a Name (line 334):
        
        # Assigning a Call to a Name (line 334):
        
        # Call to get_output_charset(...): (line 334)
        # Processing the call keyword arguments (line 334)
        kwargs_12615 = {}
        # Getting the type of 'self' (line 334)
        self_12613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 15), 'self', False)
        # Obtaining the member 'get_output_charset' of a type (line 334)
        get_output_charset_12614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 15), self_12613, 'get_output_charset')
        # Calling get_output_charset(args, kwargs) (line 334)
        get_output_charset_call_result_12616 = invoke(stypy.reporting.localization.Localization(__file__, 334, 15), get_output_charset_12614, *[], **kwargs_12615)
        
        # Assigning a type to the variable 'cset' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'cset', get_output_charset_call_result_12616)
        
        # Getting the type of 'self' (line 336)
        self_12617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'self')
        # Obtaining the member 'header_encoding' of a type (line 336)
        header_encoding_12618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 11), self_12617, 'header_encoding')
        # Getting the type of 'BASE64' (line 336)
        BASE64_12619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 35), 'BASE64')
        # Applying the binary operator '==' (line 336)
        result_eq_12620 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 11), '==', header_encoding_12618, BASE64_12619)
        
        # Testing if the type of an if condition is none (line 336)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 336, 8), result_eq_12620):
            
            # Getting the type of 'self' (line 338)
            self_12635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), 'self')
            # Obtaining the member 'header_encoding' of a type (line 338)
            header_encoding_12636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 13), self_12635, 'header_encoding')
            # Getting the type of 'QP' (line 338)
            QP_12637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 37), 'QP')
            # Applying the binary operator '==' (line 338)
            result_eq_12638 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 13), '==', header_encoding_12636, QP_12637)
            
            # Testing if the type of an if condition is none (line 338)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 338, 13), result_eq_12638):
                
                # Getting the type of 'self' (line 340)
                self_12653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'self')
                # Obtaining the member 'header_encoding' of a type (line 340)
                header_encoding_12654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 13), self_12653, 'header_encoding')
                # Getting the type of 'SHORTEST' (line 340)
                SHORTEST_12655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 37), 'SHORTEST')
                # Applying the binary operator '==' (line 340)
                result_eq_12656 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 13), '==', header_encoding_12654, SHORTEST_12655)
                
                # Testing if the type of an if condition is none (line 340)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 340, 13), result_eq_12656):
                    
                    # Call to len(...): (line 345)
                    # Processing the call arguments (line 345)
                    # Getting the type of 's' (line 345)
                    s_12683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 's', False)
                    # Processing the call keyword arguments (line 345)
                    kwargs_12684 = {}
                    # Getting the type of 'len' (line 345)
                    len_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'len', False)
                    # Calling len(args, kwargs) (line 345)
                    len_call_result_12685 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), len_12682, *[s_12683], **kwargs_12684)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 345)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', len_call_result_12685)
                else:
                    
                    # Testing the type of an if condition (line 340)
                    if_condition_12657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 13), result_eq_12656)
                    # Assigning a type to the variable 'if_condition_12657' (line 340)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'if_condition_12657', if_condition_12657)
                    # SSA begins for if statement (line 340)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 341):
                    
                    # Assigning a Call to a Name (line 341):
                    
                    # Call to base64_len(...): (line 341)
                    # Processing the call arguments (line 341)
                    # Getting the type of 's' (line 341)
                    s_12661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 49), 's', False)
                    # Processing the call keyword arguments (line 341)
                    kwargs_12662 = {}
                    # Getting the type of 'email' (line 341)
                    email_12658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 21), 'email', False)
                    # Obtaining the member 'base64mime' of a type (line 341)
                    base64mime_12659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 21), email_12658, 'base64mime')
                    # Obtaining the member 'base64_len' of a type (line 341)
                    base64_len_12660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 21), base64mime_12659, 'base64_len')
                    # Calling base64_len(args, kwargs) (line 341)
                    base64_len_call_result_12663 = invoke(stypy.reporting.localization.Localization(__file__, 341, 21), base64_len_12660, *[s_12661], **kwargs_12662)
                    
                    # Assigning a type to the variable 'lenb64' (line 341)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'lenb64', base64_len_call_result_12663)
                    
                    # Assigning a Call to a Name (line 342):
                    
                    # Assigning a Call to a Name (line 342):
                    
                    # Call to header_quopri_len(...): (line 342)
                    # Processing the call arguments (line 342)
                    # Getting the type of 's' (line 342)
                    s_12667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 55), 's', False)
                    # Processing the call keyword arguments (line 342)
                    kwargs_12668 = {}
                    # Getting the type of 'email' (line 342)
                    email_12664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'email', False)
                    # Obtaining the member 'quoprimime' of a type (line 342)
                    quoprimime_12665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 20), email_12664, 'quoprimime')
                    # Obtaining the member 'header_quopri_len' of a type (line 342)
                    header_quopri_len_12666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 20), quoprimime_12665, 'header_quopri_len')
                    # Calling header_quopri_len(args, kwargs) (line 342)
                    header_quopri_len_call_result_12669 = invoke(stypy.reporting.localization.Localization(__file__, 342, 20), header_quopri_len_12666, *[s_12667], **kwargs_12668)
                    
                    # Assigning a type to the variable 'lenqp' (line 342)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'lenqp', header_quopri_len_call_result_12669)
                    
                    # Call to min(...): (line 343)
                    # Processing the call arguments (line 343)
                    # Getting the type of 'lenb64' (line 343)
                    lenb64_12671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 23), 'lenb64', False)
                    # Getting the type of 'lenqp' (line 343)
                    lenqp_12672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 31), 'lenqp', False)
                    # Processing the call keyword arguments (line 343)
                    kwargs_12673 = {}
                    # Getting the type of 'min' (line 343)
                    min_12670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'min', False)
                    # Calling min(args, kwargs) (line 343)
                    min_call_result_12674 = invoke(stypy.reporting.localization.Localization(__file__, 343, 19), min_12670, *[lenb64_12671, lenqp_12672], **kwargs_12673)
                    
                    
                    # Call to len(...): (line 343)
                    # Processing the call arguments (line 343)
                    # Getting the type of 'cset' (line 343)
                    cset_12676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 44), 'cset', False)
                    # Processing the call keyword arguments (line 343)
                    kwargs_12677 = {}
                    # Getting the type of 'len' (line 343)
                    len_12675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 40), 'len', False)
                    # Calling len(args, kwargs) (line 343)
                    len_call_result_12678 = invoke(stypy.reporting.localization.Localization(__file__, 343, 40), len_12675, *[cset_12676], **kwargs_12677)
                    
                    # Applying the binary operator '+' (line 343)
                    result_add_12679 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 19), '+', min_call_result_12674, len_call_result_12678)
                    
                    # Getting the type of 'MISC_LEN' (line 343)
                    MISC_LEN_12680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 52), 'MISC_LEN')
                    # Applying the binary operator '+' (line 343)
                    result_add_12681 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 50), '+', result_add_12679, MISC_LEN_12680)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 343)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'stypy_return_type', result_add_12681)
                    # SSA branch for the else part of an if statement (line 340)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to len(...): (line 345)
                    # Processing the call arguments (line 345)
                    # Getting the type of 's' (line 345)
                    s_12683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 's', False)
                    # Processing the call keyword arguments (line 345)
                    kwargs_12684 = {}
                    # Getting the type of 'len' (line 345)
                    len_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'len', False)
                    # Calling len(args, kwargs) (line 345)
                    len_call_result_12685 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), len_12682, *[s_12683], **kwargs_12684)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 345)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', len_call_result_12685)
                    # SSA join for if statement (line 340)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 338)
                if_condition_12639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 13), result_eq_12638)
                # Assigning a type to the variable 'if_condition_12639' (line 338)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), 'if_condition_12639', if_condition_12639)
                # SSA begins for if statement (line 338)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to header_quopri_len(...): (line 339)
                # Processing the call arguments (line 339)
                # Getting the type of 's' (line 339)
                s_12643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 54), 's', False)
                # Processing the call keyword arguments (line 339)
                kwargs_12644 = {}
                # Getting the type of 'email' (line 339)
                email_12640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'email', False)
                # Obtaining the member 'quoprimime' of a type (line 339)
                quoprimime_12641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 19), email_12640, 'quoprimime')
                # Obtaining the member 'header_quopri_len' of a type (line 339)
                header_quopri_len_12642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 19), quoprimime_12641, 'header_quopri_len')
                # Calling header_quopri_len(args, kwargs) (line 339)
                header_quopri_len_call_result_12645 = invoke(stypy.reporting.localization.Localization(__file__, 339, 19), header_quopri_len_12642, *[s_12643], **kwargs_12644)
                
                
                # Call to len(...): (line 339)
                # Processing the call arguments (line 339)
                # Getting the type of 'cset' (line 339)
                cset_12647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 63), 'cset', False)
                # Processing the call keyword arguments (line 339)
                kwargs_12648 = {}
                # Getting the type of 'len' (line 339)
                len_12646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 59), 'len', False)
                # Calling len(args, kwargs) (line 339)
                len_call_result_12649 = invoke(stypy.reporting.localization.Localization(__file__, 339, 59), len_12646, *[cset_12647], **kwargs_12648)
                
                # Applying the binary operator '+' (line 339)
                result_add_12650 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 19), '+', header_quopri_len_call_result_12645, len_call_result_12649)
                
                # Getting the type of 'MISC_LEN' (line 339)
                MISC_LEN_12651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 71), 'MISC_LEN')
                # Applying the binary operator '+' (line 339)
                result_add_12652 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 69), '+', result_add_12650, MISC_LEN_12651)
                
                # Assigning a type to the variable 'stypy_return_type' (line 339)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'stypy_return_type', result_add_12652)
                # SSA branch for the else part of an if statement (line 338)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'self' (line 340)
                self_12653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'self')
                # Obtaining the member 'header_encoding' of a type (line 340)
                header_encoding_12654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 13), self_12653, 'header_encoding')
                # Getting the type of 'SHORTEST' (line 340)
                SHORTEST_12655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 37), 'SHORTEST')
                # Applying the binary operator '==' (line 340)
                result_eq_12656 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 13), '==', header_encoding_12654, SHORTEST_12655)
                
                # Testing if the type of an if condition is none (line 340)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 340, 13), result_eq_12656):
                    
                    # Call to len(...): (line 345)
                    # Processing the call arguments (line 345)
                    # Getting the type of 's' (line 345)
                    s_12683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 's', False)
                    # Processing the call keyword arguments (line 345)
                    kwargs_12684 = {}
                    # Getting the type of 'len' (line 345)
                    len_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'len', False)
                    # Calling len(args, kwargs) (line 345)
                    len_call_result_12685 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), len_12682, *[s_12683], **kwargs_12684)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 345)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', len_call_result_12685)
                else:
                    
                    # Testing the type of an if condition (line 340)
                    if_condition_12657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 13), result_eq_12656)
                    # Assigning a type to the variable 'if_condition_12657' (line 340)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'if_condition_12657', if_condition_12657)
                    # SSA begins for if statement (line 340)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 341):
                    
                    # Assigning a Call to a Name (line 341):
                    
                    # Call to base64_len(...): (line 341)
                    # Processing the call arguments (line 341)
                    # Getting the type of 's' (line 341)
                    s_12661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 49), 's', False)
                    # Processing the call keyword arguments (line 341)
                    kwargs_12662 = {}
                    # Getting the type of 'email' (line 341)
                    email_12658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 21), 'email', False)
                    # Obtaining the member 'base64mime' of a type (line 341)
                    base64mime_12659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 21), email_12658, 'base64mime')
                    # Obtaining the member 'base64_len' of a type (line 341)
                    base64_len_12660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 21), base64mime_12659, 'base64_len')
                    # Calling base64_len(args, kwargs) (line 341)
                    base64_len_call_result_12663 = invoke(stypy.reporting.localization.Localization(__file__, 341, 21), base64_len_12660, *[s_12661], **kwargs_12662)
                    
                    # Assigning a type to the variable 'lenb64' (line 341)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'lenb64', base64_len_call_result_12663)
                    
                    # Assigning a Call to a Name (line 342):
                    
                    # Assigning a Call to a Name (line 342):
                    
                    # Call to header_quopri_len(...): (line 342)
                    # Processing the call arguments (line 342)
                    # Getting the type of 's' (line 342)
                    s_12667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 55), 's', False)
                    # Processing the call keyword arguments (line 342)
                    kwargs_12668 = {}
                    # Getting the type of 'email' (line 342)
                    email_12664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'email', False)
                    # Obtaining the member 'quoprimime' of a type (line 342)
                    quoprimime_12665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 20), email_12664, 'quoprimime')
                    # Obtaining the member 'header_quopri_len' of a type (line 342)
                    header_quopri_len_12666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 20), quoprimime_12665, 'header_quopri_len')
                    # Calling header_quopri_len(args, kwargs) (line 342)
                    header_quopri_len_call_result_12669 = invoke(stypy.reporting.localization.Localization(__file__, 342, 20), header_quopri_len_12666, *[s_12667], **kwargs_12668)
                    
                    # Assigning a type to the variable 'lenqp' (line 342)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'lenqp', header_quopri_len_call_result_12669)
                    
                    # Call to min(...): (line 343)
                    # Processing the call arguments (line 343)
                    # Getting the type of 'lenb64' (line 343)
                    lenb64_12671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 23), 'lenb64', False)
                    # Getting the type of 'lenqp' (line 343)
                    lenqp_12672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 31), 'lenqp', False)
                    # Processing the call keyword arguments (line 343)
                    kwargs_12673 = {}
                    # Getting the type of 'min' (line 343)
                    min_12670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'min', False)
                    # Calling min(args, kwargs) (line 343)
                    min_call_result_12674 = invoke(stypy.reporting.localization.Localization(__file__, 343, 19), min_12670, *[lenb64_12671, lenqp_12672], **kwargs_12673)
                    
                    
                    # Call to len(...): (line 343)
                    # Processing the call arguments (line 343)
                    # Getting the type of 'cset' (line 343)
                    cset_12676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 44), 'cset', False)
                    # Processing the call keyword arguments (line 343)
                    kwargs_12677 = {}
                    # Getting the type of 'len' (line 343)
                    len_12675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 40), 'len', False)
                    # Calling len(args, kwargs) (line 343)
                    len_call_result_12678 = invoke(stypy.reporting.localization.Localization(__file__, 343, 40), len_12675, *[cset_12676], **kwargs_12677)
                    
                    # Applying the binary operator '+' (line 343)
                    result_add_12679 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 19), '+', min_call_result_12674, len_call_result_12678)
                    
                    # Getting the type of 'MISC_LEN' (line 343)
                    MISC_LEN_12680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 52), 'MISC_LEN')
                    # Applying the binary operator '+' (line 343)
                    result_add_12681 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 50), '+', result_add_12679, MISC_LEN_12680)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 343)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'stypy_return_type', result_add_12681)
                    # SSA branch for the else part of an if statement (line 340)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to len(...): (line 345)
                    # Processing the call arguments (line 345)
                    # Getting the type of 's' (line 345)
                    s_12683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 's', False)
                    # Processing the call keyword arguments (line 345)
                    kwargs_12684 = {}
                    # Getting the type of 'len' (line 345)
                    len_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'len', False)
                    # Calling len(args, kwargs) (line 345)
                    len_call_result_12685 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), len_12682, *[s_12683], **kwargs_12684)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 345)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', len_call_result_12685)
                    # SSA join for if statement (line 340)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 338)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 336)
            if_condition_12621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 8), result_eq_12620)
            # Assigning a type to the variable 'if_condition_12621' (line 336)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'if_condition_12621', if_condition_12621)
            # SSA begins for if statement (line 336)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to base64_len(...): (line 337)
            # Processing the call arguments (line 337)
            # Getting the type of 's' (line 337)
            s_12625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 47), 's', False)
            # Processing the call keyword arguments (line 337)
            kwargs_12626 = {}
            # Getting the type of 'email' (line 337)
            email_12622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 19), 'email', False)
            # Obtaining the member 'base64mime' of a type (line 337)
            base64mime_12623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 19), email_12622, 'base64mime')
            # Obtaining the member 'base64_len' of a type (line 337)
            base64_len_12624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 19), base64mime_12623, 'base64_len')
            # Calling base64_len(args, kwargs) (line 337)
            base64_len_call_result_12627 = invoke(stypy.reporting.localization.Localization(__file__, 337, 19), base64_len_12624, *[s_12625], **kwargs_12626)
            
            
            # Call to len(...): (line 337)
            # Processing the call arguments (line 337)
            # Getting the type of 'cset' (line 337)
            cset_12629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 56), 'cset', False)
            # Processing the call keyword arguments (line 337)
            kwargs_12630 = {}
            # Getting the type of 'len' (line 337)
            len_12628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 52), 'len', False)
            # Calling len(args, kwargs) (line 337)
            len_call_result_12631 = invoke(stypy.reporting.localization.Localization(__file__, 337, 52), len_12628, *[cset_12629], **kwargs_12630)
            
            # Applying the binary operator '+' (line 337)
            result_add_12632 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 19), '+', base64_len_call_result_12627, len_call_result_12631)
            
            # Getting the type of 'MISC_LEN' (line 337)
            MISC_LEN_12633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 64), 'MISC_LEN')
            # Applying the binary operator '+' (line 337)
            result_add_12634 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 62), '+', result_add_12632, MISC_LEN_12633)
            
            # Assigning a type to the variable 'stypy_return_type' (line 337)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'stypy_return_type', result_add_12634)
            # SSA branch for the else part of an if statement (line 336)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'self' (line 338)
            self_12635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), 'self')
            # Obtaining the member 'header_encoding' of a type (line 338)
            header_encoding_12636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 13), self_12635, 'header_encoding')
            # Getting the type of 'QP' (line 338)
            QP_12637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 37), 'QP')
            # Applying the binary operator '==' (line 338)
            result_eq_12638 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 13), '==', header_encoding_12636, QP_12637)
            
            # Testing if the type of an if condition is none (line 338)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 338, 13), result_eq_12638):
                
                # Getting the type of 'self' (line 340)
                self_12653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'self')
                # Obtaining the member 'header_encoding' of a type (line 340)
                header_encoding_12654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 13), self_12653, 'header_encoding')
                # Getting the type of 'SHORTEST' (line 340)
                SHORTEST_12655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 37), 'SHORTEST')
                # Applying the binary operator '==' (line 340)
                result_eq_12656 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 13), '==', header_encoding_12654, SHORTEST_12655)
                
                # Testing if the type of an if condition is none (line 340)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 340, 13), result_eq_12656):
                    
                    # Call to len(...): (line 345)
                    # Processing the call arguments (line 345)
                    # Getting the type of 's' (line 345)
                    s_12683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 's', False)
                    # Processing the call keyword arguments (line 345)
                    kwargs_12684 = {}
                    # Getting the type of 'len' (line 345)
                    len_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'len', False)
                    # Calling len(args, kwargs) (line 345)
                    len_call_result_12685 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), len_12682, *[s_12683], **kwargs_12684)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 345)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', len_call_result_12685)
                else:
                    
                    # Testing the type of an if condition (line 340)
                    if_condition_12657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 13), result_eq_12656)
                    # Assigning a type to the variable 'if_condition_12657' (line 340)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'if_condition_12657', if_condition_12657)
                    # SSA begins for if statement (line 340)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 341):
                    
                    # Assigning a Call to a Name (line 341):
                    
                    # Call to base64_len(...): (line 341)
                    # Processing the call arguments (line 341)
                    # Getting the type of 's' (line 341)
                    s_12661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 49), 's', False)
                    # Processing the call keyword arguments (line 341)
                    kwargs_12662 = {}
                    # Getting the type of 'email' (line 341)
                    email_12658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 21), 'email', False)
                    # Obtaining the member 'base64mime' of a type (line 341)
                    base64mime_12659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 21), email_12658, 'base64mime')
                    # Obtaining the member 'base64_len' of a type (line 341)
                    base64_len_12660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 21), base64mime_12659, 'base64_len')
                    # Calling base64_len(args, kwargs) (line 341)
                    base64_len_call_result_12663 = invoke(stypy.reporting.localization.Localization(__file__, 341, 21), base64_len_12660, *[s_12661], **kwargs_12662)
                    
                    # Assigning a type to the variable 'lenb64' (line 341)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'lenb64', base64_len_call_result_12663)
                    
                    # Assigning a Call to a Name (line 342):
                    
                    # Assigning a Call to a Name (line 342):
                    
                    # Call to header_quopri_len(...): (line 342)
                    # Processing the call arguments (line 342)
                    # Getting the type of 's' (line 342)
                    s_12667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 55), 's', False)
                    # Processing the call keyword arguments (line 342)
                    kwargs_12668 = {}
                    # Getting the type of 'email' (line 342)
                    email_12664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'email', False)
                    # Obtaining the member 'quoprimime' of a type (line 342)
                    quoprimime_12665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 20), email_12664, 'quoprimime')
                    # Obtaining the member 'header_quopri_len' of a type (line 342)
                    header_quopri_len_12666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 20), quoprimime_12665, 'header_quopri_len')
                    # Calling header_quopri_len(args, kwargs) (line 342)
                    header_quopri_len_call_result_12669 = invoke(stypy.reporting.localization.Localization(__file__, 342, 20), header_quopri_len_12666, *[s_12667], **kwargs_12668)
                    
                    # Assigning a type to the variable 'lenqp' (line 342)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'lenqp', header_quopri_len_call_result_12669)
                    
                    # Call to min(...): (line 343)
                    # Processing the call arguments (line 343)
                    # Getting the type of 'lenb64' (line 343)
                    lenb64_12671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 23), 'lenb64', False)
                    # Getting the type of 'lenqp' (line 343)
                    lenqp_12672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 31), 'lenqp', False)
                    # Processing the call keyword arguments (line 343)
                    kwargs_12673 = {}
                    # Getting the type of 'min' (line 343)
                    min_12670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'min', False)
                    # Calling min(args, kwargs) (line 343)
                    min_call_result_12674 = invoke(stypy.reporting.localization.Localization(__file__, 343, 19), min_12670, *[lenb64_12671, lenqp_12672], **kwargs_12673)
                    
                    
                    # Call to len(...): (line 343)
                    # Processing the call arguments (line 343)
                    # Getting the type of 'cset' (line 343)
                    cset_12676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 44), 'cset', False)
                    # Processing the call keyword arguments (line 343)
                    kwargs_12677 = {}
                    # Getting the type of 'len' (line 343)
                    len_12675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 40), 'len', False)
                    # Calling len(args, kwargs) (line 343)
                    len_call_result_12678 = invoke(stypy.reporting.localization.Localization(__file__, 343, 40), len_12675, *[cset_12676], **kwargs_12677)
                    
                    # Applying the binary operator '+' (line 343)
                    result_add_12679 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 19), '+', min_call_result_12674, len_call_result_12678)
                    
                    # Getting the type of 'MISC_LEN' (line 343)
                    MISC_LEN_12680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 52), 'MISC_LEN')
                    # Applying the binary operator '+' (line 343)
                    result_add_12681 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 50), '+', result_add_12679, MISC_LEN_12680)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 343)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'stypy_return_type', result_add_12681)
                    # SSA branch for the else part of an if statement (line 340)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to len(...): (line 345)
                    # Processing the call arguments (line 345)
                    # Getting the type of 's' (line 345)
                    s_12683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 's', False)
                    # Processing the call keyword arguments (line 345)
                    kwargs_12684 = {}
                    # Getting the type of 'len' (line 345)
                    len_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'len', False)
                    # Calling len(args, kwargs) (line 345)
                    len_call_result_12685 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), len_12682, *[s_12683], **kwargs_12684)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 345)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', len_call_result_12685)
                    # SSA join for if statement (line 340)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 338)
                if_condition_12639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 13), result_eq_12638)
                # Assigning a type to the variable 'if_condition_12639' (line 338)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), 'if_condition_12639', if_condition_12639)
                # SSA begins for if statement (line 338)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to header_quopri_len(...): (line 339)
                # Processing the call arguments (line 339)
                # Getting the type of 's' (line 339)
                s_12643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 54), 's', False)
                # Processing the call keyword arguments (line 339)
                kwargs_12644 = {}
                # Getting the type of 'email' (line 339)
                email_12640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'email', False)
                # Obtaining the member 'quoprimime' of a type (line 339)
                quoprimime_12641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 19), email_12640, 'quoprimime')
                # Obtaining the member 'header_quopri_len' of a type (line 339)
                header_quopri_len_12642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 19), quoprimime_12641, 'header_quopri_len')
                # Calling header_quopri_len(args, kwargs) (line 339)
                header_quopri_len_call_result_12645 = invoke(stypy.reporting.localization.Localization(__file__, 339, 19), header_quopri_len_12642, *[s_12643], **kwargs_12644)
                
                
                # Call to len(...): (line 339)
                # Processing the call arguments (line 339)
                # Getting the type of 'cset' (line 339)
                cset_12647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 63), 'cset', False)
                # Processing the call keyword arguments (line 339)
                kwargs_12648 = {}
                # Getting the type of 'len' (line 339)
                len_12646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 59), 'len', False)
                # Calling len(args, kwargs) (line 339)
                len_call_result_12649 = invoke(stypy.reporting.localization.Localization(__file__, 339, 59), len_12646, *[cset_12647], **kwargs_12648)
                
                # Applying the binary operator '+' (line 339)
                result_add_12650 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 19), '+', header_quopri_len_call_result_12645, len_call_result_12649)
                
                # Getting the type of 'MISC_LEN' (line 339)
                MISC_LEN_12651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 71), 'MISC_LEN')
                # Applying the binary operator '+' (line 339)
                result_add_12652 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 69), '+', result_add_12650, MISC_LEN_12651)
                
                # Assigning a type to the variable 'stypy_return_type' (line 339)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'stypy_return_type', result_add_12652)
                # SSA branch for the else part of an if statement (line 338)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'self' (line 340)
                self_12653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'self')
                # Obtaining the member 'header_encoding' of a type (line 340)
                header_encoding_12654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 13), self_12653, 'header_encoding')
                # Getting the type of 'SHORTEST' (line 340)
                SHORTEST_12655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 37), 'SHORTEST')
                # Applying the binary operator '==' (line 340)
                result_eq_12656 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 13), '==', header_encoding_12654, SHORTEST_12655)
                
                # Testing if the type of an if condition is none (line 340)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 340, 13), result_eq_12656):
                    
                    # Call to len(...): (line 345)
                    # Processing the call arguments (line 345)
                    # Getting the type of 's' (line 345)
                    s_12683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 's', False)
                    # Processing the call keyword arguments (line 345)
                    kwargs_12684 = {}
                    # Getting the type of 'len' (line 345)
                    len_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'len', False)
                    # Calling len(args, kwargs) (line 345)
                    len_call_result_12685 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), len_12682, *[s_12683], **kwargs_12684)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 345)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', len_call_result_12685)
                else:
                    
                    # Testing the type of an if condition (line 340)
                    if_condition_12657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 13), result_eq_12656)
                    # Assigning a type to the variable 'if_condition_12657' (line 340)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'if_condition_12657', if_condition_12657)
                    # SSA begins for if statement (line 340)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 341):
                    
                    # Assigning a Call to a Name (line 341):
                    
                    # Call to base64_len(...): (line 341)
                    # Processing the call arguments (line 341)
                    # Getting the type of 's' (line 341)
                    s_12661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 49), 's', False)
                    # Processing the call keyword arguments (line 341)
                    kwargs_12662 = {}
                    # Getting the type of 'email' (line 341)
                    email_12658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 21), 'email', False)
                    # Obtaining the member 'base64mime' of a type (line 341)
                    base64mime_12659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 21), email_12658, 'base64mime')
                    # Obtaining the member 'base64_len' of a type (line 341)
                    base64_len_12660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 21), base64mime_12659, 'base64_len')
                    # Calling base64_len(args, kwargs) (line 341)
                    base64_len_call_result_12663 = invoke(stypy.reporting.localization.Localization(__file__, 341, 21), base64_len_12660, *[s_12661], **kwargs_12662)
                    
                    # Assigning a type to the variable 'lenb64' (line 341)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'lenb64', base64_len_call_result_12663)
                    
                    # Assigning a Call to a Name (line 342):
                    
                    # Assigning a Call to a Name (line 342):
                    
                    # Call to header_quopri_len(...): (line 342)
                    # Processing the call arguments (line 342)
                    # Getting the type of 's' (line 342)
                    s_12667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 55), 's', False)
                    # Processing the call keyword arguments (line 342)
                    kwargs_12668 = {}
                    # Getting the type of 'email' (line 342)
                    email_12664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'email', False)
                    # Obtaining the member 'quoprimime' of a type (line 342)
                    quoprimime_12665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 20), email_12664, 'quoprimime')
                    # Obtaining the member 'header_quopri_len' of a type (line 342)
                    header_quopri_len_12666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 20), quoprimime_12665, 'header_quopri_len')
                    # Calling header_quopri_len(args, kwargs) (line 342)
                    header_quopri_len_call_result_12669 = invoke(stypy.reporting.localization.Localization(__file__, 342, 20), header_quopri_len_12666, *[s_12667], **kwargs_12668)
                    
                    # Assigning a type to the variable 'lenqp' (line 342)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'lenqp', header_quopri_len_call_result_12669)
                    
                    # Call to min(...): (line 343)
                    # Processing the call arguments (line 343)
                    # Getting the type of 'lenb64' (line 343)
                    lenb64_12671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 23), 'lenb64', False)
                    # Getting the type of 'lenqp' (line 343)
                    lenqp_12672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 31), 'lenqp', False)
                    # Processing the call keyword arguments (line 343)
                    kwargs_12673 = {}
                    # Getting the type of 'min' (line 343)
                    min_12670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'min', False)
                    # Calling min(args, kwargs) (line 343)
                    min_call_result_12674 = invoke(stypy.reporting.localization.Localization(__file__, 343, 19), min_12670, *[lenb64_12671, lenqp_12672], **kwargs_12673)
                    
                    
                    # Call to len(...): (line 343)
                    # Processing the call arguments (line 343)
                    # Getting the type of 'cset' (line 343)
                    cset_12676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 44), 'cset', False)
                    # Processing the call keyword arguments (line 343)
                    kwargs_12677 = {}
                    # Getting the type of 'len' (line 343)
                    len_12675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 40), 'len', False)
                    # Calling len(args, kwargs) (line 343)
                    len_call_result_12678 = invoke(stypy.reporting.localization.Localization(__file__, 343, 40), len_12675, *[cset_12676], **kwargs_12677)
                    
                    # Applying the binary operator '+' (line 343)
                    result_add_12679 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 19), '+', min_call_result_12674, len_call_result_12678)
                    
                    # Getting the type of 'MISC_LEN' (line 343)
                    MISC_LEN_12680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 52), 'MISC_LEN')
                    # Applying the binary operator '+' (line 343)
                    result_add_12681 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 50), '+', result_add_12679, MISC_LEN_12680)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 343)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'stypy_return_type', result_add_12681)
                    # SSA branch for the else part of an if statement (line 340)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to len(...): (line 345)
                    # Processing the call arguments (line 345)
                    # Getting the type of 's' (line 345)
                    s_12683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 's', False)
                    # Processing the call keyword arguments (line 345)
                    kwargs_12684 = {}
                    # Getting the type of 'len' (line 345)
                    len_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'len', False)
                    # Calling len(args, kwargs) (line 345)
                    len_call_result_12685 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), len_12682, *[s_12683], **kwargs_12684)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 345)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', len_call_result_12685)
                    # SSA join for if statement (line 340)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 338)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 336)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'encoded_header_len(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'encoded_header_len' in the type store
        # Getting the type of 'stypy_return_type' (line 332)
        stypy_return_type_12686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12686)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'encoded_header_len'
        return stypy_return_type_12686


    @norecursion
    def header_encode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 347)
        False_12687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 39), 'False')
        defaults = [False_12687]
        # Create a new context for function 'header_encode'
        module_type_store = module_type_store.open_function_context('header_encode', 347, 4, False)
        # Assigning a type to the variable 'self' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.header_encode.__dict__.__setitem__('stypy_localization', localization)
        Charset.header_encode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.header_encode.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.header_encode.__dict__.__setitem__('stypy_function_name', 'Charset.header_encode')
        Charset.header_encode.__dict__.__setitem__('stypy_param_names_list', ['s', 'convert'])
        Charset.header_encode.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.header_encode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.header_encode.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.header_encode.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.header_encode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.header_encode.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.header_encode', ['s', 'convert'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'header_encode', localization, ['s', 'convert'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'header_encode(...)' code ##################

        str_12688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, (-1)), 'str', 'Header-encode a string, optionally converting it to output_charset.\n\n        If convert is True, the string will be converted from the input\n        charset to the output charset automatically.  This is not useful for\n        multibyte character sets, which have line length issues (multibyte\n        characters must be split on a character, not a byte boundary); use the\n        high-level Header class to deal with these issues.  convert defaults\n        to False.\n\n        The type of encoding (base64 or quoted-printable) will be based on\n        self.header_encoding.\n        ')
        
        # Assigning a Call to a Name (line 360):
        
        # Assigning a Call to a Name (line 360):
        
        # Call to get_output_charset(...): (line 360)
        # Processing the call keyword arguments (line 360)
        kwargs_12691 = {}
        # Getting the type of 'self' (line 360)
        self_12689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'self', False)
        # Obtaining the member 'get_output_charset' of a type (line 360)
        get_output_charset_12690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 15), self_12689, 'get_output_charset')
        # Calling get_output_charset(args, kwargs) (line 360)
        get_output_charset_call_result_12692 = invoke(stypy.reporting.localization.Localization(__file__, 360, 15), get_output_charset_12690, *[], **kwargs_12691)
        
        # Assigning a type to the variable 'cset' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'cset', get_output_charset_call_result_12692)
        # Getting the type of 'convert' (line 361)
        convert_12693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 11), 'convert')
        # Testing if the type of an if condition is none (line 361)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 361, 8), convert_12693):
            pass
        else:
            
            # Testing the type of an if condition (line 361)
            if_condition_12694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 8), convert_12693)
            # Assigning a type to the variable 'if_condition_12694' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'if_condition_12694', if_condition_12694)
            # SSA begins for if statement (line 361)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 362):
            
            # Assigning a Call to a Name (line 362):
            
            # Call to convert(...): (line 362)
            # Processing the call arguments (line 362)
            # Getting the type of 's' (line 362)
            s_12697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 29), 's', False)
            # Processing the call keyword arguments (line 362)
            kwargs_12698 = {}
            # Getting the type of 'self' (line 362)
            self_12695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'self', False)
            # Obtaining the member 'convert' of a type (line 362)
            convert_12696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 16), self_12695, 'convert')
            # Calling convert(args, kwargs) (line 362)
            convert_call_result_12699 = invoke(stypy.reporting.localization.Localization(__file__, 362, 16), convert_12696, *[s_12697], **kwargs_12698)
            
            # Assigning a type to the variable 's' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 's', convert_call_result_12699)
            # SSA join for if statement (line 361)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 364)
        self_12700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 11), 'self')
        # Obtaining the member 'header_encoding' of a type (line 364)
        header_encoding_12701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 11), self_12700, 'header_encoding')
        # Getting the type of 'BASE64' (line 364)
        BASE64_12702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 35), 'BASE64')
        # Applying the binary operator '==' (line 364)
        result_eq_12703 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 11), '==', header_encoding_12701, BASE64_12702)
        
        # Testing if the type of an if condition is none (line 364)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 364, 8), result_eq_12703):
            
            # Getting the type of 'self' (line 366)
            self_12712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 13), 'self')
            # Obtaining the member 'header_encoding' of a type (line 366)
            header_encoding_12713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 13), self_12712, 'header_encoding')
            # Getting the type of 'QP' (line 366)
            QP_12714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 37), 'QP')
            # Applying the binary operator '==' (line 366)
            result_eq_12715 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 13), '==', header_encoding_12713, QP_12714)
            
            # Testing if the type of an if condition is none (line 366)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 366, 13), result_eq_12715):
                
                # Getting the type of 'self' (line 368)
                self_12726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'self')
                # Obtaining the member 'header_encoding' of a type (line 368)
                header_encoding_12727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 13), self_12726, 'header_encoding')
                # Getting the type of 'SHORTEST' (line 368)
                SHORTEST_12728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 37), 'SHORTEST')
                # Applying the binary operator '==' (line 368)
                result_eq_12729 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 13), '==', header_encoding_12727, SHORTEST_12728)
                
                # Testing if the type of an if condition is none (line 368)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 368, 13), result_eq_12729):
                    # Getting the type of 's' (line 376)
                    s_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 's')
                    # Assigning a type to the variable 'stypy_return_type' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', s_12763)
                else:
                    
                    # Testing the type of an if condition (line 368)
                    if_condition_12730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 13), result_eq_12729)
                    # Assigning a type to the variable 'if_condition_12730' (line 368)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'if_condition_12730', if_condition_12730)
                    # SSA begins for if statement (line 368)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 369):
                    
                    # Assigning a Call to a Name (line 369):
                    
                    # Call to base64_len(...): (line 369)
                    # Processing the call arguments (line 369)
                    # Getting the type of 's' (line 369)
                    s_12734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 49), 's', False)
                    # Processing the call keyword arguments (line 369)
                    kwargs_12735 = {}
                    # Getting the type of 'email' (line 369)
                    email_12731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'email', False)
                    # Obtaining the member 'base64mime' of a type (line 369)
                    base64mime_12732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), email_12731, 'base64mime')
                    # Obtaining the member 'base64_len' of a type (line 369)
                    base64_len_12733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), base64mime_12732, 'base64_len')
                    # Calling base64_len(args, kwargs) (line 369)
                    base64_len_call_result_12736 = invoke(stypy.reporting.localization.Localization(__file__, 369, 21), base64_len_12733, *[s_12734], **kwargs_12735)
                    
                    # Assigning a type to the variable 'lenb64' (line 369)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'lenb64', base64_len_call_result_12736)
                    
                    # Assigning a Call to a Name (line 370):
                    
                    # Assigning a Call to a Name (line 370):
                    
                    # Call to header_quopri_len(...): (line 370)
                    # Processing the call arguments (line 370)
                    # Getting the type of 's' (line 370)
                    s_12740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 55), 's', False)
                    # Processing the call keyword arguments (line 370)
                    kwargs_12741 = {}
                    # Getting the type of 'email' (line 370)
                    email_12737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 20), 'email', False)
                    # Obtaining the member 'quoprimime' of a type (line 370)
                    quoprimime_12738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 20), email_12737, 'quoprimime')
                    # Obtaining the member 'header_quopri_len' of a type (line 370)
                    header_quopri_len_12739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 20), quoprimime_12738, 'header_quopri_len')
                    # Calling header_quopri_len(args, kwargs) (line 370)
                    header_quopri_len_call_result_12742 = invoke(stypy.reporting.localization.Localization(__file__, 370, 20), header_quopri_len_12739, *[s_12740], **kwargs_12741)
                    
                    # Assigning a type to the variable 'lenqp' (line 370)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'lenqp', header_quopri_len_call_result_12742)
                    
                    # Getting the type of 'lenb64' (line 371)
                    lenb64_12743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'lenb64')
                    # Getting the type of 'lenqp' (line 371)
                    lenqp_12744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'lenqp')
                    # Applying the binary operator '<' (line 371)
                    result_lt_12745 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 15), '<', lenb64_12743, lenqp_12744)
                    
                    # Testing if the type of an if condition is none (line 371)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 371, 12), result_lt_12745):
                        
                        # Call to header_encode(...): (line 374)
                        # Processing the call arguments (line 374)
                        # Getting the type of 's' (line 374)
                        s_12757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 54), 's', False)
                        # Getting the type of 'cset' (line 374)
                        cset_12758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 57), 'cset', False)
                        # Processing the call keyword arguments (line 374)
                        # Getting the type of 'None' (line 374)
                        None_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 74), 'None', False)
                        keyword_12760 = None_12759
                        kwargs_12761 = {'maxlinelen': keyword_12760}
                        # Getting the type of 'email' (line 374)
                        email_12754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'email', False)
                        # Obtaining the member 'quoprimime' of a type (line 374)
                        quoprimime_12755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), email_12754, 'quoprimime')
                        # Obtaining the member 'header_encode' of a type (line 374)
                        header_encode_12756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), quoprimime_12755, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 374)
                        header_encode_call_result_12762 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), header_encode_12756, *[s_12757, cset_12758], **kwargs_12761)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 374)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'stypy_return_type', header_encode_call_result_12762)
                    else:
                        
                        # Testing the type of an if condition (line 371)
                        if_condition_12746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 12), result_lt_12745)
                        # Assigning a type to the variable 'if_condition_12746' (line 371)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'if_condition_12746', if_condition_12746)
                        # SSA begins for if statement (line 371)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to header_encode(...): (line 372)
                        # Processing the call arguments (line 372)
                        # Getting the type of 's' (line 372)
                        s_12750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 54), 's', False)
                        # Getting the type of 'cset' (line 372)
                        cset_12751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 57), 'cset', False)
                        # Processing the call keyword arguments (line 372)
                        kwargs_12752 = {}
                        # Getting the type of 'email' (line 372)
                        email_12747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'email', False)
                        # Obtaining the member 'base64mime' of a type (line 372)
                        base64mime_12748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), email_12747, 'base64mime')
                        # Obtaining the member 'header_encode' of a type (line 372)
                        header_encode_12749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), base64mime_12748, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 372)
                        header_encode_call_result_12753 = invoke(stypy.reporting.localization.Localization(__file__, 372, 23), header_encode_12749, *[s_12750, cset_12751], **kwargs_12752)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 372)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'stypy_return_type', header_encode_call_result_12753)
                        # SSA branch for the else part of an if statement (line 371)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to header_encode(...): (line 374)
                        # Processing the call arguments (line 374)
                        # Getting the type of 's' (line 374)
                        s_12757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 54), 's', False)
                        # Getting the type of 'cset' (line 374)
                        cset_12758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 57), 'cset', False)
                        # Processing the call keyword arguments (line 374)
                        # Getting the type of 'None' (line 374)
                        None_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 74), 'None', False)
                        keyword_12760 = None_12759
                        kwargs_12761 = {'maxlinelen': keyword_12760}
                        # Getting the type of 'email' (line 374)
                        email_12754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'email', False)
                        # Obtaining the member 'quoprimime' of a type (line 374)
                        quoprimime_12755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), email_12754, 'quoprimime')
                        # Obtaining the member 'header_encode' of a type (line 374)
                        header_encode_12756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), quoprimime_12755, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 374)
                        header_encode_call_result_12762 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), header_encode_12756, *[s_12757, cset_12758], **kwargs_12761)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 374)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'stypy_return_type', header_encode_call_result_12762)
                        # SSA join for if statement (line 371)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 368)
                    module_type_store.open_ssa_branch('else')
                    # Getting the type of 's' (line 376)
                    s_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 's')
                    # Assigning a type to the variable 'stypy_return_type' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', s_12763)
                    # SSA join for if statement (line 368)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 366)
                if_condition_12716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 13), result_eq_12715)
                # Assigning a type to the variable 'if_condition_12716' (line 366)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 13), 'if_condition_12716', if_condition_12716)
                # SSA begins for if statement (line 366)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to header_encode(...): (line 367)
                # Processing the call arguments (line 367)
                # Getting the type of 's' (line 367)
                s_12720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 50), 's', False)
                # Getting the type of 'cset' (line 367)
                cset_12721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 53), 'cset', False)
                # Processing the call keyword arguments (line 367)
                # Getting the type of 'None' (line 367)
                None_12722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 70), 'None', False)
                keyword_12723 = None_12722
                kwargs_12724 = {'maxlinelen': keyword_12723}
                # Getting the type of 'email' (line 367)
                email_12717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 19), 'email', False)
                # Obtaining the member 'quoprimime' of a type (line 367)
                quoprimime_12718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 19), email_12717, 'quoprimime')
                # Obtaining the member 'header_encode' of a type (line 367)
                header_encode_12719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 19), quoprimime_12718, 'header_encode')
                # Calling header_encode(args, kwargs) (line 367)
                header_encode_call_result_12725 = invoke(stypy.reporting.localization.Localization(__file__, 367, 19), header_encode_12719, *[s_12720, cset_12721], **kwargs_12724)
                
                # Assigning a type to the variable 'stypy_return_type' (line 367)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'stypy_return_type', header_encode_call_result_12725)
                # SSA branch for the else part of an if statement (line 366)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'self' (line 368)
                self_12726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'self')
                # Obtaining the member 'header_encoding' of a type (line 368)
                header_encoding_12727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 13), self_12726, 'header_encoding')
                # Getting the type of 'SHORTEST' (line 368)
                SHORTEST_12728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 37), 'SHORTEST')
                # Applying the binary operator '==' (line 368)
                result_eq_12729 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 13), '==', header_encoding_12727, SHORTEST_12728)
                
                # Testing if the type of an if condition is none (line 368)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 368, 13), result_eq_12729):
                    # Getting the type of 's' (line 376)
                    s_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 's')
                    # Assigning a type to the variable 'stypy_return_type' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', s_12763)
                else:
                    
                    # Testing the type of an if condition (line 368)
                    if_condition_12730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 13), result_eq_12729)
                    # Assigning a type to the variable 'if_condition_12730' (line 368)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'if_condition_12730', if_condition_12730)
                    # SSA begins for if statement (line 368)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 369):
                    
                    # Assigning a Call to a Name (line 369):
                    
                    # Call to base64_len(...): (line 369)
                    # Processing the call arguments (line 369)
                    # Getting the type of 's' (line 369)
                    s_12734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 49), 's', False)
                    # Processing the call keyword arguments (line 369)
                    kwargs_12735 = {}
                    # Getting the type of 'email' (line 369)
                    email_12731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'email', False)
                    # Obtaining the member 'base64mime' of a type (line 369)
                    base64mime_12732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), email_12731, 'base64mime')
                    # Obtaining the member 'base64_len' of a type (line 369)
                    base64_len_12733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), base64mime_12732, 'base64_len')
                    # Calling base64_len(args, kwargs) (line 369)
                    base64_len_call_result_12736 = invoke(stypy.reporting.localization.Localization(__file__, 369, 21), base64_len_12733, *[s_12734], **kwargs_12735)
                    
                    # Assigning a type to the variable 'lenb64' (line 369)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'lenb64', base64_len_call_result_12736)
                    
                    # Assigning a Call to a Name (line 370):
                    
                    # Assigning a Call to a Name (line 370):
                    
                    # Call to header_quopri_len(...): (line 370)
                    # Processing the call arguments (line 370)
                    # Getting the type of 's' (line 370)
                    s_12740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 55), 's', False)
                    # Processing the call keyword arguments (line 370)
                    kwargs_12741 = {}
                    # Getting the type of 'email' (line 370)
                    email_12737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 20), 'email', False)
                    # Obtaining the member 'quoprimime' of a type (line 370)
                    quoprimime_12738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 20), email_12737, 'quoprimime')
                    # Obtaining the member 'header_quopri_len' of a type (line 370)
                    header_quopri_len_12739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 20), quoprimime_12738, 'header_quopri_len')
                    # Calling header_quopri_len(args, kwargs) (line 370)
                    header_quopri_len_call_result_12742 = invoke(stypy.reporting.localization.Localization(__file__, 370, 20), header_quopri_len_12739, *[s_12740], **kwargs_12741)
                    
                    # Assigning a type to the variable 'lenqp' (line 370)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'lenqp', header_quopri_len_call_result_12742)
                    
                    # Getting the type of 'lenb64' (line 371)
                    lenb64_12743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'lenb64')
                    # Getting the type of 'lenqp' (line 371)
                    lenqp_12744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'lenqp')
                    # Applying the binary operator '<' (line 371)
                    result_lt_12745 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 15), '<', lenb64_12743, lenqp_12744)
                    
                    # Testing if the type of an if condition is none (line 371)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 371, 12), result_lt_12745):
                        
                        # Call to header_encode(...): (line 374)
                        # Processing the call arguments (line 374)
                        # Getting the type of 's' (line 374)
                        s_12757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 54), 's', False)
                        # Getting the type of 'cset' (line 374)
                        cset_12758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 57), 'cset', False)
                        # Processing the call keyword arguments (line 374)
                        # Getting the type of 'None' (line 374)
                        None_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 74), 'None', False)
                        keyword_12760 = None_12759
                        kwargs_12761 = {'maxlinelen': keyword_12760}
                        # Getting the type of 'email' (line 374)
                        email_12754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'email', False)
                        # Obtaining the member 'quoprimime' of a type (line 374)
                        quoprimime_12755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), email_12754, 'quoprimime')
                        # Obtaining the member 'header_encode' of a type (line 374)
                        header_encode_12756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), quoprimime_12755, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 374)
                        header_encode_call_result_12762 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), header_encode_12756, *[s_12757, cset_12758], **kwargs_12761)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 374)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'stypy_return_type', header_encode_call_result_12762)
                    else:
                        
                        # Testing the type of an if condition (line 371)
                        if_condition_12746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 12), result_lt_12745)
                        # Assigning a type to the variable 'if_condition_12746' (line 371)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'if_condition_12746', if_condition_12746)
                        # SSA begins for if statement (line 371)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to header_encode(...): (line 372)
                        # Processing the call arguments (line 372)
                        # Getting the type of 's' (line 372)
                        s_12750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 54), 's', False)
                        # Getting the type of 'cset' (line 372)
                        cset_12751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 57), 'cset', False)
                        # Processing the call keyword arguments (line 372)
                        kwargs_12752 = {}
                        # Getting the type of 'email' (line 372)
                        email_12747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'email', False)
                        # Obtaining the member 'base64mime' of a type (line 372)
                        base64mime_12748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), email_12747, 'base64mime')
                        # Obtaining the member 'header_encode' of a type (line 372)
                        header_encode_12749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), base64mime_12748, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 372)
                        header_encode_call_result_12753 = invoke(stypy.reporting.localization.Localization(__file__, 372, 23), header_encode_12749, *[s_12750, cset_12751], **kwargs_12752)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 372)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'stypy_return_type', header_encode_call_result_12753)
                        # SSA branch for the else part of an if statement (line 371)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to header_encode(...): (line 374)
                        # Processing the call arguments (line 374)
                        # Getting the type of 's' (line 374)
                        s_12757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 54), 's', False)
                        # Getting the type of 'cset' (line 374)
                        cset_12758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 57), 'cset', False)
                        # Processing the call keyword arguments (line 374)
                        # Getting the type of 'None' (line 374)
                        None_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 74), 'None', False)
                        keyword_12760 = None_12759
                        kwargs_12761 = {'maxlinelen': keyword_12760}
                        # Getting the type of 'email' (line 374)
                        email_12754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'email', False)
                        # Obtaining the member 'quoprimime' of a type (line 374)
                        quoprimime_12755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), email_12754, 'quoprimime')
                        # Obtaining the member 'header_encode' of a type (line 374)
                        header_encode_12756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), quoprimime_12755, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 374)
                        header_encode_call_result_12762 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), header_encode_12756, *[s_12757, cset_12758], **kwargs_12761)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 374)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'stypy_return_type', header_encode_call_result_12762)
                        # SSA join for if statement (line 371)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 368)
                    module_type_store.open_ssa_branch('else')
                    # Getting the type of 's' (line 376)
                    s_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 's')
                    # Assigning a type to the variable 'stypy_return_type' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', s_12763)
                    # SSA join for if statement (line 368)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 366)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 364)
            if_condition_12704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 8), result_eq_12703)
            # Assigning a type to the variable 'if_condition_12704' (line 364)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'if_condition_12704', if_condition_12704)
            # SSA begins for if statement (line 364)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to header_encode(...): (line 365)
            # Processing the call arguments (line 365)
            # Getting the type of 's' (line 365)
            s_12708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 50), 's', False)
            # Getting the type of 'cset' (line 365)
            cset_12709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 53), 'cset', False)
            # Processing the call keyword arguments (line 365)
            kwargs_12710 = {}
            # Getting the type of 'email' (line 365)
            email_12705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 19), 'email', False)
            # Obtaining the member 'base64mime' of a type (line 365)
            base64mime_12706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 19), email_12705, 'base64mime')
            # Obtaining the member 'header_encode' of a type (line 365)
            header_encode_12707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 19), base64mime_12706, 'header_encode')
            # Calling header_encode(args, kwargs) (line 365)
            header_encode_call_result_12711 = invoke(stypy.reporting.localization.Localization(__file__, 365, 19), header_encode_12707, *[s_12708, cset_12709], **kwargs_12710)
            
            # Assigning a type to the variable 'stypy_return_type' (line 365)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'stypy_return_type', header_encode_call_result_12711)
            # SSA branch for the else part of an if statement (line 364)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'self' (line 366)
            self_12712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 13), 'self')
            # Obtaining the member 'header_encoding' of a type (line 366)
            header_encoding_12713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 13), self_12712, 'header_encoding')
            # Getting the type of 'QP' (line 366)
            QP_12714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 37), 'QP')
            # Applying the binary operator '==' (line 366)
            result_eq_12715 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 13), '==', header_encoding_12713, QP_12714)
            
            # Testing if the type of an if condition is none (line 366)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 366, 13), result_eq_12715):
                
                # Getting the type of 'self' (line 368)
                self_12726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'self')
                # Obtaining the member 'header_encoding' of a type (line 368)
                header_encoding_12727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 13), self_12726, 'header_encoding')
                # Getting the type of 'SHORTEST' (line 368)
                SHORTEST_12728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 37), 'SHORTEST')
                # Applying the binary operator '==' (line 368)
                result_eq_12729 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 13), '==', header_encoding_12727, SHORTEST_12728)
                
                # Testing if the type of an if condition is none (line 368)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 368, 13), result_eq_12729):
                    # Getting the type of 's' (line 376)
                    s_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 's')
                    # Assigning a type to the variable 'stypy_return_type' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', s_12763)
                else:
                    
                    # Testing the type of an if condition (line 368)
                    if_condition_12730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 13), result_eq_12729)
                    # Assigning a type to the variable 'if_condition_12730' (line 368)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'if_condition_12730', if_condition_12730)
                    # SSA begins for if statement (line 368)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 369):
                    
                    # Assigning a Call to a Name (line 369):
                    
                    # Call to base64_len(...): (line 369)
                    # Processing the call arguments (line 369)
                    # Getting the type of 's' (line 369)
                    s_12734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 49), 's', False)
                    # Processing the call keyword arguments (line 369)
                    kwargs_12735 = {}
                    # Getting the type of 'email' (line 369)
                    email_12731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'email', False)
                    # Obtaining the member 'base64mime' of a type (line 369)
                    base64mime_12732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), email_12731, 'base64mime')
                    # Obtaining the member 'base64_len' of a type (line 369)
                    base64_len_12733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), base64mime_12732, 'base64_len')
                    # Calling base64_len(args, kwargs) (line 369)
                    base64_len_call_result_12736 = invoke(stypy.reporting.localization.Localization(__file__, 369, 21), base64_len_12733, *[s_12734], **kwargs_12735)
                    
                    # Assigning a type to the variable 'lenb64' (line 369)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'lenb64', base64_len_call_result_12736)
                    
                    # Assigning a Call to a Name (line 370):
                    
                    # Assigning a Call to a Name (line 370):
                    
                    # Call to header_quopri_len(...): (line 370)
                    # Processing the call arguments (line 370)
                    # Getting the type of 's' (line 370)
                    s_12740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 55), 's', False)
                    # Processing the call keyword arguments (line 370)
                    kwargs_12741 = {}
                    # Getting the type of 'email' (line 370)
                    email_12737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 20), 'email', False)
                    # Obtaining the member 'quoprimime' of a type (line 370)
                    quoprimime_12738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 20), email_12737, 'quoprimime')
                    # Obtaining the member 'header_quopri_len' of a type (line 370)
                    header_quopri_len_12739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 20), quoprimime_12738, 'header_quopri_len')
                    # Calling header_quopri_len(args, kwargs) (line 370)
                    header_quopri_len_call_result_12742 = invoke(stypy.reporting.localization.Localization(__file__, 370, 20), header_quopri_len_12739, *[s_12740], **kwargs_12741)
                    
                    # Assigning a type to the variable 'lenqp' (line 370)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'lenqp', header_quopri_len_call_result_12742)
                    
                    # Getting the type of 'lenb64' (line 371)
                    lenb64_12743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'lenb64')
                    # Getting the type of 'lenqp' (line 371)
                    lenqp_12744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'lenqp')
                    # Applying the binary operator '<' (line 371)
                    result_lt_12745 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 15), '<', lenb64_12743, lenqp_12744)
                    
                    # Testing if the type of an if condition is none (line 371)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 371, 12), result_lt_12745):
                        
                        # Call to header_encode(...): (line 374)
                        # Processing the call arguments (line 374)
                        # Getting the type of 's' (line 374)
                        s_12757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 54), 's', False)
                        # Getting the type of 'cset' (line 374)
                        cset_12758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 57), 'cset', False)
                        # Processing the call keyword arguments (line 374)
                        # Getting the type of 'None' (line 374)
                        None_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 74), 'None', False)
                        keyword_12760 = None_12759
                        kwargs_12761 = {'maxlinelen': keyword_12760}
                        # Getting the type of 'email' (line 374)
                        email_12754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'email', False)
                        # Obtaining the member 'quoprimime' of a type (line 374)
                        quoprimime_12755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), email_12754, 'quoprimime')
                        # Obtaining the member 'header_encode' of a type (line 374)
                        header_encode_12756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), quoprimime_12755, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 374)
                        header_encode_call_result_12762 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), header_encode_12756, *[s_12757, cset_12758], **kwargs_12761)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 374)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'stypy_return_type', header_encode_call_result_12762)
                    else:
                        
                        # Testing the type of an if condition (line 371)
                        if_condition_12746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 12), result_lt_12745)
                        # Assigning a type to the variable 'if_condition_12746' (line 371)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'if_condition_12746', if_condition_12746)
                        # SSA begins for if statement (line 371)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to header_encode(...): (line 372)
                        # Processing the call arguments (line 372)
                        # Getting the type of 's' (line 372)
                        s_12750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 54), 's', False)
                        # Getting the type of 'cset' (line 372)
                        cset_12751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 57), 'cset', False)
                        # Processing the call keyword arguments (line 372)
                        kwargs_12752 = {}
                        # Getting the type of 'email' (line 372)
                        email_12747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'email', False)
                        # Obtaining the member 'base64mime' of a type (line 372)
                        base64mime_12748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), email_12747, 'base64mime')
                        # Obtaining the member 'header_encode' of a type (line 372)
                        header_encode_12749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), base64mime_12748, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 372)
                        header_encode_call_result_12753 = invoke(stypy.reporting.localization.Localization(__file__, 372, 23), header_encode_12749, *[s_12750, cset_12751], **kwargs_12752)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 372)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'stypy_return_type', header_encode_call_result_12753)
                        # SSA branch for the else part of an if statement (line 371)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to header_encode(...): (line 374)
                        # Processing the call arguments (line 374)
                        # Getting the type of 's' (line 374)
                        s_12757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 54), 's', False)
                        # Getting the type of 'cset' (line 374)
                        cset_12758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 57), 'cset', False)
                        # Processing the call keyword arguments (line 374)
                        # Getting the type of 'None' (line 374)
                        None_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 74), 'None', False)
                        keyword_12760 = None_12759
                        kwargs_12761 = {'maxlinelen': keyword_12760}
                        # Getting the type of 'email' (line 374)
                        email_12754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'email', False)
                        # Obtaining the member 'quoprimime' of a type (line 374)
                        quoprimime_12755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), email_12754, 'quoprimime')
                        # Obtaining the member 'header_encode' of a type (line 374)
                        header_encode_12756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), quoprimime_12755, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 374)
                        header_encode_call_result_12762 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), header_encode_12756, *[s_12757, cset_12758], **kwargs_12761)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 374)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'stypy_return_type', header_encode_call_result_12762)
                        # SSA join for if statement (line 371)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 368)
                    module_type_store.open_ssa_branch('else')
                    # Getting the type of 's' (line 376)
                    s_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 's')
                    # Assigning a type to the variable 'stypy_return_type' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', s_12763)
                    # SSA join for if statement (line 368)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 366)
                if_condition_12716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 13), result_eq_12715)
                # Assigning a type to the variable 'if_condition_12716' (line 366)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 13), 'if_condition_12716', if_condition_12716)
                # SSA begins for if statement (line 366)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to header_encode(...): (line 367)
                # Processing the call arguments (line 367)
                # Getting the type of 's' (line 367)
                s_12720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 50), 's', False)
                # Getting the type of 'cset' (line 367)
                cset_12721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 53), 'cset', False)
                # Processing the call keyword arguments (line 367)
                # Getting the type of 'None' (line 367)
                None_12722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 70), 'None', False)
                keyword_12723 = None_12722
                kwargs_12724 = {'maxlinelen': keyword_12723}
                # Getting the type of 'email' (line 367)
                email_12717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 19), 'email', False)
                # Obtaining the member 'quoprimime' of a type (line 367)
                quoprimime_12718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 19), email_12717, 'quoprimime')
                # Obtaining the member 'header_encode' of a type (line 367)
                header_encode_12719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 19), quoprimime_12718, 'header_encode')
                # Calling header_encode(args, kwargs) (line 367)
                header_encode_call_result_12725 = invoke(stypy.reporting.localization.Localization(__file__, 367, 19), header_encode_12719, *[s_12720, cset_12721], **kwargs_12724)
                
                # Assigning a type to the variable 'stypy_return_type' (line 367)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'stypy_return_type', header_encode_call_result_12725)
                # SSA branch for the else part of an if statement (line 366)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'self' (line 368)
                self_12726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'self')
                # Obtaining the member 'header_encoding' of a type (line 368)
                header_encoding_12727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 13), self_12726, 'header_encoding')
                # Getting the type of 'SHORTEST' (line 368)
                SHORTEST_12728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 37), 'SHORTEST')
                # Applying the binary operator '==' (line 368)
                result_eq_12729 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 13), '==', header_encoding_12727, SHORTEST_12728)
                
                # Testing if the type of an if condition is none (line 368)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 368, 13), result_eq_12729):
                    # Getting the type of 's' (line 376)
                    s_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 's')
                    # Assigning a type to the variable 'stypy_return_type' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', s_12763)
                else:
                    
                    # Testing the type of an if condition (line 368)
                    if_condition_12730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 13), result_eq_12729)
                    # Assigning a type to the variable 'if_condition_12730' (line 368)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'if_condition_12730', if_condition_12730)
                    # SSA begins for if statement (line 368)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 369):
                    
                    # Assigning a Call to a Name (line 369):
                    
                    # Call to base64_len(...): (line 369)
                    # Processing the call arguments (line 369)
                    # Getting the type of 's' (line 369)
                    s_12734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 49), 's', False)
                    # Processing the call keyword arguments (line 369)
                    kwargs_12735 = {}
                    # Getting the type of 'email' (line 369)
                    email_12731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'email', False)
                    # Obtaining the member 'base64mime' of a type (line 369)
                    base64mime_12732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), email_12731, 'base64mime')
                    # Obtaining the member 'base64_len' of a type (line 369)
                    base64_len_12733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), base64mime_12732, 'base64_len')
                    # Calling base64_len(args, kwargs) (line 369)
                    base64_len_call_result_12736 = invoke(stypy.reporting.localization.Localization(__file__, 369, 21), base64_len_12733, *[s_12734], **kwargs_12735)
                    
                    # Assigning a type to the variable 'lenb64' (line 369)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'lenb64', base64_len_call_result_12736)
                    
                    # Assigning a Call to a Name (line 370):
                    
                    # Assigning a Call to a Name (line 370):
                    
                    # Call to header_quopri_len(...): (line 370)
                    # Processing the call arguments (line 370)
                    # Getting the type of 's' (line 370)
                    s_12740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 55), 's', False)
                    # Processing the call keyword arguments (line 370)
                    kwargs_12741 = {}
                    # Getting the type of 'email' (line 370)
                    email_12737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 20), 'email', False)
                    # Obtaining the member 'quoprimime' of a type (line 370)
                    quoprimime_12738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 20), email_12737, 'quoprimime')
                    # Obtaining the member 'header_quopri_len' of a type (line 370)
                    header_quopri_len_12739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 20), quoprimime_12738, 'header_quopri_len')
                    # Calling header_quopri_len(args, kwargs) (line 370)
                    header_quopri_len_call_result_12742 = invoke(stypy.reporting.localization.Localization(__file__, 370, 20), header_quopri_len_12739, *[s_12740], **kwargs_12741)
                    
                    # Assigning a type to the variable 'lenqp' (line 370)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'lenqp', header_quopri_len_call_result_12742)
                    
                    # Getting the type of 'lenb64' (line 371)
                    lenb64_12743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'lenb64')
                    # Getting the type of 'lenqp' (line 371)
                    lenqp_12744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'lenqp')
                    # Applying the binary operator '<' (line 371)
                    result_lt_12745 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 15), '<', lenb64_12743, lenqp_12744)
                    
                    # Testing if the type of an if condition is none (line 371)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 371, 12), result_lt_12745):
                        
                        # Call to header_encode(...): (line 374)
                        # Processing the call arguments (line 374)
                        # Getting the type of 's' (line 374)
                        s_12757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 54), 's', False)
                        # Getting the type of 'cset' (line 374)
                        cset_12758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 57), 'cset', False)
                        # Processing the call keyword arguments (line 374)
                        # Getting the type of 'None' (line 374)
                        None_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 74), 'None', False)
                        keyword_12760 = None_12759
                        kwargs_12761 = {'maxlinelen': keyword_12760}
                        # Getting the type of 'email' (line 374)
                        email_12754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'email', False)
                        # Obtaining the member 'quoprimime' of a type (line 374)
                        quoprimime_12755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), email_12754, 'quoprimime')
                        # Obtaining the member 'header_encode' of a type (line 374)
                        header_encode_12756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), quoprimime_12755, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 374)
                        header_encode_call_result_12762 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), header_encode_12756, *[s_12757, cset_12758], **kwargs_12761)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 374)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'stypy_return_type', header_encode_call_result_12762)
                    else:
                        
                        # Testing the type of an if condition (line 371)
                        if_condition_12746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 12), result_lt_12745)
                        # Assigning a type to the variable 'if_condition_12746' (line 371)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'if_condition_12746', if_condition_12746)
                        # SSA begins for if statement (line 371)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to header_encode(...): (line 372)
                        # Processing the call arguments (line 372)
                        # Getting the type of 's' (line 372)
                        s_12750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 54), 's', False)
                        # Getting the type of 'cset' (line 372)
                        cset_12751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 57), 'cset', False)
                        # Processing the call keyword arguments (line 372)
                        kwargs_12752 = {}
                        # Getting the type of 'email' (line 372)
                        email_12747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'email', False)
                        # Obtaining the member 'base64mime' of a type (line 372)
                        base64mime_12748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), email_12747, 'base64mime')
                        # Obtaining the member 'header_encode' of a type (line 372)
                        header_encode_12749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), base64mime_12748, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 372)
                        header_encode_call_result_12753 = invoke(stypy.reporting.localization.Localization(__file__, 372, 23), header_encode_12749, *[s_12750, cset_12751], **kwargs_12752)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 372)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'stypy_return_type', header_encode_call_result_12753)
                        # SSA branch for the else part of an if statement (line 371)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to header_encode(...): (line 374)
                        # Processing the call arguments (line 374)
                        # Getting the type of 's' (line 374)
                        s_12757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 54), 's', False)
                        # Getting the type of 'cset' (line 374)
                        cset_12758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 57), 'cset', False)
                        # Processing the call keyword arguments (line 374)
                        # Getting the type of 'None' (line 374)
                        None_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 74), 'None', False)
                        keyword_12760 = None_12759
                        kwargs_12761 = {'maxlinelen': keyword_12760}
                        # Getting the type of 'email' (line 374)
                        email_12754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'email', False)
                        # Obtaining the member 'quoprimime' of a type (line 374)
                        quoprimime_12755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), email_12754, 'quoprimime')
                        # Obtaining the member 'header_encode' of a type (line 374)
                        header_encode_12756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), quoprimime_12755, 'header_encode')
                        # Calling header_encode(args, kwargs) (line 374)
                        header_encode_call_result_12762 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), header_encode_12756, *[s_12757, cset_12758], **kwargs_12761)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 374)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'stypy_return_type', header_encode_call_result_12762)
                        # SSA join for if statement (line 371)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 368)
                    module_type_store.open_ssa_branch('else')
                    # Getting the type of 's' (line 376)
                    s_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 's')
                    # Assigning a type to the variable 'stypy_return_type' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', s_12763)
                    # SSA join for if statement (line 368)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 366)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 364)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'header_encode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'header_encode' in the type store
        # Getting the type of 'stypy_return_type' (line 347)
        stypy_return_type_12764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'header_encode'
        return stypy_return_type_12764


    @norecursion
    def body_encode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 378)
        True_12765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 37), 'True')
        defaults = [True_12765]
        # Create a new context for function 'body_encode'
        module_type_store = module_type_store.open_function_context('body_encode', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Charset.body_encode.__dict__.__setitem__('stypy_localization', localization)
        Charset.body_encode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Charset.body_encode.__dict__.__setitem__('stypy_type_store', module_type_store)
        Charset.body_encode.__dict__.__setitem__('stypy_function_name', 'Charset.body_encode')
        Charset.body_encode.__dict__.__setitem__('stypy_param_names_list', ['s', 'convert'])
        Charset.body_encode.__dict__.__setitem__('stypy_varargs_param_name', None)
        Charset.body_encode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Charset.body_encode.__dict__.__setitem__('stypy_call_defaults', defaults)
        Charset.body_encode.__dict__.__setitem__('stypy_call_varargs', varargs)
        Charset.body_encode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Charset.body_encode.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Charset.body_encode', ['s', 'convert'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'body_encode', localization, ['s', 'convert'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'body_encode(...)' code ##################

        str_12766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, (-1)), 'str', 'Body-encode a string and convert it to output_charset.\n\n        If convert is True (the default), the string will be converted from\n        the input charset to output charset automatically.  Unlike\n        header_encode(), there are no issues with byte boundaries and\n        multibyte charsets in email bodies, so this is usually pretty safe.\n\n        The type of encoding (base64 or quoted-printable) will be based on\n        self.body_encoding.\n        ')
        # Getting the type of 'convert' (line 389)
        convert_12767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'convert')
        # Testing if the type of an if condition is none (line 389)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 389, 8), convert_12767):
            pass
        else:
            
            # Testing the type of an if condition (line 389)
            if_condition_12768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 8), convert_12767)
            # Assigning a type to the variable 'if_condition_12768' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'if_condition_12768', if_condition_12768)
            # SSA begins for if statement (line 389)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 390):
            
            # Assigning a Call to a Name (line 390):
            
            # Call to convert(...): (line 390)
            # Processing the call arguments (line 390)
            # Getting the type of 's' (line 390)
            s_12771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 29), 's', False)
            # Processing the call keyword arguments (line 390)
            kwargs_12772 = {}
            # Getting the type of 'self' (line 390)
            self_12769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'self', False)
            # Obtaining the member 'convert' of a type (line 390)
            convert_12770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 16), self_12769, 'convert')
            # Calling convert(args, kwargs) (line 390)
            convert_call_result_12773 = invoke(stypy.reporting.localization.Localization(__file__, 390, 16), convert_12770, *[s_12771], **kwargs_12772)
            
            # Assigning a type to the variable 's' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 's', convert_call_result_12773)
            # SSA join for if statement (line 389)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 392)
        self_12774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 11), 'self')
        # Obtaining the member 'body_encoding' of a type (line 392)
        body_encoding_12775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 11), self_12774, 'body_encoding')
        # Getting the type of 'BASE64' (line 392)
        BASE64_12776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 33), 'BASE64')
        # Applying the binary operator 'is' (line 392)
        result_is__12777 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 11), 'is', body_encoding_12775, BASE64_12776)
        
        # Testing if the type of an if condition is none (line 392)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 392, 8), result_is__12777):
            
            # Getting the type of 'self' (line 394)
            self_12785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 13), 'self')
            # Obtaining the member 'body_encoding' of a type (line 394)
            body_encoding_12786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 13), self_12785, 'body_encoding')
            # Getting the type of 'QP' (line 394)
            QP_12787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 35), 'QP')
            # Applying the binary operator 'is' (line 394)
            result_is__12788 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 13), 'is', body_encoding_12786, QP_12787)
            
            # Testing if the type of an if condition is none (line 394)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 394, 13), result_is__12788):
                # Getting the type of 's' (line 397)
                s_12796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 's')
                # Assigning a type to the variable 'stypy_return_type' (line 397)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'stypy_return_type', s_12796)
            else:
                
                # Testing the type of an if condition (line 394)
                if_condition_12789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 13), result_is__12788)
                # Assigning a type to the variable 'if_condition_12789' (line 394)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 13), 'if_condition_12789', if_condition_12789)
                # SSA begins for if statement (line 394)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to body_encode(...): (line 395)
                # Processing the call arguments (line 395)
                # Getting the type of 's' (line 395)
                s_12793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 48), 's', False)
                # Processing the call keyword arguments (line 395)
                kwargs_12794 = {}
                # Getting the type of 'email' (line 395)
                email_12790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'email', False)
                # Obtaining the member 'quoprimime' of a type (line 395)
                quoprimime_12791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 19), email_12790, 'quoprimime')
                # Obtaining the member 'body_encode' of a type (line 395)
                body_encode_12792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 19), quoprimime_12791, 'body_encode')
                # Calling body_encode(args, kwargs) (line 395)
                body_encode_call_result_12795 = invoke(stypy.reporting.localization.Localization(__file__, 395, 19), body_encode_12792, *[s_12793], **kwargs_12794)
                
                # Assigning a type to the variable 'stypy_return_type' (line 395)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'stypy_return_type', body_encode_call_result_12795)
                # SSA branch for the else part of an if statement (line 394)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 's' (line 397)
                s_12796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 's')
                # Assigning a type to the variable 'stypy_return_type' (line 397)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'stypy_return_type', s_12796)
                # SSA join for if statement (line 394)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 392)
            if_condition_12778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 8), result_is__12777)
            # Assigning a type to the variable 'if_condition_12778' (line 392)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'if_condition_12778', if_condition_12778)
            # SSA begins for if statement (line 392)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to body_encode(...): (line 393)
            # Processing the call arguments (line 393)
            # Getting the type of 's' (line 393)
            s_12782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 48), 's', False)
            # Processing the call keyword arguments (line 393)
            kwargs_12783 = {}
            # Getting the type of 'email' (line 393)
            email_12779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 19), 'email', False)
            # Obtaining the member 'base64mime' of a type (line 393)
            base64mime_12780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 19), email_12779, 'base64mime')
            # Obtaining the member 'body_encode' of a type (line 393)
            body_encode_12781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 19), base64mime_12780, 'body_encode')
            # Calling body_encode(args, kwargs) (line 393)
            body_encode_call_result_12784 = invoke(stypy.reporting.localization.Localization(__file__, 393, 19), body_encode_12781, *[s_12782], **kwargs_12783)
            
            # Assigning a type to the variable 'stypy_return_type' (line 393)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'stypy_return_type', body_encode_call_result_12784)
            # SSA branch for the else part of an if statement (line 392)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'self' (line 394)
            self_12785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 13), 'self')
            # Obtaining the member 'body_encoding' of a type (line 394)
            body_encoding_12786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 13), self_12785, 'body_encoding')
            # Getting the type of 'QP' (line 394)
            QP_12787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 35), 'QP')
            # Applying the binary operator 'is' (line 394)
            result_is__12788 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 13), 'is', body_encoding_12786, QP_12787)
            
            # Testing if the type of an if condition is none (line 394)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 394, 13), result_is__12788):
                # Getting the type of 's' (line 397)
                s_12796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 's')
                # Assigning a type to the variable 'stypy_return_type' (line 397)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'stypy_return_type', s_12796)
            else:
                
                # Testing the type of an if condition (line 394)
                if_condition_12789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 13), result_is__12788)
                # Assigning a type to the variable 'if_condition_12789' (line 394)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 13), 'if_condition_12789', if_condition_12789)
                # SSA begins for if statement (line 394)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to body_encode(...): (line 395)
                # Processing the call arguments (line 395)
                # Getting the type of 's' (line 395)
                s_12793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 48), 's', False)
                # Processing the call keyword arguments (line 395)
                kwargs_12794 = {}
                # Getting the type of 'email' (line 395)
                email_12790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'email', False)
                # Obtaining the member 'quoprimime' of a type (line 395)
                quoprimime_12791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 19), email_12790, 'quoprimime')
                # Obtaining the member 'body_encode' of a type (line 395)
                body_encode_12792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 19), quoprimime_12791, 'body_encode')
                # Calling body_encode(args, kwargs) (line 395)
                body_encode_call_result_12795 = invoke(stypy.reporting.localization.Localization(__file__, 395, 19), body_encode_12792, *[s_12793], **kwargs_12794)
                
                # Assigning a type to the variable 'stypy_return_type' (line 395)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'stypy_return_type', body_encode_call_result_12795)
                # SSA branch for the else part of an if statement (line 394)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 's' (line 397)
                s_12796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 's')
                # Assigning a type to the variable 'stypy_return_type' (line 397)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'stypy_return_type', s_12796)
                # SSA join for if statement (line 394)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 392)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'body_encode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'body_encode' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_12797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12797)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'body_encode'
        return stypy_return_type_12797


# Assigning a type to the variable 'Charset' (line 156)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 0), 'Charset', Charset)

# Assigning a Name to a Name (line 241):
# Getting the type of 'Charset'
Charset_12798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Charset')
# Obtaining the member '__str__' of a type
str___12799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Charset_12798, '__str__')
# Getting the type of 'Charset'
Charset_12800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Charset')
# Setting the type of the member '__repr__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Charset_12800, '__repr__', str___12799)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
