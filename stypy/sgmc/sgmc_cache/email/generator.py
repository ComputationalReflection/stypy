
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2010 Python Software Foundation
2: # Contact: email-sig@python.org
3: 
4: '''Classes to generate plain text from a message object tree.'''
5: 
6: __all__ = ['Generator', 'DecodedGenerator']
7: 
8: import re
9: import sys
10: import time
11: import random
12: import warnings
13: 
14: from cStringIO import StringIO
15: from email.header import Header
16: 
17: UNDERSCORE = '_'
18: NL = '\n'
19: 
20: fcre = re.compile(r'^From ', re.MULTILINE)
21: 
22: def _is8bitstring(s):
23:     if isinstance(s, str):
24:         try:
25:             unicode(s, 'us-ascii')
26:         except UnicodeError:
27:             return True
28:     return False
29: 
30: 
31: 
32: class Generator:
33:     '''Generates output from a Message object tree.
34: 
35:     This basic generator writes the message to the given file object as plain
36:     text.
37:     '''
38:     #
39:     # Public interface
40:     #
41: 
42:     def __init__(self, outfp, mangle_from_=True, maxheaderlen=78):
43:         '''Create the generator for message flattening.
44: 
45:         outfp is the output file-like object for writing the message to.  It
46:         must have a write() method.
47: 
48:         Optional mangle_from_ is a flag that, when True (the default), escapes
49:         From_ lines in the body of the message by putting a `>' in front of
50:         them.
51: 
52:         Optional maxheaderlen specifies the longest length for a non-continued
53:         header.  When a header line is longer (in characters, with tabs
54:         expanded to 8 spaces) than maxheaderlen, the header will split as
55:         defined in the Header class.  Set maxheaderlen to zero to disable
56:         header wrapping.  The default is 78, as recommended (but not required)
57:         by RFC 2822.
58:         '''
59:         self._fp = outfp
60:         self._mangle_from_ = mangle_from_
61:         self._maxheaderlen = maxheaderlen
62: 
63:     def write(self, s):
64:         # Just delegate to the file object
65:         self._fp.write(s)
66: 
67:     def flatten(self, msg, unixfrom=False):
68:         '''Print the message object tree rooted at msg to the output file
69:         specified when the Generator instance was created.
70: 
71:         unixfrom is a flag that forces the printing of a Unix From_ delimiter
72:         before the first object in the message tree.  If the original message
73:         has no From_ delimiter, a `standard' one is crafted.  By default, this
74:         is False to inhibit the printing of any From_ delimiter.
75: 
76:         Note that for subobjects, no From_ line is printed.
77:         '''
78:         if unixfrom:
79:             ufrom = msg.get_unixfrom()
80:             if not ufrom:
81:                 ufrom = 'From nobody ' + time.ctime(time.time())
82:             print >> self._fp, ufrom
83:         self._write(msg)
84: 
85:     def clone(self, fp):
86:         '''Clone this generator with the exact same options.'''
87:         return self.__class__(fp, self._mangle_from_, self._maxheaderlen)
88: 
89:     #
90:     # Protected interface - undocumented ;/
91:     #
92: 
93:     def _write(self, msg):
94:         # We can't write the headers yet because of the following scenario:
95:         # say a multipart message includes the boundary string somewhere in
96:         # its body.  We'd have to calculate the new boundary /before/ we write
97:         # the headers so that we can write the correct Content-Type:
98:         # parameter.
99:         #
100:         # The way we do this, so as to make the _handle_*() methods simpler,
101:         # is to cache any subpart writes into a StringIO.  The we write the
102:         # headers and the StringIO contents.  That way, subpart handlers can
103:         # Do The Right Thing, and can still modify the Content-Type: header if
104:         # necessary.
105:         oldfp = self._fp
106:         try:
107:             self._fp = sfp = StringIO()
108:             self._dispatch(msg)
109:         finally:
110:             self._fp = oldfp
111:         # Write the headers.  First we see if the message object wants to
112:         # handle that itself.  If not, we'll do it generically.
113:         meth = getattr(msg, '_write_headers', None)
114:         if meth is None:
115:             self._write_headers(msg)
116:         else:
117:             meth(self)
118:         self._fp.write(sfp.getvalue())
119: 
120:     def _dispatch(self, msg):
121:         # Get the Content-Type: for the message, then try to dispatch to
122:         # self._handle_<maintype>_<subtype>().  If there's no handler for the
123:         # full MIME type, then dispatch to self._handle_<maintype>().  If
124:         # that's missing too, then dispatch to self._writeBody().
125:         main = msg.get_content_maintype()
126:         sub = msg.get_content_subtype()
127:         specific = UNDERSCORE.join((main, sub)).replace('-', '_')
128:         meth = getattr(self, '_handle_' + specific, None)
129:         if meth is None:
130:             generic = main.replace('-', '_')
131:             meth = getattr(self, '_handle_' + generic, None)
132:             if meth is None:
133:                 meth = self._writeBody
134:         meth(msg)
135: 
136:     #
137:     # Default handlers
138:     #
139: 
140:     def _write_headers(self, msg):
141:         for h, v in msg.items():
142:             print >> self._fp, '%s:' % h,
143:             if self._maxheaderlen == 0:
144:                 # Explicit no-wrapping
145:                 print >> self._fp, v
146:             elif isinstance(v, Header):
147:                 # Header instances know what to do
148:                 print >> self._fp, v.encode()
149:             elif _is8bitstring(v):
150:                 # If we have raw 8bit data in a byte string, we have no idea
151:                 # what the encoding is.  There is no safe way to split this
152:                 # string.  If it's ascii-subset, then we could do a normal
153:                 # ascii split, but if it's multibyte then we could break the
154:                 # string.  There's no way to know so the least harm seems to
155:                 # be to not split the string and risk it being too long.
156:                 print >> self._fp, v
157:             else:
158:                 # Header's got lots of smarts, so use it.  Note that this is
159:                 # fundamentally broken though because we lose idempotency when
160:                 # the header string is continued with tabs.  It will now be
161:                 # continued with spaces.  This was reversedly broken before we
162:                 # fixed bug 1974.  Either way, we lose.
163:                 print >> self._fp, Header(
164:                     v, maxlinelen=self._maxheaderlen, header_name=h).encode()
165:         # A blank line always separates headers from body
166:         print >> self._fp
167: 
168:     #
169:     # Handlers for writing types and subtypes
170:     #
171: 
172:     def _handle_text(self, msg):
173:         payload = msg.get_payload()
174:         if payload is None:
175:             return
176:         if not isinstance(payload, basestring):
177:             raise TypeError('string payload expected: %s' % type(payload))
178:         if self._mangle_from_:
179:             payload = fcre.sub('>From ', payload)
180:         self._fp.write(payload)
181: 
182:     # Default body handler
183:     _writeBody = _handle_text
184: 
185:     def _handle_multipart(self, msg):
186:         # The trick here is to write out each part separately, merge them all
187:         # together, and then make sure that the boundary we've chosen isn't
188:         # present in the payload.
189:         msgtexts = []
190:         subparts = msg.get_payload()
191:         if subparts is None:
192:             subparts = []
193:         elif isinstance(subparts, basestring):
194:             # e.g. a non-strict parse of a message with no starting boundary.
195:             self._fp.write(subparts)
196:             return
197:         elif not isinstance(subparts, list):
198:             # Scalar payload
199:             subparts = [subparts]
200:         for part in subparts:
201:             s = StringIO()
202:             g = self.clone(s)
203:             g.flatten(part, unixfrom=False)
204:             msgtexts.append(s.getvalue())
205:         # BAW: What about boundaries that are wrapped in double-quotes?
206:         boundary = msg.get_boundary()
207:         if not boundary:
208:             # Create a boundary that doesn't appear in any of the
209:             # message texts.
210:             alltext = NL.join(msgtexts)
211:             boundary = _make_boundary(alltext)
212:             msg.set_boundary(boundary)
213:         # If there's a preamble, write it out, with a trailing CRLF
214:         if msg.preamble is not None:
215:             if self._mangle_from_:
216:                 preamble = fcre.sub('>From ', msg.preamble)
217:             else:
218:                 preamble = msg.preamble
219:             print >> self._fp, preamble
220:         # dash-boundary transport-padding CRLF
221:         print >> self._fp, '--' + boundary
222:         # body-part
223:         if msgtexts:
224:             self._fp.write(msgtexts.pop(0))
225:         # *encapsulation
226:         # --> delimiter transport-padding
227:         # --> CRLF body-part
228:         for body_part in msgtexts:
229:             # delimiter transport-padding CRLF
230:             print >> self._fp, '\n--' + boundary
231:             # body-part
232:             self._fp.write(body_part)
233:         # close-delimiter transport-padding
234:         self._fp.write('\n--' + boundary + '--' + NL)
235:         if msg.epilogue is not None:
236:             if self._mangle_from_:
237:                 epilogue = fcre.sub('>From ', msg.epilogue)
238:             else:
239:                 epilogue = msg.epilogue
240:             self._fp.write(epilogue)
241: 
242:     def _handle_multipart_signed(self, msg):
243:         # The contents of signed parts has to stay unmodified in order to keep
244:         # the signature intact per RFC1847 2.1, so we disable header wrapping.
245:         # RDM: This isn't enough to completely preserve the part, but it helps.
246:         old_maxheaderlen = self._maxheaderlen
247:         try:
248:             self._maxheaderlen = 0
249:             self._handle_multipart(msg)
250:         finally:
251:             self._maxheaderlen = old_maxheaderlen
252: 
253:     def _handle_message_delivery_status(self, msg):
254:         # We can't just write the headers directly to self's file object
255:         # because this will leave an extra newline between the last header
256:         # block and the boundary.  Sigh.
257:         blocks = []
258:         for part in msg.get_payload():
259:             s = StringIO()
260:             g = self.clone(s)
261:             g.flatten(part, unixfrom=False)
262:             text = s.getvalue()
263:             lines = text.split('\n')
264:             # Strip off the unnecessary trailing empty line
265:             if lines and lines[-1] == '':
266:                 blocks.append(NL.join(lines[:-1]))
267:             else:
268:                 blocks.append(text)
269:         # Now join all the blocks with an empty line.  This has the lovely
270:         # effect of separating each block with an empty line, but not adding
271:         # an extra one after the last one.
272:         self._fp.write(NL.join(blocks))
273: 
274:     def _handle_message(self, msg):
275:         s = StringIO()
276:         g = self.clone(s)
277:         # The payload of a message/rfc822 part should be a multipart sequence
278:         # of length 1.  The zeroth element of the list should be the Message
279:         # object for the subpart.  Extract that object, stringify it, and
280:         # write it out.
281:         # Except, it turns out, when it's a string instead, which happens when
282:         # and only when HeaderParser is used on a message of mime type
283:         # message/rfc822.  Such messages are generated by, for example,
284:         # Groupwise when forwarding unadorned messages.  (Issue 7970.)  So
285:         # in that case we just emit the string body.
286:         payload = msg.get_payload()
287:         if isinstance(payload, list):
288:             g.flatten(msg.get_payload(0), unixfrom=False)
289:             payload = s.getvalue()
290:         self._fp.write(payload)
291: 
292: 
293: 
294: _FMT = '[Non-text (%(type)s) part of message omitted, filename %(filename)s]'
295: 
296: class DecodedGenerator(Generator):
297:     '''Generates a text representation of a message.
298: 
299:     Like the Generator base class, except that non-text parts are substituted
300:     with a format string representing the part.
301:     '''
302:     def __init__(self, outfp, mangle_from_=True, maxheaderlen=78, fmt=None):
303:         '''Like Generator.__init__() except that an additional optional
304:         argument is allowed.
305: 
306:         Walks through all subparts of a message.  If the subpart is of main
307:         type `text', then it prints the decoded payload of the subpart.
308: 
309:         Otherwise, fmt is a format string that is used instead of the message
310:         payload.  fmt is expanded with the following keywords (in
311:         %(keyword)s format):
312: 
313:         type       : Full MIME type of the non-text part
314:         maintype   : Main MIME type of the non-text part
315:         subtype    : Sub-MIME type of the non-text part
316:         filename   : Filename of the non-text part
317:         description: Description associated with the non-text part
318:         encoding   : Content transfer encoding of the non-text part
319: 
320:         The default value for fmt is None, meaning
321: 
322:         [Non-text (%(type)s) part of message omitted, filename %(filename)s]
323:         '''
324:         Generator.__init__(self, outfp, mangle_from_, maxheaderlen)
325:         if fmt is None:
326:             self._fmt = _FMT
327:         else:
328:             self._fmt = fmt
329: 
330:     def _dispatch(self, msg):
331:         for part in msg.walk():
332:             maintype = part.get_content_maintype()
333:             if maintype == 'text':
334:                 print >> self, part.get_payload(decode=True)
335:             elif maintype == 'multipart':
336:                 # Just skip this
337:                 pass
338:             else:
339:                 print >> self, self._fmt % {
340:                     'type'       : part.get_content_type(),
341:                     'maintype'   : part.get_content_maintype(),
342:                     'subtype'    : part.get_content_subtype(),
343:                     'filename'   : part.get_filename('[no filename]'),
344:                     'description': part.get('Content-Description',
345:                                             '[no description]'),
346:                     'encoding'   : part.get('Content-Transfer-Encoding',
347:                                             '[no encoding]'),
348:                     }
349: 
350: 
351: 
352: # Helper
353: _width = len(repr(sys.maxint-1))
354: _fmt = '%%0%dd' % _width
355: 
356: def _make_boundary(text=None):
357:     # Craft a random boundary.  If text is given, ensure that the chosen
358:     # boundary doesn't appear in the text.
359:     token = random.randrange(sys.maxint)
360:     boundary = ('=' * 15) + (_fmt % token) + '=='
361:     if text is None:
362:         return boundary
363:     b = boundary
364:     counter = 0
365:     while True:
366:         cre = re.compile('^--' + re.escape(b) + '(--)?$', re.MULTILINE)
367:         if not cre.search(text):
368:             break
369:         b = boundary + '.' + str(counter)
370:         counter += 1
371:     return b
372: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_14148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 0), 'str', 'Classes to generate plain text from a message object tree.')

# Assigning a List to a Name (line 6):
__all__ = ['Generator', 'DecodedGenerator']
module_type_store.set_exportable_members(['Generator', 'DecodedGenerator'])

# Obtaining an instance of the builtin type 'list' (line 6)
list_14149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_14150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'str', 'Generator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_14149, str_14150)
# Adding element type (line 6)
str_14151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 24), 'str', 'DecodedGenerator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_14149, str_14151)

# Assigning a type to the variable '__all__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__all__', list_14149)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import re' statement (line 8)
import re

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import time' statement (line 10)
import time

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import random' statement (line 11)
import random

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import warnings' statement (line 12)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from cStringIO import StringIO' statement (line 14)
try:
    from cStringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from email.header import Header' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_14152 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'email.header')

if (type(import_14152) is not StypyTypeError):

    if (import_14152 != 'pyd_module'):
        __import__(import_14152)
        sys_modules_14153 = sys.modules[import_14152]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'email.header', sys_modules_14153.module_type_store, module_type_store, ['Header'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_14153, sys_modules_14153.module_type_store, module_type_store)
    else:
        from email.header import Header

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'email.header', None, module_type_store, ['Header'], [Header])

else:
    # Assigning a type to the variable 'email.header' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'email.header', import_14152)

remove_current_file_folder_from_path('C:/Python27/lib/email/')


# Assigning a Str to a Name (line 17):
str_14154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'str', '_')
# Assigning a type to the variable 'UNDERSCORE' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'UNDERSCORE', str_14154)

# Assigning a Str to a Name (line 18):
str_14155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 5), 'str', '\n')
# Assigning a type to the variable 'NL' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'NL', str_14155)

# Assigning a Call to a Name (line 20):

# Call to compile(...): (line 20)
# Processing the call arguments (line 20)
str_14158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'str', '^From ')
# Getting the type of 're' (line 20)
re_14159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 29), 're', False)
# Obtaining the member 'MULTILINE' of a type (line 20)
MULTILINE_14160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 29), re_14159, 'MULTILINE')
# Processing the call keyword arguments (line 20)
kwargs_14161 = {}
# Getting the type of 're' (line 20)
re_14156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 're', False)
# Obtaining the member 'compile' of a type (line 20)
compile_14157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 7), re_14156, 'compile')
# Calling compile(args, kwargs) (line 20)
compile_call_result_14162 = invoke(stypy.reporting.localization.Localization(__file__, 20, 7), compile_14157, *[str_14158, MULTILINE_14160], **kwargs_14161)

# Assigning a type to the variable 'fcre' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'fcre', compile_call_result_14162)

@norecursion
def _is8bitstring(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_is8bitstring'
    module_type_store = module_type_store.open_function_context('_is8bitstring', 22, 0, False)
    
    # Passed parameters checking function
    _is8bitstring.stypy_localization = localization
    _is8bitstring.stypy_type_of_self = None
    _is8bitstring.stypy_type_store = module_type_store
    _is8bitstring.stypy_function_name = '_is8bitstring'
    _is8bitstring.stypy_param_names_list = ['s']
    _is8bitstring.stypy_varargs_param_name = None
    _is8bitstring.stypy_kwargs_param_name = None
    _is8bitstring.stypy_call_defaults = defaults
    _is8bitstring.stypy_call_varargs = varargs
    _is8bitstring.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_is8bitstring', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_is8bitstring', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_is8bitstring(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 23)
    # Getting the type of 'str' (line 23)
    str_14163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'str')
    # Getting the type of 's' (line 23)
    s_14164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 's')
    
    (may_be_14165, more_types_in_union_14166) = may_be_subtype(str_14163, s_14164)

    if may_be_14165:

        if more_types_in_union_14166:
            # Runtime conditional SSA (line 23)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 's' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 's', remove_not_subtype_from_union(s_14164, str))
        
        
        # SSA begins for try-except statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to unicode(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 's' (line 25)
        s_14168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 's', False)
        str_14169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'str', 'us-ascii')
        # Processing the call keyword arguments (line 25)
        kwargs_14170 = {}
        # Getting the type of 'unicode' (line 25)
        unicode_14167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'unicode', False)
        # Calling unicode(args, kwargs) (line 25)
        unicode_call_result_14171 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), unicode_14167, *[s_14168, str_14169], **kwargs_14170)
        
        # SSA branch for the except part of a try statement (line 24)
        # SSA branch for the except 'UnicodeError' branch of a try statement (line 24)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'True' (line 27)
        True_14172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'stypy_return_type', True_14172)
        # SSA join for try-except statement (line 24)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_14166:
            # SSA join for if statement (line 23)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'False' (line 28)
    False_14173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type', False_14173)
    
    # ################# End of '_is8bitstring(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_is8bitstring' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_14174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14174)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_is8bitstring'
    return stypy_return_type_14174

# Assigning a type to the variable '_is8bitstring' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '_is8bitstring', _is8bitstring)
# Declaration of the 'Generator' class

class Generator:
    str_14175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, (-1)), 'str', 'Generates output from a Message object tree.\n\n    This basic generator writes the message to the given file object as plain\n    text.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 42)
        True_14176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 43), 'True')
        int_14177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 62), 'int')
        defaults = [True_14176, int_14177]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator.__init__', ['outfp', 'mangle_from_', 'maxheaderlen'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['outfp', 'mangle_from_', 'maxheaderlen'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_14178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', "Create the generator for message flattening.\n\n        outfp is the output file-like object for writing the message to.  It\n        must have a write() method.\n\n        Optional mangle_from_ is a flag that, when True (the default), escapes\n        From_ lines in the body of the message by putting a `>' in front of\n        them.\n\n        Optional maxheaderlen specifies the longest length for a non-continued\n        header.  When a header line is longer (in characters, with tabs\n        expanded to 8 spaces) than maxheaderlen, the header will split as\n        defined in the Header class.  Set maxheaderlen to zero to disable\n        header wrapping.  The default is 78, as recommended (but not required)\n        by RFC 2822.\n        ")
        
        # Assigning a Name to a Attribute (line 59):
        # Getting the type of 'outfp' (line 59)
        outfp_14179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'outfp')
        # Getting the type of 'self' (line 59)
        self_14180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member '_fp' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_14180, '_fp', outfp_14179)
        
        # Assigning a Name to a Attribute (line 60):
        # Getting the type of 'mangle_from_' (line 60)
        mangle_from__14181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'mangle_from_')
        # Getting the type of 'self' (line 60)
        self_14182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
        # Setting the type of the member '_mangle_from_' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_14182, '_mangle_from_', mangle_from__14181)
        
        # Assigning a Name to a Attribute (line 61):
        # Getting the type of 'maxheaderlen' (line 61)
        maxheaderlen_14183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'maxheaderlen')
        # Getting the type of 'self' (line 61)
        self_14184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member '_maxheaderlen' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_14184, '_maxheaderlen', maxheaderlen_14183)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write'
        module_type_store = module_type_store.open_function_context('write', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator.write.__dict__.__setitem__('stypy_localization', localization)
        Generator.write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator.write.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator.write.__dict__.__setitem__('stypy_function_name', 'Generator.write')
        Generator.write.__dict__.__setitem__('stypy_param_names_list', ['s'])
        Generator.write.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator.write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator.write.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator.write.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator.write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator.write.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator.write', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write(...)' code ##################

        
        # Call to write(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 's' (line 65)
        s_14188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 's', False)
        # Processing the call keyword arguments (line 65)
        kwargs_14189 = {}
        # Getting the type of 'self' (line 65)
        self_14185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self', False)
        # Obtaining the member '_fp' of a type (line 65)
        _fp_14186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_14185, '_fp')
        # Obtaining the member 'write' of a type (line 65)
        write_14187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), _fp_14186, 'write')
        # Calling write(args, kwargs) (line 65)
        write_call_result_14190 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), write_14187, *[s_14188], **kwargs_14189)
        
        
        # ################# End of 'write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_14191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14191)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write'
        return stypy_return_type_14191


    @norecursion
    def flatten(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 67)
        False_14192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 36), 'False')
        defaults = [False_14192]
        # Create a new context for function 'flatten'
        module_type_store = module_type_store.open_function_context('flatten', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator.flatten.__dict__.__setitem__('stypy_localization', localization)
        Generator.flatten.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator.flatten.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator.flatten.__dict__.__setitem__('stypy_function_name', 'Generator.flatten')
        Generator.flatten.__dict__.__setitem__('stypy_param_names_list', ['msg', 'unixfrom'])
        Generator.flatten.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator.flatten.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator.flatten.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator.flatten.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator.flatten.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator.flatten.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator.flatten', ['msg', 'unixfrom'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'flatten', localization, ['msg', 'unixfrom'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'flatten(...)' code ##################

        str_14193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', "Print the message object tree rooted at msg to the output file\n        specified when the Generator instance was created.\n\n        unixfrom is a flag that forces the printing of a Unix From_ delimiter\n        before the first object in the message tree.  If the original message\n        has no From_ delimiter, a `standard' one is crafted.  By default, this\n        is False to inhibit the printing of any From_ delimiter.\n\n        Note that for subobjects, no From_ line is printed.\n        ")
        # Getting the type of 'unixfrom' (line 78)
        unixfrom_14194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'unixfrom')
        # Testing if the type of an if condition is none (line 78)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 78, 8), unixfrom_14194):
            pass
        else:
            
            # Testing the type of an if condition (line 78)
            if_condition_14195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), unixfrom_14194)
            # Assigning a type to the variable 'if_condition_14195' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_14195', if_condition_14195)
            # SSA begins for if statement (line 78)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 79):
            
            # Call to get_unixfrom(...): (line 79)
            # Processing the call keyword arguments (line 79)
            kwargs_14198 = {}
            # Getting the type of 'msg' (line 79)
            msg_14196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'msg', False)
            # Obtaining the member 'get_unixfrom' of a type (line 79)
            get_unixfrom_14197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), msg_14196, 'get_unixfrom')
            # Calling get_unixfrom(args, kwargs) (line 79)
            get_unixfrom_call_result_14199 = invoke(stypy.reporting.localization.Localization(__file__, 79, 20), get_unixfrom_14197, *[], **kwargs_14198)
            
            # Assigning a type to the variable 'ufrom' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'ufrom', get_unixfrom_call_result_14199)
            
            # Getting the type of 'ufrom' (line 80)
            ufrom_14200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'ufrom')
            # Applying the 'not' unary operator (line 80)
            result_not__14201 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), 'not', ufrom_14200)
            
            # Testing if the type of an if condition is none (line 80)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 12), result_not__14201):
                pass
            else:
                
                # Testing the type of an if condition (line 80)
                if_condition_14202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 12), result_not__14201)
                # Assigning a type to the variable 'if_condition_14202' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'if_condition_14202', if_condition_14202)
                # SSA begins for if statement (line 80)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 81):
                str_14203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 24), 'str', 'From nobody ')
                
                # Call to ctime(...): (line 81)
                # Processing the call arguments (line 81)
                
                # Call to time(...): (line 81)
                # Processing the call keyword arguments (line 81)
                kwargs_14208 = {}
                # Getting the type of 'time' (line 81)
                time_14206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 52), 'time', False)
                # Obtaining the member 'time' of a type (line 81)
                time_14207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 52), time_14206, 'time')
                # Calling time(args, kwargs) (line 81)
                time_call_result_14209 = invoke(stypy.reporting.localization.Localization(__file__, 81, 52), time_14207, *[], **kwargs_14208)
                
                # Processing the call keyword arguments (line 81)
                kwargs_14210 = {}
                # Getting the type of 'time' (line 81)
                time_14204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 41), 'time', False)
                # Obtaining the member 'ctime' of a type (line 81)
                ctime_14205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 41), time_14204, 'ctime')
                # Calling ctime(args, kwargs) (line 81)
                ctime_call_result_14211 = invoke(stypy.reporting.localization.Localization(__file__, 81, 41), ctime_14205, *[time_call_result_14209], **kwargs_14210)
                
                # Applying the binary operator '+' (line 81)
                result_add_14212 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 24), '+', str_14203, ctime_call_result_14211)
                
                # Assigning a type to the variable 'ufrom' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'ufrom', result_add_14212)
                # SSA join for if statement (line 80)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'ufrom' (line 82)
            ufrom_14213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 31), 'ufrom')
            # SSA join for if statement (line 78)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to _write(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'msg' (line 83)
        msg_14216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'msg', False)
        # Processing the call keyword arguments (line 83)
        kwargs_14217 = {}
        # Getting the type of 'self' (line 83)
        self_14214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self', False)
        # Obtaining the member '_write' of a type (line 83)
        _write_14215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_14214, '_write')
        # Calling _write(args, kwargs) (line 83)
        _write_call_result_14218 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), _write_14215, *[msg_14216], **kwargs_14217)
        
        
        # ################# End of 'flatten(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flatten' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_14219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flatten'
        return stypy_return_type_14219


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator.clone.__dict__.__setitem__('stypy_localization', localization)
        Generator.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator.clone.__dict__.__setitem__('stypy_function_name', 'Generator.clone')
        Generator.clone.__dict__.__setitem__('stypy_param_names_list', ['fp'])
        Generator.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator.clone.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator.clone', ['fp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clone', localization, ['fp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clone(...)' code ##################

        str_14220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'str', 'Clone this generator with the exact same options.')
        
        # Call to __class__(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'fp' (line 87)
        fp_14223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 'fp', False)
        # Getting the type of 'self' (line 87)
        self_14224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 34), 'self', False)
        # Obtaining the member '_mangle_from_' of a type (line 87)
        _mangle_from__14225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 34), self_14224, '_mangle_from_')
        # Getting the type of 'self' (line 87)
        self_14226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 54), 'self', False)
        # Obtaining the member '_maxheaderlen' of a type (line 87)
        _maxheaderlen_14227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 54), self_14226, '_maxheaderlen')
        # Processing the call keyword arguments (line 87)
        kwargs_14228 = {}
        # Getting the type of 'self' (line 87)
        self_14221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 87)
        class___14222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), self_14221, '__class__')
        # Calling __class__(args, kwargs) (line 87)
        class___call_result_14229 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), class___14222, *[fp_14223, _mangle_from__14225, _maxheaderlen_14227], **kwargs_14228)
        
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', class___call_result_14229)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_14230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14230)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_14230


    @norecursion
    def _write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write'
        module_type_store = module_type_store.open_function_context('_write', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator._write.__dict__.__setitem__('stypy_localization', localization)
        Generator._write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator._write.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator._write.__dict__.__setitem__('stypy_function_name', 'Generator._write')
        Generator._write.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Generator._write.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator._write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator._write.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator._write.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator._write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator._write.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator._write', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write(...)' code ##################

        
        # Assigning a Attribute to a Name (line 105):
        # Getting the type of 'self' (line 105)
        self_14231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'self')
        # Obtaining the member '_fp' of a type (line 105)
        _fp_14232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), self_14231, '_fp')
        # Assigning a type to the variable 'oldfp' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'oldfp', _fp_14232)
        
        # Try-finally block (line 106)
        
        # Multiple assignment of 2 elements.
        
        # Call to StringIO(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_14234 = {}
        # Getting the type of 'StringIO' (line 107)
        StringIO_14233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 107)
        StringIO_call_result_14235 = invoke(stypy.reporting.localization.Localization(__file__, 107, 29), StringIO_14233, *[], **kwargs_14234)
        
        # Assigning a type to the variable 'sfp' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'sfp', StringIO_call_result_14235)
        # Getting the type of 'sfp' (line 107)
        sfp_14236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'sfp')
        # Getting the type of 'self' (line 107)
        self_14237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'self')
        # Setting the type of the member '_fp' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), self_14237, '_fp', sfp_14236)
        
        # Call to _dispatch(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'msg' (line 108)
        msg_14240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'msg', False)
        # Processing the call keyword arguments (line 108)
        kwargs_14241 = {}
        # Getting the type of 'self' (line 108)
        self_14238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'self', False)
        # Obtaining the member '_dispatch' of a type (line 108)
        _dispatch_14239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), self_14238, '_dispatch')
        # Calling _dispatch(args, kwargs) (line 108)
        _dispatch_call_result_14242 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), _dispatch_14239, *[msg_14240], **kwargs_14241)
        
        
        # finally branch of the try-finally block (line 106)
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'oldfp' (line 110)
        oldfp_14243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'oldfp')
        # Getting the type of 'self' (line 110)
        self_14244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'self')
        # Setting the type of the member '_fp' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), self_14244, '_fp', oldfp_14243)
        
        
        # Assigning a Call to a Name (line 113):
        
        # Call to getattr(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'msg' (line 113)
        msg_14246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'msg', False)
        str_14247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 28), 'str', '_write_headers')
        # Getting the type of 'None' (line 113)
        None_14248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 46), 'None', False)
        # Processing the call keyword arguments (line 113)
        kwargs_14249 = {}
        # Getting the type of 'getattr' (line 113)
        getattr_14245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 113)
        getattr_call_result_14250 = invoke(stypy.reporting.localization.Localization(__file__, 113, 15), getattr_14245, *[msg_14246, str_14247, None_14248], **kwargs_14249)
        
        # Assigning a type to the variable 'meth' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'meth', getattr_call_result_14250)
        
        # Type idiom detected: calculating its left and rigth part (line 114)
        # Getting the type of 'meth' (line 114)
        meth_14251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'meth')
        # Getting the type of 'None' (line 114)
        None_14252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 19), 'None')
        
        (may_be_14253, more_types_in_union_14254) = may_be_none(meth_14251, None_14252)

        if may_be_14253:

            if more_types_in_union_14254:
                # Runtime conditional SSA (line 114)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _write_headers(...): (line 115)
            # Processing the call arguments (line 115)
            # Getting the type of 'msg' (line 115)
            msg_14257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'msg', False)
            # Processing the call keyword arguments (line 115)
            kwargs_14258 = {}
            # Getting the type of 'self' (line 115)
            self_14255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'self', False)
            # Obtaining the member '_write_headers' of a type (line 115)
            _write_headers_14256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), self_14255, '_write_headers')
            # Calling _write_headers(args, kwargs) (line 115)
            _write_headers_call_result_14259 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), _write_headers_14256, *[msg_14257], **kwargs_14258)
            

            if more_types_in_union_14254:
                # Runtime conditional SSA for else branch (line 114)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_14253) or more_types_in_union_14254):
            
            # Call to meth(...): (line 117)
            # Processing the call arguments (line 117)
            # Getting the type of 'self' (line 117)
            self_14261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'self', False)
            # Processing the call keyword arguments (line 117)
            kwargs_14262 = {}
            # Getting the type of 'meth' (line 117)
            meth_14260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'meth', False)
            # Calling meth(args, kwargs) (line 117)
            meth_call_result_14263 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), meth_14260, *[self_14261], **kwargs_14262)
            

            if (may_be_14253 and more_types_in_union_14254):
                # SSA join for if statement (line 114)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to write(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to getvalue(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_14269 = {}
        # Getting the type of 'sfp' (line 118)
        sfp_14267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 23), 'sfp', False)
        # Obtaining the member 'getvalue' of a type (line 118)
        getvalue_14268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 23), sfp_14267, 'getvalue')
        # Calling getvalue(args, kwargs) (line 118)
        getvalue_call_result_14270 = invoke(stypy.reporting.localization.Localization(__file__, 118, 23), getvalue_14268, *[], **kwargs_14269)
        
        # Processing the call keyword arguments (line 118)
        kwargs_14271 = {}
        # Getting the type of 'self' (line 118)
        self_14264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self', False)
        # Obtaining the member '_fp' of a type (line 118)
        _fp_14265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_14264, '_fp')
        # Obtaining the member 'write' of a type (line 118)
        write_14266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), _fp_14265, 'write')
        # Calling write(args, kwargs) (line 118)
        write_call_result_14272 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), write_14266, *[getvalue_call_result_14270], **kwargs_14271)
        
        
        # ################# End of '_write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_14273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14273)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write'
        return stypy_return_type_14273


    @norecursion
    def _dispatch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dispatch'
        module_type_store = module_type_store.open_function_context('_dispatch', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator._dispatch.__dict__.__setitem__('stypy_localization', localization)
        Generator._dispatch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator._dispatch.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator._dispatch.__dict__.__setitem__('stypy_function_name', 'Generator._dispatch')
        Generator._dispatch.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Generator._dispatch.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator._dispatch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator._dispatch.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator._dispatch.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator._dispatch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator._dispatch.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator._dispatch', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dispatch', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dispatch(...)' code ##################

        
        # Assigning a Call to a Name (line 125):
        
        # Call to get_content_maintype(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_14276 = {}
        # Getting the type of 'msg' (line 125)
        msg_14274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'msg', False)
        # Obtaining the member 'get_content_maintype' of a type (line 125)
        get_content_maintype_14275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 15), msg_14274, 'get_content_maintype')
        # Calling get_content_maintype(args, kwargs) (line 125)
        get_content_maintype_call_result_14277 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), get_content_maintype_14275, *[], **kwargs_14276)
        
        # Assigning a type to the variable 'main' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'main', get_content_maintype_call_result_14277)
        
        # Assigning a Call to a Name (line 126):
        
        # Call to get_content_subtype(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_14280 = {}
        # Getting the type of 'msg' (line 126)
        msg_14278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'msg', False)
        # Obtaining the member 'get_content_subtype' of a type (line 126)
        get_content_subtype_14279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 14), msg_14278, 'get_content_subtype')
        # Calling get_content_subtype(args, kwargs) (line 126)
        get_content_subtype_call_result_14281 = invoke(stypy.reporting.localization.Localization(__file__, 126, 14), get_content_subtype_14279, *[], **kwargs_14280)
        
        # Assigning a type to the variable 'sub' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'sub', get_content_subtype_call_result_14281)
        
        # Assigning a Call to a Name (line 127):
        
        # Call to replace(...): (line 127)
        # Processing the call arguments (line 127)
        str_14290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 56), 'str', '-')
        str_14291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 61), 'str', '_')
        # Processing the call keyword arguments (line 127)
        kwargs_14292 = {}
        
        # Call to join(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_14284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        # Getting the type of 'main' (line 127)
        main_14285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 36), 'main', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 36), tuple_14284, main_14285)
        # Adding element type (line 127)
        # Getting the type of 'sub' (line 127)
        sub_14286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 42), 'sub', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 36), tuple_14284, sub_14286)
        
        # Processing the call keyword arguments (line 127)
        kwargs_14287 = {}
        # Getting the type of 'UNDERSCORE' (line 127)
        UNDERSCORE_14282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'UNDERSCORE', False)
        # Obtaining the member 'join' of a type (line 127)
        join_14283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), UNDERSCORE_14282, 'join')
        # Calling join(args, kwargs) (line 127)
        join_call_result_14288 = invoke(stypy.reporting.localization.Localization(__file__, 127, 19), join_14283, *[tuple_14284], **kwargs_14287)
        
        # Obtaining the member 'replace' of a type (line 127)
        replace_14289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), join_call_result_14288, 'replace')
        # Calling replace(args, kwargs) (line 127)
        replace_call_result_14293 = invoke(stypy.reporting.localization.Localization(__file__, 127, 19), replace_14289, *[str_14290, str_14291], **kwargs_14292)
        
        # Assigning a type to the variable 'specific' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'specific', replace_call_result_14293)
        
        # Assigning a Call to a Name (line 128):
        
        # Call to getattr(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'self' (line 128)
        self_14295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'self', False)
        str_14296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 29), 'str', '_handle_')
        # Getting the type of 'specific' (line 128)
        specific_14297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'specific', False)
        # Applying the binary operator '+' (line 128)
        result_add_14298 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 29), '+', str_14296, specific_14297)
        
        # Getting the type of 'None' (line 128)
        None_14299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 52), 'None', False)
        # Processing the call keyword arguments (line 128)
        kwargs_14300 = {}
        # Getting the type of 'getattr' (line 128)
        getattr_14294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 128)
        getattr_call_result_14301 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), getattr_14294, *[self_14295, result_add_14298, None_14299], **kwargs_14300)
        
        # Assigning a type to the variable 'meth' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'meth', getattr_call_result_14301)
        
        # Type idiom detected: calculating its left and rigth part (line 129)
        # Getting the type of 'meth' (line 129)
        meth_14302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'meth')
        # Getting the type of 'None' (line 129)
        None_14303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 'None')
        
        (may_be_14304, more_types_in_union_14305) = may_be_none(meth_14302, None_14303)

        if may_be_14304:

            if more_types_in_union_14305:
                # Runtime conditional SSA (line 129)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 130):
            
            # Call to replace(...): (line 130)
            # Processing the call arguments (line 130)
            str_14308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 35), 'str', '-')
            str_14309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 40), 'str', '_')
            # Processing the call keyword arguments (line 130)
            kwargs_14310 = {}
            # Getting the type of 'main' (line 130)
            main_14306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'main', False)
            # Obtaining the member 'replace' of a type (line 130)
            replace_14307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 22), main_14306, 'replace')
            # Calling replace(args, kwargs) (line 130)
            replace_call_result_14311 = invoke(stypy.reporting.localization.Localization(__file__, 130, 22), replace_14307, *[str_14308, str_14309], **kwargs_14310)
            
            # Assigning a type to the variable 'generic' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'generic', replace_call_result_14311)
            
            # Assigning a Call to a Name (line 131):
            
            # Call to getattr(...): (line 131)
            # Processing the call arguments (line 131)
            # Getting the type of 'self' (line 131)
            self_14313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'self', False)
            str_14314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 33), 'str', '_handle_')
            # Getting the type of 'generic' (line 131)
            generic_14315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 46), 'generic', False)
            # Applying the binary operator '+' (line 131)
            result_add_14316 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 33), '+', str_14314, generic_14315)
            
            # Getting the type of 'None' (line 131)
            None_14317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 55), 'None', False)
            # Processing the call keyword arguments (line 131)
            kwargs_14318 = {}
            # Getting the type of 'getattr' (line 131)
            getattr_14312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'getattr', False)
            # Calling getattr(args, kwargs) (line 131)
            getattr_call_result_14319 = invoke(stypy.reporting.localization.Localization(__file__, 131, 19), getattr_14312, *[self_14313, result_add_14316, None_14317], **kwargs_14318)
            
            # Assigning a type to the variable 'meth' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'meth', getattr_call_result_14319)
            
            # Type idiom detected: calculating its left and rigth part (line 132)
            # Getting the type of 'meth' (line 132)
            meth_14320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'meth')
            # Getting the type of 'None' (line 132)
            None_14321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 23), 'None')
            
            (may_be_14322, more_types_in_union_14323) = may_be_none(meth_14320, None_14321)

            if may_be_14322:

                if more_types_in_union_14323:
                    # Runtime conditional SSA (line 132)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Attribute to a Name (line 133):
                # Getting the type of 'self' (line 133)
                self_14324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'self')
                # Obtaining the member '_writeBody' of a type (line 133)
                _writeBody_14325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), self_14324, '_writeBody')
                # Assigning a type to the variable 'meth' (line 133)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'meth', _writeBody_14325)

                if more_types_in_union_14323:
                    # SSA join for if statement (line 132)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_14305:
                # SSA join for if statement (line 129)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to meth(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'msg' (line 134)
        msg_14327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 13), 'msg', False)
        # Processing the call keyword arguments (line 134)
        kwargs_14328 = {}
        # Getting the type of 'meth' (line 134)
        meth_14326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'meth', False)
        # Calling meth(args, kwargs) (line 134)
        meth_call_result_14329 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), meth_14326, *[msg_14327], **kwargs_14328)
        
        
        # ################# End of '_dispatch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dispatch' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_14330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14330)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dispatch'
        return stypy_return_type_14330


    @norecursion
    def _write_headers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_headers'
        module_type_store = module_type_store.open_function_context('_write_headers', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator._write_headers.__dict__.__setitem__('stypy_localization', localization)
        Generator._write_headers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator._write_headers.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator._write_headers.__dict__.__setitem__('stypy_function_name', 'Generator._write_headers')
        Generator._write_headers.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Generator._write_headers.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator._write_headers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator._write_headers.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator._write_headers.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator._write_headers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator._write_headers.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator._write_headers', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_headers', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_headers(...)' code ##################

        
        
        # Call to items(...): (line 141)
        # Processing the call keyword arguments (line 141)
        kwargs_14333 = {}
        # Getting the type of 'msg' (line 141)
        msg_14331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'msg', False)
        # Obtaining the member 'items' of a type (line 141)
        items_14332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 20), msg_14331, 'items')
        # Calling items(args, kwargs) (line 141)
        items_call_result_14334 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), items_14332, *[], **kwargs_14333)
        
        # Assigning a type to the variable 'items_call_result_14334' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'items_call_result_14334', items_call_result_14334)
        # Testing if the for loop is going to be iterated (line 141)
        # Testing the type of a for loop iterable (line 141)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 8), items_call_result_14334)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 141, 8), items_call_result_14334):
            # Getting the type of the for loop variable (line 141)
            for_loop_var_14335 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 8), items_call_result_14334)
            # Assigning a type to the variable 'h' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'h', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 8), for_loop_var_14335, 2, 0))
            # Assigning a type to the variable 'v' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 8), for_loop_var_14335, 2, 1))
            # SSA begins for a for statement (line 141)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            str_14336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 31), 'str', '%s:')
            # Getting the type of 'h' (line 142)
            h_14337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 'h')
            # Applying the binary operator '%' (line 142)
            result_mod_14338 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 31), '%', str_14336, h_14337)
            
            
            # Getting the type of 'self' (line 143)
            self_14339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'self')
            # Obtaining the member '_maxheaderlen' of a type (line 143)
            _maxheaderlen_14340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), self_14339, '_maxheaderlen')
            int_14341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 37), 'int')
            # Applying the binary operator '==' (line 143)
            result_eq_14342 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 15), '==', _maxheaderlen_14340, int_14341)
            
            # Testing if the type of an if condition is none (line 143)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 143, 12), result_eq_14342):
                
                # Call to isinstance(...): (line 146)
                # Processing the call arguments (line 146)
                # Getting the type of 'v' (line 146)
                v_14346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'v', False)
                # Getting the type of 'Header' (line 146)
                Header_14347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 31), 'Header', False)
                # Processing the call keyword arguments (line 146)
                kwargs_14348 = {}
                # Getting the type of 'isinstance' (line 146)
                isinstance_14345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 146)
                isinstance_call_result_14349 = invoke(stypy.reporting.localization.Localization(__file__, 146, 17), isinstance_14345, *[v_14346, Header_14347], **kwargs_14348)
                
                # Testing if the type of an if condition is none (line 146)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 17), isinstance_call_result_14349):
                    
                    # Call to _is8bitstring(...): (line 149)
                    # Processing the call arguments (line 149)
                    # Getting the type of 'v' (line 149)
                    v_14356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 31), 'v', False)
                    # Processing the call keyword arguments (line 149)
                    kwargs_14357 = {}
                    # Getting the type of '_is8bitstring' (line 149)
                    _is8bitstring_14355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), '_is8bitstring', False)
                    # Calling _is8bitstring(args, kwargs) (line 149)
                    _is8bitstring_call_result_14358 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_14355, *[v_14356], **kwargs_14357)
                    
                    # Testing if the type of an if condition is none (line 149)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_call_result_14358):
                        
                        # Call to encode(...): (line 163)
                        # Processing the call keyword arguments (line 163)
                        kwargs_14371 = {}
                        
                        # Call to Header(...): (line 163)
                        # Processing the call arguments (line 163)
                        # Getting the type of 'v' (line 164)
                        v_14362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'v', False)
                        # Processing the call keyword arguments (line 163)
                        # Getting the type of 'self' (line 164)
                        self_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'self', False)
                        # Obtaining the member '_maxheaderlen' of a type (line 164)
                        _maxheaderlen_14364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), self_14363, '_maxheaderlen')
                        keyword_14365 = _maxheaderlen_14364
                        # Getting the type of 'h' (line 164)
                        h_14366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'h', False)
                        keyword_14367 = h_14366
                        kwargs_14368 = {'maxlinelen': keyword_14365, 'header_name': keyword_14367}
                        # Getting the type of 'Header' (line 163)
                        Header_14361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'Header', False)
                        # Calling Header(args, kwargs) (line 163)
                        Header_call_result_14369 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), Header_14361, *[v_14362], **kwargs_14368)
                        
                        # Obtaining the member 'encode' of a type (line 163)
                        encode_14370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 35), Header_call_result_14369, 'encode')
                        # Calling encode(args, kwargs) (line 163)
                        encode_call_result_14372 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), encode_14370, *[], **kwargs_14371)
                        
                    else:
                        
                        # Testing the type of an if condition (line 149)
                        if_condition_14359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_call_result_14358)
                        # Assigning a type to the variable 'if_condition_14359' (line 149)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'if_condition_14359', if_condition_14359)
                        # SSA begins for if statement (line 149)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'v' (line 156)
                        v_14360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 35), 'v')
                        # SSA branch for the else part of an if statement (line 149)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to encode(...): (line 163)
                        # Processing the call keyword arguments (line 163)
                        kwargs_14371 = {}
                        
                        # Call to Header(...): (line 163)
                        # Processing the call arguments (line 163)
                        # Getting the type of 'v' (line 164)
                        v_14362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'v', False)
                        # Processing the call keyword arguments (line 163)
                        # Getting the type of 'self' (line 164)
                        self_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'self', False)
                        # Obtaining the member '_maxheaderlen' of a type (line 164)
                        _maxheaderlen_14364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), self_14363, '_maxheaderlen')
                        keyword_14365 = _maxheaderlen_14364
                        # Getting the type of 'h' (line 164)
                        h_14366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'h', False)
                        keyword_14367 = h_14366
                        kwargs_14368 = {'maxlinelen': keyword_14365, 'header_name': keyword_14367}
                        # Getting the type of 'Header' (line 163)
                        Header_14361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'Header', False)
                        # Calling Header(args, kwargs) (line 163)
                        Header_call_result_14369 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), Header_14361, *[v_14362], **kwargs_14368)
                        
                        # Obtaining the member 'encode' of a type (line 163)
                        encode_14370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 35), Header_call_result_14369, 'encode')
                        # Calling encode(args, kwargs) (line 163)
                        encode_call_result_14372 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), encode_14370, *[], **kwargs_14371)
                        
                        # SSA join for if statement (line 149)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 146)
                    if_condition_14350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 17), isinstance_call_result_14349)
                    # Assigning a type to the variable 'if_condition_14350' (line 146)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'if_condition_14350', if_condition_14350)
                    # SSA begins for if statement (line 146)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to encode(...): (line 148)
                    # Processing the call keyword arguments (line 148)
                    kwargs_14353 = {}
                    # Getting the type of 'v' (line 148)
                    v_14351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'v', False)
                    # Obtaining the member 'encode' of a type (line 148)
                    encode_14352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 35), v_14351, 'encode')
                    # Calling encode(args, kwargs) (line 148)
                    encode_call_result_14354 = invoke(stypy.reporting.localization.Localization(__file__, 148, 35), encode_14352, *[], **kwargs_14353)
                    
                    # SSA branch for the else part of an if statement (line 146)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to _is8bitstring(...): (line 149)
                    # Processing the call arguments (line 149)
                    # Getting the type of 'v' (line 149)
                    v_14356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 31), 'v', False)
                    # Processing the call keyword arguments (line 149)
                    kwargs_14357 = {}
                    # Getting the type of '_is8bitstring' (line 149)
                    _is8bitstring_14355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), '_is8bitstring', False)
                    # Calling _is8bitstring(args, kwargs) (line 149)
                    _is8bitstring_call_result_14358 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_14355, *[v_14356], **kwargs_14357)
                    
                    # Testing if the type of an if condition is none (line 149)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_call_result_14358):
                        
                        # Call to encode(...): (line 163)
                        # Processing the call keyword arguments (line 163)
                        kwargs_14371 = {}
                        
                        # Call to Header(...): (line 163)
                        # Processing the call arguments (line 163)
                        # Getting the type of 'v' (line 164)
                        v_14362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'v', False)
                        # Processing the call keyword arguments (line 163)
                        # Getting the type of 'self' (line 164)
                        self_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'self', False)
                        # Obtaining the member '_maxheaderlen' of a type (line 164)
                        _maxheaderlen_14364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), self_14363, '_maxheaderlen')
                        keyword_14365 = _maxheaderlen_14364
                        # Getting the type of 'h' (line 164)
                        h_14366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'h', False)
                        keyword_14367 = h_14366
                        kwargs_14368 = {'maxlinelen': keyword_14365, 'header_name': keyword_14367}
                        # Getting the type of 'Header' (line 163)
                        Header_14361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'Header', False)
                        # Calling Header(args, kwargs) (line 163)
                        Header_call_result_14369 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), Header_14361, *[v_14362], **kwargs_14368)
                        
                        # Obtaining the member 'encode' of a type (line 163)
                        encode_14370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 35), Header_call_result_14369, 'encode')
                        # Calling encode(args, kwargs) (line 163)
                        encode_call_result_14372 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), encode_14370, *[], **kwargs_14371)
                        
                    else:
                        
                        # Testing the type of an if condition (line 149)
                        if_condition_14359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_call_result_14358)
                        # Assigning a type to the variable 'if_condition_14359' (line 149)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'if_condition_14359', if_condition_14359)
                        # SSA begins for if statement (line 149)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'v' (line 156)
                        v_14360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 35), 'v')
                        # SSA branch for the else part of an if statement (line 149)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to encode(...): (line 163)
                        # Processing the call keyword arguments (line 163)
                        kwargs_14371 = {}
                        
                        # Call to Header(...): (line 163)
                        # Processing the call arguments (line 163)
                        # Getting the type of 'v' (line 164)
                        v_14362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'v', False)
                        # Processing the call keyword arguments (line 163)
                        # Getting the type of 'self' (line 164)
                        self_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'self', False)
                        # Obtaining the member '_maxheaderlen' of a type (line 164)
                        _maxheaderlen_14364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), self_14363, '_maxheaderlen')
                        keyword_14365 = _maxheaderlen_14364
                        # Getting the type of 'h' (line 164)
                        h_14366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'h', False)
                        keyword_14367 = h_14366
                        kwargs_14368 = {'maxlinelen': keyword_14365, 'header_name': keyword_14367}
                        # Getting the type of 'Header' (line 163)
                        Header_14361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'Header', False)
                        # Calling Header(args, kwargs) (line 163)
                        Header_call_result_14369 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), Header_14361, *[v_14362], **kwargs_14368)
                        
                        # Obtaining the member 'encode' of a type (line 163)
                        encode_14370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 35), Header_call_result_14369, 'encode')
                        # Calling encode(args, kwargs) (line 163)
                        encode_call_result_14372 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), encode_14370, *[], **kwargs_14371)
                        
                        # SSA join for if statement (line 149)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 146)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 143)
                if_condition_14343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 12), result_eq_14342)
                # Assigning a type to the variable 'if_condition_14343' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'if_condition_14343', if_condition_14343)
                # SSA begins for if statement (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'v' (line 145)
                v_14344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 35), 'v')
                # SSA branch for the else part of an if statement (line 143)
                module_type_store.open_ssa_branch('else')
                
                # Call to isinstance(...): (line 146)
                # Processing the call arguments (line 146)
                # Getting the type of 'v' (line 146)
                v_14346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'v', False)
                # Getting the type of 'Header' (line 146)
                Header_14347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 31), 'Header', False)
                # Processing the call keyword arguments (line 146)
                kwargs_14348 = {}
                # Getting the type of 'isinstance' (line 146)
                isinstance_14345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 146)
                isinstance_call_result_14349 = invoke(stypy.reporting.localization.Localization(__file__, 146, 17), isinstance_14345, *[v_14346, Header_14347], **kwargs_14348)
                
                # Testing if the type of an if condition is none (line 146)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 17), isinstance_call_result_14349):
                    
                    # Call to _is8bitstring(...): (line 149)
                    # Processing the call arguments (line 149)
                    # Getting the type of 'v' (line 149)
                    v_14356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 31), 'v', False)
                    # Processing the call keyword arguments (line 149)
                    kwargs_14357 = {}
                    # Getting the type of '_is8bitstring' (line 149)
                    _is8bitstring_14355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), '_is8bitstring', False)
                    # Calling _is8bitstring(args, kwargs) (line 149)
                    _is8bitstring_call_result_14358 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_14355, *[v_14356], **kwargs_14357)
                    
                    # Testing if the type of an if condition is none (line 149)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_call_result_14358):
                        
                        # Call to encode(...): (line 163)
                        # Processing the call keyword arguments (line 163)
                        kwargs_14371 = {}
                        
                        # Call to Header(...): (line 163)
                        # Processing the call arguments (line 163)
                        # Getting the type of 'v' (line 164)
                        v_14362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'v', False)
                        # Processing the call keyword arguments (line 163)
                        # Getting the type of 'self' (line 164)
                        self_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'self', False)
                        # Obtaining the member '_maxheaderlen' of a type (line 164)
                        _maxheaderlen_14364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), self_14363, '_maxheaderlen')
                        keyword_14365 = _maxheaderlen_14364
                        # Getting the type of 'h' (line 164)
                        h_14366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'h', False)
                        keyword_14367 = h_14366
                        kwargs_14368 = {'maxlinelen': keyword_14365, 'header_name': keyword_14367}
                        # Getting the type of 'Header' (line 163)
                        Header_14361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'Header', False)
                        # Calling Header(args, kwargs) (line 163)
                        Header_call_result_14369 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), Header_14361, *[v_14362], **kwargs_14368)
                        
                        # Obtaining the member 'encode' of a type (line 163)
                        encode_14370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 35), Header_call_result_14369, 'encode')
                        # Calling encode(args, kwargs) (line 163)
                        encode_call_result_14372 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), encode_14370, *[], **kwargs_14371)
                        
                    else:
                        
                        # Testing the type of an if condition (line 149)
                        if_condition_14359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_call_result_14358)
                        # Assigning a type to the variable 'if_condition_14359' (line 149)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'if_condition_14359', if_condition_14359)
                        # SSA begins for if statement (line 149)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'v' (line 156)
                        v_14360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 35), 'v')
                        # SSA branch for the else part of an if statement (line 149)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to encode(...): (line 163)
                        # Processing the call keyword arguments (line 163)
                        kwargs_14371 = {}
                        
                        # Call to Header(...): (line 163)
                        # Processing the call arguments (line 163)
                        # Getting the type of 'v' (line 164)
                        v_14362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'v', False)
                        # Processing the call keyword arguments (line 163)
                        # Getting the type of 'self' (line 164)
                        self_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'self', False)
                        # Obtaining the member '_maxheaderlen' of a type (line 164)
                        _maxheaderlen_14364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), self_14363, '_maxheaderlen')
                        keyword_14365 = _maxheaderlen_14364
                        # Getting the type of 'h' (line 164)
                        h_14366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'h', False)
                        keyword_14367 = h_14366
                        kwargs_14368 = {'maxlinelen': keyword_14365, 'header_name': keyword_14367}
                        # Getting the type of 'Header' (line 163)
                        Header_14361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'Header', False)
                        # Calling Header(args, kwargs) (line 163)
                        Header_call_result_14369 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), Header_14361, *[v_14362], **kwargs_14368)
                        
                        # Obtaining the member 'encode' of a type (line 163)
                        encode_14370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 35), Header_call_result_14369, 'encode')
                        # Calling encode(args, kwargs) (line 163)
                        encode_call_result_14372 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), encode_14370, *[], **kwargs_14371)
                        
                        # SSA join for if statement (line 149)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 146)
                    if_condition_14350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 17), isinstance_call_result_14349)
                    # Assigning a type to the variable 'if_condition_14350' (line 146)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'if_condition_14350', if_condition_14350)
                    # SSA begins for if statement (line 146)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to encode(...): (line 148)
                    # Processing the call keyword arguments (line 148)
                    kwargs_14353 = {}
                    # Getting the type of 'v' (line 148)
                    v_14351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'v', False)
                    # Obtaining the member 'encode' of a type (line 148)
                    encode_14352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 35), v_14351, 'encode')
                    # Calling encode(args, kwargs) (line 148)
                    encode_call_result_14354 = invoke(stypy.reporting.localization.Localization(__file__, 148, 35), encode_14352, *[], **kwargs_14353)
                    
                    # SSA branch for the else part of an if statement (line 146)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to _is8bitstring(...): (line 149)
                    # Processing the call arguments (line 149)
                    # Getting the type of 'v' (line 149)
                    v_14356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 31), 'v', False)
                    # Processing the call keyword arguments (line 149)
                    kwargs_14357 = {}
                    # Getting the type of '_is8bitstring' (line 149)
                    _is8bitstring_14355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), '_is8bitstring', False)
                    # Calling _is8bitstring(args, kwargs) (line 149)
                    _is8bitstring_call_result_14358 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_14355, *[v_14356], **kwargs_14357)
                    
                    # Testing if the type of an if condition is none (line 149)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_call_result_14358):
                        
                        # Call to encode(...): (line 163)
                        # Processing the call keyword arguments (line 163)
                        kwargs_14371 = {}
                        
                        # Call to Header(...): (line 163)
                        # Processing the call arguments (line 163)
                        # Getting the type of 'v' (line 164)
                        v_14362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'v', False)
                        # Processing the call keyword arguments (line 163)
                        # Getting the type of 'self' (line 164)
                        self_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'self', False)
                        # Obtaining the member '_maxheaderlen' of a type (line 164)
                        _maxheaderlen_14364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), self_14363, '_maxheaderlen')
                        keyword_14365 = _maxheaderlen_14364
                        # Getting the type of 'h' (line 164)
                        h_14366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'h', False)
                        keyword_14367 = h_14366
                        kwargs_14368 = {'maxlinelen': keyword_14365, 'header_name': keyword_14367}
                        # Getting the type of 'Header' (line 163)
                        Header_14361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'Header', False)
                        # Calling Header(args, kwargs) (line 163)
                        Header_call_result_14369 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), Header_14361, *[v_14362], **kwargs_14368)
                        
                        # Obtaining the member 'encode' of a type (line 163)
                        encode_14370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 35), Header_call_result_14369, 'encode')
                        # Calling encode(args, kwargs) (line 163)
                        encode_call_result_14372 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), encode_14370, *[], **kwargs_14371)
                        
                    else:
                        
                        # Testing the type of an if condition (line 149)
                        if_condition_14359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 17), _is8bitstring_call_result_14358)
                        # Assigning a type to the variable 'if_condition_14359' (line 149)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'if_condition_14359', if_condition_14359)
                        # SSA begins for if statement (line 149)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'v' (line 156)
                        v_14360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 35), 'v')
                        # SSA branch for the else part of an if statement (line 149)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to encode(...): (line 163)
                        # Processing the call keyword arguments (line 163)
                        kwargs_14371 = {}
                        
                        # Call to Header(...): (line 163)
                        # Processing the call arguments (line 163)
                        # Getting the type of 'v' (line 164)
                        v_14362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'v', False)
                        # Processing the call keyword arguments (line 163)
                        # Getting the type of 'self' (line 164)
                        self_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'self', False)
                        # Obtaining the member '_maxheaderlen' of a type (line 164)
                        _maxheaderlen_14364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), self_14363, '_maxheaderlen')
                        keyword_14365 = _maxheaderlen_14364
                        # Getting the type of 'h' (line 164)
                        h_14366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'h', False)
                        keyword_14367 = h_14366
                        kwargs_14368 = {'maxlinelen': keyword_14365, 'header_name': keyword_14367}
                        # Getting the type of 'Header' (line 163)
                        Header_14361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'Header', False)
                        # Calling Header(args, kwargs) (line 163)
                        Header_call_result_14369 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), Header_14361, *[v_14362], **kwargs_14368)
                        
                        # Obtaining the member 'encode' of a type (line 163)
                        encode_14370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 35), Header_call_result_14369, 'encode')
                        # Calling encode(args, kwargs) (line 163)
                        encode_call_result_14372 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), encode_14370, *[], **kwargs_14371)
                        
                        # SSA join for if statement (line 149)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 146)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '_write_headers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_headers' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_14373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14373)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_headers'
        return stypy_return_type_14373


    @norecursion
    def _handle_text(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handle_text'
        module_type_store = module_type_store.open_function_context('_handle_text', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator._handle_text.__dict__.__setitem__('stypy_localization', localization)
        Generator._handle_text.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator._handle_text.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator._handle_text.__dict__.__setitem__('stypy_function_name', 'Generator._handle_text')
        Generator._handle_text.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Generator._handle_text.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator._handle_text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator._handle_text.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator._handle_text.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator._handle_text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator._handle_text.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator._handle_text', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handle_text', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handle_text(...)' code ##################

        
        # Assigning a Call to a Name (line 173):
        
        # Call to get_payload(...): (line 173)
        # Processing the call keyword arguments (line 173)
        kwargs_14376 = {}
        # Getting the type of 'msg' (line 173)
        msg_14374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 18), 'msg', False)
        # Obtaining the member 'get_payload' of a type (line 173)
        get_payload_14375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 18), msg_14374, 'get_payload')
        # Calling get_payload(args, kwargs) (line 173)
        get_payload_call_result_14377 = invoke(stypy.reporting.localization.Localization(__file__, 173, 18), get_payload_14375, *[], **kwargs_14376)
        
        # Assigning a type to the variable 'payload' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'payload', get_payload_call_result_14377)
        
        # Type idiom detected: calculating its left and rigth part (line 174)
        # Getting the type of 'payload' (line 174)
        payload_14378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'payload')
        # Getting the type of 'None' (line 174)
        None_14379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'None')
        
        (may_be_14380, more_types_in_union_14381) = may_be_none(payload_14378, None_14379)

        if may_be_14380:

            if more_types_in_union_14381:
                # Runtime conditional SSA (line 174)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_14381:
                # SSA join for if statement (line 174)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'payload' (line 174)
        payload_14382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'payload')
        # Assigning a type to the variable 'payload' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'payload', remove_type_from_union(payload_14382, types.NoneType))
        
        # Type idiom detected: calculating its left and rigth part (line 176)
        # Getting the type of 'basestring' (line 176)
        basestring_14383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'basestring')
        # Getting the type of 'payload' (line 176)
        payload_14384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'payload')
        
        (may_be_14385, more_types_in_union_14386) = may_not_be_subtype(basestring_14383, payload_14384)

        if may_be_14385:

            if more_types_in_union_14386:
                # Runtime conditional SSA (line 176)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'payload' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'payload', remove_subtype_from_union(payload_14384, basestring))
            
            # Call to TypeError(...): (line 177)
            # Processing the call arguments (line 177)
            str_14388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'str', 'string payload expected: %s')
            
            # Call to type(...): (line 177)
            # Processing the call arguments (line 177)
            # Getting the type of 'payload' (line 177)
            payload_14390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 65), 'payload', False)
            # Processing the call keyword arguments (line 177)
            kwargs_14391 = {}
            # Getting the type of 'type' (line 177)
            type_14389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 60), 'type', False)
            # Calling type(args, kwargs) (line 177)
            type_call_result_14392 = invoke(stypy.reporting.localization.Localization(__file__, 177, 60), type_14389, *[payload_14390], **kwargs_14391)
            
            # Applying the binary operator '%' (line 177)
            result_mod_14393 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 28), '%', str_14388, type_call_result_14392)
            
            # Processing the call keyword arguments (line 177)
            kwargs_14394 = {}
            # Getting the type of 'TypeError' (line 177)
            TypeError_14387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 177)
            TypeError_call_result_14395 = invoke(stypy.reporting.localization.Localization(__file__, 177, 18), TypeError_14387, *[result_mod_14393], **kwargs_14394)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 177, 12), TypeError_call_result_14395, 'raise parameter', BaseException)

            if more_types_in_union_14386:
                # SSA join for if statement (line 176)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 178)
        self_14396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'self')
        # Obtaining the member '_mangle_from_' of a type (line 178)
        _mangle_from__14397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 11), self_14396, '_mangle_from_')
        # Testing if the type of an if condition is none (line 178)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 178, 8), _mangle_from__14397):
            pass
        else:
            
            # Testing the type of an if condition (line 178)
            if_condition_14398 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 8), _mangle_from__14397)
            # Assigning a type to the variable 'if_condition_14398' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'if_condition_14398', if_condition_14398)
            # SSA begins for if statement (line 178)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 179):
            
            # Call to sub(...): (line 179)
            # Processing the call arguments (line 179)
            str_14401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'str', '>From ')
            # Getting the type of 'payload' (line 179)
            payload_14402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 41), 'payload', False)
            # Processing the call keyword arguments (line 179)
            kwargs_14403 = {}
            # Getting the type of 'fcre' (line 179)
            fcre_14399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 22), 'fcre', False)
            # Obtaining the member 'sub' of a type (line 179)
            sub_14400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 22), fcre_14399, 'sub')
            # Calling sub(args, kwargs) (line 179)
            sub_call_result_14404 = invoke(stypy.reporting.localization.Localization(__file__, 179, 22), sub_14400, *[str_14401, payload_14402], **kwargs_14403)
            
            # Assigning a type to the variable 'payload' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'payload', sub_call_result_14404)
            # SSA join for if statement (line 178)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'payload' (line 180)
        payload_14408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'payload', False)
        # Processing the call keyword arguments (line 180)
        kwargs_14409 = {}
        # Getting the type of 'self' (line 180)
        self_14405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self', False)
        # Obtaining the member '_fp' of a type (line 180)
        _fp_14406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_14405, '_fp')
        # Obtaining the member 'write' of a type (line 180)
        write_14407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), _fp_14406, 'write')
        # Calling write(args, kwargs) (line 180)
        write_call_result_14410 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), write_14407, *[payload_14408], **kwargs_14409)
        
        
        # ################# End of '_handle_text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handle_text' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_14411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14411)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handle_text'
        return stypy_return_type_14411


    @norecursion
    def _handle_multipart(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handle_multipart'
        module_type_store = module_type_store.open_function_context('_handle_multipart', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator._handle_multipart.__dict__.__setitem__('stypy_localization', localization)
        Generator._handle_multipart.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator._handle_multipart.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator._handle_multipart.__dict__.__setitem__('stypy_function_name', 'Generator._handle_multipart')
        Generator._handle_multipart.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Generator._handle_multipart.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator._handle_multipart.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator._handle_multipart.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator._handle_multipart.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator._handle_multipart.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator._handle_multipart.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator._handle_multipart', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handle_multipart', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handle_multipart(...)' code ##################

        
        # Assigning a List to a Name (line 189):
        
        # Obtaining an instance of the builtin type 'list' (line 189)
        list_14412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 189)
        
        # Assigning a type to the variable 'msgtexts' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'msgtexts', list_14412)
        
        # Assigning a Call to a Name (line 190):
        
        # Call to get_payload(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_14415 = {}
        # Getting the type of 'msg' (line 190)
        msg_14413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'msg', False)
        # Obtaining the member 'get_payload' of a type (line 190)
        get_payload_14414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 19), msg_14413, 'get_payload')
        # Calling get_payload(args, kwargs) (line 190)
        get_payload_call_result_14416 = invoke(stypy.reporting.localization.Localization(__file__, 190, 19), get_payload_14414, *[], **kwargs_14415)
        
        # Assigning a type to the variable 'subparts' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'subparts', get_payload_call_result_14416)
        
        # Type idiom detected: calculating its left and rigth part (line 191)
        # Getting the type of 'subparts' (line 191)
        subparts_14417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'subparts')
        # Getting the type of 'None' (line 191)
        None_14418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'None')
        
        (may_be_14419, more_types_in_union_14420) = may_be_none(subparts_14417, None_14418)

        if may_be_14419:

            if more_types_in_union_14420:
                # Runtime conditional SSA (line 191)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 192):
            
            # Obtaining an instance of the builtin type 'list' (line 192)
            list_14421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 192)
            
            # Assigning a type to the variable 'subparts' (line 192)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'subparts', list_14421)

            if more_types_in_union_14420:
                # Runtime conditional SSA for else branch (line 191)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_14419) or more_types_in_union_14420):
            
            # Type idiom detected: calculating its left and rigth part (line 193)
            # Getting the type of 'basestring' (line 193)
            basestring_14422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 34), 'basestring')
            # Getting the type of 'subparts' (line 193)
            subparts_14423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'subparts')
            
            (may_be_14424, more_types_in_union_14425) = may_be_subtype(basestring_14422, subparts_14423)

            if may_be_14424:

                if more_types_in_union_14425:
                    # Runtime conditional SSA (line 193)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'subparts' (line 193)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'subparts', remove_not_subtype_from_union(subparts_14423, basestring))
                
                # Call to write(...): (line 195)
                # Processing the call arguments (line 195)
                # Getting the type of 'subparts' (line 195)
                subparts_14429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 27), 'subparts', False)
                # Processing the call keyword arguments (line 195)
                kwargs_14430 = {}
                # Getting the type of 'self' (line 195)
                self_14426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'self', False)
                # Obtaining the member '_fp' of a type (line 195)
                _fp_14427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), self_14426, '_fp')
                # Obtaining the member 'write' of a type (line 195)
                write_14428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), _fp_14427, 'write')
                # Calling write(args, kwargs) (line 195)
                write_call_result_14431 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), write_14428, *[subparts_14429], **kwargs_14430)
                
                # Assigning a type to the variable 'stypy_return_type' (line 196)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'stypy_return_type', types.NoneType)

                if more_types_in_union_14425:
                    # Runtime conditional SSA for else branch (line 193)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_14424) or more_types_in_union_14425):
                # Assigning a type to the variable 'subparts' (line 193)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'subparts', remove_subtype_from_union(subparts_14423, basestring))
                
                # Type idiom detected: calculating its left and rigth part (line 197)
                # Getting the type of 'list' (line 197)
                list_14432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 38), 'list')
                # Getting the type of 'subparts' (line 197)
                subparts_14433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 28), 'subparts')
                
                (may_be_14434, more_types_in_union_14435) = may_not_be_subtype(list_14432, subparts_14433)

                if may_be_14434:

                    if more_types_in_union_14435:
                        # Runtime conditional SSA (line 197)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'subparts' (line 197)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'subparts', remove_subtype_from_union(subparts_14433, list))
                    
                    # Assigning a List to a Name (line 199):
                    
                    # Obtaining an instance of the builtin type 'list' (line 199)
                    list_14436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 199)
                    # Adding element type (line 199)
                    # Getting the type of 'subparts' (line 199)
                    subparts_14437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 24), 'subparts')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 23), list_14436, subparts_14437)
                    
                    # Assigning a type to the variable 'subparts' (line 199)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'subparts', list_14436)

                    if more_types_in_union_14435:
                        # SSA join for if statement (line 197)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_14424 and more_types_in_union_14425):
                    # SSA join for if statement (line 193)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_14419 and more_types_in_union_14420):
                # SSA join for if statement (line 191)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'subparts' (line 200)
        subparts_14438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'subparts')
        # Assigning a type to the variable 'subparts_14438' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'subparts_14438', subparts_14438)
        # Testing if the for loop is going to be iterated (line 200)
        # Testing the type of a for loop iterable (line 200)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 200, 8), subparts_14438)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 200, 8), subparts_14438):
            # Getting the type of the for loop variable (line 200)
            for_loop_var_14439 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 200, 8), subparts_14438)
            # Assigning a type to the variable 'part' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'part', for_loop_var_14439)
            # SSA begins for a for statement (line 200)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 201):
            
            # Call to StringIO(...): (line 201)
            # Processing the call keyword arguments (line 201)
            kwargs_14441 = {}
            # Getting the type of 'StringIO' (line 201)
            StringIO_14440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'StringIO', False)
            # Calling StringIO(args, kwargs) (line 201)
            StringIO_call_result_14442 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), StringIO_14440, *[], **kwargs_14441)
            
            # Assigning a type to the variable 's' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 's', StringIO_call_result_14442)
            
            # Assigning a Call to a Name (line 202):
            
            # Call to clone(...): (line 202)
            # Processing the call arguments (line 202)
            # Getting the type of 's' (line 202)
            s_14445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 's', False)
            # Processing the call keyword arguments (line 202)
            kwargs_14446 = {}
            # Getting the type of 'self' (line 202)
            self_14443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'self', False)
            # Obtaining the member 'clone' of a type (line 202)
            clone_14444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 16), self_14443, 'clone')
            # Calling clone(args, kwargs) (line 202)
            clone_call_result_14447 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), clone_14444, *[s_14445], **kwargs_14446)
            
            # Assigning a type to the variable 'g' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'g', clone_call_result_14447)
            
            # Call to flatten(...): (line 203)
            # Processing the call arguments (line 203)
            # Getting the type of 'part' (line 203)
            part_14450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'part', False)
            # Processing the call keyword arguments (line 203)
            # Getting the type of 'False' (line 203)
            False_14451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 37), 'False', False)
            keyword_14452 = False_14451
            kwargs_14453 = {'unixfrom': keyword_14452}
            # Getting the type of 'g' (line 203)
            g_14448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'g', False)
            # Obtaining the member 'flatten' of a type (line 203)
            flatten_14449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), g_14448, 'flatten')
            # Calling flatten(args, kwargs) (line 203)
            flatten_call_result_14454 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), flatten_14449, *[part_14450], **kwargs_14453)
            
            
            # Call to append(...): (line 204)
            # Processing the call arguments (line 204)
            
            # Call to getvalue(...): (line 204)
            # Processing the call keyword arguments (line 204)
            kwargs_14459 = {}
            # Getting the type of 's' (line 204)
            s_14457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 28), 's', False)
            # Obtaining the member 'getvalue' of a type (line 204)
            getvalue_14458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 28), s_14457, 'getvalue')
            # Calling getvalue(args, kwargs) (line 204)
            getvalue_call_result_14460 = invoke(stypy.reporting.localization.Localization(__file__, 204, 28), getvalue_14458, *[], **kwargs_14459)
            
            # Processing the call keyword arguments (line 204)
            kwargs_14461 = {}
            # Getting the type of 'msgtexts' (line 204)
            msgtexts_14455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'msgtexts', False)
            # Obtaining the member 'append' of a type (line 204)
            append_14456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), msgtexts_14455, 'append')
            # Calling append(args, kwargs) (line 204)
            append_call_result_14462 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), append_14456, *[getvalue_call_result_14460], **kwargs_14461)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 206):
        
        # Call to get_boundary(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_14465 = {}
        # Getting the type of 'msg' (line 206)
        msg_14463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'msg', False)
        # Obtaining the member 'get_boundary' of a type (line 206)
        get_boundary_14464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 19), msg_14463, 'get_boundary')
        # Calling get_boundary(args, kwargs) (line 206)
        get_boundary_call_result_14466 = invoke(stypy.reporting.localization.Localization(__file__, 206, 19), get_boundary_14464, *[], **kwargs_14465)
        
        # Assigning a type to the variable 'boundary' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'boundary', get_boundary_call_result_14466)
        
        # Getting the type of 'boundary' (line 207)
        boundary_14467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'boundary')
        # Applying the 'not' unary operator (line 207)
        result_not__14468 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), 'not', boundary_14467)
        
        # Testing if the type of an if condition is none (line 207)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 207, 8), result_not__14468):
            pass
        else:
            
            # Testing the type of an if condition (line 207)
            if_condition_14469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), result_not__14468)
            # Assigning a type to the variable 'if_condition_14469' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_14469', if_condition_14469)
            # SSA begins for if statement (line 207)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 210):
            
            # Call to join(...): (line 210)
            # Processing the call arguments (line 210)
            # Getting the type of 'msgtexts' (line 210)
            msgtexts_14472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'msgtexts', False)
            # Processing the call keyword arguments (line 210)
            kwargs_14473 = {}
            # Getting the type of 'NL' (line 210)
            NL_14470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'NL', False)
            # Obtaining the member 'join' of a type (line 210)
            join_14471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 22), NL_14470, 'join')
            # Calling join(args, kwargs) (line 210)
            join_call_result_14474 = invoke(stypy.reporting.localization.Localization(__file__, 210, 22), join_14471, *[msgtexts_14472], **kwargs_14473)
            
            # Assigning a type to the variable 'alltext' (line 210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'alltext', join_call_result_14474)
            
            # Assigning a Call to a Name (line 211):
            
            # Call to _make_boundary(...): (line 211)
            # Processing the call arguments (line 211)
            # Getting the type of 'alltext' (line 211)
            alltext_14476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'alltext', False)
            # Processing the call keyword arguments (line 211)
            kwargs_14477 = {}
            # Getting the type of '_make_boundary' (line 211)
            _make_boundary_14475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), '_make_boundary', False)
            # Calling _make_boundary(args, kwargs) (line 211)
            _make_boundary_call_result_14478 = invoke(stypy.reporting.localization.Localization(__file__, 211, 23), _make_boundary_14475, *[alltext_14476], **kwargs_14477)
            
            # Assigning a type to the variable 'boundary' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'boundary', _make_boundary_call_result_14478)
            
            # Call to set_boundary(...): (line 212)
            # Processing the call arguments (line 212)
            # Getting the type of 'boundary' (line 212)
            boundary_14481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'boundary', False)
            # Processing the call keyword arguments (line 212)
            kwargs_14482 = {}
            # Getting the type of 'msg' (line 212)
            msg_14479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'msg', False)
            # Obtaining the member 'set_boundary' of a type (line 212)
            set_boundary_14480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), msg_14479, 'set_boundary')
            # Calling set_boundary(args, kwargs) (line 212)
            set_boundary_call_result_14483 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), set_boundary_14480, *[boundary_14481], **kwargs_14482)
            
            # SSA join for if statement (line 207)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'msg' (line 214)
        msg_14484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'msg')
        # Obtaining the member 'preamble' of a type (line 214)
        preamble_14485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 11), msg_14484, 'preamble')
        # Getting the type of 'None' (line 214)
        None_14486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 31), 'None')
        # Applying the binary operator 'isnot' (line 214)
        result_is_not_14487 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), 'isnot', preamble_14485, None_14486)
        
        # Testing if the type of an if condition is none (line 214)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 214, 8), result_is_not_14487):
            pass
        else:
            
            # Testing the type of an if condition (line 214)
            if_condition_14488 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 8), result_is_not_14487)
            # Assigning a type to the variable 'if_condition_14488' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'if_condition_14488', if_condition_14488)
            # SSA begins for if statement (line 214)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 215)
            self_14489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'self')
            # Obtaining the member '_mangle_from_' of a type (line 215)
            _mangle_from__14490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 15), self_14489, '_mangle_from_')
            # Testing if the type of an if condition is none (line 215)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 215, 12), _mangle_from__14490):
                
                # Assigning a Attribute to a Name (line 218):
                # Getting the type of 'msg' (line 218)
                msg_14499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'msg')
                # Obtaining the member 'preamble' of a type (line 218)
                preamble_14500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 27), msg_14499, 'preamble')
                # Assigning a type to the variable 'preamble' (line 218)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'preamble', preamble_14500)
            else:
                
                # Testing the type of an if condition (line 215)
                if_condition_14491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 12), _mangle_from__14490)
                # Assigning a type to the variable 'if_condition_14491' (line 215)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'if_condition_14491', if_condition_14491)
                # SSA begins for if statement (line 215)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 216):
                
                # Call to sub(...): (line 216)
                # Processing the call arguments (line 216)
                str_14494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 36), 'str', '>From ')
                # Getting the type of 'msg' (line 216)
                msg_14495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 46), 'msg', False)
                # Obtaining the member 'preamble' of a type (line 216)
                preamble_14496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 46), msg_14495, 'preamble')
                # Processing the call keyword arguments (line 216)
                kwargs_14497 = {}
                # Getting the type of 'fcre' (line 216)
                fcre_14492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 27), 'fcre', False)
                # Obtaining the member 'sub' of a type (line 216)
                sub_14493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 27), fcre_14492, 'sub')
                # Calling sub(args, kwargs) (line 216)
                sub_call_result_14498 = invoke(stypy.reporting.localization.Localization(__file__, 216, 27), sub_14493, *[str_14494, preamble_14496], **kwargs_14497)
                
                # Assigning a type to the variable 'preamble' (line 216)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'preamble', sub_call_result_14498)
                # SSA branch for the else part of an if statement (line 215)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Attribute to a Name (line 218):
                # Getting the type of 'msg' (line 218)
                msg_14499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'msg')
                # Obtaining the member 'preamble' of a type (line 218)
                preamble_14500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 27), msg_14499, 'preamble')
                # Assigning a type to the variable 'preamble' (line 218)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'preamble', preamble_14500)
                # SSA join for if statement (line 215)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'preamble' (line 219)
            preamble_14501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'preamble')
            # SSA join for if statement (line 214)
            module_type_store = module_type_store.join_ssa_context()
            

        str_14502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 27), 'str', '--')
        # Getting the type of 'boundary' (line 221)
        boundary_14503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 34), 'boundary')
        # Applying the binary operator '+' (line 221)
        result_add_14504 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 27), '+', str_14502, boundary_14503)
        
        # Getting the type of 'msgtexts' (line 223)
        msgtexts_14505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'msgtexts')
        # Testing if the type of an if condition is none (line 223)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 223, 8), msgtexts_14505):
            pass
        else:
            
            # Testing the type of an if condition (line 223)
            if_condition_14506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), msgtexts_14505)
            # Assigning a type to the variable 'if_condition_14506' (line 223)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_14506', if_condition_14506)
            # SSA begins for if statement (line 223)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 224)
            # Processing the call arguments (line 224)
            
            # Call to pop(...): (line 224)
            # Processing the call arguments (line 224)
            int_14512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 40), 'int')
            # Processing the call keyword arguments (line 224)
            kwargs_14513 = {}
            # Getting the type of 'msgtexts' (line 224)
            msgtexts_14510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'msgtexts', False)
            # Obtaining the member 'pop' of a type (line 224)
            pop_14511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 27), msgtexts_14510, 'pop')
            # Calling pop(args, kwargs) (line 224)
            pop_call_result_14514 = invoke(stypy.reporting.localization.Localization(__file__, 224, 27), pop_14511, *[int_14512], **kwargs_14513)
            
            # Processing the call keyword arguments (line 224)
            kwargs_14515 = {}
            # Getting the type of 'self' (line 224)
            self_14507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'self', False)
            # Obtaining the member '_fp' of a type (line 224)
            _fp_14508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), self_14507, '_fp')
            # Obtaining the member 'write' of a type (line 224)
            write_14509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), _fp_14508, 'write')
            # Calling write(args, kwargs) (line 224)
            write_call_result_14516 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), write_14509, *[pop_call_result_14514], **kwargs_14515)
            
            # SSA join for if statement (line 223)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'msgtexts' (line 228)
        msgtexts_14517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 25), 'msgtexts')
        # Assigning a type to the variable 'msgtexts_14517' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'msgtexts_14517', msgtexts_14517)
        # Testing if the for loop is going to be iterated (line 228)
        # Testing the type of a for loop iterable (line 228)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 228, 8), msgtexts_14517)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 228, 8), msgtexts_14517):
            # Getting the type of the for loop variable (line 228)
            for_loop_var_14518 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 228, 8), msgtexts_14517)
            # Assigning a type to the variable 'body_part' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'body_part', for_loop_var_14518)
            # SSA begins for a for statement (line 228)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            str_14519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 31), 'str', '\n--')
            # Getting the type of 'boundary' (line 230)
            boundary_14520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'boundary')
            # Applying the binary operator '+' (line 230)
            result_add_14521 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 31), '+', str_14519, boundary_14520)
            
            
            # Call to write(...): (line 232)
            # Processing the call arguments (line 232)
            # Getting the type of 'body_part' (line 232)
            body_part_14525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'body_part', False)
            # Processing the call keyword arguments (line 232)
            kwargs_14526 = {}
            # Getting the type of 'self' (line 232)
            self_14522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self', False)
            # Obtaining the member '_fp' of a type (line 232)
            _fp_14523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), self_14522, '_fp')
            # Obtaining the member 'write' of a type (line 232)
            write_14524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), _fp_14523, 'write')
            # Calling write(args, kwargs) (line 232)
            write_call_result_14527 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), write_14524, *[body_part_14525], **kwargs_14526)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 234)
        # Processing the call arguments (line 234)
        str_14531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 23), 'str', '\n--')
        # Getting the type of 'boundary' (line 234)
        boundary_14532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 32), 'boundary', False)
        # Applying the binary operator '+' (line 234)
        result_add_14533 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 23), '+', str_14531, boundary_14532)
        
        str_14534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 43), 'str', '--')
        # Applying the binary operator '+' (line 234)
        result_add_14535 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 41), '+', result_add_14533, str_14534)
        
        # Getting the type of 'NL' (line 234)
        NL_14536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 50), 'NL', False)
        # Applying the binary operator '+' (line 234)
        result_add_14537 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 48), '+', result_add_14535, NL_14536)
        
        # Processing the call keyword arguments (line 234)
        kwargs_14538 = {}
        # Getting the type of 'self' (line 234)
        self_14528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'self', False)
        # Obtaining the member '_fp' of a type (line 234)
        _fp_14529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), self_14528, '_fp')
        # Obtaining the member 'write' of a type (line 234)
        write_14530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), _fp_14529, 'write')
        # Calling write(args, kwargs) (line 234)
        write_call_result_14539 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), write_14530, *[result_add_14537], **kwargs_14538)
        
        
        # Getting the type of 'msg' (line 235)
        msg_14540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'msg')
        # Obtaining the member 'epilogue' of a type (line 235)
        epilogue_14541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 11), msg_14540, 'epilogue')
        # Getting the type of 'None' (line 235)
        None_14542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 31), 'None')
        # Applying the binary operator 'isnot' (line 235)
        result_is_not_14543 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 11), 'isnot', epilogue_14541, None_14542)
        
        # Testing if the type of an if condition is none (line 235)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 235, 8), result_is_not_14543):
            pass
        else:
            
            # Testing the type of an if condition (line 235)
            if_condition_14544 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 8), result_is_not_14543)
            # Assigning a type to the variable 'if_condition_14544' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'if_condition_14544', if_condition_14544)
            # SSA begins for if statement (line 235)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 236)
            self_14545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'self')
            # Obtaining the member '_mangle_from_' of a type (line 236)
            _mangle_from__14546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), self_14545, '_mangle_from_')
            # Testing if the type of an if condition is none (line 236)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 236, 12), _mangle_from__14546):
                
                # Assigning a Attribute to a Name (line 239):
                # Getting the type of 'msg' (line 239)
                msg_14555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'msg')
                # Obtaining the member 'epilogue' of a type (line 239)
                epilogue_14556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 27), msg_14555, 'epilogue')
                # Assigning a type to the variable 'epilogue' (line 239)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'epilogue', epilogue_14556)
            else:
                
                # Testing the type of an if condition (line 236)
                if_condition_14547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 12), _mangle_from__14546)
                # Assigning a type to the variable 'if_condition_14547' (line 236)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'if_condition_14547', if_condition_14547)
                # SSA begins for if statement (line 236)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 237):
                
                # Call to sub(...): (line 237)
                # Processing the call arguments (line 237)
                str_14550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 36), 'str', '>From ')
                # Getting the type of 'msg' (line 237)
                msg_14551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 46), 'msg', False)
                # Obtaining the member 'epilogue' of a type (line 237)
                epilogue_14552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 46), msg_14551, 'epilogue')
                # Processing the call keyword arguments (line 237)
                kwargs_14553 = {}
                # Getting the type of 'fcre' (line 237)
                fcre_14548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'fcre', False)
                # Obtaining the member 'sub' of a type (line 237)
                sub_14549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 27), fcre_14548, 'sub')
                # Calling sub(args, kwargs) (line 237)
                sub_call_result_14554 = invoke(stypy.reporting.localization.Localization(__file__, 237, 27), sub_14549, *[str_14550, epilogue_14552], **kwargs_14553)
                
                # Assigning a type to the variable 'epilogue' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'epilogue', sub_call_result_14554)
                # SSA branch for the else part of an if statement (line 236)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Attribute to a Name (line 239):
                # Getting the type of 'msg' (line 239)
                msg_14555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'msg')
                # Obtaining the member 'epilogue' of a type (line 239)
                epilogue_14556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 27), msg_14555, 'epilogue')
                # Assigning a type to the variable 'epilogue' (line 239)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'epilogue', epilogue_14556)
                # SSA join for if statement (line 236)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 240)
            # Processing the call arguments (line 240)
            # Getting the type of 'epilogue' (line 240)
            epilogue_14560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), 'epilogue', False)
            # Processing the call keyword arguments (line 240)
            kwargs_14561 = {}
            # Getting the type of 'self' (line 240)
            self_14557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'self', False)
            # Obtaining the member '_fp' of a type (line 240)
            _fp_14558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), self_14557, '_fp')
            # Obtaining the member 'write' of a type (line 240)
            write_14559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), _fp_14558, 'write')
            # Calling write(args, kwargs) (line 240)
            write_call_result_14562 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), write_14559, *[epilogue_14560], **kwargs_14561)
            
            # SSA join for if statement (line 235)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '_handle_multipart(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handle_multipart' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_14563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14563)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handle_multipart'
        return stypy_return_type_14563


    @norecursion
    def _handle_multipart_signed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handle_multipart_signed'
        module_type_store = module_type_store.open_function_context('_handle_multipart_signed', 242, 4, False)
        # Assigning a type to the variable 'self' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_localization', localization)
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_function_name', 'Generator._handle_multipart_signed')
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator._handle_multipart_signed.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator._handle_multipart_signed', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handle_multipart_signed', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handle_multipart_signed(...)' code ##################

        
        # Assigning a Attribute to a Name (line 246):
        # Getting the type of 'self' (line 246)
        self_14564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 27), 'self')
        # Obtaining the member '_maxheaderlen' of a type (line 246)
        _maxheaderlen_14565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 27), self_14564, '_maxheaderlen')
        # Assigning a type to the variable 'old_maxheaderlen' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'old_maxheaderlen', _maxheaderlen_14565)
        
        # Try-finally block (line 247)
        
        # Assigning a Num to a Attribute (line 248):
        int_14566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 33), 'int')
        # Getting the type of 'self' (line 248)
        self_14567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'self')
        # Setting the type of the member '_maxheaderlen' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), self_14567, '_maxheaderlen', int_14566)
        
        # Call to _handle_multipart(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'msg' (line 249)
        msg_14570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 35), 'msg', False)
        # Processing the call keyword arguments (line 249)
        kwargs_14571 = {}
        # Getting the type of 'self' (line 249)
        self_14568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'self', False)
        # Obtaining the member '_handle_multipart' of a type (line 249)
        _handle_multipart_14569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), self_14568, '_handle_multipart')
        # Calling _handle_multipart(args, kwargs) (line 249)
        _handle_multipart_call_result_14572 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), _handle_multipart_14569, *[msg_14570], **kwargs_14571)
        
        
        # finally branch of the try-finally block (line 247)
        
        # Assigning a Name to a Attribute (line 251):
        # Getting the type of 'old_maxheaderlen' (line 251)
        old_maxheaderlen_14573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 33), 'old_maxheaderlen')
        # Getting the type of 'self' (line 251)
        self_14574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'self')
        # Setting the type of the member '_maxheaderlen' of a type (line 251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), self_14574, '_maxheaderlen', old_maxheaderlen_14573)
        
        
        # ################# End of '_handle_multipart_signed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handle_multipart_signed' in the type store
        # Getting the type of 'stypy_return_type' (line 242)
        stypy_return_type_14575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14575)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handle_multipart_signed'
        return stypy_return_type_14575


    @norecursion
    def _handle_message_delivery_status(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handle_message_delivery_status'
        module_type_store = module_type_store.open_function_context('_handle_message_delivery_status', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_localization', localization)
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_function_name', 'Generator._handle_message_delivery_status')
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator._handle_message_delivery_status.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator._handle_message_delivery_status', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handle_message_delivery_status', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handle_message_delivery_status(...)' code ##################

        
        # Assigning a List to a Name (line 257):
        
        # Obtaining an instance of the builtin type 'list' (line 257)
        list_14576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 257)
        
        # Assigning a type to the variable 'blocks' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'blocks', list_14576)
        
        
        # Call to get_payload(...): (line 258)
        # Processing the call keyword arguments (line 258)
        kwargs_14579 = {}
        # Getting the type of 'msg' (line 258)
        msg_14577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'msg', False)
        # Obtaining the member 'get_payload' of a type (line 258)
        get_payload_14578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), msg_14577, 'get_payload')
        # Calling get_payload(args, kwargs) (line 258)
        get_payload_call_result_14580 = invoke(stypy.reporting.localization.Localization(__file__, 258, 20), get_payload_14578, *[], **kwargs_14579)
        
        # Assigning a type to the variable 'get_payload_call_result_14580' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'get_payload_call_result_14580', get_payload_call_result_14580)
        # Testing if the for loop is going to be iterated (line 258)
        # Testing the type of a for loop iterable (line 258)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 258, 8), get_payload_call_result_14580)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 258, 8), get_payload_call_result_14580):
            # Getting the type of the for loop variable (line 258)
            for_loop_var_14581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 258, 8), get_payload_call_result_14580)
            # Assigning a type to the variable 'part' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'part', for_loop_var_14581)
            # SSA begins for a for statement (line 258)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 259):
            
            # Call to StringIO(...): (line 259)
            # Processing the call keyword arguments (line 259)
            kwargs_14583 = {}
            # Getting the type of 'StringIO' (line 259)
            StringIO_14582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'StringIO', False)
            # Calling StringIO(args, kwargs) (line 259)
            StringIO_call_result_14584 = invoke(stypy.reporting.localization.Localization(__file__, 259, 16), StringIO_14582, *[], **kwargs_14583)
            
            # Assigning a type to the variable 's' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 's', StringIO_call_result_14584)
            
            # Assigning a Call to a Name (line 260):
            
            # Call to clone(...): (line 260)
            # Processing the call arguments (line 260)
            # Getting the type of 's' (line 260)
            s_14587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 27), 's', False)
            # Processing the call keyword arguments (line 260)
            kwargs_14588 = {}
            # Getting the type of 'self' (line 260)
            self_14585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'self', False)
            # Obtaining the member 'clone' of a type (line 260)
            clone_14586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 16), self_14585, 'clone')
            # Calling clone(args, kwargs) (line 260)
            clone_call_result_14589 = invoke(stypy.reporting.localization.Localization(__file__, 260, 16), clone_14586, *[s_14587], **kwargs_14588)
            
            # Assigning a type to the variable 'g' (line 260)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'g', clone_call_result_14589)
            
            # Call to flatten(...): (line 261)
            # Processing the call arguments (line 261)
            # Getting the type of 'part' (line 261)
            part_14592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 22), 'part', False)
            # Processing the call keyword arguments (line 261)
            # Getting the type of 'False' (line 261)
            False_14593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 37), 'False', False)
            keyword_14594 = False_14593
            kwargs_14595 = {'unixfrom': keyword_14594}
            # Getting the type of 'g' (line 261)
            g_14590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'g', False)
            # Obtaining the member 'flatten' of a type (line 261)
            flatten_14591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 12), g_14590, 'flatten')
            # Calling flatten(args, kwargs) (line 261)
            flatten_call_result_14596 = invoke(stypy.reporting.localization.Localization(__file__, 261, 12), flatten_14591, *[part_14592], **kwargs_14595)
            
            
            # Assigning a Call to a Name (line 262):
            
            # Call to getvalue(...): (line 262)
            # Processing the call keyword arguments (line 262)
            kwargs_14599 = {}
            # Getting the type of 's' (line 262)
            s_14597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 19), 's', False)
            # Obtaining the member 'getvalue' of a type (line 262)
            getvalue_14598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 19), s_14597, 'getvalue')
            # Calling getvalue(args, kwargs) (line 262)
            getvalue_call_result_14600 = invoke(stypy.reporting.localization.Localization(__file__, 262, 19), getvalue_14598, *[], **kwargs_14599)
            
            # Assigning a type to the variable 'text' (line 262)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'text', getvalue_call_result_14600)
            
            # Assigning a Call to a Name (line 263):
            
            # Call to split(...): (line 263)
            # Processing the call arguments (line 263)
            str_14603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 31), 'str', '\n')
            # Processing the call keyword arguments (line 263)
            kwargs_14604 = {}
            # Getting the type of 'text' (line 263)
            text_14601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 'text', False)
            # Obtaining the member 'split' of a type (line 263)
            split_14602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 20), text_14601, 'split')
            # Calling split(args, kwargs) (line 263)
            split_call_result_14605 = invoke(stypy.reporting.localization.Localization(__file__, 263, 20), split_14602, *[str_14603], **kwargs_14604)
            
            # Assigning a type to the variable 'lines' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'lines', split_call_result_14605)
            
            # Evaluating a boolean operation
            # Getting the type of 'lines' (line 265)
            lines_14606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'lines')
            
            
            # Obtaining the type of the subscript
            int_14607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 31), 'int')
            # Getting the type of 'lines' (line 265)
            lines_14608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 'lines')
            # Obtaining the member '__getitem__' of a type (line 265)
            getitem___14609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 25), lines_14608, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 265)
            subscript_call_result_14610 = invoke(stypy.reporting.localization.Localization(__file__, 265, 25), getitem___14609, int_14607)
            
            str_14611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 38), 'str', '')
            # Applying the binary operator '==' (line 265)
            result_eq_14612 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 25), '==', subscript_call_result_14610, str_14611)
            
            # Applying the binary operator 'and' (line 265)
            result_and_keyword_14613 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 15), 'and', lines_14606, result_eq_14612)
            
            # Testing if the type of an if condition is none (line 265)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 265, 12), result_and_keyword_14613):
                
                # Call to append(...): (line 268)
                # Processing the call arguments (line 268)
                # Getting the type of 'text' (line 268)
                text_14630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'text', False)
                # Processing the call keyword arguments (line 268)
                kwargs_14631 = {}
                # Getting the type of 'blocks' (line 268)
                blocks_14628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'blocks', False)
                # Obtaining the member 'append' of a type (line 268)
                append_14629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 16), blocks_14628, 'append')
                # Calling append(args, kwargs) (line 268)
                append_call_result_14632 = invoke(stypy.reporting.localization.Localization(__file__, 268, 16), append_14629, *[text_14630], **kwargs_14631)
                
            else:
                
                # Testing the type of an if condition (line 265)
                if_condition_14614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 12), result_and_keyword_14613)
                # Assigning a type to the variable 'if_condition_14614' (line 265)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'if_condition_14614', if_condition_14614)
                # SSA begins for if statement (line 265)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 266)
                # Processing the call arguments (line 266)
                
                # Call to join(...): (line 266)
                # Processing the call arguments (line 266)
                
                # Obtaining the type of the subscript
                int_14619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 45), 'int')
                slice_14620 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 266, 38), None, int_14619, None)
                # Getting the type of 'lines' (line 266)
                lines_14621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 38), 'lines', False)
                # Obtaining the member '__getitem__' of a type (line 266)
                getitem___14622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 38), lines_14621, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 266)
                subscript_call_result_14623 = invoke(stypy.reporting.localization.Localization(__file__, 266, 38), getitem___14622, slice_14620)
                
                # Processing the call keyword arguments (line 266)
                kwargs_14624 = {}
                # Getting the type of 'NL' (line 266)
                NL_14617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 30), 'NL', False)
                # Obtaining the member 'join' of a type (line 266)
                join_14618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 30), NL_14617, 'join')
                # Calling join(args, kwargs) (line 266)
                join_call_result_14625 = invoke(stypy.reporting.localization.Localization(__file__, 266, 30), join_14618, *[subscript_call_result_14623], **kwargs_14624)
                
                # Processing the call keyword arguments (line 266)
                kwargs_14626 = {}
                # Getting the type of 'blocks' (line 266)
                blocks_14615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'blocks', False)
                # Obtaining the member 'append' of a type (line 266)
                append_14616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 16), blocks_14615, 'append')
                # Calling append(args, kwargs) (line 266)
                append_call_result_14627 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), append_14616, *[join_call_result_14625], **kwargs_14626)
                
                # SSA branch for the else part of an if statement (line 265)
                module_type_store.open_ssa_branch('else')
                
                # Call to append(...): (line 268)
                # Processing the call arguments (line 268)
                # Getting the type of 'text' (line 268)
                text_14630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'text', False)
                # Processing the call keyword arguments (line 268)
                kwargs_14631 = {}
                # Getting the type of 'blocks' (line 268)
                blocks_14628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'blocks', False)
                # Obtaining the member 'append' of a type (line 268)
                append_14629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 16), blocks_14628, 'append')
                # Calling append(args, kwargs) (line 268)
                append_call_result_14632 = invoke(stypy.reporting.localization.Localization(__file__, 268, 16), append_14629, *[text_14630], **kwargs_14631)
                
                # SSA join for if statement (line 265)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 272)
        # Processing the call arguments (line 272)
        
        # Call to join(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'blocks' (line 272)
        blocks_14638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 31), 'blocks', False)
        # Processing the call keyword arguments (line 272)
        kwargs_14639 = {}
        # Getting the type of 'NL' (line 272)
        NL_14636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 23), 'NL', False)
        # Obtaining the member 'join' of a type (line 272)
        join_14637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 23), NL_14636, 'join')
        # Calling join(args, kwargs) (line 272)
        join_call_result_14640 = invoke(stypy.reporting.localization.Localization(__file__, 272, 23), join_14637, *[blocks_14638], **kwargs_14639)
        
        # Processing the call keyword arguments (line 272)
        kwargs_14641 = {}
        # Getting the type of 'self' (line 272)
        self_14633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self', False)
        # Obtaining the member '_fp' of a type (line 272)
        _fp_14634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_14633, '_fp')
        # Obtaining the member 'write' of a type (line 272)
        write_14635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), _fp_14634, 'write')
        # Calling write(args, kwargs) (line 272)
        write_call_result_14642 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), write_14635, *[join_call_result_14640], **kwargs_14641)
        
        
        # ################# End of '_handle_message_delivery_status(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handle_message_delivery_status' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_14643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14643)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handle_message_delivery_status'
        return stypy_return_type_14643


    @norecursion
    def _handle_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handle_message'
        module_type_store = module_type_store.open_function_context('_handle_message', 274, 4, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Generator._handle_message.__dict__.__setitem__('stypy_localization', localization)
        Generator._handle_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Generator._handle_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        Generator._handle_message.__dict__.__setitem__('stypy_function_name', 'Generator._handle_message')
        Generator._handle_message.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Generator._handle_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        Generator._handle_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Generator._handle_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        Generator._handle_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        Generator._handle_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Generator._handle_message.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Generator._handle_message', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handle_message', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handle_message(...)' code ##################

        
        # Assigning a Call to a Name (line 275):
        
        # Call to StringIO(...): (line 275)
        # Processing the call keyword arguments (line 275)
        kwargs_14645 = {}
        # Getting the type of 'StringIO' (line 275)
        StringIO_14644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 275)
        StringIO_call_result_14646 = invoke(stypy.reporting.localization.Localization(__file__, 275, 12), StringIO_14644, *[], **kwargs_14645)
        
        # Assigning a type to the variable 's' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 's', StringIO_call_result_14646)
        
        # Assigning a Call to a Name (line 276):
        
        # Call to clone(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 's' (line 276)
        s_14649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 23), 's', False)
        # Processing the call keyword arguments (line 276)
        kwargs_14650 = {}
        # Getting the type of 'self' (line 276)
        self_14647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'self', False)
        # Obtaining the member 'clone' of a type (line 276)
        clone_14648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), self_14647, 'clone')
        # Calling clone(args, kwargs) (line 276)
        clone_call_result_14651 = invoke(stypy.reporting.localization.Localization(__file__, 276, 12), clone_14648, *[s_14649], **kwargs_14650)
        
        # Assigning a type to the variable 'g' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'g', clone_call_result_14651)
        
        # Assigning a Call to a Name (line 286):
        
        # Call to get_payload(...): (line 286)
        # Processing the call keyword arguments (line 286)
        kwargs_14654 = {}
        # Getting the type of 'msg' (line 286)
        msg_14652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'msg', False)
        # Obtaining the member 'get_payload' of a type (line 286)
        get_payload_14653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 18), msg_14652, 'get_payload')
        # Calling get_payload(args, kwargs) (line 286)
        get_payload_call_result_14655 = invoke(stypy.reporting.localization.Localization(__file__, 286, 18), get_payload_14653, *[], **kwargs_14654)
        
        # Assigning a type to the variable 'payload' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'payload', get_payload_call_result_14655)
        
        # Type idiom detected: calculating its left and rigth part (line 287)
        # Getting the type of 'list' (line 287)
        list_14656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'list')
        # Getting the type of 'payload' (line 287)
        payload_14657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 22), 'payload')
        
        (may_be_14658, more_types_in_union_14659) = may_be_subtype(list_14656, payload_14657)

        if may_be_14658:

            if more_types_in_union_14659:
                # Runtime conditional SSA (line 287)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'payload' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'payload', remove_not_subtype_from_union(payload_14657, list))
            
            # Call to flatten(...): (line 288)
            # Processing the call arguments (line 288)
            
            # Call to get_payload(...): (line 288)
            # Processing the call arguments (line 288)
            int_14664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 38), 'int')
            # Processing the call keyword arguments (line 288)
            kwargs_14665 = {}
            # Getting the type of 'msg' (line 288)
            msg_14662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 22), 'msg', False)
            # Obtaining the member 'get_payload' of a type (line 288)
            get_payload_14663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 22), msg_14662, 'get_payload')
            # Calling get_payload(args, kwargs) (line 288)
            get_payload_call_result_14666 = invoke(stypy.reporting.localization.Localization(__file__, 288, 22), get_payload_14663, *[int_14664], **kwargs_14665)
            
            # Processing the call keyword arguments (line 288)
            # Getting the type of 'False' (line 288)
            False_14667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 51), 'False', False)
            keyword_14668 = False_14667
            kwargs_14669 = {'unixfrom': keyword_14668}
            # Getting the type of 'g' (line 288)
            g_14660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'g', False)
            # Obtaining the member 'flatten' of a type (line 288)
            flatten_14661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), g_14660, 'flatten')
            # Calling flatten(args, kwargs) (line 288)
            flatten_call_result_14670 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), flatten_14661, *[get_payload_call_result_14666], **kwargs_14669)
            
            
            # Assigning a Call to a Name (line 289):
            
            # Call to getvalue(...): (line 289)
            # Processing the call keyword arguments (line 289)
            kwargs_14673 = {}
            # Getting the type of 's' (line 289)
            s_14671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 22), 's', False)
            # Obtaining the member 'getvalue' of a type (line 289)
            getvalue_14672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 22), s_14671, 'getvalue')
            # Calling getvalue(args, kwargs) (line 289)
            getvalue_call_result_14674 = invoke(stypy.reporting.localization.Localization(__file__, 289, 22), getvalue_14672, *[], **kwargs_14673)
            
            # Assigning a type to the variable 'payload' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'payload', getvalue_call_result_14674)

            if more_types_in_union_14659:
                # SSA join for if statement (line 287)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to write(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'payload' (line 290)
        payload_14678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 23), 'payload', False)
        # Processing the call keyword arguments (line 290)
        kwargs_14679 = {}
        # Getting the type of 'self' (line 290)
        self_14675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'self', False)
        # Obtaining the member '_fp' of a type (line 290)
        _fp_14676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), self_14675, '_fp')
        # Obtaining the member 'write' of a type (line 290)
        write_14677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), _fp_14676, 'write')
        # Calling write(args, kwargs) (line 290)
        write_call_result_14680 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), write_14677, *[payload_14678], **kwargs_14679)
        
        
        # ################# End of '_handle_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handle_message' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_14681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14681)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handle_message'
        return stypy_return_type_14681


# Assigning a type to the variable 'Generator' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'Generator', Generator)

# Assigning a Name to a Name (line 183):
# Getting the type of 'Generator'
Generator_14682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Generator')
# Obtaining the member '_handle_text' of a type
_handle_text_14683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Generator_14682, '_handle_text')
# Getting the type of 'Generator'
Generator_14684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Generator')
# Setting the type of the member '_writeBody' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Generator_14684, '_writeBody', _handle_text_14683)

# Assigning a Str to a Name (line 294):
str_14685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 7), 'str', '[Non-text (%(type)s) part of message omitted, filename %(filename)s]')
# Assigning a type to the variable '_FMT' (line 294)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 0), '_FMT', str_14685)
# Declaration of the 'DecodedGenerator' class
# Getting the type of 'Generator' (line 296)
Generator_14686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 23), 'Generator')

class DecodedGenerator(Generator_14686, ):
    str_14687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, (-1)), 'str', 'Generates a text representation of a message.\n\n    Like the Generator base class, except that non-text parts are substituted\n    with a format string representing the part.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 302)
        True_14688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 43), 'True')
        int_14689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 62), 'int')
        # Getting the type of 'None' (line 302)
        None_14690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 70), 'None')
        defaults = [True_14688, int_14689, None_14690]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 302, 4, False)
        # Assigning a type to the variable 'self' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DecodedGenerator.__init__', ['outfp', 'mangle_from_', 'maxheaderlen', 'fmt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['outfp', 'mangle_from_', 'maxheaderlen', 'fmt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_14691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, (-1)), 'str', "Like Generator.__init__() except that an additional optional\n        argument is allowed.\n\n        Walks through all subparts of a message.  If the subpart is of main\n        type `text', then it prints the decoded payload of the subpart.\n\n        Otherwise, fmt is a format string that is used instead of the message\n        payload.  fmt is expanded with the following keywords (in\n        %(keyword)s format):\n\n        type       : Full MIME type of the non-text part\n        maintype   : Main MIME type of the non-text part\n        subtype    : Sub-MIME type of the non-text part\n        filename   : Filename of the non-text part\n        description: Description associated with the non-text part\n        encoding   : Content transfer encoding of the non-text part\n\n        The default value for fmt is None, meaning\n\n        [Non-text (%(type)s) part of message omitted, filename %(filename)s]\n        ")
        
        # Call to __init__(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'self' (line 324)
        self_14694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 27), 'self', False)
        # Getting the type of 'outfp' (line 324)
        outfp_14695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 33), 'outfp', False)
        # Getting the type of 'mangle_from_' (line 324)
        mangle_from__14696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 40), 'mangle_from_', False)
        # Getting the type of 'maxheaderlen' (line 324)
        maxheaderlen_14697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 54), 'maxheaderlen', False)
        # Processing the call keyword arguments (line 324)
        kwargs_14698 = {}
        # Getting the type of 'Generator' (line 324)
        Generator_14692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'Generator', False)
        # Obtaining the member '__init__' of a type (line 324)
        init___14693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), Generator_14692, '__init__')
        # Calling __init__(args, kwargs) (line 324)
        init___call_result_14699 = invoke(stypy.reporting.localization.Localization(__file__, 324, 8), init___14693, *[self_14694, outfp_14695, mangle_from__14696, maxheaderlen_14697], **kwargs_14698)
        
        
        # Type idiom detected: calculating its left and rigth part (line 325)
        # Getting the type of 'fmt' (line 325)
        fmt_14700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'fmt')
        # Getting the type of 'None' (line 325)
        None_14701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 18), 'None')
        
        (may_be_14702, more_types_in_union_14703) = may_be_none(fmt_14700, None_14701)

        if may_be_14702:

            if more_types_in_union_14703:
                # Runtime conditional SSA (line 325)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 326):
            # Getting the type of '_FMT' (line 326)
            _FMT_14704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 24), '_FMT')
            # Getting the type of 'self' (line 326)
            self_14705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'self')
            # Setting the type of the member '_fmt' of a type (line 326)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 12), self_14705, '_fmt', _FMT_14704)

            if more_types_in_union_14703:
                # Runtime conditional SSA for else branch (line 325)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_14702) or more_types_in_union_14703):
            
            # Assigning a Name to a Attribute (line 328):
            # Getting the type of 'fmt' (line 328)
            fmt_14706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 24), 'fmt')
            # Getting the type of 'self' (line 328)
            self_14707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'self')
            # Setting the type of the member '_fmt' of a type (line 328)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 12), self_14707, '_fmt', fmt_14706)

            if (may_be_14702 and more_types_in_union_14703):
                # SSA join for if statement (line 325)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _dispatch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dispatch'
        module_type_store = module_type_store.open_function_context('_dispatch', 330, 4, False)
        # Assigning a type to the variable 'self' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_localization', localization)
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_type_store', module_type_store)
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_function_name', 'DecodedGenerator._dispatch')
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_varargs_param_name', None)
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_call_defaults', defaults)
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_call_varargs', varargs)
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DecodedGenerator._dispatch.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DecodedGenerator._dispatch', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dispatch', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dispatch(...)' code ##################

        
        
        # Call to walk(...): (line 331)
        # Processing the call keyword arguments (line 331)
        kwargs_14710 = {}
        # Getting the type of 'msg' (line 331)
        msg_14708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'msg', False)
        # Obtaining the member 'walk' of a type (line 331)
        walk_14709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 20), msg_14708, 'walk')
        # Calling walk(args, kwargs) (line 331)
        walk_call_result_14711 = invoke(stypy.reporting.localization.Localization(__file__, 331, 20), walk_14709, *[], **kwargs_14710)
        
        # Assigning a type to the variable 'walk_call_result_14711' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'walk_call_result_14711', walk_call_result_14711)
        # Testing if the for loop is going to be iterated (line 331)
        # Testing the type of a for loop iterable (line 331)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 331, 8), walk_call_result_14711)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 331, 8), walk_call_result_14711):
            # Getting the type of the for loop variable (line 331)
            for_loop_var_14712 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 331, 8), walk_call_result_14711)
            # Assigning a type to the variable 'part' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'part', for_loop_var_14712)
            # SSA begins for a for statement (line 331)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 332):
            
            # Call to get_content_maintype(...): (line 332)
            # Processing the call keyword arguments (line 332)
            kwargs_14715 = {}
            # Getting the type of 'part' (line 332)
            part_14713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 23), 'part', False)
            # Obtaining the member 'get_content_maintype' of a type (line 332)
            get_content_maintype_14714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 23), part_14713, 'get_content_maintype')
            # Calling get_content_maintype(args, kwargs) (line 332)
            get_content_maintype_call_result_14716 = invoke(stypy.reporting.localization.Localization(__file__, 332, 23), get_content_maintype_14714, *[], **kwargs_14715)
            
            # Assigning a type to the variable 'maintype' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'maintype', get_content_maintype_call_result_14716)
            
            # Getting the type of 'maintype' (line 333)
            maintype_14717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 15), 'maintype')
            str_14718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 27), 'str', 'text')
            # Applying the binary operator '==' (line 333)
            result_eq_14719 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 15), '==', maintype_14717, str_14718)
            
            # Testing if the type of an if condition is none (line 333)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 333, 12), result_eq_14719):
                
                # Getting the type of 'maintype' (line 335)
                maintype_14727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'maintype')
                str_14728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 29), 'str', 'multipart')
                # Applying the binary operator '==' (line 335)
                result_eq_14729 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 17), '==', maintype_14727, str_14728)
                
                # Testing if the type of an if condition is none (line 335)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 335, 17), result_eq_14729):
                    # Getting the type of 'self' (line 339)
                    self_14731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'self')
                    # Obtaining the member '_fmt' of a type (line 339)
                    _fmt_14732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 31), self_14731, '_fmt')
                    
                    # Obtaining an instance of the builtin type 'dict' (line 339)
                    dict_14733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 43), 'dict')
                    # Adding type elements to the builtin type 'dict' instance (line 339)
                    # Adding element type (key, value) (line 339)
                    str_14734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 20), 'str', 'type')
                    
                    # Call to get_content_type(...): (line 340)
                    # Processing the call keyword arguments (line 340)
                    kwargs_14737 = {}
                    # Getting the type of 'part' (line 340)
                    part_14735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 35), 'part', False)
                    # Obtaining the member 'get_content_type' of a type (line 340)
                    get_content_type_14736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 35), part_14735, 'get_content_type')
                    # Calling get_content_type(args, kwargs) (line 340)
                    get_content_type_call_result_14738 = invoke(stypy.reporting.localization.Localization(__file__, 340, 35), get_content_type_14736, *[], **kwargs_14737)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14734, get_content_type_call_result_14738))
                    # Adding element type (key, value) (line 339)
                    str_14739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 20), 'str', 'maintype')
                    
                    # Call to get_content_maintype(...): (line 341)
                    # Processing the call keyword arguments (line 341)
                    kwargs_14742 = {}
                    # Getting the type of 'part' (line 341)
                    part_14740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 35), 'part', False)
                    # Obtaining the member 'get_content_maintype' of a type (line 341)
                    get_content_maintype_14741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 35), part_14740, 'get_content_maintype')
                    # Calling get_content_maintype(args, kwargs) (line 341)
                    get_content_maintype_call_result_14743 = invoke(stypy.reporting.localization.Localization(__file__, 341, 35), get_content_maintype_14741, *[], **kwargs_14742)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14739, get_content_maintype_call_result_14743))
                    # Adding element type (key, value) (line 339)
                    str_14744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 20), 'str', 'subtype')
                    
                    # Call to get_content_subtype(...): (line 342)
                    # Processing the call keyword arguments (line 342)
                    kwargs_14747 = {}
                    # Getting the type of 'part' (line 342)
                    part_14745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 35), 'part', False)
                    # Obtaining the member 'get_content_subtype' of a type (line 342)
                    get_content_subtype_14746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 35), part_14745, 'get_content_subtype')
                    # Calling get_content_subtype(args, kwargs) (line 342)
                    get_content_subtype_call_result_14748 = invoke(stypy.reporting.localization.Localization(__file__, 342, 35), get_content_subtype_14746, *[], **kwargs_14747)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14744, get_content_subtype_call_result_14748))
                    # Adding element type (key, value) (line 339)
                    str_14749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 20), 'str', 'filename')
                    
                    # Call to get_filename(...): (line 343)
                    # Processing the call arguments (line 343)
                    str_14752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 53), 'str', '[no filename]')
                    # Processing the call keyword arguments (line 343)
                    kwargs_14753 = {}
                    # Getting the type of 'part' (line 343)
                    part_14750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 35), 'part', False)
                    # Obtaining the member 'get_filename' of a type (line 343)
                    get_filename_14751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 35), part_14750, 'get_filename')
                    # Calling get_filename(args, kwargs) (line 343)
                    get_filename_call_result_14754 = invoke(stypy.reporting.localization.Localization(__file__, 343, 35), get_filename_14751, *[str_14752], **kwargs_14753)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14749, get_filename_call_result_14754))
                    # Adding element type (key, value) (line 339)
                    str_14755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 20), 'str', 'description')
                    
                    # Call to get(...): (line 344)
                    # Processing the call arguments (line 344)
                    str_14758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 44), 'str', 'Content-Description')
                    str_14759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 44), 'str', '[no description]')
                    # Processing the call keyword arguments (line 344)
                    kwargs_14760 = {}
                    # Getting the type of 'part' (line 344)
                    part_14756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 35), 'part', False)
                    # Obtaining the member 'get' of a type (line 344)
                    get_14757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 35), part_14756, 'get')
                    # Calling get(args, kwargs) (line 344)
                    get_call_result_14761 = invoke(stypy.reporting.localization.Localization(__file__, 344, 35), get_14757, *[str_14758, str_14759], **kwargs_14760)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14755, get_call_result_14761))
                    # Adding element type (key, value) (line 339)
                    str_14762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 20), 'str', 'encoding')
                    
                    # Call to get(...): (line 346)
                    # Processing the call arguments (line 346)
                    str_14765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 44), 'str', 'Content-Transfer-Encoding')
                    str_14766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 44), 'str', '[no encoding]')
                    # Processing the call keyword arguments (line 346)
                    kwargs_14767 = {}
                    # Getting the type of 'part' (line 346)
                    part_14763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 35), 'part', False)
                    # Obtaining the member 'get' of a type (line 346)
                    get_14764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 35), part_14763, 'get')
                    # Calling get(args, kwargs) (line 346)
                    get_call_result_14768 = invoke(stypy.reporting.localization.Localization(__file__, 346, 35), get_14764, *[str_14765, str_14766], **kwargs_14767)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14762, get_call_result_14768))
                    
                    # Applying the binary operator '%' (line 339)
                    result_mod_14769 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 31), '%', _fmt_14732, dict_14733)
                    
                else:
                    
                    # Testing the type of an if condition (line 335)
                    if_condition_14730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 17), result_eq_14729)
                    # Assigning a type to the variable 'if_condition_14730' (line 335)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'if_condition_14730', if_condition_14730)
                    # SSA begins for if statement (line 335)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    pass
                    # SSA branch for the else part of an if statement (line 335)
                    module_type_store.open_ssa_branch('else')
                    # Getting the type of 'self' (line 339)
                    self_14731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'self')
                    # Obtaining the member '_fmt' of a type (line 339)
                    _fmt_14732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 31), self_14731, '_fmt')
                    
                    # Obtaining an instance of the builtin type 'dict' (line 339)
                    dict_14733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 43), 'dict')
                    # Adding type elements to the builtin type 'dict' instance (line 339)
                    # Adding element type (key, value) (line 339)
                    str_14734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 20), 'str', 'type')
                    
                    # Call to get_content_type(...): (line 340)
                    # Processing the call keyword arguments (line 340)
                    kwargs_14737 = {}
                    # Getting the type of 'part' (line 340)
                    part_14735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 35), 'part', False)
                    # Obtaining the member 'get_content_type' of a type (line 340)
                    get_content_type_14736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 35), part_14735, 'get_content_type')
                    # Calling get_content_type(args, kwargs) (line 340)
                    get_content_type_call_result_14738 = invoke(stypy.reporting.localization.Localization(__file__, 340, 35), get_content_type_14736, *[], **kwargs_14737)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14734, get_content_type_call_result_14738))
                    # Adding element type (key, value) (line 339)
                    str_14739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 20), 'str', 'maintype')
                    
                    # Call to get_content_maintype(...): (line 341)
                    # Processing the call keyword arguments (line 341)
                    kwargs_14742 = {}
                    # Getting the type of 'part' (line 341)
                    part_14740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 35), 'part', False)
                    # Obtaining the member 'get_content_maintype' of a type (line 341)
                    get_content_maintype_14741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 35), part_14740, 'get_content_maintype')
                    # Calling get_content_maintype(args, kwargs) (line 341)
                    get_content_maintype_call_result_14743 = invoke(stypy.reporting.localization.Localization(__file__, 341, 35), get_content_maintype_14741, *[], **kwargs_14742)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14739, get_content_maintype_call_result_14743))
                    # Adding element type (key, value) (line 339)
                    str_14744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 20), 'str', 'subtype')
                    
                    # Call to get_content_subtype(...): (line 342)
                    # Processing the call keyword arguments (line 342)
                    kwargs_14747 = {}
                    # Getting the type of 'part' (line 342)
                    part_14745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 35), 'part', False)
                    # Obtaining the member 'get_content_subtype' of a type (line 342)
                    get_content_subtype_14746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 35), part_14745, 'get_content_subtype')
                    # Calling get_content_subtype(args, kwargs) (line 342)
                    get_content_subtype_call_result_14748 = invoke(stypy.reporting.localization.Localization(__file__, 342, 35), get_content_subtype_14746, *[], **kwargs_14747)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14744, get_content_subtype_call_result_14748))
                    # Adding element type (key, value) (line 339)
                    str_14749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 20), 'str', 'filename')
                    
                    # Call to get_filename(...): (line 343)
                    # Processing the call arguments (line 343)
                    str_14752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 53), 'str', '[no filename]')
                    # Processing the call keyword arguments (line 343)
                    kwargs_14753 = {}
                    # Getting the type of 'part' (line 343)
                    part_14750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 35), 'part', False)
                    # Obtaining the member 'get_filename' of a type (line 343)
                    get_filename_14751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 35), part_14750, 'get_filename')
                    # Calling get_filename(args, kwargs) (line 343)
                    get_filename_call_result_14754 = invoke(stypy.reporting.localization.Localization(__file__, 343, 35), get_filename_14751, *[str_14752], **kwargs_14753)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14749, get_filename_call_result_14754))
                    # Adding element type (key, value) (line 339)
                    str_14755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 20), 'str', 'description')
                    
                    # Call to get(...): (line 344)
                    # Processing the call arguments (line 344)
                    str_14758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 44), 'str', 'Content-Description')
                    str_14759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 44), 'str', '[no description]')
                    # Processing the call keyword arguments (line 344)
                    kwargs_14760 = {}
                    # Getting the type of 'part' (line 344)
                    part_14756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 35), 'part', False)
                    # Obtaining the member 'get' of a type (line 344)
                    get_14757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 35), part_14756, 'get')
                    # Calling get(args, kwargs) (line 344)
                    get_call_result_14761 = invoke(stypy.reporting.localization.Localization(__file__, 344, 35), get_14757, *[str_14758, str_14759], **kwargs_14760)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14755, get_call_result_14761))
                    # Adding element type (key, value) (line 339)
                    str_14762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 20), 'str', 'encoding')
                    
                    # Call to get(...): (line 346)
                    # Processing the call arguments (line 346)
                    str_14765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 44), 'str', 'Content-Transfer-Encoding')
                    str_14766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 44), 'str', '[no encoding]')
                    # Processing the call keyword arguments (line 346)
                    kwargs_14767 = {}
                    # Getting the type of 'part' (line 346)
                    part_14763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 35), 'part', False)
                    # Obtaining the member 'get' of a type (line 346)
                    get_14764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 35), part_14763, 'get')
                    # Calling get(args, kwargs) (line 346)
                    get_call_result_14768 = invoke(stypy.reporting.localization.Localization(__file__, 346, 35), get_14764, *[str_14765, str_14766], **kwargs_14767)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14762, get_call_result_14768))
                    
                    # Applying the binary operator '%' (line 339)
                    result_mod_14769 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 31), '%', _fmt_14732, dict_14733)
                    
                    # SSA join for if statement (line 335)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 333)
                if_condition_14720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 12), result_eq_14719)
                # Assigning a type to the variable 'if_condition_14720' (line 333)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'if_condition_14720', if_condition_14720)
                # SSA begins for if statement (line 333)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to get_payload(...): (line 334)
                # Processing the call keyword arguments (line 334)
                # Getting the type of 'True' (line 334)
                True_14723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 55), 'True', False)
                keyword_14724 = True_14723
                kwargs_14725 = {'decode': keyword_14724}
                # Getting the type of 'part' (line 334)
                part_14721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 31), 'part', False)
                # Obtaining the member 'get_payload' of a type (line 334)
                get_payload_14722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 31), part_14721, 'get_payload')
                # Calling get_payload(args, kwargs) (line 334)
                get_payload_call_result_14726 = invoke(stypy.reporting.localization.Localization(__file__, 334, 31), get_payload_14722, *[], **kwargs_14725)
                
                # SSA branch for the else part of an if statement (line 333)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'maintype' (line 335)
                maintype_14727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'maintype')
                str_14728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 29), 'str', 'multipart')
                # Applying the binary operator '==' (line 335)
                result_eq_14729 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 17), '==', maintype_14727, str_14728)
                
                # Testing if the type of an if condition is none (line 335)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 335, 17), result_eq_14729):
                    # Getting the type of 'self' (line 339)
                    self_14731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'self')
                    # Obtaining the member '_fmt' of a type (line 339)
                    _fmt_14732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 31), self_14731, '_fmt')
                    
                    # Obtaining an instance of the builtin type 'dict' (line 339)
                    dict_14733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 43), 'dict')
                    # Adding type elements to the builtin type 'dict' instance (line 339)
                    # Adding element type (key, value) (line 339)
                    str_14734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 20), 'str', 'type')
                    
                    # Call to get_content_type(...): (line 340)
                    # Processing the call keyword arguments (line 340)
                    kwargs_14737 = {}
                    # Getting the type of 'part' (line 340)
                    part_14735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 35), 'part', False)
                    # Obtaining the member 'get_content_type' of a type (line 340)
                    get_content_type_14736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 35), part_14735, 'get_content_type')
                    # Calling get_content_type(args, kwargs) (line 340)
                    get_content_type_call_result_14738 = invoke(stypy.reporting.localization.Localization(__file__, 340, 35), get_content_type_14736, *[], **kwargs_14737)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14734, get_content_type_call_result_14738))
                    # Adding element type (key, value) (line 339)
                    str_14739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 20), 'str', 'maintype')
                    
                    # Call to get_content_maintype(...): (line 341)
                    # Processing the call keyword arguments (line 341)
                    kwargs_14742 = {}
                    # Getting the type of 'part' (line 341)
                    part_14740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 35), 'part', False)
                    # Obtaining the member 'get_content_maintype' of a type (line 341)
                    get_content_maintype_14741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 35), part_14740, 'get_content_maintype')
                    # Calling get_content_maintype(args, kwargs) (line 341)
                    get_content_maintype_call_result_14743 = invoke(stypy.reporting.localization.Localization(__file__, 341, 35), get_content_maintype_14741, *[], **kwargs_14742)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14739, get_content_maintype_call_result_14743))
                    # Adding element type (key, value) (line 339)
                    str_14744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 20), 'str', 'subtype')
                    
                    # Call to get_content_subtype(...): (line 342)
                    # Processing the call keyword arguments (line 342)
                    kwargs_14747 = {}
                    # Getting the type of 'part' (line 342)
                    part_14745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 35), 'part', False)
                    # Obtaining the member 'get_content_subtype' of a type (line 342)
                    get_content_subtype_14746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 35), part_14745, 'get_content_subtype')
                    # Calling get_content_subtype(args, kwargs) (line 342)
                    get_content_subtype_call_result_14748 = invoke(stypy.reporting.localization.Localization(__file__, 342, 35), get_content_subtype_14746, *[], **kwargs_14747)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14744, get_content_subtype_call_result_14748))
                    # Adding element type (key, value) (line 339)
                    str_14749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 20), 'str', 'filename')
                    
                    # Call to get_filename(...): (line 343)
                    # Processing the call arguments (line 343)
                    str_14752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 53), 'str', '[no filename]')
                    # Processing the call keyword arguments (line 343)
                    kwargs_14753 = {}
                    # Getting the type of 'part' (line 343)
                    part_14750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 35), 'part', False)
                    # Obtaining the member 'get_filename' of a type (line 343)
                    get_filename_14751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 35), part_14750, 'get_filename')
                    # Calling get_filename(args, kwargs) (line 343)
                    get_filename_call_result_14754 = invoke(stypy.reporting.localization.Localization(__file__, 343, 35), get_filename_14751, *[str_14752], **kwargs_14753)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14749, get_filename_call_result_14754))
                    # Adding element type (key, value) (line 339)
                    str_14755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 20), 'str', 'description')
                    
                    # Call to get(...): (line 344)
                    # Processing the call arguments (line 344)
                    str_14758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 44), 'str', 'Content-Description')
                    str_14759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 44), 'str', '[no description]')
                    # Processing the call keyword arguments (line 344)
                    kwargs_14760 = {}
                    # Getting the type of 'part' (line 344)
                    part_14756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 35), 'part', False)
                    # Obtaining the member 'get' of a type (line 344)
                    get_14757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 35), part_14756, 'get')
                    # Calling get(args, kwargs) (line 344)
                    get_call_result_14761 = invoke(stypy.reporting.localization.Localization(__file__, 344, 35), get_14757, *[str_14758, str_14759], **kwargs_14760)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14755, get_call_result_14761))
                    # Adding element type (key, value) (line 339)
                    str_14762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 20), 'str', 'encoding')
                    
                    # Call to get(...): (line 346)
                    # Processing the call arguments (line 346)
                    str_14765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 44), 'str', 'Content-Transfer-Encoding')
                    str_14766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 44), 'str', '[no encoding]')
                    # Processing the call keyword arguments (line 346)
                    kwargs_14767 = {}
                    # Getting the type of 'part' (line 346)
                    part_14763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 35), 'part', False)
                    # Obtaining the member 'get' of a type (line 346)
                    get_14764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 35), part_14763, 'get')
                    # Calling get(args, kwargs) (line 346)
                    get_call_result_14768 = invoke(stypy.reporting.localization.Localization(__file__, 346, 35), get_14764, *[str_14765, str_14766], **kwargs_14767)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14762, get_call_result_14768))
                    
                    # Applying the binary operator '%' (line 339)
                    result_mod_14769 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 31), '%', _fmt_14732, dict_14733)
                    
                else:
                    
                    # Testing the type of an if condition (line 335)
                    if_condition_14730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 17), result_eq_14729)
                    # Assigning a type to the variable 'if_condition_14730' (line 335)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'if_condition_14730', if_condition_14730)
                    # SSA begins for if statement (line 335)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    pass
                    # SSA branch for the else part of an if statement (line 335)
                    module_type_store.open_ssa_branch('else')
                    # Getting the type of 'self' (line 339)
                    self_14731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'self')
                    # Obtaining the member '_fmt' of a type (line 339)
                    _fmt_14732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 31), self_14731, '_fmt')
                    
                    # Obtaining an instance of the builtin type 'dict' (line 339)
                    dict_14733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 43), 'dict')
                    # Adding type elements to the builtin type 'dict' instance (line 339)
                    # Adding element type (key, value) (line 339)
                    str_14734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 20), 'str', 'type')
                    
                    # Call to get_content_type(...): (line 340)
                    # Processing the call keyword arguments (line 340)
                    kwargs_14737 = {}
                    # Getting the type of 'part' (line 340)
                    part_14735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 35), 'part', False)
                    # Obtaining the member 'get_content_type' of a type (line 340)
                    get_content_type_14736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 35), part_14735, 'get_content_type')
                    # Calling get_content_type(args, kwargs) (line 340)
                    get_content_type_call_result_14738 = invoke(stypy.reporting.localization.Localization(__file__, 340, 35), get_content_type_14736, *[], **kwargs_14737)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14734, get_content_type_call_result_14738))
                    # Adding element type (key, value) (line 339)
                    str_14739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 20), 'str', 'maintype')
                    
                    # Call to get_content_maintype(...): (line 341)
                    # Processing the call keyword arguments (line 341)
                    kwargs_14742 = {}
                    # Getting the type of 'part' (line 341)
                    part_14740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 35), 'part', False)
                    # Obtaining the member 'get_content_maintype' of a type (line 341)
                    get_content_maintype_14741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 35), part_14740, 'get_content_maintype')
                    # Calling get_content_maintype(args, kwargs) (line 341)
                    get_content_maintype_call_result_14743 = invoke(stypy.reporting.localization.Localization(__file__, 341, 35), get_content_maintype_14741, *[], **kwargs_14742)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14739, get_content_maintype_call_result_14743))
                    # Adding element type (key, value) (line 339)
                    str_14744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 20), 'str', 'subtype')
                    
                    # Call to get_content_subtype(...): (line 342)
                    # Processing the call keyword arguments (line 342)
                    kwargs_14747 = {}
                    # Getting the type of 'part' (line 342)
                    part_14745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 35), 'part', False)
                    # Obtaining the member 'get_content_subtype' of a type (line 342)
                    get_content_subtype_14746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 35), part_14745, 'get_content_subtype')
                    # Calling get_content_subtype(args, kwargs) (line 342)
                    get_content_subtype_call_result_14748 = invoke(stypy.reporting.localization.Localization(__file__, 342, 35), get_content_subtype_14746, *[], **kwargs_14747)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14744, get_content_subtype_call_result_14748))
                    # Adding element type (key, value) (line 339)
                    str_14749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 20), 'str', 'filename')
                    
                    # Call to get_filename(...): (line 343)
                    # Processing the call arguments (line 343)
                    str_14752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 53), 'str', '[no filename]')
                    # Processing the call keyword arguments (line 343)
                    kwargs_14753 = {}
                    # Getting the type of 'part' (line 343)
                    part_14750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 35), 'part', False)
                    # Obtaining the member 'get_filename' of a type (line 343)
                    get_filename_14751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 35), part_14750, 'get_filename')
                    # Calling get_filename(args, kwargs) (line 343)
                    get_filename_call_result_14754 = invoke(stypy.reporting.localization.Localization(__file__, 343, 35), get_filename_14751, *[str_14752], **kwargs_14753)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14749, get_filename_call_result_14754))
                    # Adding element type (key, value) (line 339)
                    str_14755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 20), 'str', 'description')
                    
                    # Call to get(...): (line 344)
                    # Processing the call arguments (line 344)
                    str_14758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 44), 'str', 'Content-Description')
                    str_14759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 44), 'str', '[no description]')
                    # Processing the call keyword arguments (line 344)
                    kwargs_14760 = {}
                    # Getting the type of 'part' (line 344)
                    part_14756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 35), 'part', False)
                    # Obtaining the member 'get' of a type (line 344)
                    get_14757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 35), part_14756, 'get')
                    # Calling get(args, kwargs) (line 344)
                    get_call_result_14761 = invoke(stypy.reporting.localization.Localization(__file__, 344, 35), get_14757, *[str_14758, str_14759], **kwargs_14760)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14755, get_call_result_14761))
                    # Adding element type (key, value) (line 339)
                    str_14762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 20), 'str', 'encoding')
                    
                    # Call to get(...): (line 346)
                    # Processing the call arguments (line 346)
                    str_14765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 44), 'str', 'Content-Transfer-Encoding')
                    str_14766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 44), 'str', '[no encoding]')
                    # Processing the call keyword arguments (line 346)
                    kwargs_14767 = {}
                    # Getting the type of 'part' (line 346)
                    part_14763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 35), 'part', False)
                    # Obtaining the member 'get' of a type (line 346)
                    get_14764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 35), part_14763, 'get')
                    # Calling get(args, kwargs) (line 346)
                    get_call_result_14768 = invoke(stypy.reporting.localization.Localization(__file__, 346, 35), get_14764, *[str_14765, str_14766], **kwargs_14767)
                    
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 43), dict_14733, (str_14762, get_call_result_14768))
                    
                    # Applying the binary operator '%' (line 339)
                    result_mod_14769 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 31), '%', _fmt_14732, dict_14733)
                    
                    # SSA join for if statement (line 335)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 333)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '_dispatch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dispatch' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_14770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14770)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dispatch'
        return stypy_return_type_14770


# Assigning a type to the variable 'DecodedGenerator' (line 296)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'DecodedGenerator', DecodedGenerator)

# Assigning a Call to a Name (line 353):

# Call to len(...): (line 353)
# Processing the call arguments (line 353)

# Call to repr(...): (line 353)
# Processing the call arguments (line 353)
# Getting the type of 'sys' (line 353)
sys_14773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 18), 'sys', False)
# Obtaining the member 'maxint' of a type (line 353)
maxint_14774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 18), sys_14773, 'maxint')
int_14775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 29), 'int')
# Applying the binary operator '-' (line 353)
result_sub_14776 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 18), '-', maxint_14774, int_14775)

# Processing the call keyword arguments (line 353)
kwargs_14777 = {}
# Getting the type of 'repr' (line 353)
repr_14772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 13), 'repr', False)
# Calling repr(args, kwargs) (line 353)
repr_call_result_14778 = invoke(stypy.reporting.localization.Localization(__file__, 353, 13), repr_14772, *[result_sub_14776], **kwargs_14777)

# Processing the call keyword arguments (line 353)
kwargs_14779 = {}
# Getting the type of 'len' (line 353)
len_14771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 9), 'len', False)
# Calling len(args, kwargs) (line 353)
len_call_result_14780 = invoke(stypy.reporting.localization.Localization(__file__, 353, 9), len_14771, *[repr_call_result_14778], **kwargs_14779)

# Assigning a type to the variable '_width' (line 353)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), '_width', len_call_result_14780)

# Assigning a BinOp to a Name (line 354):
str_14781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 7), 'str', '%%0%dd')
# Getting the type of '_width' (line 354)
_width_14782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 18), '_width')
# Applying the binary operator '%' (line 354)
result_mod_14783 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 7), '%', str_14781, _width_14782)

# Assigning a type to the variable '_fmt' (line 354)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), '_fmt', result_mod_14783)

@norecursion
def _make_boundary(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 356)
    None_14784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'None')
    defaults = [None_14784]
    # Create a new context for function '_make_boundary'
    module_type_store = module_type_store.open_function_context('_make_boundary', 356, 0, False)
    
    # Passed parameters checking function
    _make_boundary.stypy_localization = localization
    _make_boundary.stypy_type_of_self = None
    _make_boundary.stypy_type_store = module_type_store
    _make_boundary.stypy_function_name = '_make_boundary'
    _make_boundary.stypy_param_names_list = ['text']
    _make_boundary.stypy_varargs_param_name = None
    _make_boundary.stypy_kwargs_param_name = None
    _make_boundary.stypy_call_defaults = defaults
    _make_boundary.stypy_call_varargs = varargs
    _make_boundary.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_make_boundary', ['text'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_make_boundary', localization, ['text'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_make_boundary(...)' code ##################

    
    # Assigning a Call to a Name (line 359):
    
    # Call to randrange(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'sys' (line 359)
    sys_14787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 29), 'sys', False)
    # Obtaining the member 'maxint' of a type (line 359)
    maxint_14788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 29), sys_14787, 'maxint')
    # Processing the call keyword arguments (line 359)
    kwargs_14789 = {}
    # Getting the type of 'random' (line 359)
    random_14785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'random', False)
    # Obtaining the member 'randrange' of a type (line 359)
    randrange_14786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), random_14785, 'randrange')
    # Calling randrange(args, kwargs) (line 359)
    randrange_call_result_14790 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), randrange_14786, *[maxint_14788], **kwargs_14789)
    
    # Assigning a type to the variable 'token' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'token', randrange_call_result_14790)
    
    # Assigning a BinOp to a Name (line 360):
    str_14791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 16), 'str', '=')
    int_14792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 22), 'int')
    # Applying the binary operator '*' (line 360)
    result_mul_14793 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 16), '*', str_14791, int_14792)
    
    # Getting the type of '_fmt' (line 360)
    _fmt_14794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 29), '_fmt')
    # Getting the type of 'token' (line 360)
    token_14795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 36), 'token')
    # Applying the binary operator '%' (line 360)
    result_mod_14796 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 29), '%', _fmt_14794, token_14795)
    
    # Applying the binary operator '+' (line 360)
    result_add_14797 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 15), '+', result_mul_14793, result_mod_14796)
    
    str_14798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 45), 'str', '==')
    # Applying the binary operator '+' (line 360)
    result_add_14799 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 43), '+', result_add_14797, str_14798)
    
    # Assigning a type to the variable 'boundary' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'boundary', result_add_14799)
    
    # Type idiom detected: calculating its left and rigth part (line 361)
    # Getting the type of 'text' (line 361)
    text_14800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 7), 'text')
    # Getting the type of 'None' (line 361)
    None_14801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 15), 'None')
    
    (may_be_14802, more_types_in_union_14803) = may_be_none(text_14800, None_14801)

    if may_be_14802:

        if more_types_in_union_14803:
            # Runtime conditional SSA (line 361)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'boundary' (line 362)
        boundary_14804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'boundary')
        # Assigning a type to the variable 'stypy_return_type' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'stypy_return_type', boundary_14804)

        if more_types_in_union_14803:
            # SSA join for if statement (line 361)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'text' (line 361)
    text_14805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'text')
    # Assigning a type to the variable 'text' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'text', remove_type_from_union(text_14805, types.NoneType))
    
    # Assigning a Name to a Name (line 363):
    # Getting the type of 'boundary' (line 363)
    boundary_14806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'boundary')
    # Assigning a type to the variable 'b' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'b', boundary_14806)
    
    # Assigning a Num to a Name (line 364):
    int_14807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 14), 'int')
    # Assigning a type to the variable 'counter' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'counter', int_14807)
    
    # Getting the type of 'True' (line 365)
    True_14808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 10), 'True')
    # Assigning a type to the variable 'True_14808' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'True_14808', True_14808)
    # Testing if the while is going to be iterated (line 365)
    # Testing the type of an if condition (line 365)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 4), True_14808)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 365, 4), True_14808):
        
        # Assigning a Call to a Name (line 366):
        
        # Call to compile(...): (line 366)
        # Processing the call arguments (line 366)
        str_14811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 25), 'str', '^--')
        
        # Call to escape(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'b' (line 366)
        b_14814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 43), 'b', False)
        # Processing the call keyword arguments (line 366)
        kwargs_14815 = {}
        # Getting the type of 're' (line 366)
        re_14812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 33), 're', False)
        # Obtaining the member 'escape' of a type (line 366)
        escape_14813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 33), re_14812, 'escape')
        # Calling escape(args, kwargs) (line 366)
        escape_call_result_14816 = invoke(stypy.reporting.localization.Localization(__file__, 366, 33), escape_14813, *[b_14814], **kwargs_14815)
        
        # Applying the binary operator '+' (line 366)
        result_add_14817 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 25), '+', str_14811, escape_call_result_14816)
        
        str_14818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 48), 'str', '(--)?$')
        # Applying the binary operator '+' (line 366)
        result_add_14819 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 46), '+', result_add_14817, str_14818)
        
        # Getting the type of 're' (line 366)
        re_14820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 58), 're', False)
        # Obtaining the member 'MULTILINE' of a type (line 366)
        MULTILINE_14821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 58), re_14820, 'MULTILINE')
        # Processing the call keyword arguments (line 366)
        kwargs_14822 = {}
        # Getting the type of 're' (line 366)
        re_14809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 14), 're', False)
        # Obtaining the member 'compile' of a type (line 366)
        compile_14810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 14), re_14809, 'compile')
        # Calling compile(args, kwargs) (line 366)
        compile_call_result_14823 = invoke(stypy.reporting.localization.Localization(__file__, 366, 14), compile_14810, *[result_add_14819, MULTILINE_14821], **kwargs_14822)
        
        # Assigning a type to the variable 'cre' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'cre', compile_call_result_14823)
        
        
        # Call to search(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'text' (line 367)
        text_14826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 26), 'text', False)
        # Processing the call keyword arguments (line 367)
        kwargs_14827 = {}
        # Getting the type of 'cre' (line 367)
        cre_14824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 15), 'cre', False)
        # Obtaining the member 'search' of a type (line 367)
        search_14825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 15), cre_14824, 'search')
        # Calling search(args, kwargs) (line 367)
        search_call_result_14828 = invoke(stypy.reporting.localization.Localization(__file__, 367, 15), search_14825, *[text_14826], **kwargs_14827)
        
        # Applying the 'not' unary operator (line 367)
        result_not__14829 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 11), 'not', search_call_result_14828)
        
        # Testing if the type of an if condition is none (line 367)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 367, 8), result_not__14829):
            pass
        else:
            
            # Testing the type of an if condition (line 367)
            if_condition_14830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 8), result_not__14829)
            # Assigning a type to the variable 'if_condition_14830' (line 367)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'if_condition_14830', if_condition_14830)
            # SSA begins for if statement (line 367)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 367)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 369):
        # Getting the type of 'boundary' (line 369)
        boundary_14831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'boundary')
        str_14832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 23), 'str', '.')
        # Applying the binary operator '+' (line 369)
        result_add_14833 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 12), '+', boundary_14831, str_14832)
        
        
        # Call to str(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'counter' (line 369)
        counter_14835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 33), 'counter', False)
        # Processing the call keyword arguments (line 369)
        kwargs_14836 = {}
        # Getting the type of 'str' (line 369)
        str_14834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 29), 'str', False)
        # Calling str(args, kwargs) (line 369)
        str_call_result_14837 = invoke(stypy.reporting.localization.Localization(__file__, 369, 29), str_14834, *[counter_14835], **kwargs_14836)
        
        # Applying the binary operator '+' (line 369)
        result_add_14838 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 27), '+', result_add_14833, str_call_result_14837)
        
        # Assigning a type to the variable 'b' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'b', result_add_14838)
        
        # Getting the type of 'counter' (line 370)
        counter_14839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'counter')
        int_14840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 19), 'int')
        # Applying the binary operator '+=' (line 370)
        result_iadd_14841 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 8), '+=', counter_14839, int_14840)
        # Assigning a type to the variable 'counter' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'counter', result_iadd_14841)
        

    
    # Getting the type of 'b' (line 371)
    b_14842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'b')
    # Assigning a type to the variable 'stypy_return_type' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type', b_14842)
    
    # ################# End of '_make_boundary(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_make_boundary' in the type store
    # Getting the type of 'stypy_return_type' (line 356)
    stypy_return_type_14843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14843)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_make_boundary'
    return stypy_return_type_14843

# Assigning a type to the variable '_make_boundary' (line 356)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 0), '_make_boundary', _make_boundary)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
