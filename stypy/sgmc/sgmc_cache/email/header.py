
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2002-2006 Python Software Foundation
2: # Author: Ben Gertzfield, Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Header encoding and decoding functionality.'''
6: 
7: __all__ = [
8:     'Header',
9:     'decode_header',
10:     'make_header',
11:     ]
12: 
13: import re
14: import binascii
15: 
16: import email.quoprimime
17: import email.base64mime
18: 
19: from email.errors import HeaderParseError
20: from email.charset import Charset
21: 
22: NL = '\n'
23: SPACE = ' '
24: USPACE = u' '
25: SPACE8 = ' ' * 8
26: UEMPTYSTRING = u''
27: 
28: MAXLINELEN = 76
29: 
30: USASCII = Charset('us-ascii')
31: UTF8 = Charset('utf-8')
32: 
33: # Match encoded-word strings in the form =?charset?q?Hello_World?=
34: ecre = re.compile(r'''
35:   =\?                   # literal =?
36:   (?P<charset>[^?]*?)   # non-greedy up to the next ? is the charset
37:   \?                    # literal ?
38:   (?P<encoding>[qb])    # either a "q" or a "b", case insensitive
39:   \?                    # literal ?
40:   (?P<encoded>.*?)      # non-greedy up to the next ?= is the encoded string
41:   \?=                   # literal ?=
42:   (?=[ \t]|$)           # whitespace or the end of the string
43:   ''', re.VERBOSE | re.IGNORECASE | re.MULTILINE)
44: 
45: # Field name regexp, including trailing colon, but not separating whitespace,
46: # according to RFC 2822.  Character range is from tilde to exclamation mark.
47: # For use with .match()
48: fcre = re.compile(r'[\041-\176]+:$')
49: 
50: # Find a header embedded in a putative header value.  Used to check for
51: # header injection attack.
52: _embeded_header = re.compile(r'\n[^ \t]+:')
53: 
54: 
55: 
56: # Helpers
57: _max_append = email.quoprimime._max_append
58: 
59: 
60: 
61: def decode_header(header):
62:     '''Decode a message header value without converting charset.
63: 
64:     Returns a list of (decoded_string, charset) pairs containing each of the
65:     decoded parts of the header.  Charset is None for non-encoded parts of the
66:     header, otherwise a lower-case string containing the name of the character
67:     set specified in the encoded string.
68: 
69:     An email.errors.HeaderParseError may be raised when certain decoding error
70:     occurs (e.g. a base64 decoding exception).
71:     '''
72:     # If no encoding, just return the header
73:     header = str(header)
74:     if not ecre.search(header):
75:         return [(header, None)]
76:     decoded = []
77:     dec = ''
78:     for line in header.splitlines():
79:         # This line might not have an encoding in it
80:         if not ecre.search(line):
81:             decoded.append((line, None))
82:             continue
83:         parts = ecre.split(line)
84:         while parts:
85:             unenc = parts.pop(0).strip()
86:             if unenc:
87:                 # Should we continue a long line?
88:                 if decoded and decoded[-1][1] is None:
89:                     decoded[-1] = (decoded[-1][0] + SPACE + unenc, None)
90:                 else:
91:                     decoded.append((unenc, None))
92:             if parts:
93:                 charset, encoding = [s.lower() for s in parts[0:2]]
94:                 encoded = parts[2]
95:                 dec = None
96:                 if encoding == 'q':
97:                     dec = email.quoprimime.header_decode(encoded)
98:                 elif encoding == 'b':
99:                     paderr = len(encoded) % 4   # Postel's law: add missing padding
100:                     if paderr:
101:                         encoded += '==='[:4 - paderr]
102:                     try:
103:                         dec = email.base64mime.decode(encoded)
104:                     except binascii.Error:
105:                         # Turn this into a higher level exception.  BAW: Right
106:                         # now we throw the lower level exception away but
107:                         # when/if we get exception chaining, we'll preserve it.
108:                         raise HeaderParseError
109:                 if dec is None:
110:                     dec = encoded
111: 
112:                 if decoded and decoded[-1][1] == charset:
113:                     decoded[-1] = (decoded[-1][0] + dec, decoded[-1][1])
114:                 else:
115:                     decoded.append((dec, charset))
116:             del parts[0:3]
117:     return decoded
118: 
119: 
120: 
121: def make_header(decoded_seq, maxlinelen=None, header_name=None,
122:                 continuation_ws=' '):
123:     '''Create a Header from a sequence of pairs as returned by decode_header()
124: 
125:     decode_header() takes a header value string and returns a sequence of
126:     pairs of the format (decoded_string, charset) where charset is the string
127:     name of the character set.
128: 
129:     This function takes one of those sequence of pairs and returns a Header
130:     instance.  Optional maxlinelen, header_name, and continuation_ws are as in
131:     the Header constructor.
132:     '''
133:     h = Header(maxlinelen=maxlinelen, header_name=header_name,
134:                continuation_ws=continuation_ws)
135:     for s, charset in decoded_seq:
136:         # None means us-ascii but we can simply pass it on to h.append()
137:         if charset is not None and not isinstance(charset, Charset):
138:             charset = Charset(charset)
139:         h.append(s, charset)
140:     return h
141: 
142: 
143: 
144: class Header:
145:     def __init__(self, s=None, charset=None,
146:                  maxlinelen=None, header_name=None,
147:                  continuation_ws=' ', errors='strict'):
148:         '''Create a MIME-compliant header that can contain many character sets.
149: 
150:         Optional s is the initial header value.  If None, the initial header
151:         value is not set.  You can later append to the header with .append()
152:         method calls.  s may be a byte string or a Unicode string, but see the
153:         .append() documentation for semantics.
154: 
155:         Optional charset serves two purposes: it has the same meaning as the
156:         charset argument to the .append() method.  It also sets the default
157:         character set for all subsequent .append() calls that omit the charset
158:         argument.  If charset is not provided in the constructor, the us-ascii
159:         charset is used both as s's initial charset and as the default for
160:         subsequent .append() calls.
161: 
162:         The maximum line length can be specified explicit via maxlinelen.  For
163:         splitting the first line to a shorter value (to account for the field
164:         header which isn't included in s, e.g. `Subject') pass in the name of
165:         the field in header_name.  The default maxlinelen is 76.
166: 
167:         continuation_ws must be RFC 2822 compliant folding whitespace (usually
168:         either a space or a hard tab) which will be prepended to continuation
169:         lines.
170: 
171:         errors is passed through to the .append() call.
172:         '''
173:         if charset is None:
174:             charset = USASCII
175:         if not isinstance(charset, Charset):
176:             charset = Charset(charset)
177:         self._charset = charset
178:         self._continuation_ws = continuation_ws
179:         cws_expanded_len = len(continuation_ws.replace('\t', SPACE8))
180:         # BAW: I believe `chunks' and `maxlinelen' should be non-public.
181:         self._chunks = []
182:         if s is not None:
183:             self.append(s, charset, errors)
184:         if maxlinelen is None:
185:             maxlinelen = MAXLINELEN
186:         if header_name is None:
187:             # We don't know anything about the field header so the first line
188:             # is the same length as subsequent lines.
189:             self._firstlinelen = maxlinelen
190:         else:
191:             # The first line should be shorter to take into account the field
192:             # header.  Also subtract off 2 extra for the colon and space.
193:             self._firstlinelen = maxlinelen - len(header_name) - 2
194:         # Second and subsequent lines should subtract off the length in
195:         # columns of the continuation whitespace prefix.
196:         self._maxlinelen = maxlinelen - cws_expanded_len
197: 
198:     def __str__(self):
199:         '''A synonym for self.encode().'''
200:         return self.encode()
201: 
202:     def __unicode__(self):
203:         '''Helper for the built-in unicode function.'''
204:         uchunks = []
205:         lastcs = None
206:         for s, charset in self._chunks:
207:             # We must preserve spaces between encoded and non-encoded word
208:             # boundaries, which means for us we need to add a space when we go
209:             # from a charset to None/us-ascii, or from None/us-ascii to a
210:             # charset.  Only do this for the second and subsequent chunks.
211:             nextcs = charset
212:             if uchunks:
213:                 if lastcs not in (None, 'us-ascii'):
214:                     if nextcs in (None, 'us-ascii'):
215:                         uchunks.append(USPACE)
216:                         nextcs = None
217:                 elif nextcs not in (None, 'us-ascii'):
218:                     uchunks.append(USPACE)
219:             lastcs = nextcs
220:             uchunks.append(unicode(s, str(charset)))
221:         return UEMPTYSTRING.join(uchunks)
222: 
223:     # Rich comparison operators for equality only.  BAW: does it make sense to
224:     # have or explicitly disable <, <=, >, >= operators?
225:     def __eq__(self, other):
226:         # other may be a Header or a string.  Both are fine so coerce
227:         # ourselves to a string, swap the args and do another comparison.
228:         return other == self.encode()
229: 
230:     def __ne__(self, other):
231:         return not self == other
232: 
233:     def append(self, s, charset=None, errors='strict'):
234:         '''Append a string to the MIME header.
235: 
236:         Optional charset, if given, should be a Charset instance or the name
237:         of a character set (which will be converted to a Charset instance).  A
238:         value of None (the default) means that the charset given in the
239:         constructor is used.
240: 
241:         s may be a byte string or a Unicode string.  If it is a byte string
242:         (i.e. isinstance(s, str) is true), then charset is the encoding of
243:         that byte string, and a UnicodeError will be raised if the string
244:         cannot be decoded with that charset.  If s is a Unicode string, then
245:         charset is a hint specifying the character set of the characters in
246:         the string.  In this case, when producing an RFC 2822 compliant header
247:         using RFC 2047 rules, the Unicode string will be encoded using the
248:         following charsets in order: us-ascii, the charset hint, utf-8.  The
249:         first character set not to provoke a UnicodeError is used.
250: 
251:         Optional `errors' is passed as the third argument to any unicode() or
252:         ustr.encode() call.
253:         '''
254:         if charset is None:
255:             charset = self._charset
256:         elif not isinstance(charset, Charset):
257:             charset = Charset(charset)
258:         # If the charset is our faux 8bit charset, leave the string unchanged
259:         if charset != '8bit':
260:             # We need to test that the string can be converted to unicode and
261:             # back to a byte string, given the input and output codecs of the
262:             # charset.
263:             if isinstance(s, str):
264:                 # Possibly raise UnicodeError if the byte string can't be
265:                 # converted to a unicode with the input codec of the charset.
266:                 incodec = charset.input_codec or 'us-ascii'
267:                 ustr = unicode(s, incodec, errors)
268:                 # Now make sure that the unicode could be converted back to a
269:                 # byte string with the output codec, which may be different
270:                 # than the iput coded.  Still, use the original byte string.
271:                 outcodec = charset.output_codec or 'us-ascii'
272:                 ustr.encode(outcodec, errors)
273:             elif isinstance(s, unicode):
274:                 # Now we have to be sure the unicode string can be converted
275:                 # to a byte string with a reasonable output codec.  We want to
276:                 # use the byte string in the chunk.
277:                 for charset in USASCII, charset, UTF8:
278:                     try:
279:                         outcodec = charset.output_codec or 'us-ascii'
280:                         s = s.encode(outcodec, errors)
281:                         break
282:                     except UnicodeError:
283:                         pass
284:                 else:
285:                     assert False, 'utf-8 conversion failed'
286:         self._chunks.append((s, charset))
287: 
288:     def _split(self, s, charset, maxlinelen, splitchars):
289:         # Split up a header safely for use with encode_chunks.
290:         splittable = charset.to_splittable(s)
291:         encoded = charset.from_splittable(splittable, True)
292:         elen = charset.encoded_header_len(encoded)
293:         # If the line's encoded length first, just return it
294:         if elen <= maxlinelen:
295:             return [(encoded, charset)]
296:         # If we have undetermined raw 8bit characters sitting in a byte
297:         # string, we really don't know what the right thing to do is.  We
298:         # can't really split it because it might be multibyte data which we
299:         # could break if we split it between pairs.  The least harm seems to
300:         # be to not split the header at all, but that means they could go out
301:         # longer than maxlinelen.
302:         if charset == '8bit':
303:             return [(s, charset)]
304:         # BAW: I'm not sure what the right test here is.  What we're trying to
305:         # do is be faithful to RFC 2822's recommendation that ($2.2.3):
306:         #
307:         # "Note: Though structured field bodies are defined in such a way that
308:         #  folding can take place between many of the lexical tokens (and even
309:         #  within some of the lexical tokens), folding SHOULD be limited to
310:         #  placing the CRLF at higher-level syntactic breaks."
311:         #
312:         # For now, I can only imagine doing this when the charset is us-ascii,
313:         # although it's possible that other charsets may also benefit from the
314:         # higher-level syntactic breaks.
315:         elif charset == 'us-ascii':
316:             return self._split_ascii(s, charset, maxlinelen, splitchars)
317:         # BAW: should we use encoded?
318:         elif elen == len(s):
319:             # We can split on _maxlinelen boundaries because we know that the
320:             # encoding won't change the size of the string
321:             splitpnt = maxlinelen
322:             first = charset.from_splittable(splittable[:splitpnt], False)
323:             last = charset.from_splittable(splittable[splitpnt:], False)
324:         else:
325:             # Binary search for split point
326:             first, last = _binsplit(splittable, charset, maxlinelen)
327:         # first is of the proper length so just wrap it in the appropriate
328:         # chrome.  last must be recursively split.
329:         fsplittable = charset.to_splittable(first)
330:         fencoded = charset.from_splittable(fsplittable, True)
331:         chunk = [(fencoded, charset)]
332:         return chunk + self._split(last, charset, self._maxlinelen, splitchars)
333: 
334:     def _split_ascii(self, s, charset, firstlen, splitchars):
335:         chunks = _split_ascii(s, firstlen, self._maxlinelen,
336:                               self._continuation_ws, splitchars)
337:         return zip(chunks, [charset]*len(chunks))
338: 
339:     def _encode_chunks(self, newchunks, maxlinelen):
340:         # MIME-encode a header with many different charsets and/or encodings.
341:         #
342:         # Given a list of pairs (string, charset), return a MIME-encoded
343:         # string suitable for use in a header field.  Each pair may have
344:         # different charsets and/or encodings, and the resulting header will
345:         # accurately reflect each setting.
346:         #
347:         # Each encoding can be email.utils.QP (quoted-printable, for
348:         # ASCII-like character sets like iso-8859-1), email.utils.BASE64
349:         # (Base64, for non-ASCII like character sets like KOI8-R and
350:         # iso-2022-jp), or None (no encoding).
351:         #
352:         # Each pair will be represented on a separate line; the resulting
353:         # string will be in the format:
354:         #
355:         # =?charset1?q?Mar=EDa_Gonz=E1lez_Alonso?=\n
356:         #  =?charset2?b?SvxyZ2VuIEL2aW5n?="
357:         chunks = []
358:         for header, charset in newchunks:
359:             if not header:
360:                 continue
361:             if charset is None or charset.header_encoding is None:
362:                 s = header
363:             else:
364:                 s = charset.header_encode(header)
365:             # Don't add more folding whitespace than necessary
366:             if chunks and chunks[-1].endswith(' '):
367:                 extra = ''
368:             else:
369:                 extra = ' '
370:             _max_append(chunks, s, maxlinelen, extra)
371:         joiner = NL + self._continuation_ws
372:         return joiner.join(chunks)
373: 
374:     def encode(self, splitchars=';, '):
375:         '''Encode a message header into an RFC-compliant format.
376: 
377:         There are many issues involved in converting a given string for use in
378:         an email header.  Only certain character sets are readable in most
379:         email clients, and as header strings can only contain a subset of
380:         7-bit ASCII, care must be taken to properly convert and encode (with
381:         Base64 or quoted-printable) header strings.  In addition, there is a
382:         75-character length limit on any given encoded header field, so
383:         line-wrapping must be performed, even with double-byte character sets.
384: 
385:         This method will do its best to convert the string to the correct
386:         character set used in email, and encode and line wrap it safely with
387:         the appropriate scheme for that character set.
388: 
389:         If the given charset is not known or an error occurs during
390:         conversion, this function will return the header untouched.
391: 
392:         Optional splitchars is a string containing characters to split long
393:         ASCII lines on, in rough support of RFC 2822's `highest level
394:         syntactic breaks'.  This doesn't affect RFC 2047 encoded lines.
395:         '''
396:         newchunks = []
397:         maxlinelen = self._firstlinelen
398:         lastlen = 0
399:         for s, charset in self._chunks:
400:             # The first bit of the next chunk should be just long enough to
401:             # fill the next line.  Don't forget the space separating the
402:             # encoded words.
403:             targetlen = maxlinelen - lastlen - 1
404:             if targetlen < charset.encoded_header_len(''):
405:                 # Stick it on the next line
406:                 targetlen = maxlinelen
407:             newchunks += self._split(s, charset, targetlen, splitchars)
408:             lastchunk, lastcharset = newchunks[-1]
409:             lastlen = lastcharset.encoded_header_len(lastchunk)
410:         value = self._encode_chunks(newchunks, maxlinelen)
411:         if _embeded_header.search(value):
412:             raise HeaderParseError("header value appears to contain "
413:                 "an embedded header: {!r}".format(value))
414:         return value
415: 
416: 
417: 
418: def _split_ascii(s, firstlen, restlen, continuation_ws, splitchars):
419:     lines = []
420:     maxlen = firstlen
421:     for line in s.splitlines():
422:         # Ignore any leading whitespace (i.e. continuation whitespace) already
423:         # on the line, since we'll be adding our own.
424:         line = line.lstrip()
425:         if len(line) < maxlen:
426:             lines.append(line)
427:             maxlen = restlen
428:             continue
429:         # Attempt to split the line at the highest-level syntactic break
430:         # possible.  Note that we don't have a lot of smarts about field
431:         # syntax; we just try to break on semi-colons, then commas, then
432:         # whitespace.
433:         for ch in splitchars:
434:             if ch in line:
435:                 break
436:         else:
437:             # There's nothing useful to split the line on, not even spaces, so
438:             # just append this line unchanged
439:             lines.append(line)
440:             maxlen = restlen
441:             continue
442:         # Now split the line on the character plus trailing whitespace
443:         cre = re.compile(r'%s\s*' % ch)
444:         if ch in ';,':
445:             eol = ch
446:         else:
447:             eol = ''
448:         joiner = eol + ' '
449:         joinlen = len(joiner)
450:         wslen = len(continuation_ws.replace('\t', SPACE8))
451:         this = []
452:         linelen = 0
453:         for part in cre.split(line):
454:             curlen = linelen + max(0, len(this)-1) * joinlen
455:             partlen = len(part)
456:             onfirstline = not lines
457:             # We don't want to split after the field name, if we're on the
458:             # first line and the field name is present in the header string.
459:             if ch == ' ' and onfirstline and \
460:                    len(this) == 1 and fcre.match(this[0]):
461:                 this.append(part)
462:                 linelen += partlen
463:             elif curlen + partlen > maxlen:
464:                 if this:
465:                     lines.append(joiner.join(this) + eol)
466:                 # If this part is longer than maxlen and we aren't already
467:                 # splitting on whitespace, try to recursively split this line
468:                 # on whitespace.
469:                 if partlen > maxlen and ch != ' ':
470:                     subl = _split_ascii(part, maxlen, restlen,
471:                                         continuation_ws, ' ')
472:                     lines.extend(subl[:-1])
473:                     this = [subl[-1]]
474:                 else:
475:                     this = [part]
476:                 linelen = wslen + len(this[-1])
477:                 maxlen = restlen
478:             else:
479:                 this.append(part)
480:                 linelen += partlen
481:         # Put any left over parts on a line by themselves
482:         if this:
483:             lines.append(joiner.join(this))
484:     return lines
485: 
486: 
487: 
488: def _binsplit(splittable, charset, maxlinelen):
489:     i = 0
490:     j = len(splittable)
491:     while i < j:
492:         # Invariants:
493:         # 1. splittable[:k] fits for all k <= i (note that we *assume*,
494:         #    at the start, that splittable[:0] fits).
495:         # 2. splittable[:k] does not fit for any k > j (at the start,
496:         #    this means we shouldn't look at any k > len(splittable)).
497:         # 3. We don't know about splittable[:k] for k in i+1..j.
498:         # 4. We want to set i to the largest k that fits, with i <= k <= j.
499:         #
500:         m = (i+j+1) >> 1  # ceiling((i+j)/2); i < m <= j
501:         chunk = charset.from_splittable(splittable[:m], True)
502:         chunklen = charset.encoded_header_len(chunk)
503:         if chunklen <= maxlinelen:
504:             # m is acceptable, so is a new lower bound.
505:             i = m
506:         else:
507:             # m is not acceptable, so final i must be < m.
508:             j = m - 1
509:     # i == j.  Invariant #1 implies that splittable[:i] fits, and
510:     # invariant #2 implies that splittable[:i+1] does not fit, so i
511:     # is what we're looking for.
512:     first = charset.from_splittable(splittable[:i], False)
513:     last  = charset.from_splittable(splittable[i:], False)
514:     return first, last
515: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_14851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Header encoding and decoding functionality.')

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['Header', 'decode_header', 'make_header']
module_type_store.set_exportable_members(['Header', 'decode_header', 'make_header'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_14852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_14853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'str', 'Header')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14852, str_14853)
# Adding element type (line 7)
str_14854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'str', 'decode_header')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14852, str_14854)
# Adding element type (line 7)
str_14855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'str', 'make_header')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14852, str_14855)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_14852)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import re' statement (line 13)
import re

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import binascii' statement (line 14)
import binascii

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'binascii', binascii, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import email.quoprimime' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_14856 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'email.quoprimime')

if (type(import_14856) is not StypyTypeError):

    if (import_14856 != 'pyd_module'):
        __import__(import_14856)
        sys_modules_14857 = sys.modules[import_14856]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'email.quoprimime', sys_modules_14857.module_type_store, module_type_store)
    else:
        import email.quoprimime

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'email.quoprimime', email.quoprimime, module_type_store)

else:
    # Assigning a type to the variable 'email.quoprimime' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'email.quoprimime', import_14856)

remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import email.base64mime' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_14858 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'email.base64mime')

if (type(import_14858) is not StypyTypeError):

    if (import_14858 != 'pyd_module'):
        __import__(import_14858)
        sys_modules_14859 = sys.modules[import_14858]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'email.base64mime', sys_modules_14859.module_type_store, module_type_store)
    else:
        import email.base64mime

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'email.base64mime', email.base64mime, module_type_store)

else:
    # Assigning a type to the variable 'email.base64mime' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'email.base64mime', import_14858)

remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from email.errors import HeaderParseError' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_14860 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'email.errors')

if (type(import_14860) is not StypyTypeError):

    if (import_14860 != 'pyd_module'):
        __import__(import_14860)
        sys_modules_14861 = sys.modules[import_14860]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'email.errors', sys_modules_14861.module_type_store, module_type_store, ['HeaderParseError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_14861, sys_modules_14861.module_type_store, module_type_store)
    else:
        from email.errors import HeaderParseError

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'email.errors', None, module_type_store, ['HeaderParseError'], [HeaderParseError])

else:
    # Assigning a type to the variable 'email.errors' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'email.errors', import_14860)

remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from email.charset import Charset' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_14862 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'email.charset')

if (type(import_14862) is not StypyTypeError):

    if (import_14862 != 'pyd_module'):
        __import__(import_14862)
        sys_modules_14863 = sys.modules[import_14862]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'email.charset', sys_modules_14863.module_type_store, module_type_store, ['Charset'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_14863, sys_modules_14863.module_type_store, module_type_store)
    else:
        from email.charset import Charset

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'email.charset', None, module_type_store, ['Charset'], [Charset])

else:
    # Assigning a type to the variable 'email.charset' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'email.charset', import_14862)

remove_current_file_folder_from_path('C:/Python27/lib/email/')


# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
str_14864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 5), 'str', '\n')
# Assigning a type to the variable 'NL' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'NL', str_14864)

# Assigning a Str to a Name (line 23):

# Assigning a Str to a Name (line 23):
str_14865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'str', ' ')
# Assigning a type to the variable 'SPACE' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'SPACE', str_14865)

# Assigning a Str to a Name (line 24):

# Assigning a Str to a Name (line 24):
unicode_14866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'unicode', u' ')
# Assigning a type to the variable 'USPACE' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'USPACE', unicode_14866)

# Assigning a BinOp to a Name (line 25):

# Assigning a BinOp to a Name (line 25):
str_14867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'str', ' ')
int_14868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'int')
# Applying the binary operator '*' (line 25)
result_mul_14869 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 9), '*', str_14867, int_14868)

# Assigning a type to the variable 'SPACE8' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'SPACE8', result_mul_14869)

# Assigning a Str to a Name (line 26):

# Assigning a Str to a Name (line 26):
unicode_14870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'unicode', u'')
# Assigning a type to the variable 'UEMPTYSTRING' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'UEMPTYSTRING', unicode_14870)

# Assigning a Num to a Name (line 28):

# Assigning a Num to a Name (line 28):
int_14871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 13), 'int')
# Assigning a type to the variable 'MAXLINELEN' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'MAXLINELEN', int_14871)

# Assigning a Call to a Name (line 30):

# Assigning a Call to a Name (line 30):

# Call to Charset(...): (line 30)
# Processing the call arguments (line 30)
str_14873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'str', 'us-ascii')
# Processing the call keyword arguments (line 30)
kwargs_14874 = {}
# Getting the type of 'Charset' (line 30)
Charset_14872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 10), 'Charset', False)
# Calling Charset(args, kwargs) (line 30)
Charset_call_result_14875 = invoke(stypy.reporting.localization.Localization(__file__, 30, 10), Charset_14872, *[str_14873], **kwargs_14874)

# Assigning a type to the variable 'USASCII' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'USASCII', Charset_call_result_14875)

# Assigning a Call to a Name (line 31):

# Assigning a Call to a Name (line 31):

# Call to Charset(...): (line 31)
# Processing the call arguments (line 31)
str_14877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 15), 'str', 'utf-8')
# Processing the call keyword arguments (line 31)
kwargs_14878 = {}
# Getting the type of 'Charset' (line 31)
Charset_14876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'Charset', False)
# Calling Charset(args, kwargs) (line 31)
Charset_call_result_14879 = invoke(stypy.reporting.localization.Localization(__file__, 31, 7), Charset_14876, *[str_14877], **kwargs_14878)

# Assigning a type to the variable 'UTF8' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'UTF8', Charset_call_result_14879)

# Assigning a Call to a Name (line 34):

# Assigning a Call to a Name (line 34):

# Call to compile(...): (line 34)
# Processing the call arguments (line 34)
str_14882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', '\n  =\\?                   # literal =?\n  (?P<charset>[^?]*?)   # non-greedy up to the next ? is the charset\n  \\?                    # literal ?\n  (?P<encoding>[qb])    # either a "q" or a "b", case insensitive\n  \\?                    # literal ?\n  (?P<encoded>.*?)      # non-greedy up to the next ?= is the encoded string\n  \\?=                   # literal ?=\n  (?=[ \\t]|$)           # whitespace or the end of the string\n  ')
# Getting the type of 're' (line 43)
re_14883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 're', False)
# Obtaining the member 'VERBOSE' of a type (line 43)
VERBOSE_14884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 7), re_14883, 'VERBOSE')
# Getting the type of 're' (line 43)
re_14885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 're', False)
# Obtaining the member 'IGNORECASE' of a type (line 43)
IGNORECASE_14886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 20), re_14885, 'IGNORECASE')
# Applying the binary operator '|' (line 43)
result_or__14887 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), '|', VERBOSE_14884, IGNORECASE_14886)

# Getting the type of 're' (line 43)
re_14888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 're', False)
# Obtaining the member 'MULTILINE' of a type (line 43)
MULTILINE_14889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 36), re_14888, 'MULTILINE')
# Applying the binary operator '|' (line 43)
result_or__14890 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 34), '|', result_or__14887, MULTILINE_14889)

# Processing the call keyword arguments (line 34)
kwargs_14891 = {}
# Getting the type of 're' (line 34)
re_14880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 7), 're', False)
# Obtaining the member 'compile' of a type (line 34)
compile_14881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 7), re_14880, 'compile')
# Calling compile(args, kwargs) (line 34)
compile_call_result_14892 = invoke(stypy.reporting.localization.Localization(__file__, 34, 7), compile_14881, *[str_14882, result_or__14890], **kwargs_14891)

# Assigning a type to the variable 'ecre' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'ecre', compile_call_result_14892)

# Assigning a Call to a Name (line 48):

# Assigning a Call to a Name (line 48):

# Call to compile(...): (line 48)
# Processing the call arguments (line 48)
str_14895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 18), 'str', '[\\041-\\176]+:$')
# Processing the call keyword arguments (line 48)
kwargs_14896 = {}
# Getting the type of 're' (line 48)
re_14893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 7), 're', False)
# Obtaining the member 'compile' of a type (line 48)
compile_14894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 7), re_14893, 'compile')
# Calling compile(args, kwargs) (line 48)
compile_call_result_14897 = invoke(stypy.reporting.localization.Localization(__file__, 48, 7), compile_14894, *[str_14895], **kwargs_14896)

# Assigning a type to the variable 'fcre' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'fcre', compile_call_result_14897)

# Assigning a Call to a Name (line 52):

# Assigning a Call to a Name (line 52):

# Call to compile(...): (line 52)
# Processing the call arguments (line 52)
str_14900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'str', '\\n[^ \\t]+:')
# Processing the call keyword arguments (line 52)
kwargs_14901 = {}
# Getting the type of 're' (line 52)
re_14898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 're', False)
# Obtaining the member 'compile' of a type (line 52)
compile_14899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 18), re_14898, 'compile')
# Calling compile(args, kwargs) (line 52)
compile_call_result_14902 = invoke(stypy.reporting.localization.Localization(__file__, 52, 18), compile_14899, *[str_14900], **kwargs_14901)

# Assigning a type to the variable '_embeded_header' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), '_embeded_header', compile_call_result_14902)

# Assigning a Attribute to a Name (line 57):

# Assigning a Attribute to a Name (line 57):
# Getting the type of 'email' (line 57)
email_14903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'email')
# Obtaining the member 'quoprimime' of a type (line 57)
quoprimime_14904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 14), email_14903, 'quoprimime')
# Obtaining the member '_max_append' of a type (line 57)
_max_append_14905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 14), quoprimime_14904, '_max_append')
# Assigning a type to the variable '_max_append' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), '_max_append', _max_append_14905)

@norecursion
def decode_header(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'decode_header'
    module_type_store = module_type_store.open_function_context('decode_header', 61, 0, False)
    
    # Passed parameters checking function
    decode_header.stypy_localization = localization
    decode_header.stypy_type_of_self = None
    decode_header.stypy_type_store = module_type_store
    decode_header.stypy_function_name = 'decode_header'
    decode_header.stypy_param_names_list = ['header']
    decode_header.stypy_varargs_param_name = None
    decode_header.stypy_kwargs_param_name = None
    decode_header.stypy_call_defaults = defaults
    decode_header.stypy_call_varargs = varargs
    decode_header.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decode_header', ['header'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decode_header', localization, ['header'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decode_header(...)' code ##################

    str_14906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', 'Decode a message header value without converting charset.\n\n    Returns a list of (decoded_string, charset) pairs containing each of the\n    decoded parts of the header.  Charset is None for non-encoded parts of the\n    header, otherwise a lower-case string containing the name of the character\n    set specified in the encoded string.\n\n    An email.errors.HeaderParseError may be raised when certain decoding error\n    occurs (e.g. a base64 decoding exception).\n    ')
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to str(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'header' (line 73)
    header_14908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'header', False)
    # Processing the call keyword arguments (line 73)
    kwargs_14909 = {}
    # Getting the type of 'str' (line 73)
    str_14907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'str', False)
    # Calling str(args, kwargs) (line 73)
    str_call_result_14910 = invoke(stypy.reporting.localization.Localization(__file__, 73, 13), str_14907, *[header_14908], **kwargs_14909)
    
    # Assigning a type to the variable 'header' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'header', str_call_result_14910)
    
    
    # Call to search(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'header' (line 74)
    header_14913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'header', False)
    # Processing the call keyword arguments (line 74)
    kwargs_14914 = {}
    # Getting the type of 'ecre' (line 74)
    ecre_14911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'ecre', False)
    # Obtaining the member 'search' of a type (line 74)
    search_14912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 11), ecre_14911, 'search')
    # Calling search(args, kwargs) (line 74)
    search_call_result_14915 = invoke(stypy.reporting.localization.Localization(__file__, 74, 11), search_14912, *[header_14913], **kwargs_14914)
    
    # Applying the 'not' unary operator (line 74)
    result_not__14916 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 7), 'not', search_call_result_14915)
    
    # Testing if the type of an if condition is none (line 74)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 4), result_not__14916):
        pass
    else:
        
        # Testing the type of an if condition (line 74)
        if_condition_14917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 4), result_not__14916)
        # Assigning a type to the variable 'if_condition_14917' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'if_condition_14917', if_condition_14917)
        # SSA begins for if statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_14918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_14919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        # Getting the type of 'header' (line 75)
        header_14920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'header')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), tuple_14919, header_14920)
        # Adding element type (line 75)
        # Getting the type of 'None' (line 75)
        None_14921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), tuple_14919, None_14921)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 15), list_14918, tuple_14919)
        
        # Assigning a type to the variable 'stypy_return_type' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type', list_14918)
        # SSA join for if statement (line 74)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 76):
    
    # Assigning a List to a Name (line 76):
    
    # Obtaining an instance of the builtin type 'list' (line 76)
    list_14922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 76)
    
    # Assigning a type to the variable 'decoded' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'decoded', list_14922)
    
    # Assigning a Str to a Name (line 77):
    
    # Assigning a Str to a Name (line 77):
    str_14923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 10), 'str', '')
    # Assigning a type to the variable 'dec' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'dec', str_14923)
    
    
    # Call to splitlines(...): (line 78)
    # Processing the call keyword arguments (line 78)
    kwargs_14926 = {}
    # Getting the type of 'header' (line 78)
    header_14924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'header', False)
    # Obtaining the member 'splitlines' of a type (line 78)
    splitlines_14925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), header_14924, 'splitlines')
    # Calling splitlines(args, kwargs) (line 78)
    splitlines_call_result_14927 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), splitlines_14925, *[], **kwargs_14926)
    
    # Assigning a type to the variable 'splitlines_call_result_14927' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'splitlines_call_result_14927', splitlines_call_result_14927)
    # Testing if the for loop is going to be iterated (line 78)
    # Testing the type of a for loop iterable (line 78)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 4), splitlines_call_result_14927)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 78, 4), splitlines_call_result_14927):
        # Getting the type of the for loop variable (line 78)
        for_loop_var_14928 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 4), splitlines_call_result_14927)
        # Assigning a type to the variable 'line' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'line', for_loop_var_14928)
        # SSA begins for a for statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to search(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'line' (line 80)
        line_14931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'line', False)
        # Processing the call keyword arguments (line 80)
        kwargs_14932 = {}
        # Getting the type of 'ecre' (line 80)
        ecre_14929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'ecre', False)
        # Obtaining the member 'search' of a type (line 80)
        search_14930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), ecre_14929, 'search')
        # Calling search(args, kwargs) (line 80)
        search_call_result_14933 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), search_14930, *[line_14931], **kwargs_14932)
        
        # Applying the 'not' unary operator (line 80)
        result_not__14934 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), 'not', search_call_result_14933)
        
        # Testing if the type of an if condition is none (line 80)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 8), result_not__14934):
            pass
        else:
            
            # Testing the type of an if condition (line 80)
            if_condition_14935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), result_not__14934)
            # Assigning a type to the variable 'if_condition_14935' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_14935', if_condition_14935)
            # SSA begins for if statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 81)
            # Processing the call arguments (line 81)
            
            # Obtaining an instance of the builtin type 'tuple' (line 81)
            tuple_14938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 81)
            # Adding element type (line 81)
            # Getting the type of 'line' (line 81)
            line_14939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'line', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 28), tuple_14938, line_14939)
            # Adding element type (line 81)
            # Getting the type of 'None' (line 81)
            None_14940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 34), 'None', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 28), tuple_14938, None_14940)
            
            # Processing the call keyword arguments (line 81)
            kwargs_14941 = {}
            # Getting the type of 'decoded' (line 81)
            decoded_14936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'decoded', False)
            # Obtaining the member 'append' of a type (line 81)
            append_14937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), decoded_14936, 'append')
            # Calling append(args, kwargs) (line 81)
            append_call_result_14942 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), append_14937, *[tuple_14938], **kwargs_14941)
            
            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to split(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'line' (line 83)
        line_14945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'line', False)
        # Processing the call keyword arguments (line 83)
        kwargs_14946 = {}
        # Getting the type of 'ecre' (line 83)
        ecre_14943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'ecre', False)
        # Obtaining the member 'split' of a type (line 83)
        split_14944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), ecre_14943, 'split')
        # Calling split(args, kwargs) (line 83)
        split_call_result_14947 = invoke(stypy.reporting.localization.Localization(__file__, 83, 16), split_14944, *[line_14945], **kwargs_14946)
        
        # Assigning a type to the variable 'parts' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'parts', split_call_result_14947)
        
        # Getting the type of 'parts' (line 84)
        parts_14948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'parts')
        # Assigning a type to the variable 'parts_14948' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'parts_14948', parts_14948)
        # Testing if the while is going to be iterated (line 84)
        # Testing the type of an if condition (line 84)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 8), parts_14948)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 8), parts_14948):
            # SSA begins for while statement (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 85):
            
            # Assigning a Call to a Name (line 85):
            
            # Call to strip(...): (line 85)
            # Processing the call keyword arguments (line 85)
            kwargs_14955 = {}
            
            # Call to pop(...): (line 85)
            # Processing the call arguments (line 85)
            int_14951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'int')
            # Processing the call keyword arguments (line 85)
            kwargs_14952 = {}
            # Getting the type of 'parts' (line 85)
            parts_14949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'parts', False)
            # Obtaining the member 'pop' of a type (line 85)
            pop_14950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), parts_14949, 'pop')
            # Calling pop(args, kwargs) (line 85)
            pop_call_result_14953 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), pop_14950, *[int_14951], **kwargs_14952)
            
            # Obtaining the member 'strip' of a type (line 85)
            strip_14954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), pop_call_result_14953, 'strip')
            # Calling strip(args, kwargs) (line 85)
            strip_call_result_14956 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), strip_14954, *[], **kwargs_14955)
            
            # Assigning a type to the variable 'unenc' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'unenc', strip_call_result_14956)
            # Getting the type of 'unenc' (line 86)
            unenc_14957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'unenc')
            # Testing if the type of an if condition is none (line 86)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 12), unenc_14957):
                pass
            else:
                
                # Testing the type of an if condition (line 86)
                if_condition_14958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 12), unenc_14957)
                # Assigning a type to the variable 'if_condition_14958' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'if_condition_14958', if_condition_14958)
                # SSA begins for if statement (line 86)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Evaluating a boolean operation
                # Getting the type of 'decoded' (line 88)
                decoded_14959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'decoded')
                
                
                # Obtaining the type of the subscript
                int_14960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 43), 'int')
                
                # Obtaining the type of the subscript
                int_14961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 39), 'int')
                # Getting the type of 'decoded' (line 88)
                decoded_14962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 31), 'decoded')
                # Obtaining the member '__getitem__' of a type (line 88)
                getitem___14963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 31), decoded_14962, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 88)
                subscript_call_result_14964 = invoke(stypy.reporting.localization.Localization(__file__, 88, 31), getitem___14963, int_14961)
                
                # Obtaining the member '__getitem__' of a type (line 88)
                getitem___14965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 31), subscript_call_result_14964, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 88)
                subscript_call_result_14966 = invoke(stypy.reporting.localization.Localization(__file__, 88, 31), getitem___14965, int_14960)
                
                # Getting the type of 'None' (line 88)
                None_14967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 49), 'None')
                # Applying the binary operator 'is' (line 88)
                result_is__14968 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 31), 'is', subscript_call_result_14966, None_14967)
                
                # Applying the binary operator 'and' (line 88)
                result_and_keyword_14969 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 19), 'and', decoded_14959, result_is__14968)
                
                # Testing if the type of an if condition is none (line 88)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 16), result_and_keyword_14969):
                    
                    # Call to append(...): (line 91)
                    # Processing the call arguments (line 91)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 91)
                    tuple_14988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 36), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 91)
                    # Adding element type (line 91)
                    # Getting the type of 'unenc' (line 91)
                    unenc_14989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 36), 'unenc', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 36), tuple_14988, unenc_14989)
                    # Adding element type (line 91)
                    # Getting the type of 'None' (line 91)
                    None_14990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'None', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 36), tuple_14988, None_14990)
                    
                    # Processing the call keyword arguments (line 91)
                    kwargs_14991 = {}
                    # Getting the type of 'decoded' (line 91)
                    decoded_14986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'decoded', False)
                    # Obtaining the member 'append' of a type (line 91)
                    append_14987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), decoded_14986, 'append')
                    # Calling append(args, kwargs) (line 91)
                    append_call_result_14992 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), append_14987, *[tuple_14988], **kwargs_14991)
                    
                else:
                    
                    # Testing the type of an if condition (line 88)
                    if_condition_14970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 16), result_and_keyword_14969)
                    # Assigning a type to the variable 'if_condition_14970' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'if_condition_14970', if_condition_14970)
                    # SSA begins for if statement (line 88)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Tuple to a Subscript (line 89):
                    
                    # Assigning a Tuple to a Subscript (line 89):
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 89)
                    tuple_14971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 35), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 89)
                    # Adding element type (line 89)
                    
                    # Obtaining the type of the subscript
                    int_14972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 47), 'int')
                    
                    # Obtaining the type of the subscript
                    int_14973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 43), 'int')
                    # Getting the type of 'decoded' (line 89)
                    decoded_14974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'decoded')
                    # Obtaining the member '__getitem__' of a type (line 89)
                    getitem___14975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 35), decoded_14974, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
                    subscript_call_result_14976 = invoke(stypy.reporting.localization.Localization(__file__, 89, 35), getitem___14975, int_14973)
                    
                    # Obtaining the member '__getitem__' of a type (line 89)
                    getitem___14977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 35), subscript_call_result_14976, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
                    subscript_call_result_14978 = invoke(stypy.reporting.localization.Localization(__file__, 89, 35), getitem___14977, int_14972)
                    
                    # Getting the type of 'SPACE' (line 89)
                    SPACE_14979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 52), 'SPACE')
                    # Applying the binary operator '+' (line 89)
                    result_add_14980 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 35), '+', subscript_call_result_14978, SPACE_14979)
                    
                    # Getting the type of 'unenc' (line 89)
                    unenc_14981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 60), 'unenc')
                    # Applying the binary operator '+' (line 89)
                    result_add_14982 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 58), '+', result_add_14980, unenc_14981)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 35), tuple_14971, result_add_14982)
                    # Adding element type (line 89)
                    # Getting the type of 'None' (line 89)
                    None_14983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 67), 'None')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 35), tuple_14971, None_14983)
                    
                    # Getting the type of 'decoded' (line 89)
                    decoded_14984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'decoded')
                    int_14985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 28), 'int')
                    # Storing an element on a container (line 89)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 20), decoded_14984, (int_14985, tuple_14971))
                    # SSA branch for the else part of an if statement (line 88)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to append(...): (line 91)
                    # Processing the call arguments (line 91)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 91)
                    tuple_14988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 36), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 91)
                    # Adding element type (line 91)
                    # Getting the type of 'unenc' (line 91)
                    unenc_14989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 36), 'unenc', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 36), tuple_14988, unenc_14989)
                    # Adding element type (line 91)
                    # Getting the type of 'None' (line 91)
                    None_14990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'None', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 36), tuple_14988, None_14990)
                    
                    # Processing the call keyword arguments (line 91)
                    kwargs_14991 = {}
                    # Getting the type of 'decoded' (line 91)
                    decoded_14986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'decoded', False)
                    # Obtaining the member 'append' of a type (line 91)
                    append_14987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), decoded_14986, 'append')
                    # Calling append(args, kwargs) (line 91)
                    append_call_result_14992 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), append_14987, *[tuple_14988], **kwargs_14991)
                    
                    # SSA join for if statement (line 88)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 86)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'parts' (line 92)
            parts_14993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'parts')
            # Testing if the type of an if condition is none (line 92)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 92, 12), parts_14993):
                pass
            else:
                
                # Testing the type of an if condition (line 92)
                if_condition_14994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 12), parts_14993)
                # Assigning a type to the variable 'if_condition_14994' (line 92)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'if_condition_14994', if_condition_14994)
                # SSA begins for if statement (line 92)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a ListComp to a Tuple (line 93):
                
                # Assigning a Subscript to a Name (line 93):
                
                # Obtaining the type of the subscript
                int_14995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 16), 'int')
                # Calculating list comprehension
                # Calculating comprehension expression
                
                # Obtaining the type of the subscript
                int_15000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 62), 'int')
                int_15001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 64), 'int')
                slice_15002 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 56), int_15000, int_15001, None)
                # Getting the type of 'parts' (line 93)
                parts_15003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 56), 'parts')
                # Obtaining the member '__getitem__' of a type (line 93)
                getitem___15004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 56), parts_15003, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 93)
                subscript_call_result_15005 = invoke(stypy.reporting.localization.Localization(__file__, 93, 56), getitem___15004, slice_15002)
                
                comprehension_15006 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 37), subscript_call_result_15005)
                # Assigning a type to the variable 's' (line 93)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 's', comprehension_15006)
                
                # Call to lower(...): (line 93)
                # Processing the call keyword arguments (line 93)
                kwargs_14998 = {}
                # Getting the type of 's' (line 93)
                s_14996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 's', False)
                # Obtaining the member 'lower' of a type (line 93)
                lower_14997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 37), s_14996, 'lower')
                # Calling lower(args, kwargs) (line 93)
                lower_call_result_14999 = invoke(stypy.reporting.localization.Localization(__file__, 93, 37), lower_14997, *[], **kwargs_14998)
                
                list_15007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 37), 'list')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 37), list_15007, lower_call_result_14999)
                # Obtaining the member '__getitem__' of a type (line 93)
                getitem___15008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 16), list_15007, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 93)
                subscript_call_result_15009 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), getitem___15008, int_14995)
                
                # Assigning a type to the variable 'tuple_var_assignment_14844' (line 93)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'tuple_var_assignment_14844', subscript_call_result_15009)
                
                # Assigning a Subscript to a Name (line 93):
                
                # Obtaining the type of the subscript
                int_15010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 16), 'int')
                # Calculating list comprehension
                # Calculating comprehension expression
                
                # Obtaining the type of the subscript
                int_15015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 62), 'int')
                int_15016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 64), 'int')
                slice_15017 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 93, 56), int_15015, int_15016, None)
                # Getting the type of 'parts' (line 93)
                parts_15018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 56), 'parts')
                # Obtaining the member '__getitem__' of a type (line 93)
                getitem___15019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 56), parts_15018, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 93)
                subscript_call_result_15020 = invoke(stypy.reporting.localization.Localization(__file__, 93, 56), getitem___15019, slice_15017)
                
                comprehension_15021 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 37), subscript_call_result_15020)
                # Assigning a type to the variable 's' (line 93)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 's', comprehension_15021)
                
                # Call to lower(...): (line 93)
                # Processing the call keyword arguments (line 93)
                kwargs_15013 = {}
                # Getting the type of 's' (line 93)
                s_15011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 's', False)
                # Obtaining the member 'lower' of a type (line 93)
                lower_15012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 37), s_15011, 'lower')
                # Calling lower(args, kwargs) (line 93)
                lower_call_result_15014 = invoke(stypy.reporting.localization.Localization(__file__, 93, 37), lower_15012, *[], **kwargs_15013)
                
                list_15022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 37), 'list')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 37), list_15022, lower_call_result_15014)
                # Obtaining the member '__getitem__' of a type (line 93)
                getitem___15023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 16), list_15022, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 93)
                subscript_call_result_15024 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), getitem___15023, int_15010)
                
                # Assigning a type to the variable 'tuple_var_assignment_14845' (line 93)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'tuple_var_assignment_14845', subscript_call_result_15024)
                
                # Assigning a Name to a Name (line 93):
                # Getting the type of 'tuple_var_assignment_14844' (line 93)
                tuple_var_assignment_14844_15025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'tuple_var_assignment_14844')
                # Assigning a type to the variable 'charset' (line 93)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'charset', tuple_var_assignment_14844_15025)
                
                # Assigning a Name to a Name (line 93):
                # Getting the type of 'tuple_var_assignment_14845' (line 93)
                tuple_var_assignment_14845_15026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'tuple_var_assignment_14845')
                # Assigning a type to the variable 'encoding' (line 93)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'encoding', tuple_var_assignment_14845_15026)
                
                # Assigning a Subscript to a Name (line 94):
                
                # Assigning a Subscript to a Name (line 94):
                
                # Obtaining the type of the subscript
                int_15027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 32), 'int')
                # Getting the type of 'parts' (line 94)
                parts_15028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'parts')
                # Obtaining the member '__getitem__' of a type (line 94)
                getitem___15029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 26), parts_15028, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 94)
                subscript_call_result_15030 = invoke(stypy.reporting.localization.Localization(__file__, 94, 26), getitem___15029, int_15027)
                
                # Assigning a type to the variable 'encoded' (line 94)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'encoded', subscript_call_result_15030)
                
                # Assigning a Name to a Name (line 95):
                
                # Assigning a Name to a Name (line 95):
                # Getting the type of 'None' (line 95)
                None_15031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'None')
                # Assigning a type to the variable 'dec' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'dec', None_15031)
                
                # Getting the type of 'encoding' (line 96)
                encoding_15032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'encoding')
                str_15033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 31), 'str', 'q')
                # Applying the binary operator '==' (line 96)
                result_eq_15034 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 19), '==', encoding_15032, str_15033)
                
                # Testing if the type of an if condition is none (line 96)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 16), result_eq_15034):
                    
                    # Getting the type of 'encoding' (line 98)
                    encoding_15042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'encoding')
                    str_15043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 33), 'str', 'b')
                    # Applying the binary operator '==' (line 98)
                    result_eq_15044 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 21), '==', encoding_15042, str_15043)
                    
                    # Testing if the type of an if condition is none (line 98)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 98, 21), result_eq_15044):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 98)
                        if_condition_15045 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 21), result_eq_15044)
                        # Assigning a type to the variable 'if_condition_15045' (line 98)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'if_condition_15045', if_condition_15045)
                        # SSA begins for if statement (line 98)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 99):
                        
                        # Assigning a BinOp to a Name (line 99):
                        
                        # Call to len(...): (line 99)
                        # Processing the call arguments (line 99)
                        # Getting the type of 'encoded' (line 99)
                        encoded_15047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 33), 'encoded', False)
                        # Processing the call keyword arguments (line 99)
                        kwargs_15048 = {}
                        # Getting the type of 'len' (line 99)
                        len_15046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'len', False)
                        # Calling len(args, kwargs) (line 99)
                        len_call_result_15049 = invoke(stypy.reporting.localization.Localization(__file__, 99, 29), len_15046, *[encoded_15047], **kwargs_15048)
                        
                        int_15050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 44), 'int')
                        # Applying the binary operator '%' (line 99)
                        result_mod_15051 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 29), '%', len_call_result_15049, int_15050)
                        
                        # Assigning a type to the variable 'paderr' (line 99)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'paderr', result_mod_15051)
                        # Getting the type of 'paderr' (line 100)
                        paderr_15052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'paderr')
                        # Testing if the type of an if condition is none (line 100)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 20), paderr_15052):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 100)
                            if_condition_15053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 20), paderr_15052)
                            # Assigning a type to the variable 'if_condition_15053' (line 100)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'if_condition_15053', if_condition_15053)
                            # SSA begins for if statement (line 100)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'encoded' (line 101)
                            encoded_15054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'encoded')
                            
                            # Obtaining the type of the subscript
                            int_15055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 42), 'int')
                            # Getting the type of 'paderr' (line 101)
                            paderr_15056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 46), 'paderr')
                            # Applying the binary operator '-' (line 101)
                            result_sub_15057 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 42), '-', int_15055, paderr_15056)
                            
                            slice_15058 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 101, 35), None, result_sub_15057, None)
                            str_15059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 35), 'str', '===')
                            # Obtaining the member '__getitem__' of a type (line 101)
                            getitem___15060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 35), str_15059, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 101)
                            subscript_call_result_15061 = invoke(stypy.reporting.localization.Localization(__file__, 101, 35), getitem___15060, slice_15058)
                            
                            # Applying the binary operator '+=' (line 101)
                            result_iadd_15062 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 24), '+=', encoded_15054, subscript_call_result_15061)
                            # Assigning a type to the variable 'encoded' (line 101)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'encoded', result_iadd_15062)
                            
                            # SSA join for if statement (line 100)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        
                        # SSA begins for try-except statement (line 102)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                        
                        # Assigning a Call to a Name (line 103):
                        
                        # Assigning a Call to a Name (line 103):
                        
                        # Call to decode(...): (line 103)
                        # Processing the call arguments (line 103)
                        # Getting the type of 'encoded' (line 103)
                        encoded_15066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 54), 'encoded', False)
                        # Processing the call keyword arguments (line 103)
                        kwargs_15067 = {}
                        # Getting the type of 'email' (line 103)
                        email_15063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'email', False)
                        # Obtaining the member 'base64mime' of a type (line 103)
                        base64mime_15064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 30), email_15063, 'base64mime')
                        # Obtaining the member 'decode' of a type (line 103)
                        decode_15065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 30), base64mime_15064, 'decode')
                        # Calling decode(args, kwargs) (line 103)
                        decode_call_result_15068 = invoke(stypy.reporting.localization.Localization(__file__, 103, 30), decode_15065, *[encoded_15066], **kwargs_15067)
                        
                        # Assigning a type to the variable 'dec' (line 103)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'dec', decode_call_result_15068)
                        # SSA branch for the except part of a try statement (line 102)
                        # SSA branch for the except 'Attribute' branch of a try statement (line 102)
                        module_type_store.open_ssa_branch('except')
                        # Getting the type of 'HeaderParseError' (line 108)
                        HeaderParseError_15069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'HeaderParseError')
                        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 108, 24), HeaderParseError_15069, 'raise parameter', BaseException)
                        # SSA join for try-except statement (line 102)
                        module_type_store = module_type_store.join_ssa_context()
                        
                        # SSA join for if statement (line 98)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 96)
                    if_condition_15035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 16), result_eq_15034)
                    # Assigning a type to the variable 'if_condition_15035' (line 96)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'if_condition_15035', if_condition_15035)
                    # SSA begins for if statement (line 96)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 97):
                    
                    # Assigning a Call to a Name (line 97):
                    
                    # Call to header_decode(...): (line 97)
                    # Processing the call arguments (line 97)
                    # Getting the type of 'encoded' (line 97)
                    encoded_15039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 57), 'encoded', False)
                    # Processing the call keyword arguments (line 97)
                    kwargs_15040 = {}
                    # Getting the type of 'email' (line 97)
                    email_15036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'email', False)
                    # Obtaining the member 'quoprimime' of a type (line 97)
                    quoprimime_15037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 26), email_15036, 'quoprimime')
                    # Obtaining the member 'header_decode' of a type (line 97)
                    header_decode_15038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 26), quoprimime_15037, 'header_decode')
                    # Calling header_decode(args, kwargs) (line 97)
                    header_decode_call_result_15041 = invoke(stypy.reporting.localization.Localization(__file__, 97, 26), header_decode_15038, *[encoded_15039], **kwargs_15040)
                    
                    # Assigning a type to the variable 'dec' (line 97)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'dec', header_decode_call_result_15041)
                    # SSA branch for the else part of an if statement (line 96)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'encoding' (line 98)
                    encoding_15042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'encoding')
                    str_15043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 33), 'str', 'b')
                    # Applying the binary operator '==' (line 98)
                    result_eq_15044 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 21), '==', encoding_15042, str_15043)
                    
                    # Testing if the type of an if condition is none (line 98)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 98, 21), result_eq_15044):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 98)
                        if_condition_15045 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 21), result_eq_15044)
                        # Assigning a type to the variable 'if_condition_15045' (line 98)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'if_condition_15045', if_condition_15045)
                        # SSA begins for if statement (line 98)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 99):
                        
                        # Assigning a BinOp to a Name (line 99):
                        
                        # Call to len(...): (line 99)
                        # Processing the call arguments (line 99)
                        # Getting the type of 'encoded' (line 99)
                        encoded_15047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 33), 'encoded', False)
                        # Processing the call keyword arguments (line 99)
                        kwargs_15048 = {}
                        # Getting the type of 'len' (line 99)
                        len_15046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'len', False)
                        # Calling len(args, kwargs) (line 99)
                        len_call_result_15049 = invoke(stypy.reporting.localization.Localization(__file__, 99, 29), len_15046, *[encoded_15047], **kwargs_15048)
                        
                        int_15050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 44), 'int')
                        # Applying the binary operator '%' (line 99)
                        result_mod_15051 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 29), '%', len_call_result_15049, int_15050)
                        
                        # Assigning a type to the variable 'paderr' (line 99)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'paderr', result_mod_15051)
                        # Getting the type of 'paderr' (line 100)
                        paderr_15052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'paderr')
                        # Testing if the type of an if condition is none (line 100)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 20), paderr_15052):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 100)
                            if_condition_15053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 20), paderr_15052)
                            # Assigning a type to the variable 'if_condition_15053' (line 100)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'if_condition_15053', if_condition_15053)
                            # SSA begins for if statement (line 100)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'encoded' (line 101)
                            encoded_15054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'encoded')
                            
                            # Obtaining the type of the subscript
                            int_15055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 42), 'int')
                            # Getting the type of 'paderr' (line 101)
                            paderr_15056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 46), 'paderr')
                            # Applying the binary operator '-' (line 101)
                            result_sub_15057 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 42), '-', int_15055, paderr_15056)
                            
                            slice_15058 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 101, 35), None, result_sub_15057, None)
                            str_15059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 35), 'str', '===')
                            # Obtaining the member '__getitem__' of a type (line 101)
                            getitem___15060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 35), str_15059, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 101)
                            subscript_call_result_15061 = invoke(stypy.reporting.localization.Localization(__file__, 101, 35), getitem___15060, slice_15058)
                            
                            # Applying the binary operator '+=' (line 101)
                            result_iadd_15062 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 24), '+=', encoded_15054, subscript_call_result_15061)
                            # Assigning a type to the variable 'encoded' (line 101)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'encoded', result_iadd_15062)
                            
                            # SSA join for if statement (line 100)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        
                        # SSA begins for try-except statement (line 102)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                        
                        # Assigning a Call to a Name (line 103):
                        
                        # Assigning a Call to a Name (line 103):
                        
                        # Call to decode(...): (line 103)
                        # Processing the call arguments (line 103)
                        # Getting the type of 'encoded' (line 103)
                        encoded_15066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 54), 'encoded', False)
                        # Processing the call keyword arguments (line 103)
                        kwargs_15067 = {}
                        # Getting the type of 'email' (line 103)
                        email_15063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'email', False)
                        # Obtaining the member 'base64mime' of a type (line 103)
                        base64mime_15064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 30), email_15063, 'base64mime')
                        # Obtaining the member 'decode' of a type (line 103)
                        decode_15065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 30), base64mime_15064, 'decode')
                        # Calling decode(args, kwargs) (line 103)
                        decode_call_result_15068 = invoke(stypy.reporting.localization.Localization(__file__, 103, 30), decode_15065, *[encoded_15066], **kwargs_15067)
                        
                        # Assigning a type to the variable 'dec' (line 103)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'dec', decode_call_result_15068)
                        # SSA branch for the except part of a try statement (line 102)
                        # SSA branch for the except 'Attribute' branch of a try statement (line 102)
                        module_type_store.open_ssa_branch('except')
                        # Getting the type of 'HeaderParseError' (line 108)
                        HeaderParseError_15069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'HeaderParseError')
                        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 108, 24), HeaderParseError_15069, 'raise parameter', BaseException)
                        # SSA join for try-except statement (line 102)
                        module_type_store = module_type_store.join_ssa_context()
                        
                        # SSA join for if statement (line 98)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 96)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Type idiom detected: calculating its left and rigth part (line 109)
                # Getting the type of 'dec' (line 109)
                dec_15070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'dec')
                # Getting the type of 'None' (line 109)
                None_15071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 26), 'None')
                
                (may_be_15072, more_types_in_union_15073) = may_be_none(dec_15070, None_15071)

                if may_be_15072:

                    if more_types_in_union_15073:
                        # Runtime conditional SSA (line 109)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Assigning a Name to a Name (line 110):
                    
                    # Assigning a Name to a Name (line 110):
                    # Getting the type of 'encoded' (line 110)
                    encoded_15074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'encoded')
                    # Assigning a type to the variable 'dec' (line 110)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'dec', encoded_15074)

                    if more_types_in_union_15073:
                        # SSA join for if statement (line 109)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Evaluating a boolean operation
                # Getting the type of 'decoded' (line 112)
                decoded_15075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'decoded')
                
                
                # Obtaining the type of the subscript
                int_15076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 43), 'int')
                
                # Obtaining the type of the subscript
                int_15077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 39), 'int')
                # Getting the type of 'decoded' (line 112)
                decoded_15078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'decoded')
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___15079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 31), decoded_15078, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_15080 = invoke(stypy.reporting.localization.Localization(__file__, 112, 31), getitem___15079, int_15077)
                
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___15081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 31), subscript_call_result_15080, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_15082 = invoke(stypy.reporting.localization.Localization(__file__, 112, 31), getitem___15081, int_15076)
                
                # Getting the type of 'charset' (line 112)
                charset_15083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 49), 'charset')
                # Applying the binary operator '==' (line 112)
                result_eq_15084 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 31), '==', subscript_call_result_15082, charset_15083)
                
                # Applying the binary operator 'and' (line 112)
                result_and_keyword_15085 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 19), 'and', decoded_15075, result_eq_15084)
                
                # Testing if the type of an if condition is none (line 112)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 16), result_and_keyword_15085):
                    
                    # Call to append(...): (line 115)
                    # Processing the call arguments (line 115)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 115)
                    tuple_15108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 36), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 115)
                    # Adding element type (line 115)
                    # Getting the type of 'dec' (line 115)
                    dec_15109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 36), 'dec', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 36), tuple_15108, dec_15109)
                    # Adding element type (line 115)
                    # Getting the type of 'charset' (line 115)
                    charset_15110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 41), 'charset', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 36), tuple_15108, charset_15110)
                    
                    # Processing the call keyword arguments (line 115)
                    kwargs_15111 = {}
                    # Getting the type of 'decoded' (line 115)
                    decoded_15106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'decoded', False)
                    # Obtaining the member 'append' of a type (line 115)
                    append_15107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 20), decoded_15106, 'append')
                    # Calling append(args, kwargs) (line 115)
                    append_call_result_15112 = invoke(stypy.reporting.localization.Localization(__file__, 115, 20), append_15107, *[tuple_15108], **kwargs_15111)
                    
                else:
                    
                    # Testing the type of an if condition (line 112)
                    if_condition_15086 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 16), result_and_keyword_15085)
                    # Assigning a type to the variable 'if_condition_15086' (line 112)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'if_condition_15086', if_condition_15086)
                    # SSA begins for if statement (line 112)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Tuple to a Subscript (line 113):
                    
                    # Assigning a Tuple to a Subscript (line 113):
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 113)
                    tuple_15087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 35), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 113)
                    # Adding element type (line 113)
                    
                    # Obtaining the type of the subscript
                    int_15088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 47), 'int')
                    
                    # Obtaining the type of the subscript
                    int_15089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 43), 'int')
                    # Getting the type of 'decoded' (line 113)
                    decoded_15090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'decoded')
                    # Obtaining the member '__getitem__' of a type (line 113)
                    getitem___15091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 35), decoded_15090, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                    subscript_call_result_15092 = invoke(stypy.reporting.localization.Localization(__file__, 113, 35), getitem___15091, int_15089)
                    
                    # Obtaining the member '__getitem__' of a type (line 113)
                    getitem___15093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 35), subscript_call_result_15092, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                    subscript_call_result_15094 = invoke(stypy.reporting.localization.Localization(__file__, 113, 35), getitem___15093, int_15088)
                    
                    # Getting the type of 'dec' (line 113)
                    dec_15095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 52), 'dec')
                    # Applying the binary operator '+' (line 113)
                    result_add_15096 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 35), '+', subscript_call_result_15094, dec_15095)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 35), tuple_15087, result_add_15096)
                    # Adding element type (line 113)
                    
                    # Obtaining the type of the subscript
                    int_15097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 69), 'int')
                    
                    # Obtaining the type of the subscript
                    int_15098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 65), 'int')
                    # Getting the type of 'decoded' (line 113)
                    decoded_15099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 57), 'decoded')
                    # Obtaining the member '__getitem__' of a type (line 113)
                    getitem___15100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 57), decoded_15099, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                    subscript_call_result_15101 = invoke(stypy.reporting.localization.Localization(__file__, 113, 57), getitem___15100, int_15098)
                    
                    # Obtaining the member '__getitem__' of a type (line 113)
                    getitem___15102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 57), subscript_call_result_15101, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                    subscript_call_result_15103 = invoke(stypy.reporting.localization.Localization(__file__, 113, 57), getitem___15102, int_15097)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 35), tuple_15087, subscript_call_result_15103)
                    
                    # Getting the type of 'decoded' (line 113)
                    decoded_15104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'decoded')
                    int_15105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 28), 'int')
                    # Storing an element on a container (line 113)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), decoded_15104, (int_15105, tuple_15087))
                    # SSA branch for the else part of an if statement (line 112)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to append(...): (line 115)
                    # Processing the call arguments (line 115)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 115)
                    tuple_15108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 36), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 115)
                    # Adding element type (line 115)
                    # Getting the type of 'dec' (line 115)
                    dec_15109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 36), 'dec', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 36), tuple_15108, dec_15109)
                    # Adding element type (line 115)
                    # Getting the type of 'charset' (line 115)
                    charset_15110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 41), 'charset', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 36), tuple_15108, charset_15110)
                    
                    # Processing the call keyword arguments (line 115)
                    kwargs_15111 = {}
                    # Getting the type of 'decoded' (line 115)
                    decoded_15106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'decoded', False)
                    # Obtaining the member 'append' of a type (line 115)
                    append_15107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 20), decoded_15106, 'append')
                    # Calling append(args, kwargs) (line 115)
                    append_call_result_15112 = invoke(stypy.reporting.localization.Localization(__file__, 115, 20), append_15107, *[tuple_15108], **kwargs_15111)
                    
                    # SSA join for if statement (line 112)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 92)
                module_type_store = module_type_store.join_ssa_context()
                

            # Deleting a member
            # Getting the type of 'parts' (line 116)
            parts_15113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'parts')
            
            # Obtaining the type of the subscript
            int_15114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 22), 'int')
            int_15115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'int')
            slice_15116 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 116, 16), int_15114, int_15115, None)
            # Getting the type of 'parts' (line 116)
            parts_15117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'parts')
            # Obtaining the member '__getitem__' of a type (line 116)
            getitem___15118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), parts_15117, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 116)
            subscript_call_result_15119 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), getitem___15118, slice_15116)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 12), parts_15113, subscript_call_result_15119)
            # SSA join for while statement (line 84)
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'decoded' (line 117)
    decoded_15120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'decoded')
    # Assigning a type to the variable 'stypy_return_type' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type', decoded_15120)
    
    # ################# End of 'decode_header(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode_header' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_15121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15121)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode_header'
    return stypy_return_type_15121

# Assigning a type to the variable 'decode_header' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'decode_header', decode_header)

@norecursion
def make_header(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 121)
    None_15122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 40), 'None')
    # Getting the type of 'None' (line 121)
    None_15123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 58), 'None')
    str_15124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 32), 'str', ' ')
    defaults = [None_15122, None_15123, str_15124]
    # Create a new context for function 'make_header'
    module_type_store = module_type_store.open_function_context('make_header', 121, 0, False)
    
    # Passed parameters checking function
    make_header.stypy_localization = localization
    make_header.stypy_type_of_self = None
    make_header.stypy_type_store = module_type_store
    make_header.stypy_function_name = 'make_header'
    make_header.stypy_param_names_list = ['decoded_seq', 'maxlinelen', 'header_name', 'continuation_ws']
    make_header.stypy_varargs_param_name = None
    make_header.stypy_kwargs_param_name = None
    make_header.stypy_call_defaults = defaults
    make_header.stypy_call_varargs = varargs
    make_header.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_header', ['decoded_seq', 'maxlinelen', 'header_name', 'continuation_ws'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_header', localization, ['decoded_seq', 'maxlinelen', 'header_name', 'continuation_ws'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_header(...)' code ##################

    str_15125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'str', 'Create a Header from a sequence of pairs as returned by decode_header()\n\n    decode_header() takes a header value string and returns a sequence of\n    pairs of the format (decoded_string, charset) where charset is the string\n    name of the character set.\n\n    This function takes one of those sequence of pairs and returns a Header\n    instance.  Optional maxlinelen, header_name, and continuation_ws are as in\n    the Header constructor.\n    ')
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to Header(...): (line 133)
    # Processing the call keyword arguments (line 133)
    # Getting the type of 'maxlinelen' (line 133)
    maxlinelen_15127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 26), 'maxlinelen', False)
    keyword_15128 = maxlinelen_15127
    # Getting the type of 'header_name' (line 133)
    header_name_15129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 50), 'header_name', False)
    keyword_15130 = header_name_15129
    # Getting the type of 'continuation_ws' (line 134)
    continuation_ws_15131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), 'continuation_ws', False)
    keyword_15132 = continuation_ws_15131
    kwargs_15133 = {'continuation_ws': keyword_15132, 'maxlinelen': keyword_15128, 'header_name': keyword_15130}
    # Getting the type of 'Header' (line 133)
    Header_15126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'Header', False)
    # Calling Header(args, kwargs) (line 133)
    Header_call_result_15134 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), Header_15126, *[], **kwargs_15133)
    
    # Assigning a type to the variable 'h' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'h', Header_call_result_15134)
    
    # Getting the type of 'decoded_seq' (line 135)
    decoded_seq_15135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'decoded_seq')
    # Assigning a type to the variable 'decoded_seq_15135' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'decoded_seq_15135', decoded_seq_15135)
    # Testing if the for loop is going to be iterated (line 135)
    # Testing the type of a for loop iterable (line 135)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 135, 4), decoded_seq_15135)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 135, 4), decoded_seq_15135):
        # Getting the type of the for loop variable (line 135)
        for_loop_var_15136 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 135, 4), decoded_seq_15135)
        # Assigning a type to the variable 's' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 's', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 4), for_loop_var_15136, 2, 0))
        # Assigning a type to the variable 'charset' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'charset', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 4), for_loop_var_15136, 2, 1))
        # SSA begins for a for statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'charset' (line 137)
        charset_15137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'charset')
        # Getting the type of 'None' (line 137)
        None_15138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'None')
        # Applying the binary operator 'isnot' (line 137)
        result_is_not_15139 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 11), 'isnot', charset_15137, None_15138)
        
        
        
        # Call to isinstance(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'charset' (line 137)
        charset_15141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'charset', False)
        # Getting the type of 'Charset' (line 137)
        Charset_15142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 59), 'Charset', False)
        # Processing the call keyword arguments (line 137)
        kwargs_15143 = {}
        # Getting the type of 'isinstance' (line 137)
        isinstance_15140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 137)
        isinstance_call_result_15144 = invoke(stypy.reporting.localization.Localization(__file__, 137, 39), isinstance_15140, *[charset_15141, Charset_15142], **kwargs_15143)
        
        # Applying the 'not' unary operator (line 137)
        result_not__15145 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 35), 'not', isinstance_call_result_15144)
        
        # Applying the binary operator 'and' (line 137)
        result_and_keyword_15146 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 11), 'and', result_is_not_15139, result_not__15145)
        
        # Testing if the type of an if condition is none (line 137)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 137, 8), result_and_keyword_15146):
            pass
        else:
            
            # Testing the type of an if condition (line 137)
            if_condition_15147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), result_and_keyword_15146)
            # Assigning a type to the variable 'if_condition_15147' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_15147', if_condition_15147)
            # SSA begins for if statement (line 137)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 138):
            
            # Assigning a Call to a Name (line 138):
            
            # Call to Charset(...): (line 138)
            # Processing the call arguments (line 138)
            # Getting the type of 'charset' (line 138)
            charset_15149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 30), 'charset', False)
            # Processing the call keyword arguments (line 138)
            kwargs_15150 = {}
            # Getting the type of 'Charset' (line 138)
            Charset_15148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 22), 'Charset', False)
            # Calling Charset(args, kwargs) (line 138)
            Charset_call_result_15151 = invoke(stypy.reporting.localization.Localization(__file__, 138, 22), Charset_15148, *[charset_15149], **kwargs_15150)
            
            # Assigning a type to the variable 'charset' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'charset', Charset_call_result_15151)
            # SSA join for if statement (line 137)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to append(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 's' (line 139)
        s_15154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 17), 's', False)
        # Getting the type of 'charset' (line 139)
        charset_15155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'charset', False)
        # Processing the call keyword arguments (line 139)
        kwargs_15156 = {}
        # Getting the type of 'h' (line 139)
        h_15152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'h', False)
        # Obtaining the member 'append' of a type (line 139)
        append_15153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), h_15152, 'append')
        # Calling append(args, kwargs) (line 139)
        append_call_result_15157 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), append_15153, *[s_15154, charset_15155], **kwargs_15156)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'h' (line 140)
    h_15158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'h')
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type', h_15158)
    
    # ################# End of 'make_header(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_header' in the type store
    # Getting the type of 'stypy_return_type' (line 121)
    stypy_return_type_15159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15159)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_header'
    return stypy_return_type_15159

# Assigning a type to the variable 'make_header' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'make_header', make_header)
# Declaration of the 'Header' class

class Header:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 145)
        None_15160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'None')
        # Getting the type of 'None' (line 145)
        None_15161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 39), 'None')
        # Getting the type of 'None' (line 146)
        None_15162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'None')
        # Getting the type of 'None' (line 146)
        None_15163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 46), 'None')
        str_15164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 33), 'str', ' ')
        str_15165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 45), 'str', 'strict')
        defaults = [None_15160, None_15161, None_15162, None_15163, str_15164, str_15165]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Header.__init__', ['s', 'charset', 'maxlinelen', 'header_name', 'continuation_ws', 'errors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['s', 'charset', 'maxlinelen', 'header_name', 'continuation_ws', 'errors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_15166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, (-1)), 'str', "Create a MIME-compliant header that can contain many character sets.\n\n        Optional s is the initial header value.  If None, the initial header\n        value is not set.  You can later append to the header with .append()\n        method calls.  s may be a byte string or a Unicode string, but see the\n        .append() documentation for semantics.\n\n        Optional charset serves two purposes: it has the same meaning as the\n        charset argument to the .append() method.  It also sets the default\n        character set for all subsequent .append() calls that omit the charset\n        argument.  If charset is not provided in the constructor, the us-ascii\n        charset is used both as s's initial charset and as the default for\n        subsequent .append() calls.\n\n        The maximum line length can be specified explicit via maxlinelen.  For\n        splitting the first line to a shorter value (to account for the field\n        header which isn't included in s, e.g. `Subject') pass in the name of\n        the field in header_name.  The default maxlinelen is 76.\n\n        continuation_ws must be RFC 2822 compliant folding whitespace (usually\n        either a space or a hard tab) which will be prepended to continuation\n        lines.\n\n        errors is passed through to the .append() call.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 173)
        # Getting the type of 'charset' (line 173)
        charset_15167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'charset')
        # Getting the type of 'None' (line 173)
        None_15168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'None')
        
        (may_be_15169, more_types_in_union_15170) = may_be_none(charset_15167, None_15168)

        if may_be_15169:

            if more_types_in_union_15170:
                # Runtime conditional SSA (line 173)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 174):
            
            # Assigning a Name to a Name (line 174):
            # Getting the type of 'USASCII' (line 174)
            USASCII_15171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'USASCII')
            # Assigning a type to the variable 'charset' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'charset', USASCII_15171)

            if more_types_in_union_15170:
                # SSA join for if statement (line 173)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to isinstance(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'charset' (line 175)
        charset_15173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 26), 'charset', False)
        # Getting the type of 'Charset' (line 175)
        Charset_15174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 'Charset', False)
        # Processing the call keyword arguments (line 175)
        kwargs_15175 = {}
        # Getting the type of 'isinstance' (line 175)
        isinstance_15172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 175)
        isinstance_call_result_15176 = invoke(stypy.reporting.localization.Localization(__file__, 175, 15), isinstance_15172, *[charset_15173, Charset_15174], **kwargs_15175)
        
        # Applying the 'not' unary operator (line 175)
        result_not__15177 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 11), 'not', isinstance_call_result_15176)
        
        # Testing if the type of an if condition is none (line 175)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 175, 8), result_not__15177):
            pass
        else:
            
            # Testing the type of an if condition (line 175)
            if_condition_15178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 8), result_not__15177)
            # Assigning a type to the variable 'if_condition_15178' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'if_condition_15178', if_condition_15178)
            # SSA begins for if statement (line 175)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 176):
            
            # Assigning a Call to a Name (line 176):
            
            # Call to Charset(...): (line 176)
            # Processing the call arguments (line 176)
            # Getting the type of 'charset' (line 176)
            charset_15180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 30), 'charset', False)
            # Processing the call keyword arguments (line 176)
            kwargs_15181 = {}
            # Getting the type of 'Charset' (line 176)
            Charset_15179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 22), 'Charset', False)
            # Calling Charset(args, kwargs) (line 176)
            Charset_call_result_15182 = invoke(stypy.reporting.localization.Localization(__file__, 176, 22), Charset_15179, *[charset_15180], **kwargs_15181)
            
            # Assigning a type to the variable 'charset' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'charset', Charset_call_result_15182)
            # SSA join for if statement (line 175)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Attribute (line 177):
        
        # Assigning a Name to a Attribute (line 177):
        # Getting the type of 'charset' (line 177)
        charset_15183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'charset')
        # Getting the type of 'self' (line 177)
        self_15184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Setting the type of the member '_charset' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_15184, '_charset', charset_15183)
        
        # Assigning a Name to a Attribute (line 178):
        
        # Assigning a Name to a Attribute (line 178):
        # Getting the type of 'continuation_ws' (line 178)
        continuation_ws_15185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 32), 'continuation_ws')
        # Getting the type of 'self' (line 178)
        self_15186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self')
        # Setting the type of the member '_continuation_ws' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_15186, '_continuation_ws', continuation_ws_15185)
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to len(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Call to replace(...): (line 179)
        # Processing the call arguments (line 179)
        str_15190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 55), 'str', '\t')
        # Getting the type of 'SPACE8' (line 179)
        SPACE8_15191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 61), 'SPACE8', False)
        # Processing the call keyword arguments (line 179)
        kwargs_15192 = {}
        # Getting the type of 'continuation_ws' (line 179)
        continuation_ws_15188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'continuation_ws', False)
        # Obtaining the member 'replace' of a type (line 179)
        replace_15189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 31), continuation_ws_15188, 'replace')
        # Calling replace(args, kwargs) (line 179)
        replace_call_result_15193 = invoke(stypy.reporting.localization.Localization(__file__, 179, 31), replace_15189, *[str_15190, SPACE8_15191], **kwargs_15192)
        
        # Processing the call keyword arguments (line 179)
        kwargs_15194 = {}
        # Getting the type of 'len' (line 179)
        len_15187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'len', False)
        # Calling len(args, kwargs) (line 179)
        len_call_result_15195 = invoke(stypy.reporting.localization.Localization(__file__, 179, 27), len_15187, *[replace_call_result_15193], **kwargs_15194)
        
        # Assigning a type to the variable 'cws_expanded_len' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'cws_expanded_len', len_call_result_15195)
        
        # Assigning a List to a Attribute (line 181):
        
        # Assigning a List to a Attribute (line 181):
        
        # Obtaining an instance of the builtin type 'list' (line 181)
        list_15196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 181)
        
        # Getting the type of 'self' (line 181)
        self_15197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self')
        # Setting the type of the member '_chunks' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_15197, '_chunks', list_15196)
        
        # Type idiom detected: calculating its left and rigth part (line 182)
        # Getting the type of 's' (line 182)
        s_15198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 's')
        # Getting the type of 'None' (line 182)
        None_15199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'None')
        
        (may_be_15200, more_types_in_union_15201) = may_not_be_none(s_15198, None_15199)

        if may_be_15200:

            if more_types_in_union_15201:
                # Runtime conditional SSA (line 182)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 183)
            # Processing the call arguments (line 183)
            # Getting the type of 's' (line 183)
            s_15204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 24), 's', False)
            # Getting the type of 'charset' (line 183)
            charset_15205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 27), 'charset', False)
            # Getting the type of 'errors' (line 183)
            errors_15206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 36), 'errors', False)
            # Processing the call keyword arguments (line 183)
            kwargs_15207 = {}
            # Getting the type of 'self' (line 183)
            self_15202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'self', False)
            # Obtaining the member 'append' of a type (line 183)
            append_15203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), self_15202, 'append')
            # Calling append(args, kwargs) (line 183)
            append_call_result_15208 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), append_15203, *[s_15204, charset_15205, errors_15206], **kwargs_15207)
            

            if more_types_in_union_15201:
                # SSA join for if statement (line 182)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 184)
        # Getting the type of 'maxlinelen' (line 184)
        maxlinelen_15209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'maxlinelen')
        # Getting the type of 'None' (line 184)
        None_15210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 'None')
        
        (may_be_15211, more_types_in_union_15212) = may_be_none(maxlinelen_15209, None_15210)

        if may_be_15211:

            if more_types_in_union_15212:
                # Runtime conditional SSA (line 184)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 185):
            
            # Assigning a Name to a Name (line 185):
            # Getting the type of 'MAXLINELEN' (line 185)
            MAXLINELEN_15213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 'MAXLINELEN')
            # Assigning a type to the variable 'maxlinelen' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'maxlinelen', MAXLINELEN_15213)

            if more_types_in_union_15212:
                # SSA join for if statement (line 184)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 186)
        # Getting the type of 'header_name' (line 186)
        header_name_15214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'header_name')
        # Getting the type of 'None' (line 186)
        None_15215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 26), 'None')
        
        (may_be_15216, more_types_in_union_15217) = may_be_none(header_name_15214, None_15215)

        if may_be_15216:

            if more_types_in_union_15217:
                # Runtime conditional SSA (line 186)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 189):
            
            # Assigning a Name to a Attribute (line 189):
            # Getting the type of 'maxlinelen' (line 189)
            maxlinelen_15218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 33), 'maxlinelen')
            # Getting the type of 'self' (line 189)
            self_15219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'self')
            # Setting the type of the member '_firstlinelen' of a type (line 189)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), self_15219, '_firstlinelen', maxlinelen_15218)

            if more_types_in_union_15217:
                # Runtime conditional SSA for else branch (line 186)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_15216) or more_types_in_union_15217):
            
            # Assigning a BinOp to a Attribute (line 193):
            
            # Assigning a BinOp to a Attribute (line 193):
            # Getting the type of 'maxlinelen' (line 193)
            maxlinelen_15220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 33), 'maxlinelen')
            
            # Call to len(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 'header_name' (line 193)
            header_name_15222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 50), 'header_name', False)
            # Processing the call keyword arguments (line 193)
            kwargs_15223 = {}
            # Getting the type of 'len' (line 193)
            len_15221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 46), 'len', False)
            # Calling len(args, kwargs) (line 193)
            len_call_result_15224 = invoke(stypy.reporting.localization.Localization(__file__, 193, 46), len_15221, *[header_name_15222], **kwargs_15223)
            
            # Applying the binary operator '-' (line 193)
            result_sub_15225 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 33), '-', maxlinelen_15220, len_call_result_15224)
            
            int_15226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 65), 'int')
            # Applying the binary operator '-' (line 193)
            result_sub_15227 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 63), '-', result_sub_15225, int_15226)
            
            # Getting the type of 'self' (line 193)
            self_15228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'self')
            # Setting the type of the member '_firstlinelen' of a type (line 193)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), self_15228, '_firstlinelen', result_sub_15227)

            if (may_be_15216 and more_types_in_union_15217):
                # SSA join for if statement (line 186)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Attribute (line 196):
        
        # Assigning a BinOp to a Attribute (line 196):
        # Getting the type of 'maxlinelen' (line 196)
        maxlinelen_15229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'maxlinelen')
        # Getting the type of 'cws_expanded_len' (line 196)
        cws_expanded_len_15230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 40), 'cws_expanded_len')
        # Applying the binary operator '-' (line 196)
        result_sub_15231 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 27), '-', maxlinelen_15229, cws_expanded_len_15230)
        
        # Getting the type of 'self' (line 196)
        self_15232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self')
        # Setting the type of the member '_maxlinelen' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_15232, '_maxlinelen', result_sub_15231)
        
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
        module_type_store = module_type_store.open_function_context('__str__', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Header.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Header.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Header.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Header.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Header.stypy__str__')
        Header.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Header.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Header.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Header.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Header.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Header.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Header.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Header.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        str_15233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'str', 'A synonym for self.encode().')
        
        # Call to encode(...): (line 200)
        # Processing the call keyword arguments (line 200)
        kwargs_15236 = {}
        # Getting the type of 'self' (line 200)
        self_15234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'self', False)
        # Obtaining the member 'encode' of a type (line 200)
        encode_15235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), self_15234, 'encode')
        # Calling encode(args, kwargs) (line 200)
        encode_call_result_15237 = invoke(stypy.reporting.localization.Localization(__file__, 200, 15), encode_15235, *[], **kwargs_15236)
        
        # Assigning a type to the variable 'stypy_return_type' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'stypy_return_type', encode_call_result_15237)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_15238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15238)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_15238


    @norecursion
    def __unicode__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__unicode__'
        module_type_store = module_type_store.open_function_context('__unicode__', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Header.__unicode__.__dict__.__setitem__('stypy_localization', localization)
        Header.__unicode__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Header.__unicode__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Header.__unicode__.__dict__.__setitem__('stypy_function_name', 'Header.__unicode__')
        Header.__unicode__.__dict__.__setitem__('stypy_param_names_list', [])
        Header.__unicode__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Header.__unicode__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Header.__unicode__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Header.__unicode__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Header.__unicode__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Header.__unicode__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Header.__unicode__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__unicode__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__unicode__(...)' code ##################

        str_15239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 8), 'str', 'Helper for the built-in unicode function.')
        
        # Assigning a List to a Name (line 204):
        
        # Assigning a List to a Name (line 204):
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_15240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        
        # Assigning a type to the variable 'uchunks' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'uchunks', list_15240)
        
        # Assigning a Name to a Name (line 205):
        
        # Assigning a Name to a Name (line 205):
        # Getting the type of 'None' (line 205)
        None_15241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'None')
        # Assigning a type to the variable 'lastcs' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'lastcs', None_15241)
        
        # Getting the type of 'self' (line 206)
        self_15242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'self')
        # Obtaining the member '_chunks' of a type (line 206)
        _chunks_15243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 26), self_15242, '_chunks')
        # Assigning a type to the variable '_chunks_15243' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), '_chunks_15243', _chunks_15243)
        # Testing if the for loop is going to be iterated (line 206)
        # Testing the type of a for loop iterable (line 206)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 206, 8), _chunks_15243)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 206, 8), _chunks_15243):
            # Getting the type of the for loop variable (line 206)
            for_loop_var_15244 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 206, 8), _chunks_15243)
            # Assigning a type to the variable 's' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 's', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 8), for_loop_var_15244, 2, 0))
            # Assigning a type to the variable 'charset' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'charset', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 8), for_loop_var_15244, 2, 1))
            # SSA begins for a for statement (line 206)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Name to a Name (line 211):
            
            # Assigning a Name to a Name (line 211):
            # Getting the type of 'charset' (line 211)
            charset_15245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'charset')
            # Assigning a type to the variable 'nextcs' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'nextcs', charset_15245)
            # Getting the type of 'uchunks' (line 212)
            uchunks_15246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'uchunks')
            # Testing if the type of an if condition is none (line 212)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 212, 12), uchunks_15246):
                pass
            else:
                
                # Testing the type of an if condition (line 212)
                if_condition_15247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 12), uchunks_15246)
                # Assigning a type to the variable 'if_condition_15247' (line 212)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'if_condition_15247', if_condition_15247)
                # SSA begins for if statement (line 212)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'lastcs' (line 213)
                lastcs_15248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'lastcs')
                
                # Obtaining an instance of the builtin type 'tuple' (line 213)
                tuple_15249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 34), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 213)
                # Adding element type (line 213)
                # Getting the type of 'None' (line 213)
                None_15250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 34), 'None')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 34), tuple_15249, None_15250)
                # Adding element type (line 213)
                str_15251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 40), 'str', 'us-ascii')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 34), tuple_15249, str_15251)
                
                # Applying the binary operator 'notin' (line 213)
                result_contains_15252 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 19), 'notin', lastcs_15248, tuple_15249)
                
                # Testing if the type of an if condition is none (line 213)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 213, 16), result_contains_15252):
                    
                    # Getting the type of 'nextcs' (line 217)
                    nextcs_15266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'nextcs')
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 217)
                    tuple_15267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 36), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 217)
                    # Adding element type (line 217)
                    # Getting the type of 'None' (line 217)
                    None_15268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 36), 'None')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 36), tuple_15267, None_15268)
                    # Adding element type (line 217)
                    str_15269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 42), 'str', 'us-ascii')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 36), tuple_15267, str_15269)
                    
                    # Applying the binary operator 'notin' (line 217)
                    result_contains_15270 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 21), 'notin', nextcs_15266, tuple_15267)
                    
                    # Testing if the type of an if condition is none (line 217)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 217, 21), result_contains_15270):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 217)
                        if_condition_15271 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 21), result_contains_15270)
                        # Assigning a type to the variable 'if_condition_15271' (line 217)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'if_condition_15271', if_condition_15271)
                        # SSA begins for if statement (line 217)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 218)
                        # Processing the call arguments (line 218)
                        # Getting the type of 'USPACE' (line 218)
                        USPACE_15274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 35), 'USPACE', False)
                        # Processing the call keyword arguments (line 218)
                        kwargs_15275 = {}
                        # Getting the type of 'uchunks' (line 218)
                        uchunks_15272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 20), 'uchunks', False)
                        # Obtaining the member 'append' of a type (line 218)
                        append_15273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 20), uchunks_15272, 'append')
                        # Calling append(args, kwargs) (line 218)
                        append_call_result_15276 = invoke(stypy.reporting.localization.Localization(__file__, 218, 20), append_15273, *[USPACE_15274], **kwargs_15275)
                        
                        # SSA join for if statement (line 217)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 213)
                    if_condition_15253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 16), result_contains_15252)
                    # Assigning a type to the variable 'if_condition_15253' (line 213)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'if_condition_15253', if_condition_15253)
                    # SSA begins for if statement (line 213)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'nextcs' (line 214)
                    nextcs_15254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'nextcs')
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 214)
                    tuple_15255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 34), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 214)
                    # Adding element type (line 214)
                    # Getting the type of 'None' (line 214)
                    None_15256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 'None')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 34), tuple_15255, None_15256)
                    # Adding element type (line 214)
                    str_15257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 40), 'str', 'us-ascii')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 34), tuple_15255, str_15257)
                    
                    # Applying the binary operator 'in' (line 214)
                    result_contains_15258 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 23), 'in', nextcs_15254, tuple_15255)
                    
                    # Testing if the type of an if condition is none (line 214)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 214, 20), result_contains_15258):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 214)
                        if_condition_15259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 20), result_contains_15258)
                        # Assigning a type to the variable 'if_condition_15259' (line 214)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'if_condition_15259', if_condition_15259)
                        # SSA begins for if statement (line 214)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 215)
                        # Processing the call arguments (line 215)
                        # Getting the type of 'USPACE' (line 215)
                        USPACE_15262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 39), 'USPACE', False)
                        # Processing the call keyword arguments (line 215)
                        kwargs_15263 = {}
                        # Getting the type of 'uchunks' (line 215)
                        uchunks_15260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'uchunks', False)
                        # Obtaining the member 'append' of a type (line 215)
                        append_15261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 24), uchunks_15260, 'append')
                        # Calling append(args, kwargs) (line 215)
                        append_call_result_15264 = invoke(stypy.reporting.localization.Localization(__file__, 215, 24), append_15261, *[USPACE_15262], **kwargs_15263)
                        
                        
                        # Assigning a Name to a Name (line 216):
                        
                        # Assigning a Name to a Name (line 216):
                        # Getting the type of 'None' (line 216)
                        None_15265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 33), 'None')
                        # Assigning a type to the variable 'nextcs' (line 216)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 'nextcs', None_15265)
                        # SSA join for if statement (line 214)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 213)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'nextcs' (line 217)
                    nextcs_15266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'nextcs')
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 217)
                    tuple_15267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 36), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 217)
                    # Adding element type (line 217)
                    # Getting the type of 'None' (line 217)
                    None_15268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 36), 'None')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 36), tuple_15267, None_15268)
                    # Adding element type (line 217)
                    str_15269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 42), 'str', 'us-ascii')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 36), tuple_15267, str_15269)
                    
                    # Applying the binary operator 'notin' (line 217)
                    result_contains_15270 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 21), 'notin', nextcs_15266, tuple_15267)
                    
                    # Testing if the type of an if condition is none (line 217)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 217, 21), result_contains_15270):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 217)
                        if_condition_15271 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 21), result_contains_15270)
                        # Assigning a type to the variable 'if_condition_15271' (line 217)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'if_condition_15271', if_condition_15271)
                        # SSA begins for if statement (line 217)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 218)
                        # Processing the call arguments (line 218)
                        # Getting the type of 'USPACE' (line 218)
                        USPACE_15274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 35), 'USPACE', False)
                        # Processing the call keyword arguments (line 218)
                        kwargs_15275 = {}
                        # Getting the type of 'uchunks' (line 218)
                        uchunks_15272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 20), 'uchunks', False)
                        # Obtaining the member 'append' of a type (line 218)
                        append_15273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 20), uchunks_15272, 'append')
                        # Calling append(args, kwargs) (line 218)
                        append_call_result_15276 = invoke(stypy.reporting.localization.Localization(__file__, 218, 20), append_15273, *[USPACE_15274], **kwargs_15275)
                        
                        # SSA join for if statement (line 217)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 213)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 212)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Name (line 219):
            
            # Assigning a Name to a Name (line 219):
            # Getting the type of 'nextcs' (line 219)
            nextcs_15277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'nextcs')
            # Assigning a type to the variable 'lastcs' (line 219)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'lastcs', nextcs_15277)
            
            # Call to append(...): (line 220)
            # Processing the call arguments (line 220)
            
            # Call to unicode(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 's' (line 220)
            s_15281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 35), 's', False)
            
            # Call to str(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'charset' (line 220)
            charset_15283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 42), 'charset', False)
            # Processing the call keyword arguments (line 220)
            kwargs_15284 = {}
            # Getting the type of 'str' (line 220)
            str_15282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'str', False)
            # Calling str(args, kwargs) (line 220)
            str_call_result_15285 = invoke(stypy.reporting.localization.Localization(__file__, 220, 38), str_15282, *[charset_15283], **kwargs_15284)
            
            # Processing the call keyword arguments (line 220)
            kwargs_15286 = {}
            # Getting the type of 'unicode' (line 220)
            unicode_15280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 27), 'unicode', False)
            # Calling unicode(args, kwargs) (line 220)
            unicode_call_result_15287 = invoke(stypy.reporting.localization.Localization(__file__, 220, 27), unicode_15280, *[s_15281, str_call_result_15285], **kwargs_15286)
            
            # Processing the call keyword arguments (line 220)
            kwargs_15288 = {}
            # Getting the type of 'uchunks' (line 220)
            uchunks_15278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'uchunks', False)
            # Obtaining the member 'append' of a type (line 220)
            append_15279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), uchunks_15278, 'append')
            # Calling append(args, kwargs) (line 220)
            append_call_result_15289 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), append_15279, *[unicode_call_result_15287], **kwargs_15288)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to join(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'uchunks' (line 221)
        uchunks_15292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 33), 'uchunks', False)
        # Processing the call keyword arguments (line 221)
        kwargs_15293 = {}
        # Getting the type of 'UEMPTYSTRING' (line 221)
        UEMPTYSTRING_15290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'UEMPTYSTRING', False)
        # Obtaining the member 'join' of a type (line 221)
        join_15291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 15), UEMPTYSTRING_15290, 'join')
        # Calling join(args, kwargs) (line 221)
        join_call_result_15294 = invoke(stypy.reporting.localization.Localization(__file__, 221, 15), join_15291, *[uchunks_15292], **kwargs_15293)
        
        # Assigning a type to the variable 'stypy_return_type' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'stypy_return_type', join_call_result_15294)
        
        # ################# End of '__unicode__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__unicode__' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_15295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15295)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__unicode__'
        return stypy_return_type_15295


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Header.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        Header.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Header.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Header.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'Header.stypy__eq__')
        Header.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Header.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Header.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Header.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Header.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Header.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Header.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Header.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'other' (line 228)
        other_15296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'other')
        
        # Call to encode(...): (line 228)
        # Processing the call keyword arguments (line 228)
        kwargs_15299 = {}
        # Getting the type of 'self' (line 228)
        self_15297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'self', False)
        # Obtaining the member 'encode' of a type (line 228)
        encode_15298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 24), self_15297, 'encode')
        # Calling encode(args, kwargs) (line 228)
        encode_call_result_15300 = invoke(stypy.reporting.localization.Localization(__file__, 228, 24), encode_15298, *[], **kwargs_15299)
        
        # Applying the binary operator '==' (line 228)
        result_eq_15301 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 15), '==', other_15296, encode_call_result_15300)
        
        # Assigning a type to the variable 'stypy_return_type' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stypy_return_type', result_eq_15301)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_15302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_15302


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Header.__ne__.__dict__.__setitem__('stypy_localization', localization)
        Header.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Header.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Header.__ne__.__dict__.__setitem__('stypy_function_name', 'Header.__ne__')
        Header.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Header.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Header.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Header.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Header.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Header.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Header.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Header.__ne__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 231)
        self_15303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'self')
        # Getting the type of 'other' (line 231)
        other_15304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'other')
        # Applying the binary operator '==' (line 231)
        result_eq_15305 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 19), '==', self_15303, other_15304)
        
        # Applying the 'not' unary operator (line 231)
        result_not__15306 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 15), 'not', result_eq_15305)
        
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', result_not__15306)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_15307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15307)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_15307


    @norecursion
    def append(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 233)
        None_15308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 32), 'None')
        str_15309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 45), 'str', 'strict')
        defaults = [None_15308, str_15309]
        # Create a new context for function 'append'
        module_type_store = module_type_store.open_function_context('append', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Header.append.__dict__.__setitem__('stypy_localization', localization)
        Header.append.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Header.append.__dict__.__setitem__('stypy_type_store', module_type_store)
        Header.append.__dict__.__setitem__('stypy_function_name', 'Header.append')
        Header.append.__dict__.__setitem__('stypy_param_names_list', ['s', 'charset', 'errors'])
        Header.append.__dict__.__setitem__('stypy_varargs_param_name', None)
        Header.append.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Header.append.__dict__.__setitem__('stypy_call_defaults', defaults)
        Header.append.__dict__.__setitem__('stypy_call_varargs', varargs)
        Header.append.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Header.append.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Header.append', ['s', 'charset', 'errors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'append', localization, ['s', 'charset', 'errors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'append(...)' code ##################

        str_15310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, (-1)), 'str', "Append a string to the MIME header.\n\n        Optional charset, if given, should be a Charset instance or the name\n        of a character set (which will be converted to a Charset instance).  A\n        value of None (the default) means that the charset given in the\n        constructor is used.\n\n        s may be a byte string or a Unicode string.  If it is a byte string\n        (i.e. isinstance(s, str) is true), then charset is the encoding of\n        that byte string, and a UnicodeError will be raised if the string\n        cannot be decoded with that charset.  If s is a Unicode string, then\n        charset is a hint specifying the character set of the characters in\n        the string.  In this case, when producing an RFC 2822 compliant header\n        using RFC 2047 rules, the Unicode string will be encoded using the\n        following charsets in order: us-ascii, the charset hint, utf-8.  The\n        first character set not to provoke a UnicodeError is used.\n\n        Optional `errors' is passed as the third argument to any unicode() or\n        ustr.encode() call.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 254)
        # Getting the type of 'charset' (line 254)
        charset_15311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 'charset')
        # Getting the type of 'None' (line 254)
        None_15312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 22), 'None')
        
        (may_be_15313, more_types_in_union_15314) = may_be_none(charset_15311, None_15312)

        if may_be_15313:

            if more_types_in_union_15314:
                # Runtime conditional SSA (line 254)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 255):
            
            # Assigning a Attribute to a Name (line 255):
            # Getting the type of 'self' (line 255)
            self_15315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 22), 'self')
            # Obtaining the member '_charset' of a type (line 255)
            _charset_15316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 22), self_15315, '_charset')
            # Assigning a type to the variable 'charset' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'charset', _charset_15316)

            if more_types_in_union_15314:
                # Runtime conditional SSA for else branch (line 254)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_15313) or more_types_in_union_15314):
            
            
            # Call to isinstance(...): (line 256)
            # Processing the call arguments (line 256)
            # Getting the type of 'charset' (line 256)
            charset_15318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'charset', False)
            # Getting the type of 'Charset' (line 256)
            Charset_15319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 37), 'Charset', False)
            # Processing the call keyword arguments (line 256)
            kwargs_15320 = {}
            # Getting the type of 'isinstance' (line 256)
            isinstance_15317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 256)
            isinstance_call_result_15321 = invoke(stypy.reporting.localization.Localization(__file__, 256, 17), isinstance_15317, *[charset_15318, Charset_15319], **kwargs_15320)
            
            # Applying the 'not' unary operator (line 256)
            result_not__15322 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 13), 'not', isinstance_call_result_15321)
            
            # Testing if the type of an if condition is none (line 256)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 256, 13), result_not__15322):
                pass
            else:
                
                # Testing the type of an if condition (line 256)
                if_condition_15323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 13), result_not__15322)
                # Assigning a type to the variable 'if_condition_15323' (line 256)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 13), 'if_condition_15323', if_condition_15323)
                # SSA begins for if statement (line 256)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 257):
                
                # Assigning a Call to a Name (line 257):
                
                # Call to Charset(...): (line 257)
                # Processing the call arguments (line 257)
                # Getting the type of 'charset' (line 257)
                charset_15325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 30), 'charset', False)
                # Processing the call keyword arguments (line 257)
                kwargs_15326 = {}
                # Getting the type of 'Charset' (line 257)
                Charset_15324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 'Charset', False)
                # Calling Charset(args, kwargs) (line 257)
                Charset_call_result_15327 = invoke(stypy.reporting.localization.Localization(__file__, 257, 22), Charset_15324, *[charset_15325], **kwargs_15326)
                
                # Assigning a type to the variable 'charset' (line 257)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'charset', Charset_call_result_15327)
                # SSA join for if statement (line 256)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_15313 and more_types_in_union_15314):
                # SSA join for if statement (line 254)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'charset' (line 259)
        charset_15328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'charset')
        str_15329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 22), 'str', '8bit')
        # Applying the binary operator '!=' (line 259)
        result_ne_15330 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 11), '!=', charset_15328, str_15329)
        
        # Testing if the type of an if condition is none (line 259)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 259, 8), result_ne_15330):
            pass
        else:
            
            # Testing the type of an if condition (line 259)
            if_condition_15331 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 8), result_ne_15330)
            # Assigning a type to the variable 'if_condition_15331' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'if_condition_15331', if_condition_15331)
            # SSA begins for if statement (line 259)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 263)
            # Getting the type of 'str' (line 263)
            str_15332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 29), 'str')
            # Getting the type of 's' (line 263)
            s_15333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 26), 's')
            
            (may_be_15334, more_types_in_union_15335) = may_be_subtype(str_15332, s_15333)

            if may_be_15334:

                if more_types_in_union_15335:
                    # Runtime conditional SSA (line 263)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 's' (line 263)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 's', remove_not_subtype_from_union(s_15333, str))
                
                # Assigning a BoolOp to a Name (line 266):
                
                # Assigning a BoolOp to a Name (line 266):
                
                # Evaluating a boolean operation
                # Getting the type of 'charset' (line 266)
                charset_15336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 26), 'charset')
                # Obtaining the member 'input_codec' of a type (line 266)
                input_codec_15337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 26), charset_15336, 'input_codec')
                str_15338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 49), 'str', 'us-ascii')
                # Applying the binary operator 'or' (line 266)
                result_or_keyword_15339 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 26), 'or', input_codec_15337, str_15338)
                
                # Assigning a type to the variable 'incodec' (line 266)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'incodec', result_or_keyword_15339)
                
                # Assigning a Call to a Name (line 267):
                
                # Assigning a Call to a Name (line 267):
                
                # Call to unicode(...): (line 267)
                # Processing the call arguments (line 267)
                # Getting the type of 's' (line 267)
                s_15341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 31), 's', False)
                # Getting the type of 'incodec' (line 267)
                incodec_15342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 34), 'incodec', False)
                # Getting the type of 'errors' (line 267)
                errors_15343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 43), 'errors', False)
                # Processing the call keyword arguments (line 267)
                kwargs_15344 = {}
                # Getting the type of 'unicode' (line 267)
                unicode_15340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'unicode', False)
                # Calling unicode(args, kwargs) (line 267)
                unicode_call_result_15345 = invoke(stypy.reporting.localization.Localization(__file__, 267, 23), unicode_15340, *[s_15341, incodec_15342, errors_15343], **kwargs_15344)
                
                # Assigning a type to the variable 'ustr' (line 267)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'ustr', unicode_call_result_15345)
                
                # Assigning a BoolOp to a Name (line 271):
                
                # Assigning a BoolOp to a Name (line 271):
                
                # Evaluating a boolean operation
                # Getting the type of 'charset' (line 271)
                charset_15346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 27), 'charset')
                # Obtaining the member 'output_codec' of a type (line 271)
                output_codec_15347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 27), charset_15346, 'output_codec')
                str_15348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 51), 'str', 'us-ascii')
                # Applying the binary operator 'or' (line 271)
                result_or_keyword_15349 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 27), 'or', output_codec_15347, str_15348)
                
                # Assigning a type to the variable 'outcodec' (line 271)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'outcodec', result_or_keyword_15349)
                
                # Call to encode(...): (line 272)
                # Processing the call arguments (line 272)
                # Getting the type of 'outcodec' (line 272)
                outcodec_15352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 28), 'outcodec', False)
                # Getting the type of 'errors' (line 272)
                errors_15353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 38), 'errors', False)
                # Processing the call keyword arguments (line 272)
                kwargs_15354 = {}
                # Getting the type of 'ustr' (line 272)
                ustr_15350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'ustr', False)
                # Obtaining the member 'encode' of a type (line 272)
                encode_15351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), ustr_15350, 'encode')
                # Calling encode(args, kwargs) (line 272)
                encode_call_result_15355 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), encode_15351, *[outcodec_15352, errors_15353], **kwargs_15354)
                

                if more_types_in_union_15335:
                    # Runtime conditional SSA for else branch (line 263)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_15334) or more_types_in_union_15335):
                # Assigning a type to the variable 's' (line 263)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 's', remove_subtype_from_union(s_15333, str))
                
                # Type idiom detected: calculating its left and rigth part (line 273)
                # Getting the type of 'unicode' (line 273)
                unicode_15356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 31), 'unicode')
                # Getting the type of 's' (line 273)
                s_15357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 28), 's')
                
                (may_be_15358, more_types_in_union_15359) = may_be_subtype(unicode_15356, s_15357)

                if may_be_15358:

                    if more_types_in_union_15359:
                        # Runtime conditional SSA (line 273)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 's' (line 273)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 17), 's', remove_not_subtype_from_union(s_15357, unicode))
                    
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 277)
                    tuple_15360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 31), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 277)
                    # Adding element type (line 277)
                    # Getting the type of 'USASCII' (line 277)
                    USASCII_15361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 31), 'USASCII')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 31), tuple_15360, USASCII_15361)
                    # Adding element type (line 277)
                    # Getting the type of 'charset' (line 277)
                    charset_15362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 40), 'charset')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 31), tuple_15360, charset_15362)
                    # Adding element type (line 277)
                    # Getting the type of 'UTF8' (line 277)
                    UTF8_15363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 49), 'UTF8')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 31), tuple_15360, UTF8_15363)
                    
                    # Assigning a type to the variable 'tuple_15360' (line 277)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'tuple_15360', tuple_15360)
                    # Testing if the for loop is going to be iterated (line 277)
                    # Testing the type of a for loop iterable (line 277)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 16), tuple_15360)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 277, 16), tuple_15360):
                        # Getting the type of the for loop variable (line 277)
                        for_loop_var_15364 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 16), tuple_15360)
                        # Assigning a type to the variable 'charset' (line 277)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'charset', for_loop_var_15364)
                        # SSA begins for a for statement (line 277)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        
                        # SSA begins for try-except statement (line 278)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                        
                        # Assigning a BoolOp to a Name (line 279):
                        
                        # Assigning a BoolOp to a Name (line 279):
                        
                        # Evaluating a boolean operation
                        # Getting the type of 'charset' (line 279)
                        charset_15365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 35), 'charset')
                        # Obtaining the member 'output_codec' of a type (line 279)
                        output_codec_15366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 35), charset_15365, 'output_codec')
                        str_15367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 59), 'str', 'us-ascii')
                        # Applying the binary operator 'or' (line 279)
                        result_or_keyword_15368 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 35), 'or', output_codec_15366, str_15367)
                        
                        # Assigning a type to the variable 'outcodec' (line 279)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'outcodec', result_or_keyword_15368)
                        
                        # Assigning a Call to a Name (line 280):
                        
                        # Assigning a Call to a Name (line 280):
                        
                        # Call to encode(...): (line 280)
                        # Processing the call arguments (line 280)
                        # Getting the type of 'outcodec' (line 280)
                        outcodec_15371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 37), 'outcodec', False)
                        # Getting the type of 'errors' (line 280)
                        errors_15372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 47), 'errors', False)
                        # Processing the call keyword arguments (line 280)
                        kwargs_15373 = {}
                        # Getting the type of 's' (line 280)
                        s_15369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 28), 's', False)
                        # Obtaining the member 'encode' of a type (line 280)
                        encode_15370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 28), s_15369, 'encode')
                        # Calling encode(args, kwargs) (line 280)
                        encode_call_result_15374 = invoke(stypy.reporting.localization.Localization(__file__, 280, 28), encode_15370, *[outcodec_15371, errors_15372], **kwargs_15373)
                        
                        # Assigning a type to the variable 's' (line 280)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 's', encode_call_result_15374)
                        # SSA branch for the except part of a try statement (line 278)
                        # SSA branch for the except 'UnicodeError' branch of a try statement (line 278)
                        module_type_store.open_ssa_branch('except')
                        pass
                        # SSA join for try-except statement (line 278)
                        module_type_store = module_type_store.join_ssa_context()
                        
                        # SSA branch for the else part of a for statement (line 277)
                        module_type_store.open_ssa_branch('for loop else')
                        # Evaluating assert statement condition
                        # Getting the type of 'False' (line 285)
                        False_15375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 27), 'False')
                        assert_15376 = False_15375
                        # Assigning a type to the variable 'assert_15376' (line 285)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 20), 'assert_15376', False_15375)
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()
                    else:
                        # Evaluating assert statement condition
                        # Getting the type of 'False' (line 285)
                        False_15375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 27), 'False')
                        assert_15376 = False_15375
                        # Assigning a type to the variable 'assert_15376' (line 285)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 20), 'assert_15376', False_15375)

                    

                    if more_types_in_union_15359:
                        # SSA join for if statement (line 273)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_15334 and more_types_in_union_15335):
                    # SSA join for if statement (line 263)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 259)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to append(...): (line 286)
        # Processing the call arguments (line 286)
        
        # Obtaining an instance of the builtin type 'tuple' (line 286)
        tuple_15380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 286)
        # Adding element type (line 286)
        # Getting the type of 's' (line 286)
        s_15381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 29), 's', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 29), tuple_15380, s_15381)
        # Adding element type (line 286)
        # Getting the type of 'charset' (line 286)
        charset_15382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 32), 'charset', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 29), tuple_15380, charset_15382)
        
        # Processing the call keyword arguments (line 286)
        kwargs_15383 = {}
        # Getting the type of 'self' (line 286)
        self_15377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'self', False)
        # Obtaining the member '_chunks' of a type (line 286)
        _chunks_15378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), self_15377, '_chunks')
        # Obtaining the member 'append' of a type (line 286)
        append_15379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), _chunks_15378, 'append')
        # Calling append(args, kwargs) (line 286)
        append_call_result_15384 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), append_15379, *[tuple_15380], **kwargs_15383)
        
        
        # ################# End of 'append(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'append' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_15385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'append'
        return stypy_return_type_15385


    @norecursion
    def _split(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_split'
        module_type_store = module_type_store.open_function_context('_split', 288, 4, False)
        # Assigning a type to the variable 'self' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Header._split.__dict__.__setitem__('stypy_localization', localization)
        Header._split.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Header._split.__dict__.__setitem__('stypy_type_store', module_type_store)
        Header._split.__dict__.__setitem__('stypy_function_name', 'Header._split')
        Header._split.__dict__.__setitem__('stypy_param_names_list', ['s', 'charset', 'maxlinelen', 'splitchars'])
        Header._split.__dict__.__setitem__('stypy_varargs_param_name', None)
        Header._split.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Header._split.__dict__.__setitem__('stypy_call_defaults', defaults)
        Header._split.__dict__.__setitem__('stypy_call_varargs', varargs)
        Header._split.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Header._split.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Header._split', ['s', 'charset', 'maxlinelen', 'splitchars'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_split', localization, ['s', 'charset', 'maxlinelen', 'splitchars'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_split(...)' code ##################

        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to to_splittable(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 's' (line 290)
        s_15388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 43), 's', False)
        # Processing the call keyword arguments (line 290)
        kwargs_15389 = {}
        # Getting the type of 'charset' (line 290)
        charset_15386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 21), 'charset', False)
        # Obtaining the member 'to_splittable' of a type (line 290)
        to_splittable_15387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 21), charset_15386, 'to_splittable')
        # Calling to_splittable(args, kwargs) (line 290)
        to_splittable_call_result_15390 = invoke(stypy.reporting.localization.Localization(__file__, 290, 21), to_splittable_15387, *[s_15388], **kwargs_15389)
        
        # Assigning a type to the variable 'splittable' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'splittable', to_splittable_call_result_15390)
        
        # Assigning a Call to a Name (line 291):
        
        # Assigning a Call to a Name (line 291):
        
        # Call to from_splittable(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'splittable' (line 291)
        splittable_15393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 42), 'splittable', False)
        # Getting the type of 'True' (line 291)
        True_15394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 54), 'True', False)
        # Processing the call keyword arguments (line 291)
        kwargs_15395 = {}
        # Getting the type of 'charset' (line 291)
        charset_15391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), 'charset', False)
        # Obtaining the member 'from_splittable' of a type (line 291)
        from_splittable_15392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 18), charset_15391, 'from_splittable')
        # Calling from_splittable(args, kwargs) (line 291)
        from_splittable_call_result_15396 = invoke(stypy.reporting.localization.Localization(__file__, 291, 18), from_splittable_15392, *[splittable_15393, True_15394], **kwargs_15395)
        
        # Assigning a type to the variable 'encoded' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'encoded', from_splittable_call_result_15396)
        
        # Assigning a Call to a Name (line 292):
        
        # Assigning a Call to a Name (line 292):
        
        # Call to encoded_header_len(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'encoded' (line 292)
        encoded_15399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 42), 'encoded', False)
        # Processing the call keyword arguments (line 292)
        kwargs_15400 = {}
        # Getting the type of 'charset' (line 292)
        charset_15397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'charset', False)
        # Obtaining the member 'encoded_header_len' of a type (line 292)
        encoded_header_len_15398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 15), charset_15397, 'encoded_header_len')
        # Calling encoded_header_len(args, kwargs) (line 292)
        encoded_header_len_call_result_15401 = invoke(stypy.reporting.localization.Localization(__file__, 292, 15), encoded_header_len_15398, *[encoded_15399], **kwargs_15400)
        
        # Assigning a type to the variable 'elen' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'elen', encoded_header_len_call_result_15401)
        
        # Getting the type of 'elen' (line 294)
        elen_15402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 11), 'elen')
        # Getting the type of 'maxlinelen' (line 294)
        maxlinelen_15403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'maxlinelen')
        # Applying the binary operator '<=' (line 294)
        result_le_15404 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 11), '<=', elen_15402, maxlinelen_15403)
        
        # Testing if the type of an if condition is none (line 294)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 294, 8), result_le_15404):
            pass
        else:
            
            # Testing the type of an if condition (line 294)
            if_condition_15405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 8), result_le_15404)
            # Assigning a type to the variable 'if_condition_15405' (line 294)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'if_condition_15405', if_condition_15405)
            # SSA begins for if statement (line 294)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'list' (line 295)
            list_15406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 295)
            # Adding element type (line 295)
            
            # Obtaining an instance of the builtin type 'tuple' (line 295)
            tuple_15407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 295)
            # Adding element type (line 295)
            # Getting the type of 'encoded' (line 295)
            encoded_15408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 'encoded')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 21), tuple_15407, encoded_15408)
            # Adding element type (line 295)
            # Getting the type of 'charset' (line 295)
            charset_15409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 30), 'charset')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 21), tuple_15407, charset_15409)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 19), list_15406, tuple_15407)
            
            # Assigning a type to the variable 'stypy_return_type' (line 295)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'stypy_return_type', list_15406)
            # SSA join for if statement (line 294)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'charset' (line 302)
        charset_15410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 11), 'charset')
        str_15411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 22), 'str', '8bit')
        # Applying the binary operator '==' (line 302)
        result_eq_15412 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 11), '==', charset_15410, str_15411)
        
        # Testing if the type of an if condition is none (line 302)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 302, 8), result_eq_15412):
            
            # Getting the type of 'charset' (line 315)
            charset_15418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'charset')
            str_15419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 24), 'str', 'us-ascii')
            # Applying the binary operator '==' (line 315)
            result_eq_15420 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 13), '==', charset_15418, str_15419)
            
            # Testing if the type of an if condition is none (line 315)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 315, 13), result_eq_15420):
                
                # Getting the type of 'elen' (line 318)
                elen_15430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'elen')
                
                # Call to len(...): (line 318)
                # Processing the call arguments (line 318)
                # Getting the type of 's' (line 318)
                s_15432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 's', False)
                # Processing the call keyword arguments (line 318)
                kwargs_15433 = {}
                # Getting the type of 'len' (line 318)
                len_15431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'len', False)
                # Calling len(args, kwargs) (line 318)
                len_call_result_15434 = invoke(stypy.reporting.localization.Localization(__file__, 318, 21), len_15431, *[s_15432], **kwargs_15433)
                
                # Applying the binary operator '==' (line 318)
                result_eq_15435 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 13), '==', elen_15430, len_call_result_15434)
                
                # Testing if the type of an if condition is none (line 318)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 318, 13), result_eq_15435):
                    
                    # Assigning a Call to a Tuple (line 326):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to _binsplit(...): (line 326)
                    # Processing the call arguments (line 326)
                    # Getting the type of 'splittable' (line 326)
                    splittable_15459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'splittable', False)
                    # Getting the type of 'charset' (line 326)
                    charset_15460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'charset', False)
                    # Getting the type of 'maxlinelen' (line 326)
                    maxlinelen_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 57), 'maxlinelen', False)
                    # Processing the call keyword arguments (line 326)
                    kwargs_15462 = {}
                    # Getting the type of '_binsplit' (line 326)
                    _binsplit_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), '_binsplit', False)
                    # Calling _binsplit(args, kwargs) (line 326)
                    _binsplit_call_result_15463 = invoke(stypy.reporting.localization.Localization(__file__, 326, 26), _binsplit_15458, *[splittable_15459, charset_15460, maxlinelen_15461], **kwargs_15462)
                    
                    # Assigning a type to the variable 'call_assignment_14846' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', _binsplit_call_result_15463)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15465 = stypy_get_value_from_tuple(call_assignment_14846_15464, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_14847' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847', stypy_get_value_from_tuple_call_result_15465)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14847' (line 326)
                    call_assignment_14847_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847')
                    # Assigning a type to the variable 'first' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'first', call_assignment_14847_15466)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15468 = stypy_get_value_from_tuple(call_assignment_14846_15467, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_14848' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848', stypy_get_value_from_tuple_call_result_15468)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14848' (line 326)
                    call_assignment_14848_15469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848')
                    # Assigning a type to the variable 'last' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'last', call_assignment_14848_15469)
                else:
                    
                    # Testing the type of an if condition (line 318)
                    if_condition_15436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 13), result_eq_15435)
                    # Assigning a type to the variable 'if_condition_15436' (line 318)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'if_condition_15436', if_condition_15436)
                    # SSA begins for if statement (line 318)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 321):
                    
                    # Assigning a Name to a Name (line 321):
                    # Getting the type of 'maxlinelen' (line 321)
                    maxlinelen_15437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 23), 'maxlinelen')
                    # Assigning a type to the variable 'splitpnt' (line 321)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'splitpnt', maxlinelen_15437)
                    
                    # Assigning a Call to a Name (line 322):
                    
                    # Assigning a Call to a Name (line 322):
                    
                    # Call to from_splittable(...): (line 322)
                    # Processing the call arguments (line 322)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'splitpnt' (line 322)
                    splitpnt_15440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 56), 'splitpnt', False)
                    slice_15441 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 322, 44), None, splitpnt_15440, None)
                    # Getting the type of 'splittable' (line 322)
                    splittable_15442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 44), 'splittable', False)
                    # Obtaining the member '__getitem__' of a type (line 322)
                    getitem___15443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 44), splittable_15442, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
                    subscript_call_result_15444 = invoke(stypy.reporting.localization.Localization(__file__, 322, 44), getitem___15443, slice_15441)
                    
                    # Getting the type of 'False' (line 322)
                    False_15445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 67), 'False', False)
                    # Processing the call keyword arguments (line 322)
                    kwargs_15446 = {}
                    # Getting the type of 'charset' (line 322)
                    charset_15438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'charset', False)
                    # Obtaining the member 'from_splittable' of a type (line 322)
                    from_splittable_15439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 20), charset_15438, 'from_splittable')
                    # Calling from_splittable(args, kwargs) (line 322)
                    from_splittable_call_result_15447 = invoke(stypy.reporting.localization.Localization(__file__, 322, 20), from_splittable_15439, *[subscript_call_result_15444, False_15445], **kwargs_15446)
                    
                    # Assigning a type to the variable 'first' (line 322)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'first', from_splittable_call_result_15447)
                    
                    # Assigning a Call to a Name (line 323):
                    
                    # Assigning a Call to a Name (line 323):
                    
                    # Call to from_splittable(...): (line 323)
                    # Processing the call arguments (line 323)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'splitpnt' (line 323)
                    splitpnt_15450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 54), 'splitpnt', False)
                    slice_15451 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 323, 43), splitpnt_15450, None, None)
                    # Getting the type of 'splittable' (line 323)
                    splittable_15452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 43), 'splittable', False)
                    # Obtaining the member '__getitem__' of a type (line 323)
                    getitem___15453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 43), splittable_15452, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
                    subscript_call_result_15454 = invoke(stypy.reporting.localization.Localization(__file__, 323, 43), getitem___15453, slice_15451)
                    
                    # Getting the type of 'False' (line 323)
                    False_15455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 66), 'False', False)
                    # Processing the call keyword arguments (line 323)
                    kwargs_15456 = {}
                    # Getting the type of 'charset' (line 323)
                    charset_15448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'charset', False)
                    # Obtaining the member 'from_splittable' of a type (line 323)
                    from_splittable_15449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 19), charset_15448, 'from_splittable')
                    # Calling from_splittable(args, kwargs) (line 323)
                    from_splittable_call_result_15457 = invoke(stypy.reporting.localization.Localization(__file__, 323, 19), from_splittable_15449, *[subscript_call_result_15454, False_15455], **kwargs_15456)
                    
                    # Assigning a type to the variable 'last' (line 323)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'last', from_splittable_call_result_15457)
                    # SSA branch for the else part of an if statement (line 318)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Tuple (line 326):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to _binsplit(...): (line 326)
                    # Processing the call arguments (line 326)
                    # Getting the type of 'splittable' (line 326)
                    splittable_15459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'splittable', False)
                    # Getting the type of 'charset' (line 326)
                    charset_15460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'charset', False)
                    # Getting the type of 'maxlinelen' (line 326)
                    maxlinelen_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 57), 'maxlinelen', False)
                    # Processing the call keyword arguments (line 326)
                    kwargs_15462 = {}
                    # Getting the type of '_binsplit' (line 326)
                    _binsplit_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), '_binsplit', False)
                    # Calling _binsplit(args, kwargs) (line 326)
                    _binsplit_call_result_15463 = invoke(stypy.reporting.localization.Localization(__file__, 326, 26), _binsplit_15458, *[splittable_15459, charset_15460, maxlinelen_15461], **kwargs_15462)
                    
                    # Assigning a type to the variable 'call_assignment_14846' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', _binsplit_call_result_15463)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15465 = stypy_get_value_from_tuple(call_assignment_14846_15464, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_14847' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847', stypy_get_value_from_tuple_call_result_15465)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14847' (line 326)
                    call_assignment_14847_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847')
                    # Assigning a type to the variable 'first' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'first', call_assignment_14847_15466)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15468 = stypy_get_value_from_tuple(call_assignment_14846_15467, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_14848' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848', stypy_get_value_from_tuple_call_result_15468)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14848' (line 326)
                    call_assignment_14848_15469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848')
                    # Assigning a type to the variable 'last' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'last', call_assignment_14848_15469)
                    # SSA join for if statement (line 318)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 315)
                if_condition_15421 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 13), result_eq_15420)
                # Assigning a type to the variable 'if_condition_15421' (line 315)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'if_condition_15421', if_condition_15421)
                # SSA begins for if statement (line 315)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to _split_ascii(...): (line 316)
                # Processing the call arguments (line 316)
                # Getting the type of 's' (line 316)
                s_15424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 37), 's', False)
                # Getting the type of 'charset' (line 316)
                charset_15425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 40), 'charset', False)
                # Getting the type of 'maxlinelen' (line 316)
                maxlinelen_15426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 49), 'maxlinelen', False)
                # Getting the type of 'splitchars' (line 316)
                splitchars_15427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 61), 'splitchars', False)
                # Processing the call keyword arguments (line 316)
                kwargs_15428 = {}
                # Getting the type of 'self' (line 316)
                self_15422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 19), 'self', False)
                # Obtaining the member '_split_ascii' of a type (line 316)
                _split_ascii_15423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 19), self_15422, '_split_ascii')
                # Calling _split_ascii(args, kwargs) (line 316)
                _split_ascii_call_result_15429 = invoke(stypy.reporting.localization.Localization(__file__, 316, 19), _split_ascii_15423, *[s_15424, charset_15425, maxlinelen_15426, splitchars_15427], **kwargs_15428)
                
                # Assigning a type to the variable 'stypy_return_type' (line 316)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'stypy_return_type', _split_ascii_call_result_15429)
                # SSA branch for the else part of an if statement (line 315)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'elen' (line 318)
                elen_15430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'elen')
                
                # Call to len(...): (line 318)
                # Processing the call arguments (line 318)
                # Getting the type of 's' (line 318)
                s_15432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 's', False)
                # Processing the call keyword arguments (line 318)
                kwargs_15433 = {}
                # Getting the type of 'len' (line 318)
                len_15431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'len', False)
                # Calling len(args, kwargs) (line 318)
                len_call_result_15434 = invoke(stypy.reporting.localization.Localization(__file__, 318, 21), len_15431, *[s_15432], **kwargs_15433)
                
                # Applying the binary operator '==' (line 318)
                result_eq_15435 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 13), '==', elen_15430, len_call_result_15434)
                
                # Testing if the type of an if condition is none (line 318)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 318, 13), result_eq_15435):
                    
                    # Assigning a Call to a Tuple (line 326):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to _binsplit(...): (line 326)
                    # Processing the call arguments (line 326)
                    # Getting the type of 'splittable' (line 326)
                    splittable_15459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'splittable', False)
                    # Getting the type of 'charset' (line 326)
                    charset_15460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'charset', False)
                    # Getting the type of 'maxlinelen' (line 326)
                    maxlinelen_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 57), 'maxlinelen', False)
                    # Processing the call keyword arguments (line 326)
                    kwargs_15462 = {}
                    # Getting the type of '_binsplit' (line 326)
                    _binsplit_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), '_binsplit', False)
                    # Calling _binsplit(args, kwargs) (line 326)
                    _binsplit_call_result_15463 = invoke(stypy.reporting.localization.Localization(__file__, 326, 26), _binsplit_15458, *[splittable_15459, charset_15460, maxlinelen_15461], **kwargs_15462)
                    
                    # Assigning a type to the variable 'call_assignment_14846' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', _binsplit_call_result_15463)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15465 = stypy_get_value_from_tuple(call_assignment_14846_15464, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_14847' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847', stypy_get_value_from_tuple_call_result_15465)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14847' (line 326)
                    call_assignment_14847_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847')
                    # Assigning a type to the variable 'first' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'first', call_assignment_14847_15466)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15468 = stypy_get_value_from_tuple(call_assignment_14846_15467, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_14848' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848', stypy_get_value_from_tuple_call_result_15468)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14848' (line 326)
                    call_assignment_14848_15469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848')
                    # Assigning a type to the variable 'last' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'last', call_assignment_14848_15469)
                else:
                    
                    # Testing the type of an if condition (line 318)
                    if_condition_15436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 13), result_eq_15435)
                    # Assigning a type to the variable 'if_condition_15436' (line 318)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'if_condition_15436', if_condition_15436)
                    # SSA begins for if statement (line 318)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 321):
                    
                    # Assigning a Name to a Name (line 321):
                    # Getting the type of 'maxlinelen' (line 321)
                    maxlinelen_15437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 23), 'maxlinelen')
                    # Assigning a type to the variable 'splitpnt' (line 321)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'splitpnt', maxlinelen_15437)
                    
                    # Assigning a Call to a Name (line 322):
                    
                    # Assigning a Call to a Name (line 322):
                    
                    # Call to from_splittable(...): (line 322)
                    # Processing the call arguments (line 322)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'splitpnt' (line 322)
                    splitpnt_15440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 56), 'splitpnt', False)
                    slice_15441 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 322, 44), None, splitpnt_15440, None)
                    # Getting the type of 'splittable' (line 322)
                    splittable_15442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 44), 'splittable', False)
                    # Obtaining the member '__getitem__' of a type (line 322)
                    getitem___15443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 44), splittable_15442, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
                    subscript_call_result_15444 = invoke(stypy.reporting.localization.Localization(__file__, 322, 44), getitem___15443, slice_15441)
                    
                    # Getting the type of 'False' (line 322)
                    False_15445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 67), 'False', False)
                    # Processing the call keyword arguments (line 322)
                    kwargs_15446 = {}
                    # Getting the type of 'charset' (line 322)
                    charset_15438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'charset', False)
                    # Obtaining the member 'from_splittable' of a type (line 322)
                    from_splittable_15439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 20), charset_15438, 'from_splittable')
                    # Calling from_splittable(args, kwargs) (line 322)
                    from_splittable_call_result_15447 = invoke(stypy.reporting.localization.Localization(__file__, 322, 20), from_splittable_15439, *[subscript_call_result_15444, False_15445], **kwargs_15446)
                    
                    # Assigning a type to the variable 'first' (line 322)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'first', from_splittable_call_result_15447)
                    
                    # Assigning a Call to a Name (line 323):
                    
                    # Assigning a Call to a Name (line 323):
                    
                    # Call to from_splittable(...): (line 323)
                    # Processing the call arguments (line 323)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'splitpnt' (line 323)
                    splitpnt_15450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 54), 'splitpnt', False)
                    slice_15451 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 323, 43), splitpnt_15450, None, None)
                    # Getting the type of 'splittable' (line 323)
                    splittable_15452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 43), 'splittable', False)
                    # Obtaining the member '__getitem__' of a type (line 323)
                    getitem___15453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 43), splittable_15452, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
                    subscript_call_result_15454 = invoke(stypy.reporting.localization.Localization(__file__, 323, 43), getitem___15453, slice_15451)
                    
                    # Getting the type of 'False' (line 323)
                    False_15455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 66), 'False', False)
                    # Processing the call keyword arguments (line 323)
                    kwargs_15456 = {}
                    # Getting the type of 'charset' (line 323)
                    charset_15448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'charset', False)
                    # Obtaining the member 'from_splittable' of a type (line 323)
                    from_splittable_15449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 19), charset_15448, 'from_splittable')
                    # Calling from_splittable(args, kwargs) (line 323)
                    from_splittable_call_result_15457 = invoke(stypy.reporting.localization.Localization(__file__, 323, 19), from_splittable_15449, *[subscript_call_result_15454, False_15455], **kwargs_15456)
                    
                    # Assigning a type to the variable 'last' (line 323)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'last', from_splittable_call_result_15457)
                    # SSA branch for the else part of an if statement (line 318)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Tuple (line 326):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to _binsplit(...): (line 326)
                    # Processing the call arguments (line 326)
                    # Getting the type of 'splittable' (line 326)
                    splittable_15459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'splittable', False)
                    # Getting the type of 'charset' (line 326)
                    charset_15460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'charset', False)
                    # Getting the type of 'maxlinelen' (line 326)
                    maxlinelen_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 57), 'maxlinelen', False)
                    # Processing the call keyword arguments (line 326)
                    kwargs_15462 = {}
                    # Getting the type of '_binsplit' (line 326)
                    _binsplit_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), '_binsplit', False)
                    # Calling _binsplit(args, kwargs) (line 326)
                    _binsplit_call_result_15463 = invoke(stypy.reporting.localization.Localization(__file__, 326, 26), _binsplit_15458, *[splittable_15459, charset_15460, maxlinelen_15461], **kwargs_15462)
                    
                    # Assigning a type to the variable 'call_assignment_14846' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', _binsplit_call_result_15463)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15465 = stypy_get_value_from_tuple(call_assignment_14846_15464, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_14847' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847', stypy_get_value_from_tuple_call_result_15465)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14847' (line 326)
                    call_assignment_14847_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847')
                    # Assigning a type to the variable 'first' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'first', call_assignment_14847_15466)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15468 = stypy_get_value_from_tuple(call_assignment_14846_15467, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_14848' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848', stypy_get_value_from_tuple_call_result_15468)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14848' (line 326)
                    call_assignment_14848_15469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848')
                    # Assigning a type to the variable 'last' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'last', call_assignment_14848_15469)
                    # SSA join for if statement (line 318)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 315)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 302)
            if_condition_15413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 8), result_eq_15412)
            # Assigning a type to the variable 'if_condition_15413' (line 302)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'if_condition_15413', if_condition_15413)
            # SSA begins for if statement (line 302)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'list' (line 303)
            list_15414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 303)
            # Adding element type (line 303)
            
            # Obtaining an instance of the builtin type 'tuple' (line 303)
            tuple_15415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 303)
            # Adding element type (line 303)
            # Getting the type of 's' (line 303)
            s_15416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 21), 's')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), tuple_15415, s_15416)
            # Adding element type (line 303)
            # Getting the type of 'charset' (line 303)
            charset_15417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'charset')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), tuple_15415, charset_15417)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 19), list_15414, tuple_15415)
            
            # Assigning a type to the variable 'stypy_return_type' (line 303)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'stypy_return_type', list_15414)
            # SSA branch for the else part of an if statement (line 302)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'charset' (line 315)
            charset_15418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'charset')
            str_15419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 24), 'str', 'us-ascii')
            # Applying the binary operator '==' (line 315)
            result_eq_15420 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 13), '==', charset_15418, str_15419)
            
            # Testing if the type of an if condition is none (line 315)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 315, 13), result_eq_15420):
                
                # Getting the type of 'elen' (line 318)
                elen_15430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'elen')
                
                # Call to len(...): (line 318)
                # Processing the call arguments (line 318)
                # Getting the type of 's' (line 318)
                s_15432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 's', False)
                # Processing the call keyword arguments (line 318)
                kwargs_15433 = {}
                # Getting the type of 'len' (line 318)
                len_15431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'len', False)
                # Calling len(args, kwargs) (line 318)
                len_call_result_15434 = invoke(stypy.reporting.localization.Localization(__file__, 318, 21), len_15431, *[s_15432], **kwargs_15433)
                
                # Applying the binary operator '==' (line 318)
                result_eq_15435 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 13), '==', elen_15430, len_call_result_15434)
                
                # Testing if the type of an if condition is none (line 318)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 318, 13), result_eq_15435):
                    
                    # Assigning a Call to a Tuple (line 326):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to _binsplit(...): (line 326)
                    # Processing the call arguments (line 326)
                    # Getting the type of 'splittable' (line 326)
                    splittable_15459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'splittable', False)
                    # Getting the type of 'charset' (line 326)
                    charset_15460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'charset', False)
                    # Getting the type of 'maxlinelen' (line 326)
                    maxlinelen_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 57), 'maxlinelen', False)
                    # Processing the call keyword arguments (line 326)
                    kwargs_15462 = {}
                    # Getting the type of '_binsplit' (line 326)
                    _binsplit_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), '_binsplit', False)
                    # Calling _binsplit(args, kwargs) (line 326)
                    _binsplit_call_result_15463 = invoke(stypy.reporting.localization.Localization(__file__, 326, 26), _binsplit_15458, *[splittable_15459, charset_15460, maxlinelen_15461], **kwargs_15462)
                    
                    # Assigning a type to the variable 'call_assignment_14846' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', _binsplit_call_result_15463)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15465 = stypy_get_value_from_tuple(call_assignment_14846_15464, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_14847' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847', stypy_get_value_from_tuple_call_result_15465)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14847' (line 326)
                    call_assignment_14847_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847')
                    # Assigning a type to the variable 'first' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'first', call_assignment_14847_15466)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15468 = stypy_get_value_from_tuple(call_assignment_14846_15467, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_14848' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848', stypy_get_value_from_tuple_call_result_15468)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14848' (line 326)
                    call_assignment_14848_15469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848')
                    # Assigning a type to the variable 'last' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'last', call_assignment_14848_15469)
                else:
                    
                    # Testing the type of an if condition (line 318)
                    if_condition_15436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 13), result_eq_15435)
                    # Assigning a type to the variable 'if_condition_15436' (line 318)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'if_condition_15436', if_condition_15436)
                    # SSA begins for if statement (line 318)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 321):
                    
                    # Assigning a Name to a Name (line 321):
                    # Getting the type of 'maxlinelen' (line 321)
                    maxlinelen_15437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 23), 'maxlinelen')
                    # Assigning a type to the variable 'splitpnt' (line 321)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'splitpnt', maxlinelen_15437)
                    
                    # Assigning a Call to a Name (line 322):
                    
                    # Assigning a Call to a Name (line 322):
                    
                    # Call to from_splittable(...): (line 322)
                    # Processing the call arguments (line 322)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'splitpnt' (line 322)
                    splitpnt_15440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 56), 'splitpnt', False)
                    slice_15441 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 322, 44), None, splitpnt_15440, None)
                    # Getting the type of 'splittable' (line 322)
                    splittable_15442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 44), 'splittable', False)
                    # Obtaining the member '__getitem__' of a type (line 322)
                    getitem___15443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 44), splittable_15442, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
                    subscript_call_result_15444 = invoke(stypy.reporting.localization.Localization(__file__, 322, 44), getitem___15443, slice_15441)
                    
                    # Getting the type of 'False' (line 322)
                    False_15445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 67), 'False', False)
                    # Processing the call keyword arguments (line 322)
                    kwargs_15446 = {}
                    # Getting the type of 'charset' (line 322)
                    charset_15438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'charset', False)
                    # Obtaining the member 'from_splittable' of a type (line 322)
                    from_splittable_15439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 20), charset_15438, 'from_splittable')
                    # Calling from_splittable(args, kwargs) (line 322)
                    from_splittable_call_result_15447 = invoke(stypy.reporting.localization.Localization(__file__, 322, 20), from_splittable_15439, *[subscript_call_result_15444, False_15445], **kwargs_15446)
                    
                    # Assigning a type to the variable 'first' (line 322)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'first', from_splittable_call_result_15447)
                    
                    # Assigning a Call to a Name (line 323):
                    
                    # Assigning a Call to a Name (line 323):
                    
                    # Call to from_splittable(...): (line 323)
                    # Processing the call arguments (line 323)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'splitpnt' (line 323)
                    splitpnt_15450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 54), 'splitpnt', False)
                    slice_15451 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 323, 43), splitpnt_15450, None, None)
                    # Getting the type of 'splittable' (line 323)
                    splittable_15452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 43), 'splittable', False)
                    # Obtaining the member '__getitem__' of a type (line 323)
                    getitem___15453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 43), splittable_15452, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
                    subscript_call_result_15454 = invoke(stypy.reporting.localization.Localization(__file__, 323, 43), getitem___15453, slice_15451)
                    
                    # Getting the type of 'False' (line 323)
                    False_15455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 66), 'False', False)
                    # Processing the call keyword arguments (line 323)
                    kwargs_15456 = {}
                    # Getting the type of 'charset' (line 323)
                    charset_15448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'charset', False)
                    # Obtaining the member 'from_splittable' of a type (line 323)
                    from_splittable_15449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 19), charset_15448, 'from_splittable')
                    # Calling from_splittable(args, kwargs) (line 323)
                    from_splittable_call_result_15457 = invoke(stypy.reporting.localization.Localization(__file__, 323, 19), from_splittable_15449, *[subscript_call_result_15454, False_15455], **kwargs_15456)
                    
                    # Assigning a type to the variable 'last' (line 323)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'last', from_splittable_call_result_15457)
                    # SSA branch for the else part of an if statement (line 318)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Tuple (line 326):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to _binsplit(...): (line 326)
                    # Processing the call arguments (line 326)
                    # Getting the type of 'splittable' (line 326)
                    splittable_15459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'splittable', False)
                    # Getting the type of 'charset' (line 326)
                    charset_15460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'charset', False)
                    # Getting the type of 'maxlinelen' (line 326)
                    maxlinelen_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 57), 'maxlinelen', False)
                    # Processing the call keyword arguments (line 326)
                    kwargs_15462 = {}
                    # Getting the type of '_binsplit' (line 326)
                    _binsplit_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), '_binsplit', False)
                    # Calling _binsplit(args, kwargs) (line 326)
                    _binsplit_call_result_15463 = invoke(stypy.reporting.localization.Localization(__file__, 326, 26), _binsplit_15458, *[splittable_15459, charset_15460, maxlinelen_15461], **kwargs_15462)
                    
                    # Assigning a type to the variable 'call_assignment_14846' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', _binsplit_call_result_15463)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15465 = stypy_get_value_from_tuple(call_assignment_14846_15464, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_14847' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847', stypy_get_value_from_tuple_call_result_15465)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14847' (line 326)
                    call_assignment_14847_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847')
                    # Assigning a type to the variable 'first' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'first', call_assignment_14847_15466)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15468 = stypy_get_value_from_tuple(call_assignment_14846_15467, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_14848' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848', stypy_get_value_from_tuple_call_result_15468)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14848' (line 326)
                    call_assignment_14848_15469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848')
                    # Assigning a type to the variable 'last' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'last', call_assignment_14848_15469)
                    # SSA join for if statement (line 318)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 315)
                if_condition_15421 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 13), result_eq_15420)
                # Assigning a type to the variable 'if_condition_15421' (line 315)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'if_condition_15421', if_condition_15421)
                # SSA begins for if statement (line 315)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to _split_ascii(...): (line 316)
                # Processing the call arguments (line 316)
                # Getting the type of 's' (line 316)
                s_15424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 37), 's', False)
                # Getting the type of 'charset' (line 316)
                charset_15425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 40), 'charset', False)
                # Getting the type of 'maxlinelen' (line 316)
                maxlinelen_15426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 49), 'maxlinelen', False)
                # Getting the type of 'splitchars' (line 316)
                splitchars_15427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 61), 'splitchars', False)
                # Processing the call keyword arguments (line 316)
                kwargs_15428 = {}
                # Getting the type of 'self' (line 316)
                self_15422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 19), 'self', False)
                # Obtaining the member '_split_ascii' of a type (line 316)
                _split_ascii_15423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 19), self_15422, '_split_ascii')
                # Calling _split_ascii(args, kwargs) (line 316)
                _split_ascii_call_result_15429 = invoke(stypy.reporting.localization.Localization(__file__, 316, 19), _split_ascii_15423, *[s_15424, charset_15425, maxlinelen_15426, splitchars_15427], **kwargs_15428)
                
                # Assigning a type to the variable 'stypy_return_type' (line 316)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'stypy_return_type', _split_ascii_call_result_15429)
                # SSA branch for the else part of an if statement (line 315)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'elen' (line 318)
                elen_15430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'elen')
                
                # Call to len(...): (line 318)
                # Processing the call arguments (line 318)
                # Getting the type of 's' (line 318)
                s_15432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 's', False)
                # Processing the call keyword arguments (line 318)
                kwargs_15433 = {}
                # Getting the type of 'len' (line 318)
                len_15431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'len', False)
                # Calling len(args, kwargs) (line 318)
                len_call_result_15434 = invoke(stypy.reporting.localization.Localization(__file__, 318, 21), len_15431, *[s_15432], **kwargs_15433)
                
                # Applying the binary operator '==' (line 318)
                result_eq_15435 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 13), '==', elen_15430, len_call_result_15434)
                
                # Testing if the type of an if condition is none (line 318)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 318, 13), result_eq_15435):
                    
                    # Assigning a Call to a Tuple (line 326):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to _binsplit(...): (line 326)
                    # Processing the call arguments (line 326)
                    # Getting the type of 'splittable' (line 326)
                    splittable_15459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'splittable', False)
                    # Getting the type of 'charset' (line 326)
                    charset_15460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'charset', False)
                    # Getting the type of 'maxlinelen' (line 326)
                    maxlinelen_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 57), 'maxlinelen', False)
                    # Processing the call keyword arguments (line 326)
                    kwargs_15462 = {}
                    # Getting the type of '_binsplit' (line 326)
                    _binsplit_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), '_binsplit', False)
                    # Calling _binsplit(args, kwargs) (line 326)
                    _binsplit_call_result_15463 = invoke(stypy.reporting.localization.Localization(__file__, 326, 26), _binsplit_15458, *[splittable_15459, charset_15460, maxlinelen_15461], **kwargs_15462)
                    
                    # Assigning a type to the variable 'call_assignment_14846' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', _binsplit_call_result_15463)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15465 = stypy_get_value_from_tuple(call_assignment_14846_15464, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_14847' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847', stypy_get_value_from_tuple_call_result_15465)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14847' (line 326)
                    call_assignment_14847_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847')
                    # Assigning a type to the variable 'first' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'first', call_assignment_14847_15466)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15468 = stypy_get_value_from_tuple(call_assignment_14846_15467, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_14848' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848', stypy_get_value_from_tuple_call_result_15468)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14848' (line 326)
                    call_assignment_14848_15469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848')
                    # Assigning a type to the variable 'last' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'last', call_assignment_14848_15469)
                else:
                    
                    # Testing the type of an if condition (line 318)
                    if_condition_15436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 13), result_eq_15435)
                    # Assigning a type to the variable 'if_condition_15436' (line 318)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'if_condition_15436', if_condition_15436)
                    # SSA begins for if statement (line 318)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 321):
                    
                    # Assigning a Name to a Name (line 321):
                    # Getting the type of 'maxlinelen' (line 321)
                    maxlinelen_15437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 23), 'maxlinelen')
                    # Assigning a type to the variable 'splitpnt' (line 321)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'splitpnt', maxlinelen_15437)
                    
                    # Assigning a Call to a Name (line 322):
                    
                    # Assigning a Call to a Name (line 322):
                    
                    # Call to from_splittable(...): (line 322)
                    # Processing the call arguments (line 322)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'splitpnt' (line 322)
                    splitpnt_15440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 56), 'splitpnt', False)
                    slice_15441 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 322, 44), None, splitpnt_15440, None)
                    # Getting the type of 'splittable' (line 322)
                    splittable_15442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 44), 'splittable', False)
                    # Obtaining the member '__getitem__' of a type (line 322)
                    getitem___15443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 44), splittable_15442, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
                    subscript_call_result_15444 = invoke(stypy.reporting.localization.Localization(__file__, 322, 44), getitem___15443, slice_15441)
                    
                    # Getting the type of 'False' (line 322)
                    False_15445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 67), 'False', False)
                    # Processing the call keyword arguments (line 322)
                    kwargs_15446 = {}
                    # Getting the type of 'charset' (line 322)
                    charset_15438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'charset', False)
                    # Obtaining the member 'from_splittable' of a type (line 322)
                    from_splittable_15439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 20), charset_15438, 'from_splittable')
                    # Calling from_splittable(args, kwargs) (line 322)
                    from_splittable_call_result_15447 = invoke(stypy.reporting.localization.Localization(__file__, 322, 20), from_splittable_15439, *[subscript_call_result_15444, False_15445], **kwargs_15446)
                    
                    # Assigning a type to the variable 'first' (line 322)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'first', from_splittable_call_result_15447)
                    
                    # Assigning a Call to a Name (line 323):
                    
                    # Assigning a Call to a Name (line 323):
                    
                    # Call to from_splittable(...): (line 323)
                    # Processing the call arguments (line 323)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'splitpnt' (line 323)
                    splitpnt_15450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 54), 'splitpnt', False)
                    slice_15451 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 323, 43), splitpnt_15450, None, None)
                    # Getting the type of 'splittable' (line 323)
                    splittable_15452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 43), 'splittable', False)
                    # Obtaining the member '__getitem__' of a type (line 323)
                    getitem___15453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 43), splittable_15452, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
                    subscript_call_result_15454 = invoke(stypy.reporting.localization.Localization(__file__, 323, 43), getitem___15453, slice_15451)
                    
                    # Getting the type of 'False' (line 323)
                    False_15455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 66), 'False', False)
                    # Processing the call keyword arguments (line 323)
                    kwargs_15456 = {}
                    # Getting the type of 'charset' (line 323)
                    charset_15448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'charset', False)
                    # Obtaining the member 'from_splittable' of a type (line 323)
                    from_splittable_15449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 19), charset_15448, 'from_splittable')
                    # Calling from_splittable(args, kwargs) (line 323)
                    from_splittable_call_result_15457 = invoke(stypy.reporting.localization.Localization(__file__, 323, 19), from_splittable_15449, *[subscript_call_result_15454, False_15455], **kwargs_15456)
                    
                    # Assigning a type to the variable 'last' (line 323)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'last', from_splittable_call_result_15457)
                    # SSA branch for the else part of an if statement (line 318)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Tuple (line 326):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to _binsplit(...): (line 326)
                    # Processing the call arguments (line 326)
                    # Getting the type of 'splittable' (line 326)
                    splittable_15459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'splittable', False)
                    # Getting the type of 'charset' (line 326)
                    charset_15460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'charset', False)
                    # Getting the type of 'maxlinelen' (line 326)
                    maxlinelen_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 57), 'maxlinelen', False)
                    # Processing the call keyword arguments (line 326)
                    kwargs_15462 = {}
                    # Getting the type of '_binsplit' (line 326)
                    _binsplit_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), '_binsplit', False)
                    # Calling _binsplit(args, kwargs) (line 326)
                    _binsplit_call_result_15463 = invoke(stypy.reporting.localization.Localization(__file__, 326, 26), _binsplit_15458, *[splittable_15459, charset_15460, maxlinelen_15461], **kwargs_15462)
                    
                    # Assigning a type to the variable 'call_assignment_14846' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', _binsplit_call_result_15463)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15465 = stypy_get_value_from_tuple(call_assignment_14846_15464, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_14847' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847', stypy_get_value_from_tuple_call_result_15465)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14847' (line 326)
                    call_assignment_14847_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14847')
                    # Assigning a type to the variable 'first' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'first', call_assignment_14847_15466)
                    
                    # Assigning a Call to a Name (line 326):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_14846' (line 326)
                    call_assignment_14846_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14846', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_15468 = stypy_get_value_from_tuple(call_assignment_14846_15467, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_14848' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848', stypy_get_value_from_tuple_call_result_15468)
                    
                    # Assigning a Name to a Name (line 326):
                    # Getting the type of 'call_assignment_14848' (line 326)
                    call_assignment_14848_15469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'call_assignment_14848')
                    # Assigning a type to the variable 'last' (line 326)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'last', call_assignment_14848_15469)
                    # SSA join for if statement (line 318)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 315)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 302)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 329):
        
        # Assigning a Call to a Name (line 329):
        
        # Call to to_splittable(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'first' (line 329)
        first_15472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 44), 'first', False)
        # Processing the call keyword arguments (line 329)
        kwargs_15473 = {}
        # Getting the type of 'charset' (line 329)
        charset_15470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 22), 'charset', False)
        # Obtaining the member 'to_splittable' of a type (line 329)
        to_splittable_15471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 22), charset_15470, 'to_splittable')
        # Calling to_splittable(args, kwargs) (line 329)
        to_splittable_call_result_15474 = invoke(stypy.reporting.localization.Localization(__file__, 329, 22), to_splittable_15471, *[first_15472], **kwargs_15473)
        
        # Assigning a type to the variable 'fsplittable' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'fsplittable', to_splittable_call_result_15474)
        
        # Assigning a Call to a Name (line 330):
        
        # Assigning a Call to a Name (line 330):
        
        # Call to from_splittable(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'fsplittable' (line 330)
        fsplittable_15477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 43), 'fsplittable', False)
        # Getting the type of 'True' (line 330)
        True_15478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 56), 'True', False)
        # Processing the call keyword arguments (line 330)
        kwargs_15479 = {}
        # Getting the type of 'charset' (line 330)
        charset_15475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'charset', False)
        # Obtaining the member 'from_splittable' of a type (line 330)
        from_splittable_15476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 19), charset_15475, 'from_splittable')
        # Calling from_splittable(args, kwargs) (line 330)
        from_splittable_call_result_15480 = invoke(stypy.reporting.localization.Localization(__file__, 330, 19), from_splittable_15476, *[fsplittable_15477, True_15478], **kwargs_15479)
        
        # Assigning a type to the variable 'fencoded' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'fencoded', from_splittable_call_result_15480)
        
        # Assigning a List to a Name (line 331):
        
        # Assigning a List to a Name (line 331):
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_15481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        # Adding element type (line 331)
        
        # Obtaining an instance of the builtin type 'tuple' (line 331)
        tuple_15482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 331)
        # Adding element type (line 331)
        # Getting the type of 'fencoded' (line 331)
        fencoded_15483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 18), 'fencoded')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 18), tuple_15482, fencoded_15483)
        # Adding element type (line 331)
        # Getting the type of 'charset' (line 331)
        charset_15484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 28), 'charset')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 18), tuple_15482, charset_15484)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 16), list_15481, tuple_15482)
        
        # Assigning a type to the variable 'chunk' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'chunk', list_15481)
        # Getting the type of 'chunk' (line 332)
        chunk_15485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'chunk')
        
        # Call to _split(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'last' (line 332)
        last_15488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 35), 'last', False)
        # Getting the type of 'charset' (line 332)
        charset_15489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 41), 'charset', False)
        # Getting the type of 'self' (line 332)
        self_15490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 50), 'self', False)
        # Obtaining the member '_maxlinelen' of a type (line 332)
        _maxlinelen_15491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 50), self_15490, '_maxlinelen')
        # Getting the type of 'splitchars' (line 332)
        splitchars_15492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 68), 'splitchars', False)
        # Processing the call keyword arguments (line 332)
        kwargs_15493 = {}
        # Getting the type of 'self' (line 332)
        self_15486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 23), 'self', False)
        # Obtaining the member '_split' of a type (line 332)
        _split_15487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 23), self_15486, '_split')
        # Calling _split(args, kwargs) (line 332)
        _split_call_result_15494 = invoke(stypy.reporting.localization.Localization(__file__, 332, 23), _split_15487, *[last_15488, charset_15489, _maxlinelen_15491, splitchars_15492], **kwargs_15493)
        
        # Applying the binary operator '+' (line 332)
        result_add_15495 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 15), '+', chunk_15485, _split_call_result_15494)
        
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'stypy_return_type', result_add_15495)
        
        # ################# End of '_split(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_split' in the type store
        # Getting the type of 'stypy_return_type' (line 288)
        stypy_return_type_15496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15496)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_split'
        return stypy_return_type_15496


    @norecursion
    def _split_ascii(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_split_ascii'
        module_type_store = module_type_store.open_function_context('_split_ascii', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Header._split_ascii.__dict__.__setitem__('stypy_localization', localization)
        Header._split_ascii.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Header._split_ascii.__dict__.__setitem__('stypy_type_store', module_type_store)
        Header._split_ascii.__dict__.__setitem__('stypy_function_name', 'Header._split_ascii')
        Header._split_ascii.__dict__.__setitem__('stypy_param_names_list', ['s', 'charset', 'firstlen', 'splitchars'])
        Header._split_ascii.__dict__.__setitem__('stypy_varargs_param_name', None)
        Header._split_ascii.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Header._split_ascii.__dict__.__setitem__('stypy_call_defaults', defaults)
        Header._split_ascii.__dict__.__setitem__('stypy_call_varargs', varargs)
        Header._split_ascii.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Header._split_ascii.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Header._split_ascii', ['s', 'charset', 'firstlen', 'splitchars'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_split_ascii', localization, ['s', 'charset', 'firstlen', 'splitchars'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_split_ascii(...)' code ##################

        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to _split_ascii(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 's' (line 335)
        s_15498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 30), 's', False)
        # Getting the type of 'firstlen' (line 335)
        firstlen_15499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 33), 'firstlen', False)
        # Getting the type of 'self' (line 335)
        self_15500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 43), 'self', False)
        # Obtaining the member '_maxlinelen' of a type (line 335)
        _maxlinelen_15501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 43), self_15500, '_maxlinelen')
        # Getting the type of 'self' (line 336)
        self_15502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 30), 'self', False)
        # Obtaining the member '_continuation_ws' of a type (line 336)
        _continuation_ws_15503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 30), self_15502, '_continuation_ws')
        # Getting the type of 'splitchars' (line 336)
        splitchars_15504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 53), 'splitchars', False)
        # Processing the call keyword arguments (line 335)
        kwargs_15505 = {}
        # Getting the type of '_split_ascii' (line 335)
        _split_ascii_15497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), '_split_ascii', False)
        # Calling _split_ascii(args, kwargs) (line 335)
        _split_ascii_call_result_15506 = invoke(stypy.reporting.localization.Localization(__file__, 335, 17), _split_ascii_15497, *[s_15498, firstlen_15499, _maxlinelen_15501, _continuation_ws_15503, splitchars_15504], **kwargs_15505)
        
        # Assigning a type to the variable 'chunks' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'chunks', _split_ascii_call_result_15506)
        
        # Call to zip(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'chunks' (line 337)
        chunks_15508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 19), 'chunks', False)
        
        # Obtaining an instance of the builtin type 'list' (line 337)
        list_15509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 337)
        # Adding element type (line 337)
        # Getting the type of 'charset' (line 337)
        charset_15510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 28), 'charset', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 27), list_15509, charset_15510)
        
        
        # Call to len(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'chunks' (line 337)
        chunks_15512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 41), 'chunks', False)
        # Processing the call keyword arguments (line 337)
        kwargs_15513 = {}
        # Getting the type of 'len' (line 337)
        len_15511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 37), 'len', False)
        # Calling len(args, kwargs) (line 337)
        len_call_result_15514 = invoke(stypy.reporting.localization.Localization(__file__, 337, 37), len_15511, *[chunks_15512], **kwargs_15513)
        
        # Applying the binary operator '*' (line 337)
        result_mul_15515 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 27), '*', list_15509, len_call_result_15514)
        
        # Processing the call keyword arguments (line 337)
        kwargs_15516 = {}
        # Getting the type of 'zip' (line 337)
        zip_15507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'zip', False)
        # Calling zip(args, kwargs) (line 337)
        zip_call_result_15517 = invoke(stypy.reporting.localization.Localization(__file__, 337, 15), zip_15507, *[chunks_15508, result_mul_15515], **kwargs_15516)
        
        # Assigning a type to the variable 'stypy_return_type' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'stypy_return_type', zip_call_result_15517)
        
        # ################# End of '_split_ascii(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_split_ascii' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_15518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_split_ascii'
        return stypy_return_type_15518


    @norecursion
    def _encode_chunks(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_encode_chunks'
        module_type_store = module_type_store.open_function_context('_encode_chunks', 339, 4, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Header._encode_chunks.__dict__.__setitem__('stypy_localization', localization)
        Header._encode_chunks.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Header._encode_chunks.__dict__.__setitem__('stypy_type_store', module_type_store)
        Header._encode_chunks.__dict__.__setitem__('stypy_function_name', 'Header._encode_chunks')
        Header._encode_chunks.__dict__.__setitem__('stypy_param_names_list', ['newchunks', 'maxlinelen'])
        Header._encode_chunks.__dict__.__setitem__('stypy_varargs_param_name', None)
        Header._encode_chunks.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Header._encode_chunks.__dict__.__setitem__('stypy_call_defaults', defaults)
        Header._encode_chunks.__dict__.__setitem__('stypy_call_varargs', varargs)
        Header._encode_chunks.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Header._encode_chunks.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Header._encode_chunks', ['newchunks', 'maxlinelen'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_encode_chunks', localization, ['newchunks', 'maxlinelen'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_encode_chunks(...)' code ##################

        
        # Assigning a List to a Name (line 357):
        
        # Assigning a List to a Name (line 357):
        
        # Obtaining an instance of the builtin type 'list' (line 357)
        list_15519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 357)
        
        # Assigning a type to the variable 'chunks' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'chunks', list_15519)
        
        # Getting the type of 'newchunks' (line 358)
        newchunks_15520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 31), 'newchunks')
        # Assigning a type to the variable 'newchunks_15520' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'newchunks_15520', newchunks_15520)
        # Testing if the for loop is going to be iterated (line 358)
        # Testing the type of a for loop iterable (line 358)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 358, 8), newchunks_15520)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 358, 8), newchunks_15520):
            # Getting the type of the for loop variable (line 358)
            for_loop_var_15521 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 358, 8), newchunks_15520)
            # Assigning a type to the variable 'header' (line 358)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'header', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 8), for_loop_var_15521, 2, 0))
            # Assigning a type to the variable 'charset' (line 358)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'charset', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 8), for_loop_var_15521, 2, 1))
            # SSA begins for a for statement (line 358)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'header' (line 359)
            header_15522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 19), 'header')
            # Applying the 'not' unary operator (line 359)
            result_not__15523 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 15), 'not', header_15522)
            
            # Testing if the type of an if condition is none (line 359)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 359, 12), result_not__15523):
                pass
            else:
                
                # Testing the type of an if condition (line 359)
                if_condition_15524 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 12), result_not__15523)
                # Assigning a type to the variable 'if_condition_15524' (line 359)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'if_condition_15524', if_condition_15524)
                # SSA begins for if statement (line 359)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 359)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Evaluating a boolean operation
            
            # Getting the type of 'charset' (line 361)
            charset_15525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 15), 'charset')
            # Getting the type of 'None' (line 361)
            None_15526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 26), 'None')
            # Applying the binary operator 'is' (line 361)
            result_is__15527 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 15), 'is', charset_15525, None_15526)
            
            
            # Getting the type of 'charset' (line 361)
            charset_15528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 34), 'charset')
            # Obtaining the member 'header_encoding' of a type (line 361)
            header_encoding_15529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 34), charset_15528, 'header_encoding')
            # Getting the type of 'None' (line 361)
            None_15530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 61), 'None')
            # Applying the binary operator 'is' (line 361)
            result_is__15531 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 34), 'is', header_encoding_15529, None_15530)
            
            # Applying the binary operator 'or' (line 361)
            result_or_keyword_15532 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 15), 'or', result_is__15527, result_is__15531)
            
            # Testing if the type of an if condition is none (line 361)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 361, 12), result_or_keyword_15532):
                
                # Assigning a Call to a Name (line 364):
                
                # Assigning a Call to a Name (line 364):
                
                # Call to header_encode(...): (line 364)
                # Processing the call arguments (line 364)
                # Getting the type of 'header' (line 364)
                header_15537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 42), 'header', False)
                # Processing the call keyword arguments (line 364)
                kwargs_15538 = {}
                # Getting the type of 'charset' (line 364)
                charset_15535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 20), 'charset', False)
                # Obtaining the member 'header_encode' of a type (line 364)
                header_encode_15536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 20), charset_15535, 'header_encode')
                # Calling header_encode(args, kwargs) (line 364)
                header_encode_call_result_15539 = invoke(stypy.reporting.localization.Localization(__file__, 364, 20), header_encode_15536, *[header_15537], **kwargs_15538)
                
                # Assigning a type to the variable 's' (line 364)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 's', header_encode_call_result_15539)
            else:
                
                # Testing the type of an if condition (line 361)
                if_condition_15533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 12), result_or_keyword_15532)
                # Assigning a type to the variable 'if_condition_15533' (line 361)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'if_condition_15533', if_condition_15533)
                # SSA begins for if statement (line 361)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 362):
                
                # Assigning a Name to a Name (line 362):
                # Getting the type of 'header' (line 362)
                header_15534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 20), 'header')
                # Assigning a type to the variable 's' (line 362)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 's', header_15534)
                # SSA branch for the else part of an if statement (line 361)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 364):
                
                # Assigning a Call to a Name (line 364):
                
                # Call to header_encode(...): (line 364)
                # Processing the call arguments (line 364)
                # Getting the type of 'header' (line 364)
                header_15537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 42), 'header', False)
                # Processing the call keyword arguments (line 364)
                kwargs_15538 = {}
                # Getting the type of 'charset' (line 364)
                charset_15535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 20), 'charset', False)
                # Obtaining the member 'header_encode' of a type (line 364)
                header_encode_15536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 20), charset_15535, 'header_encode')
                # Calling header_encode(args, kwargs) (line 364)
                header_encode_call_result_15539 = invoke(stypy.reporting.localization.Localization(__file__, 364, 20), header_encode_15536, *[header_15537], **kwargs_15538)
                
                # Assigning a type to the variable 's' (line 364)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 's', header_encode_call_result_15539)
                # SSA join for if statement (line 361)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Evaluating a boolean operation
            # Getting the type of 'chunks' (line 366)
            chunks_15540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 15), 'chunks')
            
            # Call to endswith(...): (line 366)
            # Processing the call arguments (line 366)
            str_15546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 46), 'str', ' ')
            # Processing the call keyword arguments (line 366)
            kwargs_15547 = {}
            
            # Obtaining the type of the subscript
            int_15541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 33), 'int')
            # Getting the type of 'chunks' (line 366)
            chunks_15542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 26), 'chunks', False)
            # Obtaining the member '__getitem__' of a type (line 366)
            getitem___15543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 26), chunks_15542, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 366)
            subscript_call_result_15544 = invoke(stypy.reporting.localization.Localization(__file__, 366, 26), getitem___15543, int_15541)
            
            # Obtaining the member 'endswith' of a type (line 366)
            endswith_15545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 26), subscript_call_result_15544, 'endswith')
            # Calling endswith(args, kwargs) (line 366)
            endswith_call_result_15548 = invoke(stypy.reporting.localization.Localization(__file__, 366, 26), endswith_15545, *[str_15546], **kwargs_15547)
            
            # Applying the binary operator 'and' (line 366)
            result_and_keyword_15549 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 15), 'and', chunks_15540, endswith_call_result_15548)
            
            # Testing if the type of an if condition is none (line 366)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 366, 12), result_and_keyword_15549):
                
                # Assigning a Str to a Name (line 369):
                
                # Assigning a Str to a Name (line 369):
                str_15552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 24), 'str', ' ')
                # Assigning a type to the variable 'extra' (line 369)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'extra', str_15552)
            else:
                
                # Testing the type of an if condition (line 366)
                if_condition_15550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 12), result_and_keyword_15549)
                # Assigning a type to the variable 'if_condition_15550' (line 366)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'if_condition_15550', if_condition_15550)
                # SSA begins for if statement (line 366)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Str to a Name (line 367):
                
                # Assigning a Str to a Name (line 367):
                str_15551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 24), 'str', '')
                # Assigning a type to the variable 'extra' (line 367)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'extra', str_15551)
                # SSA branch for the else part of an if statement (line 366)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Str to a Name (line 369):
                
                # Assigning a Str to a Name (line 369):
                str_15552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 24), 'str', ' ')
                # Assigning a type to the variable 'extra' (line 369)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'extra', str_15552)
                # SSA join for if statement (line 366)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to _max_append(...): (line 370)
            # Processing the call arguments (line 370)
            # Getting the type of 'chunks' (line 370)
            chunks_15554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'chunks', False)
            # Getting the type of 's' (line 370)
            s_15555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 32), 's', False)
            # Getting the type of 'maxlinelen' (line 370)
            maxlinelen_15556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 35), 'maxlinelen', False)
            # Getting the type of 'extra' (line 370)
            extra_15557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 47), 'extra', False)
            # Processing the call keyword arguments (line 370)
            kwargs_15558 = {}
            # Getting the type of '_max_append' (line 370)
            _max_append_15553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), '_max_append', False)
            # Calling _max_append(args, kwargs) (line 370)
            _max_append_call_result_15559 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), _max_append_15553, *[chunks_15554, s_15555, maxlinelen_15556, extra_15557], **kwargs_15558)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a BinOp to a Name (line 371):
        
        # Assigning a BinOp to a Name (line 371):
        # Getting the type of 'NL' (line 371)
        NL_15560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 17), 'NL')
        # Getting the type of 'self' (line 371)
        self_15561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 22), 'self')
        # Obtaining the member '_continuation_ws' of a type (line 371)
        _continuation_ws_15562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 22), self_15561, '_continuation_ws')
        # Applying the binary operator '+' (line 371)
        result_add_15563 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 17), '+', NL_15560, _continuation_ws_15562)
        
        # Assigning a type to the variable 'joiner' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'joiner', result_add_15563)
        
        # Call to join(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'chunks' (line 372)
        chunks_15566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 27), 'chunks', False)
        # Processing the call keyword arguments (line 372)
        kwargs_15567 = {}
        # Getting the type of 'joiner' (line 372)
        joiner_15564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 15), 'joiner', False)
        # Obtaining the member 'join' of a type (line 372)
        join_15565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 15), joiner_15564, 'join')
        # Calling join(args, kwargs) (line 372)
        join_call_result_15568 = invoke(stypy.reporting.localization.Localization(__file__, 372, 15), join_15565, *[chunks_15566], **kwargs_15567)
        
        # Assigning a type to the variable 'stypy_return_type' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'stypy_return_type', join_call_result_15568)
        
        # ################# End of '_encode_chunks(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_encode_chunks' in the type store
        # Getting the type of 'stypy_return_type' (line 339)
        stypy_return_type_15569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15569)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_encode_chunks'
        return stypy_return_type_15569


    @norecursion
    def encode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_15570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 32), 'str', ';, ')
        defaults = [str_15570]
        # Create a new context for function 'encode'
        module_type_store = module_type_store.open_function_context('encode', 374, 4, False)
        # Assigning a type to the variable 'self' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Header.encode.__dict__.__setitem__('stypy_localization', localization)
        Header.encode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Header.encode.__dict__.__setitem__('stypy_type_store', module_type_store)
        Header.encode.__dict__.__setitem__('stypy_function_name', 'Header.encode')
        Header.encode.__dict__.__setitem__('stypy_param_names_list', ['splitchars'])
        Header.encode.__dict__.__setitem__('stypy_varargs_param_name', None)
        Header.encode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Header.encode.__dict__.__setitem__('stypy_call_defaults', defaults)
        Header.encode.__dict__.__setitem__('stypy_call_varargs', varargs)
        Header.encode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Header.encode.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Header.encode', ['splitchars'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'encode', localization, ['splitchars'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'encode(...)' code ##################

        str_15571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, (-1)), 'str', "Encode a message header into an RFC-compliant format.\n\n        There are many issues involved in converting a given string for use in\n        an email header.  Only certain character sets are readable in most\n        email clients, and as header strings can only contain a subset of\n        7-bit ASCII, care must be taken to properly convert and encode (with\n        Base64 or quoted-printable) header strings.  In addition, there is a\n        75-character length limit on any given encoded header field, so\n        line-wrapping must be performed, even with double-byte character sets.\n\n        This method will do its best to convert the string to the correct\n        character set used in email, and encode and line wrap it safely with\n        the appropriate scheme for that character set.\n\n        If the given charset is not known or an error occurs during\n        conversion, this function will return the header untouched.\n\n        Optional splitchars is a string containing characters to split long\n        ASCII lines on, in rough support of RFC 2822's `highest level\n        syntactic breaks'.  This doesn't affect RFC 2047 encoded lines.\n        ")
        
        # Assigning a List to a Name (line 396):
        
        # Assigning a List to a Name (line 396):
        
        # Obtaining an instance of the builtin type 'list' (line 396)
        list_15572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 396)
        
        # Assigning a type to the variable 'newchunks' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'newchunks', list_15572)
        
        # Assigning a Attribute to a Name (line 397):
        
        # Assigning a Attribute to a Name (line 397):
        # Getting the type of 'self' (line 397)
        self_15573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 21), 'self')
        # Obtaining the member '_firstlinelen' of a type (line 397)
        _firstlinelen_15574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 21), self_15573, '_firstlinelen')
        # Assigning a type to the variable 'maxlinelen' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'maxlinelen', _firstlinelen_15574)
        
        # Assigning a Num to a Name (line 398):
        
        # Assigning a Num to a Name (line 398):
        int_15575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 18), 'int')
        # Assigning a type to the variable 'lastlen' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'lastlen', int_15575)
        
        # Getting the type of 'self' (line 399)
        self_15576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 26), 'self')
        # Obtaining the member '_chunks' of a type (line 399)
        _chunks_15577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 26), self_15576, '_chunks')
        # Assigning a type to the variable '_chunks_15577' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), '_chunks_15577', _chunks_15577)
        # Testing if the for loop is going to be iterated (line 399)
        # Testing the type of a for loop iterable (line 399)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 399, 8), _chunks_15577)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 399, 8), _chunks_15577):
            # Getting the type of the for loop variable (line 399)
            for_loop_var_15578 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 399, 8), _chunks_15577)
            # Assigning a type to the variable 's' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 's', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 8), for_loop_var_15578, 2, 0))
            # Assigning a type to the variable 'charset' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'charset', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 8), for_loop_var_15578, 2, 1))
            # SSA begins for a for statement (line 399)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 403):
            
            # Assigning a BinOp to a Name (line 403):
            # Getting the type of 'maxlinelen' (line 403)
            maxlinelen_15579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 24), 'maxlinelen')
            # Getting the type of 'lastlen' (line 403)
            lastlen_15580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 37), 'lastlen')
            # Applying the binary operator '-' (line 403)
            result_sub_15581 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 24), '-', maxlinelen_15579, lastlen_15580)
            
            int_15582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 47), 'int')
            # Applying the binary operator '-' (line 403)
            result_sub_15583 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 45), '-', result_sub_15581, int_15582)
            
            # Assigning a type to the variable 'targetlen' (line 403)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'targetlen', result_sub_15583)
            
            # Getting the type of 'targetlen' (line 404)
            targetlen_15584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'targetlen')
            
            # Call to encoded_header_len(...): (line 404)
            # Processing the call arguments (line 404)
            str_15587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 54), 'str', '')
            # Processing the call keyword arguments (line 404)
            kwargs_15588 = {}
            # Getting the type of 'charset' (line 404)
            charset_15585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 27), 'charset', False)
            # Obtaining the member 'encoded_header_len' of a type (line 404)
            encoded_header_len_15586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 27), charset_15585, 'encoded_header_len')
            # Calling encoded_header_len(args, kwargs) (line 404)
            encoded_header_len_call_result_15589 = invoke(stypy.reporting.localization.Localization(__file__, 404, 27), encoded_header_len_15586, *[str_15587], **kwargs_15588)
            
            # Applying the binary operator '<' (line 404)
            result_lt_15590 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 15), '<', targetlen_15584, encoded_header_len_call_result_15589)
            
            # Testing if the type of an if condition is none (line 404)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 404, 12), result_lt_15590):
                pass
            else:
                
                # Testing the type of an if condition (line 404)
                if_condition_15591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 12), result_lt_15590)
                # Assigning a type to the variable 'if_condition_15591' (line 404)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'if_condition_15591', if_condition_15591)
                # SSA begins for if statement (line 404)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 406):
                
                # Assigning a Name to a Name (line 406):
                # Getting the type of 'maxlinelen' (line 406)
                maxlinelen_15592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 28), 'maxlinelen')
                # Assigning a type to the variable 'targetlen' (line 406)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'targetlen', maxlinelen_15592)
                # SSA join for if statement (line 404)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'newchunks' (line 407)
            newchunks_15593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'newchunks')
            
            # Call to _split(...): (line 407)
            # Processing the call arguments (line 407)
            # Getting the type of 's' (line 407)
            s_15596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 37), 's', False)
            # Getting the type of 'charset' (line 407)
            charset_15597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 40), 'charset', False)
            # Getting the type of 'targetlen' (line 407)
            targetlen_15598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 49), 'targetlen', False)
            # Getting the type of 'splitchars' (line 407)
            splitchars_15599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 60), 'splitchars', False)
            # Processing the call keyword arguments (line 407)
            kwargs_15600 = {}
            # Getting the type of 'self' (line 407)
            self_15594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 25), 'self', False)
            # Obtaining the member '_split' of a type (line 407)
            _split_15595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 25), self_15594, '_split')
            # Calling _split(args, kwargs) (line 407)
            _split_call_result_15601 = invoke(stypy.reporting.localization.Localization(__file__, 407, 25), _split_15595, *[s_15596, charset_15597, targetlen_15598, splitchars_15599], **kwargs_15600)
            
            # Applying the binary operator '+=' (line 407)
            result_iadd_15602 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 12), '+=', newchunks_15593, _split_call_result_15601)
            # Assigning a type to the variable 'newchunks' (line 407)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'newchunks', result_iadd_15602)
            
            
            # Assigning a Subscript to a Tuple (line 408):
            
            # Assigning a Subscript to a Name (line 408):
            
            # Obtaining the type of the subscript
            int_15603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 12), 'int')
            
            # Obtaining the type of the subscript
            int_15604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 47), 'int')
            # Getting the type of 'newchunks' (line 408)
            newchunks_15605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 37), 'newchunks')
            # Obtaining the member '__getitem__' of a type (line 408)
            getitem___15606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 37), newchunks_15605, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 408)
            subscript_call_result_15607 = invoke(stypy.reporting.localization.Localization(__file__, 408, 37), getitem___15606, int_15604)
            
            # Obtaining the member '__getitem__' of a type (line 408)
            getitem___15608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 12), subscript_call_result_15607, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 408)
            subscript_call_result_15609 = invoke(stypy.reporting.localization.Localization(__file__, 408, 12), getitem___15608, int_15603)
            
            # Assigning a type to the variable 'tuple_var_assignment_14849' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'tuple_var_assignment_14849', subscript_call_result_15609)
            
            # Assigning a Subscript to a Name (line 408):
            
            # Obtaining the type of the subscript
            int_15610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 12), 'int')
            
            # Obtaining the type of the subscript
            int_15611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 47), 'int')
            # Getting the type of 'newchunks' (line 408)
            newchunks_15612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 37), 'newchunks')
            # Obtaining the member '__getitem__' of a type (line 408)
            getitem___15613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 37), newchunks_15612, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 408)
            subscript_call_result_15614 = invoke(stypy.reporting.localization.Localization(__file__, 408, 37), getitem___15613, int_15611)
            
            # Obtaining the member '__getitem__' of a type (line 408)
            getitem___15615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 12), subscript_call_result_15614, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 408)
            subscript_call_result_15616 = invoke(stypy.reporting.localization.Localization(__file__, 408, 12), getitem___15615, int_15610)
            
            # Assigning a type to the variable 'tuple_var_assignment_14850' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'tuple_var_assignment_14850', subscript_call_result_15616)
            
            # Assigning a Name to a Name (line 408):
            # Getting the type of 'tuple_var_assignment_14849' (line 408)
            tuple_var_assignment_14849_15617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'tuple_var_assignment_14849')
            # Assigning a type to the variable 'lastchunk' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'lastchunk', tuple_var_assignment_14849_15617)
            
            # Assigning a Name to a Name (line 408):
            # Getting the type of 'tuple_var_assignment_14850' (line 408)
            tuple_var_assignment_14850_15618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'tuple_var_assignment_14850')
            # Assigning a type to the variable 'lastcharset' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 23), 'lastcharset', tuple_var_assignment_14850_15618)
            
            # Assigning a Call to a Name (line 409):
            
            # Assigning a Call to a Name (line 409):
            
            # Call to encoded_header_len(...): (line 409)
            # Processing the call arguments (line 409)
            # Getting the type of 'lastchunk' (line 409)
            lastchunk_15621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 53), 'lastchunk', False)
            # Processing the call keyword arguments (line 409)
            kwargs_15622 = {}
            # Getting the type of 'lastcharset' (line 409)
            lastcharset_15619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 22), 'lastcharset', False)
            # Obtaining the member 'encoded_header_len' of a type (line 409)
            encoded_header_len_15620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 22), lastcharset_15619, 'encoded_header_len')
            # Calling encoded_header_len(args, kwargs) (line 409)
            encoded_header_len_call_result_15623 = invoke(stypy.reporting.localization.Localization(__file__, 409, 22), encoded_header_len_15620, *[lastchunk_15621], **kwargs_15622)
            
            # Assigning a type to the variable 'lastlen' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'lastlen', encoded_header_len_call_result_15623)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 410):
        
        # Assigning a Call to a Name (line 410):
        
        # Call to _encode_chunks(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'newchunks' (line 410)
        newchunks_15626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 36), 'newchunks', False)
        # Getting the type of 'maxlinelen' (line 410)
        maxlinelen_15627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 47), 'maxlinelen', False)
        # Processing the call keyword arguments (line 410)
        kwargs_15628 = {}
        # Getting the type of 'self' (line 410)
        self_15624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 16), 'self', False)
        # Obtaining the member '_encode_chunks' of a type (line 410)
        _encode_chunks_15625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 16), self_15624, '_encode_chunks')
        # Calling _encode_chunks(args, kwargs) (line 410)
        _encode_chunks_call_result_15629 = invoke(stypy.reporting.localization.Localization(__file__, 410, 16), _encode_chunks_15625, *[newchunks_15626, maxlinelen_15627], **kwargs_15628)
        
        # Assigning a type to the variable 'value' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'value', _encode_chunks_call_result_15629)
        
        # Call to search(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'value' (line 411)
        value_15632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 34), 'value', False)
        # Processing the call keyword arguments (line 411)
        kwargs_15633 = {}
        # Getting the type of '_embeded_header' (line 411)
        _embeded_header_15630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 11), '_embeded_header', False)
        # Obtaining the member 'search' of a type (line 411)
        search_15631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 11), _embeded_header_15630, 'search')
        # Calling search(args, kwargs) (line 411)
        search_call_result_15634 = invoke(stypy.reporting.localization.Localization(__file__, 411, 11), search_15631, *[value_15632], **kwargs_15633)
        
        # Testing if the type of an if condition is none (line 411)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 411, 8), search_call_result_15634):
            pass
        else:
            
            # Testing the type of an if condition (line 411)
            if_condition_15635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 8), search_call_result_15634)
            # Assigning a type to the variable 'if_condition_15635' (line 411)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'if_condition_15635', if_condition_15635)
            # SSA begins for if statement (line 411)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to HeaderParseError(...): (line 412)
            # Processing the call arguments (line 412)
            
            # Call to format(...): (line 412)
            # Processing the call arguments (line 412)
            # Getting the type of 'value' (line 413)
            value_15639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 50), 'value', False)
            # Processing the call keyword arguments (line 412)
            kwargs_15640 = {}
            str_15637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 35), 'str', 'header value appears to contain an embedded header: {!r}')
            # Obtaining the member 'format' of a type (line 412)
            format_15638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 35), str_15637, 'format')
            # Calling format(args, kwargs) (line 412)
            format_call_result_15641 = invoke(stypy.reporting.localization.Localization(__file__, 412, 35), format_15638, *[value_15639], **kwargs_15640)
            
            # Processing the call keyword arguments (line 412)
            kwargs_15642 = {}
            # Getting the type of 'HeaderParseError' (line 412)
            HeaderParseError_15636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 18), 'HeaderParseError', False)
            # Calling HeaderParseError(args, kwargs) (line 412)
            HeaderParseError_call_result_15643 = invoke(stypy.reporting.localization.Localization(__file__, 412, 18), HeaderParseError_15636, *[format_call_result_15641], **kwargs_15642)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 412, 12), HeaderParseError_call_result_15643, 'raise parameter', BaseException)
            # SSA join for if statement (line 411)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'value' (line 414)
        value_15644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 15), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'stypy_return_type', value_15644)
        
        # ################# End of 'encode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'encode' in the type store
        # Getting the type of 'stypy_return_type' (line 374)
        stypy_return_type_15645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15645)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'encode'
        return stypy_return_type_15645


# Assigning a type to the variable 'Header' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'Header', Header)

@norecursion
def _split_ascii(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_split_ascii'
    module_type_store = module_type_store.open_function_context('_split_ascii', 418, 0, False)
    
    # Passed parameters checking function
    _split_ascii.stypy_localization = localization
    _split_ascii.stypy_type_of_self = None
    _split_ascii.stypy_type_store = module_type_store
    _split_ascii.stypy_function_name = '_split_ascii'
    _split_ascii.stypy_param_names_list = ['s', 'firstlen', 'restlen', 'continuation_ws', 'splitchars']
    _split_ascii.stypy_varargs_param_name = None
    _split_ascii.stypy_kwargs_param_name = None
    _split_ascii.stypy_call_defaults = defaults
    _split_ascii.stypy_call_varargs = varargs
    _split_ascii.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_split_ascii', ['s', 'firstlen', 'restlen', 'continuation_ws', 'splitchars'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_split_ascii', localization, ['s', 'firstlen', 'restlen', 'continuation_ws', 'splitchars'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_split_ascii(...)' code ##################

    
    # Assigning a List to a Name (line 419):
    
    # Assigning a List to a Name (line 419):
    
    # Obtaining an instance of the builtin type 'list' (line 419)
    list_15646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 419)
    
    # Assigning a type to the variable 'lines' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'lines', list_15646)
    
    # Assigning a Name to a Name (line 420):
    
    # Assigning a Name to a Name (line 420):
    # Getting the type of 'firstlen' (line 420)
    firstlen_15647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 13), 'firstlen')
    # Assigning a type to the variable 'maxlen' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'maxlen', firstlen_15647)
    
    
    # Call to splitlines(...): (line 421)
    # Processing the call keyword arguments (line 421)
    kwargs_15650 = {}
    # Getting the type of 's' (line 421)
    s_15648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 's', False)
    # Obtaining the member 'splitlines' of a type (line 421)
    splitlines_15649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 16), s_15648, 'splitlines')
    # Calling splitlines(args, kwargs) (line 421)
    splitlines_call_result_15651 = invoke(stypy.reporting.localization.Localization(__file__, 421, 16), splitlines_15649, *[], **kwargs_15650)
    
    # Assigning a type to the variable 'splitlines_call_result_15651' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'splitlines_call_result_15651', splitlines_call_result_15651)
    # Testing if the for loop is going to be iterated (line 421)
    # Testing the type of a for loop iterable (line 421)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 421, 4), splitlines_call_result_15651)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 421, 4), splitlines_call_result_15651):
        # Getting the type of the for loop variable (line 421)
        for_loop_var_15652 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 421, 4), splitlines_call_result_15651)
        # Assigning a type to the variable 'line' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'line', for_loop_var_15652)
        # SSA begins for a for statement (line 421)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 424):
        
        # Assigning a Call to a Name (line 424):
        
        # Call to lstrip(...): (line 424)
        # Processing the call keyword arguments (line 424)
        kwargs_15655 = {}
        # Getting the type of 'line' (line 424)
        line_15653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'line', False)
        # Obtaining the member 'lstrip' of a type (line 424)
        lstrip_15654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 15), line_15653, 'lstrip')
        # Calling lstrip(args, kwargs) (line 424)
        lstrip_call_result_15656 = invoke(stypy.reporting.localization.Localization(__file__, 424, 15), lstrip_15654, *[], **kwargs_15655)
        
        # Assigning a type to the variable 'line' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'line', lstrip_call_result_15656)
        
        
        # Call to len(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'line' (line 425)
        line_15658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'line', False)
        # Processing the call keyword arguments (line 425)
        kwargs_15659 = {}
        # Getting the type of 'len' (line 425)
        len_15657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 11), 'len', False)
        # Calling len(args, kwargs) (line 425)
        len_call_result_15660 = invoke(stypy.reporting.localization.Localization(__file__, 425, 11), len_15657, *[line_15658], **kwargs_15659)
        
        # Getting the type of 'maxlen' (line 425)
        maxlen_15661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 23), 'maxlen')
        # Applying the binary operator '<' (line 425)
        result_lt_15662 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 11), '<', len_call_result_15660, maxlen_15661)
        
        # Testing if the type of an if condition is none (line 425)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 425, 8), result_lt_15662):
            pass
        else:
            
            # Testing the type of an if condition (line 425)
            if_condition_15663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 8), result_lt_15662)
            # Assigning a type to the variable 'if_condition_15663' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'if_condition_15663', if_condition_15663)
            # SSA begins for if statement (line 425)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 426)
            # Processing the call arguments (line 426)
            # Getting the type of 'line' (line 426)
            line_15666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 25), 'line', False)
            # Processing the call keyword arguments (line 426)
            kwargs_15667 = {}
            # Getting the type of 'lines' (line 426)
            lines_15664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'lines', False)
            # Obtaining the member 'append' of a type (line 426)
            append_15665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), lines_15664, 'append')
            # Calling append(args, kwargs) (line 426)
            append_call_result_15668 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), append_15665, *[line_15666], **kwargs_15667)
            
            
            # Assigning a Name to a Name (line 427):
            
            # Assigning a Name to a Name (line 427):
            # Getting the type of 'restlen' (line 427)
            restlen_15669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 21), 'restlen')
            # Assigning a type to the variable 'maxlen' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'maxlen', restlen_15669)
            # SSA join for if statement (line 425)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'splitchars' (line 433)
        splitchars_15670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 18), 'splitchars')
        # Assigning a type to the variable 'splitchars_15670' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'splitchars_15670', splitchars_15670)
        # Testing if the for loop is going to be iterated (line 433)
        # Testing the type of a for loop iterable (line 433)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 433, 8), splitchars_15670)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 433, 8), splitchars_15670):
            # Getting the type of the for loop variable (line 433)
            for_loop_var_15671 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 433, 8), splitchars_15670)
            # Assigning a type to the variable 'ch' (line 433)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'ch', for_loop_var_15671)
            # SSA begins for a for statement (line 433)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'ch' (line 434)
            ch_15672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'ch')
            # Getting the type of 'line' (line 434)
            line_15673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 21), 'line')
            # Applying the binary operator 'in' (line 434)
            result_contains_15674 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 15), 'in', ch_15672, line_15673)
            
            # Testing if the type of an if condition is none (line 434)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 434, 12), result_contains_15674):
                pass
            else:
                
                # Testing the type of an if condition (line 434)
                if_condition_15675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 12), result_contains_15674)
                # Assigning a type to the variable 'if_condition_15675' (line 434)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'if_condition_15675', if_condition_15675)
                # SSA begins for if statement (line 434)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 434)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of a for statement (line 433)
            module_type_store.open_ssa_branch('for loop else')
            
            # Call to append(...): (line 439)
            # Processing the call arguments (line 439)
            # Getting the type of 'line' (line 439)
            line_15678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 25), 'line', False)
            # Processing the call keyword arguments (line 439)
            kwargs_15679 = {}
            # Getting the type of 'lines' (line 439)
            lines_15676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'lines', False)
            # Obtaining the member 'append' of a type (line 439)
            append_15677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 12), lines_15676, 'append')
            # Calling append(args, kwargs) (line 439)
            append_call_result_15680 = invoke(stypy.reporting.localization.Localization(__file__, 439, 12), append_15677, *[line_15678], **kwargs_15679)
            
            
            # Assigning a Name to a Name (line 440):
            
            # Assigning a Name to a Name (line 440):
            # Getting the type of 'restlen' (line 440)
            restlen_15681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 21), 'restlen')
            # Assigning a type to the variable 'maxlen' (line 440)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'maxlen', restlen_15681)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
        else:
            
            # Call to append(...): (line 439)
            # Processing the call arguments (line 439)
            # Getting the type of 'line' (line 439)
            line_15678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 25), 'line', False)
            # Processing the call keyword arguments (line 439)
            kwargs_15679 = {}
            # Getting the type of 'lines' (line 439)
            lines_15676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'lines', False)
            # Obtaining the member 'append' of a type (line 439)
            append_15677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 12), lines_15676, 'append')
            # Calling append(args, kwargs) (line 439)
            append_call_result_15680 = invoke(stypy.reporting.localization.Localization(__file__, 439, 12), append_15677, *[line_15678], **kwargs_15679)
            
            
            # Assigning a Name to a Name (line 440):
            
            # Assigning a Name to a Name (line 440):
            # Getting the type of 'restlen' (line 440)
            restlen_15681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 21), 'restlen')
            # Assigning a type to the variable 'maxlen' (line 440)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'maxlen', restlen_15681)

        
        
        # Assigning a Call to a Name (line 443):
        
        # Assigning a Call to a Name (line 443):
        
        # Call to compile(...): (line 443)
        # Processing the call arguments (line 443)
        str_15684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 25), 'str', '%s\\s*')
        # Getting the type of 'ch' (line 443)
        ch_15685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 36), 'ch', False)
        # Applying the binary operator '%' (line 443)
        result_mod_15686 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 25), '%', str_15684, ch_15685)
        
        # Processing the call keyword arguments (line 443)
        kwargs_15687 = {}
        # Getting the type of 're' (line 443)
        re_15682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 14), 're', False)
        # Obtaining the member 'compile' of a type (line 443)
        compile_15683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 14), re_15682, 'compile')
        # Calling compile(args, kwargs) (line 443)
        compile_call_result_15688 = invoke(stypy.reporting.localization.Localization(__file__, 443, 14), compile_15683, *[result_mod_15686], **kwargs_15687)
        
        # Assigning a type to the variable 'cre' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'cre', compile_call_result_15688)
        
        # Getting the type of 'ch' (line 444)
        ch_15689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 11), 'ch')
        str_15690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 17), 'str', ';,')
        # Applying the binary operator 'in' (line 444)
        result_contains_15691 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 11), 'in', ch_15689, str_15690)
        
        # Testing if the type of an if condition is none (line 444)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 444, 8), result_contains_15691):
            
            # Assigning a Str to a Name (line 447):
            
            # Assigning a Str to a Name (line 447):
            str_15694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 18), 'str', '')
            # Assigning a type to the variable 'eol' (line 447)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'eol', str_15694)
        else:
            
            # Testing the type of an if condition (line 444)
            if_condition_15692 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 8), result_contains_15691)
            # Assigning a type to the variable 'if_condition_15692' (line 444)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'if_condition_15692', if_condition_15692)
            # SSA begins for if statement (line 444)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 445):
            
            # Assigning a Name to a Name (line 445):
            # Getting the type of 'ch' (line 445)
            ch_15693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 18), 'ch')
            # Assigning a type to the variable 'eol' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'eol', ch_15693)
            # SSA branch for the else part of an if statement (line 444)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 447):
            
            # Assigning a Str to a Name (line 447):
            str_15694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 18), 'str', '')
            # Assigning a type to the variable 'eol' (line 447)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'eol', str_15694)
            # SSA join for if statement (line 444)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 448):
        
        # Assigning a BinOp to a Name (line 448):
        # Getting the type of 'eol' (line 448)
        eol_15695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 17), 'eol')
        str_15696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 23), 'str', ' ')
        # Applying the binary operator '+' (line 448)
        result_add_15697 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 17), '+', eol_15695, str_15696)
        
        # Assigning a type to the variable 'joiner' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'joiner', result_add_15697)
        
        # Assigning a Call to a Name (line 449):
        
        # Assigning a Call to a Name (line 449):
        
        # Call to len(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'joiner' (line 449)
        joiner_15699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 22), 'joiner', False)
        # Processing the call keyword arguments (line 449)
        kwargs_15700 = {}
        # Getting the type of 'len' (line 449)
        len_15698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 18), 'len', False)
        # Calling len(args, kwargs) (line 449)
        len_call_result_15701 = invoke(stypy.reporting.localization.Localization(__file__, 449, 18), len_15698, *[joiner_15699], **kwargs_15700)
        
        # Assigning a type to the variable 'joinlen' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'joinlen', len_call_result_15701)
        
        # Assigning a Call to a Name (line 450):
        
        # Assigning a Call to a Name (line 450):
        
        # Call to len(...): (line 450)
        # Processing the call arguments (line 450)
        
        # Call to replace(...): (line 450)
        # Processing the call arguments (line 450)
        str_15705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 44), 'str', '\t')
        # Getting the type of 'SPACE8' (line 450)
        SPACE8_15706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 50), 'SPACE8', False)
        # Processing the call keyword arguments (line 450)
        kwargs_15707 = {}
        # Getting the type of 'continuation_ws' (line 450)
        continuation_ws_15703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 20), 'continuation_ws', False)
        # Obtaining the member 'replace' of a type (line 450)
        replace_15704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 20), continuation_ws_15703, 'replace')
        # Calling replace(args, kwargs) (line 450)
        replace_call_result_15708 = invoke(stypy.reporting.localization.Localization(__file__, 450, 20), replace_15704, *[str_15705, SPACE8_15706], **kwargs_15707)
        
        # Processing the call keyword arguments (line 450)
        kwargs_15709 = {}
        # Getting the type of 'len' (line 450)
        len_15702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'len', False)
        # Calling len(args, kwargs) (line 450)
        len_call_result_15710 = invoke(stypy.reporting.localization.Localization(__file__, 450, 16), len_15702, *[replace_call_result_15708], **kwargs_15709)
        
        # Assigning a type to the variable 'wslen' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'wslen', len_call_result_15710)
        
        # Assigning a List to a Name (line 451):
        
        # Assigning a List to a Name (line 451):
        
        # Obtaining an instance of the builtin type 'list' (line 451)
        list_15711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 451)
        
        # Assigning a type to the variable 'this' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'this', list_15711)
        
        # Assigning a Num to a Name (line 452):
        
        # Assigning a Num to a Name (line 452):
        int_15712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 18), 'int')
        # Assigning a type to the variable 'linelen' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'linelen', int_15712)
        
        
        # Call to split(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'line' (line 453)
        line_15715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 30), 'line', False)
        # Processing the call keyword arguments (line 453)
        kwargs_15716 = {}
        # Getting the type of 'cre' (line 453)
        cre_15713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 20), 'cre', False)
        # Obtaining the member 'split' of a type (line 453)
        split_15714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 20), cre_15713, 'split')
        # Calling split(args, kwargs) (line 453)
        split_call_result_15717 = invoke(stypy.reporting.localization.Localization(__file__, 453, 20), split_15714, *[line_15715], **kwargs_15716)
        
        # Assigning a type to the variable 'split_call_result_15717' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'split_call_result_15717', split_call_result_15717)
        # Testing if the for loop is going to be iterated (line 453)
        # Testing the type of a for loop iterable (line 453)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 453, 8), split_call_result_15717)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 453, 8), split_call_result_15717):
            # Getting the type of the for loop variable (line 453)
            for_loop_var_15718 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 453, 8), split_call_result_15717)
            # Assigning a type to the variable 'part' (line 453)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'part', for_loop_var_15718)
            # SSA begins for a for statement (line 453)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 454):
            
            # Assigning a BinOp to a Name (line 454):
            # Getting the type of 'linelen' (line 454)
            linelen_15719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 21), 'linelen')
            
            # Call to max(...): (line 454)
            # Processing the call arguments (line 454)
            int_15721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 35), 'int')
            
            # Call to len(...): (line 454)
            # Processing the call arguments (line 454)
            # Getting the type of 'this' (line 454)
            this_15723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 42), 'this', False)
            # Processing the call keyword arguments (line 454)
            kwargs_15724 = {}
            # Getting the type of 'len' (line 454)
            len_15722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 38), 'len', False)
            # Calling len(args, kwargs) (line 454)
            len_call_result_15725 = invoke(stypy.reporting.localization.Localization(__file__, 454, 38), len_15722, *[this_15723], **kwargs_15724)
            
            int_15726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 48), 'int')
            # Applying the binary operator '-' (line 454)
            result_sub_15727 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 38), '-', len_call_result_15725, int_15726)
            
            # Processing the call keyword arguments (line 454)
            kwargs_15728 = {}
            # Getting the type of 'max' (line 454)
            max_15720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 31), 'max', False)
            # Calling max(args, kwargs) (line 454)
            max_call_result_15729 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), max_15720, *[int_15721, result_sub_15727], **kwargs_15728)
            
            # Getting the type of 'joinlen' (line 454)
            joinlen_15730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 53), 'joinlen')
            # Applying the binary operator '*' (line 454)
            result_mul_15731 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 31), '*', max_call_result_15729, joinlen_15730)
            
            # Applying the binary operator '+' (line 454)
            result_add_15732 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 21), '+', linelen_15719, result_mul_15731)
            
            # Assigning a type to the variable 'curlen' (line 454)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'curlen', result_add_15732)
            
            # Assigning a Call to a Name (line 455):
            
            # Assigning a Call to a Name (line 455):
            
            # Call to len(...): (line 455)
            # Processing the call arguments (line 455)
            # Getting the type of 'part' (line 455)
            part_15734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 26), 'part', False)
            # Processing the call keyword arguments (line 455)
            kwargs_15735 = {}
            # Getting the type of 'len' (line 455)
            len_15733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 22), 'len', False)
            # Calling len(args, kwargs) (line 455)
            len_call_result_15736 = invoke(stypy.reporting.localization.Localization(__file__, 455, 22), len_15733, *[part_15734], **kwargs_15735)
            
            # Assigning a type to the variable 'partlen' (line 455)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'partlen', len_call_result_15736)
            
            # Assigning a UnaryOp to a Name (line 456):
            
            # Assigning a UnaryOp to a Name (line 456):
            
            # Getting the type of 'lines' (line 456)
            lines_15737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'lines')
            # Applying the 'not' unary operator (line 456)
            result_not__15738 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 26), 'not', lines_15737)
            
            # Assigning a type to the variable 'onfirstline' (line 456)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'onfirstline', result_not__15738)
            
            # Evaluating a boolean operation
            
            # Getting the type of 'ch' (line 459)
            ch_15739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'ch')
            str_15740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 21), 'str', ' ')
            # Applying the binary operator '==' (line 459)
            result_eq_15741 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 15), '==', ch_15739, str_15740)
            
            # Getting the type of 'onfirstline' (line 459)
            onfirstline_15742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 29), 'onfirstline')
            # Applying the binary operator 'and' (line 459)
            result_and_keyword_15743 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 15), 'and', result_eq_15741, onfirstline_15742)
            
            
            # Call to len(...): (line 460)
            # Processing the call arguments (line 460)
            # Getting the type of 'this' (line 460)
            this_15745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'this', False)
            # Processing the call keyword arguments (line 460)
            kwargs_15746 = {}
            # Getting the type of 'len' (line 460)
            len_15744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 19), 'len', False)
            # Calling len(args, kwargs) (line 460)
            len_call_result_15747 = invoke(stypy.reporting.localization.Localization(__file__, 460, 19), len_15744, *[this_15745], **kwargs_15746)
            
            int_15748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 32), 'int')
            # Applying the binary operator '==' (line 460)
            result_eq_15749 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 19), '==', len_call_result_15747, int_15748)
            
            # Applying the binary operator 'and' (line 459)
            result_and_keyword_15750 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 15), 'and', result_and_keyword_15743, result_eq_15749)
            
            # Call to match(...): (line 460)
            # Processing the call arguments (line 460)
            
            # Obtaining the type of the subscript
            int_15753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 54), 'int')
            # Getting the type of 'this' (line 460)
            this_15754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 49), 'this', False)
            # Obtaining the member '__getitem__' of a type (line 460)
            getitem___15755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 49), this_15754, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 460)
            subscript_call_result_15756 = invoke(stypy.reporting.localization.Localization(__file__, 460, 49), getitem___15755, int_15753)
            
            # Processing the call keyword arguments (line 460)
            kwargs_15757 = {}
            # Getting the type of 'fcre' (line 460)
            fcre_15751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 38), 'fcre', False)
            # Obtaining the member 'match' of a type (line 460)
            match_15752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 38), fcre_15751, 'match')
            # Calling match(args, kwargs) (line 460)
            match_call_result_15758 = invoke(stypy.reporting.localization.Localization(__file__, 460, 38), match_15752, *[subscript_call_result_15756], **kwargs_15757)
            
            # Applying the binary operator 'and' (line 459)
            result_and_keyword_15759 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 15), 'and', result_and_keyword_15750, match_call_result_15758)
            
            # Testing if the type of an if condition is none (line 459)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 459, 12), result_and_keyword_15759):
                
                # Getting the type of 'curlen' (line 463)
                curlen_15769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 17), 'curlen')
                # Getting the type of 'partlen' (line 463)
                partlen_15770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 26), 'partlen')
                # Applying the binary operator '+' (line 463)
                result_add_15771 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 17), '+', curlen_15769, partlen_15770)
                
                # Getting the type of 'maxlen' (line 463)
                maxlen_15772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 36), 'maxlen')
                # Applying the binary operator '>' (line 463)
                result_gt_15773 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 17), '>', result_add_15771, maxlen_15772)
                
                # Testing if the type of an if condition is none (line 463)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 463, 17), result_gt_15773):
                    
                    # Call to append(...): (line 479)
                    # Processing the call arguments (line 479)
                    # Getting the type of 'part' (line 479)
                    part_15832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 28), 'part', False)
                    # Processing the call keyword arguments (line 479)
                    kwargs_15833 = {}
                    # Getting the type of 'this' (line 479)
                    this_15830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'this', False)
                    # Obtaining the member 'append' of a type (line 479)
                    append_15831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 16), this_15830, 'append')
                    # Calling append(args, kwargs) (line 479)
                    append_call_result_15834 = invoke(stypy.reporting.localization.Localization(__file__, 479, 16), append_15831, *[part_15832], **kwargs_15833)
                    
                    
                    # Getting the type of 'linelen' (line 480)
                    linelen_15835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'linelen')
                    # Getting the type of 'partlen' (line 480)
                    partlen_15836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 27), 'partlen')
                    # Applying the binary operator '+=' (line 480)
                    result_iadd_15837 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 16), '+=', linelen_15835, partlen_15836)
                    # Assigning a type to the variable 'linelen' (line 480)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'linelen', result_iadd_15837)
                    
                else:
                    
                    # Testing the type of an if condition (line 463)
                    if_condition_15774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 17), result_gt_15773)
                    # Assigning a type to the variable 'if_condition_15774' (line 463)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 17), 'if_condition_15774', if_condition_15774)
                    # SSA begins for if statement (line 463)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'this' (line 464)
                    this_15775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 19), 'this')
                    # Testing if the type of an if condition is none (line 464)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 464, 16), this_15775):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 464)
                        if_condition_15776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 464, 16), this_15775)
                        # Assigning a type to the variable 'if_condition_15776' (line 464)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'if_condition_15776', if_condition_15776)
                        # SSA begins for if statement (line 464)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 465)
                        # Processing the call arguments (line 465)
                        
                        # Call to join(...): (line 465)
                        # Processing the call arguments (line 465)
                        # Getting the type of 'this' (line 465)
                        this_15781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 45), 'this', False)
                        # Processing the call keyword arguments (line 465)
                        kwargs_15782 = {}
                        # Getting the type of 'joiner' (line 465)
                        joiner_15779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 33), 'joiner', False)
                        # Obtaining the member 'join' of a type (line 465)
                        join_15780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 33), joiner_15779, 'join')
                        # Calling join(args, kwargs) (line 465)
                        join_call_result_15783 = invoke(stypy.reporting.localization.Localization(__file__, 465, 33), join_15780, *[this_15781], **kwargs_15782)
                        
                        # Getting the type of 'eol' (line 465)
                        eol_15784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 53), 'eol', False)
                        # Applying the binary operator '+' (line 465)
                        result_add_15785 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 33), '+', join_call_result_15783, eol_15784)
                        
                        # Processing the call keyword arguments (line 465)
                        kwargs_15786 = {}
                        # Getting the type of 'lines' (line 465)
                        lines_15777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 20), 'lines', False)
                        # Obtaining the member 'append' of a type (line 465)
                        append_15778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 20), lines_15777, 'append')
                        # Calling append(args, kwargs) (line 465)
                        append_call_result_15787 = invoke(stypy.reporting.localization.Localization(__file__, 465, 20), append_15778, *[result_add_15785], **kwargs_15786)
                        
                        # SSA join for if statement (line 464)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'partlen' (line 469)
                    partlen_15788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 19), 'partlen')
                    # Getting the type of 'maxlen' (line 469)
                    maxlen_15789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 29), 'maxlen')
                    # Applying the binary operator '>' (line 469)
                    result_gt_15790 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 19), '>', partlen_15788, maxlen_15789)
                    
                    
                    # Getting the type of 'ch' (line 469)
                    ch_15791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 40), 'ch')
                    str_15792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 46), 'str', ' ')
                    # Applying the binary operator '!=' (line 469)
                    result_ne_15793 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 40), '!=', ch_15791, str_15792)
                    
                    # Applying the binary operator 'and' (line 469)
                    result_and_keyword_15794 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 19), 'and', result_gt_15790, result_ne_15793)
                    
                    # Testing if the type of an if condition is none (line 469)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 469, 16), result_and_keyword_15794):
                        
                        # Assigning a List to a Name (line 475):
                        
                        # Assigning a List to a Name (line 475):
                        
                        # Obtaining an instance of the builtin type 'list' (line 475)
                        list_15818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 27), 'list')
                        # Adding type elements to the builtin type 'list' instance (line 475)
                        # Adding element type (line 475)
                        # Getting the type of 'part' (line 475)
                        part_15819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 28), 'part')
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 27), list_15818, part_15819)
                        
                        # Assigning a type to the variable 'this' (line 475)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 20), 'this', list_15818)
                    else:
                        
                        # Testing the type of an if condition (line 469)
                        if_condition_15795 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 469, 16), result_and_keyword_15794)
                        # Assigning a type to the variable 'if_condition_15795' (line 469)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 16), 'if_condition_15795', if_condition_15795)
                        # SSA begins for if statement (line 469)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 470):
                        
                        # Assigning a Call to a Name (line 470):
                        
                        # Call to _split_ascii(...): (line 470)
                        # Processing the call arguments (line 470)
                        # Getting the type of 'part' (line 470)
                        part_15797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 40), 'part', False)
                        # Getting the type of 'maxlen' (line 470)
                        maxlen_15798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 46), 'maxlen', False)
                        # Getting the type of 'restlen' (line 470)
                        restlen_15799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 54), 'restlen', False)
                        # Getting the type of 'continuation_ws' (line 471)
                        continuation_ws_15800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 40), 'continuation_ws', False)
                        str_15801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 57), 'str', ' ')
                        # Processing the call keyword arguments (line 470)
                        kwargs_15802 = {}
                        # Getting the type of '_split_ascii' (line 470)
                        _split_ascii_15796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 27), '_split_ascii', False)
                        # Calling _split_ascii(args, kwargs) (line 470)
                        _split_ascii_call_result_15803 = invoke(stypy.reporting.localization.Localization(__file__, 470, 27), _split_ascii_15796, *[part_15797, maxlen_15798, restlen_15799, continuation_ws_15800, str_15801], **kwargs_15802)
                        
                        # Assigning a type to the variable 'subl' (line 470)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 20), 'subl', _split_ascii_call_result_15803)
                        
                        # Call to extend(...): (line 472)
                        # Processing the call arguments (line 472)
                        
                        # Obtaining the type of the subscript
                        int_15806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 39), 'int')
                        slice_15807 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 472, 33), None, int_15806, None)
                        # Getting the type of 'subl' (line 472)
                        subl_15808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 33), 'subl', False)
                        # Obtaining the member '__getitem__' of a type (line 472)
                        getitem___15809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 33), subl_15808, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 472)
                        subscript_call_result_15810 = invoke(stypy.reporting.localization.Localization(__file__, 472, 33), getitem___15809, slice_15807)
                        
                        # Processing the call keyword arguments (line 472)
                        kwargs_15811 = {}
                        # Getting the type of 'lines' (line 472)
                        lines_15804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 20), 'lines', False)
                        # Obtaining the member 'extend' of a type (line 472)
                        extend_15805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 20), lines_15804, 'extend')
                        # Calling extend(args, kwargs) (line 472)
                        extend_call_result_15812 = invoke(stypy.reporting.localization.Localization(__file__, 472, 20), extend_15805, *[subscript_call_result_15810], **kwargs_15811)
                        
                        
                        # Assigning a List to a Name (line 473):
                        
                        # Assigning a List to a Name (line 473):
                        
                        # Obtaining an instance of the builtin type 'list' (line 473)
                        list_15813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 27), 'list')
                        # Adding type elements to the builtin type 'list' instance (line 473)
                        # Adding element type (line 473)
                        
                        # Obtaining the type of the subscript
                        int_15814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 33), 'int')
                        # Getting the type of 'subl' (line 473)
                        subl_15815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 28), 'subl')
                        # Obtaining the member '__getitem__' of a type (line 473)
                        getitem___15816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 28), subl_15815, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 473)
                        subscript_call_result_15817 = invoke(stypy.reporting.localization.Localization(__file__, 473, 28), getitem___15816, int_15814)
                        
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 27), list_15813, subscript_call_result_15817)
                        
                        # Assigning a type to the variable 'this' (line 473)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 20), 'this', list_15813)
                        # SSA branch for the else part of an if statement (line 469)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a List to a Name (line 475):
                        
                        # Assigning a List to a Name (line 475):
                        
                        # Obtaining an instance of the builtin type 'list' (line 475)
                        list_15818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 27), 'list')
                        # Adding type elements to the builtin type 'list' instance (line 475)
                        # Adding element type (line 475)
                        # Getting the type of 'part' (line 475)
                        part_15819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 28), 'part')
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 27), list_15818, part_15819)
                        
                        # Assigning a type to the variable 'this' (line 475)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 20), 'this', list_15818)
                        # SSA join for if statement (line 469)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Assigning a BinOp to a Name (line 476):
                    
                    # Assigning a BinOp to a Name (line 476):
                    # Getting the type of 'wslen' (line 476)
                    wslen_15820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 26), 'wslen')
                    
                    # Call to len(...): (line 476)
                    # Processing the call arguments (line 476)
                    
                    # Obtaining the type of the subscript
                    int_15822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 43), 'int')
                    # Getting the type of 'this' (line 476)
                    this_15823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 38), 'this', False)
                    # Obtaining the member '__getitem__' of a type (line 476)
                    getitem___15824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 38), this_15823, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
                    subscript_call_result_15825 = invoke(stypy.reporting.localization.Localization(__file__, 476, 38), getitem___15824, int_15822)
                    
                    # Processing the call keyword arguments (line 476)
                    kwargs_15826 = {}
                    # Getting the type of 'len' (line 476)
                    len_15821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 34), 'len', False)
                    # Calling len(args, kwargs) (line 476)
                    len_call_result_15827 = invoke(stypy.reporting.localization.Localization(__file__, 476, 34), len_15821, *[subscript_call_result_15825], **kwargs_15826)
                    
                    # Applying the binary operator '+' (line 476)
                    result_add_15828 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 26), '+', wslen_15820, len_call_result_15827)
                    
                    # Assigning a type to the variable 'linelen' (line 476)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 16), 'linelen', result_add_15828)
                    
                    # Assigning a Name to a Name (line 477):
                    
                    # Assigning a Name to a Name (line 477):
                    # Getting the type of 'restlen' (line 477)
                    restlen_15829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 25), 'restlen')
                    # Assigning a type to the variable 'maxlen' (line 477)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 16), 'maxlen', restlen_15829)
                    # SSA branch for the else part of an if statement (line 463)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to append(...): (line 479)
                    # Processing the call arguments (line 479)
                    # Getting the type of 'part' (line 479)
                    part_15832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 28), 'part', False)
                    # Processing the call keyword arguments (line 479)
                    kwargs_15833 = {}
                    # Getting the type of 'this' (line 479)
                    this_15830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'this', False)
                    # Obtaining the member 'append' of a type (line 479)
                    append_15831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 16), this_15830, 'append')
                    # Calling append(args, kwargs) (line 479)
                    append_call_result_15834 = invoke(stypy.reporting.localization.Localization(__file__, 479, 16), append_15831, *[part_15832], **kwargs_15833)
                    
                    
                    # Getting the type of 'linelen' (line 480)
                    linelen_15835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'linelen')
                    # Getting the type of 'partlen' (line 480)
                    partlen_15836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 27), 'partlen')
                    # Applying the binary operator '+=' (line 480)
                    result_iadd_15837 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 16), '+=', linelen_15835, partlen_15836)
                    # Assigning a type to the variable 'linelen' (line 480)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'linelen', result_iadd_15837)
                    
                    # SSA join for if statement (line 463)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 459)
                if_condition_15760 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 12), result_and_keyword_15759)
                # Assigning a type to the variable 'if_condition_15760' (line 459)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'if_condition_15760', if_condition_15760)
                # SSA begins for if statement (line 459)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 461)
                # Processing the call arguments (line 461)
                # Getting the type of 'part' (line 461)
                part_15763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 28), 'part', False)
                # Processing the call keyword arguments (line 461)
                kwargs_15764 = {}
                # Getting the type of 'this' (line 461)
                this_15761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'this', False)
                # Obtaining the member 'append' of a type (line 461)
                append_15762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 16), this_15761, 'append')
                # Calling append(args, kwargs) (line 461)
                append_call_result_15765 = invoke(stypy.reporting.localization.Localization(__file__, 461, 16), append_15762, *[part_15763], **kwargs_15764)
                
                
                # Getting the type of 'linelen' (line 462)
                linelen_15766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 16), 'linelen')
                # Getting the type of 'partlen' (line 462)
                partlen_15767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'partlen')
                # Applying the binary operator '+=' (line 462)
                result_iadd_15768 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 16), '+=', linelen_15766, partlen_15767)
                # Assigning a type to the variable 'linelen' (line 462)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 16), 'linelen', result_iadd_15768)
                
                # SSA branch for the else part of an if statement (line 459)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'curlen' (line 463)
                curlen_15769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 17), 'curlen')
                # Getting the type of 'partlen' (line 463)
                partlen_15770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 26), 'partlen')
                # Applying the binary operator '+' (line 463)
                result_add_15771 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 17), '+', curlen_15769, partlen_15770)
                
                # Getting the type of 'maxlen' (line 463)
                maxlen_15772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 36), 'maxlen')
                # Applying the binary operator '>' (line 463)
                result_gt_15773 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 17), '>', result_add_15771, maxlen_15772)
                
                # Testing if the type of an if condition is none (line 463)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 463, 17), result_gt_15773):
                    
                    # Call to append(...): (line 479)
                    # Processing the call arguments (line 479)
                    # Getting the type of 'part' (line 479)
                    part_15832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 28), 'part', False)
                    # Processing the call keyword arguments (line 479)
                    kwargs_15833 = {}
                    # Getting the type of 'this' (line 479)
                    this_15830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'this', False)
                    # Obtaining the member 'append' of a type (line 479)
                    append_15831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 16), this_15830, 'append')
                    # Calling append(args, kwargs) (line 479)
                    append_call_result_15834 = invoke(stypy.reporting.localization.Localization(__file__, 479, 16), append_15831, *[part_15832], **kwargs_15833)
                    
                    
                    # Getting the type of 'linelen' (line 480)
                    linelen_15835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'linelen')
                    # Getting the type of 'partlen' (line 480)
                    partlen_15836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 27), 'partlen')
                    # Applying the binary operator '+=' (line 480)
                    result_iadd_15837 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 16), '+=', linelen_15835, partlen_15836)
                    # Assigning a type to the variable 'linelen' (line 480)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'linelen', result_iadd_15837)
                    
                else:
                    
                    # Testing the type of an if condition (line 463)
                    if_condition_15774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 17), result_gt_15773)
                    # Assigning a type to the variable 'if_condition_15774' (line 463)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 17), 'if_condition_15774', if_condition_15774)
                    # SSA begins for if statement (line 463)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'this' (line 464)
                    this_15775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 19), 'this')
                    # Testing if the type of an if condition is none (line 464)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 464, 16), this_15775):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 464)
                        if_condition_15776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 464, 16), this_15775)
                        # Assigning a type to the variable 'if_condition_15776' (line 464)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'if_condition_15776', if_condition_15776)
                        # SSA begins for if statement (line 464)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 465)
                        # Processing the call arguments (line 465)
                        
                        # Call to join(...): (line 465)
                        # Processing the call arguments (line 465)
                        # Getting the type of 'this' (line 465)
                        this_15781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 45), 'this', False)
                        # Processing the call keyword arguments (line 465)
                        kwargs_15782 = {}
                        # Getting the type of 'joiner' (line 465)
                        joiner_15779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 33), 'joiner', False)
                        # Obtaining the member 'join' of a type (line 465)
                        join_15780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 33), joiner_15779, 'join')
                        # Calling join(args, kwargs) (line 465)
                        join_call_result_15783 = invoke(stypy.reporting.localization.Localization(__file__, 465, 33), join_15780, *[this_15781], **kwargs_15782)
                        
                        # Getting the type of 'eol' (line 465)
                        eol_15784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 53), 'eol', False)
                        # Applying the binary operator '+' (line 465)
                        result_add_15785 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 33), '+', join_call_result_15783, eol_15784)
                        
                        # Processing the call keyword arguments (line 465)
                        kwargs_15786 = {}
                        # Getting the type of 'lines' (line 465)
                        lines_15777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 20), 'lines', False)
                        # Obtaining the member 'append' of a type (line 465)
                        append_15778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 20), lines_15777, 'append')
                        # Calling append(args, kwargs) (line 465)
                        append_call_result_15787 = invoke(stypy.reporting.localization.Localization(__file__, 465, 20), append_15778, *[result_add_15785], **kwargs_15786)
                        
                        # SSA join for if statement (line 464)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'partlen' (line 469)
                    partlen_15788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 19), 'partlen')
                    # Getting the type of 'maxlen' (line 469)
                    maxlen_15789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 29), 'maxlen')
                    # Applying the binary operator '>' (line 469)
                    result_gt_15790 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 19), '>', partlen_15788, maxlen_15789)
                    
                    
                    # Getting the type of 'ch' (line 469)
                    ch_15791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 40), 'ch')
                    str_15792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 46), 'str', ' ')
                    # Applying the binary operator '!=' (line 469)
                    result_ne_15793 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 40), '!=', ch_15791, str_15792)
                    
                    # Applying the binary operator 'and' (line 469)
                    result_and_keyword_15794 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 19), 'and', result_gt_15790, result_ne_15793)
                    
                    # Testing if the type of an if condition is none (line 469)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 469, 16), result_and_keyword_15794):
                        
                        # Assigning a List to a Name (line 475):
                        
                        # Assigning a List to a Name (line 475):
                        
                        # Obtaining an instance of the builtin type 'list' (line 475)
                        list_15818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 27), 'list')
                        # Adding type elements to the builtin type 'list' instance (line 475)
                        # Adding element type (line 475)
                        # Getting the type of 'part' (line 475)
                        part_15819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 28), 'part')
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 27), list_15818, part_15819)
                        
                        # Assigning a type to the variable 'this' (line 475)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 20), 'this', list_15818)
                    else:
                        
                        # Testing the type of an if condition (line 469)
                        if_condition_15795 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 469, 16), result_and_keyword_15794)
                        # Assigning a type to the variable 'if_condition_15795' (line 469)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 16), 'if_condition_15795', if_condition_15795)
                        # SSA begins for if statement (line 469)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 470):
                        
                        # Assigning a Call to a Name (line 470):
                        
                        # Call to _split_ascii(...): (line 470)
                        # Processing the call arguments (line 470)
                        # Getting the type of 'part' (line 470)
                        part_15797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 40), 'part', False)
                        # Getting the type of 'maxlen' (line 470)
                        maxlen_15798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 46), 'maxlen', False)
                        # Getting the type of 'restlen' (line 470)
                        restlen_15799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 54), 'restlen', False)
                        # Getting the type of 'continuation_ws' (line 471)
                        continuation_ws_15800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 40), 'continuation_ws', False)
                        str_15801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 57), 'str', ' ')
                        # Processing the call keyword arguments (line 470)
                        kwargs_15802 = {}
                        # Getting the type of '_split_ascii' (line 470)
                        _split_ascii_15796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 27), '_split_ascii', False)
                        # Calling _split_ascii(args, kwargs) (line 470)
                        _split_ascii_call_result_15803 = invoke(stypy.reporting.localization.Localization(__file__, 470, 27), _split_ascii_15796, *[part_15797, maxlen_15798, restlen_15799, continuation_ws_15800, str_15801], **kwargs_15802)
                        
                        # Assigning a type to the variable 'subl' (line 470)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 20), 'subl', _split_ascii_call_result_15803)
                        
                        # Call to extend(...): (line 472)
                        # Processing the call arguments (line 472)
                        
                        # Obtaining the type of the subscript
                        int_15806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 39), 'int')
                        slice_15807 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 472, 33), None, int_15806, None)
                        # Getting the type of 'subl' (line 472)
                        subl_15808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 33), 'subl', False)
                        # Obtaining the member '__getitem__' of a type (line 472)
                        getitem___15809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 33), subl_15808, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 472)
                        subscript_call_result_15810 = invoke(stypy.reporting.localization.Localization(__file__, 472, 33), getitem___15809, slice_15807)
                        
                        # Processing the call keyword arguments (line 472)
                        kwargs_15811 = {}
                        # Getting the type of 'lines' (line 472)
                        lines_15804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 20), 'lines', False)
                        # Obtaining the member 'extend' of a type (line 472)
                        extend_15805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 20), lines_15804, 'extend')
                        # Calling extend(args, kwargs) (line 472)
                        extend_call_result_15812 = invoke(stypy.reporting.localization.Localization(__file__, 472, 20), extend_15805, *[subscript_call_result_15810], **kwargs_15811)
                        
                        
                        # Assigning a List to a Name (line 473):
                        
                        # Assigning a List to a Name (line 473):
                        
                        # Obtaining an instance of the builtin type 'list' (line 473)
                        list_15813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 27), 'list')
                        # Adding type elements to the builtin type 'list' instance (line 473)
                        # Adding element type (line 473)
                        
                        # Obtaining the type of the subscript
                        int_15814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 33), 'int')
                        # Getting the type of 'subl' (line 473)
                        subl_15815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 28), 'subl')
                        # Obtaining the member '__getitem__' of a type (line 473)
                        getitem___15816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 28), subl_15815, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 473)
                        subscript_call_result_15817 = invoke(stypy.reporting.localization.Localization(__file__, 473, 28), getitem___15816, int_15814)
                        
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 27), list_15813, subscript_call_result_15817)
                        
                        # Assigning a type to the variable 'this' (line 473)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 20), 'this', list_15813)
                        # SSA branch for the else part of an if statement (line 469)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a List to a Name (line 475):
                        
                        # Assigning a List to a Name (line 475):
                        
                        # Obtaining an instance of the builtin type 'list' (line 475)
                        list_15818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 27), 'list')
                        # Adding type elements to the builtin type 'list' instance (line 475)
                        # Adding element type (line 475)
                        # Getting the type of 'part' (line 475)
                        part_15819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 28), 'part')
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 27), list_15818, part_15819)
                        
                        # Assigning a type to the variable 'this' (line 475)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 20), 'this', list_15818)
                        # SSA join for if statement (line 469)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Assigning a BinOp to a Name (line 476):
                    
                    # Assigning a BinOp to a Name (line 476):
                    # Getting the type of 'wslen' (line 476)
                    wslen_15820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 26), 'wslen')
                    
                    # Call to len(...): (line 476)
                    # Processing the call arguments (line 476)
                    
                    # Obtaining the type of the subscript
                    int_15822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 43), 'int')
                    # Getting the type of 'this' (line 476)
                    this_15823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 38), 'this', False)
                    # Obtaining the member '__getitem__' of a type (line 476)
                    getitem___15824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 38), this_15823, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
                    subscript_call_result_15825 = invoke(stypy.reporting.localization.Localization(__file__, 476, 38), getitem___15824, int_15822)
                    
                    # Processing the call keyword arguments (line 476)
                    kwargs_15826 = {}
                    # Getting the type of 'len' (line 476)
                    len_15821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 34), 'len', False)
                    # Calling len(args, kwargs) (line 476)
                    len_call_result_15827 = invoke(stypy.reporting.localization.Localization(__file__, 476, 34), len_15821, *[subscript_call_result_15825], **kwargs_15826)
                    
                    # Applying the binary operator '+' (line 476)
                    result_add_15828 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 26), '+', wslen_15820, len_call_result_15827)
                    
                    # Assigning a type to the variable 'linelen' (line 476)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 16), 'linelen', result_add_15828)
                    
                    # Assigning a Name to a Name (line 477):
                    
                    # Assigning a Name to a Name (line 477):
                    # Getting the type of 'restlen' (line 477)
                    restlen_15829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 25), 'restlen')
                    # Assigning a type to the variable 'maxlen' (line 477)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 16), 'maxlen', restlen_15829)
                    # SSA branch for the else part of an if statement (line 463)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to append(...): (line 479)
                    # Processing the call arguments (line 479)
                    # Getting the type of 'part' (line 479)
                    part_15832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 28), 'part', False)
                    # Processing the call keyword arguments (line 479)
                    kwargs_15833 = {}
                    # Getting the type of 'this' (line 479)
                    this_15830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'this', False)
                    # Obtaining the member 'append' of a type (line 479)
                    append_15831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 16), this_15830, 'append')
                    # Calling append(args, kwargs) (line 479)
                    append_call_result_15834 = invoke(stypy.reporting.localization.Localization(__file__, 479, 16), append_15831, *[part_15832], **kwargs_15833)
                    
                    
                    # Getting the type of 'linelen' (line 480)
                    linelen_15835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'linelen')
                    # Getting the type of 'partlen' (line 480)
                    partlen_15836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 27), 'partlen')
                    # Applying the binary operator '+=' (line 480)
                    result_iadd_15837 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 16), '+=', linelen_15835, partlen_15836)
                    # Assigning a type to the variable 'linelen' (line 480)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'linelen', result_iadd_15837)
                    
                    # SSA join for if statement (line 463)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 459)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'this' (line 482)
        this_15838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 11), 'this')
        # Testing if the type of an if condition is none (line 482)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 482, 8), this_15838):
            pass
        else:
            
            # Testing the type of an if condition (line 482)
            if_condition_15839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 482, 8), this_15838)
            # Assigning a type to the variable 'if_condition_15839' (line 482)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'if_condition_15839', if_condition_15839)
            # SSA begins for if statement (line 482)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 483)
            # Processing the call arguments (line 483)
            
            # Call to join(...): (line 483)
            # Processing the call arguments (line 483)
            # Getting the type of 'this' (line 483)
            this_15844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 37), 'this', False)
            # Processing the call keyword arguments (line 483)
            kwargs_15845 = {}
            # Getting the type of 'joiner' (line 483)
            joiner_15842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 25), 'joiner', False)
            # Obtaining the member 'join' of a type (line 483)
            join_15843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 25), joiner_15842, 'join')
            # Calling join(args, kwargs) (line 483)
            join_call_result_15846 = invoke(stypy.reporting.localization.Localization(__file__, 483, 25), join_15843, *[this_15844], **kwargs_15845)
            
            # Processing the call keyword arguments (line 483)
            kwargs_15847 = {}
            # Getting the type of 'lines' (line 483)
            lines_15840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'lines', False)
            # Obtaining the member 'append' of a type (line 483)
            append_15841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 12), lines_15840, 'append')
            # Calling append(args, kwargs) (line 483)
            append_call_result_15848 = invoke(stypy.reporting.localization.Localization(__file__, 483, 12), append_15841, *[join_call_result_15846], **kwargs_15847)
            
            # SSA join for if statement (line 482)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'lines' (line 484)
    lines_15849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 11), 'lines')
    # Assigning a type to the variable 'stypy_return_type' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'stypy_return_type', lines_15849)
    
    # ################# End of '_split_ascii(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_split_ascii' in the type store
    # Getting the type of 'stypy_return_type' (line 418)
    stypy_return_type_15850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15850)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_split_ascii'
    return stypy_return_type_15850

# Assigning a type to the variable '_split_ascii' (line 418)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 0), '_split_ascii', _split_ascii)

@norecursion
def _binsplit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_binsplit'
    module_type_store = module_type_store.open_function_context('_binsplit', 488, 0, False)
    
    # Passed parameters checking function
    _binsplit.stypy_localization = localization
    _binsplit.stypy_type_of_self = None
    _binsplit.stypy_type_store = module_type_store
    _binsplit.stypy_function_name = '_binsplit'
    _binsplit.stypy_param_names_list = ['splittable', 'charset', 'maxlinelen']
    _binsplit.stypy_varargs_param_name = None
    _binsplit.stypy_kwargs_param_name = None
    _binsplit.stypy_call_defaults = defaults
    _binsplit.stypy_call_varargs = varargs
    _binsplit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_binsplit', ['splittable', 'charset', 'maxlinelen'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_binsplit', localization, ['splittable', 'charset', 'maxlinelen'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_binsplit(...)' code ##################

    
    # Assigning a Num to a Name (line 489):
    
    # Assigning a Num to a Name (line 489):
    int_15851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 8), 'int')
    # Assigning a type to the variable 'i' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'i', int_15851)
    
    # Assigning a Call to a Name (line 490):
    
    # Assigning a Call to a Name (line 490):
    
    # Call to len(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'splittable' (line 490)
    splittable_15853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'splittable', False)
    # Processing the call keyword arguments (line 490)
    kwargs_15854 = {}
    # Getting the type of 'len' (line 490)
    len_15852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'len', False)
    # Calling len(args, kwargs) (line 490)
    len_call_result_15855 = invoke(stypy.reporting.localization.Localization(__file__, 490, 8), len_15852, *[splittable_15853], **kwargs_15854)
    
    # Assigning a type to the variable 'j' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'j', len_call_result_15855)
    
    
    # Getting the type of 'i' (line 491)
    i_15856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 10), 'i')
    # Getting the type of 'j' (line 491)
    j_15857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 14), 'j')
    # Applying the binary operator '<' (line 491)
    result_lt_15858 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 10), '<', i_15856, j_15857)
    
    # Assigning a type to the variable 'result_lt_15858' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'result_lt_15858', result_lt_15858)
    # Testing if the while is going to be iterated (line 491)
    # Testing the type of an if condition (line 491)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 491, 4), result_lt_15858)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 491, 4), result_lt_15858):
        # SSA begins for while statement (line 491)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 500):
        
        # Assigning a BinOp to a Name (line 500):
        # Getting the type of 'i' (line 500)
        i_15859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 13), 'i')
        # Getting the type of 'j' (line 500)
        j_15860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'j')
        # Applying the binary operator '+' (line 500)
        result_add_15861 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 13), '+', i_15859, j_15860)
        
        int_15862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 17), 'int')
        # Applying the binary operator '+' (line 500)
        result_add_15863 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 16), '+', result_add_15861, int_15862)
        
        int_15864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 23), 'int')
        # Applying the binary operator '>>' (line 500)
        result_rshift_15865 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 12), '>>', result_add_15863, int_15864)
        
        # Assigning a type to the variable 'm' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'm', result_rshift_15865)
        
        # Assigning a Call to a Name (line 501):
        
        # Assigning a Call to a Name (line 501):
        
        # Call to from_splittable(...): (line 501)
        # Processing the call arguments (line 501)
        
        # Obtaining the type of the subscript
        # Getting the type of 'm' (line 501)
        m_15868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 52), 'm', False)
        slice_15869 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 501, 40), None, m_15868, None)
        # Getting the type of 'splittable' (line 501)
        splittable_15870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 40), 'splittable', False)
        # Obtaining the member '__getitem__' of a type (line 501)
        getitem___15871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 40), splittable_15870, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 501)
        subscript_call_result_15872 = invoke(stypy.reporting.localization.Localization(__file__, 501, 40), getitem___15871, slice_15869)
        
        # Getting the type of 'True' (line 501)
        True_15873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 56), 'True', False)
        # Processing the call keyword arguments (line 501)
        kwargs_15874 = {}
        # Getting the type of 'charset' (line 501)
        charset_15866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'charset', False)
        # Obtaining the member 'from_splittable' of a type (line 501)
        from_splittable_15867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 16), charset_15866, 'from_splittable')
        # Calling from_splittable(args, kwargs) (line 501)
        from_splittable_call_result_15875 = invoke(stypy.reporting.localization.Localization(__file__, 501, 16), from_splittable_15867, *[subscript_call_result_15872, True_15873], **kwargs_15874)
        
        # Assigning a type to the variable 'chunk' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'chunk', from_splittable_call_result_15875)
        
        # Assigning a Call to a Name (line 502):
        
        # Assigning a Call to a Name (line 502):
        
        # Call to encoded_header_len(...): (line 502)
        # Processing the call arguments (line 502)
        # Getting the type of 'chunk' (line 502)
        chunk_15878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 46), 'chunk', False)
        # Processing the call keyword arguments (line 502)
        kwargs_15879 = {}
        # Getting the type of 'charset' (line 502)
        charset_15876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 19), 'charset', False)
        # Obtaining the member 'encoded_header_len' of a type (line 502)
        encoded_header_len_15877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 19), charset_15876, 'encoded_header_len')
        # Calling encoded_header_len(args, kwargs) (line 502)
        encoded_header_len_call_result_15880 = invoke(stypy.reporting.localization.Localization(__file__, 502, 19), encoded_header_len_15877, *[chunk_15878], **kwargs_15879)
        
        # Assigning a type to the variable 'chunklen' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'chunklen', encoded_header_len_call_result_15880)
        
        # Getting the type of 'chunklen' (line 503)
        chunklen_15881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 11), 'chunklen')
        # Getting the type of 'maxlinelen' (line 503)
        maxlinelen_15882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 23), 'maxlinelen')
        # Applying the binary operator '<=' (line 503)
        result_le_15883 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 11), '<=', chunklen_15881, maxlinelen_15882)
        
        # Testing if the type of an if condition is none (line 503)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 503, 8), result_le_15883):
            
            # Assigning a BinOp to a Name (line 508):
            
            # Assigning a BinOp to a Name (line 508):
            # Getting the type of 'm' (line 508)
            m_15886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'm')
            int_15887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 20), 'int')
            # Applying the binary operator '-' (line 508)
            result_sub_15888 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 16), '-', m_15886, int_15887)
            
            # Assigning a type to the variable 'j' (line 508)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'j', result_sub_15888)
        else:
            
            # Testing the type of an if condition (line 503)
            if_condition_15884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 503, 8), result_le_15883)
            # Assigning a type to the variable 'if_condition_15884' (line 503)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'if_condition_15884', if_condition_15884)
            # SSA begins for if statement (line 503)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 505):
            
            # Assigning a Name to a Name (line 505):
            # Getting the type of 'm' (line 505)
            m_15885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 16), 'm')
            # Assigning a type to the variable 'i' (line 505)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'i', m_15885)
            # SSA branch for the else part of an if statement (line 503)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 508):
            
            # Assigning a BinOp to a Name (line 508):
            # Getting the type of 'm' (line 508)
            m_15886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'm')
            int_15887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 20), 'int')
            # Applying the binary operator '-' (line 508)
            result_sub_15888 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 16), '-', m_15886, int_15887)
            
            # Assigning a type to the variable 'j' (line 508)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'j', result_sub_15888)
            # SSA join for if statement (line 503)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for while statement (line 491)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Call to a Name (line 512):
    
    # Assigning a Call to a Name (line 512):
    
    # Call to from_splittable(...): (line 512)
    # Processing the call arguments (line 512)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 512)
    i_15891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 48), 'i', False)
    slice_15892 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 512, 36), None, i_15891, None)
    # Getting the type of 'splittable' (line 512)
    splittable_15893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 36), 'splittable', False)
    # Obtaining the member '__getitem__' of a type (line 512)
    getitem___15894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 36), splittable_15893, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 512)
    subscript_call_result_15895 = invoke(stypy.reporting.localization.Localization(__file__, 512, 36), getitem___15894, slice_15892)
    
    # Getting the type of 'False' (line 512)
    False_15896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 52), 'False', False)
    # Processing the call keyword arguments (line 512)
    kwargs_15897 = {}
    # Getting the type of 'charset' (line 512)
    charset_15889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'charset', False)
    # Obtaining the member 'from_splittable' of a type (line 512)
    from_splittable_15890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 12), charset_15889, 'from_splittable')
    # Calling from_splittable(args, kwargs) (line 512)
    from_splittable_call_result_15898 = invoke(stypy.reporting.localization.Localization(__file__, 512, 12), from_splittable_15890, *[subscript_call_result_15895, False_15896], **kwargs_15897)
    
    # Assigning a type to the variable 'first' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'first', from_splittable_call_result_15898)
    
    # Assigning a Call to a Name (line 513):
    
    # Assigning a Call to a Name (line 513):
    
    # Call to from_splittable(...): (line 513)
    # Processing the call arguments (line 513)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 513)
    i_15901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 47), 'i', False)
    slice_15902 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 513, 36), i_15901, None, None)
    # Getting the type of 'splittable' (line 513)
    splittable_15903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 36), 'splittable', False)
    # Obtaining the member '__getitem__' of a type (line 513)
    getitem___15904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 36), splittable_15903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 513)
    subscript_call_result_15905 = invoke(stypy.reporting.localization.Localization(__file__, 513, 36), getitem___15904, slice_15902)
    
    # Getting the type of 'False' (line 513)
    False_15906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 52), 'False', False)
    # Processing the call keyword arguments (line 513)
    kwargs_15907 = {}
    # Getting the type of 'charset' (line 513)
    charset_15899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'charset', False)
    # Obtaining the member 'from_splittable' of a type (line 513)
    from_splittable_15900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 12), charset_15899, 'from_splittable')
    # Calling from_splittable(args, kwargs) (line 513)
    from_splittable_call_result_15908 = invoke(stypy.reporting.localization.Localization(__file__, 513, 12), from_splittable_15900, *[subscript_call_result_15905, False_15906], **kwargs_15907)
    
    # Assigning a type to the variable 'last' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'last', from_splittable_call_result_15908)
    
    # Obtaining an instance of the builtin type 'tuple' (line 514)
    tuple_15909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 514)
    # Adding element type (line 514)
    # Getting the type of 'first' (line 514)
    first_15910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 11), 'first')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 11), tuple_15909, first_15910)
    # Adding element type (line 514)
    # Getting the type of 'last' (line 514)
    last_15911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 18), 'last')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 11), tuple_15909, last_15911)
    
    # Assigning a type to the variable 'stypy_return_type' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'stypy_return_type', tuple_15909)
    
    # ################# End of '_binsplit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_binsplit' in the type store
    # Getting the type of 'stypy_return_type' (line 488)
    stypy_return_type_15912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15912)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_binsplit'
    return stypy_return_type_15912

# Assigning a type to the variable '_binsplit' (line 488)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 0), '_binsplit', _binsplit)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
