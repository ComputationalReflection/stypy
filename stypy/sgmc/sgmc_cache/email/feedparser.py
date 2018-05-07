
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2004-2006 Python Software Foundation
2: # Authors: Baxter, Wouters and Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''FeedParser - An email feed parser.
6: 
7: The feed parser implements an interface for incrementally parsing an email
8: message, line by line.  This has advantages for certain applications, such as
9: those reading email messages off a socket.
10: 
11: FeedParser.feed() is the primary interface for pushing new data into the
12: parser.  It returns when there's nothing more it can do with the available
13: data.  When you have no more data to push into the parser, call .close().
14: This completes the parsing and returns the root message object.
15: 
16: The other advantage of this parser is that it will never raise a parsing
17: exception.  Instead, when it finds something unexpected, it adds a 'defect' to
18: the current message.  Defects are just instances that live on the message
19: object's .defects attribute.
20: '''
21: 
22: __all__ = ['FeedParser']
23: 
24: import re
25: 
26: from email import errors
27: from email import message
28: 
29: NLCRE = re.compile('\r\n|\r|\n')
30: NLCRE_bol = re.compile('(\r\n|\r|\n)')
31: NLCRE_eol = re.compile('(\r\n|\r|\n)\Z')
32: NLCRE_crack = re.compile('(\r\n|\r|\n)')
33: # RFC 2822 $3.6.8 Optional fields.  ftext is %d33-57 / %d59-126, Any character
34: # except controls, SP, and ":".
35: headerRE = re.compile(r'^(From |[\041-\071\073-\176]{1,}:|[\t ])')
36: EMPTYSTRING = ''
37: NL = '\n'
38: 
39: NeedMoreData = object()
40: 
41: 
42: 
43: class BufferedSubFile(object):
44:     '''A file-ish object that can have new data loaded into it.
45: 
46:     You can also push and pop line-matching predicates onto a stack.  When the
47:     current predicate matches the current line, a false EOF response
48:     (i.e. empty string) is returned instead.  This lets the parser adhere to a
49:     simple abstraction -- it parses until EOF closes the current message.
50:     '''
51:     def __init__(self):
52:         # Chunks of the last partial line pushed into this object.
53:         self._partial = []
54:         # The list of full, pushed lines, in reverse order
55:         self._lines = []
56:         # The stack of false-EOF checking predicates.
57:         self._eofstack = []
58:         # A flag indicating whether the file has been closed or not.
59:         self._closed = False
60: 
61:     def push_eof_matcher(self, pred):
62:         self._eofstack.append(pred)
63: 
64:     def pop_eof_matcher(self):
65:         return self._eofstack.pop()
66: 
67:     def close(self):
68:         # Don't forget any trailing partial line.
69:         self.pushlines(''.join(self._partial).splitlines(True))
70:         self._partial = []
71:         self._closed = True
72: 
73:     def readline(self):
74:         if not self._lines:
75:             if self._closed:
76:                 return ''
77:             return NeedMoreData
78:         # Pop the line off the stack and see if it matches the current
79:         # false-EOF predicate.
80:         line = self._lines.pop()
81:         # RFC 2046, section 5.1.2 requires us to recognize outer level
82:         # boundaries at any level of inner nesting.  Do this, but be sure it's
83:         # in the order of most to least nested.
84:         for ateof in self._eofstack[::-1]:
85:             if ateof(line):
86:                 # We're at the false EOF.  But push the last line back first.
87:                 self._lines.append(line)
88:                 return ''
89:         return line
90: 
91:     def unreadline(self, line):
92:         # Let the consumer push a line back into the buffer.
93:         assert line is not NeedMoreData
94:         self._lines.append(line)
95: 
96:     def push(self, data):
97:         '''Push some new data into this object.'''
98:         # Crack into lines, but preserve the linesep characters on the end of each
99:         parts = data.splitlines(True)
100: 
101:         if not parts or not parts[0].endswith(('\n', '\r')):
102:             # No new complete lines, so just accumulate partials
103:             self._partial += parts
104:             return
105: 
106:         if self._partial:
107:             # If there are previous leftovers, complete them now
108:             self._partial.append(parts[0])
109:             parts[0:1] = ''.join(self._partial).splitlines(True)
110:             del self._partial[:]
111: 
112:         # If the last element of the list does not end in a newline, then treat
113:         # it as a partial line.  We only check for '\n' here because a line
114:         # ending with '\r' might be a line that was split in the middle of a
115:         # '\r\n' sequence (see bugs 1555570 and 1721862).
116:         if not parts[-1].endswith('\n'):
117:             self._partial = [parts.pop()]
118:         self.pushlines(parts)
119: 
120:     def pushlines(self, lines):
121:         # Crack into lines, but preserve the newlines on the end of each
122:         parts = NLCRE_crack.split(data)
123:         # The *ahem* interesting behaviour of re.split when supplied grouping
124:         # parentheses is that the last element of the resulting list is the
125:         # data after the final RE.  In the case of a NL/CR terminated string,
126:         # this is the empty string.
127:         self._partial = parts.pop()
128:         #GAN 29Mar09  bugs 1555570, 1721862  Confusion at 8K boundary ending with \r:
129:         # is there a \n to follow later?
130:         if not self._partial and parts and parts[-1].endswith('\r'):
131:             self._partial = parts.pop(-2)+parts.pop()
132:         # parts is a list of strings, alternating between the line contents
133:         # and the eol character(s).  Gather up a list of lines after
134:         # re-attaching the newlines.
135:         lines = []
136:         for i in range(len(parts) // 2):
137:             lines.append(parts[i*2] + parts[i*2+1])
138:         self.pushlines(lines)
139: 
140:     def pushlines(self, lines):
141:         # Reverse and insert at the front of the lines.
142:         self._lines[:0] = lines[::-1]
143: 
144:     def is_closed(self):
145:         return self._closed
146: 
147:     def __iter__(self):
148:         return self
149: 
150:     def next(self):
151:         line = self.readline()
152:         if line == '':
153:             raise StopIteration
154:         return line
155: 
156: 
157: 
158: class FeedParser:
159:     '''A feed-style parser of email.'''
160: 
161:     def __init__(self, _factory=message.Message):
162:         '''_factory is called with no arguments to create a new message obj'''
163:         self._factory = _factory
164:         self._input = BufferedSubFile()
165:         self._msgstack = []
166:         self._parse = self._parsegen().next
167:         self._cur = None
168:         self._last = None
169:         self._headersonly = False
170: 
171:     # Non-public interface for supporting Parser's headersonly flag
172:     def _set_headersonly(self):
173:         self._headersonly = True
174: 
175:     def feed(self, data):
176:         '''Push more data into the parser.'''
177:         self._input.push(data)
178:         self._call_parse()
179: 
180:     def _call_parse(self):
181:         try:
182:             self._parse()
183:         except StopIteration:
184:             pass
185: 
186:     def close(self):
187:         '''Parse all remaining data and return the root message object.'''
188:         self._input.close()
189:         self._call_parse()
190:         root = self._pop_message()
191:         assert not self._msgstack
192:         # Look for final set of defects
193:         if root.get_content_maintype() == 'multipart' \
194:                and not root.is_multipart():
195:             root.defects.append(errors.MultipartInvariantViolationDefect())
196:         return root
197: 
198:     def _new_message(self):
199:         msg = self._factory()
200:         if self._cur and self._cur.get_content_type() == 'multipart/digest':
201:             msg.set_default_type('message/rfc822')
202:         if self._msgstack:
203:             self._msgstack[-1].attach(msg)
204:         self._msgstack.append(msg)
205:         self._cur = msg
206:         self._last = msg
207: 
208:     def _pop_message(self):
209:         retval = self._msgstack.pop()
210:         if self._msgstack:
211:             self._cur = self._msgstack[-1]
212:         else:
213:             self._cur = None
214:         return retval
215: 
216:     def _parsegen(self):
217:         # Create a new message and start by parsing headers.
218:         self._new_message()
219:         headers = []
220:         # Collect the headers, searching for a line that doesn't match the RFC
221:         # 2822 header or continuation pattern (including an empty line).
222:         for line in self._input:
223:             if line is NeedMoreData:
224:                 yield NeedMoreData
225:                 continue
226:             if not headerRE.match(line):
227:                 # If we saw the RFC defined header/body separator
228:                 # (i.e. newline), just throw it away. Otherwise the line is
229:                 # part of the body so push it back.
230:                 if not NLCRE.match(line):
231:                     self._input.unreadline(line)
232:                 break
233:             headers.append(line)
234:         # Done with the headers, so parse them and figure out what we're
235:         # supposed to see in the body of the message.
236:         self._parse_headers(headers)
237:         # Headers-only parsing is a backwards compatibility hack, which was
238:         # necessary in the older parser, which could raise errors.  All
239:         # remaining lines in the input are thrown into the message body.
240:         if self._headersonly:
241:             lines = []
242:             while True:
243:                 line = self._input.readline()
244:                 if line is NeedMoreData:
245:                     yield NeedMoreData
246:                     continue
247:                 if line == '':
248:                     break
249:                 lines.append(line)
250:             self._cur.set_payload(EMPTYSTRING.join(lines))
251:             return
252:         if self._cur.get_content_type() == 'message/delivery-status':
253:             # message/delivery-status contains blocks of headers separated by
254:             # a blank line.  We'll represent each header block as a separate
255:             # nested message object, but the processing is a bit different
256:             # than standard message/* types because there is no body for the
257:             # nested messages.  A blank line separates the subparts.
258:             while True:
259:                 self._input.push_eof_matcher(NLCRE.match)
260:                 for retval in self._parsegen():
261:                     if retval is NeedMoreData:
262:                         yield NeedMoreData
263:                         continue
264:                     break
265:                 msg = self._pop_message()
266:                 # We need to pop the EOF matcher in order to tell if we're at
267:                 # the end of the current file, not the end of the last block
268:                 # of message headers.
269:                 self._input.pop_eof_matcher()
270:                 # The input stream must be sitting at the newline or at the
271:                 # EOF.  We want to see if we're at the end of this subpart, so
272:                 # first consume the blank line, then test the next line to see
273:                 # if we're at this subpart's EOF.
274:                 while True:
275:                     line = self._input.readline()
276:                     if line is NeedMoreData:
277:                         yield NeedMoreData
278:                         continue
279:                     break
280:                 while True:
281:                     line = self._input.readline()
282:                     if line is NeedMoreData:
283:                         yield NeedMoreData
284:                         continue
285:                     break
286:                 if line == '':
287:                     break
288:                 # Not at EOF so this is a line we're going to need.
289:                 self._input.unreadline(line)
290:             return
291:         if self._cur.get_content_maintype() == 'message':
292:             # The message claims to be a message/* type, then what follows is
293:             # another RFC 2822 message.
294:             for retval in self._parsegen():
295:                 if retval is NeedMoreData:
296:                     yield NeedMoreData
297:                     continue
298:                 break
299:             self._pop_message()
300:             return
301:         if self._cur.get_content_maintype() == 'multipart':
302:             boundary = self._cur.get_boundary()
303:             if boundary is None:
304:                 # The message /claims/ to be a multipart but it has not
305:                 # defined a boundary.  That's a problem which we'll handle by
306:                 # reading everything until the EOF and marking the message as
307:                 # defective.
308:                 self._cur.defects.append(errors.NoBoundaryInMultipartDefect())
309:                 lines = []
310:                 for line in self._input:
311:                     if line is NeedMoreData:
312:                         yield NeedMoreData
313:                         continue
314:                     lines.append(line)
315:                 self._cur.set_payload(EMPTYSTRING.join(lines))
316:                 return
317:             # Create a line match predicate which matches the inter-part
318:             # boundary as well as the end-of-multipart boundary.  Don't push
319:             # this onto the input stream until we've scanned past the
320:             # preamble.
321:             separator = '--' + boundary
322:             boundaryre = re.compile(
323:                 '(?P<sep>' + re.escape(separator) +
324:                 r')(?P<end>--)?(?P<ws>[ \t]*)(?P<linesep>\r\n|\r|\n)?$')
325:             capturing_preamble = True
326:             preamble = []
327:             linesep = False
328:             while True:
329:                 line = self._input.readline()
330:                 if line is NeedMoreData:
331:                     yield NeedMoreData
332:                     continue
333:                 if line == '':
334:                     break
335:                 mo = boundaryre.match(line)
336:                 if mo:
337:                     # If we're looking at the end boundary, we're done with
338:                     # this multipart.  If there was a newline at the end of
339:                     # the closing boundary, then we need to initialize the
340:                     # epilogue with the empty string (see below).
341:                     if mo.group('end'):
342:                         linesep = mo.group('linesep')
343:                         break
344:                     # We saw an inter-part boundary.  Were we in the preamble?
345:                     if capturing_preamble:
346:                         if preamble:
347:                             # According to RFC 2046, the last newline belongs
348:                             # to the boundary.
349:                             lastline = preamble[-1]
350:                             eolmo = NLCRE_eol.search(lastline)
351:                             if eolmo:
352:                                 preamble[-1] = lastline[:-len(eolmo.group(0))]
353:                             self._cur.preamble = EMPTYSTRING.join(preamble)
354:                         capturing_preamble = False
355:                         self._input.unreadline(line)
356:                         continue
357:                     # We saw a boundary separating two parts.  Consume any
358:                     # multiple boundary lines that may be following.  Our
359:                     # interpretation of RFC 2046 BNF grammar does not produce
360:                     # body parts within such double boundaries.
361:                     while True:
362:                         line = self._input.readline()
363:                         if line is NeedMoreData:
364:                             yield NeedMoreData
365:                             continue
366:                         mo = boundaryre.match(line)
367:                         if not mo:
368:                             self._input.unreadline(line)
369:                             break
370:                     # Recurse to parse this subpart; the input stream points
371:                     # at the subpart's first line.
372:                     self._input.push_eof_matcher(boundaryre.match)
373:                     for retval in self._parsegen():
374:                         if retval is NeedMoreData:
375:                             yield NeedMoreData
376:                             continue
377:                         break
378:                     # Because of RFC 2046, the newline preceding the boundary
379:                     # separator actually belongs to the boundary, not the
380:                     # previous subpart's payload (or epilogue if the previous
381:                     # part is a multipart).
382:                     if self._last.get_content_maintype() == 'multipart':
383:                         epilogue = self._last.epilogue
384:                         if epilogue == '':
385:                             self._last.epilogue = None
386:                         elif epilogue is not None:
387:                             mo = NLCRE_eol.search(epilogue)
388:                             if mo:
389:                                 end = len(mo.group(0))
390:                                 self._last.epilogue = epilogue[:-end]
391:                     else:
392:                         payload = self._last.get_payload()
393:                         if isinstance(payload, basestring):
394:                             mo = NLCRE_eol.search(payload)
395:                             if mo:
396:                                 payload = payload[:-len(mo.group(0))]
397:                                 self._last.set_payload(payload)
398:                     self._input.pop_eof_matcher()
399:                     self._pop_message()
400:                     # Set the multipart up for newline cleansing, which will
401:                     # happen if we're in a nested multipart.
402:                     self._last = self._cur
403:                 else:
404:                     # I think we must be in the preamble
405:                     assert capturing_preamble
406:                     preamble.append(line)
407:             # We've seen either the EOF or the end boundary.  If we're still
408:             # capturing the preamble, we never saw the start boundary.  Note
409:             # that as a defect and store the captured text as the payload.
410:             # Everything from here to the EOF is epilogue.
411:             if capturing_preamble:
412:                 self._cur.defects.append(errors.StartBoundaryNotFoundDefect())
413:                 self._cur.set_payload(EMPTYSTRING.join(preamble))
414:                 epilogue = []
415:                 for line in self._input:
416:                     if line is NeedMoreData:
417:                         yield NeedMoreData
418:                         continue
419:                 self._cur.epilogue = EMPTYSTRING.join(epilogue)
420:                 return
421:             # If the end boundary ended in a newline, we'll need to make sure
422:             # the epilogue isn't None
423:             if linesep:
424:                 epilogue = ['']
425:             else:
426:                 epilogue = []
427:             for line in self._input:
428:                 if line is NeedMoreData:
429:                     yield NeedMoreData
430:                     continue
431:                 epilogue.append(line)
432:             # Any CRLF at the front of the epilogue is not technically part of
433:             # the epilogue.  Also, watch out for an empty string epilogue,
434:             # which means a single newline.
435:             if epilogue:
436:                 firstline = epilogue[0]
437:                 bolmo = NLCRE_bol.match(firstline)
438:                 if bolmo:
439:                     epilogue[0] = firstline[len(bolmo.group(0)):]
440:             self._cur.epilogue = EMPTYSTRING.join(epilogue)
441:             return
442:         # Otherwise, it's some non-multipart type, so the entire rest of the
443:         # file contents becomes the payload.
444:         lines = []
445:         for line in self._input:
446:             if line is NeedMoreData:
447:                 yield NeedMoreData
448:                 continue
449:             lines.append(line)
450:         self._cur.set_payload(EMPTYSTRING.join(lines))
451: 
452:     def _parse_headers(self, lines):
453:         # Passed a list of lines that make up the headers for the current msg
454:         lastheader = ''
455:         lastvalue = []
456:         for lineno, line in enumerate(lines):
457:             # Check for continuation
458:             if line[0] in ' \t':
459:                 if not lastheader:
460:                     # The first line of the headers was a continuation.  This
461:                     # is illegal, so let's note the defect, store the illegal
462:                     # line, and ignore it for purposes of headers.
463:                     defect = errors.FirstHeaderLineIsContinuationDefect(line)
464:                     self._cur.defects.append(defect)
465:                     continue
466:                 lastvalue.append(line)
467:                 continue
468:             if lastheader:
469:                 # XXX reconsider the joining of folded lines
470:                 lhdr = EMPTYSTRING.join(lastvalue)[:-1].rstrip('\r\n')
471:                 self._cur[lastheader] = lhdr
472:                 lastheader, lastvalue = '', []
473:             # Check for envelope header, i.e. unix-from
474:             if line.startswith('From '):
475:                 if lineno == 0:
476:                     # Strip off the trailing newline
477:                     mo = NLCRE_eol.search(line)
478:                     if mo:
479:                         line = line[:-len(mo.group(0))]
480:                     self._cur.set_unixfrom(line)
481:                     continue
482:                 elif lineno == len(lines) - 1:
483:                     # Something looking like a unix-from at the end - it's
484:                     # probably the first line of the body, so push back the
485:                     # line and stop.
486:                     self._input.unreadline(line)
487:                     return
488:                 else:
489:                     # Weirdly placed unix-from line.  Note this as a defect
490:                     # and ignore it.
491:                     defect = errors.MisplacedEnvelopeHeaderDefect(line)
492:                     self._cur.defects.append(defect)
493:                     continue
494:             # Split the line on the colon separating field name from value.
495:             i = line.find(':')
496:             if i < 0:
497:                 defect = errors.MalformedHeaderDefect(line)
498:                 self._cur.defects.append(defect)
499:                 continue
500:             lastheader = line[:i]
501:             lastvalue = [line[i+1:].lstrip()]
502:         # Done with all the lines, so handle the last header.
503:         if lastheader:
504:             # XXX reconsider the joining of folded lines
505:             self._cur[lastheader] = EMPTYSTRING.join(lastvalue).rstrip('\r\n')
506: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_12947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', "FeedParser - An email feed parser.\n\nThe feed parser implements an interface for incrementally parsing an email\nmessage, line by line.  This has advantages for certain applications, such as\nthose reading email messages off a socket.\n\nFeedParser.feed() is the primary interface for pushing new data into the\nparser.  It returns when there's nothing more it can do with the available\ndata.  When you have no more data to push into the parser, call .close().\nThis completes the parsing and returns the root message object.\n\nThe other advantage of this parser is that it will never raise a parsing\nexception.  Instead, when it finds something unexpected, it adds a 'defect' to\nthe current message.  Defects are just instances that live on the message\nobject's .defects attribute.\n")

# Assigning a List to a Name (line 22):

# Assigning a List to a Name (line 22):
__all__ = ['FeedParser']
module_type_store.set_exportable_members(['FeedParser'])

# Obtaining an instance of the builtin type 'list' (line 22)
list_12948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_12949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'str', 'FeedParser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_12948, str_12949)

# Assigning a type to the variable '__all__' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '__all__', list_12948)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import re' statement (line 24)
import re

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from email import errors' statement (line 26)
try:
    from email import errors

except:
    errors = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'email', None, module_type_store, ['errors'], [errors])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from email import message' statement (line 27)
try:
    from email import message

except:
    message = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'email', None, module_type_store, ['message'], [message])


# Assigning a Call to a Name (line 29):

# Assigning a Call to a Name (line 29):

# Call to compile(...): (line 29)
# Processing the call arguments (line 29)
str_12952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'str', '\r\n|\r|\n')
# Processing the call keyword arguments (line 29)
kwargs_12953 = {}
# Getting the type of 're' (line 29)
re_12950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 're', False)
# Obtaining the member 'compile' of a type (line 29)
compile_12951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), re_12950, 'compile')
# Calling compile(args, kwargs) (line 29)
compile_call_result_12954 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), compile_12951, *[str_12952], **kwargs_12953)

# Assigning a type to the variable 'NLCRE' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'NLCRE', compile_call_result_12954)

# Assigning a Call to a Name (line 30):

# Assigning a Call to a Name (line 30):

# Call to compile(...): (line 30)
# Processing the call arguments (line 30)
str_12957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'str', '(\r\n|\r|\n)')
# Processing the call keyword arguments (line 30)
kwargs_12958 = {}
# Getting the type of 're' (line 30)
re_12955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 're', False)
# Obtaining the member 'compile' of a type (line 30)
compile_12956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), re_12955, 'compile')
# Calling compile(args, kwargs) (line 30)
compile_call_result_12959 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), compile_12956, *[str_12957], **kwargs_12958)

# Assigning a type to the variable 'NLCRE_bol' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'NLCRE_bol', compile_call_result_12959)

# Assigning a Call to a Name (line 31):

# Assigning a Call to a Name (line 31):

# Call to compile(...): (line 31)
# Processing the call arguments (line 31)
str_12962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'str', '(\r\n|\r|\n)\\Z')
# Processing the call keyword arguments (line 31)
kwargs_12963 = {}
# Getting the type of 're' (line 31)
re_12960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 're', False)
# Obtaining the member 'compile' of a type (line 31)
compile_12961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), re_12960, 'compile')
# Calling compile(args, kwargs) (line 31)
compile_call_result_12964 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), compile_12961, *[str_12962], **kwargs_12963)

# Assigning a type to the variable 'NLCRE_eol' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'NLCRE_eol', compile_call_result_12964)

# Assigning a Call to a Name (line 32):

# Assigning a Call to a Name (line 32):

# Call to compile(...): (line 32)
# Processing the call arguments (line 32)
str_12967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'str', '(\r\n|\r|\n)')
# Processing the call keyword arguments (line 32)
kwargs_12968 = {}
# Getting the type of 're' (line 32)
re_12965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 're', False)
# Obtaining the member 'compile' of a type (line 32)
compile_12966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 14), re_12965, 'compile')
# Calling compile(args, kwargs) (line 32)
compile_call_result_12969 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), compile_12966, *[str_12967], **kwargs_12968)

# Assigning a type to the variable 'NLCRE_crack' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'NLCRE_crack', compile_call_result_12969)

# Assigning a Call to a Name (line 35):

# Assigning a Call to a Name (line 35):

# Call to compile(...): (line 35)
# Processing the call arguments (line 35)
str_12972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 22), 'str', '^(From |[\\041-\\071\\073-\\176]{1,}:|[\\t ])')
# Processing the call keyword arguments (line 35)
kwargs_12973 = {}
# Getting the type of 're' (line 35)
re_12970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 're', False)
# Obtaining the member 'compile' of a type (line 35)
compile_12971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 11), re_12970, 'compile')
# Calling compile(args, kwargs) (line 35)
compile_call_result_12974 = invoke(stypy.reporting.localization.Localization(__file__, 35, 11), compile_12971, *[str_12972], **kwargs_12973)

# Assigning a type to the variable 'headerRE' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'headerRE', compile_call_result_12974)

# Assigning a Str to a Name (line 36):

# Assigning a Str to a Name (line 36):
str_12975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'str', '')
# Assigning a type to the variable 'EMPTYSTRING' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'EMPTYSTRING', str_12975)

# Assigning a Str to a Name (line 37):

# Assigning a Str to a Name (line 37):
str_12976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 5), 'str', '\n')
# Assigning a type to the variable 'NL' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'NL', str_12976)

# Assigning a Call to a Name (line 39):

# Assigning a Call to a Name (line 39):

# Call to object(...): (line 39)
# Processing the call keyword arguments (line 39)
kwargs_12978 = {}
# Getting the type of 'object' (line 39)
object_12977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'object', False)
# Calling object(args, kwargs) (line 39)
object_call_result_12979 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), object_12977, *[], **kwargs_12978)

# Assigning a type to the variable 'NeedMoreData' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'NeedMoreData', object_call_result_12979)
# Declaration of the 'BufferedSubFile' class

class BufferedSubFile(object, ):
    str_12980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', 'A file-ish object that can have new data loaded into it.\n\n    You can also push and pop line-matching predicates onto a stack.  When the\n    current predicate matches the current line, a false EOF response\n    (i.e. empty string) is returned instead.  This lets the parser adhere to a\n    simple abstraction -- it parses until EOF closes the current message.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a List to a Attribute (line 53):
        
        # Assigning a List to a Attribute (line 53):
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_12981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        
        # Getting the type of 'self' (line 53)
        self_12982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member '_partial' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_12982, '_partial', list_12981)
        
        # Assigning a List to a Attribute (line 55):
        
        # Assigning a List to a Attribute (line 55):
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_12983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        
        # Getting the type of 'self' (line 55)
        self_12984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member '_lines' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_12984, '_lines', list_12983)
        
        # Assigning a List to a Attribute (line 57):
        
        # Assigning a List to a Attribute (line 57):
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_12985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        
        # Getting the type of 'self' (line 57)
        self_12986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member '_eofstack' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_12986, '_eofstack', list_12985)
        
        # Assigning a Name to a Attribute (line 59):
        
        # Assigning a Name to a Attribute (line 59):
        # Getting the type of 'False' (line 59)
        False_12987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'False')
        # Getting the type of 'self' (line 59)
        self_12988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member '_closed' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_12988, '_closed', False_12987)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def push_eof_matcher(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'push_eof_matcher'
        module_type_store = module_type_store.open_function_context('push_eof_matcher', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.push_eof_matcher')
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_param_names_list', ['pred'])
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.push_eof_matcher.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.push_eof_matcher', ['pred'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'push_eof_matcher', localization, ['pred'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'push_eof_matcher(...)' code ##################

        
        # Call to append(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'pred' (line 62)
        pred_12992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 30), 'pred', False)
        # Processing the call keyword arguments (line 62)
        kwargs_12993 = {}
        # Getting the type of 'self' (line 62)
        self_12989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self', False)
        # Obtaining the member '_eofstack' of a type (line 62)
        _eofstack_12990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_12989, '_eofstack')
        # Obtaining the member 'append' of a type (line 62)
        append_12991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), _eofstack_12990, 'append')
        # Calling append(args, kwargs) (line 62)
        append_call_result_12994 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), append_12991, *[pred_12992], **kwargs_12993)
        
        
        # ################# End of 'push_eof_matcher(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'push_eof_matcher' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_12995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12995)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'push_eof_matcher'
        return stypy_return_type_12995


    @norecursion
    def pop_eof_matcher(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pop_eof_matcher'
        module_type_store = module_type_store.open_function_context('pop_eof_matcher', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.pop_eof_matcher')
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_param_names_list', [])
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.pop_eof_matcher.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.pop_eof_matcher', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pop_eof_matcher', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pop_eof_matcher(...)' code ##################

        
        # Call to pop(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_12999 = {}
        # Getting the type of 'self' (line 65)
        self_12996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'self', False)
        # Obtaining the member '_eofstack' of a type (line 65)
        _eofstack_12997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), self_12996, '_eofstack')
        # Obtaining the member 'pop' of a type (line 65)
        pop_12998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), _eofstack_12997, 'pop')
        # Calling pop(args, kwargs) (line 65)
        pop_call_result_13000 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), pop_12998, *[], **kwargs_12999)
        
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', pop_call_result_13000)
        
        # ################# End of 'pop_eof_matcher(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pop_eof_matcher' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_13001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13001)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pop_eof_matcher'
        return stypy_return_type_13001


    @norecursion
    def close(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'close'
        module_type_store = module_type_store.open_function_context('close', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.close.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.close.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.close.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.close.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.close')
        BufferedSubFile.close.__dict__.__setitem__('stypy_param_names_list', [])
        BufferedSubFile.close.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.close.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.close.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.close.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.close.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.close.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.close', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'close', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'close(...)' code ##################

        
        # Call to pushlines(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to splitlines(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'True' (line 69)
        True_13011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 57), 'True', False)
        # Processing the call keyword arguments (line 69)
        kwargs_13012 = {}
        
        # Call to join(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'self' (line 69)
        self_13006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'self', False)
        # Obtaining the member '_partial' of a type (line 69)
        _partial_13007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 31), self_13006, '_partial')
        # Processing the call keyword arguments (line 69)
        kwargs_13008 = {}
        str_13004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'str', '')
        # Obtaining the member 'join' of a type (line 69)
        join_13005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 23), str_13004, 'join')
        # Calling join(args, kwargs) (line 69)
        join_call_result_13009 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), join_13005, *[_partial_13007], **kwargs_13008)
        
        # Obtaining the member 'splitlines' of a type (line 69)
        splitlines_13010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 23), join_call_result_13009, 'splitlines')
        # Calling splitlines(args, kwargs) (line 69)
        splitlines_call_result_13013 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), splitlines_13010, *[True_13011], **kwargs_13012)
        
        # Processing the call keyword arguments (line 69)
        kwargs_13014 = {}
        # Getting the type of 'self' (line 69)
        self_13002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self', False)
        # Obtaining the member 'pushlines' of a type (line 69)
        pushlines_13003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_13002, 'pushlines')
        # Calling pushlines(args, kwargs) (line 69)
        pushlines_call_result_13015 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), pushlines_13003, *[splitlines_call_result_13013], **kwargs_13014)
        
        
        # Assigning a List to a Attribute (line 70):
        
        # Assigning a List to a Attribute (line 70):
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_13016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        
        # Getting the type of 'self' (line 70)
        self_13017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member '_partial' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_13017, '_partial', list_13016)
        
        # Assigning a Name to a Attribute (line 71):
        
        # Assigning a Name to a Attribute (line 71):
        # Getting the type of 'True' (line 71)
        True_13018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'True')
        # Getting the type of 'self' (line 71)
        self_13019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member '_closed' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_13019, '_closed', True_13018)
        
        # ################# End of 'close(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'close' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_13020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13020)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'close'
        return stypy_return_type_13020


    @norecursion
    def readline(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'readline'
        module_type_store = module_type_store.open_function_context('readline', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.readline.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.readline.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.readline.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.readline.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.readline')
        BufferedSubFile.readline.__dict__.__setitem__('stypy_param_names_list', [])
        BufferedSubFile.readline.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.readline.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.readline.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.readline.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.readline.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.readline.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.readline', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'readline', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'readline(...)' code ##################

        
        # Getting the type of 'self' (line 74)
        self_13021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'self')
        # Obtaining the member '_lines' of a type (line 74)
        _lines_13022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 15), self_13021, '_lines')
        # Applying the 'not' unary operator (line 74)
        result_not__13023 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), 'not', _lines_13022)
        
        # Testing if the type of an if condition is none (line 74)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 8), result_not__13023):
            pass
        else:
            
            # Testing the type of an if condition (line 74)
            if_condition_13024 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 8), result_not__13023)
            # Assigning a type to the variable 'if_condition_13024' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'if_condition_13024', if_condition_13024)
            # SSA begins for if statement (line 74)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 75)
            self_13025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self')
            # Obtaining the member '_closed' of a type (line 75)
            _closed_13026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_13025, '_closed')
            # Testing if the type of an if condition is none (line 75)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), _closed_13026):
                pass
            else:
                
                # Testing the type of an if condition (line 75)
                if_condition_13027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), _closed_13026)
                # Assigning a type to the variable 'if_condition_13027' (line 75)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_13027', if_condition_13027)
                # SSA begins for if statement (line 75)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                str_13028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 23), 'str', '')
                # Assigning a type to the variable 'stypy_return_type' (line 76)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', str_13028)
                # SSA join for if statement (line 75)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'NeedMoreData' (line 77)
            NeedMoreData_13029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'NeedMoreData')
            # Assigning a type to the variable 'stypy_return_type' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'stypy_return_type', NeedMoreData_13029)
            # SSA join for if statement (line 74)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to pop(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_13033 = {}
        # Getting the type of 'self' (line 80)
        self_13030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'self', False)
        # Obtaining the member '_lines' of a type (line 80)
        _lines_13031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), self_13030, '_lines')
        # Obtaining the member 'pop' of a type (line 80)
        pop_13032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), _lines_13031, 'pop')
        # Calling pop(args, kwargs) (line 80)
        pop_call_result_13034 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), pop_13032, *[], **kwargs_13033)
        
        # Assigning a type to the variable 'line' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'line', pop_call_result_13034)
        
        
        # Obtaining the type of the subscript
        int_13035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 38), 'int')
        slice_13036 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 84, 21), None, None, int_13035)
        # Getting the type of 'self' (line 84)
        self_13037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'self')
        # Obtaining the member '_eofstack' of a type (line 84)
        _eofstack_13038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), self_13037, '_eofstack')
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___13039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), _eofstack_13038, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_13040 = invoke(stypy.reporting.localization.Localization(__file__, 84, 21), getitem___13039, slice_13036)
        
        # Assigning a type to the variable 'subscript_call_result_13040' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'subscript_call_result_13040', subscript_call_result_13040)
        # Testing if the for loop is going to be iterated (line 84)
        # Testing the type of a for loop iterable (line 84)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 8), subscript_call_result_13040)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 8), subscript_call_result_13040):
            # Getting the type of the for loop variable (line 84)
            for_loop_var_13041 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 8), subscript_call_result_13040)
            # Assigning a type to the variable 'ateof' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'ateof', for_loop_var_13041)
            # SSA begins for a for statement (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to ateof(...): (line 85)
            # Processing the call arguments (line 85)
            # Getting the type of 'line' (line 85)
            line_13043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'line', False)
            # Processing the call keyword arguments (line 85)
            kwargs_13044 = {}
            # Getting the type of 'ateof' (line 85)
            ateof_13042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'ateof', False)
            # Calling ateof(args, kwargs) (line 85)
            ateof_call_result_13045 = invoke(stypy.reporting.localization.Localization(__file__, 85, 15), ateof_13042, *[line_13043], **kwargs_13044)
            
            # Testing if the type of an if condition is none (line 85)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 12), ateof_call_result_13045):
                pass
            else:
                
                # Testing the type of an if condition (line 85)
                if_condition_13046 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 12), ateof_call_result_13045)
                # Assigning a type to the variable 'if_condition_13046' (line 85)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'if_condition_13046', if_condition_13046)
                # SSA begins for if statement (line 85)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 87)
                # Processing the call arguments (line 87)
                # Getting the type of 'line' (line 87)
                line_13050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 35), 'line', False)
                # Processing the call keyword arguments (line 87)
                kwargs_13051 = {}
                # Getting the type of 'self' (line 87)
                self_13047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'self', False)
                # Obtaining the member '_lines' of a type (line 87)
                _lines_13048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), self_13047, '_lines')
                # Obtaining the member 'append' of a type (line 87)
                append_13049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), _lines_13048, 'append')
                # Calling append(args, kwargs) (line 87)
                append_call_result_13052 = invoke(stypy.reporting.localization.Localization(__file__, 87, 16), append_13049, *[line_13050], **kwargs_13051)
                
                str_13053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 23), 'str', '')
                # Assigning a type to the variable 'stypy_return_type' (line 88)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'stypy_return_type', str_13053)
                # SSA join for if statement (line 85)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'line' (line 89)
        line_13054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'line')
        # Assigning a type to the variable 'stypy_return_type' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'stypy_return_type', line_13054)
        
        # ################# End of 'readline(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'readline' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_13055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13055)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'readline'
        return stypy_return_type_13055


    @norecursion
    def unreadline(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unreadline'
        module_type_store = module_type_store.open_function_context('unreadline', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.unreadline')
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_param_names_list', ['line'])
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.unreadline.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.unreadline', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unreadline', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unreadline(...)' code ##################

        # Evaluating assert statement condition
        
        # Getting the type of 'line' (line 93)
        line_13056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'line')
        # Getting the type of 'NeedMoreData' (line 93)
        NeedMoreData_13057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'NeedMoreData')
        # Applying the binary operator 'isnot' (line 93)
        result_is_not_13058 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 15), 'isnot', line_13056, NeedMoreData_13057)
        
        assert_13059 = result_is_not_13058
        # Assigning a type to the variable 'assert_13059' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'assert_13059', result_is_not_13058)
        
        # Call to append(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'line' (line 94)
        line_13063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'line', False)
        # Processing the call keyword arguments (line 94)
        kwargs_13064 = {}
        # Getting the type of 'self' (line 94)
        self_13060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member '_lines' of a type (line 94)
        _lines_13061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_13060, '_lines')
        # Obtaining the member 'append' of a type (line 94)
        append_13062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), _lines_13061, 'append')
        # Calling append(args, kwargs) (line 94)
        append_call_result_13065 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), append_13062, *[line_13063], **kwargs_13064)
        
        
        # ################# End of 'unreadline(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unreadline' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_13066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13066)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unreadline'
        return stypy_return_type_13066


    @norecursion
    def push(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'push'
        module_type_store = module_type_store.open_function_context('push', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.push.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.push.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.push.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.push.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.push')
        BufferedSubFile.push.__dict__.__setitem__('stypy_param_names_list', ['data'])
        BufferedSubFile.push.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.push.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.push.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.push.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.push.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.push.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.push', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'push', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'push(...)' code ##################

        str_13067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'str', 'Push some new data into this object.')
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to splitlines(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'True' (line 99)
        True_13070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 32), 'True', False)
        # Processing the call keyword arguments (line 99)
        kwargs_13071 = {}
        # Getting the type of 'data' (line 99)
        data_13068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'data', False)
        # Obtaining the member 'splitlines' of a type (line 99)
        splitlines_13069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), data_13068, 'splitlines')
        # Calling splitlines(args, kwargs) (line 99)
        splitlines_call_result_13072 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), splitlines_13069, *[True_13070], **kwargs_13071)
        
        # Assigning a type to the variable 'parts' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'parts', splitlines_call_result_13072)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'parts' (line 101)
        parts_13073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'parts')
        # Applying the 'not' unary operator (line 101)
        result_not__13074 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), 'not', parts_13073)
        
        
        
        # Call to endswith(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Obtaining an instance of the builtin type 'tuple' (line 101)
        tuple_13080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        str_13081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 47), 'str', '\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 47), tuple_13080, str_13081)
        # Adding element type (line 101)
        str_13082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 53), 'str', '\r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 47), tuple_13080, str_13082)
        
        # Processing the call keyword arguments (line 101)
        kwargs_13083 = {}
        
        # Obtaining the type of the subscript
        int_13075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 34), 'int')
        # Getting the type of 'parts' (line 101)
        parts_13076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'parts', False)
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___13077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), parts_13076, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_13078 = invoke(stypy.reporting.localization.Localization(__file__, 101, 28), getitem___13077, int_13075)
        
        # Obtaining the member 'endswith' of a type (line 101)
        endswith_13079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), subscript_call_result_13078, 'endswith')
        # Calling endswith(args, kwargs) (line 101)
        endswith_call_result_13084 = invoke(stypy.reporting.localization.Localization(__file__, 101, 28), endswith_13079, *[tuple_13080], **kwargs_13083)
        
        # Applying the 'not' unary operator (line 101)
        result_not__13085 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 24), 'not', endswith_call_result_13084)
        
        # Applying the binary operator 'or' (line 101)
        result_or_keyword_13086 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), 'or', result_not__13074, result_not__13085)
        
        # Testing if the type of an if condition is none (line 101)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 8), result_or_keyword_13086):
            pass
        else:
            
            # Testing the type of an if condition (line 101)
            if_condition_13087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), result_or_keyword_13086)
            # Assigning a type to the variable 'if_condition_13087' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'if_condition_13087', if_condition_13087)
            # SSA begins for if statement (line 101)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'self' (line 103)
            self_13088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self')
            # Obtaining the member '_partial' of a type (line 103)
            _partial_13089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_13088, '_partial')
            # Getting the type of 'parts' (line 103)
            parts_13090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'parts')
            # Applying the binary operator '+=' (line 103)
            result_iadd_13091 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 12), '+=', _partial_13089, parts_13090)
            # Getting the type of 'self' (line 103)
            self_13092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self')
            # Setting the type of the member '_partial' of a type (line 103)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_13092, '_partial', result_iadd_13091)
            
            # Assigning a type to the variable 'stypy_return_type' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 101)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'self' (line 106)
        self_13093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'self')
        # Obtaining the member '_partial' of a type (line 106)
        _partial_13094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 11), self_13093, '_partial')
        # Testing if the type of an if condition is none (line 106)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 8), _partial_13094):
            pass
        else:
            
            # Testing the type of an if condition (line 106)
            if_condition_13095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), _partial_13094)
            # Assigning a type to the variable 'if_condition_13095' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_13095', if_condition_13095)
            # SSA begins for if statement (line 106)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 108)
            # Processing the call arguments (line 108)
            
            # Obtaining the type of the subscript
            int_13099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 39), 'int')
            # Getting the type of 'parts' (line 108)
            parts_13100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 33), 'parts', False)
            # Obtaining the member '__getitem__' of a type (line 108)
            getitem___13101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 33), parts_13100, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 108)
            subscript_call_result_13102 = invoke(stypy.reporting.localization.Localization(__file__, 108, 33), getitem___13101, int_13099)
            
            # Processing the call keyword arguments (line 108)
            kwargs_13103 = {}
            # Getting the type of 'self' (line 108)
            self_13096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'self', False)
            # Obtaining the member '_partial' of a type (line 108)
            _partial_13097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), self_13096, '_partial')
            # Obtaining the member 'append' of a type (line 108)
            append_13098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), _partial_13097, 'append')
            # Calling append(args, kwargs) (line 108)
            append_call_result_13104 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), append_13098, *[subscript_call_result_13102], **kwargs_13103)
            
            
            # Assigning a Call to a Subscript (line 109):
            
            # Assigning a Call to a Subscript (line 109):
            
            # Call to splitlines(...): (line 109)
            # Processing the call arguments (line 109)
            # Getting the type of 'True' (line 109)
            True_13112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 59), 'True', False)
            # Processing the call keyword arguments (line 109)
            kwargs_13113 = {}
            
            # Call to join(...): (line 109)
            # Processing the call arguments (line 109)
            # Getting the type of 'self' (line 109)
            self_13107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 33), 'self', False)
            # Obtaining the member '_partial' of a type (line 109)
            _partial_13108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 33), self_13107, '_partial')
            # Processing the call keyword arguments (line 109)
            kwargs_13109 = {}
            str_13105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'str', '')
            # Obtaining the member 'join' of a type (line 109)
            join_13106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 25), str_13105, 'join')
            # Calling join(args, kwargs) (line 109)
            join_call_result_13110 = invoke(stypy.reporting.localization.Localization(__file__, 109, 25), join_13106, *[_partial_13108], **kwargs_13109)
            
            # Obtaining the member 'splitlines' of a type (line 109)
            splitlines_13111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 25), join_call_result_13110, 'splitlines')
            # Calling splitlines(args, kwargs) (line 109)
            splitlines_call_result_13114 = invoke(stypy.reporting.localization.Localization(__file__, 109, 25), splitlines_13111, *[True_13112], **kwargs_13113)
            
            # Getting the type of 'parts' (line 109)
            parts_13115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'parts')
            int_13116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'int')
            int_13117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'int')
            slice_13118 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 109, 12), int_13116, int_13117, None)
            # Storing an element on a container (line 109)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 12), parts_13115, (slice_13118, splitlines_call_result_13114))
            # Deleting a member
            # Getting the type of 'self' (line 110)
            self_13119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'self')
            # Obtaining the member '_partial' of a type (line 110)
            _partial_13120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), self_13119, '_partial')
            
            # Obtaining the type of the subscript
            slice_13121 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 110, 16), None, None, None)
            # Getting the type of 'self' (line 110)
            self_13122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'self')
            # Obtaining the member '_partial' of a type (line 110)
            _partial_13123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), self_13122, '_partial')
            # Obtaining the member '__getitem__' of a type (line 110)
            getitem___13124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), _partial_13123, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 110)
            subscript_call_result_13125 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), getitem___13124, slice_13121)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 12), _partial_13120, subscript_call_result_13125)
            # SSA join for if statement (line 106)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to endswith(...): (line 116)
        # Processing the call arguments (line 116)
        str_13131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 34), 'str', '\n')
        # Processing the call keyword arguments (line 116)
        kwargs_13132 = {}
        
        # Obtaining the type of the subscript
        int_13126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
        # Getting the type of 'parts' (line 116)
        parts_13127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'parts', False)
        # Obtaining the member '__getitem__' of a type (line 116)
        getitem___13128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 15), parts_13127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
        subscript_call_result_13129 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), getitem___13128, int_13126)
        
        # Obtaining the member 'endswith' of a type (line 116)
        endswith_13130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 15), subscript_call_result_13129, 'endswith')
        # Calling endswith(args, kwargs) (line 116)
        endswith_call_result_13133 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), endswith_13130, *[str_13131], **kwargs_13132)
        
        # Applying the 'not' unary operator (line 116)
        result_not__13134 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 11), 'not', endswith_call_result_13133)
        
        # Testing if the type of an if condition is none (line 116)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 116, 8), result_not__13134):
            pass
        else:
            
            # Testing the type of an if condition (line 116)
            if_condition_13135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), result_not__13134)
            # Assigning a type to the variable 'if_condition_13135' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_13135', if_condition_13135)
            # SSA begins for if statement (line 116)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a List to a Attribute (line 117):
            
            # Assigning a List to a Attribute (line 117):
            
            # Obtaining an instance of the builtin type 'list' (line 117)
            list_13136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 117)
            # Adding element type (line 117)
            
            # Call to pop(...): (line 117)
            # Processing the call keyword arguments (line 117)
            kwargs_13139 = {}
            # Getting the type of 'parts' (line 117)
            parts_13137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'parts', False)
            # Obtaining the member 'pop' of a type (line 117)
            pop_13138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 29), parts_13137, 'pop')
            # Calling pop(args, kwargs) (line 117)
            pop_call_result_13140 = invoke(stypy.reporting.localization.Localization(__file__, 117, 29), pop_13138, *[], **kwargs_13139)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 28), list_13136, pop_call_result_13140)
            
            # Getting the type of 'self' (line 117)
            self_13141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'self')
            # Setting the type of the member '_partial' of a type (line 117)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), self_13141, '_partial', list_13136)
            # SSA join for if statement (line 116)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to pushlines(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'parts' (line 118)
        parts_13144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 23), 'parts', False)
        # Processing the call keyword arguments (line 118)
        kwargs_13145 = {}
        # Getting the type of 'self' (line 118)
        self_13142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self', False)
        # Obtaining the member 'pushlines' of a type (line 118)
        pushlines_13143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_13142, 'pushlines')
        # Calling pushlines(args, kwargs) (line 118)
        pushlines_call_result_13146 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), pushlines_13143, *[parts_13144], **kwargs_13145)
        
        
        # ################# End of 'push(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'push' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_13147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13147)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'push'
        return stypy_return_type_13147


    @norecursion
    def pushlines(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pushlines'
        module_type_store = module_type_store.open_function_context('pushlines', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.pushlines')
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_param_names_list', ['lines'])
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.pushlines', ['lines'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pushlines', localization, ['lines'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pushlines(...)' code ##################

        
        # Assigning a Call to a Name (line 122):
        
        # Assigning a Call to a Name (line 122):
        
        # Call to split(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'data' (line 122)
        data_13150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 34), 'data', False)
        # Processing the call keyword arguments (line 122)
        kwargs_13151 = {}
        # Getting the type of 'NLCRE_crack' (line 122)
        NLCRE_crack_13148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'NLCRE_crack', False)
        # Obtaining the member 'split' of a type (line 122)
        split_13149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), NLCRE_crack_13148, 'split')
        # Calling split(args, kwargs) (line 122)
        split_call_result_13152 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), split_13149, *[data_13150], **kwargs_13151)
        
        # Assigning a type to the variable 'parts' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'parts', split_call_result_13152)
        
        # Assigning a Call to a Attribute (line 127):
        
        # Assigning a Call to a Attribute (line 127):
        
        # Call to pop(...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_13155 = {}
        # Getting the type of 'parts' (line 127)
        parts_13153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'parts', False)
        # Obtaining the member 'pop' of a type (line 127)
        pop_13154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), parts_13153, 'pop')
        # Calling pop(args, kwargs) (line 127)
        pop_call_result_13156 = invoke(stypy.reporting.localization.Localization(__file__, 127, 24), pop_13154, *[], **kwargs_13155)
        
        # Getting the type of 'self' (line 127)
        self_13157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self')
        # Setting the type of the member '_partial' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_13157, '_partial', pop_call_result_13156)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 130)
        self_13158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'self')
        # Obtaining the member '_partial' of a type (line 130)
        _partial_13159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 15), self_13158, '_partial')
        # Applying the 'not' unary operator (line 130)
        result_not__13160 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), 'not', _partial_13159)
        
        # Getting the type of 'parts' (line 130)
        parts_13161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'parts')
        # Applying the binary operator 'and' (line 130)
        result_and_keyword_13162 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), 'and', result_not__13160, parts_13161)
        
        # Call to endswith(...): (line 130)
        # Processing the call arguments (line 130)
        str_13168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 62), 'str', '\r')
        # Processing the call keyword arguments (line 130)
        kwargs_13169 = {}
        
        # Obtaining the type of the subscript
        int_13163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 49), 'int')
        # Getting the type of 'parts' (line 130)
        parts_13164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 43), 'parts', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___13165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 43), parts_13164, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_13166 = invoke(stypy.reporting.localization.Localization(__file__, 130, 43), getitem___13165, int_13163)
        
        # Obtaining the member 'endswith' of a type (line 130)
        endswith_13167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 43), subscript_call_result_13166, 'endswith')
        # Calling endswith(args, kwargs) (line 130)
        endswith_call_result_13170 = invoke(stypy.reporting.localization.Localization(__file__, 130, 43), endswith_13167, *[str_13168], **kwargs_13169)
        
        # Applying the binary operator 'and' (line 130)
        result_and_keyword_13171 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), 'and', result_and_keyword_13162, endswith_call_result_13170)
        
        # Testing if the type of an if condition is none (line 130)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 130, 8), result_and_keyword_13171):
            pass
        else:
            
            # Testing the type of an if condition (line 130)
            if_condition_13172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_and_keyword_13171)
            # Assigning a type to the variable 'if_condition_13172' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_13172', if_condition_13172)
            # SSA begins for if statement (line 130)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Attribute (line 131):
            
            # Assigning a BinOp to a Attribute (line 131):
            
            # Call to pop(...): (line 131)
            # Processing the call arguments (line 131)
            int_13175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 38), 'int')
            # Processing the call keyword arguments (line 131)
            kwargs_13176 = {}
            # Getting the type of 'parts' (line 131)
            parts_13173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'parts', False)
            # Obtaining the member 'pop' of a type (line 131)
            pop_13174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 28), parts_13173, 'pop')
            # Calling pop(args, kwargs) (line 131)
            pop_call_result_13177 = invoke(stypy.reporting.localization.Localization(__file__, 131, 28), pop_13174, *[int_13175], **kwargs_13176)
            
            
            # Call to pop(...): (line 131)
            # Processing the call keyword arguments (line 131)
            kwargs_13180 = {}
            # Getting the type of 'parts' (line 131)
            parts_13178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 42), 'parts', False)
            # Obtaining the member 'pop' of a type (line 131)
            pop_13179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 42), parts_13178, 'pop')
            # Calling pop(args, kwargs) (line 131)
            pop_call_result_13181 = invoke(stypy.reporting.localization.Localization(__file__, 131, 42), pop_13179, *[], **kwargs_13180)
            
            # Applying the binary operator '+' (line 131)
            result_add_13182 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 28), '+', pop_call_result_13177, pop_call_result_13181)
            
            # Getting the type of 'self' (line 131)
            self_13183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'self')
            # Setting the type of the member '_partial' of a type (line 131)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), self_13183, '_partial', result_add_13182)
            # SSA join for if statement (line 130)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a List to a Name (line 135):
        
        # Assigning a List to a Name (line 135):
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_13184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        
        # Assigning a type to the variable 'lines' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'lines', list_13184)
        
        
        # Call to range(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Call to len(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'parts' (line 136)
        parts_13187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'parts', False)
        # Processing the call keyword arguments (line 136)
        kwargs_13188 = {}
        # Getting the type of 'len' (line 136)
        len_13186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'len', False)
        # Calling len(args, kwargs) (line 136)
        len_call_result_13189 = invoke(stypy.reporting.localization.Localization(__file__, 136, 23), len_13186, *[parts_13187], **kwargs_13188)
        
        int_13190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 37), 'int')
        # Applying the binary operator '//' (line 136)
        result_floordiv_13191 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 23), '//', len_call_result_13189, int_13190)
        
        # Processing the call keyword arguments (line 136)
        kwargs_13192 = {}
        # Getting the type of 'range' (line 136)
        range_13185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'range', False)
        # Calling range(args, kwargs) (line 136)
        range_call_result_13193 = invoke(stypy.reporting.localization.Localization(__file__, 136, 17), range_13185, *[result_floordiv_13191], **kwargs_13192)
        
        # Assigning a type to the variable 'range_call_result_13193' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'range_call_result_13193', range_call_result_13193)
        # Testing if the for loop is going to be iterated (line 136)
        # Testing the type of a for loop iterable (line 136)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 136, 8), range_call_result_13193)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 136, 8), range_call_result_13193):
            # Getting the type of the for loop variable (line 136)
            for_loop_var_13194 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 136, 8), range_call_result_13193)
            # Assigning a type to the variable 'i' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'i', for_loop_var_13194)
            # SSA begins for a for statement (line 136)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 137)
            # Processing the call arguments (line 137)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 137)
            i_13197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'i', False)
            int_13198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 33), 'int')
            # Applying the binary operator '*' (line 137)
            result_mul_13199 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 31), '*', i_13197, int_13198)
            
            # Getting the type of 'parts' (line 137)
            parts_13200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'parts', False)
            # Obtaining the member '__getitem__' of a type (line 137)
            getitem___13201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 25), parts_13200, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 137)
            subscript_call_result_13202 = invoke(stypy.reporting.localization.Localization(__file__, 137, 25), getitem___13201, result_mul_13199)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 137)
            i_13203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 44), 'i', False)
            int_13204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 46), 'int')
            # Applying the binary operator '*' (line 137)
            result_mul_13205 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 44), '*', i_13203, int_13204)
            
            int_13206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 48), 'int')
            # Applying the binary operator '+' (line 137)
            result_add_13207 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 44), '+', result_mul_13205, int_13206)
            
            # Getting the type of 'parts' (line 137)
            parts_13208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 38), 'parts', False)
            # Obtaining the member '__getitem__' of a type (line 137)
            getitem___13209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 38), parts_13208, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 137)
            subscript_call_result_13210 = invoke(stypy.reporting.localization.Localization(__file__, 137, 38), getitem___13209, result_add_13207)
            
            # Applying the binary operator '+' (line 137)
            result_add_13211 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 25), '+', subscript_call_result_13202, subscript_call_result_13210)
            
            # Processing the call keyword arguments (line 137)
            kwargs_13212 = {}
            # Getting the type of 'lines' (line 137)
            lines_13195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'lines', False)
            # Obtaining the member 'append' of a type (line 137)
            append_13196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), lines_13195, 'append')
            # Calling append(args, kwargs) (line 137)
            append_call_result_13213 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), append_13196, *[result_add_13211], **kwargs_13212)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to pushlines(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'lines' (line 138)
        lines_13216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'lines', False)
        # Processing the call keyword arguments (line 138)
        kwargs_13217 = {}
        # Getting the type of 'self' (line 138)
        self_13214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self', False)
        # Obtaining the member 'pushlines' of a type (line 138)
        pushlines_13215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_13214, 'pushlines')
        # Calling pushlines(args, kwargs) (line 138)
        pushlines_call_result_13218 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), pushlines_13215, *[lines_13216], **kwargs_13217)
        
        
        # ################# End of 'pushlines(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pushlines' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_13219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pushlines'
        return stypy_return_type_13219


    @norecursion
    def pushlines(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pushlines'
        module_type_store = module_type_store.open_function_context('pushlines', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.pushlines')
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_param_names_list', ['lines'])
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.pushlines.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.pushlines', ['lines'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pushlines', localization, ['lines'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pushlines(...)' code ##################

        
        # Assigning a Subscript to a Subscript (line 142):
        
        # Assigning a Subscript to a Subscript (line 142):
        
        # Obtaining the type of the subscript
        int_13220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 34), 'int')
        slice_13221 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 142, 26), None, None, int_13220)
        # Getting the type of 'lines' (line 142)
        lines_13222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'lines')
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___13223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 26), lines_13222, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_13224 = invoke(stypy.reporting.localization.Localization(__file__, 142, 26), getitem___13223, slice_13221)
        
        # Getting the type of 'self' (line 142)
        self_13225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self')
        # Obtaining the member '_lines' of a type (line 142)
        _lines_13226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_13225, '_lines')
        int_13227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'int')
        slice_13228 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 142, 8), None, int_13227, None)
        # Storing an element on a container (line 142)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 8), _lines_13226, (slice_13228, subscript_call_result_13224))
        
        # ################# End of 'pushlines(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pushlines' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_13229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13229)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pushlines'
        return stypy_return_type_13229


    @norecursion
    def is_closed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_closed'
        module_type_store = module_type_store.open_function_context('is_closed', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.is_closed')
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_param_names_list', [])
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.is_closed.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.is_closed', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_closed', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_closed(...)' code ##################

        # Getting the type of 'self' (line 145)
        self_13230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'self')
        # Obtaining the member '_closed' of a type (line 145)
        _closed_13231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 15), self_13230, '_closed')
        # Assigning a type to the variable 'stypy_return_type' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'stypy_return_type', _closed_13231)
        
        # ################# End of 'is_closed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_closed' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_13232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13232)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_closed'
        return stypy_return_type_13232


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.__iter__')
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        # Getting the type of 'self' (line 148)
        self_13233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', self_13233)
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_13234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13234)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_13234


    @norecursion
    def next(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'next'
        module_type_store = module_type_store.open_function_context('next', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BufferedSubFile.next.__dict__.__setitem__('stypy_localization', localization)
        BufferedSubFile.next.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BufferedSubFile.next.__dict__.__setitem__('stypy_type_store', module_type_store)
        BufferedSubFile.next.__dict__.__setitem__('stypy_function_name', 'BufferedSubFile.next')
        BufferedSubFile.next.__dict__.__setitem__('stypy_param_names_list', [])
        BufferedSubFile.next.__dict__.__setitem__('stypy_varargs_param_name', None)
        BufferedSubFile.next.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BufferedSubFile.next.__dict__.__setitem__('stypy_call_defaults', defaults)
        BufferedSubFile.next.__dict__.__setitem__('stypy_call_varargs', varargs)
        BufferedSubFile.next.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BufferedSubFile.next.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BufferedSubFile.next', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'next', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'next(...)' code ##################

        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to readline(...): (line 151)
        # Processing the call keyword arguments (line 151)
        kwargs_13237 = {}
        # Getting the type of 'self' (line 151)
        self_13235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'self', False)
        # Obtaining the member 'readline' of a type (line 151)
        readline_13236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 15), self_13235, 'readline')
        # Calling readline(args, kwargs) (line 151)
        readline_call_result_13238 = invoke(stypy.reporting.localization.Localization(__file__, 151, 15), readline_13236, *[], **kwargs_13237)
        
        # Assigning a type to the variable 'line' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'line', readline_call_result_13238)
        
        # Getting the type of 'line' (line 152)
        line_13239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'line')
        str_13240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 19), 'str', '')
        # Applying the binary operator '==' (line 152)
        result_eq_13241 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 11), '==', line_13239, str_13240)
        
        # Testing if the type of an if condition is none (line 152)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 152, 8), result_eq_13241):
            pass
        else:
            
            # Testing the type of an if condition (line 152)
            if_condition_13242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 8), result_eq_13241)
            # Assigning a type to the variable 'if_condition_13242' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'if_condition_13242', if_condition_13242)
            # SSA begins for if statement (line 152)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'StopIteration' (line 153)
            StopIteration_13243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'StopIteration')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 153, 12), StopIteration_13243, 'raise parameter', BaseException)
            # SSA join for if statement (line 152)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'line' (line 154)
        line_13244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'line')
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', line_13244)
        
        # ################# End of 'next(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'next' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_13245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13245)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'next'
        return stypy_return_type_13245


# Assigning a type to the variable 'BufferedSubFile' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'BufferedSubFile', BufferedSubFile)
# Declaration of the 'FeedParser' class

class FeedParser:
    str_13246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 4), 'str', 'A feed-style parser of email.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'message' (line 161)
        message_13247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 32), 'message')
        # Obtaining the member 'Message' of a type (line 161)
        Message_13248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 32), message_13247, 'Message')
        defaults = [Message_13248]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FeedParser.__init__', ['_factory'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['_factory'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_13249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 8), 'str', '_factory is called with no arguments to create a new message obj')
        
        # Assigning a Name to a Attribute (line 163):
        
        # Assigning a Name to a Attribute (line 163):
        # Getting the type of '_factory' (line 163)
        _factory_13250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), '_factory')
        # Getting the type of 'self' (line 163)
        self_13251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self')
        # Setting the type of the member '_factory' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_13251, '_factory', _factory_13250)
        
        # Assigning a Call to a Attribute (line 164):
        
        # Assigning a Call to a Attribute (line 164):
        
        # Call to BufferedSubFile(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_13253 = {}
        # Getting the type of 'BufferedSubFile' (line 164)
        BufferedSubFile_13252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'BufferedSubFile', False)
        # Calling BufferedSubFile(args, kwargs) (line 164)
        BufferedSubFile_call_result_13254 = invoke(stypy.reporting.localization.Localization(__file__, 164, 22), BufferedSubFile_13252, *[], **kwargs_13253)
        
        # Getting the type of 'self' (line 164)
        self_13255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self')
        # Setting the type of the member '_input' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_13255, '_input', BufferedSubFile_call_result_13254)
        
        # Assigning a List to a Attribute (line 165):
        
        # Assigning a List to a Attribute (line 165):
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_13256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        
        # Getting the type of 'self' (line 165)
        self_13257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self')
        # Setting the type of the member '_msgstack' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_13257, '_msgstack', list_13256)
        
        # Assigning a Attribute to a Attribute (line 166):
        
        # Assigning a Attribute to a Attribute (line 166):
        
        # Call to _parsegen(...): (line 166)
        # Processing the call keyword arguments (line 166)
        kwargs_13260 = {}
        # Getting the type of 'self' (line 166)
        self_13258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'self', False)
        # Obtaining the member '_parsegen' of a type (line 166)
        _parsegen_13259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 22), self_13258, '_parsegen')
        # Calling _parsegen(args, kwargs) (line 166)
        _parsegen_call_result_13261 = invoke(stypy.reporting.localization.Localization(__file__, 166, 22), _parsegen_13259, *[], **kwargs_13260)
        
        # Obtaining the member 'next' of a type (line 166)
        next_13262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 22), _parsegen_call_result_13261, 'next')
        # Getting the type of 'self' (line 166)
        self_13263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self')
        # Setting the type of the member '_parse' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_13263, '_parse', next_13262)
        
        # Assigning a Name to a Attribute (line 167):
        
        # Assigning a Name to a Attribute (line 167):
        # Getting the type of 'None' (line 167)
        None_13264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'None')
        # Getting the type of 'self' (line 167)
        self_13265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member '_cur' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_13265, '_cur', None_13264)
        
        # Assigning a Name to a Attribute (line 168):
        
        # Assigning a Name to a Attribute (line 168):
        # Getting the type of 'None' (line 168)
        None_13266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'None')
        # Getting the type of 'self' (line 168)
        self_13267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self')
        # Setting the type of the member '_last' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_13267, '_last', None_13266)
        
        # Assigning a Name to a Attribute (line 169):
        
        # Assigning a Name to a Attribute (line 169):
        # Getting the type of 'False' (line 169)
        False_13268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'False')
        # Getting the type of 'self' (line 169)
        self_13269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self')
        # Setting the type of the member '_headersonly' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_13269, '_headersonly', False_13268)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _set_headersonly(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_headersonly'
        module_type_store = module_type_store.open_function_context('_set_headersonly', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_localization', localization)
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_type_store', module_type_store)
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_function_name', 'FeedParser._set_headersonly')
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_param_names_list', [])
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_varargs_param_name', None)
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_call_defaults', defaults)
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_call_varargs', varargs)
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FeedParser._set_headersonly.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FeedParser._set_headersonly', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_headersonly', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_headersonly(...)' code ##################

        
        # Assigning a Name to a Attribute (line 173):
        
        # Assigning a Name to a Attribute (line 173):
        # Getting the type of 'True' (line 173)
        True_13270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 28), 'True')
        # Getting the type of 'self' (line 173)
        self_13271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self')
        # Setting the type of the member '_headersonly' of a type (line 173)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_13271, '_headersonly', True_13270)
        
        # ################# End of '_set_headersonly(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_headersonly' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_13272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13272)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_headersonly'
        return stypy_return_type_13272


    @norecursion
    def feed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'feed'
        module_type_store = module_type_store.open_function_context('feed', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FeedParser.feed.__dict__.__setitem__('stypy_localization', localization)
        FeedParser.feed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FeedParser.feed.__dict__.__setitem__('stypy_type_store', module_type_store)
        FeedParser.feed.__dict__.__setitem__('stypy_function_name', 'FeedParser.feed')
        FeedParser.feed.__dict__.__setitem__('stypy_param_names_list', ['data'])
        FeedParser.feed.__dict__.__setitem__('stypy_varargs_param_name', None)
        FeedParser.feed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FeedParser.feed.__dict__.__setitem__('stypy_call_defaults', defaults)
        FeedParser.feed.__dict__.__setitem__('stypy_call_varargs', varargs)
        FeedParser.feed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FeedParser.feed.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FeedParser.feed', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'feed', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'feed(...)' code ##################

        str_13273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'str', 'Push more data into the parser.')
        
        # Call to push(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'data' (line 177)
        data_13277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'data', False)
        # Processing the call keyword arguments (line 177)
        kwargs_13278 = {}
        # Getting the type of 'self' (line 177)
        self_13274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self', False)
        # Obtaining the member '_input' of a type (line 177)
        _input_13275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_13274, '_input')
        # Obtaining the member 'push' of a type (line 177)
        push_13276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), _input_13275, 'push')
        # Calling push(args, kwargs) (line 177)
        push_call_result_13279 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), push_13276, *[data_13277], **kwargs_13278)
        
        
        # Call to _call_parse(...): (line 178)
        # Processing the call keyword arguments (line 178)
        kwargs_13282 = {}
        # Getting the type of 'self' (line 178)
        self_13280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self', False)
        # Obtaining the member '_call_parse' of a type (line 178)
        _call_parse_13281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_13280, '_call_parse')
        # Calling _call_parse(args, kwargs) (line 178)
        _call_parse_call_result_13283 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), _call_parse_13281, *[], **kwargs_13282)
        
        
        # ################# End of 'feed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'feed' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_13284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'feed'
        return stypy_return_type_13284


    @norecursion
    def _call_parse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_call_parse'
        module_type_store = module_type_store.open_function_context('_call_parse', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FeedParser._call_parse.__dict__.__setitem__('stypy_localization', localization)
        FeedParser._call_parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FeedParser._call_parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        FeedParser._call_parse.__dict__.__setitem__('stypy_function_name', 'FeedParser._call_parse')
        FeedParser._call_parse.__dict__.__setitem__('stypy_param_names_list', [])
        FeedParser._call_parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        FeedParser._call_parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FeedParser._call_parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        FeedParser._call_parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        FeedParser._call_parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FeedParser._call_parse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FeedParser._call_parse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_call_parse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_call_parse(...)' code ##################

        
        
        # SSA begins for try-except statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to _parse(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_13287 = {}
        # Getting the type of 'self' (line 182)
        self_13285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'self', False)
        # Obtaining the member '_parse' of a type (line 182)
        _parse_13286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), self_13285, '_parse')
        # Calling _parse(args, kwargs) (line 182)
        _parse_call_result_13288 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), _parse_13286, *[], **kwargs_13287)
        
        # SSA branch for the except part of a try statement (line 181)
        # SSA branch for the except 'StopIteration' branch of a try statement (line 181)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_call_parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_call_parse' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_13289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13289)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_call_parse'
        return stypy_return_type_13289


    @norecursion
    def close(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'close'
        module_type_store = module_type_store.open_function_context('close', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FeedParser.close.__dict__.__setitem__('stypy_localization', localization)
        FeedParser.close.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FeedParser.close.__dict__.__setitem__('stypy_type_store', module_type_store)
        FeedParser.close.__dict__.__setitem__('stypy_function_name', 'FeedParser.close')
        FeedParser.close.__dict__.__setitem__('stypy_param_names_list', [])
        FeedParser.close.__dict__.__setitem__('stypy_varargs_param_name', None)
        FeedParser.close.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FeedParser.close.__dict__.__setitem__('stypy_call_defaults', defaults)
        FeedParser.close.__dict__.__setitem__('stypy_call_varargs', varargs)
        FeedParser.close.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FeedParser.close.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FeedParser.close', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'close', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'close(...)' code ##################

        str_13290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 8), 'str', 'Parse all remaining data and return the root message object.')
        
        # Call to close(...): (line 188)
        # Processing the call keyword arguments (line 188)
        kwargs_13294 = {}
        # Getting the type of 'self' (line 188)
        self_13291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self', False)
        # Obtaining the member '_input' of a type (line 188)
        _input_13292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_13291, '_input')
        # Obtaining the member 'close' of a type (line 188)
        close_13293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), _input_13292, 'close')
        # Calling close(args, kwargs) (line 188)
        close_call_result_13295 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), close_13293, *[], **kwargs_13294)
        
        
        # Call to _call_parse(...): (line 189)
        # Processing the call keyword arguments (line 189)
        kwargs_13298 = {}
        # Getting the type of 'self' (line 189)
        self_13296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'self', False)
        # Obtaining the member '_call_parse' of a type (line 189)
        _call_parse_13297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), self_13296, '_call_parse')
        # Calling _call_parse(args, kwargs) (line 189)
        _call_parse_call_result_13299 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), _call_parse_13297, *[], **kwargs_13298)
        
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to _pop_message(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_13302 = {}
        # Getting the type of 'self' (line 190)
        self_13300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), 'self', False)
        # Obtaining the member '_pop_message' of a type (line 190)
        _pop_message_13301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 15), self_13300, '_pop_message')
        # Calling _pop_message(args, kwargs) (line 190)
        _pop_message_call_result_13303 = invoke(stypy.reporting.localization.Localization(__file__, 190, 15), _pop_message_13301, *[], **kwargs_13302)
        
        # Assigning a type to the variable 'root' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'root', _pop_message_call_result_13303)
        # Evaluating assert statement condition
        
        # Getting the type of 'self' (line 191)
        self_13304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), 'self')
        # Obtaining the member '_msgstack' of a type (line 191)
        _msgstack_13305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 19), self_13304, '_msgstack')
        # Applying the 'not' unary operator (line 191)
        result_not__13306 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), 'not', _msgstack_13305)
        
        assert_13307 = result_not__13306
        # Assigning a type to the variable 'assert_13307' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'assert_13307', result_not__13306)
        
        # Evaluating a boolean operation
        
        
        # Call to get_content_maintype(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_13310 = {}
        # Getting the type of 'root' (line 193)
        root_13308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'root', False)
        # Obtaining the member 'get_content_maintype' of a type (line 193)
        get_content_maintype_13309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 11), root_13308, 'get_content_maintype')
        # Calling get_content_maintype(args, kwargs) (line 193)
        get_content_maintype_call_result_13311 = invoke(stypy.reporting.localization.Localization(__file__, 193, 11), get_content_maintype_13309, *[], **kwargs_13310)
        
        str_13312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 42), 'str', 'multipart')
        # Applying the binary operator '==' (line 193)
        result_eq_13313 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), '==', get_content_maintype_call_result_13311, str_13312)
        
        
        
        # Call to is_multipart(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_13316 = {}
        # Getting the type of 'root' (line 194)
        root_13314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 23), 'root', False)
        # Obtaining the member 'is_multipart' of a type (line 194)
        is_multipart_13315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 23), root_13314, 'is_multipart')
        # Calling is_multipart(args, kwargs) (line 194)
        is_multipart_call_result_13317 = invoke(stypy.reporting.localization.Localization(__file__, 194, 23), is_multipart_13315, *[], **kwargs_13316)
        
        # Applying the 'not' unary operator (line 194)
        result_not__13318 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 19), 'not', is_multipart_call_result_13317)
        
        # Applying the binary operator 'and' (line 193)
        result_and_keyword_13319 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), 'and', result_eq_13313, result_not__13318)
        
        # Testing if the type of an if condition is none (line 193)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 193, 8), result_and_keyword_13319):
            pass
        else:
            
            # Testing the type of an if condition (line 193)
            if_condition_13320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), result_and_keyword_13319)
            # Assigning a type to the variable 'if_condition_13320' (line 193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_13320', if_condition_13320)
            # SSA begins for if statement (line 193)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 195)
            # Processing the call arguments (line 195)
            
            # Call to MultipartInvariantViolationDefect(...): (line 195)
            # Processing the call keyword arguments (line 195)
            kwargs_13326 = {}
            # Getting the type of 'errors' (line 195)
            errors_13324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 32), 'errors', False)
            # Obtaining the member 'MultipartInvariantViolationDefect' of a type (line 195)
            MultipartInvariantViolationDefect_13325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 32), errors_13324, 'MultipartInvariantViolationDefect')
            # Calling MultipartInvariantViolationDefect(args, kwargs) (line 195)
            MultipartInvariantViolationDefect_call_result_13327 = invoke(stypy.reporting.localization.Localization(__file__, 195, 32), MultipartInvariantViolationDefect_13325, *[], **kwargs_13326)
            
            # Processing the call keyword arguments (line 195)
            kwargs_13328 = {}
            # Getting the type of 'root' (line 195)
            root_13321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'root', False)
            # Obtaining the member 'defects' of a type (line 195)
            defects_13322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), root_13321, 'defects')
            # Obtaining the member 'append' of a type (line 195)
            append_13323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), defects_13322, 'append')
            # Calling append(args, kwargs) (line 195)
            append_call_result_13329 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), append_13323, *[MultipartInvariantViolationDefect_call_result_13327], **kwargs_13328)
            
            # SSA join for if statement (line 193)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'root' (line 196)
        root_13330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'root')
        # Assigning a type to the variable 'stypy_return_type' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'stypy_return_type', root_13330)
        
        # ################# End of 'close(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'close' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_13331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13331)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'close'
        return stypy_return_type_13331


    @norecursion
    def _new_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_new_message'
        module_type_store = module_type_store.open_function_context('_new_message', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FeedParser._new_message.__dict__.__setitem__('stypy_localization', localization)
        FeedParser._new_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FeedParser._new_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        FeedParser._new_message.__dict__.__setitem__('stypy_function_name', 'FeedParser._new_message')
        FeedParser._new_message.__dict__.__setitem__('stypy_param_names_list', [])
        FeedParser._new_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        FeedParser._new_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FeedParser._new_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        FeedParser._new_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        FeedParser._new_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FeedParser._new_message.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FeedParser._new_message', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_new_message', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_new_message(...)' code ##################

        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to _factory(...): (line 199)
        # Processing the call keyword arguments (line 199)
        kwargs_13334 = {}
        # Getting the type of 'self' (line 199)
        self_13332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 14), 'self', False)
        # Obtaining the member '_factory' of a type (line 199)
        _factory_13333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 14), self_13332, '_factory')
        # Calling _factory(args, kwargs) (line 199)
        _factory_call_result_13335 = invoke(stypy.reporting.localization.Localization(__file__, 199, 14), _factory_13333, *[], **kwargs_13334)
        
        # Assigning a type to the variable 'msg' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'msg', _factory_call_result_13335)
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 200)
        self_13336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'self')
        # Obtaining the member '_cur' of a type (line 200)
        _cur_13337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), self_13336, '_cur')
        
        
        # Call to get_content_type(...): (line 200)
        # Processing the call keyword arguments (line 200)
        kwargs_13341 = {}
        # Getting the type of 'self' (line 200)
        self_13338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'self', False)
        # Obtaining the member '_cur' of a type (line 200)
        _cur_13339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 25), self_13338, '_cur')
        # Obtaining the member 'get_content_type' of a type (line 200)
        get_content_type_13340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 25), _cur_13339, 'get_content_type')
        # Calling get_content_type(args, kwargs) (line 200)
        get_content_type_call_result_13342 = invoke(stypy.reporting.localization.Localization(__file__, 200, 25), get_content_type_13340, *[], **kwargs_13341)
        
        str_13343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 57), 'str', 'multipart/digest')
        # Applying the binary operator '==' (line 200)
        result_eq_13344 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 25), '==', get_content_type_call_result_13342, str_13343)
        
        # Applying the binary operator 'and' (line 200)
        result_and_keyword_13345 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 11), 'and', _cur_13337, result_eq_13344)
        
        # Testing if the type of an if condition is none (line 200)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 200, 8), result_and_keyword_13345):
            pass
        else:
            
            # Testing the type of an if condition (line 200)
            if_condition_13346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 8), result_and_keyword_13345)
            # Assigning a type to the variable 'if_condition_13346' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'if_condition_13346', if_condition_13346)
            # SSA begins for if statement (line 200)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to set_default_type(...): (line 201)
            # Processing the call arguments (line 201)
            str_13349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 33), 'str', 'message/rfc822')
            # Processing the call keyword arguments (line 201)
            kwargs_13350 = {}
            # Getting the type of 'msg' (line 201)
            msg_13347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'msg', False)
            # Obtaining the member 'set_default_type' of a type (line 201)
            set_default_type_13348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 12), msg_13347, 'set_default_type')
            # Calling set_default_type(args, kwargs) (line 201)
            set_default_type_call_result_13351 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), set_default_type_13348, *[str_13349], **kwargs_13350)
            
            # SSA join for if statement (line 200)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'self' (line 202)
        self_13352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'self')
        # Obtaining the member '_msgstack' of a type (line 202)
        _msgstack_13353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 11), self_13352, '_msgstack')
        # Testing if the type of an if condition is none (line 202)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 202, 8), _msgstack_13353):
            pass
        else:
            
            # Testing the type of an if condition (line 202)
            if_condition_13354 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), _msgstack_13353)
            # Assigning a type to the variable 'if_condition_13354' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_13354', if_condition_13354)
            # SSA begins for if statement (line 202)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to attach(...): (line 203)
            # Processing the call arguments (line 203)
            # Getting the type of 'msg' (line 203)
            msg_13361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 38), 'msg', False)
            # Processing the call keyword arguments (line 203)
            kwargs_13362 = {}
            
            # Obtaining the type of the subscript
            int_13355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 27), 'int')
            # Getting the type of 'self' (line 203)
            self_13356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'self', False)
            # Obtaining the member '_msgstack' of a type (line 203)
            _msgstack_13357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), self_13356, '_msgstack')
            # Obtaining the member '__getitem__' of a type (line 203)
            getitem___13358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), _msgstack_13357, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 203)
            subscript_call_result_13359 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), getitem___13358, int_13355)
            
            # Obtaining the member 'attach' of a type (line 203)
            attach_13360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), subscript_call_result_13359, 'attach')
            # Calling attach(args, kwargs) (line 203)
            attach_call_result_13363 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), attach_13360, *[msg_13361], **kwargs_13362)
            
            # SSA join for if statement (line 202)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to append(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'msg' (line 204)
        msg_13367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 30), 'msg', False)
        # Processing the call keyword arguments (line 204)
        kwargs_13368 = {}
        # Getting the type of 'self' (line 204)
        self_13364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self', False)
        # Obtaining the member '_msgstack' of a type (line 204)
        _msgstack_13365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_13364, '_msgstack')
        # Obtaining the member 'append' of a type (line 204)
        append_13366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), _msgstack_13365, 'append')
        # Calling append(args, kwargs) (line 204)
        append_call_result_13369 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), append_13366, *[msg_13367], **kwargs_13368)
        
        
        # Assigning a Name to a Attribute (line 205):
        
        # Assigning a Name to a Attribute (line 205):
        # Getting the type of 'msg' (line 205)
        msg_13370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'msg')
        # Getting the type of 'self' (line 205)
        self_13371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member '_cur' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_13371, '_cur', msg_13370)
        
        # Assigning a Name to a Attribute (line 206):
        
        # Assigning a Name to a Attribute (line 206):
        # Getting the type of 'msg' (line 206)
        msg_13372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 21), 'msg')
        # Getting the type of 'self' (line 206)
        self_13373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member '_last' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_13373, '_last', msg_13372)
        
        # ################# End of '_new_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_new_message' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_13374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13374)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_new_message'
        return stypy_return_type_13374


    @norecursion
    def _pop_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pop_message'
        module_type_store = module_type_store.open_function_context('_pop_message', 208, 4, False)
        # Assigning a type to the variable 'self' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FeedParser._pop_message.__dict__.__setitem__('stypy_localization', localization)
        FeedParser._pop_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FeedParser._pop_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        FeedParser._pop_message.__dict__.__setitem__('stypy_function_name', 'FeedParser._pop_message')
        FeedParser._pop_message.__dict__.__setitem__('stypy_param_names_list', [])
        FeedParser._pop_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        FeedParser._pop_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FeedParser._pop_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        FeedParser._pop_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        FeedParser._pop_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FeedParser._pop_message.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FeedParser._pop_message', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pop_message', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pop_message(...)' code ##################

        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to pop(...): (line 209)
        # Processing the call keyword arguments (line 209)
        kwargs_13378 = {}
        # Getting the type of 'self' (line 209)
        self_13375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 17), 'self', False)
        # Obtaining the member '_msgstack' of a type (line 209)
        _msgstack_13376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 17), self_13375, '_msgstack')
        # Obtaining the member 'pop' of a type (line 209)
        pop_13377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 17), _msgstack_13376, 'pop')
        # Calling pop(args, kwargs) (line 209)
        pop_call_result_13379 = invoke(stypy.reporting.localization.Localization(__file__, 209, 17), pop_13377, *[], **kwargs_13378)
        
        # Assigning a type to the variable 'retval' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'retval', pop_call_result_13379)
        # Getting the type of 'self' (line 210)
        self_13380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'self')
        # Obtaining the member '_msgstack' of a type (line 210)
        _msgstack_13381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 11), self_13380, '_msgstack')
        # Testing if the type of an if condition is none (line 210)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 210, 8), _msgstack_13381):
            
            # Assigning a Name to a Attribute (line 213):
            
            # Assigning a Name to a Attribute (line 213):
            # Getting the type of 'None' (line 213)
            None_13389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'None')
            # Getting the type of 'self' (line 213)
            self_13390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'self')
            # Setting the type of the member '_cur' of a type (line 213)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), self_13390, '_cur', None_13389)
        else:
            
            # Testing the type of an if condition (line 210)
            if_condition_13382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), _msgstack_13381)
            # Assigning a type to the variable 'if_condition_13382' (line 210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_13382', if_condition_13382)
            # SSA begins for if statement (line 210)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Attribute (line 211):
            
            # Assigning a Subscript to a Attribute (line 211):
            
            # Obtaining the type of the subscript
            int_13383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 39), 'int')
            # Getting the type of 'self' (line 211)
            self_13384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'self')
            # Obtaining the member '_msgstack' of a type (line 211)
            _msgstack_13385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 24), self_13384, '_msgstack')
            # Obtaining the member '__getitem__' of a type (line 211)
            getitem___13386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 24), _msgstack_13385, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 211)
            subscript_call_result_13387 = invoke(stypy.reporting.localization.Localization(__file__, 211, 24), getitem___13386, int_13383)
            
            # Getting the type of 'self' (line 211)
            self_13388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'self')
            # Setting the type of the member '_cur' of a type (line 211)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), self_13388, '_cur', subscript_call_result_13387)
            # SSA branch for the else part of an if statement (line 210)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Attribute (line 213):
            
            # Assigning a Name to a Attribute (line 213):
            # Getting the type of 'None' (line 213)
            None_13389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'None')
            # Getting the type of 'self' (line 213)
            self_13390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'self')
            # Setting the type of the member '_cur' of a type (line 213)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), self_13390, '_cur', None_13389)
            # SSA join for if statement (line 210)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'retval' (line 214)
        retval_13391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'retval')
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'stypy_return_type', retval_13391)
        
        # ################# End of '_pop_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pop_message' in the type store
        # Getting the type of 'stypy_return_type' (line 208)
        stypy_return_type_13392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13392)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pop_message'
        return stypy_return_type_13392


    @norecursion
    def _parsegen(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parsegen'
        module_type_store = module_type_store.open_function_context('_parsegen', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FeedParser._parsegen.__dict__.__setitem__('stypy_localization', localization)
        FeedParser._parsegen.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FeedParser._parsegen.__dict__.__setitem__('stypy_type_store', module_type_store)
        FeedParser._parsegen.__dict__.__setitem__('stypy_function_name', 'FeedParser._parsegen')
        FeedParser._parsegen.__dict__.__setitem__('stypy_param_names_list', [])
        FeedParser._parsegen.__dict__.__setitem__('stypy_varargs_param_name', None)
        FeedParser._parsegen.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FeedParser._parsegen.__dict__.__setitem__('stypy_call_defaults', defaults)
        FeedParser._parsegen.__dict__.__setitem__('stypy_call_varargs', varargs)
        FeedParser._parsegen.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FeedParser._parsegen.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FeedParser._parsegen', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parsegen', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parsegen(...)' code ##################

        
        # Call to _new_message(...): (line 218)
        # Processing the call keyword arguments (line 218)
        kwargs_13395 = {}
        # Getting the type of 'self' (line 218)
        self_13393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'self', False)
        # Obtaining the member '_new_message' of a type (line 218)
        _new_message_13394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), self_13393, '_new_message')
        # Calling _new_message(args, kwargs) (line 218)
        _new_message_call_result_13396 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), _new_message_13394, *[], **kwargs_13395)
        
        
        # Assigning a List to a Name (line 219):
        
        # Assigning a List to a Name (line 219):
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_13397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        
        # Assigning a type to the variable 'headers' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'headers', list_13397)
        
        # Getting the type of 'self' (line 222)
        self_13398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'self')
        # Obtaining the member '_input' of a type (line 222)
        _input_13399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 20), self_13398, '_input')
        # Assigning a type to the variable '_input_13399' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), '_input_13399', _input_13399)
        # Testing if the for loop is going to be iterated (line 222)
        # Testing the type of a for loop iterable (line 222)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 222, 8), _input_13399)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 222, 8), _input_13399):
            # Getting the type of the for loop variable (line 222)
            for_loop_var_13400 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 222, 8), _input_13399)
            # Assigning a type to the variable 'line' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'line', for_loop_var_13400)
            # SSA begins for a for statement (line 222)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'line' (line 223)
            line_13401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'line')
            # Getting the type of 'NeedMoreData' (line 223)
            NeedMoreData_13402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'NeedMoreData')
            # Applying the binary operator 'is' (line 223)
            result_is__13403 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 15), 'is', line_13401, NeedMoreData_13402)
            
            # Testing if the type of an if condition is none (line 223)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 223, 12), result_is__13403):
                pass
            else:
                
                # Testing the type of an if condition (line 223)
                if_condition_13404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 12), result_is__13403)
                # Assigning a type to the variable 'if_condition_13404' (line 223)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'if_condition_13404', if_condition_13404)
                # SSA begins for if statement (line 223)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Creating a generator
                # Getting the type of 'NeedMoreData' (line 224)
                NeedMoreData_13405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 22), 'NeedMoreData')
                GeneratorType_13406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 16), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 16), GeneratorType_13406, NeedMoreData_13405)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'stypy_return_type', GeneratorType_13406)
                # SSA join for if statement (line 223)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to match(...): (line 226)
            # Processing the call arguments (line 226)
            # Getting the type of 'line' (line 226)
            line_13409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 34), 'line', False)
            # Processing the call keyword arguments (line 226)
            kwargs_13410 = {}
            # Getting the type of 'headerRE' (line 226)
            headerRE_13407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'headerRE', False)
            # Obtaining the member 'match' of a type (line 226)
            match_13408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 19), headerRE_13407, 'match')
            # Calling match(args, kwargs) (line 226)
            match_call_result_13411 = invoke(stypy.reporting.localization.Localization(__file__, 226, 19), match_13408, *[line_13409], **kwargs_13410)
            
            # Applying the 'not' unary operator (line 226)
            result_not__13412 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 15), 'not', match_call_result_13411)
            
            # Testing if the type of an if condition is none (line 226)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 226, 12), result_not__13412):
                pass
            else:
                
                # Testing the type of an if condition (line 226)
                if_condition_13413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 12), result_not__13412)
                # Assigning a type to the variable 'if_condition_13413' (line 226)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'if_condition_13413', if_condition_13413)
                # SSA begins for if statement (line 226)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to match(...): (line 230)
                # Processing the call arguments (line 230)
                # Getting the type of 'line' (line 230)
                line_13416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'line', False)
                # Processing the call keyword arguments (line 230)
                kwargs_13417 = {}
                # Getting the type of 'NLCRE' (line 230)
                NLCRE_13414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'NLCRE', False)
                # Obtaining the member 'match' of a type (line 230)
                match_13415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 23), NLCRE_13414, 'match')
                # Calling match(args, kwargs) (line 230)
                match_call_result_13418 = invoke(stypy.reporting.localization.Localization(__file__, 230, 23), match_13415, *[line_13416], **kwargs_13417)
                
                # Applying the 'not' unary operator (line 230)
                result_not__13419 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 19), 'not', match_call_result_13418)
                
                # Testing if the type of an if condition is none (line 230)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 230, 16), result_not__13419):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 230)
                    if_condition_13420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 16), result_not__13419)
                    # Assigning a type to the variable 'if_condition_13420' (line 230)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'if_condition_13420', if_condition_13420)
                    # SSA begins for if statement (line 230)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to unreadline(...): (line 231)
                    # Processing the call arguments (line 231)
                    # Getting the type of 'line' (line 231)
                    line_13424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 43), 'line', False)
                    # Processing the call keyword arguments (line 231)
                    kwargs_13425 = {}
                    # Getting the type of 'self' (line 231)
                    self_13421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'self', False)
                    # Obtaining the member '_input' of a type (line 231)
                    _input_13422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 20), self_13421, '_input')
                    # Obtaining the member 'unreadline' of a type (line 231)
                    unreadline_13423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 20), _input_13422, 'unreadline')
                    # Calling unreadline(args, kwargs) (line 231)
                    unreadline_call_result_13426 = invoke(stypy.reporting.localization.Localization(__file__, 231, 20), unreadline_13423, *[line_13424], **kwargs_13425)
                    
                    # SSA join for if statement (line 230)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 226)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to append(...): (line 233)
            # Processing the call arguments (line 233)
            # Getting the type of 'line' (line 233)
            line_13429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'line', False)
            # Processing the call keyword arguments (line 233)
            kwargs_13430 = {}
            # Getting the type of 'headers' (line 233)
            headers_13427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'headers', False)
            # Obtaining the member 'append' of a type (line 233)
            append_13428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 12), headers_13427, 'append')
            # Calling append(args, kwargs) (line 233)
            append_call_result_13431 = invoke(stypy.reporting.localization.Localization(__file__, 233, 12), append_13428, *[line_13429], **kwargs_13430)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to _parse_headers(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'headers' (line 236)
        headers_13434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 28), 'headers', False)
        # Processing the call keyword arguments (line 236)
        kwargs_13435 = {}
        # Getting the type of 'self' (line 236)
        self_13432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self', False)
        # Obtaining the member '_parse_headers' of a type (line 236)
        _parse_headers_13433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_13432, '_parse_headers')
        # Calling _parse_headers(args, kwargs) (line 236)
        _parse_headers_call_result_13436 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), _parse_headers_13433, *[headers_13434], **kwargs_13435)
        
        # Getting the type of 'self' (line 240)
        self_13437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'self')
        # Obtaining the member '_headersonly' of a type (line 240)
        _headersonly_13438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), self_13437, '_headersonly')
        # Testing if the type of an if condition is none (line 240)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 240, 8), _headersonly_13438):
            pass
        else:
            
            # Testing the type of an if condition (line 240)
            if_condition_13439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), _headersonly_13438)
            # Assigning a type to the variable 'if_condition_13439' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_13439', if_condition_13439)
            # SSA begins for if statement (line 240)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a List to a Name (line 241):
            
            # Assigning a List to a Name (line 241):
            
            # Obtaining an instance of the builtin type 'list' (line 241)
            list_13440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 241)
            
            # Assigning a type to the variable 'lines' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'lines', list_13440)
            
            # Getting the type of 'True' (line 242)
            True_13441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 18), 'True')
            # Assigning a type to the variable 'True_13441' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'True_13441', True_13441)
            # Testing if the while is going to be iterated (line 242)
            # Testing the type of an if condition (line 242)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 12), True_13441)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 242, 12), True_13441):
                
                # Assigning a Call to a Name (line 243):
                
                # Assigning a Call to a Name (line 243):
                
                # Call to readline(...): (line 243)
                # Processing the call keyword arguments (line 243)
                kwargs_13445 = {}
                # Getting the type of 'self' (line 243)
                self_13442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 23), 'self', False)
                # Obtaining the member '_input' of a type (line 243)
                _input_13443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 23), self_13442, '_input')
                # Obtaining the member 'readline' of a type (line 243)
                readline_13444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 23), _input_13443, 'readline')
                # Calling readline(args, kwargs) (line 243)
                readline_call_result_13446 = invoke(stypy.reporting.localization.Localization(__file__, 243, 23), readline_13444, *[], **kwargs_13445)
                
                # Assigning a type to the variable 'line' (line 243)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'line', readline_call_result_13446)
                
                # Getting the type of 'line' (line 244)
                line_13447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'line')
                # Getting the type of 'NeedMoreData' (line 244)
                NeedMoreData_13448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 27), 'NeedMoreData')
                # Applying the binary operator 'is' (line 244)
                result_is__13449 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 19), 'is', line_13447, NeedMoreData_13448)
                
                # Testing if the type of an if condition is none (line 244)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 244, 16), result_is__13449):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 244)
                    if_condition_13450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 16), result_is__13449)
                    # Assigning a type to the variable 'if_condition_13450' (line 244)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'if_condition_13450', if_condition_13450)
                    # SSA begins for if statement (line 244)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Creating a generator
                    # Getting the type of 'NeedMoreData' (line 245)
                    NeedMoreData_13451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 26), 'NeedMoreData')
                    GeneratorType_13452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 20), 'GeneratorType')
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 20), GeneratorType_13452, NeedMoreData_13451)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'stypy_return_type', GeneratorType_13452)
                    # SSA join for if statement (line 244)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'line' (line 247)
                line_13453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 19), 'line')
                str_13454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 27), 'str', '')
                # Applying the binary operator '==' (line 247)
                result_eq_13455 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 19), '==', line_13453, str_13454)
                
                # Testing if the type of an if condition is none (line 247)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 247, 16), result_eq_13455):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 247)
                    if_condition_13456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 16), result_eq_13455)
                    # Assigning a type to the variable 'if_condition_13456' (line 247)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'if_condition_13456', if_condition_13456)
                    # SSA begins for if statement (line 247)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 247)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to append(...): (line 249)
                # Processing the call arguments (line 249)
                # Getting the type of 'line' (line 249)
                line_13459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 29), 'line', False)
                # Processing the call keyword arguments (line 249)
                kwargs_13460 = {}
                # Getting the type of 'lines' (line 249)
                lines_13457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'lines', False)
                # Obtaining the member 'append' of a type (line 249)
                append_13458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 16), lines_13457, 'append')
                # Calling append(args, kwargs) (line 249)
                append_call_result_13461 = invoke(stypy.reporting.localization.Localization(__file__, 249, 16), append_13458, *[line_13459], **kwargs_13460)
                

            
            
            # Call to set_payload(...): (line 250)
            # Processing the call arguments (line 250)
            
            # Call to join(...): (line 250)
            # Processing the call arguments (line 250)
            # Getting the type of 'lines' (line 250)
            lines_13467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 51), 'lines', False)
            # Processing the call keyword arguments (line 250)
            kwargs_13468 = {}
            # Getting the type of 'EMPTYSTRING' (line 250)
            EMPTYSTRING_13465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 34), 'EMPTYSTRING', False)
            # Obtaining the member 'join' of a type (line 250)
            join_13466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 34), EMPTYSTRING_13465, 'join')
            # Calling join(args, kwargs) (line 250)
            join_call_result_13469 = invoke(stypy.reporting.localization.Localization(__file__, 250, 34), join_13466, *[lines_13467], **kwargs_13468)
            
            # Processing the call keyword arguments (line 250)
            kwargs_13470 = {}
            # Getting the type of 'self' (line 250)
            self_13462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'self', False)
            # Obtaining the member '_cur' of a type (line 250)
            _cur_13463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), self_13462, '_cur')
            # Obtaining the member 'set_payload' of a type (line 250)
            set_payload_13464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), _cur_13463, 'set_payload')
            # Calling set_payload(args, kwargs) (line 250)
            set_payload_call_result_13471 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), set_payload_13464, *[join_call_result_13469], **kwargs_13470)
            
            # Assigning a type to the variable 'stypy_return_type' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 240)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to get_content_type(...): (line 252)
        # Processing the call keyword arguments (line 252)
        kwargs_13475 = {}
        # Getting the type of 'self' (line 252)
        self_13472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 11), 'self', False)
        # Obtaining the member '_cur' of a type (line 252)
        _cur_13473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 11), self_13472, '_cur')
        # Obtaining the member 'get_content_type' of a type (line 252)
        get_content_type_13474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 11), _cur_13473, 'get_content_type')
        # Calling get_content_type(args, kwargs) (line 252)
        get_content_type_call_result_13476 = invoke(stypy.reporting.localization.Localization(__file__, 252, 11), get_content_type_13474, *[], **kwargs_13475)
        
        str_13477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 43), 'str', 'message/delivery-status')
        # Applying the binary operator '==' (line 252)
        result_eq_13478 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 11), '==', get_content_type_call_result_13476, str_13477)
        
        # Testing if the type of an if condition is none (line 252)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 252, 8), result_eq_13478):
            pass
        else:
            
            # Testing the type of an if condition (line 252)
            if_condition_13479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 8), result_eq_13478)
            # Assigning a type to the variable 'if_condition_13479' (line 252)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'if_condition_13479', if_condition_13479)
            # SSA begins for if statement (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'True' (line 258)
            True_13480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'True')
            # Assigning a type to the variable 'True_13480' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'True_13480', True_13480)
            # Testing if the while is going to be iterated (line 258)
            # Testing the type of an if condition (line 258)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 12), True_13480)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 258, 12), True_13480):
                
                # Call to push_eof_matcher(...): (line 259)
                # Processing the call arguments (line 259)
                # Getting the type of 'NLCRE' (line 259)
                NLCRE_13484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 45), 'NLCRE', False)
                # Obtaining the member 'match' of a type (line 259)
                match_13485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 45), NLCRE_13484, 'match')
                # Processing the call keyword arguments (line 259)
                kwargs_13486 = {}
                # Getting the type of 'self' (line 259)
                self_13481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'self', False)
                # Obtaining the member '_input' of a type (line 259)
                _input_13482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 16), self_13481, '_input')
                # Obtaining the member 'push_eof_matcher' of a type (line 259)
                push_eof_matcher_13483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 16), _input_13482, 'push_eof_matcher')
                # Calling push_eof_matcher(args, kwargs) (line 259)
                push_eof_matcher_call_result_13487 = invoke(stypy.reporting.localization.Localization(__file__, 259, 16), push_eof_matcher_13483, *[match_13485], **kwargs_13486)
                
                
                
                # Call to _parsegen(...): (line 260)
                # Processing the call keyword arguments (line 260)
                kwargs_13490 = {}
                # Getting the type of 'self' (line 260)
                self_13488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 30), 'self', False)
                # Obtaining the member '_parsegen' of a type (line 260)
                _parsegen_13489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 30), self_13488, '_parsegen')
                # Calling _parsegen(args, kwargs) (line 260)
                _parsegen_call_result_13491 = invoke(stypy.reporting.localization.Localization(__file__, 260, 30), _parsegen_13489, *[], **kwargs_13490)
                
                # Assigning a type to the variable '_parsegen_call_result_13491' (line 260)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), '_parsegen_call_result_13491', _parsegen_call_result_13491)
                # Testing if the for loop is going to be iterated (line 260)
                # Testing the type of a for loop iterable (line 260)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 260, 16), _parsegen_call_result_13491)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 260, 16), _parsegen_call_result_13491):
                    # Getting the type of the for loop variable (line 260)
                    for_loop_var_13492 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 260, 16), _parsegen_call_result_13491)
                    # Assigning a type to the variable 'retval' (line 260)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'retval', for_loop_var_13492)
                    # SSA begins for a for statement (line 260)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'retval' (line 261)
                    retval_13493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'retval')
                    # Getting the type of 'NeedMoreData' (line 261)
                    NeedMoreData_13494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 33), 'NeedMoreData')
                    # Applying the binary operator 'is' (line 261)
                    result_is__13495 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 23), 'is', retval_13493, NeedMoreData_13494)
                    
                    # Testing if the type of an if condition is none (line 261)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 261, 20), result_is__13495):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 261)
                        if_condition_13496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 20), result_is__13495)
                        # Assigning a type to the variable 'if_condition_13496' (line 261)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'if_condition_13496', if_condition_13496)
                        # SSA begins for if statement (line 261)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Creating a generator
                        # Getting the type of 'NeedMoreData' (line 262)
                        NeedMoreData_13497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 30), 'NeedMoreData')
                        GeneratorType_13498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 24), 'GeneratorType')
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 24), GeneratorType_13498, NeedMoreData_13497)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'stypy_return_type', GeneratorType_13498)
                        # SSA join for if statement (line 261)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a Call to a Name (line 265):
                
                # Assigning a Call to a Name (line 265):
                
                # Call to _pop_message(...): (line 265)
                # Processing the call keyword arguments (line 265)
                kwargs_13501 = {}
                # Getting the type of 'self' (line 265)
                self_13499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'self', False)
                # Obtaining the member '_pop_message' of a type (line 265)
                _pop_message_13500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 22), self_13499, '_pop_message')
                # Calling _pop_message(args, kwargs) (line 265)
                _pop_message_call_result_13502 = invoke(stypy.reporting.localization.Localization(__file__, 265, 22), _pop_message_13500, *[], **kwargs_13501)
                
                # Assigning a type to the variable 'msg' (line 265)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'msg', _pop_message_call_result_13502)
                
                # Call to pop_eof_matcher(...): (line 269)
                # Processing the call keyword arguments (line 269)
                kwargs_13506 = {}
                # Getting the type of 'self' (line 269)
                self_13503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'self', False)
                # Obtaining the member '_input' of a type (line 269)
                _input_13504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 16), self_13503, '_input')
                # Obtaining the member 'pop_eof_matcher' of a type (line 269)
                pop_eof_matcher_13505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 16), _input_13504, 'pop_eof_matcher')
                # Calling pop_eof_matcher(args, kwargs) (line 269)
                pop_eof_matcher_call_result_13507 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), pop_eof_matcher_13505, *[], **kwargs_13506)
                
                
                # Getting the type of 'True' (line 274)
                True_13508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'True')
                # Assigning a type to the variable 'True_13508' (line 274)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'True_13508', True_13508)
                # Testing if the while is going to be iterated (line 274)
                # Testing the type of an if condition (line 274)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 16), True_13508)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 274, 16), True_13508):
                    
                    # Assigning a Call to a Name (line 275):
                    
                    # Assigning a Call to a Name (line 275):
                    
                    # Call to readline(...): (line 275)
                    # Processing the call keyword arguments (line 275)
                    kwargs_13512 = {}
                    # Getting the type of 'self' (line 275)
                    self_13509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'self', False)
                    # Obtaining the member '_input' of a type (line 275)
                    _input_13510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 27), self_13509, '_input')
                    # Obtaining the member 'readline' of a type (line 275)
                    readline_13511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 27), _input_13510, 'readline')
                    # Calling readline(args, kwargs) (line 275)
                    readline_call_result_13513 = invoke(stypy.reporting.localization.Localization(__file__, 275, 27), readline_13511, *[], **kwargs_13512)
                    
                    # Assigning a type to the variable 'line' (line 275)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 20), 'line', readline_call_result_13513)
                    
                    # Getting the type of 'line' (line 276)
                    line_13514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 23), 'line')
                    # Getting the type of 'NeedMoreData' (line 276)
                    NeedMoreData_13515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 31), 'NeedMoreData')
                    # Applying the binary operator 'is' (line 276)
                    result_is__13516 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 23), 'is', line_13514, NeedMoreData_13515)
                    
                    # Testing if the type of an if condition is none (line 276)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 276, 20), result_is__13516):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 276)
                        if_condition_13517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 276, 20), result_is__13516)
                        # Assigning a type to the variable 'if_condition_13517' (line 276)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'if_condition_13517', if_condition_13517)
                        # SSA begins for if statement (line 276)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Creating a generator
                        # Getting the type of 'NeedMoreData' (line 277)
                        NeedMoreData_13518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 30), 'NeedMoreData')
                        GeneratorType_13519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 24), 'GeneratorType')
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 24), GeneratorType_13519, NeedMoreData_13518)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 24), 'stypy_return_type', GeneratorType_13519)
                        # SSA join for if statement (line 276)
                        module_type_store = module_type_store.join_ssa_context()
                        


                
                
                # Getting the type of 'True' (line 280)
                True_13520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 22), 'True')
                # Assigning a type to the variable 'True_13520' (line 280)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'True_13520', True_13520)
                # Testing if the while is going to be iterated (line 280)
                # Testing the type of an if condition (line 280)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 16), True_13520)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 280, 16), True_13520):
                    
                    # Assigning a Call to a Name (line 281):
                    
                    # Assigning a Call to a Name (line 281):
                    
                    # Call to readline(...): (line 281)
                    # Processing the call keyword arguments (line 281)
                    kwargs_13524 = {}
                    # Getting the type of 'self' (line 281)
                    self_13521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'self', False)
                    # Obtaining the member '_input' of a type (line 281)
                    _input_13522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 27), self_13521, '_input')
                    # Obtaining the member 'readline' of a type (line 281)
                    readline_13523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 27), _input_13522, 'readline')
                    # Calling readline(args, kwargs) (line 281)
                    readline_call_result_13525 = invoke(stypy.reporting.localization.Localization(__file__, 281, 27), readline_13523, *[], **kwargs_13524)
                    
                    # Assigning a type to the variable 'line' (line 281)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'line', readline_call_result_13525)
                    
                    # Getting the type of 'line' (line 282)
                    line_13526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 23), 'line')
                    # Getting the type of 'NeedMoreData' (line 282)
                    NeedMoreData_13527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 31), 'NeedMoreData')
                    # Applying the binary operator 'is' (line 282)
                    result_is__13528 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 23), 'is', line_13526, NeedMoreData_13527)
                    
                    # Testing if the type of an if condition is none (line 282)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 282, 20), result_is__13528):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 282)
                        if_condition_13529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 20), result_is__13528)
                        # Assigning a type to the variable 'if_condition_13529' (line 282)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'if_condition_13529', if_condition_13529)
                        # SSA begins for if statement (line 282)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Creating a generator
                        # Getting the type of 'NeedMoreData' (line 283)
                        NeedMoreData_13530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 30), 'NeedMoreData')
                        GeneratorType_13531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 24), 'GeneratorType')
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 24), GeneratorType_13531, NeedMoreData_13530)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'stypy_return_type', GeneratorType_13531)
                        # SSA join for if statement (line 282)
                        module_type_store = module_type_store.join_ssa_context()
                        


                
                
                # Getting the type of 'line' (line 286)
                line_13532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'line')
                str_13533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 27), 'str', '')
                # Applying the binary operator '==' (line 286)
                result_eq_13534 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 19), '==', line_13532, str_13533)
                
                # Testing if the type of an if condition is none (line 286)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 286, 16), result_eq_13534):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 286)
                    if_condition_13535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 16), result_eq_13534)
                    # Assigning a type to the variable 'if_condition_13535' (line 286)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'if_condition_13535', if_condition_13535)
                    # SSA begins for if statement (line 286)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 286)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to unreadline(...): (line 289)
                # Processing the call arguments (line 289)
                # Getting the type of 'line' (line 289)
                line_13539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 39), 'line', False)
                # Processing the call keyword arguments (line 289)
                kwargs_13540 = {}
                # Getting the type of 'self' (line 289)
                self_13536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'self', False)
                # Obtaining the member '_input' of a type (line 289)
                _input_13537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 16), self_13536, '_input')
                # Obtaining the member 'unreadline' of a type (line 289)
                unreadline_13538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 16), _input_13537, 'unreadline')
                # Calling unreadline(args, kwargs) (line 289)
                unreadline_call_result_13541 = invoke(stypy.reporting.localization.Localization(__file__, 289, 16), unreadline_13538, *[line_13539], **kwargs_13540)
                

            
            # Assigning a type to the variable 'stypy_return_type' (line 290)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 252)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to get_content_maintype(...): (line 291)
        # Processing the call keyword arguments (line 291)
        kwargs_13545 = {}
        # Getting the type of 'self' (line 291)
        self_13542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 11), 'self', False)
        # Obtaining the member '_cur' of a type (line 291)
        _cur_13543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 11), self_13542, '_cur')
        # Obtaining the member 'get_content_maintype' of a type (line 291)
        get_content_maintype_13544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 11), _cur_13543, 'get_content_maintype')
        # Calling get_content_maintype(args, kwargs) (line 291)
        get_content_maintype_call_result_13546 = invoke(stypy.reporting.localization.Localization(__file__, 291, 11), get_content_maintype_13544, *[], **kwargs_13545)
        
        str_13547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 47), 'str', 'message')
        # Applying the binary operator '==' (line 291)
        result_eq_13548 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 11), '==', get_content_maintype_call_result_13546, str_13547)
        
        # Testing if the type of an if condition is none (line 291)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 291, 8), result_eq_13548):
            pass
        else:
            
            # Testing the type of an if condition (line 291)
            if_condition_13549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 8), result_eq_13548)
            # Assigning a type to the variable 'if_condition_13549' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'if_condition_13549', if_condition_13549)
            # SSA begins for if statement (line 291)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to _parsegen(...): (line 294)
            # Processing the call keyword arguments (line 294)
            kwargs_13552 = {}
            # Getting the type of 'self' (line 294)
            self_13550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'self', False)
            # Obtaining the member '_parsegen' of a type (line 294)
            _parsegen_13551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 26), self_13550, '_parsegen')
            # Calling _parsegen(args, kwargs) (line 294)
            _parsegen_call_result_13553 = invoke(stypy.reporting.localization.Localization(__file__, 294, 26), _parsegen_13551, *[], **kwargs_13552)
            
            # Assigning a type to the variable '_parsegen_call_result_13553' (line 294)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), '_parsegen_call_result_13553', _parsegen_call_result_13553)
            # Testing if the for loop is going to be iterated (line 294)
            # Testing the type of a for loop iterable (line 294)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 294, 12), _parsegen_call_result_13553)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 294, 12), _parsegen_call_result_13553):
                # Getting the type of the for loop variable (line 294)
                for_loop_var_13554 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 294, 12), _parsegen_call_result_13553)
                # Assigning a type to the variable 'retval' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'retval', for_loop_var_13554)
                # SSA begins for a for statement (line 294)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'retval' (line 295)
                retval_13555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'retval')
                # Getting the type of 'NeedMoreData' (line 295)
                NeedMoreData_13556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 29), 'NeedMoreData')
                # Applying the binary operator 'is' (line 295)
                result_is__13557 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 19), 'is', retval_13555, NeedMoreData_13556)
                
                # Testing if the type of an if condition is none (line 295)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 295, 16), result_is__13557):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 295)
                    if_condition_13558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 16), result_is__13557)
                    # Assigning a type to the variable 'if_condition_13558' (line 295)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'if_condition_13558', if_condition_13558)
                    # SSA begins for if statement (line 295)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Creating a generator
                    # Getting the type of 'NeedMoreData' (line 296)
                    NeedMoreData_13559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 26), 'NeedMoreData')
                    GeneratorType_13560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 20), 'GeneratorType')
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 20), GeneratorType_13560, NeedMoreData_13559)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'stypy_return_type', GeneratorType_13560)
                    # SSA join for if statement (line 295)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to _pop_message(...): (line 299)
            # Processing the call keyword arguments (line 299)
            kwargs_13563 = {}
            # Getting the type of 'self' (line 299)
            self_13561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'self', False)
            # Obtaining the member '_pop_message' of a type (line 299)
            _pop_message_13562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), self_13561, '_pop_message')
            # Calling _pop_message(args, kwargs) (line 299)
            _pop_message_call_result_13564 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), _pop_message_13562, *[], **kwargs_13563)
            
            # Assigning a type to the variable 'stypy_return_type' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 291)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to get_content_maintype(...): (line 301)
        # Processing the call keyword arguments (line 301)
        kwargs_13568 = {}
        # Getting the type of 'self' (line 301)
        self_13565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'self', False)
        # Obtaining the member '_cur' of a type (line 301)
        _cur_13566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 11), self_13565, '_cur')
        # Obtaining the member 'get_content_maintype' of a type (line 301)
        get_content_maintype_13567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 11), _cur_13566, 'get_content_maintype')
        # Calling get_content_maintype(args, kwargs) (line 301)
        get_content_maintype_call_result_13569 = invoke(stypy.reporting.localization.Localization(__file__, 301, 11), get_content_maintype_13567, *[], **kwargs_13568)
        
        str_13570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 47), 'str', 'multipart')
        # Applying the binary operator '==' (line 301)
        result_eq_13571 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 11), '==', get_content_maintype_call_result_13569, str_13570)
        
        # Testing if the type of an if condition is none (line 301)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 301, 8), result_eq_13571):
            pass
        else:
            
            # Testing the type of an if condition (line 301)
            if_condition_13572 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 8), result_eq_13571)
            # Assigning a type to the variable 'if_condition_13572' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'if_condition_13572', if_condition_13572)
            # SSA begins for if statement (line 301)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 302):
            
            # Assigning a Call to a Name (line 302):
            
            # Call to get_boundary(...): (line 302)
            # Processing the call keyword arguments (line 302)
            kwargs_13576 = {}
            # Getting the type of 'self' (line 302)
            self_13573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 23), 'self', False)
            # Obtaining the member '_cur' of a type (line 302)
            _cur_13574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 23), self_13573, '_cur')
            # Obtaining the member 'get_boundary' of a type (line 302)
            get_boundary_13575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 23), _cur_13574, 'get_boundary')
            # Calling get_boundary(args, kwargs) (line 302)
            get_boundary_call_result_13577 = invoke(stypy.reporting.localization.Localization(__file__, 302, 23), get_boundary_13575, *[], **kwargs_13576)
            
            # Assigning a type to the variable 'boundary' (line 302)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'boundary', get_boundary_call_result_13577)
            
            # Type idiom detected: calculating its left and rigth part (line 303)
            # Getting the type of 'boundary' (line 303)
            boundary_13578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 15), 'boundary')
            # Getting the type of 'None' (line 303)
            None_13579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 27), 'None')
            
            (may_be_13580, more_types_in_union_13581) = may_be_none(boundary_13578, None_13579)

            if may_be_13580:

                if more_types_in_union_13581:
                    # Runtime conditional SSA (line 303)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 308)
                # Processing the call arguments (line 308)
                
                # Call to NoBoundaryInMultipartDefect(...): (line 308)
                # Processing the call keyword arguments (line 308)
                kwargs_13588 = {}
                # Getting the type of 'errors' (line 308)
                errors_13586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 41), 'errors', False)
                # Obtaining the member 'NoBoundaryInMultipartDefect' of a type (line 308)
                NoBoundaryInMultipartDefect_13587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 41), errors_13586, 'NoBoundaryInMultipartDefect')
                # Calling NoBoundaryInMultipartDefect(args, kwargs) (line 308)
                NoBoundaryInMultipartDefect_call_result_13589 = invoke(stypy.reporting.localization.Localization(__file__, 308, 41), NoBoundaryInMultipartDefect_13587, *[], **kwargs_13588)
                
                # Processing the call keyword arguments (line 308)
                kwargs_13590 = {}
                # Getting the type of 'self' (line 308)
                self_13582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'self', False)
                # Obtaining the member '_cur' of a type (line 308)
                _cur_13583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 16), self_13582, '_cur')
                # Obtaining the member 'defects' of a type (line 308)
                defects_13584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 16), _cur_13583, 'defects')
                # Obtaining the member 'append' of a type (line 308)
                append_13585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 16), defects_13584, 'append')
                # Calling append(args, kwargs) (line 308)
                append_call_result_13591 = invoke(stypy.reporting.localization.Localization(__file__, 308, 16), append_13585, *[NoBoundaryInMultipartDefect_call_result_13589], **kwargs_13590)
                
                
                # Assigning a List to a Name (line 309):
                
                # Assigning a List to a Name (line 309):
                
                # Obtaining an instance of the builtin type 'list' (line 309)
                list_13592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 24), 'list')
                # Adding type elements to the builtin type 'list' instance (line 309)
                
                # Assigning a type to the variable 'lines' (line 309)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'lines', list_13592)
                
                # Getting the type of 'self' (line 310)
                self_13593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 28), 'self')
                # Obtaining the member '_input' of a type (line 310)
                _input_13594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 28), self_13593, '_input')
                # Assigning a type to the variable '_input_13594' (line 310)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), '_input_13594', _input_13594)
                # Testing if the for loop is going to be iterated (line 310)
                # Testing the type of a for loop iterable (line 310)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 310, 16), _input_13594)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 310, 16), _input_13594):
                    # Getting the type of the for loop variable (line 310)
                    for_loop_var_13595 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 310, 16), _input_13594)
                    # Assigning a type to the variable 'line' (line 310)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'line', for_loop_var_13595)
                    # SSA begins for a for statement (line 310)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'line' (line 311)
                    line_13596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'line')
                    # Getting the type of 'NeedMoreData' (line 311)
                    NeedMoreData_13597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 31), 'NeedMoreData')
                    # Applying the binary operator 'is' (line 311)
                    result_is__13598 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 23), 'is', line_13596, NeedMoreData_13597)
                    
                    # Testing if the type of an if condition is none (line 311)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 311, 20), result_is__13598):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 311)
                        if_condition_13599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 20), result_is__13598)
                        # Assigning a type to the variable 'if_condition_13599' (line 311)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 20), 'if_condition_13599', if_condition_13599)
                        # SSA begins for if statement (line 311)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Creating a generator
                        # Getting the type of 'NeedMoreData' (line 312)
                        NeedMoreData_13600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 30), 'NeedMoreData')
                        GeneratorType_13601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 24), 'GeneratorType')
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 24), GeneratorType_13601, NeedMoreData_13600)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), 'stypy_return_type', GeneratorType_13601)
                        # SSA join for if statement (line 311)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Call to append(...): (line 314)
                    # Processing the call arguments (line 314)
                    # Getting the type of 'line' (line 314)
                    line_13604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 33), 'line', False)
                    # Processing the call keyword arguments (line 314)
                    kwargs_13605 = {}
                    # Getting the type of 'lines' (line 314)
                    lines_13602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'lines', False)
                    # Obtaining the member 'append' of a type (line 314)
                    append_13603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 20), lines_13602, 'append')
                    # Calling append(args, kwargs) (line 314)
                    append_call_result_13606 = invoke(stypy.reporting.localization.Localization(__file__, 314, 20), append_13603, *[line_13604], **kwargs_13605)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Call to set_payload(...): (line 315)
                # Processing the call arguments (line 315)
                
                # Call to join(...): (line 315)
                # Processing the call arguments (line 315)
                # Getting the type of 'lines' (line 315)
                lines_13612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 55), 'lines', False)
                # Processing the call keyword arguments (line 315)
                kwargs_13613 = {}
                # Getting the type of 'EMPTYSTRING' (line 315)
                EMPTYSTRING_13610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 38), 'EMPTYSTRING', False)
                # Obtaining the member 'join' of a type (line 315)
                join_13611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 38), EMPTYSTRING_13610, 'join')
                # Calling join(args, kwargs) (line 315)
                join_call_result_13614 = invoke(stypy.reporting.localization.Localization(__file__, 315, 38), join_13611, *[lines_13612], **kwargs_13613)
                
                # Processing the call keyword arguments (line 315)
                kwargs_13615 = {}
                # Getting the type of 'self' (line 315)
                self_13607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'self', False)
                # Obtaining the member '_cur' of a type (line 315)
                _cur_13608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 16), self_13607, '_cur')
                # Obtaining the member 'set_payload' of a type (line 315)
                set_payload_13609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 16), _cur_13608, 'set_payload')
                # Calling set_payload(args, kwargs) (line 315)
                set_payload_call_result_13616 = invoke(stypy.reporting.localization.Localization(__file__, 315, 16), set_payload_13609, *[join_call_result_13614], **kwargs_13615)
                
                # Assigning a type to the variable 'stypy_return_type' (line 316)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), 'stypy_return_type', types.NoneType)

                if more_types_in_union_13581:
                    # SSA join for if statement (line 303)
                    module_type_store = module_type_store.join_ssa_context()


            
            # Getting the type of 'boundary' (line 303)
            boundary_13617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'boundary')
            # Assigning a type to the variable 'boundary' (line 303)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'boundary', remove_type_from_union(boundary_13617, types.NoneType))
            
            # Assigning a BinOp to a Name (line 321):
            
            # Assigning a BinOp to a Name (line 321):
            str_13618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 24), 'str', '--')
            # Getting the type of 'boundary' (line 321)
            boundary_13619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 31), 'boundary')
            # Applying the binary operator '+' (line 321)
            result_add_13620 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 24), '+', str_13618, boundary_13619)
            
            # Assigning a type to the variable 'separator' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'separator', result_add_13620)
            
            # Assigning a Call to a Name (line 322):
            
            # Assigning a Call to a Name (line 322):
            
            # Call to compile(...): (line 322)
            # Processing the call arguments (line 322)
            str_13623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 16), 'str', '(?P<sep>')
            
            # Call to escape(...): (line 323)
            # Processing the call arguments (line 323)
            # Getting the type of 'separator' (line 323)
            separator_13626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 39), 'separator', False)
            # Processing the call keyword arguments (line 323)
            kwargs_13627 = {}
            # Getting the type of 're' (line 323)
            re_13624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 29), 're', False)
            # Obtaining the member 'escape' of a type (line 323)
            escape_13625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 29), re_13624, 'escape')
            # Calling escape(args, kwargs) (line 323)
            escape_call_result_13628 = invoke(stypy.reporting.localization.Localization(__file__, 323, 29), escape_13625, *[separator_13626], **kwargs_13627)
            
            # Applying the binary operator '+' (line 323)
            result_add_13629 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 16), '+', str_13623, escape_call_result_13628)
            
            str_13630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 16), 'str', ')(?P<end>--)?(?P<ws>[ \\t]*)(?P<linesep>\\r\\n|\\r|\\n)?$')
            # Applying the binary operator '+' (line 323)
            result_add_13631 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 50), '+', result_add_13629, str_13630)
            
            # Processing the call keyword arguments (line 322)
            kwargs_13632 = {}
            # Getting the type of 're' (line 322)
            re_13621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 're', False)
            # Obtaining the member 'compile' of a type (line 322)
            compile_13622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 25), re_13621, 'compile')
            # Calling compile(args, kwargs) (line 322)
            compile_call_result_13633 = invoke(stypy.reporting.localization.Localization(__file__, 322, 25), compile_13622, *[result_add_13631], **kwargs_13632)
            
            # Assigning a type to the variable 'boundaryre' (line 322)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'boundaryre', compile_call_result_13633)
            
            # Assigning a Name to a Name (line 325):
            
            # Assigning a Name to a Name (line 325):
            # Getting the type of 'True' (line 325)
            True_13634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 33), 'True')
            # Assigning a type to the variable 'capturing_preamble' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'capturing_preamble', True_13634)
            
            # Assigning a List to a Name (line 326):
            
            # Assigning a List to a Name (line 326):
            
            # Obtaining an instance of the builtin type 'list' (line 326)
            list_13635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 326)
            
            # Assigning a type to the variable 'preamble' (line 326)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'preamble', list_13635)
            
            # Assigning a Name to a Name (line 327):
            
            # Assigning a Name to a Name (line 327):
            # Getting the type of 'False' (line 327)
            False_13636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 22), 'False')
            # Assigning a type to the variable 'linesep' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'linesep', False_13636)
            
            # Getting the type of 'True' (line 328)
            True_13637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 18), 'True')
            # Assigning a type to the variable 'True_13637' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'True_13637', True_13637)
            # Testing if the while is going to be iterated (line 328)
            # Testing the type of an if condition (line 328)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 12), True_13637)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 328, 12), True_13637):
                
                # Assigning a Call to a Name (line 329):
                
                # Assigning a Call to a Name (line 329):
                
                # Call to readline(...): (line 329)
                # Processing the call keyword arguments (line 329)
                kwargs_13641 = {}
                # Getting the type of 'self' (line 329)
                self_13638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 23), 'self', False)
                # Obtaining the member '_input' of a type (line 329)
                _input_13639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 23), self_13638, '_input')
                # Obtaining the member 'readline' of a type (line 329)
                readline_13640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 23), _input_13639, 'readline')
                # Calling readline(args, kwargs) (line 329)
                readline_call_result_13642 = invoke(stypy.reporting.localization.Localization(__file__, 329, 23), readline_13640, *[], **kwargs_13641)
                
                # Assigning a type to the variable 'line' (line 329)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'line', readline_call_result_13642)
                
                # Getting the type of 'line' (line 330)
                line_13643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'line')
                # Getting the type of 'NeedMoreData' (line 330)
                NeedMoreData_13644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 27), 'NeedMoreData')
                # Applying the binary operator 'is' (line 330)
                result_is__13645 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 19), 'is', line_13643, NeedMoreData_13644)
                
                # Testing if the type of an if condition is none (line 330)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 330, 16), result_is__13645):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 330)
                    if_condition_13646 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 16), result_is__13645)
                    # Assigning a type to the variable 'if_condition_13646' (line 330)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'if_condition_13646', if_condition_13646)
                    # SSA begins for if statement (line 330)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Creating a generator
                    # Getting the type of 'NeedMoreData' (line 331)
                    NeedMoreData_13647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'NeedMoreData')
                    GeneratorType_13648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 20), 'GeneratorType')
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 20), GeneratorType_13648, NeedMoreData_13647)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'stypy_return_type', GeneratorType_13648)
                    # SSA join for if statement (line 330)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'line' (line 333)
                line_13649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 19), 'line')
                str_13650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 27), 'str', '')
                # Applying the binary operator '==' (line 333)
                result_eq_13651 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 19), '==', line_13649, str_13650)
                
                # Testing if the type of an if condition is none (line 333)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 333, 16), result_eq_13651):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 333)
                    if_condition_13652 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 16), result_eq_13651)
                    # Assigning a type to the variable 'if_condition_13652' (line 333)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'if_condition_13652', if_condition_13652)
                    # SSA begins for if statement (line 333)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 333)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Name (line 335):
                
                # Assigning a Call to a Name (line 335):
                
                # Call to match(...): (line 335)
                # Processing the call arguments (line 335)
                # Getting the type of 'line' (line 335)
                line_13655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 38), 'line', False)
                # Processing the call keyword arguments (line 335)
                kwargs_13656 = {}
                # Getting the type of 'boundaryre' (line 335)
                boundaryre_13653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'boundaryre', False)
                # Obtaining the member 'match' of a type (line 335)
                match_13654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 21), boundaryre_13653, 'match')
                # Calling match(args, kwargs) (line 335)
                match_call_result_13657 = invoke(stypy.reporting.localization.Localization(__file__, 335, 21), match_13654, *[line_13655], **kwargs_13656)
                
                # Assigning a type to the variable 'mo' (line 335)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'mo', match_call_result_13657)
                # Getting the type of 'mo' (line 336)
                mo_13658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'mo')
                # Testing if the type of an if condition is none (line 336)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 336, 16), mo_13658):
                    # Evaluating assert statement condition
                    # Getting the type of 'capturing_preamble' (line 405)
                    capturing_preamble_13851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 27), 'capturing_preamble')
                    assert_13852 = capturing_preamble_13851
                    # Assigning a type to the variable 'assert_13852' (line 405)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 20), 'assert_13852', capturing_preamble_13851)
                    
                    # Call to append(...): (line 406)
                    # Processing the call arguments (line 406)
                    # Getting the type of 'line' (line 406)
                    line_13855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 36), 'line', False)
                    # Processing the call keyword arguments (line 406)
                    kwargs_13856 = {}
                    # Getting the type of 'preamble' (line 406)
                    preamble_13853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'preamble', False)
                    # Obtaining the member 'append' of a type (line 406)
                    append_13854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 20), preamble_13853, 'append')
                    # Calling append(args, kwargs) (line 406)
                    append_call_result_13857 = invoke(stypy.reporting.localization.Localization(__file__, 406, 20), append_13854, *[line_13855], **kwargs_13856)
                    
                else:
                    
                    # Testing the type of an if condition (line 336)
                    if_condition_13659 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 16), mo_13658)
                    # Assigning a type to the variable 'if_condition_13659' (line 336)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'if_condition_13659', if_condition_13659)
                    # SSA begins for if statement (line 336)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to group(...): (line 341)
                    # Processing the call arguments (line 341)
                    str_13662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 32), 'str', 'end')
                    # Processing the call keyword arguments (line 341)
                    kwargs_13663 = {}
                    # Getting the type of 'mo' (line 341)
                    mo_13660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 23), 'mo', False)
                    # Obtaining the member 'group' of a type (line 341)
                    group_13661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 23), mo_13660, 'group')
                    # Calling group(args, kwargs) (line 341)
                    group_call_result_13664 = invoke(stypy.reporting.localization.Localization(__file__, 341, 23), group_13661, *[str_13662], **kwargs_13663)
                    
                    # Testing if the type of an if condition is none (line 341)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 341, 20), group_call_result_13664):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 341)
                        if_condition_13665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 20), group_call_result_13664)
                        # Assigning a type to the variable 'if_condition_13665' (line 341)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 20), 'if_condition_13665', if_condition_13665)
                        # SSA begins for if statement (line 341)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 342):
                        
                        # Assigning a Call to a Name (line 342):
                        
                        # Call to group(...): (line 342)
                        # Processing the call arguments (line 342)
                        str_13668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 43), 'str', 'linesep')
                        # Processing the call keyword arguments (line 342)
                        kwargs_13669 = {}
                        # Getting the type of 'mo' (line 342)
                        mo_13666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 34), 'mo', False)
                        # Obtaining the member 'group' of a type (line 342)
                        group_13667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 34), mo_13666, 'group')
                        # Calling group(args, kwargs) (line 342)
                        group_call_result_13670 = invoke(stypy.reporting.localization.Localization(__file__, 342, 34), group_13667, *[str_13668], **kwargs_13669)
                        
                        # Assigning a type to the variable 'linesep' (line 342)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 24), 'linesep', group_call_result_13670)
                        # SSA join for if statement (line 341)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # Getting the type of 'capturing_preamble' (line 345)
                    capturing_preamble_13671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 'capturing_preamble')
                    # Testing if the type of an if condition is none (line 345)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 345, 20), capturing_preamble_13671):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 345)
                        if_condition_13672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 345, 20), capturing_preamble_13671)
                        # Assigning a type to the variable 'if_condition_13672' (line 345)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 20), 'if_condition_13672', if_condition_13672)
                        # SSA begins for if statement (line 345)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'preamble' (line 346)
                        preamble_13673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 27), 'preamble')
                        # Testing if the type of an if condition is none (line 346)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 346, 24), preamble_13673):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 346)
                            if_condition_13674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 24), preamble_13673)
                            # Assigning a type to the variable 'if_condition_13674' (line 346)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 24), 'if_condition_13674', if_condition_13674)
                            # SSA begins for if statement (line 346)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Subscript to a Name (line 349):
                            
                            # Assigning a Subscript to a Name (line 349):
                            
                            # Obtaining the type of the subscript
                            int_13675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 48), 'int')
                            # Getting the type of 'preamble' (line 349)
                            preamble_13676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 39), 'preamble')
                            # Obtaining the member '__getitem__' of a type (line 349)
                            getitem___13677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 39), preamble_13676, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 349)
                            subscript_call_result_13678 = invoke(stypy.reporting.localization.Localization(__file__, 349, 39), getitem___13677, int_13675)
                            
                            # Assigning a type to the variable 'lastline' (line 349)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 28), 'lastline', subscript_call_result_13678)
                            
                            # Assigning a Call to a Name (line 350):
                            
                            # Assigning a Call to a Name (line 350):
                            
                            # Call to search(...): (line 350)
                            # Processing the call arguments (line 350)
                            # Getting the type of 'lastline' (line 350)
                            lastline_13681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 53), 'lastline', False)
                            # Processing the call keyword arguments (line 350)
                            kwargs_13682 = {}
                            # Getting the type of 'NLCRE_eol' (line 350)
                            NLCRE_eol_13679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 36), 'NLCRE_eol', False)
                            # Obtaining the member 'search' of a type (line 350)
                            search_13680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 36), NLCRE_eol_13679, 'search')
                            # Calling search(args, kwargs) (line 350)
                            search_call_result_13683 = invoke(stypy.reporting.localization.Localization(__file__, 350, 36), search_13680, *[lastline_13681], **kwargs_13682)
                            
                            # Assigning a type to the variable 'eolmo' (line 350)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 28), 'eolmo', search_call_result_13683)
                            # Getting the type of 'eolmo' (line 351)
                            eolmo_13684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 31), 'eolmo')
                            # Testing if the type of an if condition is none (line 351)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 351, 28), eolmo_13684):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 351)
                                if_condition_13685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 28), eolmo_13684)
                                # Assigning a type to the variable 'if_condition_13685' (line 351)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 28), 'if_condition_13685', if_condition_13685)
                                # SSA begins for if statement (line 351)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Subscript to a Subscript (line 352):
                                
                                # Assigning a Subscript to a Subscript (line 352):
                                
                                # Obtaining the type of the subscript
                                
                                
                                # Call to len(...): (line 352)
                                # Processing the call arguments (line 352)
                                
                                # Call to group(...): (line 352)
                                # Processing the call arguments (line 352)
                                int_13689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 74), 'int')
                                # Processing the call keyword arguments (line 352)
                                kwargs_13690 = {}
                                # Getting the type of 'eolmo' (line 352)
                                eolmo_13687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 62), 'eolmo', False)
                                # Obtaining the member 'group' of a type (line 352)
                                group_13688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 62), eolmo_13687, 'group')
                                # Calling group(args, kwargs) (line 352)
                                group_call_result_13691 = invoke(stypy.reporting.localization.Localization(__file__, 352, 62), group_13688, *[int_13689], **kwargs_13690)
                                
                                # Processing the call keyword arguments (line 352)
                                kwargs_13692 = {}
                                # Getting the type of 'len' (line 352)
                                len_13686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 58), 'len', False)
                                # Calling len(args, kwargs) (line 352)
                                len_call_result_13693 = invoke(stypy.reporting.localization.Localization(__file__, 352, 58), len_13686, *[group_call_result_13691], **kwargs_13692)
                                
                                # Applying the 'usub' unary operator (line 352)
                                result___neg___13694 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 57), 'usub', len_call_result_13693)
                                
                                slice_13695 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 352, 47), None, result___neg___13694, None)
                                # Getting the type of 'lastline' (line 352)
                                lastline_13696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 47), 'lastline')
                                # Obtaining the member '__getitem__' of a type (line 352)
                                getitem___13697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 47), lastline_13696, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 352)
                                subscript_call_result_13698 = invoke(stypy.reporting.localization.Localization(__file__, 352, 47), getitem___13697, slice_13695)
                                
                                # Getting the type of 'preamble' (line 352)
                                preamble_13699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 32), 'preamble')
                                int_13700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 41), 'int')
                                # Storing an element on a container (line 352)
                                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 32), preamble_13699, (int_13700, subscript_call_result_13698))
                                # SSA join for if statement (line 351)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            
                            # Assigning a Call to a Attribute (line 353):
                            
                            # Assigning a Call to a Attribute (line 353):
                            
                            # Call to join(...): (line 353)
                            # Processing the call arguments (line 353)
                            # Getting the type of 'preamble' (line 353)
                            preamble_13703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 66), 'preamble', False)
                            # Processing the call keyword arguments (line 353)
                            kwargs_13704 = {}
                            # Getting the type of 'EMPTYSTRING' (line 353)
                            EMPTYSTRING_13701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 49), 'EMPTYSTRING', False)
                            # Obtaining the member 'join' of a type (line 353)
                            join_13702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 49), EMPTYSTRING_13701, 'join')
                            # Calling join(args, kwargs) (line 353)
                            join_call_result_13705 = invoke(stypy.reporting.localization.Localization(__file__, 353, 49), join_13702, *[preamble_13703], **kwargs_13704)
                            
                            # Getting the type of 'self' (line 353)
                            self_13706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 28), 'self')
                            # Obtaining the member '_cur' of a type (line 353)
                            _cur_13707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 28), self_13706, '_cur')
                            # Setting the type of the member 'preamble' of a type (line 353)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 28), _cur_13707, 'preamble', join_call_result_13705)
                            # SSA join for if statement (line 346)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        # Assigning a Name to a Name (line 354):
                        
                        # Assigning a Name to a Name (line 354):
                        # Getting the type of 'False' (line 354)
                        False_13708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 45), 'False')
                        # Assigning a type to the variable 'capturing_preamble' (line 354)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'capturing_preamble', False_13708)
                        
                        # Call to unreadline(...): (line 355)
                        # Processing the call arguments (line 355)
                        # Getting the type of 'line' (line 355)
                        line_13712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 47), 'line', False)
                        # Processing the call keyword arguments (line 355)
                        kwargs_13713 = {}
                        # Getting the type of 'self' (line 355)
                        self_13709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'self', False)
                        # Obtaining the member '_input' of a type (line 355)
                        _input_13710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 24), self_13709, '_input')
                        # Obtaining the member 'unreadline' of a type (line 355)
                        unreadline_13711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 24), _input_13710, 'unreadline')
                        # Calling unreadline(args, kwargs) (line 355)
                        unreadline_call_result_13714 = invoke(stypy.reporting.localization.Localization(__file__, 355, 24), unreadline_13711, *[line_13712], **kwargs_13713)
                        
                        # SSA join for if statement (line 345)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Getting the type of 'True' (line 361)
                    True_13715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 26), 'True')
                    # Assigning a type to the variable 'True_13715' (line 361)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'True_13715', True_13715)
                    # Testing if the while is going to be iterated (line 361)
                    # Testing the type of an if condition (line 361)
                    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 20), True_13715)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 361, 20), True_13715):
                        
                        # Assigning a Call to a Name (line 362):
                        
                        # Assigning a Call to a Name (line 362):
                        
                        # Call to readline(...): (line 362)
                        # Processing the call keyword arguments (line 362)
                        kwargs_13719 = {}
                        # Getting the type of 'self' (line 362)
                        self_13716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 31), 'self', False)
                        # Obtaining the member '_input' of a type (line 362)
                        _input_13717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 31), self_13716, '_input')
                        # Obtaining the member 'readline' of a type (line 362)
                        readline_13718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 31), _input_13717, 'readline')
                        # Calling readline(args, kwargs) (line 362)
                        readline_call_result_13720 = invoke(stypy.reporting.localization.Localization(__file__, 362, 31), readline_13718, *[], **kwargs_13719)
                        
                        # Assigning a type to the variable 'line' (line 362)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 24), 'line', readline_call_result_13720)
                        
                        # Getting the type of 'line' (line 363)
                        line_13721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'line')
                        # Getting the type of 'NeedMoreData' (line 363)
                        NeedMoreData_13722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 35), 'NeedMoreData')
                        # Applying the binary operator 'is' (line 363)
                        result_is__13723 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 27), 'is', line_13721, NeedMoreData_13722)
                        
                        # Testing if the type of an if condition is none (line 363)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 363, 24), result_is__13723):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 363)
                            if_condition_13724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 24), result_is__13723)
                            # Assigning a type to the variable 'if_condition_13724' (line 363)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 24), 'if_condition_13724', if_condition_13724)
                            # SSA begins for if statement (line 363)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # Creating a generator
                            # Getting the type of 'NeedMoreData' (line 364)
                            NeedMoreData_13725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 34), 'NeedMoreData')
                            GeneratorType_13726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 28), 'GeneratorType')
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 28), GeneratorType_13726, NeedMoreData_13725)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 28), 'stypy_return_type', GeneratorType_13726)
                            # SSA join for if statement (line 363)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        # Assigning a Call to a Name (line 366):
                        
                        # Assigning a Call to a Name (line 366):
                        
                        # Call to match(...): (line 366)
                        # Processing the call arguments (line 366)
                        # Getting the type of 'line' (line 366)
                        line_13729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 46), 'line', False)
                        # Processing the call keyword arguments (line 366)
                        kwargs_13730 = {}
                        # Getting the type of 'boundaryre' (line 366)
                        boundaryre_13727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 29), 'boundaryre', False)
                        # Obtaining the member 'match' of a type (line 366)
                        match_13728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 29), boundaryre_13727, 'match')
                        # Calling match(args, kwargs) (line 366)
                        match_call_result_13731 = invoke(stypy.reporting.localization.Localization(__file__, 366, 29), match_13728, *[line_13729], **kwargs_13730)
                        
                        # Assigning a type to the variable 'mo' (line 366)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 24), 'mo', match_call_result_13731)
                        
                        # Getting the type of 'mo' (line 367)
                        mo_13732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 31), 'mo')
                        # Applying the 'not' unary operator (line 367)
                        result_not__13733 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 27), 'not', mo_13732)
                        
                        # Testing if the type of an if condition is none (line 367)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 367, 24), result_not__13733):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 367)
                            if_condition_13734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 24), result_not__13733)
                            # Assigning a type to the variable 'if_condition_13734' (line 367)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 24), 'if_condition_13734', if_condition_13734)
                            # SSA begins for if statement (line 367)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to unreadline(...): (line 368)
                            # Processing the call arguments (line 368)
                            # Getting the type of 'line' (line 368)
                            line_13738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 51), 'line', False)
                            # Processing the call keyword arguments (line 368)
                            kwargs_13739 = {}
                            # Getting the type of 'self' (line 368)
                            self_13735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 28), 'self', False)
                            # Obtaining the member '_input' of a type (line 368)
                            _input_13736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 28), self_13735, '_input')
                            # Obtaining the member 'unreadline' of a type (line 368)
                            unreadline_13737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 28), _input_13736, 'unreadline')
                            # Calling unreadline(args, kwargs) (line 368)
                            unreadline_call_result_13740 = invoke(stypy.reporting.localization.Localization(__file__, 368, 28), unreadline_13737, *[line_13738], **kwargs_13739)
                            
                            # SSA join for if statement (line 367)
                            module_type_store = module_type_store.join_ssa_context()
                            


                    
                    
                    # Call to push_eof_matcher(...): (line 372)
                    # Processing the call arguments (line 372)
                    # Getting the type of 'boundaryre' (line 372)
                    boundaryre_13744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 49), 'boundaryre', False)
                    # Obtaining the member 'match' of a type (line 372)
                    match_13745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 49), boundaryre_13744, 'match')
                    # Processing the call keyword arguments (line 372)
                    kwargs_13746 = {}
                    # Getting the type of 'self' (line 372)
                    self_13741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'self', False)
                    # Obtaining the member '_input' of a type (line 372)
                    _input_13742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 20), self_13741, '_input')
                    # Obtaining the member 'push_eof_matcher' of a type (line 372)
                    push_eof_matcher_13743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 20), _input_13742, 'push_eof_matcher')
                    # Calling push_eof_matcher(args, kwargs) (line 372)
                    push_eof_matcher_call_result_13747 = invoke(stypy.reporting.localization.Localization(__file__, 372, 20), push_eof_matcher_13743, *[match_13745], **kwargs_13746)
                    
                    
                    
                    # Call to _parsegen(...): (line 373)
                    # Processing the call keyword arguments (line 373)
                    kwargs_13750 = {}
                    # Getting the type of 'self' (line 373)
                    self_13748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 34), 'self', False)
                    # Obtaining the member '_parsegen' of a type (line 373)
                    _parsegen_13749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 34), self_13748, '_parsegen')
                    # Calling _parsegen(args, kwargs) (line 373)
                    _parsegen_call_result_13751 = invoke(stypy.reporting.localization.Localization(__file__, 373, 34), _parsegen_13749, *[], **kwargs_13750)
                    
                    # Assigning a type to the variable '_parsegen_call_result_13751' (line 373)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 20), '_parsegen_call_result_13751', _parsegen_call_result_13751)
                    # Testing if the for loop is going to be iterated (line 373)
                    # Testing the type of a for loop iterable (line 373)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 373, 20), _parsegen_call_result_13751)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 373, 20), _parsegen_call_result_13751):
                        # Getting the type of the for loop variable (line 373)
                        for_loop_var_13752 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 373, 20), _parsegen_call_result_13751)
                        # Assigning a type to the variable 'retval' (line 373)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 20), 'retval', for_loop_var_13752)
                        # SSA begins for a for statement (line 373)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'retval' (line 374)
                        retval_13753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 27), 'retval')
                        # Getting the type of 'NeedMoreData' (line 374)
                        NeedMoreData_13754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 37), 'NeedMoreData')
                        # Applying the binary operator 'is' (line 374)
                        result_is__13755 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 27), 'is', retval_13753, NeedMoreData_13754)
                        
                        # Testing if the type of an if condition is none (line 374)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 374, 24), result_is__13755):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 374)
                            if_condition_13756 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 24), result_is__13755)
                            # Assigning a type to the variable 'if_condition_13756' (line 374)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 24), 'if_condition_13756', if_condition_13756)
                            # SSA begins for if statement (line 374)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # Creating a generator
                            # Getting the type of 'NeedMoreData' (line 375)
                            NeedMoreData_13757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 34), 'NeedMoreData')
                            GeneratorType_13758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 28), 'GeneratorType')
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 28), GeneratorType_13758, NeedMoreData_13757)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 28), 'stypy_return_type', GeneratorType_13758)
                            # SSA join for if statement (line 374)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to get_content_maintype(...): (line 382)
                    # Processing the call keyword arguments (line 382)
                    kwargs_13762 = {}
                    # Getting the type of 'self' (line 382)
                    self_13759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'self', False)
                    # Obtaining the member '_last' of a type (line 382)
                    _last_13760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 23), self_13759, '_last')
                    # Obtaining the member 'get_content_maintype' of a type (line 382)
                    get_content_maintype_13761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 23), _last_13760, 'get_content_maintype')
                    # Calling get_content_maintype(args, kwargs) (line 382)
                    get_content_maintype_call_result_13763 = invoke(stypy.reporting.localization.Localization(__file__, 382, 23), get_content_maintype_13761, *[], **kwargs_13762)
                    
                    str_13764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 60), 'str', 'multipart')
                    # Applying the binary operator '==' (line 382)
                    result_eq_13765 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 23), '==', get_content_maintype_call_result_13763, str_13764)
                    
                    # Testing if the type of an if condition is none (line 382)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 382, 20), result_eq_13765):
                        
                        # Assigning a Call to a Name (line 392):
                        
                        # Assigning a Call to a Name (line 392):
                        
                        # Call to get_payload(...): (line 392)
                        # Processing the call keyword arguments (line 392)
                        kwargs_13807 = {}
                        # Getting the type of 'self' (line 392)
                        self_13804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 34), 'self', False)
                        # Obtaining the member '_last' of a type (line 392)
                        _last_13805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 34), self_13804, '_last')
                        # Obtaining the member 'get_payload' of a type (line 392)
                        get_payload_13806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 34), _last_13805, 'get_payload')
                        # Calling get_payload(args, kwargs) (line 392)
                        get_payload_call_result_13808 = invoke(stypy.reporting.localization.Localization(__file__, 392, 34), get_payload_13806, *[], **kwargs_13807)
                        
                        # Assigning a type to the variable 'payload' (line 392)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 24), 'payload', get_payload_call_result_13808)
                        
                        # Type idiom detected: calculating its left and rigth part (line 393)
                        # Getting the type of 'basestring' (line 393)
                        basestring_13809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 47), 'basestring')
                        # Getting the type of 'payload' (line 393)
                        payload_13810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 38), 'payload')
                        
                        (may_be_13811, more_types_in_union_13812) = may_be_subtype(basestring_13809, payload_13810)

                        if may_be_13811:

                            if more_types_in_union_13812:
                                # Runtime conditional SSA (line 393)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                            else:
                                module_type_store = module_type_store

                            # Assigning a type to the variable 'payload' (line 393)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 24), 'payload', remove_not_subtype_from_union(payload_13810, basestring))
                            
                            # Assigning a Call to a Name (line 394):
                            
                            # Assigning a Call to a Name (line 394):
                            
                            # Call to search(...): (line 394)
                            # Processing the call arguments (line 394)
                            # Getting the type of 'payload' (line 394)
                            payload_13815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 50), 'payload', False)
                            # Processing the call keyword arguments (line 394)
                            kwargs_13816 = {}
                            # Getting the type of 'NLCRE_eol' (line 394)
                            NLCRE_eol_13813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 33), 'NLCRE_eol', False)
                            # Obtaining the member 'search' of a type (line 394)
                            search_13814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 33), NLCRE_eol_13813, 'search')
                            # Calling search(args, kwargs) (line 394)
                            search_call_result_13817 = invoke(stypy.reporting.localization.Localization(__file__, 394, 33), search_13814, *[payload_13815], **kwargs_13816)
                            
                            # Assigning a type to the variable 'mo' (line 394)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 28), 'mo', search_call_result_13817)
                            # Getting the type of 'mo' (line 395)
                            mo_13818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 31), 'mo')
                            # Testing if the type of an if condition is none (line 395)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 395, 28), mo_13818):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 395)
                                if_condition_13819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 28), mo_13818)
                                # Assigning a type to the variable 'if_condition_13819' (line 395)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 28), 'if_condition_13819', if_condition_13819)
                                # SSA begins for if statement (line 395)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Subscript to a Name (line 396):
                                
                                # Assigning a Subscript to a Name (line 396):
                                
                                # Obtaining the type of the subscript
                                
                                
                                # Call to len(...): (line 396)
                                # Processing the call arguments (line 396)
                                
                                # Call to group(...): (line 396)
                                # Processing the call arguments (line 396)
                                int_13823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 65), 'int')
                                # Processing the call keyword arguments (line 396)
                                kwargs_13824 = {}
                                # Getting the type of 'mo' (line 396)
                                mo_13821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 56), 'mo', False)
                                # Obtaining the member 'group' of a type (line 396)
                                group_13822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 56), mo_13821, 'group')
                                # Calling group(args, kwargs) (line 396)
                                group_call_result_13825 = invoke(stypy.reporting.localization.Localization(__file__, 396, 56), group_13822, *[int_13823], **kwargs_13824)
                                
                                # Processing the call keyword arguments (line 396)
                                kwargs_13826 = {}
                                # Getting the type of 'len' (line 396)
                                len_13820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 52), 'len', False)
                                # Calling len(args, kwargs) (line 396)
                                len_call_result_13827 = invoke(stypy.reporting.localization.Localization(__file__, 396, 52), len_13820, *[group_call_result_13825], **kwargs_13826)
                                
                                # Applying the 'usub' unary operator (line 396)
                                result___neg___13828 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 51), 'usub', len_call_result_13827)
                                
                                slice_13829 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 396, 42), None, result___neg___13828, None)
                                # Getting the type of 'payload' (line 396)
                                payload_13830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 42), 'payload')
                                # Obtaining the member '__getitem__' of a type (line 396)
                                getitem___13831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 42), payload_13830, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 396)
                                subscript_call_result_13832 = invoke(stypy.reporting.localization.Localization(__file__, 396, 42), getitem___13831, slice_13829)
                                
                                # Assigning a type to the variable 'payload' (line 396)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 32), 'payload', subscript_call_result_13832)
                                
                                # Call to set_payload(...): (line 397)
                                # Processing the call arguments (line 397)
                                # Getting the type of 'payload' (line 397)
                                payload_13836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 55), 'payload', False)
                                # Processing the call keyword arguments (line 397)
                                kwargs_13837 = {}
                                # Getting the type of 'self' (line 397)
                                self_13833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 32), 'self', False)
                                # Obtaining the member '_last' of a type (line 397)
                                _last_13834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 32), self_13833, '_last')
                                # Obtaining the member 'set_payload' of a type (line 397)
                                set_payload_13835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 32), _last_13834, 'set_payload')
                                # Calling set_payload(args, kwargs) (line 397)
                                set_payload_call_result_13838 = invoke(stypy.reporting.localization.Localization(__file__, 397, 32), set_payload_13835, *[payload_13836], **kwargs_13837)
                                
                                # SSA join for if statement (line 395)
                                module_type_store = module_type_store.join_ssa_context()
                                


                            if more_types_in_union_13812:
                                # SSA join for if statement (line 393)
                                module_type_store = module_type_store.join_ssa_context()


                        
                    else:
                        
                        # Testing the type of an if condition (line 382)
                        if_condition_13766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 20), result_eq_13765)
                        # Assigning a type to the variable 'if_condition_13766' (line 382)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'if_condition_13766', if_condition_13766)
                        # SSA begins for if statement (line 382)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Attribute to a Name (line 383):
                        
                        # Assigning a Attribute to a Name (line 383):
                        # Getting the type of 'self' (line 383)
                        self_13767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 35), 'self')
                        # Obtaining the member '_last' of a type (line 383)
                        _last_13768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 35), self_13767, '_last')
                        # Obtaining the member 'epilogue' of a type (line 383)
                        epilogue_13769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 35), _last_13768, 'epilogue')
                        # Assigning a type to the variable 'epilogue' (line 383)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 24), 'epilogue', epilogue_13769)
                        
                        # Getting the type of 'epilogue' (line 384)
                        epilogue_13770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'epilogue')
                        str_13771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 39), 'str', '')
                        # Applying the binary operator '==' (line 384)
                        result_eq_13772 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 27), '==', epilogue_13770, str_13771)
                        
                        # Testing if the type of an if condition is none (line 384)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 384, 24), result_eq_13772):
                            
                            # Type idiom detected: calculating its left and rigth part (line 386)
                            # Getting the type of 'epilogue' (line 386)
                            epilogue_13777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 29), 'epilogue')
                            # Getting the type of 'None' (line 386)
                            None_13778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 45), 'None')
                            
                            (may_be_13779, more_types_in_union_13780) = may_not_be_none(epilogue_13777, None_13778)

                            if may_be_13779:

                                if more_types_in_union_13780:
                                    # Runtime conditional SSA (line 386)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                                else:
                                    module_type_store = module_type_store

                                
                                # Assigning a Call to a Name (line 387):
                                
                                # Assigning a Call to a Name (line 387):
                                
                                # Call to search(...): (line 387)
                                # Processing the call arguments (line 387)
                                # Getting the type of 'epilogue' (line 387)
                                epilogue_13783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 50), 'epilogue', False)
                                # Processing the call keyword arguments (line 387)
                                kwargs_13784 = {}
                                # Getting the type of 'NLCRE_eol' (line 387)
                                NLCRE_eol_13781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 33), 'NLCRE_eol', False)
                                # Obtaining the member 'search' of a type (line 387)
                                search_13782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 33), NLCRE_eol_13781, 'search')
                                # Calling search(args, kwargs) (line 387)
                                search_call_result_13785 = invoke(stypy.reporting.localization.Localization(__file__, 387, 33), search_13782, *[epilogue_13783], **kwargs_13784)
                                
                                # Assigning a type to the variable 'mo' (line 387)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 28), 'mo', search_call_result_13785)
                                # Getting the type of 'mo' (line 388)
                                mo_13786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 31), 'mo')
                                # Testing if the type of an if condition is none (line 388)

                                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 388, 28), mo_13786):
                                    pass
                                else:
                                    
                                    # Testing the type of an if condition (line 388)
                                    if_condition_13787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 28), mo_13786)
                                    # Assigning a type to the variable 'if_condition_13787' (line 388)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 28), 'if_condition_13787', if_condition_13787)
                                    # SSA begins for if statement (line 388)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                    
                                    # Assigning a Call to a Name (line 389):
                                    
                                    # Assigning a Call to a Name (line 389):
                                    
                                    # Call to len(...): (line 389)
                                    # Processing the call arguments (line 389)
                                    
                                    # Call to group(...): (line 389)
                                    # Processing the call arguments (line 389)
                                    int_13791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 51), 'int')
                                    # Processing the call keyword arguments (line 389)
                                    kwargs_13792 = {}
                                    # Getting the type of 'mo' (line 389)
                                    mo_13789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 42), 'mo', False)
                                    # Obtaining the member 'group' of a type (line 389)
                                    group_13790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 42), mo_13789, 'group')
                                    # Calling group(args, kwargs) (line 389)
                                    group_call_result_13793 = invoke(stypy.reporting.localization.Localization(__file__, 389, 42), group_13790, *[int_13791], **kwargs_13792)
                                    
                                    # Processing the call keyword arguments (line 389)
                                    kwargs_13794 = {}
                                    # Getting the type of 'len' (line 389)
                                    len_13788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 38), 'len', False)
                                    # Calling len(args, kwargs) (line 389)
                                    len_call_result_13795 = invoke(stypy.reporting.localization.Localization(__file__, 389, 38), len_13788, *[group_call_result_13793], **kwargs_13794)
                                    
                                    # Assigning a type to the variable 'end' (line 389)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 32), 'end', len_call_result_13795)
                                    
                                    # Assigning a Subscript to a Attribute (line 390):
                                    
                                    # Assigning a Subscript to a Attribute (line 390):
                                    
                                    # Obtaining the type of the subscript
                                    
                                    # Getting the type of 'end' (line 390)
                                    end_13796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 65), 'end')
                                    # Applying the 'usub' unary operator (line 390)
                                    result___neg___13797 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 64), 'usub', end_13796)
                                    
                                    slice_13798 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 390, 54), None, result___neg___13797, None)
                                    # Getting the type of 'epilogue' (line 390)
                                    epilogue_13799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 54), 'epilogue')
                                    # Obtaining the member '__getitem__' of a type (line 390)
                                    getitem___13800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 54), epilogue_13799, '__getitem__')
                                    # Calling the subscript (__getitem__) to obtain the elements type (line 390)
                                    subscript_call_result_13801 = invoke(stypy.reporting.localization.Localization(__file__, 390, 54), getitem___13800, slice_13798)
                                    
                                    # Getting the type of 'self' (line 390)
                                    self_13802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 32), 'self')
                                    # Obtaining the member '_last' of a type (line 390)
                                    _last_13803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 32), self_13802, '_last')
                                    # Setting the type of the member 'epilogue' of a type (line 390)
                                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 32), _last_13803, 'epilogue', subscript_call_result_13801)
                                    # SSA join for if statement (line 388)
                                    module_type_store = module_type_store.join_ssa_context()
                                    


                                if more_types_in_union_13780:
                                    # SSA join for if statement (line 386)
                                    module_type_store = module_type_store.join_ssa_context()


                            
                        else:
                            
                            # Testing the type of an if condition (line 384)
                            if_condition_13773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 24), result_eq_13772)
                            # Assigning a type to the variable 'if_condition_13773' (line 384)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'if_condition_13773', if_condition_13773)
                            # SSA begins for if statement (line 384)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Attribute (line 385):
                            
                            # Assigning a Name to a Attribute (line 385):
                            # Getting the type of 'None' (line 385)
                            None_13774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 50), 'None')
                            # Getting the type of 'self' (line 385)
                            self_13775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 28), 'self')
                            # Obtaining the member '_last' of a type (line 385)
                            _last_13776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 28), self_13775, '_last')
                            # Setting the type of the member 'epilogue' of a type (line 385)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 28), _last_13776, 'epilogue', None_13774)
                            # SSA branch for the else part of an if statement (line 384)
                            module_type_store.open_ssa_branch('else')
                            
                            # Type idiom detected: calculating its left and rigth part (line 386)
                            # Getting the type of 'epilogue' (line 386)
                            epilogue_13777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 29), 'epilogue')
                            # Getting the type of 'None' (line 386)
                            None_13778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 45), 'None')
                            
                            (may_be_13779, more_types_in_union_13780) = may_not_be_none(epilogue_13777, None_13778)

                            if may_be_13779:

                                if more_types_in_union_13780:
                                    # Runtime conditional SSA (line 386)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                                else:
                                    module_type_store = module_type_store

                                
                                # Assigning a Call to a Name (line 387):
                                
                                # Assigning a Call to a Name (line 387):
                                
                                # Call to search(...): (line 387)
                                # Processing the call arguments (line 387)
                                # Getting the type of 'epilogue' (line 387)
                                epilogue_13783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 50), 'epilogue', False)
                                # Processing the call keyword arguments (line 387)
                                kwargs_13784 = {}
                                # Getting the type of 'NLCRE_eol' (line 387)
                                NLCRE_eol_13781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 33), 'NLCRE_eol', False)
                                # Obtaining the member 'search' of a type (line 387)
                                search_13782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 33), NLCRE_eol_13781, 'search')
                                # Calling search(args, kwargs) (line 387)
                                search_call_result_13785 = invoke(stypy.reporting.localization.Localization(__file__, 387, 33), search_13782, *[epilogue_13783], **kwargs_13784)
                                
                                # Assigning a type to the variable 'mo' (line 387)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 28), 'mo', search_call_result_13785)
                                # Getting the type of 'mo' (line 388)
                                mo_13786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 31), 'mo')
                                # Testing if the type of an if condition is none (line 388)

                                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 388, 28), mo_13786):
                                    pass
                                else:
                                    
                                    # Testing the type of an if condition (line 388)
                                    if_condition_13787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 28), mo_13786)
                                    # Assigning a type to the variable 'if_condition_13787' (line 388)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 28), 'if_condition_13787', if_condition_13787)
                                    # SSA begins for if statement (line 388)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                    
                                    # Assigning a Call to a Name (line 389):
                                    
                                    # Assigning a Call to a Name (line 389):
                                    
                                    # Call to len(...): (line 389)
                                    # Processing the call arguments (line 389)
                                    
                                    # Call to group(...): (line 389)
                                    # Processing the call arguments (line 389)
                                    int_13791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 51), 'int')
                                    # Processing the call keyword arguments (line 389)
                                    kwargs_13792 = {}
                                    # Getting the type of 'mo' (line 389)
                                    mo_13789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 42), 'mo', False)
                                    # Obtaining the member 'group' of a type (line 389)
                                    group_13790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 42), mo_13789, 'group')
                                    # Calling group(args, kwargs) (line 389)
                                    group_call_result_13793 = invoke(stypy.reporting.localization.Localization(__file__, 389, 42), group_13790, *[int_13791], **kwargs_13792)
                                    
                                    # Processing the call keyword arguments (line 389)
                                    kwargs_13794 = {}
                                    # Getting the type of 'len' (line 389)
                                    len_13788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 38), 'len', False)
                                    # Calling len(args, kwargs) (line 389)
                                    len_call_result_13795 = invoke(stypy.reporting.localization.Localization(__file__, 389, 38), len_13788, *[group_call_result_13793], **kwargs_13794)
                                    
                                    # Assigning a type to the variable 'end' (line 389)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 32), 'end', len_call_result_13795)
                                    
                                    # Assigning a Subscript to a Attribute (line 390):
                                    
                                    # Assigning a Subscript to a Attribute (line 390):
                                    
                                    # Obtaining the type of the subscript
                                    
                                    # Getting the type of 'end' (line 390)
                                    end_13796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 65), 'end')
                                    # Applying the 'usub' unary operator (line 390)
                                    result___neg___13797 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 64), 'usub', end_13796)
                                    
                                    slice_13798 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 390, 54), None, result___neg___13797, None)
                                    # Getting the type of 'epilogue' (line 390)
                                    epilogue_13799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 54), 'epilogue')
                                    # Obtaining the member '__getitem__' of a type (line 390)
                                    getitem___13800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 54), epilogue_13799, '__getitem__')
                                    # Calling the subscript (__getitem__) to obtain the elements type (line 390)
                                    subscript_call_result_13801 = invoke(stypy.reporting.localization.Localization(__file__, 390, 54), getitem___13800, slice_13798)
                                    
                                    # Getting the type of 'self' (line 390)
                                    self_13802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 32), 'self')
                                    # Obtaining the member '_last' of a type (line 390)
                                    _last_13803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 32), self_13802, '_last')
                                    # Setting the type of the member 'epilogue' of a type (line 390)
                                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 32), _last_13803, 'epilogue', subscript_call_result_13801)
                                    # SSA join for if statement (line 388)
                                    module_type_store = module_type_store.join_ssa_context()
                                    


                                if more_types_in_union_13780:
                                    # SSA join for if statement (line 386)
                                    module_type_store = module_type_store.join_ssa_context()


                            
                            # SSA join for if statement (line 384)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA branch for the else part of an if statement (line 382)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a Call to a Name (line 392):
                        
                        # Assigning a Call to a Name (line 392):
                        
                        # Call to get_payload(...): (line 392)
                        # Processing the call keyword arguments (line 392)
                        kwargs_13807 = {}
                        # Getting the type of 'self' (line 392)
                        self_13804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 34), 'self', False)
                        # Obtaining the member '_last' of a type (line 392)
                        _last_13805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 34), self_13804, '_last')
                        # Obtaining the member 'get_payload' of a type (line 392)
                        get_payload_13806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 34), _last_13805, 'get_payload')
                        # Calling get_payload(args, kwargs) (line 392)
                        get_payload_call_result_13808 = invoke(stypy.reporting.localization.Localization(__file__, 392, 34), get_payload_13806, *[], **kwargs_13807)
                        
                        # Assigning a type to the variable 'payload' (line 392)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 24), 'payload', get_payload_call_result_13808)
                        
                        # Type idiom detected: calculating its left and rigth part (line 393)
                        # Getting the type of 'basestring' (line 393)
                        basestring_13809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 47), 'basestring')
                        # Getting the type of 'payload' (line 393)
                        payload_13810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 38), 'payload')
                        
                        (may_be_13811, more_types_in_union_13812) = may_be_subtype(basestring_13809, payload_13810)

                        if may_be_13811:

                            if more_types_in_union_13812:
                                # Runtime conditional SSA (line 393)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                            else:
                                module_type_store = module_type_store

                            # Assigning a type to the variable 'payload' (line 393)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 24), 'payload', remove_not_subtype_from_union(payload_13810, basestring))
                            
                            # Assigning a Call to a Name (line 394):
                            
                            # Assigning a Call to a Name (line 394):
                            
                            # Call to search(...): (line 394)
                            # Processing the call arguments (line 394)
                            # Getting the type of 'payload' (line 394)
                            payload_13815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 50), 'payload', False)
                            # Processing the call keyword arguments (line 394)
                            kwargs_13816 = {}
                            # Getting the type of 'NLCRE_eol' (line 394)
                            NLCRE_eol_13813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 33), 'NLCRE_eol', False)
                            # Obtaining the member 'search' of a type (line 394)
                            search_13814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 33), NLCRE_eol_13813, 'search')
                            # Calling search(args, kwargs) (line 394)
                            search_call_result_13817 = invoke(stypy.reporting.localization.Localization(__file__, 394, 33), search_13814, *[payload_13815], **kwargs_13816)
                            
                            # Assigning a type to the variable 'mo' (line 394)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 28), 'mo', search_call_result_13817)
                            # Getting the type of 'mo' (line 395)
                            mo_13818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 31), 'mo')
                            # Testing if the type of an if condition is none (line 395)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 395, 28), mo_13818):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 395)
                                if_condition_13819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 28), mo_13818)
                                # Assigning a type to the variable 'if_condition_13819' (line 395)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 28), 'if_condition_13819', if_condition_13819)
                                # SSA begins for if statement (line 395)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Assigning a Subscript to a Name (line 396):
                                
                                # Assigning a Subscript to a Name (line 396):
                                
                                # Obtaining the type of the subscript
                                
                                
                                # Call to len(...): (line 396)
                                # Processing the call arguments (line 396)
                                
                                # Call to group(...): (line 396)
                                # Processing the call arguments (line 396)
                                int_13823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 65), 'int')
                                # Processing the call keyword arguments (line 396)
                                kwargs_13824 = {}
                                # Getting the type of 'mo' (line 396)
                                mo_13821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 56), 'mo', False)
                                # Obtaining the member 'group' of a type (line 396)
                                group_13822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 56), mo_13821, 'group')
                                # Calling group(args, kwargs) (line 396)
                                group_call_result_13825 = invoke(stypy.reporting.localization.Localization(__file__, 396, 56), group_13822, *[int_13823], **kwargs_13824)
                                
                                # Processing the call keyword arguments (line 396)
                                kwargs_13826 = {}
                                # Getting the type of 'len' (line 396)
                                len_13820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 52), 'len', False)
                                # Calling len(args, kwargs) (line 396)
                                len_call_result_13827 = invoke(stypy.reporting.localization.Localization(__file__, 396, 52), len_13820, *[group_call_result_13825], **kwargs_13826)
                                
                                # Applying the 'usub' unary operator (line 396)
                                result___neg___13828 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 51), 'usub', len_call_result_13827)
                                
                                slice_13829 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 396, 42), None, result___neg___13828, None)
                                # Getting the type of 'payload' (line 396)
                                payload_13830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 42), 'payload')
                                # Obtaining the member '__getitem__' of a type (line 396)
                                getitem___13831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 42), payload_13830, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 396)
                                subscript_call_result_13832 = invoke(stypy.reporting.localization.Localization(__file__, 396, 42), getitem___13831, slice_13829)
                                
                                # Assigning a type to the variable 'payload' (line 396)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 32), 'payload', subscript_call_result_13832)
                                
                                # Call to set_payload(...): (line 397)
                                # Processing the call arguments (line 397)
                                # Getting the type of 'payload' (line 397)
                                payload_13836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 55), 'payload', False)
                                # Processing the call keyword arguments (line 397)
                                kwargs_13837 = {}
                                # Getting the type of 'self' (line 397)
                                self_13833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 32), 'self', False)
                                # Obtaining the member '_last' of a type (line 397)
                                _last_13834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 32), self_13833, '_last')
                                # Obtaining the member 'set_payload' of a type (line 397)
                                set_payload_13835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 32), _last_13834, 'set_payload')
                                # Calling set_payload(args, kwargs) (line 397)
                                set_payload_call_result_13838 = invoke(stypy.reporting.localization.Localization(__file__, 397, 32), set_payload_13835, *[payload_13836], **kwargs_13837)
                                
                                # SSA join for if statement (line 395)
                                module_type_store = module_type_store.join_ssa_context()
                                


                            if more_types_in_union_13812:
                                # SSA join for if statement (line 393)
                                module_type_store = module_type_store.join_ssa_context()


                        
                        # SSA join for if statement (line 382)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Call to pop_eof_matcher(...): (line 398)
                    # Processing the call keyword arguments (line 398)
                    kwargs_13842 = {}
                    # Getting the type of 'self' (line 398)
                    self_13839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 20), 'self', False)
                    # Obtaining the member '_input' of a type (line 398)
                    _input_13840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 20), self_13839, '_input')
                    # Obtaining the member 'pop_eof_matcher' of a type (line 398)
                    pop_eof_matcher_13841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 20), _input_13840, 'pop_eof_matcher')
                    # Calling pop_eof_matcher(args, kwargs) (line 398)
                    pop_eof_matcher_call_result_13843 = invoke(stypy.reporting.localization.Localization(__file__, 398, 20), pop_eof_matcher_13841, *[], **kwargs_13842)
                    
                    
                    # Call to _pop_message(...): (line 399)
                    # Processing the call keyword arguments (line 399)
                    kwargs_13846 = {}
                    # Getting the type of 'self' (line 399)
                    self_13844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 20), 'self', False)
                    # Obtaining the member '_pop_message' of a type (line 399)
                    _pop_message_13845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 20), self_13844, '_pop_message')
                    # Calling _pop_message(args, kwargs) (line 399)
                    _pop_message_call_result_13847 = invoke(stypy.reporting.localization.Localization(__file__, 399, 20), _pop_message_13845, *[], **kwargs_13846)
                    
                    
                    # Assigning a Attribute to a Attribute (line 402):
                    
                    # Assigning a Attribute to a Attribute (line 402):
                    # Getting the type of 'self' (line 402)
                    self_13848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 33), 'self')
                    # Obtaining the member '_cur' of a type (line 402)
                    _cur_13849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 33), self_13848, '_cur')
                    # Getting the type of 'self' (line 402)
                    self_13850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'self')
                    # Setting the type of the member '_last' of a type (line 402)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 20), self_13850, '_last', _cur_13849)
                    # SSA branch for the else part of an if statement (line 336)
                    module_type_store.open_ssa_branch('else')
                    # Evaluating assert statement condition
                    # Getting the type of 'capturing_preamble' (line 405)
                    capturing_preamble_13851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 27), 'capturing_preamble')
                    assert_13852 = capturing_preamble_13851
                    # Assigning a type to the variable 'assert_13852' (line 405)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 20), 'assert_13852', capturing_preamble_13851)
                    
                    # Call to append(...): (line 406)
                    # Processing the call arguments (line 406)
                    # Getting the type of 'line' (line 406)
                    line_13855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 36), 'line', False)
                    # Processing the call keyword arguments (line 406)
                    kwargs_13856 = {}
                    # Getting the type of 'preamble' (line 406)
                    preamble_13853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'preamble', False)
                    # Obtaining the member 'append' of a type (line 406)
                    append_13854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 20), preamble_13853, 'append')
                    # Calling append(args, kwargs) (line 406)
                    append_call_result_13857 = invoke(stypy.reporting.localization.Localization(__file__, 406, 20), append_13854, *[line_13855], **kwargs_13856)
                    
                    # SSA join for if statement (line 336)
                    module_type_store = module_type_store.join_ssa_context()
                    


            
            # Getting the type of 'capturing_preamble' (line 411)
            capturing_preamble_13858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 15), 'capturing_preamble')
            # Testing if the type of an if condition is none (line 411)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 411, 12), capturing_preamble_13858):
                pass
            else:
                
                # Testing the type of an if condition (line 411)
                if_condition_13859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 12), capturing_preamble_13858)
                # Assigning a type to the variable 'if_condition_13859' (line 411)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'if_condition_13859', if_condition_13859)
                # SSA begins for if statement (line 411)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 412)
                # Processing the call arguments (line 412)
                
                # Call to StartBoundaryNotFoundDefect(...): (line 412)
                # Processing the call keyword arguments (line 412)
                kwargs_13866 = {}
                # Getting the type of 'errors' (line 412)
                errors_13864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 41), 'errors', False)
                # Obtaining the member 'StartBoundaryNotFoundDefect' of a type (line 412)
                StartBoundaryNotFoundDefect_13865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 41), errors_13864, 'StartBoundaryNotFoundDefect')
                # Calling StartBoundaryNotFoundDefect(args, kwargs) (line 412)
                StartBoundaryNotFoundDefect_call_result_13867 = invoke(stypy.reporting.localization.Localization(__file__, 412, 41), StartBoundaryNotFoundDefect_13865, *[], **kwargs_13866)
                
                # Processing the call keyword arguments (line 412)
                kwargs_13868 = {}
                # Getting the type of 'self' (line 412)
                self_13860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 16), 'self', False)
                # Obtaining the member '_cur' of a type (line 412)
                _cur_13861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 16), self_13860, '_cur')
                # Obtaining the member 'defects' of a type (line 412)
                defects_13862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 16), _cur_13861, 'defects')
                # Obtaining the member 'append' of a type (line 412)
                append_13863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 16), defects_13862, 'append')
                # Calling append(args, kwargs) (line 412)
                append_call_result_13869 = invoke(stypy.reporting.localization.Localization(__file__, 412, 16), append_13863, *[StartBoundaryNotFoundDefect_call_result_13867], **kwargs_13868)
                
                
                # Call to set_payload(...): (line 413)
                # Processing the call arguments (line 413)
                
                # Call to join(...): (line 413)
                # Processing the call arguments (line 413)
                # Getting the type of 'preamble' (line 413)
                preamble_13875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 55), 'preamble', False)
                # Processing the call keyword arguments (line 413)
                kwargs_13876 = {}
                # Getting the type of 'EMPTYSTRING' (line 413)
                EMPTYSTRING_13873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 38), 'EMPTYSTRING', False)
                # Obtaining the member 'join' of a type (line 413)
                join_13874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 38), EMPTYSTRING_13873, 'join')
                # Calling join(args, kwargs) (line 413)
                join_call_result_13877 = invoke(stypy.reporting.localization.Localization(__file__, 413, 38), join_13874, *[preamble_13875], **kwargs_13876)
                
                # Processing the call keyword arguments (line 413)
                kwargs_13878 = {}
                # Getting the type of 'self' (line 413)
                self_13870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 16), 'self', False)
                # Obtaining the member '_cur' of a type (line 413)
                _cur_13871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 16), self_13870, '_cur')
                # Obtaining the member 'set_payload' of a type (line 413)
                set_payload_13872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 16), _cur_13871, 'set_payload')
                # Calling set_payload(args, kwargs) (line 413)
                set_payload_call_result_13879 = invoke(stypy.reporting.localization.Localization(__file__, 413, 16), set_payload_13872, *[join_call_result_13877], **kwargs_13878)
                
                
                # Assigning a List to a Name (line 414):
                
                # Assigning a List to a Name (line 414):
                
                # Obtaining an instance of the builtin type 'list' (line 414)
                list_13880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 27), 'list')
                # Adding type elements to the builtin type 'list' instance (line 414)
                
                # Assigning a type to the variable 'epilogue' (line 414)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 16), 'epilogue', list_13880)
                
                # Getting the type of 'self' (line 415)
                self_13881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 28), 'self')
                # Obtaining the member '_input' of a type (line 415)
                _input_13882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 28), self_13881, '_input')
                # Assigning a type to the variable '_input_13882' (line 415)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), '_input_13882', _input_13882)
                # Testing if the for loop is going to be iterated (line 415)
                # Testing the type of a for loop iterable (line 415)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 415, 16), _input_13882)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 415, 16), _input_13882):
                    # Getting the type of the for loop variable (line 415)
                    for_loop_var_13883 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 415, 16), _input_13882)
                    # Assigning a type to the variable 'line' (line 415)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'line', for_loop_var_13883)
                    # SSA begins for a for statement (line 415)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'line' (line 416)
                    line_13884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 23), 'line')
                    # Getting the type of 'NeedMoreData' (line 416)
                    NeedMoreData_13885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 31), 'NeedMoreData')
                    # Applying the binary operator 'is' (line 416)
                    result_is__13886 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 23), 'is', line_13884, NeedMoreData_13885)
                    
                    # Testing if the type of an if condition is none (line 416)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 416, 20), result_is__13886):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 416)
                        if_condition_13887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 20), result_is__13886)
                        # Assigning a type to the variable 'if_condition_13887' (line 416)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 20), 'if_condition_13887', if_condition_13887)
                        # SSA begins for if statement (line 416)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Creating a generator
                        # Getting the type of 'NeedMoreData' (line 417)
                        NeedMoreData_13888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 30), 'NeedMoreData')
                        GeneratorType_13889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 24), 'GeneratorType')
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 24), GeneratorType_13889, NeedMoreData_13888)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 24), 'stypy_return_type', GeneratorType_13889)
                        # SSA join for if statement (line 416)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a Call to a Attribute (line 419):
                
                # Assigning a Call to a Attribute (line 419):
                
                # Call to join(...): (line 419)
                # Processing the call arguments (line 419)
                # Getting the type of 'epilogue' (line 419)
                epilogue_13892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 54), 'epilogue', False)
                # Processing the call keyword arguments (line 419)
                kwargs_13893 = {}
                # Getting the type of 'EMPTYSTRING' (line 419)
                EMPTYSTRING_13890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 37), 'EMPTYSTRING', False)
                # Obtaining the member 'join' of a type (line 419)
                join_13891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 37), EMPTYSTRING_13890, 'join')
                # Calling join(args, kwargs) (line 419)
                join_call_result_13894 = invoke(stypy.reporting.localization.Localization(__file__, 419, 37), join_13891, *[epilogue_13892], **kwargs_13893)
                
                # Getting the type of 'self' (line 419)
                self_13895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'self')
                # Obtaining the member '_cur' of a type (line 419)
                _cur_13896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 16), self_13895, '_cur')
                # Setting the type of the member 'epilogue' of a type (line 419)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 16), _cur_13896, 'epilogue', join_call_result_13894)
                # Assigning a type to the variable 'stypy_return_type' (line 420)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'stypy_return_type', types.NoneType)
                # SSA join for if statement (line 411)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'linesep' (line 423)
            linesep_13897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'linesep')
            # Testing if the type of an if condition is none (line 423)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 423, 12), linesep_13897):
                
                # Assigning a List to a Name (line 426):
                
                # Assigning a List to a Name (line 426):
                
                # Obtaining an instance of the builtin type 'list' (line 426)
                list_13901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 27), 'list')
                # Adding type elements to the builtin type 'list' instance (line 426)
                
                # Assigning a type to the variable 'epilogue' (line 426)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 16), 'epilogue', list_13901)
            else:
                
                # Testing the type of an if condition (line 423)
                if_condition_13898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 12), linesep_13897)
                # Assigning a type to the variable 'if_condition_13898' (line 423)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'if_condition_13898', if_condition_13898)
                # SSA begins for if statement (line 423)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a List to a Name (line 424):
                
                # Assigning a List to a Name (line 424):
                
                # Obtaining an instance of the builtin type 'list' (line 424)
                list_13899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 27), 'list')
                # Adding type elements to the builtin type 'list' instance (line 424)
                # Adding element type (line 424)
                str_13900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 28), 'str', '')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 27), list_13899, str_13900)
                
                # Assigning a type to the variable 'epilogue' (line 424)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 16), 'epilogue', list_13899)
                # SSA branch for the else part of an if statement (line 423)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a List to a Name (line 426):
                
                # Assigning a List to a Name (line 426):
                
                # Obtaining an instance of the builtin type 'list' (line 426)
                list_13901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 27), 'list')
                # Adding type elements to the builtin type 'list' instance (line 426)
                
                # Assigning a type to the variable 'epilogue' (line 426)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 16), 'epilogue', list_13901)
                # SSA join for if statement (line 423)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'self' (line 427)
            self_13902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 24), 'self')
            # Obtaining the member '_input' of a type (line 427)
            _input_13903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 24), self_13902, '_input')
            # Assigning a type to the variable '_input_13903' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), '_input_13903', _input_13903)
            # Testing if the for loop is going to be iterated (line 427)
            # Testing the type of a for loop iterable (line 427)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 427, 12), _input_13903)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 427, 12), _input_13903):
                # Getting the type of the for loop variable (line 427)
                for_loop_var_13904 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 427, 12), _input_13903)
                # Assigning a type to the variable 'line' (line 427)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'line', for_loop_var_13904)
                # SSA begins for a for statement (line 427)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'line' (line 428)
                line_13905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 19), 'line')
                # Getting the type of 'NeedMoreData' (line 428)
                NeedMoreData_13906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 27), 'NeedMoreData')
                # Applying the binary operator 'is' (line 428)
                result_is__13907 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 19), 'is', line_13905, NeedMoreData_13906)
                
                # Testing if the type of an if condition is none (line 428)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 428, 16), result_is__13907):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 428)
                    if_condition_13908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 16), result_is__13907)
                    # Assigning a type to the variable 'if_condition_13908' (line 428)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 16), 'if_condition_13908', if_condition_13908)
                    # SSA begins for if statement (line 428)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Creating a generator
                    # Getting the type of 'NeedMoreData' (line 429)
                    NeedMoreData_13909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 26), 'NeedMoreData')
                    GeneratorType_13910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 20), 'GeneratorType')
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 20), GeneratorType_13910, NeedMoreData_13909)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'stypy_return_type', GeneratorType_13910)
                    # SSA join for if statement (line 428)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to append(...): (line 431)
                # Processing the call arguments (line 431)
                # Getting the type of 'line' (line 431)
                line_13913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 32), 'line', False)
                # Processing the call keyword arguments (line 431)
                kwargs_13914 = {}
                # Getting the type of 'epilogue' (line 431)
                epilogue_13911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'epilogue', False)
                # Obtaining the member 'append' of a type (line 431)
                append_13912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 16), epilogue_13911, 'append')
                # Calling append(args, kwargs) (line 431)
                append_call_result_13915 = invoke(stypy.reporting.localization.Localization(__file__, 431, 16), append_13912, *[line_13913], **kwargs_13914)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'epilogue' (line 435)
            epilogue_13916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'epilogue')
            # Testing if the type of an if condition is none (line 435)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 435, 12), epilogue_13916):
                pass
            else:
                
                # Testing the type of an if condition (line 435)
                if_condition_13917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 12), epilogue_13916)
                # Assigning a type to the variable 'if_condition_13917' (line 435)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'if_condition_13917', if_condition_13917)
                # SSA begins for if statement (line 435)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 436):
                
                # Assigning a Subscript to a Name (line 436):
                
                # Obtaining the type of the subscript
                int_13918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 37), 'int')
                # Getting the type of 'epilogue' (line 436)
                epilogue_13919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 28), 'epilogue')
                # Obtaining the member '__getitem__' of a type (line 436)
                getitem___13920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 28), epilogue_13919, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 436)
                subscript_call_result_13921 = invoke(stypy.reporting.localization.Localization(__file__, 436, 28), getitem___13920, int_13918)
                
                # Assigning a type to the variable 'firstline' (line 436)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'firstline', subscript_call_result_13921)
                
                # Assigning a Call to a Name (line 437):
                
                # Assigning a Call to a Name (line 437):
                
                # Call to match(...): (line 437)
                # Processing the call arguments (line 437)
                # Getting the type of 'firstline' (line 437)
                firstline_13924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 40), 'firstline', False)
                # Processing the call keyword arguments (line 437)
                kwargs_13925 = {}
                # Getting the type of 'NLCRE_bol' (line 437)
                NLCRE_bol_13922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 24), 'NLCRE_bol', False)
                # Obtaining the member 'match' of a type (line 437)
                match_13923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 24), NLCRE_bol_13922, 'match')
                # Calling match(args, kwargs) (line 437)
                match_call_result_13926 = invoke(stypy.reporting.localization.Localization(__file__, 437, 24), match_13923, *[firstline_13924], **kwargs_13925)
                
                # Assigning a type to the variable 'bolmo' (line 437)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), 'bolmo', match_call_result_13926)
                # Getting the type of 'bolmo' (line 438)
                bolmo_13927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'bolmo')
                # Testing if the type of an if condition is none (line 438)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 438, 16), bolmo_13927):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 438)
                    if_condition_13928 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 16), bolmo_13927)
                    # Assigning a type to the variable 'if_condition_13928' (line 438)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'if_condition_13928', if_condition_13928)
                    # SSA begins for if statement (line 438)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Subscript to a Subscript (line 439):
                    
                    # Assigning a Subscript to a Subscript (line 439):
                    
                    # Obtaining the type of the subscript
                    
                    # Call to len(...): (line 439)
                    # Processing the call arguments (line 439)
                    
                    # Call to group(...): (line 439)
                    # Processing the call arguments (line 439)
                    int_13932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 60), 'int')
                    # Processing the call keyword arguments (line 439)
                    kwargs_13933 = {}
                    # Getting the type of 'bolmo' (line 439)
                    bolmo_13930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 48), 'bolmo', False)
                    # Obtaining the member 'group' of a type (line 439)
                    group_13931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 48), bolmo_13930, 'group')
                    # Calling group(args, kwargs) (line 439)
                    group_call_result_13934 = invoke(stypy.reporting.localization.Localization(__file__, 439, 48), group_13931, *[int_13932], **kwargs_13933)
                    
                    # Processing the call keyword arguments (line 439)
                    kwargs_13935 = {}
                    # Getting the type of 'len' (line 439)
                    len_13929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 44), 'len', False)
                    # Calling len(args, kwargs) (line 439)
                    len_call_result_13936 = invoke(stypy.reporting.localization.Localization(__file__, 439, 44), len_13929, *[group_call_result_13934], **kwargs_13935)
                    
                    slice_13937 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 439, 34), len_call_result_13936, None, None)
                    # Getting the type of 'firstline' (line 439)
                    firstline_13938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 34), 'firstline')
                    # Obtaining the member '__getitem__' of a type (line 439)
                    getitem___13939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 34), firstline_13938, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
                    subscript_call_result_13940 = invoke(stypy.reporting.localization.Localization(__file__, 439, 34), getitem___13939, slice_13937)
                    
                    # Getting the type of 'epilogue' (line 439)
                    epilogue_13941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'epilogue')
                    int_13942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 29), 'int')
                    # Storing an element on a container (line 439)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 20), epilogue_13941, (int_13942, subscript_call_result_13940))
                    # SSA join for if statement (line 438)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 435)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Attribute (line 440):
            
            # Assigning a Call to a Attribute (line 440):
            
            # Call to join(...): (line 440)
            # Processing the call arguments (line 440)
            # Getting the type of 'epilogue' (line 440)
            epilogue_13945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 50), 'epilogue', False)
            # Processing the call keyword arguments (line 440)
            kwargs_13946 = {}
            # Getting the type of 'EMPTYSTRING' (line 440)
            EMPTYSTRING_13943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 33), 'EMPTYSTRING', False)
            # Obtaining the member 'join' of a type (line 440)
            join_13944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 33), EMPTYSTRING_13943, 'join')
            # Calling join(args, kwargs) (line 440)
            join_call_result_13947 = invoke(stypy.reporting.localization.Localization(__file__, 440, 33), join_13944, *[epilogue_13945], **kwargs_13946)
            
            # Getting the type of 'self' (line 440)
            self_13948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'self')
            # Obtaining the member '_cur' of a type (line 440)
            _cur_13949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 12), self_13948, '_cur')
            # Setting the type of the member 'epilogue' of a type (line 440)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 12), _cur_13949, 'epilogue', join_call_result_13947)
            # Assigning a type to the variable 'stypy_return_type' (line 441)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 301)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a List to a Name (line 444):
        
        # Assigning a List to a Name (line 444):
        
        # Obtaining an instance of the builtin type 'list' (line 444)
        list_13950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 444)
        
        # Assigning a type to the variable 'lines' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'lines', list_13950)
        
        # Getting the type of 'self' (line 445)
        self_13951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'self')
        # Obtaining the member '_input' of a type (line 445)
        _input_13952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 20), self_13951, '_input')
        # Assigning a type to the variable '_input_13952' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), '_input_13952', _input_13952)
        # Testing if the for loop is going to be iterated (line 445)
        # Testing the type of a for loop iterable (line 445)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 445, 8), _input_13952)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 445, 8), _input_13952):
            # Getting the type of the for loop variable (line 445)
            for_loop_var_13953 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 445, 8), _input_13952)
            # Assigning a type to the variable 'line' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'line', for_loop_var_13953)
            # SSA begins for a for statement (line 445)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'line' (line 446)
            line_13954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 15), 'line')
            # Getting the type of 'NeedMoreData' (line 446)
            NeedMoreData_13955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 23), 'NeedMoreData')
            # Applying the binary operator 'is' (line 446)
            result_is__13956 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 15), 'is', line_13954, NeedMoreData_13955)
            
            # Testing if the type of an if condition is none (line 446)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 446, 12), result_is__13956):
                pass
            else:
                
                # Testing the type of an if condition (line 446)
                if_condition_13957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 446, 12), result_is__13956)
                # Assigning a type to the variable 'if_condition_13957' (line 446)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'if_condition_13957', if_condition_13957)
                # SSA begins for if statement (line 446)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Creating a generator
                # Getting the type of 'NeedMoreData' (line 447)
                NeedMoreData_13958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 22), 'NeedMoreData')
                GeneratorType_13959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 16), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 16), GeneratorType_13959, NeedMoreData_13958)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 16), 'stypy_return_type', GeneratorType_13959)
                # SSA join for if statement (line 446)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to append(...): (line 449)
            # Processing the call arguments (line 449)
            # Getting the type of 'line' (line 449)
            line_13962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 25), 'line', False)
            # Processing the call keyword arguments (line 449)
            kwargs_13963 = {}
            # Getting the type of 'lines' (line 449)
            lines_13960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'lines', False)
            # Obtaining the member 'append' of a type (line 449)
            append_13961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 12), lines_13960, 'append')
            # Calling append(args, kwargs) (line 449)
            append_call_result_13964 = invoke(stypy.reporting.localization.Localization(__file__, 449, 12), append_13961, *[line_13962], **kwargs_13963)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to set_payload(...): (line 450)
        # Processing the call arguments (line 450)
        
        # Call to join(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'lines' (line 450)
        lines_13970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 47), 'lines', False)
        # Processing the call keyword arguments (line 450)
        kwargs_13971 = {}
        # Getting the type of 'EMPTYSTRING' (line 450)
        EMPTYSTRING_13968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 30), 'EMPTYSTRING', False)
        # Obtaining the member 'join' of a type (line 450)
        join_13969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 30), EMPTYSTRING_13968, 'join')
        # Calling join(args, kwargs) (line 450)
        join_call_result_13972 = invoke(stypy.reporting.localization.Localization(__file__, 450, 30), join_13969, *[lines_13970], **kwargs_13971)
        
        # Processing the call keyword arguments (line 450)
        kwargs_13973 = {}
        # Getting the type of 'self' (line 450)
        self_13965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'self', False)
        # Obtaining the member '_cur' of a type (line 450)
        _cur_13966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), self_13965, '_cur')
        # Obtaining the member 'set_payload' of a type (line 450)
        set_payload_13967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), _cur_13966, 'set_payload')
        # Calling set_payload(args, kwargs) (line 450)
        set_payload_call_result_13974 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), set_payload_13967, *[join_call_result_13972], **kwargs_13973)
        
        
        # ################# End of '_parsegen(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parsegen' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_13975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parsegen'
        return stypy_return_type_13975


    @norecursion
    def _parse_headers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parse_headers'
        module_type_store = module_type_store.open_function_context('_parse_headers', 452, 4, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FeedParser._parse_headers.__dict__.__setitem__('stypy_localization', localization)
        FeedParser._parse_headers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FeedParser._parse_headers.__dict__.__setitem__('stypy_type_store', module_type_store)
        FeedParser._parse_headers.__dict__.__setitem__('stypy_function_name', 'FeedParser._parse_headers')
        FeedParser._parse_headers.__dict__.__setitem__('stypy_param_names_list', ['lines'])
        FeedParser._parse_headers.__dict__.__setitem__('stypy_varargs_param_name', None)
        FeedParser._parse_headers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FeedParser._parse_headers.__dict__.__setitem__('stypy_call_defaults', defaults)
        FeedParser._parse_headers.__dict__.__setitem__('stypy_call_varargs', varargs)
        FeedParser._parse_headers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FeedParser._parse_headers.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FeedParser._parse_headers', ['lines'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse_headers', localization, ['lines'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse_headers(...)' code ##################

        
        # Assigning a Str to a Name (line 454):
        
        # Assigning a Str to a Name (line 454):
        str_13976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 21), 'str', '')
        # Assigning a type to the variable 'lastheader' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'lastheader', str_13976)
        
        # Assigning a List to a Name (line 455):
        
        # Assigning a List to a Name (line 455):
        
        # Obtaining an instance of the builtin type 'list' (line 455)
        list_13977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 455)
        
        # Assigning a type to the variable 'lastvalue' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'lastvalue', list_13977)
        
        
        # Call to enumerate(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'lines' (line 456)
        lines_13979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 38), 'lines', False)
        # Processing the call keyword arguments (line 456)
        kwargs_13980 = {}
        # Getting the type of 'enumerate' (line 456)
        enumerate_13978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 28), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 456)
        enumerate_call_result_13981 = invoke(stypy.reporting.localization.Localization(__file__, 456, 28), enumerate_13978, *[lines_13979], **kwargs_13980)
        
        # Assigning a type to the variable 'enumerate_call_result_13981' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'enumerate_call_result_13981', enumerate_call_result_13981)
        # Testing if the for loop is going to be iterated (line 456)
        # Testing the type of a for loop iterable (line 456)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 456, 8), enumerate_call_result_13981)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 456, 8), enumerate_call_result_13981):
            # Getting the type of the for loop variable (line 456)
            for_loop_var_13982 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 456, 8), enumerate_call_result_13981)
            # Assigning a type to the variable 'lineno' (line 456)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'lineno', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 8), for_loop_var_13982, 2, 0))
            # Assigning a type to the variable 'line' (line 456)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'line', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 8), for_loop_var_13982, 2, 1))
            # SSA begins for a for statement (line 456)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            int_13983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 20), 'int')
            # Getting the type of 'line' (line 458)
            line_13984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 15), 'line')
            # Obtaining the member '__getitem__' of a type (line 458)
            getitem___13985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 15), line_13984, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 458)
            subscript_call_result_13986 = invoke(stypy.reporting.localization.Localization(__file__, 458, 15), getitem___13985, int_13983)
            
            str_13987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 26), 'str', ' \t')
            # Applying the binary operator 'in' (line 458)
            result_contains_13988 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 15), 'in', subscript_call_result_13986, str_13987)
            
            # Testing if the type of an if condition is none (line 458)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 458, 12), result_contains_13988):
                pass
            else:
                
                # Testing the type of an if condition (line 458)
                if_condition_13989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 12), result_contains_13988)
                # Assigning a type to the variable 'if_condition_13989' (line 458)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'if_condition_13989', if_condition_13989)
                # SSA begins for if statement (line 458)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'lastheader' (line 459)
                lastheader_13990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 23), 'lastheader')
                # Applying the 'not' unary operator (line 459)
                result_not__13991 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 19), 'not', lastheader_13990)
                
                # Testing if the type of an if condition is none (line 459)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 459, 16), result_not__13991):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 459)
                    if_condition_13992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 16), result_not__13991)
                    # Assigning a type to the variable 'if_condition_13992' (line 459)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 'if_condition_13992', if_condition_13992)
                    # SSA begins for if statement (line 459)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 463):
                    
                    # Assigning a Call to a Name (line 463):
                    
                    # Call to FirstHeaderLineIsContinuationDefect(...): (line 463)
                    # Processing the call arguments (line 463)
                    # Getting the type of 'line' (line 463)
                    line_13995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 72), 'line', False)
                    # Processing the call keyword arguments (line 463)
                    kwargs_13996 = {}
                    # Getting the type of 'errors' (line 463)
                    errors_13993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 29), 'errors', False)
                    # Obtaining the member 'FirstHeaderLineIsContinuationDefect' of a type (line 463)
                    FirstHeaderLineIsContinuationDefect_13994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 29), errors_13993, 'FirstHeaderLineIsContinuationDefect')
                    # Calling FirstHeaderLineIsContinuationDefect(args, kwargs) (line 463)
                    FirstHeaderLineIsContinuationDefect_call_result_13997 = invoke(stypy.reporting.localization.Localization(__file__, 463, 29), FirstHeaderLineIsContinuationDefect_13994, *[line_13995], **kwargs_13996)
                    
                    # Assigning a type to the variable 'defect' (line 463)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 20), 'defect', FirstHeaderLineIsContinuationDefect_call_result_13997)
                    
                    # Call to append(...): (line 464)
                    # Processing the call arguments (line 464)
                    # Getting the type of 'defect' (line 464)
                    defect_14002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 45), 'defect', False)
                    # Processing the call keyword arguments (line 464)
                    kwargs_14003 = {}
                    # Getting the type of 'self' (line 464)
                    self_13998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 20), 'self', False)
                    # Obtaining the member '_cur' of a type (line 464)
                    _cur_13999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 20), self_13998, '_cur')
                    # Obtaining the member 'defects' of a type (line 464)
                    defects_14000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 20), _cur_13999, 'defects')
                    # Obtaining the member 'append' of a type (line 464)
                    append_14001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 20), defects_14000, 'append')
                    # Calling append(args, kwargs) (line 464)
                    append_call_result_14004 = invoke(stypy.reporting.localization.Localization(__file__, 464, 20), append_14001, *[defect_14002], **kwargs_14003)
                    
                    # SSA join for if statement (line 459)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to append(...): (line 466)
                # Processing the call arguments (line 466)
                # Getting the type of 'line' (line 466)
                line_14007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 33), 'line', False)
                # Processing the call keyword arguments (line 466)
                kwargs_14008 = {}
                # Getting the type of 'lastvalue' (line 466)
                lastvalue_14005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'lastvalue', False)
                # Obtaining the member 'append' of a type (line 466)
                append_14006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 16), lastvalue_14005, 'append')
                # Calling append(args, kwargs) (line 466)
                append_call_result_14009 = invoke(stypy.reporting.localization.Localization(__file__, 466, 16), append_14006, *[line_14007], **kwargs_14008)
                
                # SSA join for if statement (line 458)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'lastheader' (line 468)
            lastheader_14010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 15), 'lastheader')
            # Testing if the type of an if condition is none (line 468)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 468, 12), lastheader_14010):
                pass
            else:
                
                # Testing the type of an if condition (line 468)
                if_condition_14011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 12), lastheader_14010)
                # Assigning a type to the variable 'if_condition_14011' (line 468)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'if_condition_14011', if_condition_14011)
                # SSA begins for if statement (line 468)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 470):
                
                # Assigning a Call to a Name (line 470):
                
                # Call to rstrip(...): (line 470)
                # Processing the call arguments (line 470)
                str_14022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 63), 'str', '\r\n')
                # Processing the call keyword arguments (line 470)
                kwargs_14023 = {}
                
                # Obtaining the type of the subscript
                int_14012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 52), 'int')
                slice_14013 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 470, 23), None, int_14012, None)
                
                # Call to join(...): (line 470)
                # Processing the call arguments (line 470)
                # Getting the type of 'lastvalue' (line 470)
                lastvalue_14016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 40), 'lastvalue', False)
                # Processing the call keyword arguments (line 470)
                kwargs_14017 = {}
                # Getting the type of 'EMPTYSTRING' (line 470)
                EMPTYSTRING_14014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 23), 'EMPTYSTRING', False)
                # Obtaining the member 'join' of a type (line 470)
                join_14015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 23), EMPTYSTRING_14014, 'join')
                # Calling join(args, kwargs) (line 470)
                join_call_result_14018 = invoke(stypy.reporting.localization.Localization(__file__, 470, 23), join_14015, *[lastvalue_14016], **kwargs_14017)
                
                # Obtaining the member '__getitem__' of a type (line 470)
                getitem___14019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 23), join_call_result_14018, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 470)
                subscript_call_result_14020 = invoke(stypy.reporting.localization.Localization(__file__, 470, 23), getitem___14019, slice_14013)
                
                # Obtaining the member 'rstrip' of a type (line 470)
                rstrip_14021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 23), subscript_call_result_14020, 'rstrip')
                # Calling rstrip(args, kwargs) (line 470)
                rstrip_call_result_14024 = invoke(stypy.reporting.localization.Localization(__file__, 470, 23), rstrip_14021, *[str_14022], **kwargs_14023)
                
                # Assigning a type to the variable 'lhdr' (line 470)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 16), 'lhdr', rstrip_call_result_14024)
                
                # Assigning a Name to a Subscript (line 471):
                
                # Assigning a Name to a Subscript (line 471):
                # Getting the type of 'lhdr' (line 471)
                lhdr_14025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 40), 'lhdr')
                # Getting the type of 'self' (line 471)
                self_14026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'self')
                # Obtaining the member '_cur' of a type (line 471)
                _cur_14027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 16), self_14026, '_cur')
                # Getting the type of 'lastheader' (line 471)
                lastheader_14028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 26), 'lastheader')
                # Storing an element on a container (line 471)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 16), _cur_14027, (lastheader_14028, lhdr_14025))
                
                # Assigning a Tuple to a Tuple (line 472):
                
                # Assigning a Str to a Name (line 472):
                str_14029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 40), 'str', '')
                # Assigning a type to the variable 'tuple_assignment_12945' (line 472)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 16), 'tuple_assignment_12945', str_14029)
                
                # Assigning a List to a Name (line 472):
                
                # Obtaining an instance of the builtin type 'list' (line 472)
                list_14030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 44), 'list')
                # Adding type elements to the builtin type 'list' instance (line 472)
                
                # Assigning a type to the variable 'tuple_assignment_12946' (line 472)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 16), 'tuple_assignment_12946', list_14030)
                
                # Assigning a Name to a Name (line 472):
                # Getting the type of 'tuple_assignment_12945' (line 472)
                tuple_assignment_12945_14031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 16), 'tuple_assignment_12945')
                # Assigning a type to the variable 'lastheader' (line 472)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 16), 'lastheader', tuple_assignment_12945_14031)
                
                # Assigning a Name to a Name (line 472):
                # Getting the type of 'tuple_assignment_12946' (line 472)
                tuple_assignment_12946_14032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 16), 'tuple_assignment_12946')
                # Assigning a type to the variable 'lastvalue' (line 472)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 28), 'lastvalue', tuple_assignment_12946_14032)
                # SSA join for if statement (line 468)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to startswith(...): (line 474)
            # Processing the call arguments (line 474)
            str_14035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 31), 'str', 'From ')
            # Processing the call keyword arguments (line 474)
            kwargs_14036 = {}
            # Getting the type of 'line' (line 474)
            line_14033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 15), 'line', False)
            # Obtaining the member 'startswith' of a type (line 474)
            startswith_14034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 15), line_14033, 'startswith')
            # Calling startswith(args, kwargs) (line 474)
            startswith_call_result_14037 = invoke(stypy.reporting.localization.Localization(__file__, 474, 15), startswith_14034, *[str_14035], **kwargs_14036)
            
            # Testing if the type of an if condition is none (line 474)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 474, 12), startswith_call_result_14037):
                pass
            else:
                
                # Testing the type of an if condition (line 474)
                if_condition_14038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 12), startswith_call_result_14037)
                # Assigning a type to the variable 'if_condition_14038' (line 474)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 12), 'if_condition_14038', if_condition_14038)
                # SSA begins for if statement (line 474)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'lineno' (line 475)
                lineno_14039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'lineno')
                int_14040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 29), 'int')
                # Applying the binary operator '==' (line 475)
                result_eq_14041 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 19), '==', lineno_14039, int_14040)
                
                # Testing if the type of an if condition is none (line 475)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 475, 16), result_eq_14041):
                    
                    # Getting the type of 'lineno' (line 482)
                    lineno_14069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 21), 'lineno')
                    
                    # Call to len(...): (line 482)
                    # Processing the call arguments (line 482)
                    # Getting the type of 'lines' (line 482)
                    lines_14071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 35), 'lines', False)
                    # Processing the call keyword arguments (line 482)
                    kwargs_14072 = {}
                    # Getting the type of 'len' (line 482)
                    len_14070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 31), 'len', False)
                    # Calling len(args, kwargs) (line 482)
                    len_call_result_14073 = invoke(stypy.reporting.localization.Localization(__file__, 482, 31), len_14070, *[lines_14071], **kwargs_14072)
                    
                    int_14074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 44), 'int')
                    # Applying the binary operator '-' (line 482)
                    result_sub_14075 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 31), '-', len_call_result_14073, int_14074)
                    
                    # Applying the binary operator '==' (line 482)
                    result_eq_14076 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 21), '==', lineno_14069, result_sub_14075)
                    
                    # Testing if the type of an if condition is none (line 482)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 482, 21), result_eq_14076):
                        
                        # Assigning a Call to a Name (line 491):
                        
                        # Assigning a Call to a Name (line 491):
                        
                        # Call to MisplacedEnvelopeHeaderDefect(...): (line 491)
                        # Processing the call arguments (line 491)
                        # Getting the type of 'line' (line 491)
                        line_14086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 66), 'line', False)
                        # Processing the call keyword arguments (line 491)
                        kwargs_14087 = {}
                        # Getting the type of 'errors' (line 491)
                        errors_14084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 29), 'errors', False)
                        # Obtaining the member 'MisplacedEnvelopeHeaderDefect' of a type (line 491)
                        MisplacedEnvelopeHeaderDefect_14085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 29), errors_14084, 'MisplacedEnvelopeHeaderDefect')
                        # Calling MisplacedEnvelopeHeaderDefect(args, kwargs) (line 491)
                        MisplacedEnvelopeHeaderDefect_call_result_14088 = invoke(stypy.reporting.localization.Localization(__file__, 491, 29), MisplacedEnvelopeHeaderDefect_14085, *[line_14086], **kwargs_14087)
                        
                        # Assigning a type to the variable 'defect' (line 491)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 20), 'defect', MisplacedEnvelopeHeaderDefect_call_result_14088)
                        
                        # Call to append(...): (line 492)
                        # Processing the call arguments (line 492)
                        # Getting the type of 'defect' (line 492)
                        defect_14093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 45), 'defect', False)
                        # Processing the call keyword arguments (line 492)
                        kwargs_14094 = {}
                        # Getting the type of 'self' (line 492)
                        self_14089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 20), 'self', False)
                        # Obtaining the member '_cur' of a type (line 492)
                        _cur_14090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), self_14089, '_cur')
                        # Obtaining the member 'defects' of a type (line 492)
                        defects_14091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), _cur_14090, 'defects')
                        # Obtaining the member 'append' of a type (line 492)
                        append_14092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), defects_14091, 'append')
                        # Calling append(args, kwargs) (line 492)
                        append_call_result_14095 = invoke(stypy.reporting.localization.Localization(__file__, 492, 20), append_14092, *[defect_14093], **kwargs_14094)
                        
                    else:
                        
                        # Testing the type of an if condition (line 482)
                        if_condition_14077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 482, 21), result_eq_14076)
                        # Assigning a type to the variable 'if_condition_14077' (line 482)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 21), 'if_condition_14077', if_condition_14077)
                        # SSA begins for if statement (line 482)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to unreadline(...): (line 486)
                        # Processing the call arguments (line 486)
                        # Getting the type of 'line' (line 486)
                        line_14081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 43), 'line', False)
                        # Processing the call keyword arguments (line 486)
                        kwargs_14082 = {}
                        # Getting the type of 'self' (line 486)
                        self_14078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'self', False)
                        # Obtaining the member '_input' of a type (line 486)
                        _input_14079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 20), self_14078, '_input')
                        # Obtaining the member 'unreadline' of a type (line 486)
                        unreadline_14080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 20), _input_14079, 'unreadline')
                        # Calling unreadline(args, kwargs) (line 486)
                        unreadline_call_result_14083 = invoke(stypy.reporting.localization.Localization(__file__, 486, 20), unreadline_14080, *[line_14081], **kwargs_14082)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 487)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 20), 'stypy_return_type', types.NoneType)
                        # SSA branch for the else part of an if statement (line 482)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a Call to a Name (line 491):
                        
                        # Assigning a Call to a Name (line 491):
                        
                        # Call to MisplacedEnvelopeHeaderDefect(...): (line 491)
                        # Processing the call arguments (line 491)
                        # Getting the type of 'line' (line 491)
                        line_14086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 66), 'line', False)
                        # Processing the call keyword arguments (line 491)
                        kwargs_14087 = {}
                        # Getting the type of 'errors' (line 491)
                        errors_14084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 29), 'errors', False)
                        # Obtaining the member 'MisplacedEnvelopeHeaderDefect' of a type (line 491)
                        MisplacedEnvelopeHeaderDefect_14085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 29), errors_14084, 'MisplacedEnvelopeHeaderDefect')
                        # Calling MisplacedEnvelopeHeaderDefect(args, kwargs) (line 491)
                        MisplacedEnvelopeHeaderDefect_call_result_14088 = invoke(stypy.reporting.localization.Localization(__file__, 491, 29), MisplacedEnvelopeHeaderDefect_14085, *[line_14086], **kwargs_14087)
                        
                        # Assigning a type to the variable 'defect' (line 491)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 20), 'defect', MisplacedEnvelopeHeaderDefect_call_result_14088)
                        
                        # Call to append(...): (line 492)
                        # Processing the call arguments (line 492)
                        # Getting the type of 'defect' (line 492)
                        defect_14093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 45), 'defect', False)
                        # Processing the call keyword arguments (line 492)
                        kwargs_14094 = {}
                        # Getting the type of 'self' (line 492)
                        self_14089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 20), 'self', False)
                        # Obtaining the member '_cur' of a type (line 492)
                        _cur_14090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), self_14089, '_cur')
                        # Obtaining the member 'defects' of a type (line 492)
                        defects_14091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), _cur_14090, 'defects')
                        # Obtaining the member 'append' of a type (line 492)
                        append_14092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), defects_14091, 'append')
                        # Calling append(args, kwargs) (line 492)
                        append_call_result_14095 = invoke(stypy.reporting.localization.Localization(__file__, 492, 20), append_14092, *[defect_14093], **kwargs_14094)
                        
                        # SSA join for if statement (line 482)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 475)
                    if_condition_14042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 475, 16), result_eq_14041)
                    # Assigning a type to the variable 'if_condition_14042' (line 475)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 16), 'if_condition_14042', if_condition_14042)
                    # SSA begins for if statement (line 475)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 477):
                    
                    # Assigning a Call to a Name (line 477):
                    
                    # Call to search(...): (line 477)
                    # Processing the call arguments (line 477)
                    # Getting the type of 'line' (line 477)
                    line_14045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 42), 'line', False)
                    # Processing the call keyword arguments (line 477)
                    kwargs_14046 = {}
                    # Getting the type of 'NLCRE_eol' (line 477)
                    NLCRE_eol_14043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 25), 'NLCRE_eol', False)
                    # Obtaining the member 'search' of a type (line 477)
                    search_14044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 25), NLCRE_eol_14043, 'search')
                    # Calling search(args, kwargs) (line 477)
                    search_call_result_14047 = invoke(stypy.reporting.localization.Localization(__file__, 477, 25), search_14044, *[line_14045], **kwargs_14046)
                    
                    # Assigning a type to the variable 'mo' (line 477)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 20), 'mo', search_call_result_14047)
                    # Getting the type of 'mo' (line 478)
                    mo_14048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 23), 'mo')
                    # Testing if the type of an if condition is none (line 478)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 478, 20), mo_14048):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 478)
                        if_condition_14049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 478, 20), mo_14048)
                        # Assigning a type to the variable 'if_condition_14049' (line 478)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'if_condition_14049', if_condition_14049)
                        # SSA begins for if statement (line 478)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Subscript to a Name (line 479):
                        
                        # Assigning a Subscript to a Name (line 479):
                        
                        # Obtaining the type of the subscript
                        
                        
                        # Call to len(...): (line 479)
                        # Processing the call arguments (line 479)
                        
                        # Call to group(...): (line 479)
                        # Processing the call arguments (line 479)
                        int_14053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 51), 'int')
                        # Processing the call keyword arguments (line 479)
                        kwargs_14054 = {}
                        # Getting the type of 'mo' (line 479)
                        mo_14051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 42), 'mo', False)
                        # Obtaining the member 'group' of a type (line 479)
                        group_14052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 42), mo_14051, 'group')
                        # Calling group(args, kwargs) (line 479)
                        group_call_result_14055 = invoke(stypy.reporting.localization.Localization(__file__, 479, 42), group_14052, *[int_14053], **kwargs_14054)
                        
                        # Processing the call keyword arguments (line 479)
                        kwargs_14056 = {}
                        # Getting the type of 'len' (line 479)
                        len_14050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 38), 'len', False)
                        # Calling len(args, kwargs) (line 479)
                        len_call_result_14057 = invoke(stypy.reporting.localization.Localization(__file__, 479, 38), len_14050, *[group_call_result_14055], **kwargs_14056)
                        
                        # Applying the 'usub' unary operator (line 479)
                        result___neg___14058 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 37), 'usub', len_call_result_14057)
                        
                        slice_14059 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 479, 31), None, result___neg___14058, None)
                        # Getting the type of 'line' (line 479)
                        line_14060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 31), 'line')
                        # Obtaining the member '__getitem__' of a type (line 479)
                        getitem___14061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 31), line_14060, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 479)
                        subscript_call_result_14062 = invoke(stypy.reporting.localization.Localization(__file__, 479, 31), getitem___14061, slice_14059)
                        
                        # Assigning a type to the variable 'line' (line 479)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 24), 'line', subscript_call_result_14062)
                        # SSA join for if statement (line 478)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Call to set_unixfrom(...): (line 480)
                    # Processing the call arguments (line 480)
                    # Getting the type of 'line' (line 480)
                    line_14066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 43), 'line', False)
                    # Processing the call keyword arguments (line 480)
                    kwargs_14067 = {}
                    # Getting the type of 'self' (line 480)
                    self_14063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 20), 'self', False)
                    # Obtaining the member '_cur' of a type (line 480)
                    _cur_14064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 20), self_14063, '_cur')
                    # Obtaining the member 'set_unixfrom' of a type (line 480)
                    set_unixfrom_14065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 20), _cur_14064, 'set_unixfrom')
                    # Calling set_unixfrom(args, kwargs) (line 480)
                    set_unixfrom_call_result_14068 = invoke(stypy.reporting.localization.Localization(__file__, 480, 20), set_unixfrom_14065, *[line_14066], **kwargs_14067)
                    
                    # SSA branch for the else part of an if statement (line 475)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'lineno' (line 482)
                    lineno_14069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 21), 'lineno')
                    
                    # Call to len(...): (line 482)
                    # Processing the call arguments (line 482)
                    # Getting the type of 'lines' (line 482)
                    lines_14071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 35), 'lines', False)
                    # Processing the call keyword arguments (line 482)
                    kwargs_14072 = {}
                    # Getting the type of 'len' (line 482)
                    len_14070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 31), 'len', False)
                    # Calling len(args, kwargs) (line 482)
                    len_call_result_14073 = invoke(stypy.reporting.localization.Localization(__file__, 482, 31), len_14070, *[lines_14071], **kwargs_14072)
                    
                    int_14074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 44), 'int')
                    # Applying the binary operator '-' (line 482)
                    result_sub_14075 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 31), '-', len_call_result_14073, int_14074)
                    
                    # Applying the binary operator '==' (line 482)
                    result_eq_14076 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 21), '==', lineno_14069, result_sub_14075)
                    
                    # Testing if the type of an if condition is none (line 482)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 482, 21), result_eq_14076):
                        
                        # Assigning a Call to a Name (line 491):
                        
                        # Assigning a Call to a Name (line 491):
                        
                        # Call to MisplacedEnvelopeHeaderDefect(...): (line 491)
                        # Processing the call arguments (line 491)
                        # Getting the type of 'line' (line 491)
                        line_14086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 66), 'line', False)
                        # Processing the call keyword arguments (line 491)
                        kwargs_14087 = {}
                        # Getting the type of 'errors' (line 491)
                        errors_14084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 29), 'errors', False)
                        # Obtaining the member 'MisplacedEnvelopeHeaderDefect' of a type (line 491)
                        MisplacedEnvelopeHeaderDefect_14085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 29), errors_14084, 'MisplacedEnvelopeHeaderDefect')
                        # Calling MisplacedEnvelopeHeaderDefect(args, kwargs) (line 491)
                        MisplacedEnvelopeHeaderDefect_call_result_14088 = invoke(stypy.reporting.localization.Localization(__file__, 491, 29), MisplacedEnvelopeHeaderDefect_14085, *[line_14086], **kwargs_14087)
                        
                        # Assigning a type to the variable 'defect' (line 491)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 20), 'defect', MisplacedEnvelopeHeaderDefect_call_result_14088)
                        
                        # Call to append(...): (line 492)
                        # Processing the call arguments (line 492)
                        # Getting the type of 'defect' (line 492)
                        defect_14093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 45), 'defect', False)
                        # Processing the call keyword arguments (line 492)
                        kwargs_14094 = {}
                        # Getting the type of 'self' (line 492)
                        self_14089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 20), 'self', False)
                        # Obtaining the member '_cur' of a type (line 492)
                        _cur_14090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), self_14089, '_cur')
                        # Obtaining the member 'defects' of a type (line 492)
                        defects_14091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), _cur_14090, 'defects')
                        # Obtaining the member 'append' of a type (line 492)
                        append_14092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), defects_14091, 'append')
                        # Calling append(args, kwargs) (line 492)
                        append_call_result_14095 = invoke(stypy.reporting.localization.Localization(__file__, 492, 20), append_14092, *[defect_14093], **kwargs_14094)
                        
                    else:
                        
                        # Testing the type of an if condition (line 482)
                        if_condition_14077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 482, 21), result_eq_14076)
                        # Assigning a type to the variable 'if_condition_14077' (line 482)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 21), 'if_condition_14077', if_condition_14077)
                        # SSA begins for if statement (line 482)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to unreadline(...): (line 486)
                        # Processing the call arguments (line 486)
                        # Getting the type of 'line' (line 486)
                        line_14081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 43), 'line', False)
                        # Processing the call keyword arguments (line 486)
                        kwargs_14082 = {}
                        # Getting the type of 'self' (line 486)
                        self_14078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 20), 'self', False)
                        # Obtaining the member '_input' of a type (line 486)
                        _input_14079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 20), self_14078, '_input')
                        # Obtaining the member 'unreadline' of a type (line 486)
                        unreadline_14080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 20), _input_14079, 'unreadline')
                        # Calling unreadline(args, kwargs) (line 486)
                        unreadline_call_result_14083 = invoke(stypy.reporting.localization.Localization(__file__, 486, 20), unreadline_14080, *[line_14081], **kwargs_14082)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 487)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 20), 'stypy_return_type', types.NoneType)
                        # SSA branch for the else part of an if statement (line 482)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a Call to a Name (line 491):
                        
                        # Assigning a Call to a Name (line 491):
                        
                        # Call to MisplacedEnvelopeHeaderDefect(...): (line 491)
                        # Processing the call arguments (line 491)
                        # Getting the type of 'line' (line 491)
                        line_14086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 66), 'line', False)
                        # Processing the call keyword arguments (line 491)
                        kwargs_14087 = {}
                        # Getting the type of 'errors' (line 491)
                        errors_14084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 29), 'errors', False)
                        # Obtaining the member 'MisplacedEnvelopeHeaderDefect' of a type (line 491)
                        MisplacedEnvelopeHeaderDefect_14085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 29), errors_14084, 'MisplacedEnvelopeHeaderDefect')
                        # Calling MisplacedEnvelopeHeaderDefect(args, kwargs) (line 491)
                        MisplacedEnvelopeHeaderDefect_call_result_14088 = invoke(stypy.reporting.localization.Localization(__file__, 491, 29), MisplacedEnvelopeHeaderDefect_14085, *[line_14086], **kwargs_14087)
                        
                        # Assigning a type to the variable 'defect' (line 491)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 20), 'defect', MisplacedEnvelopeHeaderDefect_call_result_14088)
                        
                        # Call to append(...): (line 492)
                        # Processing the call arguments (line 492)
                        # Getting the type of 'defect' (line 492)
                        defect_14093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 45), 'defect', False)
                        # Processing the call keyword arguments (line 492)
                        kwargs_14094 = {}
                        # Getting the type of 'self' (line 492)
                        self_14089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 20), 'self', False)
                        # Obtaining the member '_cur' of a type (line 492)
                        _cur_14090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), self_14089, '_cur')
                        # Obtaining the member 'defects' of a type (line 492)
                        defects_14091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), _cur_14090, 'defects')
                        # Obtaining the member 'append' of a type (line 492)
                        append_14092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 20), defects_14091, 'append')
                        # Calling append(args, kwargs) (line 492)
                        append_call_result_14095 = invoke(stypy.reporting.localization.Localization(__file__, 492, 20), append_14092, *[defect_14093], **kwargs_14094)
                        
                        # SSA join for if statement (line 482)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 475)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 474)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 495):
            
            # Assigning a Call to a Name (line 495):
            
            # Call to find(...): (line 495)
            # Processing the call arguments (line 495)
            str_14098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 26), 'str', ':')
            # Processing the call keyword arguments (line 495)
            kwargs_14099 = {}
            # Getting the type of 'line' (line 495)
            line_14096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'line', False)
            # Obtaining the member 'find' of a type (line 495)
            find_14097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 16), line_14096, 'find')
            # Calling find(args, kwargs) (line 495)
            find_call_result_14100 = invoke(stypy.reporting.localization.Localization(__file__, 495, 16), find_14097, *[str_14098], **kwargs_14099)
            
            # Assigning a type to the variable 'i' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'i', find_call_result_14100)
            
            # Getting the type of 'i' (line 496)
            i_14101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'i')
            int_14102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 19), 'int')
            # Applying the binary operator '<' (line 496)
            result_lt_14103 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 15), '<', i_14101, int_14102)
            
            # Testing if the type of an if condition is none (line 496)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 496, 12), result_lt_14103):
                pass
            else:
                
                # Testing the type of an if condition (line 496)
                if_condition_14104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 496, 12), result_lt_14103)
                # Assigning a type to the variable 'if_condition_14104' (line 496)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'if_condition_14104', if_condition_14104)
                # SSA begins for if statement (line 496)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 497):
                
                # Assigning a Call to a Name (line 497):
                
                # Call to MalformedHeaderDefect(...): (line 497)
                # Processing the call arguments (line 497)
                # Getting the type of 'line' (line 497)
                line_14107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 54), 'line', False)
                # Processing the call keyword arguments (line 497)
                kwargs_14108 = {}
                # Getting the type of 'errors' (line 497)
                errors_14105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 25), 'errors', False)
                # Obtaining the member 'MalformedHeaderDefect' of a type (line 497)
                MalformedHeaderDefect_14106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 25), errors_14105, 'MalformedHeaderDefect')
                # Calling MalformedHeaderDefect(args, kwargs) (line 497)
                MalformedHeaderDefect_call_result_14109 = invoke(stypy.reporting.localization.Localization(__file__, 497, 25), MalformedHeaderDefect_14106, *[line_14107], **kwargs_14108)
                
                # Assigning a type to the variable 'defect' (line 497)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'defect', MalformedHeaderDefect_call_result_14109)
                
                # Call to append(...): (line 498)
                # Processing the call arguments (line 498)
                # Getting the type of 'defect' (line 498)
                defect_14114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 41), 'defect', False)
                # Processing the call keyword arguments (line 498)
                kwargs_14115 = {}
                # Getting the type of 'self' (line 498)
                self_14110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'self', False)
                # Obtaining the member '_cur' of a type (line 498)
                _cur_14111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 16), self_14110, '_cur')
                # Obtaining the member 'defects' of a type (line 498)
                defects_14112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 16), _cur_14111, 'defects')
                # Obtaining the member 'append' of a type (line 498)
                append_14113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 16), defects_14112, 'append')
                # Calling append(args, kwargs) (line 498)
                append_call_result_14116 = invoke(stypy.reporting.localization.Localization(__file__, 498, 16), append_14113, *[defect_14114], **kwargs_14115)
                
                # SSA join for if statement (line 496)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Subscript to a Name (line 500):
            
            # Assigning a Subscript to a Name (line 500):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 500)
            i_14117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 31), 'i')
            slice_14118 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 500, 25), None, i_14117, None)
            # Getting the type of 'line' (line 500)
            line_14119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 25), 'line')
            # Obtaining the member '__getitem__' of a type (line 500)
            getitem___14120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 25), line_14119, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 500)
            subscript_call_result_14121 = invoke(stypy.reporting.localization.Localization(__file__, 500, 25), getitem___14120, slice_14118)
            
            # Assigning a type to the variable 'lastheader' (line 500)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'lastheader', subscript_call_result_14121)
            
            # Assigning a List to a Name (line 501):
            
            # Assigning a List to a Name (line 501):
            
            # Obtaining an instance of the builtin type 'list' (line 501)
            list_14122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 501)
            # Adding element type (line 501)
            
            # Call to lstrip(...): (line 501)
            # Processing the call keyword arguments (line 501)
            kwargs_14131 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 501)
            i_14123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 30), 'i', False)
            int_14124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 32), 'int')
            # Applying the binary operator '+' (line 501)
            result_add_14125 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 30), '+', i_14123, int_14124)
            
            slice_14126 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 501, 25), result_add_14125, None, None)
            # Getting the type of 'line' (line 501)
            line_14127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 25), 'line', False)
            # Obtaining the member '__getitem__' of a type (line 501)
            getitem___14128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 25), line_14127, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 501)
            subscript_call_result_14129 = invoke(stypy.reporting.localization.Localization(__file__, 501, 25), getitem___14128, slice_14126)
            
            # Obtaining the member 'lstrip' of a type (line 501)
            lstrip_14130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 25), subscript_call_result_14129, 'lstrip')
            # Calling lstrip(args, kwargs) (line 501)
            lstrip_call_result_14132 = invoke(stypy.reporting.localization.Localization(__file__, 501, 25), lstrip_14130, *[], **kwargs_14131)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 24), list_14122, lstrip_call_result_14132)
            
            # Assigning a type to the variable 'lastvalue' (line 501)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'lastvalue', list_14122)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'lastheader' (line 503)
        lastheader_14133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 11), 'lastheader')
        # Testing if the type of an if condition is none (line 503)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 503, 8), lastheader_14133):
            pass
        else:
            
            # Testing the type of an if condition (line 503)
            if_condition_14134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 503, 8), lastheader_14133)
            # Assigning a type to the variable 'if_condition_14134' (line 503)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'if_condition_14134', if_condition_14134)
            # SSA begins for if statement (line 503)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 505):
            
            # Assigning a Call to a Subscript (line 505):
            
            # Call to rstrip(...): (line 505)
            # Processing the call arguments (line 505)
            str_14141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 71), 'str', '\r\n')
            # Processing the call keyword arguments (line 505)
            kwargs_14142 = {}
            
            # Call to join(...): (line 505)
            # Processing the call arguments (line 505)
            # Getting the type of 'lastvalue' (line 505)
            lastvalue_14137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 53), 'lastvalue', False)
            # Processing the call keyword arguments (line 505)
            kwargs_14138 = {}
            # Getting the type of 'EMPTYSTRING' (line 505)
            EMPTYSTRING_14135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 36), 'EMPTYSTRING', False)
            # Obtaining the member 'join' of a type (line 505)
            join_14136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 36), EMPTYSTRING_14135, 'join')
            # Calling join(args, kwargs) (line 505)
            join_call_result_14139 = invoke(stypy.reporting.localization.Localization(__file__, 505, 36), join_14136, *[lastvalue_14137], **kwargs_14138)
            
            # Obtaining the member 'rstrip' of a type (line 505)
            rstrip_14140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 36), join_call_result_14139, 'rstrip')
            # Calling rstrip(args, kwargs) (line 505)
            rstrip_call_result_14143 = invoke(stypy.reporting.localization.Localization(__file__, 505, 36), rstrip_14140, *[str_14141], **kwargs_14142)
            
            # Getting the type of 'self' (line 505)
            self_14144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'self')
            # Obtaining the member '_cur' of a type (line 505)
            _cur_14145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 12), self_14144, '_cur')
            # Getting the type of 'lastheader' (line 505)
            lastheader_14146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 22), 'lastheader')
            # Storing an element on a container (line 505)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 12), _cur_14145, (lastheader_14146, rstrip_call_result_14143))
            # SSA join for if statement (line 503)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '_parse_headers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse_headers' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_14147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14147)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse_headers'
        return stypy_return_type_14147


# Assigning a type to the variable 'FeedParser' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'FeedParser', FeedParser)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
