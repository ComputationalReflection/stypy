
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Basic message object for the email package object model.'''
6: 
7: __all__ = ['Message']
8: 
9: import re
10: import uu
11: import binascii
12: import warnings
13: from cStringIO import StringIO
14: 
15: # Intrapackage imports
16: import email.charset
17: from email import utils
18: from email import errors
19: 
20: SEMISPACE = '; '
21: 
22: # Regular expression that matches `special' characters in parameters, the
23: # existence of which force quoting of the parameter value.
24: tspecials = re.compile(r'[ \(\)<>@,;:\\"/\[\]\?=]')
25: 
26: 
27: # Helper functions
28: def _splitparam(param):
29:     # Split header parameters.  BAW: this may be too simple.  It isn't
30:     # strictly RFC 2045 (section 5.1) compliant, but it catches most headers
31:     # found in the wild.  We may eventually need a full fledged parser
32:     # eventually.
33:     a, sep, b = param.partition(';')
34:     if not sep:
35:         return a.strip(), None
36:     return a.strip(), b.strip()
37: 
38: def _formatparam(param, value=None, quote=True):
39:     '''Convenience function to format and return a key=value pair.
40: 
41:     This will quote the value if needed or if quote is true.  If value is a
42:     three tuple (charset, language, value), it will be encoded according
43:     to RFC2231 rules.
44:     '''
45:     if value is not None and len(value) > 0:
46:         # A tuple is used for RFC 2231 encoded parameter values where items
47:         # are (charset, language, value).  charset is a string, not a Charset
48:         # instance.
49:         if isinstance(value, tuple):
50:             # Encode as per RFC 2231
51:             param += '*'
52:             value = utils.encode_rfc2231(value[2], value[0], value[1])
53:         # BAW: Please check this.  I think that if quote is set it should
54:         # force quoting even if not necessary.
55:         if quote or tspecials.search(value):
56:             return '%s="%s"' % (param, utils.quote(value))
57:         else:
58:             return '%s=%s' % (param, value)
59:     else:
60:         return param
61: 
62: def _parseparam(s):
63:     plist = []
64:     while s[:1] == ';':
65:         s = s[1:]
66:         end = s.find(';')
67:         while end > 0 and (s.count('"', 0, end) - s.count('\\"', 0, end)) % 2:
68:             end = s.find(';', end + 1)
69:         if end < 0:
70:             end = len(s)
71:         f = s[:end]
72:         if '=' in f:
73:             i = f.index('=')
74:             f = f[:i].strip().lower() + '=' + f[i+1:].strip()
75:         plist.append(f.strip())
76:         s = s[end:]
77:     return plist
78: 
79: 
80: def _unquotevalue(value):
81:     # This is different than utils.collapse_rfc2231_value() because it doesn't
82:     # try to convert the value to a unicode.  Message.get_param() and
83:     # Message.get_params() are both currently defined to return the tuple in
84:     # the face of RFC 2231 parameters.
85:     if isinstance(value, tuple):
86:         return value[0], value[1], utils.unquote(value[2])
87:     else:
88:         return utils.unquote(value)
89: 
90: 
91: 
92: class Message:
93:     '''Basic message object.
94: 
95:     A message object is defined as something that has a bunch of RFC 2822
96:     headers and a payload.  It may optionally have an envelope header
97:     (a.k.a. Unix-From or From_ header).  If the message is a container (i.e. a
98:     multipart or a message/rfc822), then the payload is a list of Message
99:     objects, otherwise it is a string.
100: 
101:     Message objects implement part of the `mapping' interface, which assumes
102:     there is exactly one occurrence of the header per message.  Some headers
103:     do in fact appear multiple times (e.g. Received) and for those headers,
104:     you must use the explicit API to set or get all the headers.  Not all of
105:     the mapping methods are implemented.
106:     '''
107:     def __init__(self):
108:         self._headers = []
109:         self._unixfrom = None
110:         self._payload = None
111:         self._charset = None
112:         # Defaults for multipart messages
113:         self.preamble = self.epilogue = None
114:         self.defects = []
115:         # Default content type
116:         self._default_type = 'text/plain'
117: 
118:     def __str__(self):
119:         '''Return the entire formatted message as a string.
120:         This includes the headers, body, and envelope header.
121:         '''
122:         return self.as_string(unixfrom=True)
123: 
124:     def as_string(self, unixfrom=False):
125:         '''Return the entire formatted message as a string.
126:         Optional `unixfrom' when True, means include the Unix From_ envelope
127:         header.
128: 
129:         This is a convenience method and may not generate the message exactly
130:         as you intend because by default it mangles lines that begin with
131:         "From ".  For more flexibility, use the flatten() method of a
132:         Generator instance.
133:         '''
134:         from email.generator import Generator
135:         fp = StringIO()
136:         g = Generator(fp)
137:         g.flatten(self, unixfrom=unixfrom)
138:         return fp.getvalue()
139: 
140:     def is_multipart(self):
141:         '''Return True if the message consists of multiple parts.'''
142:         return isinstance(self._payload, list)
143: 
144:     #
145:     # Unix From_ line
146:     #
147:     def set_unixfrom(self, unixfrom):
148:         self._unixfrom = unixfrom
149: 
150:     def get_unixfrom(self):
151:         return self._unixfrom
152: 
153:     #
154:     # Payload manipulation.
155:     #
156:     def attach(self, payload):
157:         '''Add the given payload to the current payload.
158: 
159:         The current payload will always be a list of objects after this method
160:         is called.  If you want to set the payload to a scalar object, use
161:         set_payload() instead.
162:         '''
163:         if self._payload is None:
164:             self._payload = [payload]
165:         else:
166:             self._payload.append(payload)
167: 
168:     def get_payload(self, i=None, decode=False):
169:         '''Return a reference to the payload.
170: 
171:         The payload will either be a list object or a string.  If you mutate
172:         the list object, you modify the message's payload in place.  Optional
173:         i returns that index into the payload.
174: 
175:         Optional decode is a flag indicating whether the payload should be
176:         decoded or not, according to the Content-Transfer-Encoding header
177:         (default is False).
178: 
179:         When True and the message is not a multipart, the payload will be
180:         decoded if this header's value is `quoted-printable' or `base64'.  If
181:         some other encoding is used, or the header is missing, or if the
182:         payload has bogus data (i.e. bogus base64 or uuencoded data), the
183:         payload is returned as-is.
184: 
185:         If the message is a multipart and the decode flag is True, then None
186:         is returned.
187:         '''
188:         if i is None:
189:             payload = self._payload
190:         elif not isinstance(self._payload, list):
191:             raise TypeError('Expected list, got %s' % type(self._payload))
192:         else:
193:             payload = self._payload[i]
194:         if decode:
195:             if self.is_multipart():
196:                 return None
197:             cte = self.get('content-transfer-encoding', '').lower()
198:             if cte == 'quoted-printable':
199:                 return utils._qdecode(payload)
200:             elif cte == 'base64':
201:                 try:
202:                     return utils._bdecode(payload)
203:                 except binascii.Error:
204:                     # Incorrect padding
205:                     return payload
206:             elif cte in ('x-uuencode', 'uuencode', 'uue', 'x-uue'):
207:                 sfp = StringIO()
208:                 try:
209:                     uu.decode(StringIO(payload+'\n'), sfp, quiet=True)
210:                     payload = sfp.getvalue()
211:                 except uu.Error:
212:                     # Some decoding problem
213:                     return payload
214:         # Everything else, including encodings with 8bit or 7bit are returned
215:         # unchanged.
216:         return payload
217: 
218:     def set_payload(self, payload, charset=None):
219:         '''Set the payload to the given value.
220: 
221:         Optional charset sets the message's default character set.  See
222:         set_charset() for details.
223:         '''
224:         self._payload = payload
225:         if charset is not None:
226:             self.set_charset(charset)
227: 
228:     def set_charset(self, charset):
229:         '''Set the charset of the payload to a given character set.
230: 
231:         charset can be a Charset instance, a string naming a character set, or
232:         None.  If it is a string it will be converted to a Charset instance.
233:         If charset is None, the charset parameter will be removed from the
234:         Content-Type field.  Anything else will generate a TypeError.
235: 
236:         The message will be assumed to be of type text/* encoded with
237:         charset.input_charset.  It will be converted to charset.output_charset
238:         and encoded properly, if needed, when generating the plain text
239:         representation of the message.  MIME headers (MIME-Version,
240:         Content-Type, Content-Transfer-Encoding) will be added as needed.
241: 
242:         '''
243:         if charset is None:
244:             self.del_param('charset')
245:             self._charset = None
246:             return
247:         if isinstance(charset, basestring):
248:             charset = email.charset.Charset(charset)
249:         if not isinstance(charset, email.charset.Charset):
250:             raise TypeError(charset)
251:         # BAW: should we accept strings that can serve as arguments to the
252:         # Charset constructor?
253:         self._charset = charset
254:         if 'MIME-Version' not in self:
255:             self.add_header('MIME-Version', '1.0')
256:         if 'Content-Type' not in self:
257:             self.add_header('Content-Type', 'text/plain',
258:                             charset=charset.get_output_charset())
259:         else:
260:             self.set_param('charset', charset.get_output_charset())
261:         if isinstance(self._payload, unicode):
262:             self._payload = self._payload.encode(charset.output_charset)
263:         if str(charset) != charset.get_output_charset():
264:             self._payload = charset.body_encode(self._payload)
265:         if 'Content-Transfer-Encoding' not in self:
266:             cte = charset.get_body_encoding()
267:             try:
268:                 cte(self)
269:             except TypeError:
270:                 self._payload = charset.body_encode(self._payload)
271:                 self.add_header('Content-Transfer-Encoding', cte)
272: 
273:     def get_charset(self):
274:         '''Return the Charset instance associated with the message's payload.
275:         '''
276:         return self._charset
277: 
278:     #
279:     # MAPPING INTERFACE (partial)
280:     #
281:     def __len__(self):
282:         '''Return the total number of headers, including duplicates.'''
283:         return len(self._headers)
284: 
285:     def __getitem__(self, name):
286:         '''Get a header value.
287: 
288:         Return None if the header is missing instead of raising an exception.
289: 
290:         Note that if the header appeared multiple times, exactly which
291:         occurrence gets returned is undefined.  Use get_all() to get all
292:         the values matching a header field name.
293:         '''
294:         return self.get(name)
295: 
296:     def __setitem__(self, name, val):
297:         '''Set the value of a header.
298: 
299:         Note: this does not overwrite an existing header with the same field
300:         name.  Use __delitem__() first to delete any existing headers.
301:         '''
302:         self._headers.append((name, val))
303: 
304:     def __delitem__(self, name):
305:         '''Delete all occurrences of a header, if present.
306: 
307:         Does not raise an exception if the header is missing.
308:         '''
309:         name = name.lower()
310:         newheaders = []
311:         for k, v in self._headers:
312:             if k.lower() != name:
313:                 newheaders.append((k, v))
314:         self._headers = newheaders
315: 
316:     def __contains__(self, name):
317:         return name.lower() in [k.lower() for k, v in self._headers]
318: 
319:     def has_key(self, name):
320:         '''Return true if the message contains the header.'''
321:         missing = object()
322:         return self.get(name, missing) is not missing
323: 
324:     def keys(self):
325:         '''Return a list of all the message's header field names.
326: 
327:         These will be sorted in the order they appeared in the original
328:         message, or were added to the message, and may contain duplicates.
329:         Any fields deleted and re-inserted are always appended to the header
330:         list.
331:         '''
332:         return [k for k, v in self._headers]
333: 
334:     def values(self):
335:         '''Return a list of all the message's header values.
336: 
337:         These will be sorted in the order they appeared in the original
338:         message, or were added to the message, and may contain duplicates.
339:         Any fields deleted and re-inserted are always appended to the header
340:         list.
341:         '''
342:         return [v for k, v in self._headers]
343: 
344:     def items(self):
345:         '''Get all the message's header fields and values.
346: 
347:         These will be sorted in the order they appeared in the original
348:         message, or were added to the message, and may contain duplicates.
349:         Any fields deleted and re-inserted are always appended to the header
350:         list.
351:         '''
352:         return self._headers[:]
353: 
354:     def get(self, name, failobj=None):
355:         '''Get a header value.
356: 
357:         Like __getitem__() but return failobj instead of None when the field
358:         is missing.
359:         '''
360:         name = name.lower()
361:         for k, v in self._headers:
362:             if k.lower() == name:
363:                 return v
364:         return failobj
365: 
366:     #
367:     # Additional useful stuff
368:     #
369: 
370:     def get_all(self, name, failobj=None):
371:         '''Return a list of all the values for the named field.
372: 
373:         These will be sorted in the order they appeared in the original
374:         message, and may contain duplicates.  Any fields deleted and
375:         re-inserted are always appended to the header list.
376: 
377:         If no such fields exist, failobj is returned (defaults to None).
378:         '''
379:         values = []
380:         name = name.lower()
381:         for k, v in self._headers:
382:             if k.lower() == name:
383:                 values.append(v)
384:         if not values:
385:             return failobj
386:         return values
387: 
388:     def add_header(self, _name, _value, **_params):
389:         '''Extended header setting.
390: 
391:         name is the header field to add.  keyword arguments can be used to set
392:         additional parameters for the header field, with underscores converted
393:         to dashes.  Normally the parameter will be added as key="value" unless
394:         value is None, in which case only the key will be added.  If a
395:         parameter value contains non-ASCII characters it must be specified as a
396:         three-tuple of (charset, language, value), in which case it will be
397:         encoded according to RFC2231 rules.
398: 
399:         Example:
400: 
401:         msg.add_header('content-disposition', 'attachment', filename='bud.gif')
402:         '''
403:         parts = []
404:         for k, v in _params.items():
405:             if v is None:
406:                 parts.append(k.replace('_', '-'))
407:             else:
408:                 parts.append(_formatparam(k.replace('_', '-'), v))
409:         if _value is not None:
410:             parts.insert(0, _value)
411:         self._headers.append((_name, SEMISPACE.join(parts)))
412: 
413:     def replace_header(self, _name, _value):
414:         '''Replace a header.
415: 
416:         Replace the first matching header found in the message, retaining
417:         header order and case.  If no matching header was found, a KeyError is
418:         raised.
419:         '''
420:         _name = _name.lower()
421:         for i, (k, v) in zip(range(len(self._headers)), self._headers):
422:             if k.lower() == _name:
423:                 self._headers[i] = (k, _value)
424:                 break
425:         else:
426:             raise KeyError(_name)
427: 
428:     #
429:     # Use these three methods instead of the three above.
430:     #
431: 
432:     def get_content_type(self):
433:         '''Return the message's content type.
434: 
435:         The returned string is coerced to lower case of the form
436:         `maintype/subtype'.  If there was no Content-Type header in the
437:         message, the default type as given by get_default_type() will be
438:         returned.  Since according to RFC 2045, messages always have a default
439:         type this will always return a value.
440: 
441:         RFC 2045 defines a message's default type to be text/plain unless it
442:         appears inside a multipart/digest container, in which case it would be
443:         message/rfc822.
444:         '''
445:         missing = object()
446:         value = self.get('content-type', missing)
447:         if value is missing:
448:             # This should have no parameters
449:             return self.get_default_type()
450:         ctype = _splitparam(value)[0].lower()
451:         # RFC 2045, section 5.2 says if its invalid, use text/plain
452:         if ctype.count('/') != 1:
453:             return 'text/plain'
454:         return ctype
455: 
456:     def get_content_maintype(self):
457:         '''Return the message's main content type.
458: 
459:         This is the `maintype' part of the string returned by
460:         get_content_type().
461:         '''
462:         ctype = self.get_content_type()
463:         return ctype.split('/')[0]
464: 
465:     def get_content_subtype(self):
466:         '''Returns the message's sub-content type.
467: 
468:         This is the `subtype' part of the string returned by
469:         get_content_type().
470:         '''
471:         ctype = self.get_content_type()
472:         return ctype.split('/')[1]
473: 
474:     def get_default_type(self):
475:         '''Return the `default' content type.
476: 
477:         Most messages have a default content type of text/plain, except for
478:         messages that are subparts of multipart/digest containers.  Such
479:         subparts have a default content type of message/rfc822.
480:         '''
481:         return self._default_type
482: 
483:     def set_default_type(self, ctype):
484:         '''Set the `default' content type.
485: 
486:         ctype should be either "text/plain" or "message/rfc822", although this
487:         is not enforced.  The default content type is not stored in the
488:         Content-Type header.
489:         '''
490:         self._default_type = ctype
491: 
492:     def _get_params_preserve(self, failobj, header):
493:         # Like get_params() but preserves the quoting of values.  BAW:
494:         # should this be part of the public interface?
495:         missing = object()
496:         value = self.get(header, missing)
497:         if value is missing:
498:             return failobj
499:         params = []
500:         for p in _parseparam(';' + value):
501:             try:
502:                 name, val = p.split('=', 1)
503:                 name = name.strip()
504:                 val = val.strip()
505:             except ValueError:
506:                 # Must have been a bare attribute
507:                 name = p.strip()
508:                 val = ''
509:             params.append((name, val))
510:         params = utils.decode_params(params)
511:         return params
512: 
513:     def get_params(self, failobj=None, header='content-type', unquote=True):
514:         '''Return the message's Content-Type parameters, as a list.
515: 
516:         The elements of the returned list are 2-tuples of key/value pairs, as
517:         split on the `=' sign.  The left hand side of the `=' is the key,
518:         while the right hand side is the value.  If there is no `=' sign in
519:         the parameter the value is the empty string.  The value is as
520:         described in the get_param() method.
521: 
522:         Optional failobj is the object to return if there is no Content-Type
523:         header.  Optional header is the header to search instead of
524:         Content-Type.  If unquote is True, the value is unquoted.
525:         '''
526:         missing = object()
527:         params = self._get_params_preserve(missing, header)
528:         if params is missing:
529:             return failobj
530:         if unquote:
531:             return [(k, _unquotevalue(v)) for k, v in params]
532:         else:
533:             return params
534: 
535:     def get_param(self, param, failobj=None, header='content-type',
536:                   unquote=True):
537:         '''Return the parameter value if found in the Content-Type header.
538: 
539:         Optional failobj is the object to return if there is no Content-Type
540:         header, or the Content-Type header has no such parameter.  Optional
541:         header is the header to search instead of Content-Type.
542: 
543:         Parameter keys are always compared case insensitively.  The return
544:         value can either be a string, or a 3-tuple if the parameter was RFC
545:         2231 encoded.  When it's a 3-tuple, the elements of the value are of
546:         the form (CHARSET, LANGUAGE, VALUE).  Note that both CHARSET and
547:         LANGUAGE can be None, in which case you should consider VALUE to be
548:         encoded in the us-ascii charset.  You can usually ignore LANGUAGE.
549: 
550:         Your application should be prepared to deal with 3-tuple return
551:         values, and can convert the parameter to a Unicode string like so:
552: 
553:             param = msg.get_param('foo')
554:             if isinstance(param, tuple):
555:                 param = unicode(param[2], param[0] or 'us-ascii')
556: 
557:         In any case, the parameter value (either the returned string, or the
558:         VALUE item in the 3-tuple) is always unquoted, unless unquote is set
559:         to False.
560:         '''
561:         if header not in self:
562:             return failobj
563:         for k, v in self._get_params_preserve(failobj, header):
564:             if k.lower() == param.lower():
565:                 if unquote:
566:                     return _unquotevalue(v)
567:                 else:
568:                     return v
569:         return failobj
570: 
571:     def set_param(self, param, value, header='Content-Type', requote=True,
572:                   charset=None, language=''):
573:         '''Set a parameter in the Content-Type header.
574: 
575:         If the parameter already exists in the header, its value will be
576:         replaced with the new value.
577: 
578:         If header is Content-Type and has not yet been defined for this
579:         message, it will be set to "text/plain" and the new parameter and
580:         value will be appended as per RFC 2045.
581: 
582:         An alternate header can be specified in the header argument, and all
583:         parameters will be quoted as necessary unless requote is False.
584: 
585:         If charset is specified, the parameter will be encoded according to RFC
586:         2231.  Optional language specifies the RFC 2231 language, defaulting
587:         to the empty string.  Both charset and language should be strings.
588:         '''
589:         if not isinstance(value, tuple) and charset:
590:             value = (charset, language, value)
591: 
592:         if header not in self and header.lower() == 'content-type':
593:             ctype = 'text/plain'
594:         else:
595:             ctype = self.get(header)
596:         if not self.get_param(param, header=header):
597:             if not ctype:
598:                 ctype = _formatparam(param, value, requote)
599:             else:
600:                 ctype = SEMISPACE.join(
601:                     [ctype, _formatparam(param, value, requote)])
602:         else:
603:             ctype = ''
604:             for old_param, old_value in self.get_params(header=header,
605:                                                         unquote=requote):
606:                 append_param = ''
607:                 if old_param.lower() == param.lower():
608:                     append_param = _formatparam(param, value, requote)
609:                 else:
610:                     append_param = _formatparam(old_param, old_value, requote)
611:                 if not ctype:
612:                     ctype = append_param
613:                 else:
614:                     ctype = SEMISPACE.join([ctype, append_param])
615:         if ctype != self.get(header):
616:             del self[header]
617:             self[header] = ctype
618: 
619:     def del_param(self, param, header='content-type', requote=True):
620:         '''Remove the given parameter completely from the Content-Type header.
621: 
622:         The header will be re-written in place without the parameter or its
623:         value. All values will be quoted as necessary unless requote is
624:         False.  Optional header specifies an alternative to the Content-Type
625:         header.
626:         '''
627:         if header not in self:
628:             return
629:         new_ctype = ''
630:         for p, v in self.get_params(header=header, unquote=requote):
631:             if p.lower() != param.lower():
632:                 if not new_ctype:
633:                     new_ctype = _formatparam(p, v, requote)
634:                 else:
635:                     new_ctype = SEMISPACE.join([new_ctype,
636:                                                 _formatparam(p, v, requote)])
637:         if new_ctype != self.get(header):
638:             del self[header]
639:             self[header] = new_ctype
640: 
641:     def set_type(self, type, header='Content-Type', requote=True):
642:         '''Set the main type and subtype for the Content-Type header.
643: 
644:         type must be a string in the form "maintype/subtype", otherwise a
645:         ValueError is raised.
646: 
647:         This method replaces the Content-Type header, keeping all the
648:         parameters in place.  If requote is False, this leaves the existing
649:         header's quoting as is.  Otherwise, the parameters will be quoted (the
650:         default).
651: 
652:         An alternative header can be specified in the header argument.  When
653:         the Content-Type header is set, we'll always also add a MIME-Version
654:         header.
655:         '''
656:         # BAW: should we be strict?
657:         if not type.count('/') == 1:
658:             raise ValueError
659:         # Set the Content-Type, you get a MIME-Version
660:         if header.lower() == 'content-type':
661:             del self['mime-version']
662:             self['MIME-Version'] = '1.0'
663:         if header not in self:
664:             self[header] = type
665:             return
666:         params = self.get_params(header=header, unquote=requote)
667:         del self[header]
668:         self[header] = type
669:         # Skip the first param; it's the old type.
670:         for p, v in params[1:]:
671:             self.set_param(p, v, header, requote)
672: 
673:     def get_filename(self, failobj=None):
674:         '''Return the filename associated with the payload if present.
675: 
676:         The filename is extracted from the Content-Disposition header's
677:         `filename' parameter, and it is unquoted.  If that header is missing
678:         the `filename' parameter, this method falls back to looking for the
679:         `name' parameter.
680:         '''
681:         missing = object()
682:         filename = self.get_param('filename', missing, 'content-disposition')
683:         if filename is missing:
684:             filename = self.get_param('name', missing, 'content-type')
685:         if filename is missing:
686:             return failobj
687:         return utils.collapse_rfc2231_value(filename).strip()
688: 
689:     def get_boundary(self, failobj=None):
690:         '''Return the boundary associated with the payload if present.
691: 
692:         The boundary is extracted from the Content-Type header's `boundary'
693:         parameter, and it is unquoted.
694:         '''
695:         missing = object()
696:         boundary = self.get_param('boundary', missing)
697:         if boundary is missing:
698:             return failobj
699:         # RFC 2046 says that boundaries may begin but not end in w/s
700:         return utils.collapse_rfc2231_value(boundary).rstrip()
701: 
702:     def set_boundary(self, boundary):
703:         '''Set the boundary parameter in Content-Type to 'boundary'.
704: 
705:         This is subtly different than deleting the Content-Type header and
706:         adding a new one with a new boundary parameter via add_header().  The
707:         main difference is that using the set_boundary() method preserves the
708:         order of the Content-Type header in the original message.
709: 
710:         HeaderParseError is raised if the message has no Content-Type header.
711:         '''
712:         missing = object()
713:         params = self._get_params_preserve(missing, 'content-type')
714:         if params is missing:
715:             # There was no Content-Type header, and we don't know what type
716:             # to set it to, so raise an exception.
717:             raise errors.HeaderParseError('No Content-Type header found')
718:         newparams = []
719:         foundp = False
720:         for pk, pv in params:
721:             if pk.lower() == 'boundary':
722:                 newparams.append(('boundary', '"%s"' % boundary))
723:                 foundp = True
724:             else:
725:                 newparams.append((pk, pv))
726:         if not foundp:
727:             # The original Content-Type header had no boundary attribute.
728:             # Tack one on the end.  BAW: should we raise an exception
729:             # instead???
730:             newparams.append(('boundary', '"%s"' % boundary))
731:         # Replace the existing Content-Type header with the new value
732:         newheaders = []
733:         for h, v in self._headers:
734:             if h.lower() == 'content-type':
735:                 parts = []
736:                 for k, v in newparams:
737:                     if v == '':
738:                         parts.append(k)
739:                     else:
740:                         parts.append('%s=%s' % (k, v))
741:                 newheaders.append((h, SEMISPACE.join(parts)))
742: 
743:             else:
744:                 newheaders.append((h, v))
745:         self._headers = newheaders
746: 
747:     def get_content_charset(self, failobj=None):
748:         '''Return the charset parameter of the Content-Type header.
749: 
750:         The returned string is always coerced to lower case.  If there is no
751:         Content-Type header, or if that header has no charset parameter,
752:         failobj is returned.
753:         '''
754:         missing = object()
755:         charset = self.get_param('charset', missing)
756:         if charset is missing:
757:             return failobj
758:         if isinstance(charset, tuple):
759:             # RFC 2231 encoded, so decode it, and it better end up as ascii.
760:             pcharset = charset[0] or 'us-ascii'
761:             try:
762:                 # LookupError will be raised if the charset isn't known to
763:                 # Python.  UnicodeError will be raised if the encoded text
764:                 # contains a character not in the charset.
765:                 charset = unicode(charset[2], pcharset).encode('us-ascii')
766:             except (LookupError, UnicodeError):
767:                 charset = charset[2]
768:         # charset character must be in us-ascii range
769:         try:
770:             if isinstance(charset, str):
771:                 charset = unicode(charset, 'us-ascii')
772:             charset = charset.encode('us-ascii')
773:         except UnicodeError:
774:             return failobj
775:         # RFC 2046, $4.1.2 says charsets are not case sensitive
776:         return charset.lower()
777: 
778:     def get_charsets(self, failobj=None):
779:         '''Return a list containing the charset(s) used in this message.
780: 
781:         The returned list of items describes the Content-Type headers'
782:         charset parameter for this message and all the subparts in its
783:         payload.
784: 
785:         Each item will either be a string (the value of the charset parameter
786:         in the Content-Type header of that part) or the value of the
787:         'failobj' parameter (defaults to None), if the part does not have a
788:         main MIME type of "text", or the charset is not defined.
789: 
790:         The list will contain one string for each part of the message, plus
791:         one for the container message (i.e. self), so that a non-multipart
792:         message will still return a list of length 1.
793:         '''
794:         return [part.get_content_charset(failobj) for part in self.walk()]
795: 
796:     # I.e. def walk(self): ...
797:     from email.iterators import walk
798: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_16049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Basic message object for the email package object model.')

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['Message']
module_type_store.set_exportable_members(['Message'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_16050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_16051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'Message')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_16050, str_16051)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_16050)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import re' statement (line 9)
import re

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import uu' statement (line 10)
import uu

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'uu', uu, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import binascii' statement (line 11)
import binascii

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'binascii', binascii, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import warnings' statement (line 12)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from cStringIO import StringIO' statement (line 13)
try:
    from cStringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import email.charset' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_16052 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'email.charset')

if (type(import_16052) is not StypyTypeError):

    if (import_16052 != 'pyd_module'):
        __import__(import_16052)
        sys_modules_16053 = sys.modules[import_16052]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'email.charset', sys_modules_16053.module_type_store, module_type_store)
    else:
        import email.charset

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'email.charset', email.charset, module_type_store)

else:
    # Assigning a type to the variable 'email.charset' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'email.charset', import_16052)

remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from email import utils' statement (line 17)
try:
    from email import utils

except:
    utils = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'email', None, module_type_store, ['utils'], [utils])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from email import errors' statement (line 18)
try:
    from email import errors

except:
    errors = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'email', None, module_type_store, ['errors'], [errors])


# Assigning a Str to a Name (line 20):

# Assigning a Str to a Name (line 20):
str_16054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'str', '; ')
# Assigning a type to the variable 'SEMISPACE' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'SEMISPACE', str_16054)

# Assigning a Call to a Name (line 24):

# Assigning a Call to a Name (line 24):

# Call to compile(...): (line 24)
# Processing the call arguments (line 24)
str_16057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'str', '[ \\(\\)<>@,;:\\\\"/\\[\\]\\?=]')
# Processing the call keyword arguments (line 24)
kwargs_16058 = {}
# Getting the type of 're' (line 24)
re_16055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 're', False)
# Obtaining the member 'compile' of a type (line 24)
compile_16056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), re_16055, 'compile')
# Calling compile(args, kwargs) (line 24)
compile_call_result_16059 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), compile_16056, *[str_16057], **kwargs_16058)

# Assigning a type to the variable 'tspecials' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'tspecials', compile_call_result_16059)

@norecursion
def _splitparam(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_splitparam'
    module_type_store = module_type_store.open_function_context('_splitparam', 28, 0, False)
    
    # Passed parameters checking function
    _splitparam.stypy_localization = localization
    _splitparam.stypy_type_of_self = None
    _splitparam.stypy_type_store = module_type_store
    _splitparam.stypy_function_name = '_splitparam'
    _splitparam.stypy_param_names_list = ['param']
    _splitparam.stypy_varargs_param_name = None
    _splitparam.stypy_kwargs_param_name = None
    _splitparam.stypy_call_defaults = defaults
    _splitparam.stypy_call_varargs = varargs
    _splitparam.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_splitparam', ['param'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_splitparam', localization, ['param'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_splitparam(...)' code ##################

    
    # Assigning a Call to a Tuple (line 33):
    
    # Assigning a Call to a Name:
    
    # Call to partition(...): (line 33)
    # Processing the call arguments (line 33)
    str_16062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'str', ';')
    # Processing the call keyword arguments (line 33)
    kwargs_16063 = {}
    # Getting the type of 'param' (line 33)
    param_16060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'param', False)
    # Obtaining the member 'partition' of a type (line 33)
    partition_16061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), param_16060, 'partition')
    # Calling partition(args, kwargs) (line 33)
    partition_call_result_16064 = invoke(stypy.reporting.localization.Localization(__file__, 33, 16), partition_16061, *[str_16062], **kwargs_16063)
    
    # Assigning a type to the variable 'call_assignment_16042' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'call_assignment_16042', partition_call_result_16064)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16042' (line 33)
    call_assignment_16042_16065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'call_assignment_16042', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_16066 = stypy_get_value_from_tuple(call_assignment_16042_16065, 3, 0)
    
    # Assigning a type to the variable 'call_assignment_16043' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'call_assignment_16043', stypy_get_value_from_tuple_call_result_16066)
    
    # Assigning a Name to a Name (line 33):
    # Getting the type of 'call_assignment_16043' (line 33)
    call_assignment_16043_16067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'call_assignment_16043')
    # Assigning a type to the variable 'a' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'a', call_assignment_16043_16067)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16042' (line 33)
    call_assignment_16042_16068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'call_assignment_16042', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_16069 = stypy_get_value_from_tuple(call_assignment_16042_16068, 3, 1)
    
    # Assigning a type to the variable 'call_assignment_16044' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'call_assignment_16044', stypy_get_value_from_tuple_call_result_16069)
    
    # Assigning a Name to a Name (line 33):
    # Getting the type of 'call_assignment_16044' (line 33)
    call_assignment_16044_16070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'call_assignment_16044')
    # Assigning a type to the variable 'sep' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'sep', call_assignment_16044_16070)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16042' (line 33)
    call_assignment_16042_16071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'call_assignment_16042', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_16072 = stypy_get_value_from_tuple(call_assignment_16042_16071, 3, 2)
    
    # Assigning a type to the variable 'call_assignment_16045' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'call_assignment_16045', stypy_get_value_from_tuple_call_result_16072)
    
    # Assigning a Name to a Name (line 33):
    # Getting the type of 'call_assignment_16045' (line 33)
    call_assignment_16045_16073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'call_assignment_16045')
    # Assigning a type to the variable 'b' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'b', call_assignment_16045_16073)
    
    # Getting the type of 'sep' (line 34)
    sep_16074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'sep')
    # Applying the 'not' unary operator (line 34)
    result_not__16075 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 7), 'not', sep_16074)
    
    # Testing if the type of an if condition is none (line 34)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 34, 4), result_not__16075):
        pass
    else:
        
        # Testing the type of an if condition (line 34)
        if_condition_16076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 4), result_not__16075)
        # Assigning a type to the variable 'if_condition_16076' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'if_condition_16076', if_condition_16076)
        # SSA begins for if statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 35)
        tuple_16077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 35)
        # Adding element type (line 35)
        
        # Call to strip(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_16080 = {}
        # Getting the type of 'a' (line 35)
        a_16078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'a', False)
        # Obtaining the member 'strip' of a type (line 35)
        strip_16079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), a_16078, 'strip')
        # Calling strip(args, kwargs) (line 35)
        strip_call_result_16081 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), strip_16079, *[], **kwargs_16080)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_16077, strip_call_result_16081)
        # Adding element type (line 35)
        # Getting the type of 'None' (line 35)
        None_16082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_16077, None_16082)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', tuple_16077)
        # SSA join for if statement (line 34)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_16083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    
    # Call to strip(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_16086 = {}
    # Getting the type of 'a' (line 36)
    a_16084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'a', False)
    # Obtaining the member 'strip' of a type (line 36)
    strip_16085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 11), a_16084, 'strip')
    # Calling strip(args, kwargs) (line 36)
    strip_call_result_16087 = invoke(stypy.reporting.localization.Localization(__file__, 36, 11), strip_16085, *[], **kwargs_16086)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 11), tuple_16083, strip_call_result_16087)
    # Adding element type (line 36)
    
    # Call to strip(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_16090 = {}
    # Getting the type of 'b' (line 36)
    b_16088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'b', False)
    # Obtaining the member 'strip' of a type (line 36)
    strip_16089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 22), b_16088, 'strip')
    # Calling strip(args, kwargs) (line 36)
    strip_call_result_16091 = invoke(stypy.reporting.localization.Localization(__file__, 36, 22), strip_16089, *[], **kwargs_16090)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 11), tuple_16083, strip_call_result_16091)
    
    # Assigning a type to the variable 'stypy_return_type' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type', tuple_16083)
    
    # ################# End of '_splitparam(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_splitparam' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_16092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16092)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_splitparam'
    return stypy_return_type_16092

# Assigning a type to the variable '_splitparam' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '_splitparam', _splitparam)

@norecursion
def _formatparam(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 38)
    None_16093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'None')
    # Getting the type of 'True' (line 38)
    True_16094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'True')
    defaults = [None_16093, True_16094]
    # Create a new context for function '_formatparam'
    module_type_store = module_type_store.open_function_context('_formatparam', 38, 0, False)
    
    # Passed parameters checking function
    _formatparam.stypy_localization = localization
    _formatparam.stypy_type_of_self = None
    _formatparam.stypy_type_store = module_type_store
    _formatparam.stypy_function_name = '_formatparam'
    _formatparam.stypy_param_names_list = ['param', 'value', 'quote']
    _formatparam.stypy_varargs_param_name = None
    _formatparam.stypy_kwargs_param_name = None
    _formatparam.stypy_call_defaults = defaults
    _formatparam.stypy_call_varargs = varargs
    _formatparam.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_formatparam', ['param', 'value', 'quote'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_formatparam', localization, ['param', 'value', 'quote'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_formatparam(...)' code ##################

    str_16095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'str', 'Convenience function to format and return a key=value pair.\n\n    This will quote the value if needed or if quote is true.  If value is a\n    three tuple (charset, language, value), it will be encoded according\n    to RFC2231 rules.\n    ')
    
    # Evaluating a boolean operation
    
    # Getting the type of 'value' (line 45)
    value_16096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 7), 'value')
    # Getting the type of 'None' (line 45)
    None_16097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'None')
    # Applying the binary operator 'isnot' (line 45)
    result_is_not_16098 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 7), 'isnot', value_16096, None_16097)
    
    
    
    # Call to len(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'value' (line 45)
    value_16100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'value', False)
    # Processing the call keyword arguments (line 45)
    kwargs_16101 = {}
    # Getting the type of 'len' (line 45)
    len_16099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'len', False)
    # Calling len(args, kwargs) (line 45)
    len_call_result_16102 = invoke(stypy.reporting.localization.Localization(__file__, 45, 29), len_16099, *[value_16100], **kwargs_16101)
    
    int_16103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 42), 'int')
    # Applying the binary operator '>' (line 45)
    result_gt_16104 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 29), '>', len_call_result_16102, int_16103)
    
    # Applying the binary operator 'and' (line 45)
    result_and_keyword_16105 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 7), 'and', result_is_not_16098, result_gt_16104)
    
    # Testing if the type of an if condition is none (line 45)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 45, 4), result_and_keyword_16105):
        # Getting the type of 'param' (line 60)
        param_16152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'param')
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', param_16152)
    else:
        
        # Testing the type of an if condition (line 45)
        if_condition_16106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 4), result_and_keyword_16105)
        # Assigning a type to the variable 'if_condition_16106' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'if_condition_16106', if_condition_16106)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 49)
        # Getting the type of 'tuple' (line 49)
        tuple_16107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'tuple')
        # Getting the type of 'value' (line 49)
        value_16108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'value')
        
        (may_be_16109, more_types_in_union_16110) = may_be_subtype(tuple_16107, value_16108)

        if may_be_16109:

            if more_types_in_union_16110:
                # Runtime conditional SSA (line 49)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'value' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'value', remove_not_subtype_from_union(value_16108, tuple))
            
            # Getting the type of 'param' (line 51)
            param_16111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'param')
            str_16112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'str', '*')
            # Applying the binary operator '+=' (line 51)
            result_iadd_16113 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 12), '+=', param_16111, str_16112)
            # Assigning a type to the variable 'param' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'param', result_iadd_16113)
            
            
            # Assigning a Call to a Name (line 52):
            
            # Assigning a Call to a Name (line 52):
            
            # Call to encode_rfc2231(...): (line 52)
            # Processing the call arguments (line 52)
            
            # Obtaining the type of the subscript
            int_16116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 47), 'int')
            # Getting the type of 'value' (line 52)
            value_16117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'value', False)
            # Obtaining the member '__getitem__' of a type (line 52)
            getitem___16118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 41), value_16117, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 52)
            subscript_call_result_16119 = invoke(stypy.reporting.localization.Localization(__file__, 52, 41), getitem___16118, int_16116)
            
            
            # Obtaining the type of the subscript
            int_16120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 57), 'int')
            # Getting the type of 'value' (line 52)
            value_16121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 51), 'value', False)
            # Obtaining the member '__getitem__' of a type (line 52)
            getitem___16122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 51), value_16121, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 52)
            subscript_call_result_16123 = invoke(stypy.reporting.localization.Localization(__file__, 52, 51), getitem___16122, int_16120)
            
            
            # Obtaining the type of the subscript
            int_16124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 67), 'int')
            # Getting the type of 'value' (line 52)
            value_16125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 61), 'value', False)
            # Obtaining the member '__getitem__' of a type (line 52)
            getitem___16126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 61), value_16125, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 52)
            subscript_call_result_16127 = invoke(stypy.reporting.localization.Localization(__file__, 52, 61), getitem___16126, int_16124)
            
            # Processing the call keyword arguments (line 52)
            kwargs_16128 = {}
            # Getting the type of 'utils' (line 52)
            utils_16114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'utils', False)
            # Obtaining the member 'encode_rfc2231' of a type (line 52)
            encode_rfc2231_16115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 20), utils_16114, 'encode_rfc2231')
            # Calling encode_rfc2231(args, kwargs) (line 52)
            encode_rfc2231_call_result_16129 = invoke(stypy.reporting.localization.Localization(__file__, 52, 20), encode_rfc2231_16115, *[subscript_call_result_16119, subscript_call_result_16123, subscript_call_result_16127], **kwargs_16128)
            
            # Assigning a type to the variable 'value' (line 52)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'value', encode_rfc2231_call_result_16129)

            if more_types_in_union_16110:
                # SSA join for if statement (line 49)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Evaluating a boolean operation
        # Getting the type of 'quote' (line 55)
        quote_16130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'quote')
        
        # Call to search(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'value' (line 55)
        value_16133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 37), 'value', False)
        # Processing the call keyword arguments (line 55)
        kwargs_16134 = {}
        # Getting the type of 'tspecials' (line 55)
        tspecials_16131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'tspecials', False)
        # Obtaining the member 'search' of a type (line 55)
        search_16132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 20), tspecials_16131, 'search')
        # Calling search(args, kwargs) (line 55)
        search_call_result_16135 = invoke(stypy.reporting.localization.Localization(__file__, 55, 20), search_16132, *[value_16133], **kwargs_16134)
        
        # Applying the binary operator 'or' (line 55)
        result_or_keyword_16136 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 11), 'or', quote_16130, search_call_result_16135)
        
        # Testing if the type of an if condition is none (line 55)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 55, 8), result_or_keyword_16136):
            str_16147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 19), 'str', '%s=%s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 58)
            tuple_16148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 58)
            # Adding element type (line 58)
            # Getting the type of 'param' (line 58)
            param_16149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 30), 'param')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 30), tuple_16148, param_16149)
            # Adding element type (line 58)
            # Getting the type of 'value' (line 58)
            value_16150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'value')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 30), tuple_16148, value_16150)
            
            # Applying the binary operator '%' (line 58)
            result_mod_16151 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 19), '%', str_16147, tuple_16148)
            
            # Assigning a type to the variable 'stypy_return_type' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'stypy_return_type', result_mod_16151)
        else:
            
            # Testing the type of an if condition (line 55)
            if_condition_16137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 8), result_or_keyword_16136)
            # Assigning a type to the variable 'if_condition_16137' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'if_condition_16137', if_condition_16137)
            # SSA begins for if statement (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_16138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'str', '%s="%s"')
            
            # Obtaining an instance of the builtin type 'tuple' (line 56)
            tuple_16139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 32), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 56)
            # Adding element type (line 56)
            # Getting the type of 'param' (line 56)
            param_16140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 32), 'param')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 32), tuple_16139, param_16140)
            # Adding element type (line 56)
            
            # Call to quote(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'value' (line 56)
            value_16143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 51), 'value', False)
            # Processing the call keyword arguments (line 56)
            kwargs_16144 = {}
            # Getting the type of 'utils' (line 56)
            utils_16141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'utils', False)
            # Obtaining the member 'quote' of a type (line 56)
            quote_16142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 39), utils_16141, 'quote')
            # Calling quote(args, kwargs) (line 56)
            quote_call_result_16145 = invoke(stypy.reporting.localization.Localization(__file__, 56, 39), quote_16142, *[value_16143], **kwargs_16144)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 32), tuple_16139, quote_call_result_16145)
            
            # Applying the binary operator '%' (line 56)
            result_mod_16146 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 19), '%', str_16138, tuple_16139)
            
            # Assigning a type to the variable 'stypy_return_type' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'stypy_return_type', result_mod_16146)
            # SSA branch for the else part of an if statement (line 55)
            module_type_store.open_ssa_branch('else')
            str_16147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 19), 'str', '%s=%s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 58)
            tuple_16148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 58)
            # Adding element type (line 58)
            # Getting the type of 'param' (line 58)
            param_16149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 30), 'param')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 30), tuple_16148, param_16149)
            # Adding element type (line 58)
            # Getting the type of 'value' (line 58)
            value_16150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'value')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 30), tuple_16148, value_16150)
            
            # Applying the binary operator '%' (line 58)
            result_mod_16151 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 19), '%', str_16147, tuple_16148)
            
            # Assigning a type to the variable 'stypy_return_type' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'stypy_return_type', result_mod_16151)
            # SSA join for if statement (line 55)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 45)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'param' (line 60)
        param_16152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'param')
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', param_16152)
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of '_formatparam(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_formatparam' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_16153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16153)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_formatparam'
    return stypy_return_type_16153

# Assigning a type to the variable '_formatparam' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), '_formatparam', _formatparam)

@norecursion
def _parseparam(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_parseparam'
    module_type_store = module_type_store.open_function_context('_parseparam', 62, 0, False)
    
    # Passed parameters checking function
    _parseparam.stypy_localization = localization
    _parseparam.stypy_type_of_self = None
    _parseparam.stypy_type_store = module_type_store
    _parseparam.stypy_function_name = '_parseparam'
    _parseparam.stypy_param_names_list = ['s']
    _parseparam.stypy_varargs_param_name = None
    _parseparam.stypy_kwargs_param_name = None
    _parseparam.stypy_call_defaults = defaults
    _parseparam.stypy_call_varargs = varargs
    _parseparam.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_parseparam', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_parseparam', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_parseparam(...)' code ##################

    
    # Assigning a List to a Name (line 63):
    
    # Assigning a List to a Name (line 63):
    
    # Obtaining an instance of the builtin type 'list' (line 63)
    list_16154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 63)
    
    # Assigning a type to the variable 'plist' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'plist', list_16154)
    
    
    
    # Obtaining the type of the subscript
    int_16155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 13), 'int')
    slice_16156 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 64, 10), None, int_16155, None)
    # Getting the type of 's' (line 64)
    s_16157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 10), 's')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___16158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 10), s_16157, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_16159 = invoke(stypy.reporting.localization.Localization(__file__, 64, 10), getitem___16158, slice_16156)
    
    str_16160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'str', ';')
    # Applying the binary operator '==' (line 64)
    result_eq_16161 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 10), '==', subscript_call_result_16159, str_16160)
    
    # Assigning a type to the variable 'result_eq_16161' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'result_eq_16161', result_eq_16161)
    # Testing if the while is going to be iterated (line 64)
    # Testing the type of an if condition (line 64)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 4), result_eq_16161)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 64, 4), result_eq_16161):
        # SSA begins for while statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Subscript to a Name (line 65):
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_16162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 14), 'int')
        slice_16163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 65, 12), int_16162, None, None)
        # Getting the type of 's' (line 65)
        s_16164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 's')
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___16165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), s_16164, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_16166 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), getitem___16165, slice_16163)
        
        # Assigning a type to the variable 's' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 's', subscript_call_result_16166)
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to find(...): (line 66)
        # Processing the call arguments (line 66)
        str_16169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 21), 'str', ';')
        # Processing the call keyword arguments (line 66)
        kwargs_16170 = {}
        # Getting the type of 's' (line 66)
        s_16167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 14), 's', False)
        # Obtaining the member 'find' of a type (line 66)
        find_16168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 14), s_16167, 'find')
        # Calling find(args, kwargs) (line 66)
        find_call_result_16171 = invoke(stypy.reporting.localization.Localization(__file__, 66, 14), find_16168, *[str_16169], **kwargs_16170)
        
        # Assigning a type to the variable 'end' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'end', find_call_result_16171)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'end' (line 67)
        end_16172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'end')
        int_16173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'int')
        # Applying the binary operator '>' (line 67)
        result_gt_16174 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 14), '>', end_16172, int_16173)
        
        
        # Call to count(...): (line 67)
        # Processing the call arguments (line 67)
        str_16177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 35), 'str', '"')
        int_16178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 40), 'int')
        # Getting the type of 'end' (line 67)
        end_16179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 43), 'end', False)
        # Processing the call keyword arguments (line 67)
        kwargs_16180 = {}
        # Getting the type of 's' (line 67)
        s_16175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 27), 's', False)
        # Obtaining the member 'count' of a type (line 67)
        count_16176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 27), s_16175, 'count')
        # Calling count(args, kwargs) (line 67)
        count_call_result_16181 = invoke(stypy.reporting.localization.Localization(__file__, 67, 27), count_16176, *[str_16177, int_16178, end_16179], **kwargs_16180)
        
        
        # Call to count(...): (line 67)
        # Processing the call arguments (line 67)
        str_16184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 58), 'str', '\\"')
        int_16185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 65), 'int')
        # Getting the type of 'end' (line 67)
        end_16186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 68), 'end', False)
        # Processing the call keyword arguments (line 67)
        kwargs_16187 = {}
        # Getting the type of 's' (line 67)
        s_16182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 50), 's', False)
        # Obtaining the member 'count' of a type (line 67)
        count_16183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 50), s_16182, 'count')
        # Calling count(args, kwargs) (line 67)
        count_call_result_16188 = invoke(stypy.reporting.localization.Localization(__file__, 67, 50), count_16183, *[str_16184, int_16185, end_16186], **kwargs_16187)
        
        # Applying the binary operator '-' (line 67)
        result_sub_16189 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 27), '-', count_call_result_16181, count_call_result_16188)
        
        int_16190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 76), 'int')
        # Applying the binary operator '%' (line 67)
        result_mod_16191 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 26), '%', result_sub_16189, int_16190)
        
        # Applying the binary operator 'and' (line 67)
        result_and_keyword_16192 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 14), 'and', result_gt_16174, result_mod_16191)
        
        # Assigning a type to the variable 'result_and_keyword_16192' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'result_and_keyword_16192', result_and_keyword_16192)
        # Testing if the while is going to be iterated (line 67)
        # Testing the type of an if condition (line 67)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_and_keyword_16192)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 67, 8), result_and_keyword_16192):
            # SSA begins for while statement (line 67)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 68):
            
            # Assigning a Call to a Name (line 68):
            
            # Call to find(...): (line 68)
            # Processing the call arguments (line 68)
            str_16195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'str', ';')
            # Getting the type of 'end' (line 68)
            end_16196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'end', False)
            int_16197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 36), 'int')
            # Applying the binary operator '+' (line 68)
            result_add_16198 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 30), '+', end_16196, int_16197)
            
            # Processing the call keyword arguments (line 68)
            kwargs_16199 = {}
            # Getting the type of 's' (line 68)
            s_16193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), 's', False)
            # Obtaining the member 'find' of a type (line 68)
            find_16194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 18), s_16193, 'find')
            # Calling find(args, kwargs) (line 68)
            find_call_result_16200 = invoke(stypy.reporting.localization.Localization(__file__, 68, 18), find_16194, *[str_16195, result_add_16198], **kwargs_16199)
            
            # Assigning a type to the variable 'end' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'end', find_call_result_16200)
            # SSA join for while statement (line 67)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'end' (line 69)
        end_16201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'end')
        int_16202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 17), 'int')
        # Applying the binary operator '<' (line 69)
        result_lt_16203 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 11), '<', end_16201, int_16202)
        
        # Testing if the type of an if condition is none (line 69)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 8), result_lt_16203):
            pass
        else:
            
            # Testing the type of an if condition (line 69)
            if_condition_16204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), result_lt_16203)
            # Assigning a type to the variable 'if_condition_16204' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_16204', if_condition_16204)
            # SSA begins for if statement (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 70):
            
            # Assigning a Call to a Name (line 70):
            
            # Call to len(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of 's' (line 70)
            s_16206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 22), 's', False)
            # Processing the call keyword arguments (line 70)
            kwargs_16207 = {}
            # Getting the type of 'len' (line 70)
            len_16205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'len', False)
            # Calling len(args, kwargs) (line 70)
            len_call_result_16208 = invoke(stypy.reporting.localization.Localization(__file__, 70, 18), len_16205, *[s_16206], **kwargs_16207)
            
            # Assigning a type to the variable 'end' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'end', len_call_result_16208)
            # SSA join for if statement (line 69)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Subscript to a Name (line 71):
        
        # Assigning a Subscript to a Name (line 71):
        
        # Obtaining the type of the subscript
        # Getting the type of 'end' (line 71)
        end_16209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'end')
        slice_16210 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 71, 12), None, end_16209, None)
        # Getting the type of 's' (line 71)
        s_16211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 's')
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___16212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), s_16211, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_16213 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), getitem___16212, slice_16210)
        
        # Assigning a type to the variable 'f' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'f', subscript_call_result_16213)
        
        str_16214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 11), 'str', '=')
        # Getting the type of 'f' (line 72)
        f_16215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'f')
        # Applying the binary operator 'in' (line 72)
        result_contains_16216 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 11), 'in', str_16214, f_16215)
        
        # Testing if the type of an if condition is none (line 72)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 8), result_contains_16216):
            pass
        else:
            
            # Testing the type of an if condition (line 72)
            if_condition_16217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 8), result_contains_16216)
            # Assigning a type to the variable 'if_condition_16217' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'if_condition_16217', if_condition_16217)
            # SSA begins for if statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 73):
            
            # Assigning a Call to a Name (line 73):
            
            # Call to index(...): (line 73)
            # Processing the call arguments (line 73)
            str_16220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'str', '=')
            # Processing the call keyword arguments (line 73)
            kwargs_16221 = {}
            # Getting the type of 'f' (line 73)
            f_16218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'f', False)
            # Obtaining the member 'index' of a type (line 73)
            index_16219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), f_16218, 'index')
            # Calling index(args, kwargs) (line 73)
            index_call_result_16222 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), index_16219, *[str_16220], **kwargs_16221)
            
            # Assigning a type to the variable 'i' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'i', index_call_result_16222)
            
            # Assigning a BinOp to a Name (line 74):
            
            # Assigning a BinOp to a Name (line 74):
            
            # Call to lower(...): (line 74)
            # Processing the call keyword arguments (line 74)
            kwargs_16232 = {}
            
            # Call to strip(...): (line 74)
            # Processing the call keyword arguments (line 74)
            kwargs_16229 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 74)
            i_16223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'i', False)
            slice_16224 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 74, 16), None, i_16223, None)
            # Getting the type of 'f' (line 74)
            f_16225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'f', False)
            # Obtaining the member '__getitem__' of a type (line 74)
            getitem___16226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), f_16225, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 74)
            subscript_call_result_16227 = invoke(stypy.reporting.localization.Localization(__file__, 74, 16), getitem___16226, slice_16224)
            
            # Obtaining the member 'strip' of a type (line 74)
            strip_16228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), subscript_call_result_16227, 'strip')
            # Calling strip(args, kwargs) (line 74)
            strip_call_result_16230 = invoke(stypy.reporting.localization.Localization(__file__, 74, 16), strip_16228, *[], **kwargs_16229)
            
            # Obtaining the member 'lower' of a type (line 74)
            lower_16231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), strip_call_result_16230, 'lower')
            # Calling lower(args, kwargs) (line 74)
            lower_call_result_16233 = invoke(stypy.reporting.localization.Localization(__file__, 74, 16), lower_16231, *[], **kwargs_16232)
            
            str_16234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 40), 'str', '=')
            # Applying the binary operator '+' (line 74)
            result_add_16235 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 16), '+', lower_call_result_16233, str_16234)
            
            
            # Call to strip(...): (line 74)
            # Processing the call keyword arguments (line 74)
            kwargs_16244 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 74)
            i_16236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 48), 'i', False)
            int_16237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 50), 'int')
            # Applying the binary operator '+' (line 74)
            result_add_16238 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 48), '+', i_16236, int_16237)
            
            slice_16239 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 74, 46), result_add_16238, None, None)
            # Getting the type of 'f' (line 74)
            f_16240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 46), 'f', False)
            # Obtaining the member '__getitem__' of a type (line 74)
            getitem___16241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 46), f_16240, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 74)
            subscript_call_result_16242 = invoke(stypy.reporting.localization.Localization(__file__, 74, 46), getitem___16241, slice_16239)
            
            # Obtaining the member 'strip' of a type (line 74)
            strip_16243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 46), subscript_call_result_16242, 'strip')
            # Calling strip(args, kwargs) (line 74)
            strip_call_result_16245 = invoke(stypy.reporting.localization.Localization(__file__, 74, 46), strip_16243, *[], **kwargs_16244)
            
            # Applying the binary operator '+' (line 74)
            result_add_16246 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 44), '+', result_add_16235, strip_call_result_16245)
            
            # Assigning a type to the variable 'f' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'f', result_add_16246)
            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to append(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Call to strip(...): (line 75)
        # Processing the call keyword arguments (line 75)
        kwargs_16251 = {}
        # Getting the type of 'f' (line 75)
        f_16249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'f', False)
        # Obtaining the member 'strip' of a type (line 75)
        strip_16250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 21), f_16249, 'strip')
        # Calling strip(args, kwargs) (line 75)
        strip_call_result_16252 = invoke(stypy.reporting.localization.Localization(__file__, 75, 21), strip_16250, *[], **kwargs_16251)
        
        # Processing the call keyword arguments (line 75)
        kwargs_16253 = {}
        # Getting the type of 'plist' (line 75)
        plist_16247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'plist', False)
        # Obtaining the member 'append' of a type (line 75)
        append_16248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), plist_16247, 'append')
        # Calling append(args, kwargs) (line 75)
        append_call_result_16254 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), append_16248, *[strip_call_result_16252], **kwargs_16253)
        
        
        # Assigning a Subscript to a Name (line 76):
        
        # Assigning a Subscript to a Name (line 76):
        
        # Obtaining the type of the subscript
        # Getting the type of 'end' (line 76)
        end_16255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'end')
        slice_16256 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 76, 12), end_16255, None, None)
        # Getting the type of 's' (line 76)
        s_16257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 's')
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___16258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), s_16257, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_16259 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), getitem___16258, slice_16256)
        
        # Assigning a type to the variable 's' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 's', subscript_call_result_16259)
        # SSA join for while statement (line 64)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'plist' (line 77)
    plist_16260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'plist')
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type', plist_16260)
    
    # ################# End of '_parseparam(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_parseparam' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_16261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16261)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_parseparam'
    return stypy_return_type_16261

# Assigning a type to the variable '_parseparam' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), '_parseparam', _parseparam)

@norecursion
def _unquotevalue(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_unquotevalue'
    module_type_store = module_type_store.open_function_context('_unquotevalue', 80, 0, False)
    
    # Passed parameters checking function
    _unquotevalue.stypy_localization = localization
    _unquotevalue.stypy_type_of_self = None
    _unquotevalue.stypy_type_store = module_type_store
    _unquotevalue.stypy_function_name = '_unquotevalue'
    _unquotevalue.stypy_param_names_list = ['value']
    _unquotevalue.stypy_varargs_param_name = None
    _unquotevalue.stypy_kwargs_param_name = None
    _unquotevalue.stypy_call_defaults = defaults
    _unquotevalue.stypy_call_varargs = varargs
    _unquotevalue.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_unquotevalue', ['value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_unquotevalue', localization, ['value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_unquotevalue(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 85)
    # Getting the type of 'tuple' (line 85)
    tuple_16262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'tuple')
    # Getting the type of 'value' (line 85)
    value_16263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'value')
    
    (may_be_16264, more_types_in_union_16265) = may_be_subtype(tuple_16262, value_16263)

    if may_be_16264:

        if more_types_in_union_16265:
            # Runtime conditional SSA (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'value' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'value', remove_not_subtype_from_union(value_16263, tuple))
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_16266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        
        # Obtaining the type of the subscript
        int_16267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'int')
        # Getting the type of 'value' (line 86)
        value_16268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'value')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___16269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), value_16268, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_16270 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), getitem___16269, int_16267)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), tuple_16266, subscript_call_result_16270)
        # Adding element type (line 86)
        
        # Obtaining the type of the subscript
        int_16271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'int')
        # Getting the type of 'value' (line 86)
        value_16272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'value')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___16273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 25), value_16272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_16274 = invoke(stypy.reporting.localization.Localization(__file__, 86, 25), getitem___16273, int_16271)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), tuple_16266, subscript_call_result_16274)
        # Adding element type (line 86)
        
        # Call to unquote(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Obtaining the type of the subscript
        int_16277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 55), 'int')
        # Getting the type of 'value' (line 86)
        value_16278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 49), 'value', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___16279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 49), value_16278, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_16280 = invoke(stypy.reporting.localization.Localization(__file__, 86, 49), getitem___16279, int_16277)
        
        # Processing the call keyword arguments (line 86)
        kwargs_16281 = {}
        # Getting the type of 'utils' (line 86)
        utils_16275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 35), 'utils', False)
        # Obtaining the member 'unquote' of a type (line 86)
        unquote_16276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 35), utils_16275, 'unquote')
        # Calling unquote(args, kwargs) (line 86)
        unquote_call_result_16282 = invoke(stypy.reporting.localization.Localization(__file__, 86, 35), unquote_16276, *[subscript_call_result_16280], **kwargs_16281)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), tuple_16266, unquote_call_result_16282)
        
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', tuple_16266)

        if more_types_in_union_16265:
            # Runtime conditional SSA for else branch (line 85)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_16264) or more_types_in_union_16265):
        # Assigning a type to the variable 'value' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'value', remove_subtype_from_union(value_16263, tuple))
        
        # Call to unquote(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'value' (line 88)
        value_16285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'value', False)
        # Processing the call keyword arguments (line 88)
        kwargs_16286 = {}
        # Getting the type of 'utils' (line 88)
        utils_16283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'utils', False)
        # Obtaining the member 'unquote' of a type (line 88)
        unquote_16284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), utils_16283, 'unquote')
        # Calling unquote(args, kwargs) (line 88)
        unquote_call_result_16287 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), unquote_16284, *[value_16285], **kwargs_16286)
        
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', unquote_call_result_16287)

        if (may_be_16264 and more_types_in_union_16265):
            # SSA join for if statement (line 85)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_unquotevalue(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_unquotevalue' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_16288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16288)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_unquotevalue'
    return stypy_return_type_16288

# Assigning a type to the variable '_unquotevalue' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), '_unquotevalue', _unquotevalue)
# Declaration of the 'Message' class

class Message:
    str_16289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, (-1)), 'str', "Basic message object.\n\n    A message object is defined as something that has a bunch of RFC 2822\n    headers and a payload.  It may optionally have an envelope header\n    (a.k.a. Unix-From or From_ header).  If the message is a container (i.e. a\n    multipart or a message/rfc822), then the payload is a list of Message\n    objects, otherwise it is a string.\n\n    Message objects implement part of the `mapping' interface, which assumes\n    there is exactly one occurrence of the header per message.  Some headers\n    do in fact appear multiple times (e.g. Received) and for those headers,\n    you must use the explicit API to set or get all the headers.  Not all of\n    the mapping methods are implemented.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 108):
        
        # Assigning a List to a Attribute (line 108):
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_16290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        
        # Getting the type of 'self' (line 108)
        self_16291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self')
        # Setting the type of the member '_headers' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_16291, '_headers', list_16290)
        
        # Assigning a Name to a Attribute (line 109):
        
        # Assigning a Name to a Attribute (line 109):
        # Getting the type of 'None' (line 109)
        None_16292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'None')
        # Getting the type of 'self' (line 109)
        self_16293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member '_unixfrom' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_16293, '_unixfrom', None_16292)
        
        # Assigning a Name to a Attribute (line 110):
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'None' (line 110)
        None_16294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 24), 'None')
        # Getting the type of 'self' (line 110)
        self_16295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member '_payload' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_16295, '_payload', None_16294)
        
        # Assigning a Name to a Attribute (line 111):
        
        # Assigning a Name to a Attribute (line 111):
        # Getting the type of 'None' (line 111)
        None_16296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'None')
        # Getting the type of 'self' (line 111)
        self_16297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Setting the type of the member '_charset' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_16297, '_charset', None_16296)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Name to a Attribute (line 113):
        # Getting the type of 'None' (line 113)
        None_16298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 40), 'None')
        # Getting the type of 'self' (line 113)
        self_16299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'self')
        # Setting the type of the member 'epilogue' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 24), self_16299, 'epilogue', None_16298)
        
        # Assigning a Attribute to a Attribute (line 113):
        # Getting the type of 'self' (line 113)
        self_16300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'self')
        # Obtaining the member 'epilogue' of a type (line 113)
        epilogue_16301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 24), self_16300, 'epilogue')
        # Getting the type of 'self' (line 113)
        self_16302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'preamble' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_16302, 'preamble', epilogue_16301)
        
        # Assigning a List to a Attribute (line 114):
        
        # Assigning a List to a Attribute (line 114):
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_16303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        
        # Getting the type of 'self' (line 114)
        self_16304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Setting the type of the member 'defects' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_16304, 'defects', list_16303)
        
        # Assigning a Str to a Attribute (line 116):
        
        # Assigning a Str to a Attribute (line 116):
        str_16305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 29), 'str', 'text/plain')
        # Getting the type of 'self' (line 116)
        self_16306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self')
        # Setting the type of the member '_default_type' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_16306, '_default_type', str_16305)
        
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
        module_type_store = module_type_store.open_function_context('__str__', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Message.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Message.stypy__str__')
        Message.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Message.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        str_16307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, (-1)), 'str', 'Return the entire formatted message as a string.\n        This includes the headers, body, and envelope header.\n        ')
        
        # Call to as_string(...): (line 122)
        # Processing the call keyword arguments (line 122)
        # Getting the type of 'True' (line 122)
        True_16310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'True', False)
        keyword_16311 = True_16310
        kwargs_16312 = {'unixfrom': keyword_16311}
        # Getting the type of 'self' (line 122)
        self_16308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'self', False)
        # Obtaining the member 'as_string' of a type (line 122)
        as_string_16309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 15), self_16308, 'as_string')
        # Calling as_string(args, kwargs) (line 122)
        as_string_call_result_16313 = invoke(stypy.reporting.localization.Localization(__file__, 122, 15), as_string_16309, *[], **kwargs_16312)
        
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', as_string_call_result_16313)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_16314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16314)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_16314


    @norecursion
    def as_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 124)
        False_16315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 33), 'False')
        defaults = [False_16315]
        # Create a new context for function 'as_string'
        module_type_store = module_type_store.open_function_context('as_string', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.as_string.__dict__.__setitem__('stypy_localization', localization)
        Message.as_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.as_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.as_string.__dict__.__setitem__('stypy_function_name', 'Message.as_string')
        Message.as_string.__dict__.__setitem__('stypy_param_names_list', ['unixfrom'])
        Message.as_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.as_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.as_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.as_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.as_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.as_string.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.as_string', ['unixfrom'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'as_string', localization, ['unixfrom'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'as_string(...)' code ##################

        str_16316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, (-1)), 'str', 'Return the entire formatted message as a string.\n        Optional `unixfrom\' when True, means include the Unix From_ envelope\n        header.\n\n        This is a convenience method and may not generate the message exactly\n        as you intend because by default it mangles lines that begin with\n        "From ".  For more flexibility, use the flatten() method of a\n        Generator instance.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 134, 8))
        
        # 'from email.generator import Generator' statement (line 134)
        update_path_to_current_file_folder('C:/Python27/lib/email/')
        import_16317 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 134, 8), 'email.generator')

        if (type(import_16317) is not StypyTypeError):

            if (import_16317 != 'pyd_module'):
                __import__(import_16317)
                sys_modules_16318 = sys.modules[import_16317]
                import_from_module(stypy.reporting.localization.Localization(__file__, 134, 8), 'email.generator', sys_modules_16318.module_type_store, module_type_store, ['Generator'])
                nest_module(stypy.reporting.localization.Localization(__file__, 134, 8), __file__, sys_modules_16318, sys_modules_16318.module_type_store, module_type_store)
            else:
                from email.generator import Generator

                import_from_module(stypy.reporting.localization.Localization(__file__, 134, 8), 'email.generator', None, module_type_store, ['Generator'], [Generator])

        else:
            # Assigning a type to the variable 'email.generator' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'email.generator', import_16317)

        remove_current_file_folder_from_path('C:/Python27/lib/email/')
        
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to StringIO(...): (line 135)
        # Processing the call keyword arguments (line 135)
        kwargs_16320 = {}
        # Getting the type of 'StringIO' (line 135)
        StringIO_16319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 13), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 135)
        StringIO_call_result_16321 = invoke(stypy.reporting.localization.Localization(__file__, 135, 13), StringIO_16319, *[], **kwargs_16320)
        
        # Assigning a type to the variable 'fp' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'fp', StringIO_call_result_16321)
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to Generator(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'fp' (line 136)
        fp_16323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 22), 'fp', False)
        # Processing the call keyword arguments (line 136)
        kwargs_16324 = {}
        # Getting the type of 'Generator' (line 136)
        Generator_16322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'Generator', False)
        # Calling Generator(args, kwargs) (line 136)
        Generator_call_result_16325 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), Generator_16322, *[fp_16323], **kwargs_16324)
        
        # Assigning a type to the variable 'g' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'g', Generator_call_result_16325)
        
        # Call to flatten(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'self' (line 137)
        self_16328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 18), 'self', False)
        # Processing the call keyword arguments (line 137)
        # Getting the type of 'unixfrom' (line 137)
        unixfrom_16329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'unixfrom', False)
        keyword_16330 = unixfrom_16329
        kwargs_16331 = {'unixfrom': keyword_16330}
        # Getting the type of 'g' (line 137)
        g_16326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'g', False)
        # Obtaining the member 'flatten' of a type (line 137)
        flatten_16327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), g_16326, 'flatten')
        # Calling flatten(args, kwargs) (line 137)
        flatten_call_result_16332 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), flatten_16327, *[self_16328], **kwargs_16331)
        
        
        # Call to getvalue(...): (line 138)
        # Processing the call keyword arguments (line 138)
        kwargs_16335 = {}
        # Getting the type of 'fp' (line 138)
        fp_16333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'fp', False)
        # Obtaining the member 'getvalue' of a type (line 138)
        getvalue_16334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), fp_16333, 'getvalue')
        # Calling getvalue(args, kwargs) (line 138)
        getvalue_call_result_16336 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), getvalue_16334, *[], **kwargs_16335)
        
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', getvalue_call_result_16336)
        
        # ################# End of 'as_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'as_string' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_16337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16337)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'as_string'
        return stypy_return_type_16337


    @norecursion
    def is_multipart(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_multipart'
        module_type_store = module_type_store.open_function_context('is_multipart', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.is_multipart.__dict__.__setitem__('stypy_localization', localization)
        Message.is_multipart.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.is_multipart.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.is_multipart.__dict__.__setitem__('stypy_function_name', 'Message.is_multipart')
        Message.is_multipart.__dict__.__setitem__('stypy_param_names_list', [])
        Message.is_multipart.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.is_multipart.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.is_multipart.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.is_multipart.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.is_multipart.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.is_multipart.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.is_multipart', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_multipart', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_multipart(...)' code ##################

        str_16338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'str', 'Return True if the message consists of multiple parts.')
        
        # Call to isinstance(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_16340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'self', False)
        # Obtaining the member '_payload' of a type (line 142)
        _payload_16341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 26), self_16340, '_payload')
        # Getting the type of 'list' (line 142)
        list_16342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 41), 'list', False)
        # Processing the call keyword arguments (line 142)
        kwargs_16343 = {}
        # Getting the type of 'isinstance' (line 142)
        isinstance_16339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 142)
        isinstance_call_result_16344 = invoke(stypy.reporting.localization.Localization(__file__, 142, 15), isinstance_16339, *[_payload_16341, list_16342], **kwargs_16343)
        
        # Assigning a type to the variable 'stypy_return_type' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type', isinstance_call_result_16344)
        
        # ################# End of 'is_multipart(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_multipart' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_16345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16345)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_multipart'
        return stypy_return_type_16345


    @norecursion
    def set_unixfrom(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_unixfrom'
        module_type_store = module_type_store.open_function_context('set_unixfrom', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.set_unixfrom.__dict__.__setitem__('stypy_localization', localization)
        Message.set_unixfrom.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.set_unixfrom.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.set_unixfrom.__dict__.__setitem__('stypy_function_name', 'Message.set_unixfrom')
        Message.set_unixfrom.__dict__.__setitem__('stypy_param_names_list', ['unixfrom'])
        Message.set_unixfrom.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.set_unixfrom.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.set_unixfrom.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.set_unixfrom.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.set_unixfrom.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.set_unixfrom.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.set_unixfrom', ['unixfrom'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_unixfrom', localization, ['unixfrom'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_unixfrom(...)' code ##################

        
        # Assigning a Name to a Attribute (line 148):
        
        # Assigning a Name to a Attribute (line 148):
        # Getting the type of 'unixfrom' (line 148)
        unixfrom_16346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'unixfrom')
        # Getting the type of 'self' (line 148)
        self_16347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self')
        # Setting the type of the member '_unixfrom' of a type (line 148)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_16347, '_unixfrom', unixfrom_16346)
        
        # ################# End of 'set_unixfrom(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_unixfrom' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_16348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16348)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_unixfrom'
        return stypy_return_type_16348


    @norecursion
    def get_unixfrom(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_unixfrom'
        module_type_store = module_type_store.open_function_context('get_unixfrom', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_unixfrom.__dict__.__setitem__('stypy_localization', localization)
        Message.get_unixfrom.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_unixfrom.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_unixfrom.__dict__.__setitem__('stypy_function_name', 'Message.get_unixfrom')
        Message.get_unixfrom.__dict__.__setitem__('stypy_param_names_list', [])
        Message.get_unixfrom.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_unixfrom.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_unixfrom.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_unixfrom.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_unixfrom.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_unixfrom.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_unixfrom', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_unixfrom', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_unixfrom(...)' code ##################

        # Getting the type of 'self' (line 151)
        self_16349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'self')
        # Obtaining the member '_unixfrom' of a type (line 151)
        _unixfrom_16350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 15), self_16349, '_unixfrom')
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', _unixfrom_16350)
        
        # ################# End of 'get_unixfrom(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_unixfrom' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_16351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_unixfrom'
        return stypy_return_type_16351


    @norecursion
    def attach(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'attach'
        module_type_store = module_type_store.open_function_context('attach', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.attach.__dict__.__setitem__('stypy_localization', localization)
        Message.attach.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.attach.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.attach.__dict__.__setitem__('stypy_function_name', 'Message.attach')
        Message.attach.__dict__.__setitem__('stypy_param_names_list', ['payload'])
        Message.attach.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.attach.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.attach.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.attach.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.attach.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.attach.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.attach', ['payload'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'attach', localization, ['payload'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'attach(...)' code ##################

        str_16352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, (-1)), 'str', 'Add the given payload to the current payload.\n\n        The current payload will always be a list of objects after this method\n        is called.  If you want to set the payload to a scalar object, use\n        set_payload() instead.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 163)
        # Getting the type of 'self' (line 163)
        self_16353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'self')
        # Obtaining the member '_payload' of a type (line 163)
        _payload_16354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 11), self_16353, '_payload')
        # Getting the type of 'None' (line 163)
        None_16355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'None')
        
        (may_be_16356, more_types_in_union_16357) = may_be_none(_payload_16354, None_16355)

        if may_be_16356:

            if more_types_in_union_16357:
                # Runtime conditional SSA (line 163)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 164):
            
            # Assigning a List to a Attribute (line 164):
            
            # Obtaining an instance of the builtin type 'list' (line 164)
            list_16358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 164)
            # Adding element type (line 164)
            # Getting the type of 'payload' (line 164)
            payload_16359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 29), 'payload')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), list_16358, payload_16359)
            
            # Getting the type of 'self' (line 164)
            self_16360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'self')
            # Setting the type of the member '_payload' of a type (line 164)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), self_16360, '_payload', list_16358)

            if more_types_in_union_16357:
                # Runtime conditional SSA for else branch (line 163)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_16356) or more_types_in_union_16357):
            
            # Call to append(...): (line 166)
            # Processing the call arguments (line 166)
            # Getting the type of 'payload' (line 166)
            payload_16364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 33), 'payload', False)
            # Processing the call keyword arguments (line 166)
            kwargs_16365 = {}
            # Getting the type of 'self' (line 166)
            self_16361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'self', False)
            # Obtaining the member '_payload' of a type (line 166)
            _payload_16362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), self_16361, '_payload')
            # Obtaining the member 'append' of a type (line 166)
            append_16363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), _payload_16362, 'append')
            # Calling append(args, kwargs) (line 166)
            append_call_result_16366 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), append_16363, *[payload_16364], **kwargs_16365)
            

            if (may_be_16356 and more_types_in_union_16357):
                # SSA join for if statement (line 163)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'attach(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'attach' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_16367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16367)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'attach'
        return stypy_return_type_16367


    @norecursion
    def get_payload(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 168)
        None_16368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 'None')
        # Getting the type of 'False' (line 168)
        False_16369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 41), 'False')
        defaults = [None_16368, False_16369]
        # Create a new context for function 'get_payload'
        module_type_store = module_type_store.open_function_context('get_payload', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_payload.__dict__.__setitem__('stypy_localization', localization)
        Message.get_payload.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_payload.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_payload.__dict__.__setitem__('stypy_function_name', 'Message.get_payload')
        Message.get_payload.__dict__.__setitem__('stypy_param_names_list', ['i', 'decode'])
        Message.get_payload.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_payload.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_payload.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_payload.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_payload.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_payload.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_payload', ['i', 'decode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_payload', localization, ['i', 'decode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_payload(...)' code ##################

        str_16370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'str', "Return a reference to the payload.\n\n        The payload will either be a list object or a string.  If you mutate\n        the list object, you modify the message's payload in place.  Optional\n        i returns that index into the payload.\n\n        Optional decode is a flag indicating whether the payload should be\n        decoded or not, according to the Content-Transfer-Encoding header\n        (default is False).\n\n        When True and the message is not a multipart, the payload will be\n        decoded if this header's value is `quoted-printable' or `base64'.  If\n        some other encoding is used, or the header is missing, or if the\n        payload has bogus data (i.e. bogus base64 or uuencoded data), the\n        payload is returned as-is.\n\n        If the message is a multipart and the decode flag is True, then None\n        is returned.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 188)
        # Getting the type of 'i' (line 188)
        i_16371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'i')
        # Getting the type of 'None' (line 188)
        None_16372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'None')
        
        (may_be_16373, more_types_in_union_16374) = may_be_none(i_16371, None_16372)

        if may_be_16373:

            if more_types_in_union_16374:
                # Runtime conditional SSA (line 188)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 189):
            
            # Assigning a Attribute to a Name (line 189):
            # Getting the type of 'self' (line 189)
            self_16375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 22), 'self')
            # Obtaining the member '_payload' of a type (line 189)
            _payload_16376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 22), self_16375, '_payload')
            # Assigning a type to the variable 'payload' (line 189)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'payload', _payload_16376)

            if more_types_in_union_16374:
                # Runtime conditional SSA for else branch (line 188)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_16373) or more_types_in_union_16374):
            
            # Type idiom detected: calculating its left and rigth part (line 190)
            # Getting the type of 'list' (line 190)
            list_16377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 43), 'list')
            # Getting the type of 'self' (line 190)
            self_16378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 28), 'self')
            # Obtaining the member '_payload' of a type (line 190)
            _payload_16379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 28), self_16378, '_payload')
            
            (may_be_16380, more_types_in_union_16381) = may_not_be_subtype(list_16377, _payload_16379)

            if may_be_16380:

                if more_types_in_union_16381:
                    # Runtime conditional SSA (line 190)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'self' (line 190)
                self_16382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 13), 'self')
                # Obtaining the member '_payload' of a type (line 190)
                _payload_16383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 13), self_16382, '_payload')
                # Setting the type of the member '_payload' of a type (line 190)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 13), self_16382, '_payload', remove_subtype_from_union(_payload_16379, list))
                
                # Call to TypeError(...): (line 191)
                # Processing the call arguments (line 191)
                str_16385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 28), 'str', 'Expected list, got %s')
                
                # Call to type(...): (line 191)
                # Processing the call arguments (line 191)
                # Getting the type of 'self' (line 191)
                self_16387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 59), 'self', False)
                # Obtaining the member '_payload' of a type (line 191)
                _payload_16388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 59), self_16387, '_payload')
                # Processing the call keyword arguments (line 191)
                kwargs_16389 = {}
                # Getting the type of 'type' (line 191)
                type_16386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 54), 'type', False)
                # Calling type(args, kwargs) (line 191)
                type_call_result_16390 = invoke(stypy.reporting.localization.Localization(__file__, 191, 54), type_16386, *[_payload_16388], **kwargs_16389)
                
                # Applying the binary operator '%' (line 191)
                result_mod_16391 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 28), '%', str_16385, type_call_result_16390)
                
                # Processing the call keyword arguments (line 191)
                kwargs_16392 = {}
                # Getting the type of 'TypeError' (line 191)
                TypeError_16384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 18), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 191)
                TypeError_call_result_16393 = invoke(stypy.reporting.localization.Localization(__file__, 191, 18), TypeError_16384, *[result_mod_16391], **kwargs_16392)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 191, 12), TypeError_call_result_16393, 'raise parameter', BaseException)

                if more_types_in_union_16381:
                    # Runtime conditional SSA for else branch (line 190)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_16380) or more_types_in_union_16381):
                # Getting the type of 'self' (line 190)
                self_16394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 13), 'self')
                # Obtaining the member '_payload' of a type (line 190)
                _payload_16395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 13), self_16394, '_payload')
                # Setting the type of the member '_payload' of a type (line 190)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 13), self_16394, '_payload', remove_not_subtype_from_union(_payload_16379, list))
                
                # Assigning a Subscript to a Name (line 193):
                
                # Assigning a Subscript to a Name (line 193):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 193)
                i_16396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 36), 'i')
                # Getting the type of 'self' (line 193)
                self_16397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'self')
                # Obtaining the member '_payload' of a type (line 193)
                _payload_16398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 22), self_16397, '_payload')
                # Obtaining the member '__getitem__' of a type (line 193)
                getitem___16399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 22), _payload_16398, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 193)
                subscript_call_result_16400 = invoke(stypy.reporting.localization.Localization(__file__, 193, 22), getitem___16399, i_16396)
                
                # Assigning a type to the variable 'payload' (line 193)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'payload', subscript_call_result_16400)

                if (may_be_16380 and more_types_in_union_16381):
                    # SSA join for if statement (line 190)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_16373 and more_types_in_union_16374):
                # SSA join for if statement (line 188)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'decode' (line 194)
        decode_16401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'decode')
        # Testing if the type of an if condition is none (line 194)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 194, 8), decode_16401):
            pass
        else:
            
            # Testing the type of an if condition (line 194)
            if_condition_16402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), decode_16401)
            # Assigning a type to the variable 'if_condition_16402' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_16402', if_condition_16402)
            # SSA begins for if statement (line 194)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to is_multipart(...): (line 195)
            # Processing the call keyword arguments (line 195)
            kwargs_16405 = {}
            # Getting the type of 'self' (line 195)
            self_16403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'self', False)
            # Obtaining the member 'is_multipart' of a type (line 195)
            is_multipart_16404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 15), self_16403, 'is_multipart')
            # Calling is_multipart(args, kwargs) (line 195)
            is_multipart_call_result_16406 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), is_multipart_16404, *[], **kwargs_16405)
            
            # Testing if the type of an if condition is none (line 195)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 195, 12), is_multipart_call_result_16406):
                pass
            else:
                
                # Testing the type of an if condition (line 195)
                if_condition_16407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 12), is_multipart_call_result_16406)
                # Assigning a type to the variable 'if_condition_16407' (line 195)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'if_condition_16407', if_condition_16407)
                # SSA begins for if statement (line 195)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'None' (line 196)
                None_16408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'None')
                # Assigning a type to the variable 'stypy_return_type' (line 196)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'stypy_return_type', None_16408)
                # SSA join for if statement (line 195)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 197):
            
            # Assigning a Call to a Name (line 197):
            
            # Call to lower(...): (line 197)
            # Processing the call keyword arguments (line 197)
            kwargs_16416 = {}
            
            # Call to get(...): (line 197)
            # Processing the call arguments (line 197)
            str_16411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 27), 'str', 'content-transfer-encoding')
            str_16412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 56), 'str', '')
            # Processing the call keyword arguments (line 197)
            kwargs_16413 = {}
            # Getting the type of 'self' (line 197)
            self_16409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 18), 'self', False)
            # Obtaining the member 'get' of a type (line 197)
            get_16410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 18), self_16409, 'get')
            # Calling get(args, kwargs) (line 197)
            get_call_result_16414 = invoke(stypy.reporting.localization.Localization(__file__, 197, 18), get_16410, *[str_16411, str_16412], **kwargs_16413)
            
            # Obtaining the member 'lower' of a type (line 197)
            lower_16415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 18), get_call_result_16414, 'lower')
            # Calling lower(args, kwargs) (line 197)
            lower_call_result_16417 = invoke(stypy.reporting.localization.Localization(__file__, 197, 18), lower_16415, *[], **kwargs_16416)
            
            # Assigning a type to the variable 'cte' (line 197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'cte', lower_call_result_16417)
            
            # Getting the type of 'cte' (line 198)
            cte_16418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'cte')
            str_16419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 22), 'str', 'quoted-printable')
            # Applying the binary operator '==' (line 198)
            result_eq_16420 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 15), '==', cte_16418, str_16419)
            
            # Testing if the type of an if condition is none (line 198)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 198, 12), result_eq_16420):
                
                # Getting the type of 'cte' (line 200)
                cte_16427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 17), 'cte')
                str_16428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 24), 'str', 'base64')
                # Applying the binary operator '==' (line 200)
                result_eq_16429 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 17), '==', cte_16427, str_16428)
                
                # Testing if the type of an if condition is none (line 200)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 200, 17), result_eq_16429):
                    
                    # Getting the type of 'cte' (line 206)
                    cte_16437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'cte')
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 206)
                    tuple_16438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 206)
                    # Adding element type (line 206)
                    str_16439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'str', 'x-uuencode')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16439)
                    # Adding element type (line 206)
                    str_16440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 39), 'str', 'uuencode')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16440)
                    # Adding element type (line 206)
                    str_16441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 51), 'str', 'uue')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16441)
                    # Adding element type (line 206)
                    str_16442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 58), 'str', 'x-uue')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16442)
                    
                    # Applying the binary operator 'in' (line 206)
                    result_contains_16443 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 17), 'in', cte_16437, tuple_16438)
                    
                    # Testing if the type of an if condition is none (line 206)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 206, 17), result_contains_16443):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 206)
                        if_condition_16444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 17), result_contains_16443)
                        # Assigning a type to the variable 'if_condition_16444' (line 206)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'if_condition_16444', if_condition_16444)
                        # SSA begins for if statement (line 206)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 207):
                        
                        # Assigning a Call to a Name (line 207):
                        
                        # Call to StringIO(...): (line 207)
                        # Processing the call keyword arguments (line 207)
                        kwargs_16446 = {}
                        # Getting the type of 'StringIO' (line 207)
                        StringIO_16445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'StringIO', False)
                        # Calling StringIO(args, kwargs) (line 207)
                        StringIO_call_result_16447 = invoke(stypy.reporting.localization.Localization(__file__, 207, 22), StringIO_16445, *[], **kwargs_16446)
                        
                        # Assigning a type to the variable 'sfp' (line 207)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'sfp', StringIO_call_result_16447)
                        
                        
                        # SSA begins for try-except statement (line 208)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                        
                        # Call to decode(...): (line 209)
                        # Processing the call arguments (line 209)
                        
                        # Call to StringIO(...): (line 209)
                        # Processing the call arguments (line 209)
                        # Getting the type of 'payload' (line 209)
                        payload_16451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 39), 'payload', False)
                        str_16452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 47), 'str', '\n')
                        # Applying the binary operator '+' (line 209)
                        result_add_16453 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 39), '+', payload_16451, str_16452)
                        
                        # Processing the call keyword arguments (line 209)
                        kwargs_16454 = {}
                        # Getting the type of 'StringIO' (line 209)
                        StringIO_16450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'StringIO', False)
                        # Calling StringIO(args, kwargs) (line 209)
                        StringIO_call_result_16455 = invoke(stypy.reporting.localization.Localization(__file__, 209, 30), StringIO_16450, *[result_add_16453], **kwargs_16454)
                        
                        # Getting the type of 'sfp' (line 209)
                        sfp_16456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 54), 'sfp', False)
                        # Processing the call keyword arguments (line 209)
                        # Getting the type of 'True' (line 209)
                        True_16457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 65), 'True', False)
                        keyword_16458 = True_16457
                        kwargs_16459 = {'quiet': keyword_16458}
                        # Getting the type of 'uu' (line 209)
                        uu_16448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'uu', False)
                        # Obtaining the member 'decode' of a type (line 209)
                        decode_16449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), uu_16448, 'decode')
                        # Calling decode(args, kwargs) (line 209)
                        decode_call_result_16460 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), decode_16449, *[StringIO_call_result_16455, sfp_16456], **kwargs_16459)
                        
                        
                        # Assigning a Call to a Name (line 210):
                        
                        # Assigning a Call to a Name (line 210):
                        
                        # Call to getvalue(...): (line 210)
                        # Processing the call keyword arguments (line 210)
                        kwargs_16463 = {}
                        # Getting the type of 'sfp' (line 210)
                        sfp_16461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'sfp', False)
                        # Obtaining the member 'getvalue' of a type (line 210)
                        getvalue_16462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 30), sfp_16461, 'getvalue')
                        # Calling getvalue(args, kwargs) (line 210)
                        getvalue_call_result_16464 = invoke(stypy.reporting.localization.Localization(__file__, 210, 30), getvalue_16462, *[], **kwargs_16463)
                        
                        # Assigning a type to the variable 'payload' (line 210)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'payload', getvalue_call_result_16464)
                        # SSA branch for the except part of a try statement (line 208)
                        # SSA branch for the except 'Attribute' branch of a try statement (line 208)
                        module_type_store.open_ssa_branch('except')
                        # Getting the type of 'payload' (line 213)
                        payload_16465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'payload')
                        # Assigning a type to the variable 'stypy_return_type' (line 213)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'stypy_return_type', payload_16465)
                        # SSA join for try-except statement (line 208)
                        module_type_store = module_type_store.join_ssa_context()
                        
                        # SSA join for if statement (line 206)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 200)
                    if_condition_16430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 17), result_eq_16429)
                    # Assigning a type to the variable 'if_condition_16430' (line 200)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 17), 'if_condition_16430', if_condition_16430)
                    # SSA begins for if statement (line 200)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    
                    # SSA begins for try-except statement (line 201)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                    
                    # Call to _bdecode(...): (line 202)
                    # Processing the call arguments (line 202)
                    # Getting the type of 'payload' (line 202)
                    payload_16433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 42), 'payload', False)
                    # Processing the call keyword arguments (line 202)
                    kwargs_16434 = {}
                    # Getting the type of 'utils' (line 202)
                    utils_16431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'utils', False)
                    # Obtaining the member '_bdecode' of a type (line 202)
                    _bdecode_16432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 27), utils_16431, '_bdecode')
                    # Calling _bdecode(args, kwargs) (line 202)
                    _bdecode_call_result_16435 = invoke(stypy.reporting.localization.Localization(__file__, 202, 27), _bdecode_16432, *[payload_16433], **kwargs_16434)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 202)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'stypy_return_type', _bdecode_call_result_16435)
                    # SSA branch for the except part of a try statement (line 201)
                    # SSA branch for the except 'Attribute' branch of a try statement (line 201)
                    module_type_store.open_ssa_branch('except')
                    # Getting the type of 'payload' (line 205)
                    payload_16436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'payload')
                    # Assigning a type to the variable 'stypy_return_type' (line 205)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'stypy_return_type', payload_16436)
                    # SSA join for try-except statement (line 201)
                    module_type_store = module_type_store.join_ssa_context()
                    
                    # SSA branch for the else part of an if statement (line 200)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'cte' (line 206)
                    cte_16437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'cte')
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 206)
                    tuple_16438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 206)
                    # Adding element type (line 206)
                    str_16439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'str', 'x-uuencode')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16439)
                    # Adding element type (line 206)
                    str_16440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 39), 'str', 'uuencode')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16440)
                    # Adding element type (line 206)
                    str_16441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 51), 'str', 'uue')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16441)
                    # Adding element type (line 206)
                    str_16442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 58), 'str', 'x-uue')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16442)
                    
                    # Applying the binary operator 'in' (line 206)
                    result_contains_16443 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 17), 'in', cte_16437, tuple_16438)
                    
                    # Testing if the type of an if condition is none (line 206)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 206, 17), result_contains_16443):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 206)
                        if_condition_16444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 17), result_contains_16443)
                        # Assigning a type to the variable 'if_condition_16444' (line 206)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'if_condition_16444', if_condition_16444)
                        # SSA begins for if statement (line 206)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 207):
                        
                        # Assigning a Call to a Name (line 207):
                        
                        # Call to StringIO(...): (line 207)
                        # Processing the call keyword arguments (line 207)
                        kwargs_16446 = {}
                        # Getting the type of 'StringIO' (line 207)
                        StringIO_16445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'StringIO', False)
                        # Calling StringIO(args, kwargs) (line 207)
                        StringIO_call_result_16447 = invoke(stypy.reporting.localization.Localization(__file__, 207, 22), StringIO_16445, *[], **kwargs_16446)
                        
                        # Assigning a type to the variable 'sfp' (line 207)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'sfp', StringIO_call_result_16447)
                        
                        
                        # SSA begins for try-except statement (line 208)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                        
                        # Call to decode(...): (line 209)
                        # Processing the call arguments (line 209)
                        
                        # Call to StringIO(...): (line 209)
                        # Processing the call arguments (line 209)
                        # Getting the type of 'payload' (line 209)
                        payload_16451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 39), 'payload', False)
                        str_16452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 47), 'str', '\n')
                        # Applying the binary operator '+' (line 209)
                        result_add_16453 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 39), '+', payload_16451, str_16452)
                        
                        # Processing the call keyword arguments (line 209)
                        kwargs_16454 = {}
                        # Getting the type of 'StringIO' (line 209)
                        StringIO_16450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'StringIO', False)
                        # Calling StringIO(args, kwargs) (line 209)
                        StringIO_call_result_16455 = invoke(stypy.reporting.localization.Localization(__file__, 209, 30), StringIO_16450, *[result_add_16453], **kwargs_16454)
                        
                        # Getting the type of 'sfp' (line 209)
                        sfp_16456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 54), 'sfp', False)
                        # Processing the call keyword arguments (line 209)
                        # Getting the type of 'True' (line 209)
                        True_16457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 65), 'True', False)
                        keyword_16458 = True_16457
                        kwargs_16459 = {'quiet': keyword_16458}
                        # Getting the type of 'uu' (line 209)
                        uu_16448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'uu', False)
                        # Obtaining the member 'decode' of a type (line 209)
                        decode_16449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), uu_16448, 'decode')
                        # Calling decode(args, kwargs) (line 209)
                        decode_call_result_16460 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), decode_16449, *[StringIO_call_result_16455, sfp_16456], **kwargs_16459)
                        
                        
                        # Assigning a Call to a Name (line 210):
                        
                        # Assigning a Call to a Name (line 210):
                        
                        # Call to getvalue(...): (line 210)
                        # Processing the call keyword arguments (line 210)
                        kwargs_16463 = {}
                        # Getting the type of 'sfp' (line 210)
                        sfp_16461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'sfp', False)
                        # Obtaining the member 'getvalue' of a type (line 210)
                        getvalue_16462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 30), sfp_16461, 'getvalue')
                        # Calling getvalue(args, kwargs) (line 210)
                        getvalue_call_result_16464 = invoke(stypy.reporting.localization.Localization(__file__, 210, 30), getvalue_16462, *[], **kwargs_16463)
                        
                        # Assigning a type to the variable 'payload' (line 210)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'payload', getvalue_call_result_16464)
                        # SSA branch for the except part of a try statement (line 208)
                        # SSA branch for the except 'Attribute' branch of a try statement (line 208)
                        module_type_store.open_ssa_branch('except')
                        # Getting the type of 'payload' (line 213)
                        payload_16465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'payload')
                        # Assigning a type to the variable 'stypy_return_type' (line 213)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'stypy_return_type', payload_16465)
                        # SSA join for try-except statement (line 208)
                        module_type_store = module_type_store.join_ssa_context()
                        
                        # SSA join for if statement (line 206)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 200)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 198)
                if_condition_16421 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 12), result_eq_16420)
                # Assigning a type to the variable 'if_condition_16421' (line 198)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'if_condition_16421', if_condition_16421)
                # SSA begins for if statement (line 198)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to _qdecode(...): (line 199)
                # Processing the call arguments (line 199)
                # Getting the type of 'payload' (line 199)
                payload_16424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 38), 'payload', False)
                # Processing the call keyword arguments (line 199)
                kwargs_16425 = {}
                # Getting the type of 'utils' (line 199)
                utils_16422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'utils', False)
                # Obtaining the member '_qdecode' of a type (line 199)
                _qdecode_16423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 23), utils_16422, '_qdecode')
                # Calling _qdecode(args, kwargs) (line 199)
                _qdecode_call_result_16426 = invoke(stypy.reporting.localization.Localization(__file__, 199, 23), _qdecode_16423, *[payload_16424], **kwargs_16425)
                
                # Assigning a type to the variable 'stypy_return_type' (line 199)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'stypy_return_type', _qdecode_call_result_16426)
                # SSA branch for the else part of an if statement (line 198)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'cte' (line 200)
                cte_16427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 17), 'cte')
                str_16428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 24), 'str', 'base64')
                # Applying the binary operator '==' (line 200)
                result_eq_16429 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 17), '==', cte_16427, str_16428)
                
                # Testing if the type of an if condition is none (line 200)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 200, 17), result_eq_16429):
                    
                    # Getting the type of 'cte' (line 206)
                    cte_16437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'cte')
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 206)
                    tuple_16438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 206)
                    # Adding element type (line 206)
                    str_16439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'str', 'x-uuencode')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16439)
                    # Adding element type (line 206)
                    str_16440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 39), 'str', 'uuencode')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16440)
                    # Adding element type (line 206)
                    str_16441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 51), 'str', 'uue')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16441)
                    # Adding element type (line 206)
                    str_16442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 58), 'str', 'x-uue')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16442)
                    
                    # Applying the binary operator 'in' (line 206)
                    result_contains_16443 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 17), 'in', cte_16437, tuple_16438)
                    
                    # Testing if the type of an if condition is none (line 206)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 206, 17), result_contains_16443):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 206)
                        if_condition_16444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 17), result_contains_16443)
                        # Assigning a type to the variable 'if_condition_16444' (line 206)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'if_condition_16444', if_condition_16444)
                        # SSA begins for if statement (line 206)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 207):
                        
                        # Assigning a Call to a Name (line 207):
                        
                        # Call to StringIO(...): (line 207)
                        # Processing the call keyword arguments (line 207)
                        kwargs_16446 = {}
                        # Getting the type of 'StringIO' (line 207)
                        StringIO_16445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'StringIO', False)
                        # Calling StringIO(args, kwargs) (line 207)
                        StringIO_call_result_16447 = invoke(stypy.reporting.localization.Localization(__file__, 207, 22), StringIO_16445, *[], **kwargs_16446)
                        
                        # Assigning a type to the variable 'sfp' (line 207)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'sfp', StringIO_call_result_16447)
                        
                        
                        # SSA begins for try-except statement (line 208)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                        
                        # Call to decode(...): (line 209)
                        # Processing the call arguments (line 209)
                        
                        # Call to StringIO(...): (line 209)
                        # Processing the call arguments (line 209)
                        # Getting the type of 'payload' (line 209)
                        payload_16451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 39), 'payload', False)
                        str_16452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 47), 'str', '\n')
                        # Applying the binary operator '+' (line 209)
                        result_add_16453 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 39), '+', payload_16451, str_16452)
                        
                        # Processing the call keyword arguments (line 209)
                        kwargs_16454 = {}
                        # Getting the type of 'StringIO' (line 209)
                        StringIO_16450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'StringIO', False)
                        # Calling StringIO(args, kwargs) (line 209)
                        StringIO_call_result_16455 = invoke(stypy.reporting.localization.Localization(__file__, 209, 30), StringIO_16450, *[result_add_16453], **kwargs_16454)
                        
                        # Getting the type of 'sfp' (line 209)
                        sfp_16456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 54), 'sfp', False)
                        # Processing the call keyword arguments (line 209)
                        # Getting the type of 'True' (line 209)
                        True_16457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 65), 'True', False)
                        keyword_16458 = True_16457
                        kwargs_16459 = {'quiet': keyword_16458}
                        # Getting the type of 'uu' (line 209)
                        uu_16448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'uu', False)
                        # Obtaining the member 'decode' of a type (line 209)
                        decode_16449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), uu_16448, 'decode')
                        # Calling decode(args, kwargs) (line 209)
                        decode_call_result_16460 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), decode_16449, *[StringIO_call_result_16455, sfp_16456], **kwargs_16459)
                        
                        
                        # Assigning a Call to a Name (line 210):
                        
                        # Assigning a Call to a Name (line 210):
                        
                        # Call to getvalue(...): (line 210)
                        # Processing the call keyword arguments (line 210)
                        kwargs_16463 = {}
                        # Getting the type of 'sfp' (line 210)
                        sfp_16461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'sfp', False)
                        # Obtaining the member 'getvalue' of a type (line 210)
                        getvalue_16462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 30), sfp_16461, 'getvalue')
                        # Calling getvalue(args, kwargs) (line 210)
                        getvalue_call_result_16464 = invoke(stypy.reporting.localization.Localization(__file__, 210, 30), getvalue_16462, *[], **kwargs_16463)
                        
                        # Assigning a type to the variable 'payload' (line 210)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'payload', getvalue_call_result_16464)
                        # SSA branch for the except part of a try statement (line 208)
                        # SSA branch for the except 'Attribute' branch of a try statement (line 208)
                        module_type_store.open_ssa_branch('except')
                        # Getting the type of 'payload' (line 213)
                        payload_16465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'payload')
                        # Assigning a type to the variable 'stypy_return_type' (line 213)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'stypy_return_type', payload_16465)
                        # SSA join for try-except statement (line 208)
                        module_type_store = module_type_store.join_ssa_context()
                        
                        # SSA join for if statement (line 206)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 200)
                    if_condition_16430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 17), result_eq_16429)
                    # Assigning a type to the variable 'if_condition_16430' (line 200)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 17), 'if_condition_16430', if_condition_16430)
                    # SSA begins for if statement (line 200)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    
                    # SSA begins for try-except statement (line 201)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                    
                    # Call to _bdecode(...): (line 202)
                    # Processing the call arguments (line 202)
                    # Getting the type of 'payload' (line 202)
                    payload_16433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 42), 'payload', False)
                    # Processing the call keyword arguments (line 202)
                    kwargs_16434 = {}
                    # Getting the type of 'utils' (line 202)
                    utils_16431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'utils', False)
                    # Obtaining the member '_bdecode' of a type (line 202)
                    _bdecode_16432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 27), utils_16431, '_bdecode')
                    # Calling _bdecode(args, kwargs) (line 202)
                    _bdecode_call_result_16435 = invoke(stypy.reporting.localization.Localization(__file__, 202, 27), _bdecode_16432, *[payload_16433], **kwargs_16434)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 202)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'stypy_return_type', _bdecode_call_result_16435)
                    # SSA branch for the except part of a try statement (line 201)
                    # SSA branch for the except 'Attribute' branch of a try statement (line 201)
                    module_type_store.open_ssa_branch('except')
                    # Getting the type of 'payload' (line 205)
                    payload_16436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'payload')
                    # Assigning a type to the variable 'stypy_return_type' (line 205)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'stypy_return_type', payload_16436)
                    # SSA join for try-except statement (line 201)
                    module_type_store = module_type_store.join_ssa_context()
                    
                    # SSA branch for the else part of an if statement (line 200)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'cte' (line 206)
                    cte_16437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'cte')
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 206)
                    tuple_16438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 206)
                    # Adding element type (line 206)
                    str_16439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'str', 'x-uuencode')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16439)
                    # Adding element type (line 206)
                    str_16440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 39), 'str', 'uuencode')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16440)
                    # Adding element type (line 206)
                    str_16441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 51), 'str', 'uue')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16441)
                    # Adding element type (line 206)
                    str_16442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 58), 'str', 'x-uue')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 25), tuple_16438, str_16442)
                    
                    # Applying the binary operator 'in' (line 206)
                    result_contains_16443 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 17), 'in', cte_16437, tuple_16438)
                    
                    # Testing if the type of an if condition is none (line 206)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 206, 17), result_contains_16443):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 206)
                        if_condition_16444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 17), result_contains_16443)
                        # Assigning a type to the variable 'if_condition_16444' (line 206)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 17), 'if_condition_16444', if_condition_16444)
                        # SSA begins for if statement (line 206)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 207):
                        
                        # Assigning a Call to a Name (line 207):
                        
                        # Call to StringIO(...): (line 207)
                        # Processing the call keyword arguments (line 207)
                        kwargs_16446 = {}
                        # Getting the type of 'StringIO' (line 207)
                        StringIO_16445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'StringIO', False)
                        # Calling StringIO(args, kwargs) (line 207)
                        StringIO_call_result_16447 = invoke(stypy.reporting.localization.Localization(__file__, 207, 22), StringIO_16445, *[], **kwargs_16446)
                        
                        # Assigning a type to the variable 'sfp' (line 207)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'sfp', StringIO_call_result_16447)
                        
                        
                        # SSA begins for try-except statement (line 208)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                        
                        # Call to decode(...): (line 209)
                        # Processing the call arguments (line 209)
                        
                        # Call to StringIO(...): (line 209)
                        # Processing the call arguments (line 209)
                        # Getting the type of 'payload' (line 209)
                        payload_16451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 39), 'payload', False)
                        str_16452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 47), 'str', '\n')
                        # Applying the binary operator '+' (line 209)
                        result_add_16453 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 39), '+', payload_16451, str_16452)
                        
                        # Processing the call keyword arguments (line 209)
                        kwargs_16454 = {}
                        # Getting the type of 'StringIO' (line 209)
                        StringIO_16450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'StringIO', False)
                        # Calling StringIO(args, kwargs) (line 209)
                        StringIO_call_result_16455 = invoke(stypy.reporting.localization.Localization(__file__, 209, 30), StringIO_16450, *[result_add_16453], **kwargs_16454)
                        
                        # Getting the type of 'sfp' (line 209)
                        sfp_16456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 54), 'sfp', False)
                        # Processing the call keyword arguments (line 209)
                        # Getting the type of 'True' (line 209)
                        True_16457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 65), 'True', False)
                        keyword_16458 = True_16457
                        kwargs_16459 = {'quiet': keyword_16458}
                        # Getting the type of 'uu' (line 209)
                        uu_16448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'uu', False)
                        # Obtaining the member 'decode' of a type (line 209)
                        decode_16449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), uu_16448, 'decode')
                        # Calling decode(args, kwargs) (line 209)
                        decode_call_result_16460 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), decode_16449, *[StringIO_call_result_16455, sfp_16456], **kwargs_16459)
                        
                        
                        # Assigning a Call to a Name (line 210):
                        
                        # Assigning a Call to a Name (line 210):
                        
                        # Call to getvalue(...): (line 210)
                        # Processing the call keyword arguments (line 210)
                        kwargs_16463 = {}
                        # Getting the type of 'sfp' (line 210)
                        sfp_16461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'sfp', False)
                        # Obtaining the member 'getvalue' of a type (line 210)
                        getvalue_16462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 30), sfp_16461, 'getvalue')
                        # Calling getvalue(args, kwargs) (line 210)
                        getvalue_call_result_16464 = invoke(stypy.reporting.localization.Localization(__file__, 210, 30), getvalue_16462, *[], **kwargs_16463)
                        
                        # Assigning a type to the variable 'payload' (line 210)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'payload', getvalue_call_result_16464)
                        # SSA branch for the except part of a try statement (line 208)
                        # SSA branch for the except 'Attribute' branch of a try statement (line 208)
                        module_type_store.open_ssa_branch('except')
                        # Getting the type of 'payload' (line 213)
                        payload_16465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'payload')
                        # Assigning a type to the variable 'stypy_return_type' (line 213)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'stypy_return_type', payload_16465)
                        # SSA join for try-except statement (line 208)
                        module_type_store = module_type_store.join_ssa_context()
                        
                        # SSA join for if statement (line 206)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 200)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 198)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 194)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'payload' (line 216)
        payload_16466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'payload')
        # Assigning a type to the variable 'stypy_return_type' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', payload_16466)
        
        # ################# End of 'get_payload(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_payload' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_16467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_payload'
        return stypy_return_type_16467


    @norecursion
    def set_payload(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 218)
        None_16468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 43), 'None')
        defaults = [None_16468]
        # Create a new context for function 'set_payload'
        module_type_store = module_type_store.open_function_context('set_payload', 218, 4, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.set_payload.__dict__.__setitem__('stypy_localization', localization)
        Message.set_payload.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.set_payload.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.set_payload.__dict__.__setitem__('stypy_function_name', 'Message.set_payload')
        Message.set_payload.__dict__.__setitem__('stypy_param_names_list', ['payload', 'charset'])
        Message.set_payload.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.set_payload.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.set_payload.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.set_payload.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.set_payload.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.set_payload.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.set_payload', ['payload', 'charset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_payload', localization, ['payload', 'charset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_payload(...)' code ##################

        str_16469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, (-1)), 'str', "Set the payload to the given value.\n\n        Optional charset sets the message's default character set.  See\n        set_charset() for details.\n        ")
        
        # Assigning a Name to a Attribute (line 224):
        
        # Assigning a Name to a Attribute (line 224):
        # Getting the type of 'payload' (line 224)
        payload_16470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 'payload')
        # Getting the type of 'self' (line 224)
        self_16471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'self')
        # Setting the type of the member '_payload' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), self_16471, '_payload', payload_16470)
        
        # Type idiom detected: calculating its left and rigth part (line 225)
        # Getting the type of 'charset' (line 225)
        charset_16472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'charset')
        # Getting the type of 'None' (line 225)
        None_16473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 26), 'None')
        
        (may_be_16474, more_types_in_union_16475) = may_not_be_none(charset_16472, None_16473)

        if may_be_16474:

            if more_types_in_union_16475:
                # Runtime conditional SSA (line 225)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to set_charset(...): (line 226)
            # Processing the call arguments (line 226)
            # Getting the type of 'charset' (line 226)
            charset_16478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'charset', False)
            # Processing the call keyword arguments (line 226)
            kwargs_16479 = {}
            # Getting the type of 'self' (line 226)
            self_16476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'self', False)
            # Obtaining the member 'set_charset' of a type (line 226)
            set_charset_16477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), self_16476, 'set_charset')
            # Calling set_charset(args, kwargs) (line 226)
            set_charset_call_result_16480 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), set_charset_16477, *[charset_16478], **kwargs_16479)
            

            if more_types_in_union_16475:
                # SSA join for if statement (line 225)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'set_payload(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_payload' in the type store
        # Getting the type of 'stypy_return_type' (line 218)
        stypy_return_type_16481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16481)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_payload'
        return stypy_return_type_16481


    @norecursion
    def set_charset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_charset'
        module_type_store = module_type_store.open_function_context('set_charset', 228, 4, False)
        # Assigning a type to the variable 'self' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.set_charset.__dict__.__setitem__('stypy_localization', localization)
        Message.set_charset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.set_charset.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.set_charset.__dict__.__setitem__('stypy_function_name', 'Message.set_charset')
        Message.set_charset.__dict__.__setitem__('stypy_param_names_list', ['charset'])
        Message.set_charset.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.set_charset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.set_charset.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.set_charset.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.set_charset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.set_charset.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.set_charset', ['charset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_charset', localization, ['charset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_charset(...)' code ##################

        str_16482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, (-1)), 'str', 'Set the charset of the payload to a given character set.\n\n        charset can be a Charset instance, a string naming a character set, or\n        None.  If it is a string it will be converted to a Charset instance.\n        If charset is None, the charset parameter will be removed from the\n        Content-Type field.  Anything else will generate a TypeError.\n\n        The message will be assumed to be of type text/* encoded with\n        charset.input_charset.  It will be converted to charset.output_charset\n        and encoded properly, if needed, when generating the plain text\n        representation of the message.  MIME headers (MIME-Version,\n        Content-Type, Content-Transfer-Encoding) will be added as needed.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 243)
        # Getting the type of 'charset' (line 243)
        charset_16483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'charset')
        # Getting the type of 'None' (line 243)
        None_16484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 22), 'None')
        
        (may_be_16485, more_types_in_union_16486) = may_be_none(charset_16483, None_16484)

        if may_be_16485:

            if more_types_in_union_16486:
                # Runtime conditional SSA (line 243)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to del_param(...): (line 244)
            # Processing the call arguments (line 244)
            str_16489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 27), 'str', 'charset')
            # Processing the call keyword arguments (line 244)
            kwargs_16490 = {}
            # Getting the type of 'self' (line 244)
            self_16487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'self', False)
            # Obtaining the member 'del_param' of a type (line 244)
            del_param_16488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), self_16487, 'del_param')
            # Calling del_param(args, kwargs) (line 244)
            del_param_call_result_16491 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), del_param_16488, *[str_16489], **kwargs_16490)
            
            
            # Assigning a Name to a Attribute (line 245):
            
            # Assigning a Name to a Attribute (line 245):
            # Getting the type of 'None' (line 245)
            None_16492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'None')
            # Getting the type of 'self' (line 245)
            self_16493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'self')
            # Setting the type of the member '_charset' of a type (line 245)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 12), self_16493, '_charset', None_16492)
            # Assigning a type to the variable 'stypy_return_type' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_16486:
                # SSA join for if statement (line 243)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'charset' (line 243)
        charset_16494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'charset')
        # Assigning a type to the variable 'charset' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'charset', remove_type_from_union(charset_16494, types.NoneType))
        
        # Type idiom detected: calculating its left and rigth part (line 247)
        # Getting the type of 'basestring' (line 247)
        basestring_16495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 31), 'basestring')
        # Getting the type of 'charset' (line 247)
        charset_16496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'charset')
        
        (may_be_16497, more_types_in_union_16498) = may_be_subtype(basestring_16495, charset_16496)

        if may_be_16497:

            if more_types_in_union_16498:
                # Runtime conditional SSA (line 247)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'charset' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'charset', remove_not_subtype_from_union(charset_16496, basestring))
            
            # Assigning a Call to a Name (line 248):
            
            # Assigning a Call to a Name (line 248):
            
            # Call to Charset(...): (line 248)
            # Processing the call arguments (line 248)
            # Getting the type of 'charset' (line 248)
            charset_16502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 44), 'charset', False)
            # Processing the call keyword arguments (line 248)
            kwargs_16503 = {}
            # Getting the type of 'email' (line 248)
            email_16499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'email', False)
            # Obtaining the member 'charset' of a type (line 248)
            charset_16500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 22), email_16499, 'charset')
            # Obtaining the member 'Charset' of a type (line 248)
            Charset_16501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 22), charset_16500, 'Charset')
            # Calling Charset(args, kwargs) (line 248)
            Charset_call_result_16504 = invoke(stypy.reporting.localization.Localization(__file__, 248, 22), Charset_16501, *[charset_16502], **kwargs_16503)
            
            # Assigning a type to the variable 'charset' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'charset', Charset_call_result_16504)

            if more_types_in_union_16498:
                # SSA join for if statement (line 247)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to isinstance(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'charset' (line 249)
        charset_16506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 26), 'charset', False)
        # Getting the type of 'email' (line 249)
        email_16507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 35), 'email', False)
        # Obtaining the member 'charset' of a type (line 249)
        charset_16508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 35), email_16507, 'charset')
        # Obtaining the member 'Charset' of a type (line 249)
        Charset_16509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 35), charset_16508, 'Charset')
        # Processing the call keyword arguments (line 249)
        kwargs_16510 = {}
        # Getting the type of 'isinstance' (line 249)
        isinstance_16505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 249)
        isinstance_call_result_16511 = invoke(stypy.reporting.localization.Localization(__file__, 249, 15), isinstance_16505, *[charset_16506, Charset_16509], **kwargs_16510)
        
        # Applying the 'not' unary operator (line 249)
        result_not__16512 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 11), 'not', isinstance_call_result_16511)
        
        # Testing if the type of an if condition is none (line 249)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 249, 8), result_not__16512):
            pass
        else:
            
            # Testing the type of an if condition (line 249)
            if_condition_16513 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 8), result_not__16512)
            # Assigning a type to the variable 'if_condition_16513' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'if_condition_16513', if_condition_16513)
            # SSA begins for if statement (line 249)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 250)
            # Processing the call arguments (line 250)
            # Getting the type of 'charset' (line 250)
            charset_16515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 28), 'charset', False)
            # Processing the call keyword arguments (line 250)
            kwargs_16516 = {}
            # Getting the type of 'TypeError' (line 250)
            TypeError_16514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 250)
            TypeError_call_result_16517 = invoke(stypy.reporting.localization.Localization(__file__, 250, 18), TypeError_16514, *[charset_16515], **kwargs_16516)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 250, 12), TypeError_call_result_16517, 'raise parameter', BaseException)
            # SSA join for if statement (line 249)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Attribute (line 253):
        
        # Assigning a Name to a Attribute (line 253):
        # Getting the type of 'charset' (line 253)
        charset_16518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 24), 'charset')
        # Getting the type of 'self' (line 253)
        self_16519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'self')
        # Setting the type of the member '_charset' of a type (line 253)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), self_16519, '_charset', charset_16518)
        
        str_16520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 11), 'str', 'MIME-Version')
        # Getting the type of 'self' (line 254)
        self_16521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 33), 'self')
        # Applying the binary operator 'notin' (line 254)
        result_contains_16522 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 11), 'notin', str_16520, self_16521)
        
        # Testing if the type of an if condition is none (line 254)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 254, 8), result_contains_16522):
            pass
        else:
            
            # Testing the type of an if condition (line 254)
            if_condition_16523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 8), result_contains_16522)
            # Assigning a type to the variable 'if_condition_16523' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'if_condition_16523', if_condition_16523)
            # SSA begins for if statement (line 254)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add_header(...): (line 255)
            # Processing the call arguments (line 255)
            str_16526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 28), 'str', 'MIME-Version')
            str_16527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 44), 'str', '1.0')
            # Processing the call keyword arguments (line 255)
            kwargs_16528 = {}
            # Getting the type of 'self' (line 255)
            self_16524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'self', False)
            # Obtaining the member 'add_header' of a type (line 255)
            add_header_16525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), self_16524, 'add_header')
            # Calling add_header(args, kwargs) (line 255)
            add_header_call_result_16529 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), add_header_16525, *[str_16526, str_16527], **kwargs_16528)
            
            # SSA join for if statement (line 254)
            module_type_store = module_type_store.join_ssa_context()
            

        
        str_16530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 11), 'str', 'Content-Type')
        # Getting the type of 'self' (line 256)
        self_16531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 33), 'self')
        # Applying the binary operator 'notin' (line 256)
        result_contains_16532 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 11), 'notin', str_16530, self_16531)
        
        # Testing if the type of an if condition is none (line 256)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 256, 8), result_contains_16532):
            
            # Call to set_param(...): (line 260)
            # Processing the call arguments (line 260)
            str_16547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 27), 'str', 'charset')
            
            # Call to get_output_charset(...): (line 260)
            # Processing the call keyword arguments (line 260)
            kwargs_16550 = {}
            # Getting the type of 'charset' (line 260)
            charset_16548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 38), 'charset', False)
            # Obtaining the member 'get_output_charset' of a type (line 260)
            get_output_charset_16549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 38), charset_16548, 'get_output_charset')
            # Calling get_output_charset(args, kwargs) (line 260)
            get_output_charset_call_result_16551 = invoke(stypy.reporting.localization.Localization(__file__, 260, 38), get_output_charset_16549, *[], **kwargs_16550)
            
            # Processing the call keyword arguments (line 260)
            kwargs_16552 = {}
            # Getting the type of 'self' (line 260)
            self_16545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'self', False)
            # Obtaining the member 'set_param' of a type (line 260)
            set_param_16546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), self_16545, 'set_param')
            # Calling set_param(args, kwargs) (line 260)
            set_param_call_result_16553 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), set_param_16546, *[str_16547, get_output_charset_call_result_16551], **kwargs_16552)
            
        else:
            
            # Testing the type of an if condition (line 256)
            if_condition_16533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 8), result_contains_16532)
            # Assigning a type to the variable 'if_condition_16533' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'if_condition_16533', if_condition_16533)
            # SSA begins for if statement (line 256)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add_header(...): (line 257)
            # Processing the call arguments (line 257)
            str_16536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'str', 'Content-Type')
            str_16537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 44), 'str', 'text/plain')
            # Processing the call keyword arguments (line 257)
            
            # Call to get_output_charset(...): (line 258)
            # Processing the call keyword arguments (line 258)
            kwargs_16540 = {}
            # Getting the type of 'charset' (line 258)
            charset_16538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 36), 'charset', False)
            # Obtaining the member 'get_output_charset' of a type (line 258)
            get_output_charset_16539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 36), charset_16538, 'get_output_charset')
            # Calling get_output_charset(args, kwargs) (line 258)
            get_output_charset_call_result_16541 = invoke(stypy.reporting.localization.Localization(__file__, 258, 36), get_output_charset_16539, *[], **kwargs_16540)
            
            keyword_16542 = get_output_charset_call_result_16541
            kwargs_16543 = {'charset': keyword_16542}
            # Getting the type of 'self' (line 257)
            self_16534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'self', False)
            # Obtaining the member 'add_header' of a type (line 257)
            add_header_16535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), self_16534, 'add_header')
            # Calling add_header(args, kwargs) (line 257)
            add_header_call_result_16544 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), add_header_16535, *[str_16536, str_16537], **kwargs_16543)
            
            # SSA branch for the else part of an if statement (line 256)
            module_type_store.open_ssa_branch('else')
            
            # Call to set_param(...): (line 260)
            # Processing the call arguments (line 260)
            str_16547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 27), 'str', 'charset')
            
            # Call to get_output_charset(...): (line 260)
            # Processing the call keyword arguments (line 260)
            kwargs_16550 = {}
            # Getting the type of 'charset' (line 260)
            charset_16548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 38), 'charset', False)
            # Obtaining the member 'get_output_charset' of a type (line 260)
            get_output_charset_16549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 38), charset_16548, 'get_output_charset')
            # Calling get_output_charset(args, kwargs) (line 260)
            get_output_charset_call_result_16551 = invoke(stypy.reporting.localization.Localization(__file__, 260, 38), get_output_charset_16549, *[], **kwargs_16550)
            
            # Processing the call keyword arguments (line 260)
            kwargs_16552 = {}
            # Getting the type of 'self' (line 260)
            self_16545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'self', False)
            # Obtaining the member 'set_param' of a type (line 260)
            set_param_16546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), self_16545, 'set_param')
            # Calling set_param(args, kwargs) (line 260)
            set_param_call_result_16553 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), set_param_16546, *[str_16547, get_output_charset_call_result_16551], **kwargs_16552)
            
            # SSA join for if statement (line 256)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 261)
        # Getting the type of 'unicode' (line 261)
        unicode_16554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 37), 'unicode')
        # Getting the type of 'self' (line 261)
        self_16555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 22), 'self')
        # Obtaining the member '_payload' of a type (line 261)
        _payload_16556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 22), self_16555, '_payload')
        
        (may_be_16557, more_types_in_union_16558) = may_be_subtype(unicode_16554, _payload_16556)

        if may_be_16557:

            if more_types_in_union_16558:
                # Runtime conditional SSA (line 261)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 261)
            self_16559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self')
            # Obtaining the member '_payload' of a type (line 261)
            _payload_16560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_16559, '_payload')
            # Setting the type of the member '_payload' of a type (line 261)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_16559, '_payload', remove_not_subtype_from_union(_payload_16556, unicode))
            
            # Assigning a Call to a Attribute (line 262):
            
            # Assigning a Call to a Attribute (line 262):
            
            # Call to encode(...): (line 262)
            # Processing the call arguments (line 262)
            # Getting the type of 'charset' (line 262)
            charset_16564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 49), 'charset', False)
            # Obtaining the member 'output_charset' of a type (line 262)
            output_charset_16565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 49), charset_16564, 'output_charset')
            # Processing the call keyword arguments (line 262)
            kwargs_16566 = {}
            # Getting the type of 'self' (line 262)
            self_16561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 28), 'self', False)
            # Obtaining the member '_payload' of a type (line 262)
            _payload_16562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 28), self_16561, '_payload')
            # Obtaining the member 'encode' of a type (line 262)
            encode_16563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 28), _payload_16562, 'encode')
            # Calling encode(args, kwargs) (line 262)
            encode_call_result_16567 = invoke(stypy.reporting.localization.Localization(__file__, 262, 28), encode_16563, *[output_charset_16565], **kwargs_16566)
            
            # Getting the type of 'self' (line 262)
            self_16568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'self')
            # Setting the type of the member '_payload' of a type (line 262)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 12), self_16568, '_payload', encode_call_result_16567)

            if more_types_in_union_16558:
                # SSA join for if statement (line 261)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to str(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'charset' (line 263)
        charset_16570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'charset', False)
        # Processing the call keyword arguments (line 263)
        kwargs_16571 = {}
        # Getting the type of 'str' (line 263)
        str_16569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 11), 'str', False)
        # Calling str(args, kwargs) (line 263)
        str_call_result_16572 = invoke(stypy.reporting.localization.Localization(__file__, 263, 11), str_16569, *[charset_16570], **kwargs_16571)
        
        
        # Call to get_output_charset(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_16575 = {}
        # Getting the type of 'charset' (line 263)
        charset_16573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 27), 'charset', False)
        # Obtaining the member 'get_output_charset' of a type (line 263)
        get_output_charset_16574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 27), charset_16573, 'get_output_charset')
        # Calling get_output_charset(args, kwargs) (line 263)
        get_output_charset_call_result_16576 = invoke(stypy.reporting.localization.Localization(__file__, 263, 27), get_output_charset_16574, *[], **kwargs_16575)
        
        # Applying the binary operator '!=' (line 263)
        result_ne_16577 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 11), '!=', str_call_result_16572, get_output_charset_call_result_16576)
        
        # Testing if the type of an if condition is none (line 263)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 263, 8), result_ne_16577):
            pass
        else:
            
            # Testing the type of an if condition (line 263)
            if_condition_16578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 8), result_ne_16577)
            # Assigning a type to the variable 'if_condition_16578' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'if_condition_16578', if_condition_16578)
            # SSA begins for if statement (line 263)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 264):
            
            # Assigning a Call to a Attribute (line 264):
            
            # Call to body_encode(...): (line 264)
            # Processing the call arguments (line 264)
            # Getting the type of 'self' (line 264)
            self_16581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 48), 'self', False)
            # Obtaining the member '_payload' of a type (line 264)
            _payload_16582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 48), self_16581, '_payload')
            # Processing the call keyword arguments (line 264)
            kwargs_16583 = {}
            # Getting the type of 'charset' (line 264)
            charset_16579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 28), 'charset', False)
            # Obtaining the member 'body_encode' of a type (line 264)
            body_encode_16580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 28), charset_16579, 'body_encode')
            # Calling body_encode(args, kwargs) (line 264)
            body_encode_call_result_16584 = invoke(stypy.reporting.localization.Localization(__file__, 264, 28), body_encode_16580, *[_payload_16582], **kwargs_16583)
            
            # Getting the type of 'self' (line 264)
            self_16585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'self')
            # Setting the type of the member '_payload' of a type (line 264)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), self_16585, '_payload', body_encode_call_result_16584)
            # SSA join for if statement (line 263)
            module_type_store = module_type_store.join_ssa_context()
            

        
        str_16586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 11), 'str', 'Content-Transfer-Encoding')
        # Getting the type of 'self' (line 265)
        self_16587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 46), 'self')
        # Applying the binary operator 'notin' (line 265)
        result_contains_16588 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), 'notin', str_16586, self_16587)
        
        # Testing if the type of an if condition is none (line 265)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 265, 8), result_contains_16588):
            pass
        else:
            
            # Testing the type of an if condition (line 265)
            if_condition_16589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), result_contains_16588)
            # Assigning a type to the variable 'if_condition_16589' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_16589', if_condition_16589)
            # SSA begins for if statement (line 265)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 266):
            
            # Assigning a Call to a Name (line 266):
            
            # Call to get_body_encoding(...): (line 266)
            # Processing the call keyword arguments (line 266)
            kwargs_16592 = {}
            # Getting the type of 'charset' (line 266)
            charset_16590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 18), 'charset', False)
            # Obtaining the member 'get_body_encoding' of a type (line 266)
            get_body_encoding_16591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 18), charset_16590, 'get_body_encoding')
            # Calling get_body_encoding(args, kwargs) (line 266)
            get_body_encoding_call_result_16593 = invoke(stypy.reporting.localization.Localization(__file__, 266, 18), get_body_encoding_16591, *[], **kwargs_16592)
            
            # Assigning a type to the variable 'cte' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'cte', get_body_encoding_call_result_16593)
            
            
            # SSA begins for try-except statement (line 267)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to cte(...): (line 268)
            # Processing the call arguments (line 268)
            # Getting the type of 'self' (line 268)
            self_16595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'self', False)
            # Processing the call keyword arguments (line 268)
            kwargs_16596 = {}
            # Getting the type of 'cte' (line 268)
            cte_16594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'cte', False)
            # Calling cte(args, kwargs) (line 268)
            cte_call_result_16597 = invoke(stypy.reporting.localization.Localization(__file__, 268, 16), cte_16594, *[self_16595], **kwargs_16596)
            
            # SSA branch for the except part of a try statement (line 267)
            # SSA branch for the except 'TypeError' branch of a try statement (line 267)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Call to a Attribute (line 270):
            
            # Assigning a Call to a Attribute (line 270):
            
            # Call to body_encode(...): (line 270)
            # Processing the call arguments (line 270)
            # Getting the type of 'self' (line 270)
            self_16600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 52), 'self', False)
            # Obtaining the member '_payload' of a type (line 270)
            _payload_16601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 52), self_16600, '_payload')
            # Processing the call keyword arguments (line 270)
            kwargs_16602 = {}
            # Getting the type of 'charset' (line 270)
            charset_16598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 32), 'charset', False)
            # Obtaining the member 'body_encode' of a type (line 270)
            body_encode_16599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 32), charset_16598, 'body_encode')
            # Calling body_encode(args, kwargs) (line 270)
            body_encode_call_result_16603 = invoke(stypy.reporting.localization.Localization(__file__, 270, 32), body_encode_16599, *[_payload_16601], **kwargs_16602)
            
            # Getting the type of 'self' (line 270)
            self_16604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'self')
            # Setting the type of the member '_payload' of a type (line 270)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), self_16604, '_payload', body_encode_call_result_16603)
            
            # Call to add_header(...): (line 271)
            # Processing the call arguments (line 271)
            str_16607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 32), 'str', 'Content-Transfer-Encoding')
            # Getting the type of 'cte' (line 271)
            cte_16608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 61), 'cte', False)
            # Processing the call keyword arguments (line 271)
            kwargs_16609 = {}
            # Getting the type of 'self' (line 271)
            self_16605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'self', False)
            # Obtaining the member 'add_header' of a type (line 271)
            add_header_16606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), self_16605, 'add_header')
            # Calling add_header(args, kwargs) (line 271)
            add_header_call_result_16610 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), add_header_16606, *[str_16607, cte_16608], **kwargs_16609)
            
            # SSA join for try-except statement (line 267)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 265)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'set_charset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_charset' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_16611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_charset'
        return stypy_return_type_16611


    @norecursion
    def get_charset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_charset'
        module_type_store = module_type_store.open_function_context('get_charset', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_charset.__dict__.__setitem__('stypy_localization', localization)
        Message.get_charset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_charset.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_charset.__dict__.__setitem__('stypy_function_name', 'Message.get_charset')
        Message.get_charset.__dict__.__setitem__('stypy_param_names_list', [])
        Message.get_charset.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_charset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_charset.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_charset.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_charset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_charset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_charset', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_charset', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_charset(...)' code ##################

        str_16612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, (-1)), 'str', "Return the Charset instance associated with the message's payload.\n        ")
        # Getting the type of 'self' (line 276)
        self_16613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'self')
        # Obtaining the member '_charset' of a type (line 276)
        _charset_16614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 15), self_16613, '_charset')
        # Assigning a type to the variable 'stypy_return_type' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'stypy_return_type', _charset_16614)
        
        # ################# End of 'get_charset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_charset' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_16615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16615)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_charset'
        return stypy_return_type_16615


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 281, 4, False)
        # Assigning a type to the variable 'self' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.__len__.__dict__.__setitem__('stypy_localization', localization)
        Message.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.__len__.__dict__.__setitem__('stypy_function_name', 'Message.__len__')
        Message.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        Message.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        str_16616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 8), 'str', 'Return the total number of headers, including duplicates.')
        
        # Call to len(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'self' (line 283)
        self_16618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'self', False)
        # Obtaining the member '_headers' of a type (line 283)
        _headers_16619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 19), self_16618, '_headers')
        # Processing the call keyword arguments (line 283)
        kwargs_16620 = {}
        # Getting the type of 'len' (line 283)
        len_16617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'len', False)
        # Calling len(args, kwargs) (line 283)
        len_call_result_16621 = invoke(stypy.reporting.localization.Localization(__file__, 283, 15), len_16617, *[_headers_16619], **kwargs_16620)
        
        # Assigning a type to the variable 'stypy_return_type' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'stypy_return_type', len_call_result_16621)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 281)
        stypy_return_type_16622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_16622


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        Message.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.__getitem__.__dict__.__setitem__('stypy_function_name', 'Message.__getitem__')
        Message.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['name'])
        Message.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.__getitem__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        str_16623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, (-1)), 'str', 'Get a header value.\n\n        Return None if the header is missing instead of raising an exception.\n\n        Note that if the header appeared multiple times, exactly which\n        occurrence gets returned is undefined.  Use get_all() to get all\n        the values matching a header field name.\n        ')
        
        # Call to get(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'name' (line 294)
        name_16626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'name', False)
        # Processing the call keyword arguments (line 294)
        kwargs_16627 = {}
        # Getting the type of 'self' (line 294)
        self_16624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'self', False)
        # Obtaining the member 'get' of a type (line 294)
        get_16625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), self_16624, 'get')
        # Calling get(args, kwargs) (line 294)
        get_call_result_16628 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), get_16625, *[name_16626], **kwargs_16627)
        
        # Assigning a type to the variable 'stypy_return_type' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'stypy_return_type', get_call_result_16628)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_16629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16629)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_16629


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 296, 4, False)
        # Assigning a type to the variable 'self' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        Message.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.__setitem__.__dict__.__setitem__('stypy_function_name', 'Message.__setitem__')
        Message.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['name', 'val'])
        Message.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.__setitem__', ['name', 'val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['name', 'val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        str_16630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, (-1)), 'str', 'Set the value of a header.\n\n        Note: this does not overwrite an existing header with the same field\n        name.  Use __delitem__() first to delete any existing headers.\n        ')
        
        # Call to append(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Obtaining an instance of the builtin type 'tuple' (line 302)
        tuple_16634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 302)
        # Adding element type (line 302)
        # Getting the type of 'name' (line 302)
        name_16635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 30), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 30), tuple_16634, name_16635)
        # Adding element type (line 302)
        # Getting the type of 'val' (line 302)
        val_16636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 36), 'val', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 30), tuple_16634, val_16636)
        
        # Processing the call keyword arguments (line 302)
        kwargs_16637 = {}
        # Getting the type of 'self' (line 302)
        self_16631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'self', False)
        # Obtaining the member '_headers' of a type (line 302)
        _headers_16632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), self_16631, '_headers')
        # Obtaining the member 'append' of a type (line 302)
        append_16633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), _headers_16632, 'append')
        # Calling append(args, kwargs) (line 302)
        append_call_result_16638 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), append_16633, *[tuple_16634], **kwargs_16637)
        
        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 296)
        stypy_return_type_16639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16639)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_16639


    @norecursion
    def __delitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__delitem__'
        module_type_store = module_type_store.open_function_context('__delitem__', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.__delitem__.__dict__.__setitem__('stypy_localization', localization)
        Message.__delitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.__delitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.__delitem__.__dict__.__setitem__('stypy_function_name', 'Message.__delitem__')
        Message.__delitem__.__dict__.__setitem__('stypy_param_names_list', ['name'])
        Message.__delitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.__delitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.__delitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.__delitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.__delitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.__delitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.__delitem__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__delitem__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__delitem__(...)' code ##################

        str_16640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, (-1)), 'str', 'Delete all occurrences of a header, if present.\n\n        Does not raise an exception if the header is missing.\n        ')
        
        # Assigning a Call to a Name (line 309):
        
        # Assigning a Call to a Name (line 309):
        
        # Call to lower(...): (line 309)
        # Processing the call keyword arguments (line 309)
        kwargs_16643 = {}
        # Getting the type of 'name' (line 309)
        name_16641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 15), 'name', False)
        # Obtaining the member 'lower' of a type (line 309)
        lower_16642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 15), name_16641, 'lower')
        # Calling lower(args, kwargs) (line 309)
        lower_call_result_16644 = invoke(stypy.reporting.localization.Localization(__file__, 309, 15), lower_16642, *[], **kwargs_16643)
        
        # Assigning a type to the variable 'name' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'name', lower_call_result_16644)
        
        # Assigning a List to a Name (line 310):
        
        # Assigning a List to a Name (line 310):
        
        # Obtaining an instance of the builtin type 'list' (line 310)
        list_16645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 310)
        
        # Assigning a type to the variable 'newheaders' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'newheaders', list_16645)
        
        # Getting the type of 'self' (line 311)
        self_16646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 20), 'self')
        # Obtaining the member '_headers' of a type (line 311)
        _headers_16647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 20), self_16646, '_headers')
        # Assigning a type to the variable '_headers_16647' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), '_headers_16647', _headers_16647)
        # Testing if the for loop is going to be iterated (line 311)
        # Testing the type of a for loop iterable (line 311)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 311, 8), _headers_16647)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 311, 8), _headers_16647):
            # Getting the type of the for loop variable (line 311)
            for_loop_var_16648 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 311, 8), _headers_16647)
            # Assigning a type to the variable 'k' (line 311)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 8), for_loop_var_16648, 2, 0))
            # Assigning a type to the variable 'v' (line 311)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 8), for_loop_var_16648, 2, 1))
            # SSA begins for a for statement (line 311)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to lower(...): (line 312)
            # Processing the call keyword arguments (line 312)
            kwargs_16651 = {}
            # Getting the type of 'k' (line 312)
            k_16649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 15), 'k', False)
            # Obtaining the member 'lower' of a type (line 312)
            lower_16650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 15), k_16649, 'lower')
            # Calling lower(args, kwargs) (line 312)
            lower_call_result_16652 = invoke(stypy.reporting.localization.Localization(__file__, 312, 15), lower_16650, *[], **kwargs_16651)
            
            # Getting the type of 'name' (line 312)
            name_16653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 28), 'name')
            # Applying the binary operator '!=' (line 312)
            result_ne_16654 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 15), '!=', lower_call_result_16652, name_16653)
            
            # Testing if the type of an if condition is none (line 312)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 312, 12), result_ne_16654):
                pass
            else:
                
                # Testing the type of an if condition (line 312)
                if_condition_16655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 12), result_ne_16654)
                # Assigning a type to the variable 'if_condition_16655' (line 312)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'if_condition_16655', if_condition_16655)
                # SSA begins for if statement (line 312)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 313)
                # Processing the call arguments (line 313)
                
                # Obtaining an instance of the builtin type 'tuple' (line 313)
                tuple_16658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 35), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 313)
                # Adding element type (line 313)
                # Getting the type of 'k' (line 313)
                k_16659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 35), 'k', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 35), tuple_16658, k_16659)
                # Adding element type (line 313)
                # Getting the type of 'v' (line 313)
                v_16660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 38), 'v', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 35), tuple_16658, v_16660)
                
                # Processing the call keyword arguments (line 313)
                kwargs_16661 = {}
                # Getting the type of 'newheaders' (line 313)
                newheaders_16656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'newheaders', False)
                # Obtaining the member 'append' of a type (line 313)
                append_16657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 16), newheaders_16656, 'append')
                # Calling append(args, kwargs) (line 313)
                append_call_result_16662 = invoke(stypy.reporting.localization.Localization(__file__, 313, 16), append_16657, *[tuple_16658], **kwargs_16661)
                
                # SSA join for if statement (line 312)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Attribute (line 314):
        
        # Assigning a Name to a Attribute (line 314):
        # Getting the type of 'newheaders' (line 314)
        newheaders_16663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 24), 'newheaders')
        # Getting the type of 'self' (line 314)
        self_16664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'self')
        # Setting the type of the member '_headers' of a type (line 314)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), self_16664, '_headers', newheaders_16663)
        
        # ################# End of '__delitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__delitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_16665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16665)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__delitem__'
        return stypy_return_type_16665


    @norecursion
    def __contains__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__contains__'
        module_type_store = module_type_store.open_function_context('__contains__', 316, 4, False)
        # Assigning a type to the variable 'self' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.__contains__.__dict__.__setitem__('stypy_localization', localization)
        Message.__contains__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.__contains__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.__contains__.__dict__.__setitem__('stypy_function_name', 'Message.__contains__')
        Message.__contains__.__dict__.__setitem__('stypy_param_names_list', ['name'])
        Message.__contains__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.__contains__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.__contains__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.__contains__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.__contains__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.__contains__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.__contains__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__contains__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__contains__(...)' code ##################

        
        
        # Call to lower(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_16668 = {}
        # Getting the type of 'name' (line 317)
        name_16666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'name', False)
        # Obtaining the member 'lower' of a type (line 317)
        lower_16667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 15), name_16666, 'lower')
        # Calling lower(args, kwargs) (line 317)
        lower_call_result_16669 = invoke(stypy.reporting.localization.Localization(__file__, 317, 15), lower_16667, *[], **kwargs_16668)
        
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 317)
        self_16674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 54), 'self')
        # Obtaining the member '_headers' of a type (line 317)
        _headers_16675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 54), self_16674, '_headers')
        comprehension_16676 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 32), _headers_16675)
        # Assigning a type to the variable 'k' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 32), comprehension_16676))
        # Assigning a type to the variable 'v' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 32), comprehension_16676))
        
        # Call to lower(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_16672 = {}
        # Getting the type of 'k' (line 317)
        k_16670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'k', False)
        # Obtaining the member 'lower' of a type (line 317)
        lower_16671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 32), k_16670, 'lower')
        # Calling lower(args, kwargs) (line 317)
        lower_call_result_16673 = invoke(stypy.reporting.localization.Localization(__file__, 317, 32), lower_16671, *[], **kwargs_16672)
        
        list_16677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 32), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 32), list_16677, lower_call_result_16673)
        # Applying the binary operator 'in' (line 317)
        result_contains_16678 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 15), 'in', lower_call_result_16669, list_16677)
        
        # Assigning a type to the variable 'stypy_return_type' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'stypy_return_type', result_contains_16678)
        
        # ################# End of '__contains__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__contains__' in the type store
        # Getting the type of 'stypy_return_type' (line 316)
        stypy_return_type_16679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__contains__'
        return stypy_return_type_16679


    @norecursion
    def has_key(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_key'
        module_type_store = module_type_store.open_function_context('has_key', 319, 4, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.has_key.__dict__.__setitem__('stypy_localization', localization)
        Message.has_key.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.has_key.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.has_key.__dict__.__setitem__('stypy_function_name', 'Message.has_key')
        Message.has_key.__dict__.__setitem__('stypy_param_names_list', ['name'])
        Message.has_key.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.has_key.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.has_key.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.has_key.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.has_key.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.has_key.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.has_key', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_key', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_key(...)' code ##################

        str_16680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 8), 'str', 'Return true if the message contains the header.')
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Call to object(...): (line 321)
        # Processing the call keyword arguments (line 321)
        kwargs_16682 = {}
        # Getting the type of 'object' (line 321)
        object_16681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 18), 'object', False)
        # Calling object(args, kwargs) (line 321)
        object_call_result_16683 = invoke(stypy.reporting.localization.Localization(__file__, 321, 18), object_16681, *[], **kwargs_16682)
        
        # Assigning a type to the variable 'missing' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'missing', object_call_result_16683)
        
        
        # Call to get(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'name' (line 322)
        name_16686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 24), 'name', False)
        # Getting the type of 'missing' (line 322)
        missing_16687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 30), 'missing', False)
        # Processing the call keyword arguments (line 322)
        kwargs_16688 = {}
        # Getting the type of 'self' (line 322)
        self_16684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'self', False)
        # Obtaining the member 'get' of a type (line 322)
        get_16685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 15), self_16684, 'get')
        # Calling get(args, kwargs) (line 322)
        get_call_result_16689 = invoke(stypy.reporting.localization.Localization(__file__, 322, 15), get_16685, *[name_16686, missing_16687], **kwargs_16688)
        
        # Getting the type of 'missing' (line 322)
        missing_16690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 46), 'missing')
        # Applying the binary operator 'isnot' (line 322)
        result_is_not_16691 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), 'isnot', get_call_result_16689, missing_16690)
        
        # Assigning a type to the variable 'stypy_return_type' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'stypy_return_type', result_is_not_16691)
        
        # ################# End of 'has_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_key' in the type store
        # Getting the type of 'stypy_return_type' (line 319)
        stypy_return_type_16692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_key'
        return stypy_return_type_16692


    @norecursion
    def keys(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'keys'
        module_type_store = module_type_store.open_function_context('keys', 324, 4, False)
        # Assigning a type to the variable 'self' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.keys.__dict__.__setitem__('stypy_localization', localization)
        Message.keys.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.keys.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.keys.__dict__.__setitem__('stypy_function_name', 'Message.keys')
        Message.keys.__dict__.__setitem__('stypy_param_names_list', [])
        Message.keys.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.keys.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.keys.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.keys.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.keys.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.keys.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.keys', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'keys', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'keys(...)' code ##################

        str_16693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, (-1)), 'str', "Return a list of all the message's header field names.\n\n        These will be sorted in the order they appeared in the original\n        message, or were added to the message, and may contain duplicates.\n        Any fields deleted and re-inserted are always appended to the header\n        list.\n        ")
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 332)
        self_16695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 30), 'self')
        # Obtaining the member '_headers' of a type (line 332)
        _headers_16696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 30), self_16695, '_headers')
        comprehension_16697 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 16), _headers_16696)
        # Assigning a type to the variable 'k' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 16), comprehension_16697))
        # Assigning a type to the variable 'v' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 16), comprehension_16697))
        # Getting the type of 'k' (line 332)
        k_16694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'k')
        list_16698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 16), list_16698, k_16694)
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'stypy_return_type', list_16698)
        
        # ################# End of 'keys(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'keys' in the type store
        # Getting the type of 'stypy_return_type' (line 324)
        stypy_return_type_16699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'keys'
        return stypy_return_type_16699


    @norecursion
    def values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'values'
        module_type_store = module_type_store.open_function_context('values', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.values.__dict__.__setitem__('stypy_localization', localization)
        Message.values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.values.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.values.__dict__.__setitem__('stypy_function_name', 'Message.values')
        Message.values.__dict__.__setitem__('stypy_param_names_list', [])
        Message.values.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.values.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.values.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.values.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.values', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'values', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'values(...)' code ##################

        str_16700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, (-1)), 'str', "Return a list of all the message's header values.\n\n        These will be sorted in the order they appeared in the original\n        message, or were added to the message, and may contain duplicates.\n        Any fields deleted and re-inserted are always appended to the header\n        list.\n        ")
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 342)
        self_16702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 30), 'self')
        # Obtaining the member '_headers' of a type (line 342)
        _headers_16703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 30), self_16702, '_headers')
        comprehension_16704 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 16), _headers_16703)
        # Assigning a type to the variable 'k' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 16), comprehension_16704))
        # Assigning a type to the variable 'v' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 16), comprehension_16704))
        # Getting the type of 'v' (line 342)
        v_16701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'v')
        list_16705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 16), list_16705, v_16701)
        # Assigning a type to the variable 'stypy_return_type' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'stypy_return_type', list_16705)
        
        # ################# End of 'values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'values' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_16706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16706)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'values'
        return stypy_return_type_16706


    @norecursion
    def items(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'items'
        module_type_store = module_type_store.open_function_context('items', 344, 4, False)
        # Assigning a type to the variable 'self' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.items.__dict__.__setitem__('stypy_localization', localization)
        Message.items.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.items.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.items.__dict__.__setitem__('stypy_function_name', 'Message.items')
        Message.items.__dict__.__setitem__('stypy_param_names_list', [])
        Message.items.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.items.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.items.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.items.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.items.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.items.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.items', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'items', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'items(...)' code ##################

        str_16707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, (-1)), 'str', "Get all the message's header fields and values.\n\n        These will be sorted in the order they appeared in the original\n        message, or were added to the message, and may contain duplicates.\n        Any fields deleted and re-inserted are always appended to the header\n        list.\n        ")
        
        # Obtaining the type of the subscript
        slice_16708 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 352, 15), None, None, None)
        # Getting the type of 'self' (line 352)
        self_16709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'self')
        # Obtaining the member '_headers' of a type (line 352)
        _headers_16710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 15), self_16709, '_headers')
        # Obtaining the member '__getitem__' of a type (line 352)
        getitem___16711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 15), _headers_16710, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 352)
        subscript_call_result_16712 = invoke(stypy.reporting.localization.Localization(__file__, 352, 15), getitem___16711, slice_16708)
        
        # Assigning a type to the variable 'stypy_return_type' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'stypy_return_type', subscript_call_result_16712)
        
        # ################# End of 'items(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'items' in the type store
        # Getting the type of 'stypy_return_type' (line 344)
        stypy_return_type_16713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'items'
        return stypy_return_type_16713


    @norecursion
    def get(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 354)
        None_16714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 32), 'None')
        defaults = [None_16714]
        # Create a new context for function 'get'
        module_type_store = module_type_store.open_function_context('get', 354, 4, False)
        # Assigning a type to the variable 'self' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get.__dict__.__setitem__('stypy_localization', localization)
        Message.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get.__dict__.__setitem__('stypy_function_name', 'Message.get')
        Message.get.__dict__.__setitem__('stypy_param_names_list', ['name', 'failobj'])
        Message.get.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get', ['name', 'failobj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get', localization, ['name', 'failobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get(...)' code ##################

        str_16715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, (-1)), 'str', 'Get a header value.\n\n        Like __getitem__() but return failobj instead of None when the field\n        is missing.\n        ')
        
        # Assigning a Call to a Name (line 360):
        
        # Assigning a Call to a Name (line 360):
        
        # Call to lower(...): (line 360)
        # Processing the call keyword arguments (line 360)
        kwargs_16718 = {}
        # Getting the type of 'name' (line 360)
        name_16716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'name', False)
        # Obtaining the member 'lower' of a type (line 360)
        lower_16717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 15), name_16716, 'lower')
        # Calling lower(args, kwargs) (line 360)
        lower_call_result_16719 = invoke(stypy.reporting.localization.Localization(__file__, 360, 15), lower_16717, *[], **kwargs_16718)
        
        # Assigning a type to the variable 'name' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'name', lower_call_result_16719)
        
        # Getting the type of 'self' (line 361)
        self_16720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'self')
        # Obtaining the member '_headers' of a type (line 361)
        _headers_16721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 20), self_16720, '_headers')
        # Assigning a type to the variable '_headers_16721' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), '_headers_16721', _headers_16721)
        # Testing if the for loop is going to be iterated (line 361)
        # Testing the type of a for loop iterable (line 361)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 361, 8), _headers_16721)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 361, 8), _headers_16721):
            # Getting the type of the for loop variable (line 361)
            for_loop_var_16722 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 361, 8), _headers_16721)
            # Assigning a type to the variable 'k' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 8), for_loop_var_16722, 2, 0))
            # Assigning a type to the variable 'v' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 8), for_loop_var_16722, 2, 1))
            # SSA begins for a for statement (line 361)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to lower(...): (line 362)
            # Processing the call keyword arguments (line 362)
            kwargs_16725 = {}
            # Getting the type of 'k' (line 362)
            k_16723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'k', False)
            # Obtaining the member 'lower' of a type (line 362)
            lower_16724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 15), k_16723, 'lower')
            # Calling lower(args, kwargs) (line 362)
            lower_call_result_16726 = invoke(stypy.reporting.localization.Localization(__file__, 362, 15), lower_16724, *[], **kwargs_16725)
            
            # Getting the type of 'name' (line 362)
            name_16727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 28), 'name')
            # Applying the binary operator '==' (line 362)
            result_eq_16728 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 15), '==', lower_call_result_16726, name_16727)
            
            # Testing if the type of an if condition is none (line 362)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 362, 12), result_eq_16728):
                pass
            else:
                
                # Testing the type of an if condition (line 362)
                if_condition_16729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 12), result_eq_16728)
                # Assigning a type to the variable 'if_condition_16729' (line 362)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'if_condition_16729', if_condition_16729)
                # SSA begins for if statement (line 362)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'v' (line 363)
                v_16730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 23), 'v')
                # Assigning a type to the variable 'stypy_return_type' (line 363)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'stypy_return_type', v_16730)
                # SSA join for if statement (line 362)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'failobj' (line 364)
        failobj_16731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'failobj')
        # Assigning a type to the variable 'stypy_return_type' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'stypy_return_type', failobj_16731)
        
        # ################# End of 'get(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get' in the type store
        # Getting the type of 'stypy_return_type' (line 354)
        stypy_return_type_16732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get'
        return stypy_return_type_16732


    @norecursion
    def get_all(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 370)
        None_16733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 36), 'None')
        defaults = [None_16733]
        # Create a new context for function 'get_all'
        module_type_store = module_type_store.open_function_context('get_all', 370, 4, False)
        # Assigning a type to the variable 'self' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_all.__dict__.__setitem__('stypy_localization', localization)
        Message.get_all.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_all.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_all.__dict__.__setitem__('stypy_function_name', 'Message.get_all')
        Message.get_all.__dict__.__setitem__('stypy_param_names_list', ['name', 'failobj'])
        Message.get_all.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_all.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_all.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_all.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_all.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_all.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_all', ['name', 'failobj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_all', localization, ['name', 'failobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_all(...)' code ##################

        str_16734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, (-1)), 'str', 'Return a list of all the values for the named field.\n\n        These will be sorted in the order they appeared in the original\n        message, and may contain duplicates.  Any fields deleted and\n        re-inserted are always appended to the header list.\n\n        If no such fields exist, failobj is returned (defaults to None).\n        ')
        
        # Assigning a List to a Name (line 379):
        
        # Assigning a List to a Name (line 379):
        
        # Obtaining an instance of the builtin type 'list' (line 379)
        list_16735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 379)
        
        # Assigning a type to the variable 'values' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'values', list_16735)
        
        # Assigning a Call to a Name (line 380):
        
        # Assigning a Call to a Name (line 380):
        
        # Call to lower(...): (line 380)
        # Processing the call keyword arguments (line 380)
        kwargs_16738 = {}
        # Getting the type of 'name' (line 380)
        name_16736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'name', False)
        # Obtaining the member 'lower' of a type (line 380)
        lower_16737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 15), name_16736, 'lower')
        # Calling lower(args, kwargs) (line 380)
        lower_call_result_16739 = invoke(stypy.reporting.localization.Localization(__file__, 380, 15), lower_16737, *[], **kwargs_16738)
        
        # Assigning a type to the variable 'name' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'name', lower_call_result_16739)
        
        # Getting the type of 'self' (line 381)
        self_16740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 20), 'self')
        # Obtaining the member '_headers' of a type (line 381)
        _headers_16741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 20), self_16740, '_headers')
        # Assigning a type to the variable '_headers_16741' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), '_headers_16741', _headers_16741)
        # Testing if the for loop is going to be iterated (line 381)
        # Testing the type of a for loop iterable (line 381)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 381, 8), _headers_16741)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 381, 8), _headers_16741):
            # Getting the type of the for loop variable (line 381)
            for_loop_var_16742 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 381, 8), _headers_16741)
            # Assigning a type to the variable 'k' (line 381)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 8), for_loop_var_16742, 2, 0))
            # Assigning a type to the variable 'v' (line 381)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 8), for_loop_var_16742, 2, 1))
            # SSA begins for a for statement (line 381)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to lower(...): (line 382)
            # Processing the call keyword arguments (line 382)
            kwargs_16745 = {}
            # Getting the type of 'k' (line 382)
            k_16743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 15), 'k', False)
            # Obtaining the member 'lower' of a type (line 382)
            lower_16744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 15), k_16743, 'lower')
            # Calling lower(args, kwargs) (line 382)
            lower_call_result_16746 = invoke(stypy.reporting.localization.Localization(__file__, 382, 15), lower_16744, *[], **kwargs_16745)
            
            # Getting the type of 'name' (line 382)
            name_16747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 28), 'name')
            # Applying the binary operator '==' (line 382)
            result_eq_16748 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 15), '==', lower_call_result_16746, name_16747)
            
            # Testing if the type of an if condition is none (line 382)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 382, 12), result_eq_16748):
                pass
            else:
                
                # Testing the type of an if condition (line 382)
                if_condition_16749 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 12), result_eq_16748)
                # Assigning a type to the variable 'if_condition_16749' (line 382)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'if_condition_16749', if_condition_16749)
                # SSA begins for if statement (line 382)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 383)
                # Processing the call arguments (line 383)
                # Getting the type of 'v' (line 383)
                v_16752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 30), 'v', False)
                # Processing the call keyword arguments (line 383)
                kwargs_16753 = {}
                # Getting the type of 'values' (line 383)
                values_16750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'values', False)
                # Obtaining the member 'append' of a type (line 383)
                append_16751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 16), values_16750, 'append')
                # Calling append(args, kwargs) (line 383)
                append_call_result_16754 = invoke(stypy.reporting.localization.Localization(__file__, 383, 16), append_16751, *[v_16752], **kwargs_16753)
                
                # SSA join for if statement (line 382)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'values' (line 384)
        values_16755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'values')
        # Applying the 'not' unary operator (line 384)
        result_not__16756 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 11), 'not', values_16755)
        
        # Testing if the type of an if condition is none (line 384)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 384, 8), result_not__16756):
            pass
        else:
            
            # Testing the type of an if condition (line 384)
            if_condition_16757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 8), result_not__16756)
            # Assigning a type to the variable 'if_condition_16757' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'if_condition_16757', if_condition_16757)
            # SSA begins for if statement (line 384)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'failobj' (line 385)
            failobj_16758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'failobj')
            # Assigning a type to the variable 'stypy_return_type' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'stypy_return_type', failobj_16758)
            # SSA join for if statement (line 384)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'values' (line 386)
        values_16759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'values')
        # Assigning a type to the variable 'stypy_return_type' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'stypy_return_type', values_16759)
        
        # ################# End of 'get_all(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_all' in the type store
        # Getting the type of 'stypy_return_type' (line 370)
        stypy_return_type_16760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16760)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_all'
        return stypy_return_type_16760


    @norecursion
    def add_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_header'
        module_type_store = module_type_store.open_function_context('add_header', 388, 4, False)
        # Assigning a type to the variable 'self' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.add_header.__dict__.__setitem__('stypy_localization', localization)
        Message.add_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.add_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.add_header.__dict__.__setitem__('stypy_function_name', 'Message.add_header')
        Message.add_header.__dict__.__setitem__('stypy_param_names_list', ['_name', '_value'])
        Message.add_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.add_header.__dict__.__setitem__('stypy_kwargs_param_name', '_params')
        Message.add_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.add_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.add_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.add_header.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.add_header', ['_name', '_value'], None, '_params', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_header', localization, ['_name', '_value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_header(...)' code ##################

        str_16761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, (-1)), 'str', 'Extended header setting.\n\n        name is the header field to add.  keyword arguments can be used to set\n        additional parameters for the header field, with underscores converted\n        to dashes.  Normally the parameter will be added as key="value" unless\n        value is None, in which case only the key will be added.  If a\n        parameter value contains non-ASCII characters it must be specified as a\n        three-tuple of (charset, language, value), in which case it will be\n        encoded according to RFC2231 rules.\n\n        Example:\n\n        msg.add_header(\'content-disposition\', \'attachment\', filename=\'bud.gif\')\n        ')
        
        # Assigning a List to a Name (line 403):
        
        # Assigning a List to a Name (line 403):
        
        # Obtaining an instance of the builtin type 'list' (line 403)
        list_16762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 403)
        
        # Assigning a type to the variable 'parts' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'parts', list_16762)
        
        
        # Call to items(...): (line 404)
        # Processing the call keyword arguments (line 404)
        kwargs_16765 = {}
        # Getting the type of '_params' (line 404)
        _params_16763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 20), '_params', False)
        # Obtaining the member 'items' of a type (line 404)
        items_16764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 20), _params_16763, 'items')
        # Calling items(args, kwargs) (line 404)
        items_call_result_16766 = invoke(stypy.reporting.localization.Localization(__file__, 404, 20), items_16764, *[], **kwargs_16765)
        
        # Assigning a type to the variable 'items_call_result_16766' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'items_call_result_16766', items_call_result_16766)
        # Testing if the for loop is going to be iterated (line 404)
        # Testing the type of a for loop iterable (line 404)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 404, 8), items_call_result_16766)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 404, 8), items_call_result_16766):
            # Getting the type of the for loop variable (line 404)
            for_loop_var_16767 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 404, 8), items_call_result_16766)
            # Assigning a type to the variable 'k' (line 404)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 8), for_loop_var_16767, 2, 0))
            # Assigning a type to the variable 'v' (line 404)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 8), for_loop_var_16767, 2, 1))
            # SSA begins for a for statement (line 404)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Type idiom detected: calculating its left and rigth part (line 405)
            # Getting the type of 'v' (line 405)
            v_16768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'v')
            # Getting the type of 'None' (line 405)
            None_16769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 20), 'None')
            
            (may_be_16770, more_types_in_union_16771) = may_be_none(v_16768, None_16769)

            if may_be_16770:

                if more_types_in_union_16771:
                    # Runtime conditional SSA (line 405)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 406)
                # Processing the call arguments (line 406)
                
                # Call to replace(...): (line 406)
                # Processing the call arguments (line 406)
                str_16776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 39), 'str', '_')
                str_16777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 44), 'str', '-')
                # Processing the call keyword arguments (line 406)
                kwargs_16778 = {}
                # Getting the type of 'k' (line 406)
                k_16774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 29), 'k', False)
                # Obtaining the member 'replace' of a type (line 406)
                replace_16775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 29), k_16774, 'replace')
                # Calling replace(args, kwargs) (line 406)
                replace_call_result_16779 = invoke(stypy.reporting.localization.Localization(__file__, 406, 29), replace_16775, *[str_16776, str_16777], **kwargs_16778)
                
                # Processing the call keyword arguments (line 406)
                kwargs_16780 = {}
                # Getting the type of 'parts' (line 406)
                parts_16772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'parts', False)
                # Obtaining the member 'append' of a type (line 406)
                append_16773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 16), parts_16772, 'append')
                # Calling append(args, kwargs) (line 406)
                append_call_result_16781 = invoke(stypy.reporting.localization.Localization(__file__, 406, 16), append_16773, *[replace_call_result_16779], **kwargs_16780)
                

                if more_types_in_union_16771:
                    # Runtime conditional SSA for else branch (line 405)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_16770) or more_types_in_union_16771):
                
                # Call to append(...): (line 408)
                # Processing the call arguments (line 408)
                
                # Call to _formatparam(...): (line 408)
                # Processing the call arguments (line 408)
                
                # Call to replace(...): (line 408)
                # Processing the call arguments (line 408)
                str_16787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 52), 'str', '_')
                str_16788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 57), 'str', '-')
                # Processing the call keyword arguments (line 408)
                kwargs_16789 = {}
                # Getting the type of 'k' (line 408)
                k_16785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 42), 'k', False)
                # Obtaining the member 'replace' of a type (line 408)
                replace_16786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 42), k_16785, 'replace')
                # Calling replace(args, kwargs) (line 408)
                replace_call_result_16790 = invoke(stypy.reporting.localization.Localization(__file__, 408, 42), replace_16786, *[str_16787, str_16788], **kwargs_16789)
                
                # Getting the type of 'v' (line 408)
                v_16791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 63), 'v', False)
                # Processing the call keyword arguments (line 408)
                kwargs_16792 = {}
                # Getting the type of '_formatparam' (line 408)
                _formatparam_16784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 29), '_formatparam', False)
                # Calling _formatparam(args, kwargs) (line 408)
                _formatparam_call_result_16793 = invoke(stypy.reporting.localization.Localization(__file__, 408, 29), _formatparam_16784, *[replace_call_result_16790, v_16791], **kwargs_16792)
                
                # Processing the call keyword arguments (line 408)
                kwargs_16794 = {}
                # Getting the type of 'parts' (line 408)
                parts_16782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'parts', False)
                # Obtaining the member 'append' of a type (line 408)
                append_16783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 16), parts_16782, 'append')
                # Calling append(args, kwargs) (line 408)
                append_call_result_16795 = invoke(stypy.reporting.localization.Localization(__file__, 408, 16), append_16783, *[_formatparam_call_result_16793], **kwargs_16794)
                

                if (may_be_16770 and more_types_in_union_16771):
                    # SSA join for if statement (line 405)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Type idiom detected: calculating its left and rigth part (line 409)
        # Getting the type of '_value' (line 409)
        _value_16796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), '_value')
        # Getting the type of 'None' (line 409)
        None_16797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 25), 'None')
        
        (may_be_16798, more_types_in_union_16799) = may_not_be_none(_value_16796, None_16797)

        if may_be_16798:

            if more_types_in_union_16799:
                # Runtime conditional SSA (line 409)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to insert(...): (line 410)
            # Processing the call arguments (line 410)
            int_16802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 25), 'int')
            # Getting the type of '_value' (line 410)
            _value_16803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 28), '_value', False)
            # Processing the call keyword arguments (line 410)
            kwargs_16804 = {}
            # Getting the type of 'parts' (line 410)
            parts_16800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'parts', False)
            # Obtaining the member 'insert' of a type (line 410)
            insert_16801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), parts_16800, 'insert')
            # Calling insert(args, kwargs) (line 410)
            insert_call_result_16805 = invoke(stypy.reporting.localization.Localization(__file__, 410, 12), insert_16801, *[int_16802, _value_16803], **kwargs_16804)
            

            if more_types_in_union_16799:
                # SSA join for if statement (line 409)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to append(...): (line 411)
        # Processing the call arguments (line 411)
        
        # Obtaining an instance of the builtin type 'tuple' (line 411)
        tuple_16809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 411)
        # Adding element type (line 411)
        # Getting the type of '_name' (line 411)
        _name_16810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 30), '_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 30), tuple_16809, _name_16810)
        # Adding element type (line 411)
        
        # Call to join(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'parts' (line 411)
        parts_16813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 52), 'parts', False)
        # Processing the call keyword arguments (line 411)
        kwargs_16814 = {}
        # Getting the type of 'SEMISPACE' (line 411)
        SEMISPACE_16811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 37), 'SEMISPACE', False)
        # Obtaining the member 'join' of a type (line 411)
        join_16812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 37), SEMISPACE_16811, 'join')
        # Calling join(args, kwargs) (line 411)
        join_call_result_16815 = invoke(stypy.reporting.localization.Localization(__file__, 411, 37), join_16812, *[parts_16813], **kwargs_16814)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 30), tuple_16809, join_call_result_16815)
        
        # Processing the call keyword arguments (line 411)
        kwargs_16816 = {}
        # Getting the type of 'self' (line 411)
        self_16806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'self', False)
        # Obtaining the member '_headers' of a type (line 411)
        _headers_16807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), self_16806, '_headers')
        # Obtaining the member 'append' of a type (line 411)
        append_16808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), _headers_16807, 'append')
        # Calling append(args, kwargs) (line 411)
        append_call_result_16817 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), append_16808, *[tuple_16809], **kwargs_16816)
        
        
        # ################# End of 'add_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_header' in the type store
        # Getting the type of 'stypy_return_type' (line 388)
        stypy_return_type_16818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16818)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_header'
        return stypy_return_type_16818


    @norecursion
    def replace_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'replace_header'
        module_type_store = module_type_store.open_function_context('replace_header', 413, 4, False)
        # Assigning a type to the variable 'self' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.replace_header.__dict__.__setitem__('stypy_localization', localization)
        Message.replace_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.replace_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.replace_header.__dict__.__setitem__('stypy_function_name', 'Message.replace_header')
        Message.replace_header.__dict__.__setitem__('stypy_param_names_list', ['_name', '_value'])
        Message.replace_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.replace_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.replace_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.replace_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.replace_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.replace_header.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.replace_header', ['_name', '_value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'replace_header', localization, ['_name', '_value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'replace_header(...)' code ##################

        str_16819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, (-1)), 'str', 'Replace a header.\n\n        Replace the first matching header found in the message, retaining\n        header order and case.  If no matching header was found, a KeyError is\n        raised.\n        ')
        
        # Assigning a Call to a Name (line 420):
        
        # Assigning a Call to a Name (line 420):
        
        # Call to lower(...): (line 420)
        # Processing the call keyword arguments (line 420)
        kwargs_16822 = {}
        # Getting the type of '_name' (line 420)
        _name_16820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), '_name', False)
        # Obtaining the member 'lower' of a type (line 420)
        lower_16821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 16), _name_16820, 'lower')
        # Calling lower(args, kwargs) (line 420)
        lower_call_result_16823 = invoke(stypy.reporting.localization.Localization(__file__, 420, 16), lower_16821, *[], **kwargs_16822)
        
        # Assigning a type to the variable '_name' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), '_name', lower_call_result_16823)
        
        
        # Call to zip(...): (line 421)
        # Processing the call arguments (line 421)
        
        # Call to range(...): (line 421)
        # Processing the call arguments (line 421)
        
        # Call to len(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'self' (line 421)
        self_16827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 39), 'self', False)
        # Obtaining the member '_headers' of a type (line 421)
        _headers_16828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 39), self_16827, '_headers')
        # Processing the call keyword arguments (line 421)
        kwargs_16829 = {}
        # Getting the type of 'len' (line 421)
        len_16826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 35), 'len', False)
        # Calling len(args, kwargs) (line 421)
        len_call_result_16830 = invoke(stypy.reporting.localization.Localization(__file__, 421, 35), len_16826, *[_headers_16828], **kwargs_16829)
        
        # Processing the call keyword arguments (line 421)
        kwargs_16831 = {}
        # Getting the type of 'range' (line 421)
        range_16825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 29), 'range', False)
        # Calling range(args, kwargs) (line 421)
        range_call_result_16832 = invoke(stypy.reporting.localization.Localization(__file__, 421, 29), range_16825, *[len_call_result_16830], **kwargs_16831)
        
        # Getting the type of 'self' (line 421)
        self_16833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 56), 'self', False)
        # Obtaining the member '_headers' of a type (line 421)
        _headers_16834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 56), self_16833, '_headers')
        # Processing the call keyword arguments (line 421)
        kwargs_16835 = {}
        # Getting the type of 'zip' (line 421)
        zip_16824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 25), 'zip', False)
        # Calling zip(args, kwargs) (line 421)
        zip_call_result_16836 = invoke(stypy.reporting.localization.Localization(__file__, 421, 25), zip_16824, *[range_call_result_16832, _headers_16834], **kwargs_16835)
        
        # Assigning a type to the variable 'zip_call_result_16836' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'zip_call_result_16836', zip_call_result_16836)
        # Testing if the for loop is going to be iterated (line 421)
        # Testing the type of a for loop iterable (line 421)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 421, 8), zip_call_result_16836)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 421, 8), zip_call_result_16836):
            # Getting the type of the for loop variable (line 421)
            for_loop_var_16837 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 421, 8), zip_call_result_16836)
            # Assigning a type to the variable 'i' (line 421)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 8), for_loop_var_16837, 2, 0))
            # Assigning a type to the variable 'k' (line 421)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 8), for_loop_var_16837, 2, 1))
            # Assigning a type to the variable 'v' (line 421)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 8), for_loop_var_16837, 2, 1))
            # SSA begins for a for statement (line 421)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to lower(...): (line 422)
            # Processing the call keyword arguments (line 422)
            kwargs_16840 = {}
            # Getting the type of 'k' (line 422)
            k_16838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'k', False)
            # Obtaining the member 'lower' of a type (line 422)
            lower_16839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), k_16838, 'lower')
            # Calling lower(args, kwargs) (line 422)
            lower_call_result_16841 = invoke(stypy.reporting.localization.Localization(__file__, 422, 15), lower_16839, *[], **kwargs_16840)
            
            # Getting the type of '_name' (line 422)
            _name_16842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 28), '_name')
            # Applying the binary operator '==' (line 422)
            result_eq_16843 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 15), '==', lower_call_result_16841, _name_16842)
            
            # Testing if the type of an if condition is none (line 422)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 422, 12), result_eq_16843):
                pass
            else:
                
                # Testing the type of an if condition (line 422)
                if_condition_16844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 12), result_eq_16843)
                # Assigning a type to the variable 'if_condition_16844' (line 422)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'if_condition_16844', if_condition_16844)
                # SSA begins for if statement (line 422)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Tuple to a Subscript (line 423):
                
                # Assigning a Tuple to a Subscript (line 423):
                
                # Obtaining an instance of the builtin type 'tuple' (line 423)
                tuple_16845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 36), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 423)
                # Adding element type (line 423)
                # Getting the type of 'k' (line 423)
                k_16846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 36), 'k')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 36), tuple_16845, k_16846)
                # Adding element type (line 423)
                # Getting the type of '_value' (line 423)
                _value_16847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 39), '_value')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 36), tuple_16845, _value_16847)
                
                # Getting the type of 'self' (line 423)
                self_16848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'self')
                # Obtaining the member '_headers' of a type (line 423)
                _headers_16849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), self_16848, '_headers')
                # Getting the type of 'i' (line 423)
                i_16850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 30), 'i')
                # Storing an element on a container (line 423)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 16), _headers_16849, (i_16850, tuple_16845))
                # SSA join for if statement (line 422)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of a for statement (line 421)
            module_type_store.open_ssa_branch('for loop else')
            
            # Call to KeyError(...): (line 426)
            # Processing the call arguments (line 426)
            # Getting the type of '_name' (line 426)
            _name_16852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 27), '_name', False)
            # Processing the call keyword arguments (line 426)
            kwargs_16853 = {}
            # Getting the type of 'KeyError' (line 426)
            KeyError_16851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 18), 'KeyError', False)
            # Calling KeyError(args, kwargs) (line 426)
            KeyError_call_result_16854 = invoke(stypy.reporting.localization.Localization(__file__, 426, 18), KeyError_16851, *[_name_16852], **kwargs_16853)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 426, 12), KeyError_call_result_16854, 'raise parameter', BaseException)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
        else:
            
            # Call to KeyError(...): (line 426)
            # Processing the call arguments (line 426)
            # Getting the type of '_name' (line 426)
            _name_16852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 27), '_name', False)
            # Processing the call keyword arguments (line 426)
            kwargs_16853 = {}
            # Getting the type of 'KeyError' (line 426)
            KeyError_16851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 18), 'KeyError', False)
            # Calling KeyError(args, kwargs) (line 426)
            KeyError_call_result_16854 = invoke(stypy.reporting.localization.Localization(__file__, 426, 18), KeyError_16851, *[_name_16852], **kwargs_16853)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 426, 12), KeyError_call_result_16854, 'raise parameter', BaseException)

        
        
        # ################# End of 'replace_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'replace_header' in the type store
        # Getting the type of 'stypy_return_type' (line 413)
        stypy_return_type_16855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16855)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'replace_header'
        return stypy_return_type_16855


    @norecursion
    def get_content_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_content_type'
        module_type_store = module_type_store.open_function_context('get_content_type', 432, 4, False)
        # Assigning a type to the variable 'self' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_content_type.__dict__.__setitem__('stypy_localization', localization)
        Message.get_content_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_content_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_content_type.__dict__.__setitem__('stypy_function_name', 'Message.get_content_type')
        Message.get_content_type.__dict__.__setitem__('stypy_param_names_list', [])
        Message.get_content_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_content_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_content_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_content_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_content_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_content_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_content_type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_content_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_content_type(...)' code ##################

        str_16856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, (-1)), 'str', "Return the message's content type.\n\n        The returned string is coerced to lower case of the form\n        `maintype/subtype'.  If there was no Content-Type header in the\n        message, the default type as given by get_default_type() will be\n        returned.  Since according to RFC 2045, messages always have a default\n        type this will always return a value.\n\n        RFC 2045 defines a message's default type to be text/plain unless it\n        appears inside a multipart/digest container, in which case it would be\n        message/rfc822.\n        ")
        
        # Assigning a Call to a Name (line 445):
        
        # Assigning a Call to a Name (line 445):
        
        # Call to object(...): (line 445)
        # Processing the call keyword arguments (line 445)
        kwargs_16858 = {}
        # Getting the type of 'object' (line 445)
        object_16857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 18), 'object', False)
        # Calling object(args, kwargs) (line 445)
        object_call_result_16859 = invoke(stypy.reporting.localization.Localization(__file__, 445, 18), object_16857, *[], **kwargs_16858)
        
        # Assigning a type to the variable 'missing' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'missing', object_call_result_16859)
        
        # Assigning a Call to a Name (line 446):
        
        # Assigning a Call to a Name (line 446):
        
        # Call to get(...): (line 446)
        # Processing the call arguments (line 446)
        str_16862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 25), 'str', 'content-type')
        # Getting the type of 'missing' (line 446)
        missing_16863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 41), 'missing', False)
        # Processing the call keyword arguments (line 446)
        kwargs_16864 = {}
        # Getting the type of 'self' (line 446)
        self_16860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'self', False)
        # Obtaining the member 'get' of a type (line 446)
        get_16861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 16), self_16860, 'get')
        # Calling get(args, kwargs) (line 446)
        get_call_result_16865 = invoke(stypy.reporting.localization.Localization(__file__, 446, 16), get_16861, *[str_16862, missing_16863], **kwargs_16864)
        
        # Assigning a type to the variable 'value' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'value', get_call_result_16865)
        
        # Getting the type of 'value' (line 447)
        value_16866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 11), 'value')
        # Getting the type of 'missing' (line 447)
        missing_16867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 20), 'missing')
        # Applying the binary operator 'is' (line 447)
        result_is__16868 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 11), 'is', value_16866, missing_16867)
        
        # Testing if the type of an if condition is none (line 447)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 447, 8), result_is__16868):
            pass
        else:
            
            # Testing the type of an if condition (line 447)
            if_condition_16869 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 8), result_is__16868)
            # Assigning a type to the variable 'if_condition_16869' (line 447)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'if_condition_16869', if_condition_16869)
            # SSA begins for if statement (line 447)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to get_default_type(...): (line 449)
            # Processing the call keyword arguments (line 449)
            kwargs_16872 = {}
            # Getting the type of 'self' (line 449)
            self_16870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 19), 'self', False)
            # Obtaining the member 'get_default_type' of a type (line 449)
            get_default_type_16871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 19), self_16870, 'get_default_type')
            # Calling get_default_type(args, kwargs) (line 449)
            get_default_type_call_result_16873 = invoke(stypy.reporting.localization.Localization(__file__, 449, 19), get_default_type_16871, *[], **kwargs_16872)
            
            # Assigning a type to the variable 'stypy_return_type' (line 449)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'stypy_return_type', get_default_type_call_result_16873)
            # SSA join for if statement (line 447)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 450):
        
        # Assigning a Call to a Name (line 450):
        
        # Call to lower(...): (line 450)
        # Processing the call keyword arguments (line 450)
        kwargs_16882 = {}
        
        # Obtaining the type of the subscript
        int_16874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 35), 'int')
        
        # Call to _splitparam(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'value' (line 450)
        value_16876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 28), 'value', False)
        # Processing the call keyword arguments (line 450)
        kwargs_16877 = {}
        # Getting the type of '_splitparam' (line 450)
        _splitparam_16875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), '_splitparam', False)
        # Calling _splitparam(args, kwargs) (line 450)
        _splitparam_call_result_16878 = invoke(stypy.reporting.localization.Localization(__file__, 450, 16), _splitparam_16875, *[value_16876], **kwargs_16877)
        
        # Obtaining the member '__getitem__' of a type (line 450)
        getitem___16879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 16), _splitparam_call_result_16878, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 450)
        subscript_call_result_16880 = invoke(stypy.reporting.localization.Localization(__file__, 450, 16), getitem___16879, int_16874)
        
        # Obtaining the member 'lower' of a type (line 450)
        lower_16881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 16), subscript_call_result_16880, 'lower')
        # Calling lower(args, kwargs) (line 450)
        lower_call_result_16883 = invoke(stypy.reporting.localization.Localization(__file__, 450, 16), lower_16881, *[], **kwargs_16882)
        
        # Assigning a type to the variable 'ctype' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'ctype', lower_call_result_16883)
        
        
        # Call to count(...): (line 452)
        # Processing the call arguments (line 452)
        str_16886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 23), 'str', '/')
        # Processing the call keyword arguments (line 452)
        kwargs_16887 = {}
        # Getting the type of 'ctype' (line 452)
        ctype_16884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 11), 'ctype', False)
        # Obtaining the member 'count' of a type (line 452)
        count_16885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 11), ctype_16884, 'count')
        # Calling count(args, kwargs) (line 452)
        count_call_result_16888 = invoke(stypy.reporting.localization.Localization(__file__, 452, 11), count_16885, *[str_16886], **kwargs_16887)
        
        int_16889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 31), 'int')
        # Applying the binary operator '!=' (line 452)
        result_ne_16890 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 11), '!=', count_call_result_16888, int_16889)
        
        # Testing if the type of an if condition is none (line 452)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 452, 8), result_ne_16890):
            pass
        else:
            
            # Testing the type of an if condition (line 452)
            if_condition_16891 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 452, 8), result_ne_16890)
            # Assigning a type to the variable 'if_condition_16891' (line 452)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'if_condition_16891', if_condition_16891)
            # SSA begins for if statement (line 452)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_16892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 19), 'str', 'text/plain')
            # Assigning a type to the variable 'stypy_return_type' (line 453)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'stypy_return_type', str_16892)
            # SSA join for if statement (line 452)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'ctype' (line 454)
        ctype_16893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 15), 'ctype')
        # Assigning a type to the variable 'stypy_return_type' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'stypy_return_type', ctype_16893)
        
        # ################# End of 'get_content_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_content_type' in the type store
        # Getting the type of 'stypy_return_type' (line 432)
        stypy_return_type_16894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16894)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_content_type'
        return stypy_return_type_16894


    @norecursion
    def get_content_maintype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_content_maintype'
        module_type_store = module_type_store.open_function_context('get_content_maintype', 456, 4, False)
        # Assigning a type to the variable 'self' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_content_maintype.__dict__.__setitem__('stypy_localization', localization)
        Message.get_content_maintype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_content_maintype.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_content_maintype.__dict__.__setitem__('stypy_function_name', 'Message.get_content_maintype')
        Message.get_content_maintype.__dict__.__setitem__('stypy_param_names_list', [])
        Message.get_content_maintype.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_content_maintype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_content_maintype.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_content_maintype.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_content_maintype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_content_maintype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_content_maintype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_content_maintype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_content_maintype(...)' code ##################

        str_16895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, (-1)), 'str', "Return the message's main content type.\n\n        This is the `maintype' part of the string returned by\n        get_content_type().\n        ")
        
        # Assigning a Call to a Name (line 462):
        
        # Assigning a Call to a Name (line 462):
        
        # Call to get_content_type(...): (line 462)
        # Processing the call keyword arguments (line 462)
        kwargs_16898 = {}
        # Getting the type of 'self' (line 462)
        self_16896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 16), 'self', False)
        # Obtaining the member 'get_content_type' of a type (line 462)
        get_content_type_16897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 16), self_16896, 'get_content_type')
        # Calling get_content_type(args, kwargs) (line 462)
        get_content_type_call_result_16899 = invoke(stypy.reporting.localization.Localization(__file__, 462, 16), get_content_type_16897, *[], **kwargs_16898)
        
        # Assigning a type to the variable 'ctype' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'ctype', get_content_type_call_result_16899)
        
        # Obtaining the type of the subscript
        int_16900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 32), 'int')
        
        # Call to split(...): (line 463)
        # Processing the call arguments (line 463)
        str_16903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 27), 'str', '/')
        # Processing the call keyword arguments (line 463)
        kwargs_16904 = {}
        # Getting the type of 'ctype' (line 463)
        ctype_16901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 15), 'ctype', False)
        # Obtaining the member 'split' of a type (line 463)
        split_16902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 15), ctype_16901, 'split')
        # Calling split(args, kwargs) (line 463)
        split_call_result_16905 = invoke(stypy.reporting.localization.Localization(__file__, 463, 15), split_16902, *[str_16903], **kwargs_16904)
        
        # Obtaining the member '__getitem__' of a type (line 463)
        getitem___16906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 15), split_call_result_16905, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 463)
        subscript_call_result_16907 = invoke(stypy.reporting.localization.Localization(__file__, 463, 15), getitem___16906, int_16900)
        
        # Assigning a type to the variable 'stypy_return_type' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'stypy_return_type', subscript_call_result_16907)
        
        # ################# End of 'get_content_maintype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_content_maintype' in the type store
        # Getting the type of 'stypy_return_type' (line 456)
        stypy_return_type_16908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16908)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_content_maintype'
        return stypy_return_type_16908


    @norecursion
    def get_content_subtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_content_subtype'
        module_type_store = module_type_store.open_function_context('get_content_subtype', 465, 4, False)
        # Assigning a type to the variable 'self' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_content_subtype.__dict__.__setitem__('stypy_localization', localization)
        Message.get_content_subtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_content_subtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_content_subtype.__dict__.__setitem__('stypy_function_name', 'Message.get_content_subtype')
        Message.get_content_subtype.__dict__.__setitem__('stypy_param_names_list', [])
        Message.get_content_subtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_content_subtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_content_subtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_content_subtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_content_subtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_content_subtype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_content_subtype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_content_subtype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_content_subtype(...)' code ##################

        str_16909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, (-1)), 'str', "Returns the message's sub-content type.\n\n        This is the `subtype' part of the string returned by\n        get_content_type().\n        ")
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Call to get_content_type(...): (line 471)
        # Processing the call keyword arguments (line 471)
        kwargs_16912 = {}
        # Getting the type of 'self' (line 471)
        self_16910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'self', False)
        # Obtaining the member 'get_content_type' of a type (line 471)
        get_content_type_16911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 16), self_16910, 'get_content_type')
        # Calling get_content_type(args, kwargs) (line 471)
        get_content_type_call_result_16913 = invoke(stypy.reporting.localization.Localization(__file__, 471, 16), get_content_type_16911, *[], **kwargs_16912)
        
        # Assigning a type to the variable 'ctype' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'ctype', get_content_type_call_result_16913)
        
        # Obtaining the type of the subscript
        int_16914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 32), 'int')
        
        # Call to split(...): (line 472)
        # Processing the call arguments (line 472)
        str_16917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 27), 'str', '/')
        # Processing the call keyword arguments (line 472)
        kwargs_16918 = {}
        # Getting the type of 'ctype' (line 472)
        ctype_16915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'ctype', False)
        # Obtaining the member 'split' of a type (line 472)
        split_16916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 15), ctype_16915, 'split')
        # Calling split(args, kwargs) (line 472)
        split_call_result_16919 = invoke(stypy.reporting.localization.Localization(__file__, 472, 15), split_16916, *[str_16917], **kwargs_16918)
        
        # Obtaining the member '__getitem__' of a type (line 472)
        getitem___16920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 15), split_call_result_16919, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 472)
        subscript_call_result_16921 = invoke(stypy.reporting.localization.Localization(__file__, 472, 15), getitem___16920, int_16914)
        
        # Assigning a type to the variable 'stypy_return_type' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'stypy_return_type', subscript_call_result_16921)
        
        # ################# End of 'get_content_subtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_content_subtype' in the type store
        # Getting the type of 'stypy_return_type' (line 465)
        stypy_return_type_16922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16922)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_content_subtype'
        return stypy_return_type_16922


    @norecursion
    def get_default_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_default_type'
        module_type_store = module_type_store.open_function_context('get_default_type', 474, 4, False)
        # Assigning a type to the variable 'self' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_default_type.__dict__.__setitem__('stypy_localization', localization)
        Message.get_default_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_default_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_default_type.__dict__.__setitem__('stypy_function_name', 'Message.get_default_type')
        Message.get_default_type.__dict__.__setitem__('stypy_param_names_list', [])
        Message.get_default_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_default_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_default_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_default_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_default_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_default_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_default_type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_default_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_default_type(...)' code ##################

        str_16923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, (-1)), 'str', "Return the `default' content type.\n\n        Most messages have a default content type of text/plain, except for\n        messages that are subparts of multipart/digest containers.  Such\n        subparts have a default content type of message/rfc822.\n        ")
        # Getting the type of 'self' (line 481)
        self_16924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), 'self')
        # Obtaining the member '_default_type' of a type (line 481)
        _default_type_16925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 15), self_16924, '_default_type')
        # Assigning a type to the variable 'stypy_return_type' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'stypy_return_type', _default_type_16925)
        
        # ################# End of 'get_default_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_default_type' in the type store
        # Getting the type of 'stypy_return_type' (line 474)
        stypy_return_type_16926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16926)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_default_type'
        return stypy_return_type_16926


    @norecursion
    def set_default_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_default_type'
        module_type_store = module_type_store.open_function_context('set_default_type', 483, 4, False)
        # Assigning a type to the variable 'self' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.set_default_type.__dict__.__setitem__('stypy_localization', localization)
        Message.set_default_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.set_default_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.set_default_type.__dict__.__setitem__('stypy_function_name', 'Message.set_default_type')
        Message.set_default_type.__dict__.__setitem__('stypy_param_names_list', ['ctype'])
        Message.set_default_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.set_default_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.set_default_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.set_default_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.set_default_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.set_default_type.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.set_default_type', ['ctype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_default_type', localization, ['ctype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_default_type(...)' code ##################

        str_16927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, (-1)), 'str', 'Set the `default\' content type.\n\n        ctype should be either "text/plain" or "message/rfc822", although this\n        is not enforced.  The default content type is not stored in the\n        Content-Type header.\n        ')
        
        # Assigning a Name to a Attribute (line 490):
        
        # Assigning a Name to a Attribute (line 490):
        # Getting the type of 'ctype' (line 490)
        ctype_16928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 29), 'ctype')
        # Getting the type of 'self' (line 490)
        self_16929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'self')
        # Setting the type of the member '_default_type' of a type (line 490)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), self_16929, '_default_type', ctype_16928)
        
        # ################# End of 'set_default_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_default_type' in the type store
        # Getting the type of 'stypy_return_type' (line 483)
        stypy_return_type_16930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16930)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_default_type'
        return stypy_return_type_16930


    @norecursion
    def _get_params_preserve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_params_preserve'
        module_type_store = module_type_store.open_function_context('_get_params_preserve', 492, 4, False)
        # Assigning a type to the variable 'self' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message._get_params_preserve.__dict__.__setitem__('stypy_localization', localization)
        Message._get_params_preserve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message._get_params_preserve.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message._get_params_preserve.__dict__.__setitem__('stypy_function_name', 'Message._get_params_preserve')
        Message._get_params_preserve.__dict__.__setitem__('stypy_param_names_list', ['failobj', 'header'])
        Message._get_params_preserve.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message._get_params_preserve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message._get_params_preserve.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message._get_params_preserve.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message._get_params_preserve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message._get_params_preserve.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message._get_params_preserve', ['failobj', 'header'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_params_preserve', localization, ['failobj', 'header'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_params_preserve(...)' code ##################

        
        # Assigning a Call to a Name (line 495):
        
        # Assigning a Call to a Name (line 495):
        
        # Call to object(...): (line 495)
        # Processing the call keyword arguments (line 495)
        kwargs_16932 = {}
        # Getting the type of 'object' (line 495)
        object_16931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 18), 'object', False)
        # Calling object(args, kwargs) (line 495)
        object_call_result_16933 = invoke(stypy.reporting.localization.Localization(__file__, 495, 18), object_16931, *[], **kwargs_16932)
        
        # Assigning a type to the variable 'missing' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'missing', object_call_result_16933)
        
        # Assigning a Call to a Name (line 496):
        
        # Assigning a Call to a Name (line 496):
        
        # Call to get(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'header' (line 496)
        header_16936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 25), 'header', False)
        # Getting the type of 'missing' (line 496)
        missing_16937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 33), 'missing', False)
        # Processing the call keyword arguments (line 496)
        kwargs_16938 = {}
        # Getting the type of 'self' (line 496)
        self_16934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 16), 'self', False)
        # Obtaining the member 'get' of a type (line 496)
        get_16935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 16), self_16934, 'get')
        # Calling get(args, kwargs) (line 496)
        get_call_result_16939 = invoke(stypy.reporting.localization.Localization(__file__, 496, 16), get_16935, *[header_16936, missing_16937], **kwargs_16938)
        
        # Assigning a type to the variable 'value' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'value', get_call_result_16939)
        
        # Getting the type of 'value' (line 497)
        value_16940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 11), 'value')
        # Getting the type of 'missing' (line 497)
        missing_16941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 20), 'missing')
        # Applying the binary operator 'is' (line 497)
        result_is__16942 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 11), 'is', value_16940, missing_16941)
        
        # Testing if the type of an if condition is none (line 497)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 497, 8), result_is__16942):
            pass
        else:
            
            # Testing the type of an if condition (line 497)
            if_condition_16943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 8), result_is__16942)
            # Assigning a type to the variable 'if_condition_16943' (line 497)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'if_condition_16943', if_condition_16943)
            # SSA begins for if statement (line 497)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'failobj' (line 498)
            failobj_16944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 19), 'failobj')
            # Assigning a type to the variable 'stypy_return_type' (line 498)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'stypy_return_type', failobj_16944)
            # SSA join for if statement (line 497)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a List to a Name (line 499):
        
        # Assigning a List to a Name (line 499):
        
        # Obtaining an instance of the builtin type 'list' (line 499)
        list_16945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 499)
        
        # Assigning a type to the variable 'params' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'params', list_16945)
        
        
        # Call to _parseparam(...): (line 500)
        # Processing the call arguments (line 500)
        str_16947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 29), 'str', ';')
        # Getting the type of 'value' (line 500)
        value_16948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 35), 'value', False)
        # Applying the binary operator '+' (line 500)
        result_add_16949 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 29), '+', str_16947, value_16948)
        
        # Processing the call keyword arguments (line 500)
        kwargs_16950 = {}
        # Getting the type of '_parseparam' (line 500)
        _parseparam_16946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 17), '_parseparam', False)
        # Calling _parseparam(args, kwargs) (line 500)
        _parseparam_call_result_16951 = invoke(stypy.reporting.localization.Localization(__file__, 500, 17), _parseparam_16946, *[result_add_16949], **kwargs_16950)
        
        # Assigning a type to the variable '_parseparam_call_result_16951' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), '_parseparam_call_result_16951', _parseparam_call_result_16951)
        # Testing if the for loop is going to be iterated (line 500)
        # Testing the type of a for loop iterable (line 500)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 500, 8), _parseparam_call_result_16951)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 500, 8), _parseparam_call_result_16951):
            # Getting the type of the for loop variable (line 500)
            for_loop_var_16952 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 500, 8), _parseparam_call_result_16951)
            # Assigning a type to the variable 'p' (line 500)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'p', for_loop_var_16952)
            # SSA begins for a for statement (line 500)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # SSA begins for try-except statement (line 501)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Tuple (line 502):
            
            # Assigning a Call to a Name:
            
            # Call to split(...): (line 502)
            # Processing the call arguments (line 502)
            str_16955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 36), 'str', '=')
            int_16956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 41), 'int')
            # Processing the call keyword arguments (line 502)
            kwargs_16957 = {}
            # Getting the type of 'p' (line 502)
            p_16953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 28), 'p', False)
            # Obtaining the member 'split' of a type (line 502)
            split_16954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 28), p_16953, 'split')
            # Calling split(args, kwargs) (line 502)
            split_call_result_16958 = invoke(stypy.reporting.localization.Localization(__file__, 502, 28), split_16954, *[str_16955, int_16956], **kwargs_16957)
            
            # Assigning a type to the variable 'call_assignment_16046' (line 502)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'call_assignment_16046', split_call_result_16958)
            
            # Assigning a Call to a Name (line 502):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_16046' (line 502)
            call_assignment_16046_16959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'call_assignment_16046', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_16960 = stypy_get_value_from_tuple(call_assignment_16046_16959, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_16047' (line 502)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'call_assignment_16047', stypy_get_value_from_tuple_call_result_16960)
            
            # Assigning a Name to a Name (line 502):
            # Getting the type of 'call_assignment_16047' (line 502)
            call_assignment_16047_16961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'call_assignment_16047')
            # Assigning a type to the variable 'name' (line 502)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'name', call_assignment_16047_16961)
            
            # Assigning a Call to a Name (line 502):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_16046' (line 502)
            call_assignment_16046_16962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'call_assignment_16046', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_16963 = stypy_get_value_from_tuple(call_assignment_16046_16962, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_16048' (line 502)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'call_assignment_16048', stypy_get_value_from_tuple_call_result_16963)
            
            # Assigning a Name to a Name (line 502):
            # Getting the type of 'call_assignment_16048' (line 502)
            call_assignment_16048_16964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'call_assignment_16048')
            # Assigning a type to the variable 'val' (line 502)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 22), 'val', call_assignment_16048_16964)
            
            # Assigning a Call to a Name (line 503):
            
            # Assigning a Call to a Name (line 503):
            
            # Call to strip(...): (line 503)
            # Processing the call keyword arguments (line 503)
            kwargs_16967 = {}
            # Getting the type of 'name' (line 503)
            name_16965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 23), 'name', False)
            # Obtaining the member 'strip' of a type (line 503)
            strip_16966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 23), name_16965, 'strip')
            # Calling strip(args, kwargs) (line 503)
            strip_call_result_16968 = invoke(stypy.reporting.localization.Localization(__file__, 503, 23), strip_16966, *[], **kwargs_16967)
            
            # Assigning a type to the variable 'name' (line 503)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 16), 'name', strip_call_result_16968)
            
            # Assigning a Call to a Name (line 504):
            
            # Assigning a Call to a Name (line 504):
            
            # Call to strip(...): (line 504)
            # Processing the call keyword arguments (line 504)
            kwargs_16971 = {}
            # Getting the type of 'val' (line 504)
            val_16969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 22), 'val', False)
            # Obtaining the member 'strip' of a type (line 504)
            strip_16970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 22), val_16969, 'strip')
            # Calling strip(args, kwargs) (line 504)
            strip_call_result_16972 = invoke(stypy.reporting.localization.Localization(__file__, 504, 22), strip_16970, *[], **kwargs_16971)
            
            # Assigning a type to the variable 'val' (line 504)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 16), 'val', strip_call_result_16972)
            # SSA branch for the except part of a try statement (line 501)
            # SSA branch for the except 'ValueError' branch of a try statement (line 501)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Call to a Name (line 507):
            
            # Assigning a Call to a Name (line 507):
            
            # Call to strip(...): (line 507)
            # Processing the call keyword arguments (line 507)
            kwargs_16975 = {}
            # Getting the type of 'p' (line 507)
            p_16973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 23), 'p', False)
            # Obtaining the member 'strip' of a type (line 507)
            strip_16974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 23), p_16973, 'strip')
            # Calling strip(args, kwargs) (line 507)
            strip_call_result_16976 = invoke(stypy.reporting.localization.Localization(__file__, 507, 23), strip_16974, *[], **kwargs_16975)
            
            # Assigning a type to the variable 'name' (line 507)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 16), 'name', strip_call_result_16976)
            
            # Assigning a Str to a Name (line 508):
            
            # Assigning a Str to a Name (line 508):
            str_16977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 22), 'str', '')
            # Assigning a type to the variable 'val' (line 508)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'val', str_16977)
            # SSA join for try-except statement (line 501)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to append(...): (line 509)
            # Processing the call arguments (line 509)
            
            # Obtaining an instance of the builtin type 'tuple' (line 509)
            tuple_16980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 509)
            # Adding element type (line 509)
            # Getting the type of 'name' (line 509)
            name_16981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 27), 'name', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 27), tuple_16980, name_16981)
            # Adding element type (line 509)
            # Getting the type of 'val' (line 509)
            val_16982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 33), 'val', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 27), tuple_16980, val_16982)
            
            # Processing the call keyword arguments (line 509)
            kwargs_16983 = {}
            # Getting the type of 'params' (line 509)
            params_16978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'params', False)
            # Obtaining the member 'append' of a type (line 509)
            append_16979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 12), params_16978, 'append')
            # Calling append(args, kwargs) (line 509)
            append_call_result_16984 = invoke(stypy.reporting.localization.Localization(__file__, 509, 12), append_16979, *[tuple_16980], **kwargs_16983)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 510):
        
        # Assigning a Call to a Name (line 510):
        
        # Call to decode_params(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'params' (line 510)
        params_16987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 37), 'params', False)
        # Processing the call keyword arguments (line 510)
        kwargs_16988 = {}
        # Getting the type of 'utils' (line 510)
        utils_16985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 17), 'utils', False)
        # Obtaining the member 'decode_params' of a type (line 510)
        decode_params_16986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 17), utils_16985, 'decode_params')
        # Calling decode_params(args, kwargs) (line 510)
        decode_params_call_result_16989 = invoke(stypy.reporting.localization.Localization(__file__, 510, 17), decode_params_16986, *[params_16987], **kwargs_16988)
        
        # Assigning a type to the variable 'params' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'params', decode_params_call_result_16989)
        # Getting the type of 'params' (line 511)
        params_16990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 15), 'params')
        # Assigning a type to the variable 'stypy_return_type' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'stypy_return_type', params_16990)
        
        # ################# End of '_get_params_preserve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_params_preserve' in the type store
        # Getting the type of 'stypy_return_type' (line 492)
        stypy_return_type_16991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16991)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_params_preserve'
        return stypy_return_type_16991


    @norecursion
    def get_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 513)
        None_16992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 33), 'None')
        str_16993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 46), 'str', 'content-type')
        # Getting the type of 'True' (line 513)
        True_16994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 70), 'True')
        defaults = [None_16992, str_16993, True_16994]
        # Create a new context for function 'get_params'
        module_type_store = module_type_store.open_function_context('get_params', 513, 4, False)
        # Assigning a type to the variable 'self' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_params.__dict__.__setitem__('stypy_localization', localization)
        Message.get_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_params.__dict__.__setitem__('stypy_function_name', 'Message.get_params')
        Message.get_params.__dict__.__setitem__('stypy_param_names_list', ['failobj', 'header', 'unquote'])
        Message.get_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_params.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_params', ['failobj', 'header', 'unquote'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_params', localization, ['failobj', 'header', 'unquote'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_params(...)' code ##################

        str_16995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, (-1)), 'str', "Return the message's Content-Type parameters, as a list.\n\n        The elements of the returned list are 2-tuples of key/value pairs, as\n        split on the `=' sign.  The left hand side of the `=' is the key,\n        while the right hand side is the value.  If there is no `=' sign in\n        the parameter the value is the empty string.  The value is as\n        described in the get_param() method.\n\n        Optional failobj is the object to return if there is no Content-Type\n        header.  Optional header is the header to search instead of\n        Content-Type.  If unquote is True, the value is unquoted.\n        ")
        
        # Assigning a Call to a Name (line 526):
        
        # Assigning a Call to a Name (line 526):
        
        # Call to object(...): (line 526)
        # Processing the call keyword arguments (line 526)
        kwargs_16997 = {}
        # Getting the type of 'object' (line 526)
        object_16996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 18), 'object', False)
        # Calling object(args, kwargs) (line 526)
        object_call_result_16998 = invoke(stypy.reporting.localization.Localization(__file__, 526, 18), object_16996, *[], **kwargs_16997)
        
        # Assigning a type to the variable 'missing' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'missing', object_call_result_16998)
        
        # Assigning a Call to a Name (line 527):
        
        # Assigning a Call to a Name (line 527):
        
        # Call to _get_params_preserve(...): (line 527)
        # Processing the call arguments (line 527)
        # Getting the type of 'missing' (line 527)
        missing_17001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 43), 'missing', False)
        # Getting the type of 'header' (line 527)
        header_17002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 52), 'header', False)
        # Processing the call keyword arguments (line 527)
        kwargs_17003 = {}
        # Getting the type of 'self' (line 527)
        self_16999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 17), 'self', False)
        # Obtaining the member '_get_params_preserve' of a type (line 527)
        _get_params_preserve_17000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 17), self_16999, '_get_params_preserve')
        # Calling _get_params_preserve(args, kwargs) (line 527)
        _get_params_preserve_call_result_17004 = invoke(stypy.reporting.localization.Localization(__file__, 527, 17), _get_params_preserve_17000, *[missing_17001, header_17002], **kwargs_17003)
        
        # Assigning a type to the variable 'params' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'params', _get_params_preserve_call_result_17004)
        
        # Getting the type of 'params' (line 528)
        params_17005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 11), 'params')
        # Getting the type of 'missing' (line 528)
        missing_17006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 21), 'missing')
        # Applying the binary operator 'is' (line 528)
        result_is__17007 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 11), 'is', params_17005, missing_17006)
        
        # Testing if the type of an if condition is none (line 528)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 528, 8), result_is__17007):
            pass
        else:
            
            # Testing the type of an if condition (line 528)
            if_condition_17008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 8), result_is__17007)
            # Assigning a type to the variable 'if_condition_17008' (line 528)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'if_condition_17008', if_condition_17008)
            # SSA begins for if statement (line 528)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'failobj' (line 529)
            failobj_17009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 19), 'failobj')
            # Assigning a type to the variable 'stypy_return_type' (line 529)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'stypy_return_type', failobj_17009)
            # SSA join for if statement (line 528)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'unquote' (line 530)
        unquote_17010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 11), 'unquote')
        # Testing if the type of an if condition is none (line 530)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 530, 8), unquote_17010):
            # Getting the type of 'params' (line 533)
            params_17021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 19), 'params')
            # Assigning a type to the variable 'stypy_return_type' (line 533)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'stypy_return_type', params_17021)
        else:
            
            # Testing the type of an if condition (line 530)
            if_condition_17011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 530, 8), unquote_17010)
            # Assigning a type to the variable 'if_condition_17011' (line 530)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'if_condition_17011', if_condition_17011)
            # SSA begins for if statement (line 530)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'params' (line 531)
            params_17018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 54), 'params')
            comprehension_17019 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 20), params_17018)
            # Assigning a type to the variable 'k' (line 531)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 20), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 20), comprehension_17019))
            # Assigning a type to the variable 'v' (line 531)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 20), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 20), comprehension_17019))
            
            # Obtaining an instance of the builtin type 'tuple' (line 531)
            tuple_17012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 531)
            # Adding element type (line 531)
            # Getting the type of 'k' (line 531)
            k_17013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 21), 'k')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 21), tuple_17012, k_17013)
            # Adding element type (line 531)
            
            # Call to _unquotevalue(...): (line 531)
            # Processing the call arguments (line 531)
            # Getting the type of 'v' (line 531)
            v_17015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 38), 'v', False)
            # Processing the call keyword arguments (line 531)
            kwargs_17016 = {}
            # Getting the type of '_unquotevalue' (line 531)
            _unquotevalue_17014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 24), '_unquotevalue', False)
            # Calling _unquotevalue(args, kwargs) (line 531)
            _unquotevalue_call_result_17017 = invoke(stypy.reporting.localization.Localization(__file__, 531, 24), _unquotevalue_17014, *[v_17015], **kwargs_17016)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 21), tuple_17012, _unquotevalue_call_result_17017)
            
            list_17020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 20), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 20), list_17020, tuple_17012)
            # Assigning a type to the variable 'stypy_return_type' (line 531)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'stypy_return_type', list_17020)
            # SSA branch for the else part of an if statement (line 530)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'params' (line 533)
            params_17021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 19), 'params')
            # Assigning a type to the variable 'stypy_return_type' (line 533)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'stypy_return_type', params_17021)
            # SSA join for if statement (line 530)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'get_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_params' in the type store
        # Getting the type of 'stypy_return_type' (line 513)
        stypy_return_type_17022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17022)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_params'
        return stypy_return_type_17022


    @norecursion
    def get_param(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 535)
        None_17023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 39), 'None')
        str_17024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 52), 'str', 'content-type')
        # Getting the type of 'True' (line 536)
        True_17025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 26), 'True')
        defaults = [None_17023, str_17024, True_17025]
        # Create a new context for function 'get_param'
        module_type_store = module_type_store.open_function_context('get_param', 535, 4, False)
        # Assigning a type to the variable 'self' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_param.__dict__.__setitem__('stypy_localization', localization)
        Message.get_param.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_param.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_param.__dict__.__setitem__('stypy_function_name', 'Message.get_param')
        Message.get_param.__dict__.__setitem__('stypy_param_names_list', ['param', 'failobj', 'header', 'unquote'])
        Message.get_param.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_param.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_param.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_param.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_param.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_param.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_param', ['param', 'failobj', 'header', 'unquote'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_param', localization, ['param', 'failobj', 'header', 'unquote'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_param(...)' code ##################

        str_17026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, (-1)), 'str', "Return the parameter value if found in the Content-Type header.\n\n        Optional failobj is the object to return if there is no Content-Type\n        header, or the Content-Type header has no such parameter.  Optional\n        header is the header to search instead of Content-Type.\n\n        Parameter keys are always compared case insensitively.  The return\n        value can either be a string, or a 3-tuple if the parameter was RFC\n        2231 encoded.  When it's a 3-tuple, the elements of the value are of\n        the form (CHARSET, LANGUAGE, VALUE).  Note that both CHARSET and\n        LANGUAGE can be None, in which case you should consider VALUE to be\n        encoded in the us-ascii charset.  You can usually ignore LANGUAGE.\n\n        Your application should be prepared to deal with 3-tuple return\n        values, and can convert the parameter to a Unicode string like so:\n\n            param = msg.get_param('foo')\n            if isinstance(param, tuple):\n                param = unicode(param[2], param[0] or 'us-ascii')\n\n        In any case, the parameter value (either the returned string, or the\n        VALUE item in the 3-tuple) is always unquoted, unless unquote is set\n        to False.\n        ")
        
        # Getting the type of 'header' (line 561)
        header_17027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 11), 'header')
        # Getting the type of 'self' (line 561)
        self_17028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 25), 'self')
        # Applying the binary operator 'notin' (line 561)
        result_contains_17029 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 11), 'notin', header_17027, self_17028)
        
        # Testing if the type of an if condition is none (line 561)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 561, 8), result_contains_17029):
            pass
        else:
            
            # Testing the type of an if condition (line 561)
            if_condition_17030 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 561, 8), result_contains_17029)
            # Assigning a type to the variable 'if_condition_17030' (line 561)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'if_condition_17030', if_condition_17030)
            # SSA begins for if statement (line 561)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'failobj' (line 562)
            failobj_17031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 19), 'failobj')
            # Assigning a type to the variable 'stypy_return_type' (line 562)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'stypy_return_type', failobj_17031)
            # SSA join for if statement (line 561)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to _get_params_preserve(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'failobj' (line 563)
        failobj_17034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 46), 'failobj', False)
        # Getting the type of 'header' (line 563)
        header_17035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 55), 'header', False)
        # Processing the call keyword arguments (line 563)
        kwargs_17036 = {}
        # Getting the type of 'self' (line 563)
        self_17032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 20), 'self', False)
        # Obtaining the member '_get_params_preserve' of a type (line 563)
        _get_params_preserve_17033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 20), self_17032, '_get_params_preserve')
        # Calling _get_params_preserve(args, kwargs) (line 563)
        _get_params_preserve_call_result_17037 = invoke(stypy.reporting.localization.Localization(__file__, 563, 20), _get_params_preserve_17033, *[failobj_17034, header_17035], **kwargs_17036)
        
        # Assigning a type to the variable '_get_params_preserve_call_result_17037' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), '_get_params_preserve_call_result_17037', _get_params_preserve_call_result_17037)
        # Testing if the for loop is going to be iterated (line 563)
        # Testing the type of a for loop iterable (line 563)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 563, 8), _get_params_preserve_call_result_17037)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 563, 8), _get_params_preserve_call_result_17037):
            # Getting the type of the for loop variable (line 563)
            for_loop_var_17038 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 563, 8), _get_params_preserve_call_result_17037)
            # Assigning a type to the variable 'k' (line 563)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 8), for_loop_var_17038, 2, 0))
            # Assigning a type to the variable 'v' (line 563)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 8), for_loop_var_17038, 2, 1))
            # SSA begins for a for statement (line 563)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to lower(...): (line 564)
            # Processing the call keyword arguments (line 564)
            kwargs_17041 = {}
            # Getting the type of 'k' (line 564)
            k_17039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 15), 'k', False)
            # Obtaining the member 'lower' of a type (line 564)
            lower_17040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 15), k_17039, 'lower')
            # Calling lower(args, kwargs) (line 564)
            lower_call_result_17042 = invoke(stypy.reporting.localization.Localization(__file__, 564, 15), lower_17040, *[], **kwargs_17041)
            
            
            # Call to lower(...): (line 564)
            # Processing the call keyword arguments (line 564)
            kwargs_17045 = {}
            # Getting the type of 'param' (line 564)
            param_17043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 28), 'param', False)
            # Obtaining the member 'lower' of a type (line 564)
            lower_17044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 28), param_17043, 'lower')
            # Calling lower(args, kwargs) (line 564)
            lower_call_result_17046 = invoke(stypy.reporting.localization.Localization(__file__, 564, 28), lower_17044, *[], **kwargs_17045)
            
            # Applying the binary operator '==' (line 564)
            result_eq_17047 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 15), '==', lower_call_result_17042, lower_call_result_17046)
            
            # Testing if the type of an if condition is none (line 564)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 564, 12), result_eq_17047):
                pass
            else:
                
                # Testing the type of an if condition (line 564)
                if_condition_17048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 564, 12), result_eq_17047)
                # Assigning a type to the variable 'if_condition_17048' (line 564)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'if_condition_17048', if_condition_17048)
                # SSA begins for if statement (line 564)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'unquote' (line 565)
                unquote_17049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 19), 'unquote')
                # Testing if the type of an if condition is none (line 565)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 565, 16), unquote_17049):
                    # Getting the type of 'v' (line 568)
                    v_17055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 27), 'v')
                    # Assigning a type to the variable 'stypy_return_type' (line 568)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 20), 'stypy_return_type', v_17055)
                else:
                    
                    # Testing the type of an if condition (line 565)
                    if_condition_17050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 565, 16), unquote_17049)
                    # Assigning a type to the variable 'if_condition_17050' (line 565)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'if_condition_17050', if_condition_17050)
                    # SSA begins for if statement (line 565)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to _unquotevalue(...): (line 566)
                    # Processing the call arguments (line 566)
                    # Getting the type of 'v' (line 566)
                    v_17052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 41), 'v', False)
                    # Processing the call keyword arguments (line 566)
                    kwargs_17053 = {}
                    # Getting the type of '_unquotevalue' (line 566)
                    _unquotevalue_17051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 27), '_unquotevalue', False)
                    # Calling _unquotevalue(args, kwargs) (line 566)
                    _unquotevalue_call_result_17054 = invoke(stypy.reporting.localization.Localization(__file__, 566, 27), _unquotevalue_17051, *[v_17052], **kwargs_17053)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 566)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 20), 'stypy_return_type', _unquotevalue_call_result_17054)
                    # SSA branch for the else part of an if statement (line 565)
                    module_type_store.open_ssa_branch('else')
                    # Getting the type of 'v' (line 568)
                    v_17055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 27), 'v')
                    # Assigning a type to the variable 'stypy_return_type' (line 568)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 20), 'stypy_return_type', v_17055)
                    # SSA join for if statement (line 565)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 564)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'failobj' (line 569)
        failobj_17056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 15), 'failobj')
        # Assigning a type to the variable 'stypy_return_type' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'stypy_return_type', failobj_17056)
        
        # ################# End of 'get_param(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_param' in the type store
        # Getting the type of 'stypy_return_type' (line 535)
        stypy_return_type_17057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17057)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_param'
        return stypy_return_type_17057


    @norecursion
    def set_param(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_17058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 45), 'str', 'Content-Type')
        # Getting the type of 'True' (line 571)
        True_17059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 69), 'True')
        # Getting the type of 'None' (line 572)
        None_17060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 26), 'None')
        str_17061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 41), 'str', '')
        defaults = [str_17058, True_17059, None_17060, str_17061]
        # Create a new context for function 'set_param'
        module_type_store = module_type_store.open_function_context('set_param', 571, 4, False)
        # Assigning a type to the variable 'self' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.set_param.__dict__.__setitem__('stypy_localization', localization)
        Message.set_param.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.set_param.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.set_param.__dict__.__setitem__('stypy_function_name', 'Message.set_param')
        Message.set_param.__dict__.__setitem__('stypy_param_names_list', ['param', 'value', 'header', 'requote', 'charset', 'language'])
        Message.set_param.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.set_param.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.set_param.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.set_param.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.set_param.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.set_param.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.set_param', ['param', 'value', 'header', 'requote', 'charset', 'language'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_param', localization, ['param', 'value', 'header', 'requote', 'charset', 'language'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_param(...)' code ##################

        str_17062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, (-1)), 'str', 'Set a parameter in the Content-Type header.\n\n        If the parameter already exists in the header, its value will be\n        replaced with the new value.\n\n        If header is Content-Type and has not yet been defined for this\n        message, it will be set to "text/plain" and the new parameter and\n        value will be appended as per RFC 2045.\n\n        An alternate header can be specified in the header argument, and all\n        parameters will be quoted as necessary unless requote is False.\n\n        If charset is specified, the parameter will be encoded according to RFC\n        2231.  Optional language specifies the RFC 2231 language, defaulting\n        to the empty string.  Both charset and language should be strings.\n        ')
        
        # Evaluating a boolean operation
        
        
        # Call to isinstance(...): (line 589)
        # Processing the call arguments (line 589)
        # Getting the type of 'value' (line 589)
        value_17064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 26), 'value', False)
        # Getting the type of 'tuple' (line 589)
        tuple_17065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 33), 'tuple', False)
        # Processing the call keyword arguments (line 589)
        kwargs_17066 = {}
        # Getting the type of 'isinstance' (line 589)
        isinstance_17063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 589)
        isinstance_call_result_17067 = invoke(stypy.reporting.localization.Localization(__file__, 589, 15), isinstance_17063, *[value_17064, tuple_17065], **kwargs_17066)
        
        # Applying the 'not' unary operator (line 589)
        result_not__17068 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 11), 'not', isinstance_call_result_17067)
        
        # Getting the type of 'charset' (line 589)
        charset_17069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 44), 'charset')
        # Applying the binary operator 'and' (line 589)
        result_and_keyword_17070 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 11), 'and', result_not__17068, charset_17069)
        
        # Testing if the type of an if condition is none (line 589)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 589, 8), result_and_keyword_17070):
            pass
        else:
            
            # Testing the type of an if condition (line 589)
            if_condition_17071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 589, 8), result_and_keyword_17070)
            # Assigning a type to the variable 'if_condition_17071' (line 589)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'if_condition_17071', if_condition_17071)
            # SSA begins for if statement (line 589)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Name (line 590):
            
            # Assigning a Tuple to a Name (line 590):
            
            # Obtaining an instance of the builtin type 'tuple' (line 590)
            tuple_17072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 590)
            # Adding element type (line 590)
            # Getting the type of 'charset' (line 590)
            charset_17073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 21), 'charset')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 21), tuple_17072, charset_17073)
            # Adding element type (line 590)
            # Getting the type of 'language' (line 590)
            language_17074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 30), 'language')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 21), tuple_17072, language_17074)
            # Adding element type (line 590)
            # Getting the type of 'value' (line 590)
            value_17075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 40), 'value')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 21), tuple_17072, value_17075)
            
            # Assigning a type to the variable 'value' (line 590)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'value', tuple_17072)
            # SSA join for if statement (line 589)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Getting the type of 'header' (line 592)
        header_17076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 11), 'header')
        # Getting the type of 'self' (line 592)
        self_17077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 25), 'self')
        # Applying the binary operator 'notin' (line 592)
        result_contains_17078 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 11), 'notin', header_17076, self_17077)
        
        
        
        # Call to lower(...): (line 592)
        # Processing the call keyword arguments (line 592)
        kwargs_17081 = {}
        # Getting the type of 'header' (line 592)
        header_17079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 34), 'header', False)
        # Obtaining the member 'lower' of a type (line 592)
        lower_17080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 34), header_17079, 'lower')
        # Calling lower(args, kwargs) (line 592)
        lower_call_result_17082 = invoke(stypy.reporting.localization.Localization(__file__, 592, 34), lower_17080, *[], **kwargs_17081)
        
        str_17083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 52), 'str', 'content-type')
        # Applying the binary operator '==' (line 592)
        result_eq_17084 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 34), '==', lower_call_result_17082, str_17083)
        
        # Applying the binary operator 'and' (line 592)
        result_and_keyword_17085 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 11), 'and', result_contains_17078, result_eq_17084)
        
        # Testing if the type of an if condition is none (line 592)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 592, 8), result_and_keyword_17085):
            
            # Assigning a Call to a Name (line 595):
            
            # Assigning a Call to a Name (line 595):
            
            # Call to get(...): (line 595)
            # Processing the call arguments (line 595)
            # Getting the type of 'header' (line 595)
            header_17090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 29), 'header', False)
            # Processing the call keyword arguments (line 595)
            kwargs_17091 = {}
            # Getting the type of 'self' (line 595)
            self_17088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 20), 'self', False)
            # Obtaining the member 'get' of a type (line 595)
            get_17089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 20), self_17088, 'get')
            # Calling get(args, kwargs) (line 595)
            get_call_result_17092 = invoke(stypy.reporting.localization.Localization(__file__, 595, 20), get_17089, *[header_17090], **kwargs_17091)
            
            # Assigning a type to the variable 'ctype' (line 595)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'ctype', get_call_result_17092)
        else:
            
            # Testing the type of an if condition (line 592)
            if_condition_17086 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 592, 8), result_and_keyword_17085)
            # Assigning a type to the variable 'if_condition_17086' (line 592)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'if_condition_17086', if_condition_17086)
            # SSA begins for if statement (line 592)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 593):
            
            # Assigning a Str to a Name (line 593):
            str_17087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 20), 'str', 'text/plain')
            # Assigning a type to the variable 'ctype' (line 593)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'ctype', str_17087)
            # SSA branch for the else part of an if statement (line 592)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 595):
            
            # Assigning a Call to a Name (line 595):
            
            # Call to get(...): (line 595)
            # Processing the call arguments (line 595)
            # Getting the type of 'header' (line 595)
            header_17090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 29), 'header', False)
            # Processing the call keyword arguments (line 595)
            kwargs_17091 = {}
            # Getting the type of 'self' (line 595)
            self_17088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 20), 'self', False)
            # Obtaining the member 'get' of a type (line 595)
            get_17089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 20), self_17088, 'get')
            # Calling get(args, kwargs) (line 595)
            get_call_result_17092 = invoke(stypy.reporting.localization.Localization(__file__, 595, 20), get_17089, *[header_17090], **kwargs_17091)
            
            # Assigning a type to the variable 'ctype' (line 595)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'ctype', get_call_result_17092)
            # SSA join for if statement (line 592)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to get_param(...): (line 596)
        # Processing the call arguments (line 596)
        # Getting the type of 'param' (line 596)
        param_17095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 30), 'param', False)
        # Processing the call keyword arguments (line 596)
        # Getting the type of 'header' (line 596)
        header_17096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 44), 'header', False)
        keyword_17097 = header_17096
        kwargs_17098 = {'header': keyword_17097}
        # Getting the type of 'self' (line 596)
        self_17093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 15), 'self', False)
        # Obtaining the member 'get_param' of a type (line 596)
        get_param_17094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 15), self_17093, 'get_param')
        # Calling get_param(args, kwargs) (line 596)
        get_param_call_result_17099 = invoke(stypy.reporting.localization.Localization(__file__, 596, 15), get_param_17094, *[param_17095], **kwargs_17098)
        
        # Applying the 'not' unary operator (line 596)
        result_not__17100 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 11), 'not', get_param_call_result_17099)
        
        # Testing if the type of an if condition is none (line 596)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 596, 8), result_not__17100):
            
            # Assigning a Str to a Name (line 603):
            
            # Assigning a Str to a Name (line 603):
            str_17123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 20), 'str', '')
            # Assigning a type to the variable 'ctype' (line 603)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'ctype', str_17123)
            
            
            # Call to get_params(...): (line 604)
            # Processing the call keyword arguments (line 604)
            # Getting the type of 'header' (line 604)
            header_17126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 63), 'header', False)
            keyword_17127 = header_17126
            # Getting the type of 'requote' (line 605)
            requote_17128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 64), 'requote', False)
            keyword_17129 = requote_17128
            kwargs_17130 = {'unquote': keyword_17129, 'header': keyword_17127}
            # Getting the type of 'self' (line 604)
            self_17124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 40), 'self', False)
            # Obtaining the member 'get_params' of a type (line 604)
            get_params_17125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 40), self_17124, 'get_params')
            # Calling get_params(args, kwargs) (line 604)
            get_params_call_result_17131 = invoke(stypy.reporting.localization.Localization(__file__, 604, 40), get_params_17125, *[], **kwargs_17130)
            
            # Assigning a type to the variable 'get_params_call_result_17131' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'get_params_call_result_17131', get_params_call_result_17131)
            # Testing if the for loop is going to be iterated (line 604)
            # Testing the type of a for loop iterable (line 604)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 604, 12), get_params_call_result_17131)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 604, 12), get_params_call_result_17131):
                # Getting the type of the for loop variable (line 604)
                for_loop_var_17132 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 604, 12), get_params_call_result_17131)
                # Assigning a type to the variable 'old_param' (line 604)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'old_param', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 12), for_loop_var_17132, 2, 0))
                # Assigning a type to the variable 'old_value' (line 604)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'old_value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 12), for_loop_var_17132, 2, 1))
                # SSA begins for a for statement (line 604)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Str to a Name (line 606):
                
                # Assigning a Str to a Name (line 606):
                str_17133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 31), 'str', '')
                # Assigning a type to the variable 'append_param' (line 606)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 16), 'append_param', str_17133)
                
                
                # Call to lower(...): (line 607)
                # Processing the call keyword arguments (line 607)
                kwargs_17136 = {}
                # Getting the type of 'old_param' (line 607)
                old_param_17134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 19), 'old_param', False)
                # Obtaining the member 'lower' of a type (line 607)
                lower_17135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 19), old_param_17134, 'lower')
                # Calling lower(args, kwargs) (line 607)
                lower_call_result_17137 = invoke(stypy.reporting.localization.Localization(__file__, 607, 19), lower_17135, *[], **kwargs_17136)
                
                
                # Call to lower(...): (line 607)
                # Processing the call keyword arguments (line 607)
                kwargs_17140 = {}
                # Getting the type of 'param' (line 607)
                param_17138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 40), 'param', False)
                # Obtaining the member 'lower' of a type (line 607)
                lower_17139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 40), param_17138, 'lower')
                # Calling lower(args, kwargs) (line 607)
                lower_call_result_17141 = invoke(stypy.reporting.localization.Localization(__file__, 607, 40), lower_17139, *[], **kwargs_17140)
                
                # Applying the binary operator '==' (line 607)
                result_eq_17142 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 19), '==', lower_call_result_17137, lower_call_result_17141)
                
                # Testing if the type of an if condition is none (line 607)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 607, 16), result_eq_17142):
                    
                    # Assigning a Call to a Name (line 610):
                    
                    # Assigning a Call to a Name (line 610):
                    
                    # Call to _formatparam(...): (line 610)
                    # Processing the call arguments (line 610)
                    # Getting the type of 'old_param' (line 610)
                    old_param_17151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 48), 'old_param', False)
                    # Getting the type of 'old_value' (line 610)
                    old_value_17152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 59), 'old_value', False)
                    # Getting the type of 'requote' (line 610)
                    requote_17153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 70), 'requote', False)
                    # Processing the call keyword arguments (line 610)
                    kwargs_17154 = {}
                    # Getting the type of '_formatparam' (line 610)
                    _formatparam_17150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 35), '_formatparam', False)
                    # Calling _formatparam(args, kwargs) (line 610)
                    _formatparam_call_result_17155 = invoke(stypy.reporting.localization.Localization(__file__, 610, 35), _formatparam_17150, *[old_param_17151, old_value_17152, requote_17153], **kwargs_17154)
                    
                    # Assigning a type to the variable 'append_param' (line 610)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'append_param', _formatparam_call_result_17155)
                else:
                    
                    # Testing the type of an if condition (line 607)
                    if_condition_17143 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 607, 16), result_eq_17142)
                    # Assigning a type to the variable 'if_condition_17143' (line 607)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'if_condition_17143', if_condition_17143)
                    # SSA begins for if statement (line 607)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 608):
                    
                    # Assigning a Call to a Name (line 608):
                    
                    # Call to _formatparam(...): (line 608)
                    # Processing the call arguments (line 608)
                    # Getting the type of 'param' (line 608)
                    param_17145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 48), 'param', False)
                    # Getting the type of 'value' (line 608)
                    value_17146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 55), 'value', False)
                    # Getting the type of 'requote' (line 608)
                    requote_17147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 62), 'requote', False)
                    # Processing the call keyword arguments (line 608)
                    kwargs_17148 = {}
                    # Getting the type of '_formatparam' (line 608)
                    _formatparam_17144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 35), '_formatparam', False)
                    # Calling _formatparam(args, kwargs) (line 608)
                    _formatparam_call_result_17149 = invoke(stypy.reporting.localization.Localization(__file__, 608, 35), _formatparam_17144, *[param_17145, value_17146, requote_17147], **kwargs_17148)
                    
                    # Assigning a type to the variable 'append_param' (line 608)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 20), 'append_param', _formatparam_call_result_17149)
                    # SSA branch for the else part of an if statement (line 607)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 610):
                    
                    # Assigning a Call to a Name (line 610):
                    
                    # Call to _formatparam(...): (line 610)
                    # Processing the call arguments (line 610)
                    # Getting the type of 'old_param' (line 610)
                    old_param_17151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 48), 'old_param', False)
                    # Getting the type of 'old_value' (line 610)
                    old_value_17152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 59), 'old_value', False)
                    # Getting the type of 'requote' (line 610)
                    requote_17153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 70), 'requote', False)
                    # Processing the call keyword arguments (line 610)
                    kwargs_17154 = {}
                    # Getting the type of '_formatparam' (line 610)
                    _formatparam_17150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 35), '_formatparam', False)
                    # Calling _formatparam(args, kwargs) (line 610)
                    _formatparam_call_result_17155 = invoke(stypy.reporting.localization.Localization(__file__, 610, 35), _formatparam_17150, *[old_param_17151, old_value_17152, requote_17153], **kwargs_17154)
                    
                    # Assigning a type to the variable 'append_param' (line 610)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'append_param', _formatparam_call_result_17155)
                    # SSA join for if statement (line 607)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'ctype' (line 611)
                ctype_17156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 23), 'ctype')
                # Applying the 'not' unary operator (line 611)
                result_not__17157 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 19), 'not', ctype_17156)
                
                # Testing if the type of an if condition is none (line 611)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 611, 16), result_not__17157):
                    
                    # Assigning a Call to a Name (line 614):
                    
                    # Assigning a Call to a Name (line 614):
                    
                    # Call to join(...): (line 614)
                    # Processing the call arguments (line 614)
                    
                    # Obtaining an instance of the builtin type 'list' (line 614)
                    list_17162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 43), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 614)
                    # Adding element type (line 614)
                    # Getting the type of 'ctype' (line 614)
                    ctype_17163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 44), 'ctype', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 43), list_17162, ctype_17163)
                    # Adding element type (line 614)
                    # Getting the type of 'append_param' (line 614)
                    append_param_17164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 51), 'append_param', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 43), list_17162, append_param_17164)
                    
                    # Processing the call keyword arguments (line 614)
                    kwargs_17165 = {}
                    # Getting the type of 'SEMISPACE' (line 614)
                    SEMISPACE_17160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 28), 'SEMISPACE', False)
                    # Obtaining the member 'join' of a type (line 614)
                    join_17161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 28), SEMISPACE_17160, 'join')
                    # Calling join(args, kwargs) (line 614)
                    join_call_result_17166 = invoke(stypy.reporting.localization.Localization(__file__, 614, 28), join_17161, *[list_17162], **kwargs_17165)
                    
                    # Assigning a type to the variable 'ctype' (line 614)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 20), 'ctype', join_call_result_17166)
                else:
                    
                    # Testing the type of an if condition (line 611)
                    if_condition_17158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 16), result_not__17157)
                    # Assigning a type to the variable 'if_condition_17158' (line 611)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'if_condition_17158', if_condition_17158)
                    # SSA begins for if statement (line 611)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 612):
                    
                    # Assigning a Name to a Name (line 612):
                    # Getting the type of 'append_param' (line 612)
                    append_param_17159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 28), 'append_param')
                    # Assigning a type to the variable 'ctype' (line 612)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 20), 'ctype', append_param_17159)
                    # SSA branch for the else part of an if statement (line 611)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 614):
                    
                    # Assigning a Call to a Name (line 614):
                    
                    # Call to join(...): (line 614)
                    # Processing the call arguments (line 614)
                    
                    # Obtaining an instance of the builtin type 'list' (line 614)
                    list_17162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 43), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 614)
                    # Adding element type (line 614)
                    # Getting the type of 'ctype' (line 614)
                    ctype_17163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 44), 'ctype', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 43), list_17162, ctype_17163)
                    # Adding element type (line 614)
                    # Getting the type of 'append_param' (line 614)
                    append_param_17164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 51), 'append_param', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 43), list_17162, append_param_17164)
                    
                    # Processing the call keyword arguments (line 614)
                    kwargs_17165 = {}
                    # Getting the type of 'SEMISPACE' (line 614)
                    SEMISPACE_17160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 28), 'SEMISPACE', False)
                    # Obtaining the member 'join' of a type (line 614)
                    join_17161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 28), SEMISPACE_17160, 'join')
                    # Calling join(args, kwargs) (line 614)
                    join_call_result_17166 = invoke(stypy.reporting.localization.Localization(__file__, 614, 28), join_17161, *[list_17162], **kwargs_17165)
                    
                    # Assigning a type to the variable 'ctype' (line 614)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 20), 'ctype', join_call_result_17166)
                    # SSA join for if statement (line 611)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 596)
            if_condition_17101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 596, 8), result_not__17100)
            # Assigning a type to the variable 'if_condition_17101' (line 596)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'if_condition_17101', if_condition_17101)
            # SSA begins for if statement (line 596)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'ctype' (line 597)
            ctype_17102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 19), 'ctype')
            # Applying the 'not' unary operator (line 597)
            result_not__17103 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 15), 'not', ctype_17102)
            
            # Testing if the type of an if condition is none (line 597)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 597, 12), result_not__17103):
                
                # Assigning a Call to a Name (line 600):
                
                # Assigning a Call to a Name (line 600):
                
                # Call to join(...): (line 600)
                # Processing the call arguments (line 600)
                
                # Obtaining an instance of the builtin type 'list' (line 601)
                list_17113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 20), 'list')
                # Adding type elements to the builtin type 'list' instance (line 601)
                # Adding element type (line 601)
                # Getting the type of 'ctype' (line 601)
                ctype_17114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 21), 'ctype', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 20), list_17113, ctype_17114)
                # Adding element type (line 601)
                
                # Call to _formatparam(...): (line 601)
                # Processing the call arguments (line 601)
                # Getting the type of 'param' (line 601)
                param_17116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 41), 'param', False)
                # Getting the type of 'value' (line 601)
                value_17117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 48), 'value', False)
                # Getting the type of 'requote' (line 601)
                requote_17118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 55), 'requote', False)
                # Processing the call keyword arguments (line 601)
                kwargs_17119 = {}
                # Getting the type of '_formatparam' (line 601)
                _formatparam_17115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 28), '_formatparam', False)
                # Calling _formatparam(args, kwargs) (line 601)
                _formatparam_call_result_17120 = invoke(stypy.reporting.localization.Localization(__file__, 601, 28), _formatparam_17115, *[param_17116, value_17117, requote_17118], **kwargs_17119)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 20), list_17113, _formatparam_call_result_17120)
                
                # Processing the call keyword arguments (line 600)
                kwargs_17121 = {}
                # Getting the type of 'SEMISPACE' (line 600)
                SEMISPACE_17111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 24), 'SEMISPACE', False)
                # Obtaining the member 'join' of a type (line 600)
                join_17112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 24), SEMISPACE_17111, 'join')
                # Calling join(args, kwargs) (line 600)
                join_call_result_17122 = invoke(stypy.reporting.localization.Localization(__file__, 600, 24), join_17112, *[list_17113], **kwargs_17121)
                
                # Assigning a type to the variable 'ctype' (line 600)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'ctype', join_call_result_17122)
            else:
                
                # Testing the type of an if condition (line 597)
                if_condition_17104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 12), result_not__17103)
                # Assigning a type to the variable 'if_condition_17104' (line 597)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'if_condition_17104', if_condition_17104)
                # SSA begins for if statement (line 597)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 598):
                
                # Assigning a Call to a Name (line 598):
                
                # Call to _formatparam(...): (line 598)
                # Processing the call arguments (line 598)
                # Getting the type of 'param' (line 598)
                param_17106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 37), 'param', False)
                # Getting the type of 'value' (line 598)
                value_17107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 44), 'value', False)
                # Getting the type of 'requote' (line 598)
                requote_17108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 51), 'requote', False)
                # Processing the call keyword arguments (line 598)
                kwargs_17109 = {}
                # Getting the type of '_formatparam' (line 598)
                _formatparam_17105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 24), '_formatparam', False)
                # Calling _formatparam(args, kwargs) (line 598)
                _formatparam_call_result_17110 = invoke(stypy.reporting.localization.Localization(__file__, 598, 24), _formatparam_17105, *[param_17106, value_17107, requote_17108], **kwargs_17109)
                
                # Assigning a type to the variable 'ctype' (line 598)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'ctype', _formatparam_call_result_17110)
                # SSA branch for the else part of an if statement (line 597)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 600):
                
                # Assigning a Call to a Name (line 600):
                
                # Call to join(...): (line 600)
                # Processing the call arguments (line 600)
                
                # Obtaining an instance of the builtin type 'list' (line 601)
                list_17113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 20), 'list')
                # Adding type elements to the builtin type 'list' instance (line 601)
                # Adding element type (line 601)
                # Getting the type of 'ctype' (line 601)
                ctype_17114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 21), 'ctype', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 20), list_17113, ctype_17114)
                # Adding element type (line 601)
                
                # Call to _formatparam(...): (line 601)
                # Processing the call arguments (line 601)
                # Getting the type of 'param' (line 601)
                param_17116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 41), 'param', False)
                # Getting the type of 'value' (line 601)
                value_17117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 48), 'value', False)
                # Getting the type of 'requote' (line 601)
                requote_17118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 55), 'requote', False)
                # Processing the call keyword arguments (line 601)
                kwargs_17119 = {}
                # Getting the type of '_formatparam' (line 601)
                _formatparam_17115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 28), '_formatparam', False)
                # Calling _formatparam(args, kwargs) (line 601)
                _formatparam_call_result_17120 = invoke(stypy.reporting.localization.Localization(__file__, 601, 28), _formatparam_17115, *[param_17116, value_17117, requote_17118], **kwargs_17119)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 20), list_17113, _formatparam_call_result_17120)
                
                # Processing the call keyword arguments (line 600)
                kwargs_17121 = {}
                # Getting the type of 'SEMISPACE' (line 600)
                SEMISPACE_17111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 24), 'SEMISPACE', False)
                # Obtaining the member 'join' of a type (line 600)
                join_17112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 24), SEMISPACE_17111, 'join')
                # Calling join(args, kwargs) (line 600)
                join_call_result_17122 = invoke(stypy.reporting.localization.Localization(__file__, 600, 24), join_17112, *[list_17113], **kwargs_17121)
                
                # Assigning a type to the variable 'ctype' (line 600)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'ctype', join_call_result_17122)
                # SSA join for if statement (line 597)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of an if statement (line 596)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 603):
            
            # Assigning a Str to a Name (line 603):
            str_17123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 20), 'str', '')
            # Assigning a type to the variable 'ctype' (line 603)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'ctype', str_17123)
            
            
            # Call to get_params(...): (line 604)
            # Processing the call keyword arguments (line 604)
            # Getting the type of 'header' (line 604)
            header_17126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 63), 'header', False)
            keyword_17127 = header_17126
            # Getting the type of 'requote' (line 605)
            requote_17128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 64), 'requote', False)
            keyword_17129 = requote_17128
            kwargs_17130 = {'unquote': keyword_17129, 'header': keyword_17127}
            # Getting the type of 'self' (line 604)
            self_17124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 40), 'self', False)
            # Obtaining the member 'get_params' of a type (line 604)
            get_params_17125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 40), self_17124, 'get_params')
            # Calling get_params(args, kwargs) (line 604)
            get_params_call_result_17131 = invoke(stypy.reporting.localization.Localization(__file__, 604, 40), get_params_17125, *[], **kwargs_17130)
            
            # Assigning a type to the variable 'get_params_call_result_17131' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'get_params_call_result_17131', get_params_call_result_17131)
            # Testing if the for loop is going to be iterated (line 604)
            # Testing the type of a for loop iterable (line 604)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 604, 12), get_params_call_result_17131)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 604, 12), get_params_call_result_17131):
                # Getting the type of the for loop variable (line 604)
                for_loop_var_17132 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 604, 12), get_params_call_result_17131)
                # Assigning a type to the variable 'old_param' (line 604)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'old_param', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 12), for_loop_var_17132, 2, 0))
                # Assigning a type to the variable 'old_value' (line 604)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'old_value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 12), for_loop_var_17132, 2, 1))
                # SSA begins for a for statement (line 604)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Str to a Name (line 606):
                
                # Assigning a Str to a Name (line 606):
                str_17133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 31), 'str', '')
                # Assigning a type to the variable 'append_param' (line 606)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 16), 'append_param', str_17133)
                
                
                # Call to lower(...): (line 607)
                # Processing the call keyword arguments (line 607)
                kwargs_17136 = {}
                # Getting the type of 'old_param' (line 607)
                old_param_17134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 19), 'old_param', False)
                # Obtaining the member 'lower' of a type (line 607)
                lower_17135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 19), old_param_17134, 'lower')
                # Calling lower(args, kwargs) (line 607)
                lower_call_result_17137 = invoke(stypy.reporting.localization.Localization(__file__, 607, 19), lower_17135, *[], **kwargs_17136)
                
                
                # Call to lower(...): (line 607)
                # Processing the call keyword arguments (line 607)
                kwargs_17140 = {}
                # Getting the type of 'param' (line 607)
                param_17138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 40), 'param', False)
                # Obtaining the member 'lower' of a type (line 607)
                lower_17139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 40), param_17138, 'lower')
                # Calling lower(args, kwargs) (line 607)
                lower_call_result_17141 = invoke(stypy.reporting.localization.Localization(__file__, 607, 40), lower_17139, *[], **kwargs_17140)
                
                # Applying the binary operator '==' (line 607)
                result_eq_17142 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 19), '==', lower_call_result_17137, lower_call_result_17141)
                
                # Testing if the type of an if condition is none (line 607)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 607, 16), result_eq_17142):
                    
                    # Assigning a Call to a Name (line 610):
                    
                    # Assigning a Call to a Name (line 610):
                    
                    # Call to _formatparam(...): (line 610)
                    # Processing the call arguments (line 610)
                    # Getting the type of 'old_param' (line 610)
                    old_param_17151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 48), 'old_param', False)
                    # Getting the type of 'old_value' (line 610)
                    old_value_17152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 59), 'old_value', False)
                    # Getting the type of 'requote' (line 610)
                    requote_17153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 70), 'requote', False)
                    # Processing the call keyword arguments (line 610)
                    kwargs_17154 = {}
                    # Getting the type of '_formatparam' (line 610)
                    _formatparam_17150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 35), '_formatparam', False)
                    # Calling _formatparam(args, kwargs) (line 610)
                    _formatparam_call_result_17155 = invoke(stypy.reporting.localization.Localization(__file__, 610, 35), _formatparam_17150, *[old_param_17151, old_value_17152, requote_17153], **kwargs_17154)
                    
                    # Assigning a type to the variable 'append_param' (line 610)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'append_param', _formatparam_call_result_17155)
                else:
                    
                    # Testing the type of an if condition (line 607)
                    if_condition_17143 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 607, 16), result_eq_17142)
                    # Assigning a type to the variable 'if_condition_17143' (line 607)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'if_condition_17143', if_condition_17143)
                    # SSA begins for if statement (line 607)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 608):
                    
                    # Assigning a Call to a Name (line 608):
                    
                    # Call to _formatparam(...): (line 608)
                    # Processing the call arguments (line 608)
                    # Getting the type of 'param' (line 608)
                    param_17145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 48), 'param', False)
                    # Getting the type of 'value' (line 608)
                    value_17146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 55), 'value', False)
                    # Getting the type of 'requote' (line 608)
                    requote_17147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 62), 'requote', False)
                    # Processing the call keyword arguments (line 608)
                    kwargs_17148 = {}
                    # Getting the type of '_formatparam' (line 608)
                    _formatparam_17144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 35), '_formatparam', False)
                    # Calling _formatparam(args, kwargs) (line 608)
                    _formatparam_call_result_17149 = invoke(stypy.reporting.localization.Localization(__file__, 608, 35), _formatparam_17144, *[param_17145, value_17146, requote_17147], **kwargs_17148)
                    
                    # Assigning a type to the variable 'append_param' (line 608)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 20), 'append_param', _formatparam_call_result_17149)
                    # SSA branch for the else part of an if statement (line 607)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 610):
                    
                    # Assigning a Call to a Name (line 610):
                    
                    # Call to _formatparam(...): (line 610)
                    # Processing the call arguments (line 610)
                    # Getting the type of 'old_param' (line 610)
                    old_param_17151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 48), 'old_param', False)
                    # Getting the type of 'old_value' (line 610)
                    old_value_17152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 59), 'old_value', False)
                    # Getting the type of 'requote' (line 610)
                    requote_17153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 70), 'requote', False)
                    # Processing the call keyword arguments (line 610)
                    kwargs_17154 = {}
                    # Getting the type of '_formatparam' (line 610)
                    _formatparam_17150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 35), '_formatparam', False)
                    # Calling _formatparam(args, kwargs) (line 610)
                    _formatparam_call_result_17155 = invoke(stypy.reporting.localization.Localization(__file__, 610, 35), _formatparam_17150, *[old_param_17151, old_value_17152, requote_17153], **kwargs_17154)
                    
                    # Assigning a type to the variable 'append_param' (line 610)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'append_param', _formatparam_call_result_17155)
                    # SSA join for if statement (line 607)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'ctype' (line 611)
                ctype_17156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 23), 'ctype')
                # Applying the 'not' unary operator (line 611)
                result_not__17157 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 19), 'not', ctype_17156)
                
                # Testing if the type of an if condition is none (line 611)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 611, 16), result_not__17157):
                    
                    # Assigning a Call to a Name (line 614):
                    
                    # Assigning a Call to a Name (line 614):
                    
                    # Call to join(...): (line 614)
                    # Processing the call arguments (line 614)
                    
                    # Obtaining an instance of the builtin type 'list' (line 614)
                    list_17162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 43), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 614)
                    # Adding element type (line 614)
                    # Getting the type of 'ctype' (line 614)
                    ctype_17163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 44), 'ctype', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 43), list_17162, ctype_17163)
                    # Adding element type (line 614)
                    # Getting the type of 'append_param' (line 614)
                    append_param_17164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 51), 'append_param', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 43), list_17162, append_param_17164)
                    
                    # Processing the call keyword arguments (line 614)
                    kwargs_17165 = {}
                    # Getting the type of 'SEMISPACE' (line 614)
                    SEMISPACE_17160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 28), 'SEMISPACE', False)
                    # Obtaining the member 'join' of a type (line 614)
                    join_17161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 28), SEMISPACE_17160, 'join')
                    # Calling join(args, kwargs) (line 614)
                    join_call_result_17166 = invoke(stypy.reporting.localization.Localization(__file__, 614, 28), join_17161, *[list_17162], **kwargs_17165)
                    
                    # Assigning a type to the variable 'ctype' (line 614)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 20), 'ctype', join_call_result_17166)
                else:
                    
                    # Testing the type of an if condition (line 611)
                    if_condition_17158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 16), result_not__17157)
                    # Assigning a type to the variable 'if_condition_17158' (line 611)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'if_condition_17158', if_condition_17158)
                    # SSA begins for if statement (line 611)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 612):
                    
                    # Assigning a Name to a Name (line 612):
                    # Getting the type of 'append_param' (line 612)
                    append_param_17159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 28), 'append_param')
                    # Assigning a type to the variable 'ctype' (line 612)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 20), 'ctype', append_param_17159)
                    # SSA branch for the else part of an if statement (line 611)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 614):
                    
                    # Assigning a Call to a Name (line 614):
                    
                    # Call to join(...): (line 614)
                    # Processing the call arguments (line 614)
                    
                    # Obtaining an instance of the builtin type 'list' (line 614)
                    list_17162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 43), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 614)
                    # Adding element type (line 614)
                    # Getting the type of 'ctype' (line 614)
                    ctype_17163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 44), 'ctype', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 43), list_17162, ctype_17163)
                    # Adding element type (line 614)
                    # Getting the type of 'append_param' (line 614)
                    append_param_17164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 51), 'append_param', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 43), list_17162, append_param_17164)
                    
                    # Processing the call keyword arguments (line 614)
                    kwargs_17165 = {}
                    # Getting the type of 'SEMISPACE' (line 614)
                    SEMISPACE_17160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 28), 'SEMISPACE', False)
                    # Obtaining the member 'join' of a type (line 614)
                    join_17161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 28), SEMISPACE_17160, 'join')
                    # Calling join(args, kwargs) (line 614)
                    join_call_result_17166 = invoke(stypy.reporting.localization.Localization(__file__, 614, 28), join_17161, *[list_17162], **kwargs_17165)
                    
                    # Assigning a type to the variable 'ctype' (line 614)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 20), 'ctype', join_call_result_17166)
                    # SSA join for if statement (line 611)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 596)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'ctype' (line 615)
        ctype_17167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 11), 'ctype')
        
        # Call to get(...): (line 615)
        # Processing the call arguments (line 615)
        # Getting the type of 'header' (line 615)
        header_17170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 29), 'header', False)
        # Processing the call keyword arguments (line 615)
        kwargs_17171 = {}
        # Getting the type of 'self' (line 615)
        self_17168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 20), 'self', False)
        # Obtaining the member 'get' of a type (line 615)
        get_17169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 20), self_17168, 'get')
        # Calling get(args, kwargs) (line 615)
        get_call_result_17172 = invoke(stypy.reporting.localization.Localization(__file__, 615, 20), get_17169, *[header_17170], **kwargs_17171)
        
        # Applying the binary operator '!=' (line 615)
        result_ne_17173 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 11), '!=', ctype_17167, get_call_result_17172)
        
        # Testing if the type of an if condition is none (line 615)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 615, 8), result_ne_17173):
            pass
        else:
            
            # Testing the type of an if condition (line 615)
            if_condition_17174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 615, 8), result_ne_17173)
            # Assigning a type to the variable 'if_condition_17174' (line 615)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'if_condition_17174', if_condition_17174)
            # SSA begins for if statement (line 615)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Deleting a member
            # Getting the type of 'self' (line 616)
            self_17175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'self')
            
            # Obtaining the type of the subscript
            # Getting the type of 'header' (line 616)
            header_17176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 21), 'header')
            # Getting the type of 'self' (line 616)
            self_17177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'self')
            # Obtaining the member '__getitem__' of a type (line 616)
            getitem___17178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 16), self_17177, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 616)
            subscript_call_result_17179 = invoke(stypy.reporting.localization.Localization(__file__, 616, 16), getitem___17178, header_17176)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 12), self_17175, subscript_call_result_17179)
            
            # Assigning a Name to a Subscript (line 617):
            
            # Assigning a Name to a Subscript (line 617):
            # Getting the type of 'ctype' (line 617)
            ctype_17180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 27), 'ctype')
            # Getting the type of 'self' (line 617)
            self_17181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'self')
            # Getting the type of 'header' (line 617)
            header_17182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 17), 'header')
            # Storing an element on a container (line 617)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 12), self_17181, (header_17182, ctype_17180))
            # SSA join for if statement (line 615)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'set_param(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_param' in the type store
        # Getting the type of 'stypy_return_type' (line 571)
        stypy_return_type_17183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17183)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_param'
        return stypy_return_type_17183


    @norecursion
    def del_param(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_17184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 38), 'str', 'content-type')
        # Getting the type of 'True' (line 619)
        True_17185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 62), 'True')
        defaults = [str_17184, True_17185]
        # Create a new context for function 'del_param'
        module_type_store = module_type_store.open_function_context('del_param', 619, 4, False)
        # Assigning a type to the variable 'self' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.del_param.__dict__.__setitem__('stypy_localization', localization)
        Message.del_param.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.del_param.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.del_param.__dict__.__setitem__('stypy_function_name', 'Message.del_param')
        Message.del_param.__dict__.__setitem__('stypy_param_names_list', ['param', 'header', 'requote'])
        Message.del_param.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.del_param.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.del_param.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.del_param.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.del_param.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.del_param.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.del_param', ['param', 'header', 'requote'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'del_param', localization, ['param', 'header', 'requote'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'del_param(...)' code ##################

        str_17186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, (-1)), 'str', 'Remove the given parameter completely from the Content-Type header.\n\n        The header will be re-written in place without the parameter or its\n        value. All values will be quoted as necessary unless requote is\n        False.  Optional header specifies an alternative to the Content-Type\n        header.\n        ')
        
        # Getting the type of 'header' (line 627)
        header_17187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 11), 'header')
        # Getting the type of 'self' (line 627)
        self_17188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 25), 'self')
        # Applying the binary operator 'notin' (line 627)
        result_contains_17189 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 11), 'notin', header_17187, self_17188)
        
        # Testing if the type of an if condition is none (line 627)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 627, 8), result_contains_17189):
            pass
        else:
            
            # Testing the type of an if condition (line 627)
            if_condition_17190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 627, 8), result_contains_17189)
            # Assigning a type to the variable 'if_condition_17190' (line 627)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'if_condition_17190', if_condition_17190)
            # SSA begins for if statement (line 627)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 628)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 627)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 629):
        
        # Assigning a Str to a Name (line 629):
        str_17191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 20), 'str', '')
        # Assigning a type to the variable 'new_ctype' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'new_ctype', str_17191)
        
        
        # Call to get_params(...): (line 630)
        # Processing the call keyword arguments (line 630)
        # Getting the type of 'header' (line 630)
        header_17194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 43), 'header', False)
        keyword_17195 = header_17194
        # Getting the type of 'requote' (line 630)
        requote_17196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 59), 'requote', False)
        keyword_17197 = requote_17196
        kwargs_17198 = {'unquote': keyword_17197, 'header': keyword_17195}
        # Getting the type of 'self' (line 630)
        self_17192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 20), 'self', False)
        # Obtaining the member 'get_params' of a type (line 630)
        get_params_17193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 20), self_17192, 'get_params')
        # Calling get_params(args, kwargs) (line 630)
        get_params_call_result_17199 = invoke(stypy.reporting.localization.Localization(__file__, 630, 20), get_params_17193, *[], **kwargs_17198)
        
        # Assigning a type to the variable 'get_params_call_result_17199' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'get_params_call_result_17199', get_params_call_result_17199)
        # Testing if the for loop is going to be iterated (line 630)
        # Testing the type of a for loop iterable (line 630)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 630, 8), get_params_call_result_17199)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 630, 8), get_params_call_result_17199):
            # Getting the type of the for loop variable (line 630)
            for_loop_var_17200 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 630, 8), get_params_call_result_17199)
            # Assigning a type to the variable 'p' (line 630)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 8), for_loop_var_17200, 2, 0))
            # Assigning a type to the variable 'v' (line 630)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 8), for_loop_var_17200, 2, 1))
            # SSA begins for a for statement (line 630)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to lower(...): (line 631)
            # Processing the call keyword arguments (line 631)
            kwargs_17203 = {}
            # Getting the type of 'p' (line 631)
            p_17201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 15), 'p', False)
            # Obtaining the member 'lower' of a type (line 631)
            lower_17202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 15), p_17201, 'lower')
            # Calling lower(args, kwargs) (line 631)
            lower_call_result_17204 = invoke(stypy.reporting.localization.Localization(__file__, 631, 15), lower_17202, *[], **kwargs_17203)
            
            
            # Call to lower(...): (line 631)
            # Processing the call keyword arguments (line 631)
            kwargs_17207 = {}
            # Getting the type of 'param' (line 631)
            param_17205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 28), 'param', False)
            # Obtaining the member 'lower' of a type (line 631)
            lower_17206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 28), param_17205, 'lower')
            # Calling lower(args, kwargs) (line 631)
            lower_call_result_17208 = invoke(stypy.reporting.localization.Localization(__file__, 631, 28), lower_17206, *[], **kwargs_17207)
            
            # Applying the binary operator '!=' (line 631)
            result_ne_17209 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 15), '!=', lower_call_result_17204, lower_call_result_17208)
            
            # Testing if the type of an if condition is none (line 631)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 631, 12), result_ne_17209):
                pass
            else:
                
                # Testing the type of an if condition (line 631)
                if_condition_17210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 631, 12), result_ne_17209)
                # Assigning a type to the variable 'if_condition_17210' (line 631)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 12), 'if_condition_17210', if_condition_17210)
                # SSA begins for if statement (line 631)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'new_ctype' (line 632)
                new_ctype_17211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 23), 'new_ctype')
                # Applying the 'not' unary operator (line 632)
                result_not__17212 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 19), 'not', new_ctype_17211)
                
                # Testing if the type of an if condition is none (line 632)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 632, 16), result_not__17212):
                    
                    # Assigning a Call to a Name (line 635):
                    
                    # Assigning a Call to a Name (line 635):
                    
                    # Call to join(...): (line 635)
                    # Processing the call arguments (line 635)
                    
                    # Obtaining an instance of the builtin type 'list' (line 635)
                    list_17222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 47), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 635)
                    # Adding element type (line 635)
                    # Getting the type of 'new_ctype' (line 635)
                    new_ctype_17223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 48), 'new_ctype', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 635, 47), list_17222, new_ctype_17223)
                    # Adding element type (line 635)
                    
                    # Call to _formatparam(...): (line 636)
                    # Processing the call arguments (line 636)
                    # Getting the type of 'p' (line 636)
                    p_17225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 61), 'p', False)
                    # Getting the type of 'v' (line 636)
                    v_17226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 64), 'v', False)
                    # Getting the type of 'requote' (line 636)
                    requote_17227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 67), 'requote', False)
                    # Processing the call keyword arguments (line 636)
                    kwargs_17228 = {}
                    # Getting the type of '_formatparam' (line 636)
                    _formatparam_17224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 48), '_formatparam', False)
                    # Calling _formatparam(args, kwargs) (line 636)
                    _formatparam_call_result_17229 = invoke(stypy.reporting.localization.Localization(__file__, 636, 48), _formatparam_17224, *[p_17225, v_17226, requote_17227], **kwargs_17228)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 635, 47), list_17222, _formatparam_call_result_17229)
                    
                    # Processing the call keyword arguments (line 635)
                    kwargs_17230 = {}
                    # Getting the type of 'SEMISPACE' (line 635)
                    SEMISPACE_17220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 32), 'SEMISPACE', False)
                    # Obtaining the member 'join' of a type (line 635)
                    join_17221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 32), SEMISPACE_17220, 'join')
                    # Calling join(args, kwargs) (line 635)
                    join_call_result_17231 = invoke(stypy.reporting.localization.Localization(__file__, 635, 32), join_17221, *[list_17222], **kwargs_17230)
                    
                    # Assigning a type to the variable 'new_ctype' (line 635)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 20), 'new_ctype', join_call_result_17231)
                else:
                    
                    # Testing the type of an if condition (line 632)
                    if_condition_17213 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 632, 16), result_not__17212)
                    # Assigning a type to the variable 'if_condition_17213' (line 632)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 16), 'if_condition_17213', if_condition_17213)
                    # SSA begins for if statement (line 632)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 633):
                    
                    # Assigning a Call to a Name (line 633):
                    
                    # Call to _formatparam(...): (line 633)
                    # Processing the call arguments (line 633)
                    # Getting the type of 'p' (line 633)
                    p_17215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 45), 'p', False)
                    # Getting the type of 'v' (line 633)
                    v_17216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 48), 'v', False)
                    # Getting the type of 'requote' (line 633)
                    requote_17217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 51), 'requote', False)
                    # Processing the call keyword arguments (line 633)
                    kwargs_17218 = {}
                    # Getting the type of '_formatparam' (line 633)
                    _formatparam_17214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 32), '_formatparam', False)
                    # Calling _formatparam(args, kwargs) (line 633)
                    _formatparam_call_result_17219 = invoke(stypy.reporting.localization.Localization(__file__, 633, 32), _formatparam_17214, *[p_17215, v_17216, requote_17217], **kwargs_17218)
                    
                    # Assigning a type to the variable 'new_ctype' (line 633)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 20), 'new_ctype', _formatparam_call_result_17219)
                    # SSA branch for the else part of an if statement (line 632)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 635):
                    
                    # Assigning a Call to a Name (line 635):
                    
                    # Call to join(...): (line 635)
                    # Processing the call arguments (line 635)
                    
                    # Obtaining an instance of the builtin type 'list' (line 635)
                    list_17222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 47), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 635)
                    # Adding element type (line 635)
                    # Getting the type of 'new_ctype' (line 635)
                    new_ctype_17223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 48), 'new_ctype', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 635, 47), list_17222, new_ctype_17223)
                    # Adding element type (line 635)
                    
                    # Call to _formatparam(...): (line 636)
                    # Processing the call arguments (line 636)
                    # Getting the type of 'p' (line 636)
                    p_17225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 61), 'p', False)
                    # Getting the type of 'v' (line 636)
                    v_17226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 64), 'v', False)
                    # Getting the type of 'requote' (line 636)
                    requote_17227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 67), 'requote', False)
                    # Processing the call keyword arguments (line 636)
                    kwargs_17228 = {}
                    # Getting the type of '_formatparam' (line 636)
                    _formatparam_17224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 48), '_formatparam', False)
                    # Calling _formatparam(args, kwargs) (line 636)
                    _formatparam_call_result_17229 = invoke(stypy.reporting.localization.Localization(__file__, 636, 48), _formatparam_17224, *[p_17225, v_17226, requote_17227], **kwargs_17228)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 635, 47), list_17222, _formatparam_call_result_17229)
                    
                    # Processing the call keyword arguments (line 635)
                    kwargs_17230 = {}
                    # Getting the type of 'SEMISPACE' (line 635)
                    SEMISPACE_17220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 32), 'SEMISPACE', False)
                    # Obtaining the member 'join' of a type (line 635)
                    join_17221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 32), SEMISPACE_17220, 'join')
                    # Calling join(args, kwargs) (line 635)
                    join_call_result_17231 = invoke(stypy.reporting.localization.Localization(__file__, 635, 32), join_17221, *[list_17222], **kwargs_17230)
                    
                    # Assigning a type to the variable 'new_ctype' (line 635)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 20), 'new_ctype', join_call_result_17231)
                    # SSA join for if statement (line 632)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 631)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'new_ctype' (line 637)
        new_ctype_17232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 11), 'new_ctype')
        
        # Call to get(...): (line 637)
        # Processing the call arguments (line 637)
        # Getting the type of 'header' (line 637)
        header_17235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 33), 'header', False)
        # Processing the call keyword arguments (line 637)
        kwargs_17236 = {}
        # Getting the type of 'self' (line 637)
        self_17233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 24), 'self', False)
        # Obtaining the member 'get' of a type (line 637)
        get_17234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 24), self_17233, 'get')
        # Calling get(args, kwargs) (line 637)
        get_call_result_17237 = invoke(stypy.reporting.localization.Localization(__file__, 637, 24), get_17234, *[header_17235], **kwargs_17236)
        
        # Applying the binary operator '!=' (line 637)
        result_ne_17238 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 11), '!=', new_ctype_17232, get_call_result_17237)
        
        # Testing if the type of an if condition is none (line 637)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 637, 8), result_ne_17238):
            pass
        else:
            
            # Testing the type of an if condition (line 637)
            if_condition_17239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 637, 8), result_ne_17238)
            # Assigning a type to the variable 'if_condition_17239' (line 637)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'if_condition_17239', if_condition_17239)
            # SSA begins for if statement (line 637)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Deleting a member
            # Getting the type of 'self' (line 638)
            self_17240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 16), 'self')
            
            # Obtaining the type of the subscript
            # Getting the type of 'header' (line 638)
            header_17241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 21), 'header')
            # Getting the type of 'self' (line 638)
            self_17242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 16), 'self')
            # Obtaining the member '__getitem__' of a type (line 638)
            getitem___17243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 16), self_17242, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 638)
            subscript_call_result_17244 = invoke(stypy.reporting.localization.Localization(__file__, 638, 16), getitem___17243, header_17241)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 12), self_17240, subscript_call_result_17244)
            
            # Assigning a Name to a Subscript (line 639):
            
            # Assigning a Name to a Subscript (line 639):
            # Getting the type of 'new_ctype' (line 639)
            new_ctype_17245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 27), 'new_ctype')
            # Getting the type of 'self' (line 639)
            self_17246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'self')
            # Getting the type of 'header' (line 639)
            header_17247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 17), 'header')
            # Storing an element on a container (line 639)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 639, 12), self_17246, (header_17247, new_ctype_17245))
            # SSA join for if statement (line 637)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'del_param(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'del_param' in the type store
        # Getting the type of 'stypy_return_type' (line 619)
        stypy_return_type_17248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17248)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'del_param'
        return stypy_return_type_17248


    @norecursion
    def set_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_17249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 36), 'str', 'Content-Type')
        # Getting the type of 'True' (line 641)
        True_17250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 60), 'True')
        defaults = [str_17249, True_17250]
        # Create a new context for function 'set_type'
        module_type_store = module_type_store.open_function_context('set_type', 641, 4, False)
        # Assigning a type to the variable 'self' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.set_type.__dict__.__setitem__('stypy_localization', localization)
        Message.set_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.set_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.set_type.__dict__.__setitem__('stypy_function_name', 'Message.set_type')
        Message.set_type.__dict__.__setitem__('stypy_param_names_list', ['type', 'header', 'requote'])
        Message.set_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.set_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.set_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.set_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.set_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.set_type.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.set_type', ['type', 'header', 'requote'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_type', localization, ['type', 'header', 'requote'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_type(...)' code ##################

        str_17251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, (-1)), 'str', 'Set the main type and subtype for the Content-Type header.\n\n        type must be a string in the form "maintype/subtype", otherwise a\n        ValueError is raised.\n\n        This method replaces the Content-Type header, keeping all the\n        parameters in place.  If requote is False, this leaves the existing\n        header\'s quoting as is.  Otherwise, the parameters will be quoted (the\n        default).\n\n        An alternative header can be specified in the header argument.  When\n        the Content-Type header is set, we\'ll always also add a MIME-Version\n        header.\n        ')
        
        
        
        # Call to count(...): (line 657)
        # Processing the call arguments (line 657)
        str_17254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 26), 'str', '/')
        # Processing the call keyword arguments (line 657)
        kwargs_17255 = {}
        # Getting the type of 'type' (line 657)
        type_17252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 15), 'type', False)
        # Obtaining the member 'count' of a type (line 657)
        count_17253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 15), type_17252, 'count')
        # Calling count(args, kwargs) (line 657)
        count_call_result_17256 = invoke(stypy.reporting.localization.Localization(__file__, 657, 15), count_17253, *[str_17254], **kwargs_17255)
        
        int_17257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 34), 'int')
        # Applying the binary operator '==' (line 657)
        result_eq_17258 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 15), '==', count_call_result_17256, int_17257)
        
        # Applying the 'not' unary operator (line 657)
        result_not__17259 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 11), 'not', result_eq_17258)
        
        # Testing if the type of an if condition is none (line 657)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 657, 8), result_not__17259):
            pass
        else:
            
            # Testing the type of an if condition (line 657)
            if_condition_17260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 8), result_not__17259)
            # Assigning a type to the variable 'if_condition_17260' (line 657)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'if_condition_17260', if_condition_17260)
            # SSA begins for if statement (line 657)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'ValueError' (line 658)
            ValueError_17261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 18), 'ValueError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 658, 12), ValueError_17261, 'raise parameter', BaseException)
            # SSA join for if statement (line 657)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to lower(...): (line 660)
        # Processing the call keyword arguments (line 660)
        kwargs_17264 = {}
        # Getting the type of 'header' (line 660)
        header_17262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 11), 'header', False)
        # Obtaining the member 'lower' of a type (line 660)
        lower_17263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 11), header_17262, 'lower')
        # Calling lower(args, kwargs) (line 660)
        lower_call_result_17265 = invoke(stypy.reporting.localization.Localization(__file__, 660, 11), lower_17263, *[], **kwargs_17264)
        
        str_17266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 29), 'str', 'content-type')
        # Applying the binary operator '==' (line 660)
        result_eq_17267 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 11), '==', lower_call_result_17265, str_17266)
        
        # Testing if the type of an if condition is none (line 660)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 660, 8), result_eq_17267):
            pass
        else:
            
            # Testing the type of an if condition (line 660)
            if_condition_17268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 660, 8), result_eq_17267)
            # Assigning a type to the variable 'if_condition_17268' (line 660)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'if_condition_17268', if_condition_17268)
            # SSA begins for if statement (line 660)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Deleting a member
            # Getting the type of 'self' (line 661)
            self_17269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'self')
            
            # Obtaining the type of the subscript
            str_17270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 21), 'str', 'mime-version')
            # Getting the type of 'self' (line 661)
            self_17271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'self')
            # Obtaining the member '__getitem__' of a type (line 661)
            getitem___17272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 16), self_17271, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 661)
            subscript_call_result_17273 = invoke(stypy.reporting.localization.Localization(__file__, 661, 16), getitem___17272, str_17270)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 661, 12), self_17269, subscript_call_result_17273)
            
            # Assigning a Str to a Subscript (line 662):
            
            # Assigning a Str to a Subscript (line 662):
            str_17274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 35), 'str', '1.0')
            # Getting the type of 'self' (line 662)
            self_17275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'self')
            str_17276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 17), 'str', 'MIME-Version')
            # Storing an element on a container (line 662)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 12), self_17275, (str_17276, str_17274))
            # SSA join for if statement (line 660)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'header' (line 663)
        header_17277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 11), 'header')
        # Getting the type of 'self' (line 663)
        self_17278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 25), 'self')
        # Applying the binary operator 'notin' (line 663)
        result_contains_17279 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 11), 'notin', header_17277, self_17278)
        
        # Testing if the type of an if condition is none (line 663)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 663, 8), result_contains_17279):
            pass
        else:
            
            # Testing the type of an if condition (line 663)
            if_condition_17280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 663, 8), result_contains_17279)
            # Assigning a type to the variable 'if_condition_17280' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'if_condition_17280', if_condition_17280)
            # SSA begins for if statement (line 663)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Subscript (line 664):
            
            # Assigning a Name to a Subscript (line 664):
            # Getting the type of 'type' (line 664)
            type_17281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 27), 'type')
            # Getting the type of 'self' (line 664)
            self_17282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'self')
            # Getting the type of 'header' (line 664)
            header_17283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 17), 'header')
            # Storing an element on a container (line 664)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 12), self_17282, (header_17283, type_17281))
            # Assigning a type to the variable 'stypy_return_type' (line 665)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 663)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 666):
        
        # Assigning a Call to a Name (line 666):
        
        # Call to get_params(...): (line 666)
        # Processing the call keyword arguments (line 666)
        # Getting the type of 'header' (line 666)
        header_17286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 40), 'header', False)
        keyword_17287 = header_17286
        # Getting the type of 'requote' (line 666)
        requote_17288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 56), 'requote', False)
        keyword_17289 = requote_17288
        kwargs_17290 = {'unquote': keyword_17289, 'header': keyword_17287}
        # Getting the type of 'self' (line 666)
        self_17284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 17), 'self', False)
        # Obtaining the member 'get_params' of a type (line 666)
        get_params_17285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 17), self_17284, 'get_params')
        # Calling get_params(args, kwargs) (line 666)
        get_params_call_result_17291 = invoke(stypy.reporting.localization.Localization(__file__, 666, 17), get_params_17285, *[], **kwargs_17290)
        
        # Assigning a type to the variable 'params' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'params', get_params_call_result_17291)
        # Deleting a member
        # Getting the type of 'self' (line 667)
        self_17292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'self')
        
        # Obtaining the type of the subscript
        # Getting the type of 'header' (line 667)
        header_17293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 17), 'header')
        # Getting the type of 'self' (line 667)
        self_17294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'self')
        # Obtaining the member '__getitem__' of a type (line 667)
        getitem___17295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 12), self_17294, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 667)
        subscript_call_result_17296 = invoke(stypy.reporting.localization.Localization(__file__, 667, 12), getitem___17295, header_17293)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 8), self_17292, subscript_call_result_17296)
        
        # Assigning a Name to a Subscript (line 668):
        
        # Assigning a Name to a Subscript (line 668):
        # Getting the type of 'type' (line 668)
        type_17297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 23), 'type')
        # Getting the type of 'self' (line 668)
        self_17298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'self')
        # Getting the type of 'header' (line 668)
        header_17299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 13), 'header')
        # Storing an element on a container (line 668)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 668, 8), self_17298, (header_17299, type_17297))
        
        
        # Obtaining the type of the subscript
        int_17300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 27), 'int')
        slice_17301 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 670, 20), int_17300, None, None)
        # Getting the type of 'params' (line 670)
        params_17302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 20), 'params')
        # Obtaining the member '__getitem__' of a type (line 670)
        getitem___17303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 20), params_17302, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 670)
        subscript_call_result_17304 = invoke(stypy.reporting.localization.Localization(__file__, 670, 20), getitem___17303, slice_17301)
        
        # Assigning a type to the variable 'subscript_call_result_17304' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'subscript_call_result_17304', subscript_call_result_17304)
        # Testing if the for loop is going to be iterated (line 670)
        # Testing the type of a for loop iterable (line 670)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 670, 8), subscript_call_result_17304)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 670, 8), subscript_call_result_17304):
            # Getting the type of the for loop variable (line 670)
            for_loop_var_17305 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 670, 8), subscript_call_result_17304)
            # Assigning a type to the variable 'p' (line 670)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 670, 8), for_loop_var_17305, 2, 0))
            # Assigning a type to the variable 'v' (line 670)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 670, 8), for_loop_var_17305, 2, 1))
            # SSA begins for a for statement (line 670)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to set_param(...): (line 671)
            # Processing the call arguments (line 671)
            # Getting the type of 'p' (line 671)
            p_17308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 27), 'p', False)
            # Getting the type of 'v' (line 671)
            v_17309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 30), 'v', False)
            # Getting the type of 'header' (line 671)
            header_17310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 33), 'header', False)
            # Getting the type of 'requote' (line 671)
            requote_17311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 41), 'requote', False)
            # Processing the call keyword arguments (line 671)
            kwargs_17312 = {}
            # Getting the type of 'self' (line 671)
            self_17306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'self', False)
            # Obtaining the member 'set_param' of a type (line 671)
            set_param_17307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 12), self_17306, 'set_param')
            # Calling set_param(args, kwargs) (line 671)
            set_param_call_result_17313 = invoke(stypy.reporting.localization.Localization(__file__, 671, 12), set_param_17307, *[p_17308, v_17309, header_17310, requote_17311], **kwargs_17312)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'set_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type' in the type store
        # Getting the type of 'stypy_return_type' (line 641)
        stypy_return_type_17314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17314)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type'
        return stypy_return_type_17314


    @norecursion
    def get_filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 673)
        None_17315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 35), 'None')
        defaults = [None_17315]
        # Create a new context for function 'get_filename'
        module_type_store = module_type_store.open_function_context('get_filename', 673, 4, False)
        # Assigning a type to the variable 'self' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_filename.__dict__.__setitem__('stypy_localization', localization)
        Message.get_filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_filename.__dict__.__setitem__('stypy_function_name', 'Message.get_filename')
        Message.get_filename.__dict__.__setitem__('stypy_param_names_list', ['failobj'])
        Message.get_filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_filename.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_filename', ['failobj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_filename', localization, ['failobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_filename(...)' code ##################

        str_17316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, (-1)), 'str', "Return the filename associated with the payload if present.\n\n        The filename is extracted from the Content-Disposition header's\n        `filename' parameter, and it is unquoted.  If that header is missing\n        the `filename' parameter, this method falls back to looking for the\n        `name' parameter.\n        ")
        
        # Assigning a Call to a Name (line 681):
        
        # Assigning a Call to a Name (line 681):
        
        # Call to object(...): (line 681)
        # Processing the call keyword arguments (line 681)
        kwargs_17318 = {}
        # Getting the type of 'object' (line 681)
        object_17317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 18), 'object', False)
        # Calling object(args, kwargs) (line 681)
        object_call_result_17319 = invoke(stypy.reporting.localization.Localization(__file__, 681, 18), object_17317, *[], **kwargs_17318)
        
        # Assigning a type to the variable 'missing' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'missing', object_call_result_17319)
        
        # Assigning a Call to a Name (line 682):
        
        # Assigning a Call to a Name (line 682):
        
        # Call to get_param(...): (line 682)
        # Processing the call arguments (line 682)
        str_17322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 34), 'str', 'filename')
        # Getting the type of 'missing' (line 682)
        missing_17323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 46), 'missing', False)
        str_17324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 55), 'str', 'content-disposition')
        # Processing the call keyword arguments (line 682)
        kwargs_17325 = {}
        # Getting the type of 'self' (line 682)
        self_17320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 19), 'self', False)
        # Obtaining the member 'get_param' of a type (line 682)
        get_param_17321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 19), self_17320, 'get_param')
        # Calling get_param(args, kwargs) (line 682)
        get_param_call_result_17326 = invoke(stypy.reporting.localization.Localization(__file__, 682, 19), get_param_17321, *[str_17322, missing_17323, str_17324], **kwargs_17325)
        
        # Assigning a type to the variable 'filename' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'filename', get_param_call_result_17326)
        
        # Getting the type of 'filename' (line 683)
        filename_17327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 11), 'filename')
        # Getting the type of 'missing' (line 683)
        missing_17328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 23), 'missing')
        # Applying the binary operator 'is' (line 683)
        result_is__17329 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 11), 'is', filename_17327, missing_17328)
        
        # Testing if the type of an if condition is none (line 683)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 683, 8), result_is__17329):
            pass
        else:
            
            # Testing the type of an if condition (line 683)
            if_condition_17330 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 683, 8), result_is__17329)
            # Assigning a type to the variable 'if_condition_17330' (line 683)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'if_condition_17330', if_condition_17330)
            # SSA begins for if statement (line 683)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 684):
            
            # Assigning a Call to a Name (line 684):
            
            # Call to get_param(...): (line 684)
            # Processing the call arguments (line 684)
            str_17333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 38), 'str', 'name')
            # Getting the type of 'missing' (line 684)
            missing_17334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 46), 'missing', False)
            str_17335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 55), 'str', 'content-type')
            # Processing the call keyword arguments (line 684)
            kwargs_17336 = {}
            # Getting the type of 'self' (line 684)
            self_17331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 23), 'self', False)
            # Obtaining the member 'get_param' of a type (line 684)
            get_param_17332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 23), self_17331, 'get_param')
            # Calling get_param(args, kwargs) (line 684)
            get_param_call_result_17337 = invoke(stypy.reporting.localization.Localization(__file__, 684, 23), get_param_17332, *[str_17333, missing_17334, str_17335], **kwargs_17336)
            
            # Assigning a type to the variable 'filename' (line 684)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'filename', get_param_call_result_17337)
            # SSA join for if statement (line 683)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'filename' (line 685)
        filename_17338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 11), 'filename')
        # Getting the type of 'missing' (line 685)
        missing_17339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 23), 'missing')
        # Applying the binary operator 'is' (line 685)
        result_is__17340 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 11), 'is', filename_17338, missing_17339)
        
        # Testing if the type of an if condition is none (line 685)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 685, 8), result_is__17340):
            pass
        else:
            
            # Testing the type of an if condition (line 685)
            if_condition_17341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 685, 8), result_is__17340)
            # Assigning a type to the variable 'if_condition_17341' (line 685)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 8), 'if_condition_17341', if_condition_17341)
            # SSA begins for if statement (line 685)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'failobj' (line 686)
            failobj_17342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 19), 'failobj')
            # Assigning a type to the variable 'stypy_return_type' (line 686)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 12), 'stypy_return_type', failobj_17342)
            # SSA join for if statement (line 685)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to strip(...): (line 687)
        # Processing the call keyword arguments (line 687)
        kwargs_17349 = {}
        
        # Call to collapse_rfc2231_value(...): (line 687)
        # Processing the call arguments (line 687)
        # Getting the type of 'filename' (line 687)
        filename_17345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 44), 'filename', False)
        # Processing the call keyword arguments (line 687)
        kwargs_17346 = {}
        # Getting the type of 'utils' (line 687)
        utils_17343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 15), 'utils', False)
        # Obtaining the member 'collapse_rfc2231_value' of a type (line 687)
        collapse_rfc2231_value_17344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 15), utils_17343, 'collapse_rfc2231_value')
        # Calling collapse_rfc2231_value(args, kwargs) (line 687)
        collapse_rfc2231_value_call_result_17347 = invoke(stypy.reporting.localization.Localization(__file__, 687, 15), collapse_rfc2231_value_17344, *[filename_17345], **kwargs_17346)
        
        # Obtaining the member 'strip' of a type (line 687)
        strip_17348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 15), collapse_rfc2231_value_call_result_17347, 'strip')
        # Calling strip(args, kwargs) (line 687)
        strip_call_result_17350 = invoke(stypy.reporting.localization.Localization(__file__, 687, 15), strip_17348, *[], **kwargs_17349)
        
        # Assigning a type to the variable 'stypy_return_type' (line 687)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 8), 'stypy_return_type', strip_call_result_17350)
        
        # ################# End of 'get_filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_filename' in the type store
        # Getting the type of 'stypy_return_type' (line 673)
        stypy_return_type_17351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_filename'
        return stypy_return_type_17351


    @norecursion
    def get_boundary(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 689)
        None_17352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 35), 'None')
        defaults = [None_17352]
        # Create a new context for function 'get_boundary'
        module_type_store = module_type_store.open_function_context('get_boundary', 689, 4, False)
        # Assigning a type to the variable 'self' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_boundary.__dict__.__setitem__('stypy_localization', localization)
        Message.get_boundary.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_boundary.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_boundary.__dict__.__setitem__('stypy_function_name', 'Message.get_boundary')
        Message.get_boundary.__dict__.__setitem__('stypy_param_names_list', ['failobj'])
        Message.get_boundary.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_boundary.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_boundary.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_boundary.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_boundary.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_boundary.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_boundary', ['failobj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_boundary', localization, ['failobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_boundary(...)' code ##################

        str_17353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, (-1)), 'str', "Return the boundary associated with the payload if present.\n\n        The boundary is extracted from the Content-Type header's `boundary'\n        parameter, and it is unquoted.\n        ")
        
        # Assigning a Call to a Name (line 695):
        
        # Assigning a Call to a Name (line 695):
        
        # Call to object(...): (line 695)
        # Processing the call keyword arguments (line 695)
        kwargs_17355 = {}
        # Getting the type of 'object' (line 695)
        object_17354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 18), 'object', False)
        # Calling object(args, kwargs) (line 695)
        object_call_result_17356 = invoke(stypy.reporting.localization.Localization(__file__, 695, 18), object_17354, *[], **kwargs_17355)
        
        # Assigning a type to the variable 'missing' (line 695)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'missing', object_call_result_17356)
        
        # Assigning a Call to a Name (line 696):
        
        # Assigning a Call to a Name (line 696):
        
        # Call to get_param(...): (line 696)
        # Processing the call arguments (line 696)
        str_17359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 34), 'str', 'boundary')
        # Getting the type of 'missing' (line 696)
        missing_17360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 46), 'missing', False)
        # Processing the call keyword arguments (line 696)
        kwargs_17361 = {}
        # Getting the type of 'self' (line 696)
        self_17357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 19), 'self', False)
        # Obtaining the member 'get_param' of a type (line 696)
        get_param_17358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 19), self_17357, 'get_param')
        # Calling get_param(args, kwargs) (line 696)
        get_param_call_result_17362 = invoke(stypy.reporting.localization.Localization(__file__, 696, 19), get_param_17358, *[str_17359, missing_17360], **kwargs_17361)
        
        # Assigning a type to the variable 'boundary' (line 696)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'boundary', get_param_call_result_17362)
        
        # Getting the type of 'boundary' (line 697)
        boundary_17363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 11), 'boundary')
        # Getting the type of 'missing' (line 697)
        missing_17364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 23), 'missing')
        # Applying the binary operator 'is' (line 697)
        result_is__17365 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 11), 'is', boundary_17363, missing_17364)
        
        # Testing if the type of an if condition is none (line 697)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 697, 8), result_is__17365):
            pass
        else:
            
            # Testing the type of an if condition (line 697)
            if_condition_17366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 8), result_is__17365)
            # Assigning a type to the variable 'if_condition_17366' (line 697)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'if_condition_17366', if_condition_17366)
            # SSA begins for if statement (line 697)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'failobj' (line 698)
            failobj_17367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 19), 'failobj')
            # Assigning a type to the variable 'stypy_return_type' (line 698)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 12), 'stypy_return_type', failobj_17367)
            # SSA join for if statement (line 697)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to rstrip(...): (line 700)
        # Processing the call keyword arguments (line 700)
        kwargs_17374 = {}
        
        # Call to collapse_rfc2231_value(...): (line 700)
        # Processing the call arguments (line 700)
        # Getting the type of 'boundary' (line 700)
        boundary_17370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 44), 'boundary', False)
        # Processing the call keyword arguments (line 700)
        kwargs_17371 = {}
        # Getting the type of 'utils' (line 700)
        utils_17368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 15), 'utils', False)
        # Obtaining the member 'collapse_rfc2231_value' of a type (line 700)
        collapse_rfc2231_value_17369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 15), utils_17368, 'collapse_rfc2231_value')
        # Calling collapse_rfc2231_value(args, kwargs) (line 700)
        collapse_rfc2231_value_call_result_17372 = invoke(stypy.reporting.localization.Localization(__file__, 700, 15), collapse_rfc2231_value_17369, *[boundary_17370], **kwargs_17371)
        
        # Obtaining the member 'rstrip' of a type (line 700)
        rstrip_17373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 15), collapse_rfc2231_value_call_result_17372, 'rstrip')
        # Calling rstrip(args, kwargs) (line 700)
        rstrip_call_result_17375 = invoke(stypy.reporting.localization.Localization(__file__, 700, 15), rstrip_17373, *[], **kwargs_17374)
        
        # Assigning a type to the variable 'stypy_return_type' (line 700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'stypy_return_type', rstrip_call_result_17375)
        
        # ################# End of 'get_boundary(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_boundary' in the type store
        # Getting the type of 'stypy_return_type' (line 689)
        stypy_return_type_17376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_boundary'
        return stypy_return_type_17376


    @norecursion
    def set_boundary(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_boundary'
        module_type_store = module_type_store.open_function_context('set_boundary', 702, 4, False)
        # Assigning a type to the variable 'self' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.set_boundary.__dict__.__setitem__('stypy_localization', localization)
        Message.set_boundary.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.set_boundary.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.set_boundary.__dict__.__setitem__('stypy_function_name', 'Message.set_boundary')
        Message.set_boundary.__dict__.__setitem__('stypy_param_names_list', ['boundary'])
        Message.set_boundary.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.set_boundary.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.set_boundary.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.set_boundary.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.set_boundary.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.set_boundary.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.set_boundary', ['boundary'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_boundary', localization, ['boundary'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_boundary(...)' code ##################

        str_17377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, (-1)), 'str', "Set the boundary parameter in Content-Type to 'boundary'.\n\n        This is subtly different than deleting the Content-Type header and\n        adding a new one with a new boundary parameter via add_header().  The\n        main difference is that using the set_boundary() method preserves the\n        order of the Content-Type header in the original message.\n\n        HeaderParseError is raised if the message has no Content-Type header.\n        ")
        
        # Assigning a Call to a Name (line 712):
        
        # Assigning a Call to a Name (line 712):
        
        # Call to object(...): (line 712)
        # Processing the call keyword arguments (line 712)
        kwargs_17379 = {}
        # Getting the type of 'object' (line 712)
        object_17378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 18), 'object', False)
        # Calling object(args, kwargs) (line 712)
        object_call_result_17380 = invoke(stypy.reporting.localization.Localization(__file__, 712, 18), object_17378, *[], **kwargs_17379)
        
        # Assigning a type to the variable 'missing' (line 712)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'missing', object_call_result_17380)
        
        # Assigning a Call to a Name (line 713):
        
        # Assigning a Call to a Name (line 713):
        
        # Call to _get_params_preserve(...): (line 713)
        # Processing the call arguments (line 713)
        # Getting the type of 'missing' (line 713)
        missing_17383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 43), 'missing', False)
        str_17384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 52), 'str', 'content-type')
        # Processing the call keyword arguments (line 713)
        kwargs_17385 = {}
        # Getting the type of 'self' (line 713)
        self_17381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 17), 'self', False)
        # Obtaining the member '_get_params_preserve' of a type (line 713)
        _get_params_preserve_17382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 17), self_17381, '_get_params_preserve')
        # Calling _get_params_preserve(args, kwargs) (line 713)
        _get_params_preserve_call_result_17386 = invoke(stypy.reporting.localization.Localization(__file__, 713, 17), _get_params_preserve_17382, *[missing_17383, str_17384], **kwargs_17385)
        
        # Assigning a type to the variable 'params' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'params', _get_params_preserve_call_result_17386)
        
        # Getting the type of 'params' (line 714)
        params_17387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 11), 'params')
        # Getting the type of 'missing' (line 714)
        missing_17388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 21), 'missing')
        # Applying the binary operator 'is' (line 714)
        result_is__17389 = python_operator(stypy.reporting.localization.Localization(__file__, 714, 11), 'is', params_17387, missing_17388)
        
        # Testing if the type of an if condition is none (line 714)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 714, 8), result_is__17389):
            pass
        else:
            
            # Testing the type of an if condition (line 714)
            if_condition_17390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 714, 8), result_is__17389)
            # Assigning a type to the variable 'if_condition_17390' (line 714)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'if_condition_17390', if_condition_17390)
            # SSA begins for if statement (line 714)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to HeaderParseError(...): (line 717)
            # Processing the call arguments (line 717)
            str_17393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 42), 'str', 'No Content-Type header found')
            # Processing the call keyword arguments (line 717)
            kwargs_17394 = {}
            # Getting the type of 'errors' (line 717)
            errors_17391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 18), 'errors', False)
            # Obtaining the member 'HeaderParseError' of a type (line 717)
            HeaderParseError_17392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 18), errors_17391, 'HeaderParseError')
            # Calling HeaderParseError(args, kwargs) (line 717)
            HeaderParseError_call_result_17395 = invoke(stypy.reporting.localization.Localization(__file__, 717, 18), HeaderParseError_17392, *[str_17393], **kwargs_17394)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 717, 12), HeaderParseError_call_result_17395, 'raise parameter', BaseException)
            # SSA join for if statement (line 714)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a List to a Name (line 718):
        
        # Assigning a List to a Name (line 718):
        
        # Obtaining an instance of the builtin type 'list' (line 718)
        list_17396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 718)
        
        # Assigning a type to the variable 'newparams' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'newparams', list_17396)
        
        # Assigning a Name to a Name (line 719):
        
        # Assigning a Name to a Name (line 719):
        # Getting the type of 'False' (line 719)
        False_17397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 17), 'False')
        # Assigning a type to the variable 'foundp' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'foundp', False_17397)
        
        # Getting the type of 'params' (line 720)
        params_17398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 22), 'params')
        # Assigning a type to the variable 'params_17398' (line 720)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'params_17398', params_17398)
        # Testing if the for loop is going to be iterated (line 720)
        # Testing the type of a for loop iterable (line 720)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 720, 8), params_17398)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 720, 8), params_17398):
            # Getting the type of the for loop variable (line 720)
            for_loop_var_17399 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 720, 8), params_17398)
            # Assigning a type to the variable 'pk' (line 720)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'pk', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 8), for_loop_var_17399, 2, 0))
            # Assigning a type to the variable 'pv' (line 720)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'pv', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 8), for_loop_var_17399, 2, 1))
            # SSA begins for a for statement (line 720)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to lower(...): (line 721)
            # Processing the call keyword arguments (line 721)
            kwargs_17402 = {}
            # Getting the type of 'pk' (line 721)
            pk_17400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 15), 'pk', False)
            # Obtaining the member 'lower' of a type (line 721)
            lower_17401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 15), pk_17400, 'lower')
            # Calling lower(args, kwargs) (line 721)
            lower_call_result_17403 = invoke(stypy.reporting.localization.Localization(__file__, 721, 15), lower_17401, *[], **kwargs_17402)
            
            str_17404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 29), 'str', 'boundary')
            # Applying the binary operator '==' (line 721)
            result_eq_17405 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 15), '==', lower_call_result_17403, str_17404)
            
            # Testing if the type of an if condition is none (line 721)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 721, 12), result_eq_17405):
                
                # Call to append(...): (line 725)
                # Processing the call arguments (line 725)
                
                # Obtaining an instance of the builtin type 'tuple' (line 725)
                tuple_17419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 34), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 725)
                # Adding element type (line 725)
                # Getting the type of 'pk' (line 725)
                pk_17420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 34), 'pk', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 34), tuple_17419, pk_17420)
                # Adding element type (line 725)
                # Getting the type of 'pv' (line 725)
                pv_17421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 38), 'pv', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 34), tuple_17419, pv_17421)
                
                # Processing the call keyword arguments (line 725)
                kwargs_17422 = {}
                # Getting the type of 'newparams' (line 725)
                newparams_17417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 16), 'newparams', False)
                # Obtaining the member 'append' of a type (line 725)
                append_17418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 16), newparams_17417, 'append')
                # Calling append(args, kwargs) (line 725)
                append_call_result_17423 = invoke(stypy.reporting.localization.Localization(__file__, 725, 16), append_17418, *[tuple_17419], **kwargs_17422)
                
            else:
                
                # Testing the type of an if condition (line 721)
                if_condition_17406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 721, 12), result_eq_17405)
                # Assigning a type to the variable 'if_condition_17406' (line 721)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 12), 'if_condition_17406', if_condition_17406)
                # SSA begins for if statement (line 721)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 722)
                # Processing the call arguments (line 722)
                
                # Obtaining an instance of the builtin type 'tuple' (line 722)
                tuple_17409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 34), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 722)
                # Adding element type (line 722)
                str_17410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 34), 'str', 'boundary')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 34), tuple_17409, str_17410)
                # Adding element type (line 722)
                str_17411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 46), 'str', '"%s"')
                # Getting the type of 'boundary' (line 722)
                boundary_17412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 55), 'boundary', False)
                # Applying the binary operator '%' (line 722)
                result_mod_17413 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 46), '%', str_17411, boundary_17412)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 34), tuple_17409, result_mod_17413)
                
                # Processing the call keyword arguments (line 722)
                kwargs_17414 = {}
                # Getting the type of 'newparams' (line 722)
                newparams_17407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 16), 'newparams', False)
                # Obtaining the member 'append' of a type (line 722)
                append_17408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 16), newparams_17407, 'append')
                # Calling append(args, kwargs) (line 722)
                append_call_result_17415 = invoke(stypy.reporting.localization.Localization(__file__, 722, 16), append_17408, *[tuple_17409], **kwargs_17414)
                
                
                # Assigning a Name to a Name (line 723):
                
                # Assigning a Name to a Name (line 723):
                # Getting the type of 'True' (line 723)
                True_17416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 25), 'True')
                # Assigning a type to the variable 'foundp' (line 723)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 16), 'foundp', True_17416)
                # SSA branch for the else part of an if statement (line 721)
                module_type_store.open_ssa_branch('else')
                
                # Call to append(...): (line 725)
                # Processing the call arguments (line 725)
                
                # Obtaining an instance of the builtin type 'tuple' (line 725)
                tuple_17419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 34), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 725)
                # Adding element type (line 725)
                # Getting the type of 'pk' (line 725)
                pk_17420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 34), 'pk', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 34), tuple_17419, pk_17420)
                # Adding element type (line 725)
                # Getting the type of 'pv' (line 725)
                pv_17421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 38), 'pv', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 34), tuple_17419, pv_17421)
                
                # Processing the call keyword arguments (line 725)
                kwargs_17422 = {}
                # Getting the type of 'newparams' (line 725)
                newparams_17417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 16), 'newparams', False)
                # Obtaining the member 'append' of a type (line 725)
                append_17418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 16), newparams_17417, 'append')
                # Calling append(args, kwargs) (line 725)
                append_call_result_17423 = invoke(stypy.reporting.localization.Localization(__file__, 725, 16), append_17418, *[tuple_17419], **kwargs_17422)
                
                # SSA join for if statement (line 721)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'foundp' (line 726)
        foundp_17424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 15), 'foundp')
        # Applying the 'not' unary operator (line 726)
        result_not__17425 = python_operator(stypy.reporting.localization.Localization(__file__, 726, 11), 'not', foundp_17424)
        
        # Testing if the type of an if condition is none (line 726)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 726, 8), result_not__17425):
            pass
        else:
            
            # Testing the type of an if condition (line 726)
            if_condition_17426 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 726, 8), result_not__17425)
            # Assigning a type to the variable 'if_condition_17426' (line 726)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'if_condition_17426', if_condition_17426)
            # SSA begins for if statement (line 726)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 730)
            # Processing the call arguments (line 730)
            
            # Obtaining an instance of the builtin type 'tuple' (line 730)
            tuple_17429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 730)
            # Adding element type (line 730)
            str_17430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 30), 'str', 'boundary')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 730, 30), tuple_17429, str_17430)
            # Adding element type (line 730)
            str_17431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 42), 'str', '"%s"')
            # Getting the type of 'boundary' (line 730)
            boundary_17432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 51), 'boundary', False)
            # Applying the binary operator '%' (line 730)
            result_mod_17433 = python_operator(stypy.reporting.localization.Localization(__file__, 730, 42), '%', str_17431, boundary_17432)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 730, 30), tuple_17429, result_mod_17433)
            
            # Processing the call keyword arguments (line 730)
            kwargs_17434 = {}
            # Getting the type of 'newparams' (line 730)
            newparams_17427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'newparams', False)
            # Obtaining the member 'append' of a type (line 730)
            append_17428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 12), newparams_17427, 'append')
            # Calling append(args, kwargs) (line 730)
            append_call_result_17435 = invoke(stypy.reporting.localization.Localization(__file__, 730, 12), append_17428, *[tuple_17429], **kwargs_17434)
            
            # SSA join for if statement (line 726)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a List to a Name (line 732):
        
        # Assigning a List to a Name (line 732):
        
        # Obtaining an instance of the builtin type 'list' (line 732)
        list_17436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 732)
        
        # Assigning a type to the variable 'newheaders' (line 732)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 8), 'newheaders', list_17436)
        
        # Getting the type of 'self' (line 733)
        self_17437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 20), 'self')
        # Obtaining the member '_headers' of a type (line 733)
        _headers_17438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 20), self_17437, '_headers')
        # Assigning a type to the variable '_headers_17438' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), '_headers_17438', _headers_17438)
        # Testing if the for loop is going to be iterated (line 733)
        # Testing the type of a for loop iterable (line 733)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 733, 8), _headers_17438)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 733, 8), _headers_17438):
            # Getting the type of the for loop variable (line 733)
            for_loop_var_17439 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 733, 8), _headers_17438)
            # Assigning a type to the variable 'h' (line 733)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'h', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 8), for_loop_var_17439, 2, 0))
            # Assigning a type to the variable 'v' (line 733)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 8), for_loop_var_17439, 2, 1))
            # SSA begins for a for statement (line 733)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to lower(...): (line 734)
            # Processing the call keyword arguments (line 734)
            kwargs_17442 = {}
            # Getting the type of 'h' (line 734)
            h_17440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 15), 'h', False)
            # Obtaining the member 'lower' of a type (line 734)
            lower_17441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 15), h_17440, 'lower')
            # Calling lower(args, kwargs) (line 734)
            lower_call_result_17443 = invoke(stypy.reporting.localization.Localization(__file__, 734, 15), lower_17441, *[], **kwargs_17442)
            
            str_17444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 28), 'str', 'content-type')
            # Applying the binary operator '==' (line 734)
            result_eq_17445 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 15), '==', lower_call_result_17443, str_17444)
            
            # Testing if the type of an if condition is none (line 734)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 734, 12), result_eq_17445):
                
                # Call to append(...): (line 744)
                # Processing the call arguments (line 744)
                
                # Obtaining an instance of the builtin type 'tuple' (line 744)
                tuple_17481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 35), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 744)
                # Adding element type (line 744)
                # Getting the type of 'h' (line 744)
                h_17482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 35), 'h', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 35), tuple_17481, h_17482)
                # Adding element type (line 744)
                # Getting the type of 'v' (line 744)
                v_17483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 38), 'v', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 35), tuple_17481, v_17483)
                
                # Processing the call keyword arguments (line 744)
                kwargs_17484 = {}
                # Getting the type of 'newheaders' (line 744)
                newheaders_17479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 16), 'newheaders', False)
                # Obtaining the member 'append' of a type (line 744)
                append_17480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 16), newheaders_17479, 'append')
                # Calling append(args, kwargs) (line 744)
                append_call_result_17485 = invoke(stypy.reporting.localization.Localization(__file__, 744, 16), append_17480, *[tuple_17481], **kwargs_17484)
                
            else:
                
                # Testing the type of an if condition (line 734)
                if_condition_17446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 734, 12), result_eq_17445)
                # Assigning a type to the variable 'if_condition_17446' (line 734)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 12), 'if_condition_17446', if_condition_17446)
                # SSA begins for if statement (line 734)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a List to a Name (line 735):
                
                # Assigning a List to a Name (line 735):
                
                # Obtaining an instance of the builtin type 'list' (line 735)
                list_17447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 24), 'list')
                # Adding type elements to the builtin type 'list' instance (line 735)
                
                # Assigning a type to the variable 'parts' (line 735)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 16), 'parts', list_17447)
                
                # Getting the type of 'newparams' (line 736)
                newparams_17448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 28), 'newparams')
                # Assigning a type to the variable 'newparams_17448' (line 736)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 16), 'newparams_17448', newparams_17448)
                # Testing if the for loop is going to be iterated (line 736)
                # Testing the type of a for loop iterable (line 736)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 736, 16), newparams_17448)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 736, 16), newparams_17448):
                    # Getting the type of the for loop variable (line 736)
                    for_loop_var_17449 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 736, 16), newparams_17448)
                    # Assigning a type to the variable 'k' (line 736)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 16), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 736, 16), for_loop_var_17449, 2, 0))
                    # Assigning a type to the variable 'v' (line 736)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 16), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 736, 16), for_loop_var_17449, 2, 1))
                    # SSA begins for a for statement (line 736)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'v' (line 737)
                    v_17450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 23), 'v')
                    str_17451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 28), 'str', '')
                    # Applying the binary operator '==' (line 737)
                    result_eq_17452 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 23), '==', v_17450, str_17451)
                    
                    # Testing if the type of an if condition is none (line 737)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 737, 20), result_eq_17452):
                        
                        # Call to append(...): (line 740)
                        # Processing the call arguments (line 740)
                        str_17461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 37), 'str', '%s=%s')
                        
                        # Obtaining an instance of the builtin type 'tuple' (line 740)
                        tuple_17462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 48), 'tuple')
                        # Adding type elements to the builtin type 'tuple' instance (line 740)
                        # Adding element type (line 740)
                        # Getting the type of 'k' (line 740)
                        k_17463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 48), 'k', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 48), tuple_17462, k_17463)
                        # Adding element type (line 740)
                        # Getting the type of 'v' (line 740)
                        v_17464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 51), 'v', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 48), tuple_17462, v_17464)
                        
                        # Applying the binary operator '%' (line 740)
                        result_mod_17465 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 37), '%', str_17461, tuple_17462)
                        
                        # Processing the call keyword arguments (line 740)
                        kwargs_17466 = {}
                        # Getting the type of 'parts' (line 740)
                        parts_17459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 24), 'parts', False)
                        # Obtaining the member 'append' of a type (line 740)
                        append_17460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 24), parts_17459, 'append')
                        # Calling append(args, kwargs) (line 740)
                        append_call_result_17467 = invoke(stypy.reporting.localization.Localization(__file__, 740, 24), append_17460, *[result_mod_17465], **kwargs_17466)
                        
                    else:
                        
                        # Testing the type of an if condition (line 737)
                        if_condition_17453 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 737, 20), result_eq_17452)
                        # Assigning a type to the variable 'if_condition_17453' (line 737)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 20), 'if_condition_17453', if_condition_17453)
                        # SSA begins for if statement (line 737)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 738)
                        # Processing the call arguments (line 738)
                        # Getting the type of 'k' (line 738)
                        k_17456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 37), 'k', False)
                        # Processing the call keyword arguments (line 738)
                        kwargs_17457 = {}
                        # Getting the type of 'parts' (line 738)
                        parts_17454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 24), 'parts', False)
                        # Obtaining the member 'append' of a type (line 738)
                        append_17455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 24), parts_17454, 'append')
                        # Calling append(args, kwargs) (line 738)
                        append_call_result_17458 = invoke(stypy.reporting.localization.Localization(__file__, 738, 24), append_17455, *[k_17456], **kwargs_17457)
                        
                        # SSA branch for the else part of an if statement (line 737)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to append(...): (line 740)
                        # Processing the call arguments (line 740)
                        str_17461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 37), 'str', '%s=%s')
                        
                        # Obtaining an instance of the builtin type 'tuple' (line 740)
                        tuple_17462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 48), 'tuple')
                        # Adding type elements to the builtin type 'tuple' instance (line 740)
                        # Adding element type (line 740)
                        # Getting the type of 'k' (line 740)
                        k_17463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 48), 'k', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 48), tuple_17462, k_17463)
                        # Adding element type (line 740)
                        # Getting the type of 'v' (line 740)
                        v_17464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 51), 'v', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 48), tuple_17462, v_17464)
                        
                        # Applying the binary operator '%' (line 740)
                        result_mod_17465 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 37), '%', str_17461, tuple_17462)
                        
                        # Processing the call keyword arguments (line 740)
                        kwargs_17466 = {}
                        # Getting the type of 'parts' (line 740)
                        parts_17459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 24), 'parts', False)
                        # Obtaining the member 'append' of a type (line 740)
                        append_17460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 24), parts_17459, 'append')
                        # Calling append(args, kwargs) (line 740)
                        append_call_result_17467 = invoke(stypy.reporting.localization.Localization(__file__, 740, 24), append_17460, *[result_mod_17465], **kwargs_17466)
                        
                        # SSA join for if statement (line 737)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Call to append(...): (line 741)
                # Processing the call arguments (line 741)
                
                # Obtaining an instance of the builtin type 'tuple' (line 741)
                tuple_17470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 35), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 741)
                # Adding element type (line 741)
                # Getting the type of 'h' (line 741)
                h_17471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 35), 'h', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 741, 35), tuple_17470, h_17471)
                # Adding element type (line 741)
                
                # Call to join(...): (line 741)
                # Processing the call arguments (line 741)
                # Getting the type of 'parts' (line 741)
                parts_17474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 53), 'parts', False)
                # Processing the call keyword arguments (line 741)
                kwargs_17475 = {}
                # Getting the type of 'SEMISPACE' (line 741)
                SEMISPACE_17472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 38), 'SEMISPACE', False)
                # Obtaining the member 'join' of a type (line 741)
                join_17473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 38), SEMISPACE_17472, 'join')
                # Calling join(args, kwargs) (line 741)
                join_call_result_17476 = invoke(stypy.reporting.localization.Localization(__file__, 741, 38), join_17473, *[parts_17474], **kwargs_17475)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 741, 35), tuple_17470, join_call_result_17476)
                
                # Processing the call keyword arguments (line 741)
                kwargs_17477 = {}
                # Getting the type of 'newheaders' (line 741)
                newheaders_17468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 16), 'newheaders', False)
                # Obtaining the member 'append' of a type (line 741)
                append_17469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 16), newheaders_17468, 'append')
                # Calling append(args, kwargs) (line 741)
                append_call_result_17478 = invoke(stypy.reporting.localization.Localization(__file__, 741, 16), append_17469, *[tuple_17470], **kwargs_17477)
                
                # SSA branch for the else part of an if statement (line 734)
                module_type_store.open_ssa_branch('else')
                
                # Call to append(...): (line 744)
                # Processing the call arguments (line 744)
                
                # Obtaining an instance of the builtin type 'tuple' (line 744)
                tuple_17481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 35), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 744)
                # Adding element type (line 744)
                # Getting the type of 'h' (line 744)
                h_17482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 35), 'h', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 35), tuple_17481, h_17482)
                # Adding element type (line 744)
                # Getting the type of 'v' (line 744)
                v_17483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 38), 'v', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 744, 35), tuple_17481, v_17483)
                
                # Processing the call keyword arguments (line 744)
                kwargs_17484 = {}
                # Getting the type of 'newheaders' (line 744)
                newheaders_17479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 16), 'newheaders', False)
                # Obtaining the member 'append' of a type (line 744)
                append_17480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 16), newheaders_17479, 'append')
                # Calling append(args, kwargs) (line 744)
                append_call_result_17485 = invoke(stypy.reporting.localization.Localization(__file__, 744, 16), append_17480, *[tuple_17481], **kwargs_17484)
                
                # SSA join for if statement (line 734)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Attribute (line 745):
        
        # Assigning a Name to a Attribute (line 745):
        # Getting the type of 'newheaders' (line 745)
        newheaders_17486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 24), 'newheaders')
        # Getting the type of 'self' (line 745)
        self_17487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'self')
        # Setting the type of the member '_headers' of a type (line 745)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 8), self_17487, '_headers', newheaders_17486)
        
        # ################# End of 'set_boundary(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_boundary' in the type store
        # Getting the type of 'stypy_return_type' (line 702)
        stypy_return_type_17488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17488)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_boundary'
        return stypy_return_type_17488


    @norecursion
    def get_content_charset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 747)
        None_17489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 42), 'None')
        defaults = [None_17489]
        # Create a new context for function 'get_content_charset'
        module_type_store = module_type_store.open_function_context('get_content_charset', 747, 4, False)
        # Assigning a type to the variable 'self' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_content_charset.__dict__.__setitem__('stypy_localization', localization)
        Message.get_content_charset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_content_charset.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_content_charset.__dict__.__setitem__('stypy_function_name', 'Message.get_content_charset')
        Message.get_content_charset.__dict__.__setitem__('stypy_param_names_list', ['failobj'])
        Message.get_content_charset.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_content_charset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_content_charset.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_content_charset.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_content_charset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_content_charset.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_content_charset', ['failobj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_content_charset', localization, ['failobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_content_charset(...)' code ##################

        str_17490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, (-1)), 'str', 'Return the charset parameter of the Content-Type header.\n\n        The returned string is always coerced to lower case.  If there is no\n        Content-Type header, or if that header has no charset parameter,\n        failobj is returned.\n        ')
        
        # Assigning a Call to a Name (line 754):
        
        # Assigning a Call to a Name (line 754):
        
        # Call to object(...): (line 754)
        # Processing the call keyword arguments (line 754)
        kwargs_17492 = {}
        # Getting the type of 'object' (line 754)
        object_17491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 18), 'object', False)
        # Calling object(args, kwargs) (line 754)
        object_call_result_17493 = invoke(stypy.reporting.localization.Localization(__file__, 754, 18), object_17491, *[], **kwargs_17492)
        
        # Assigning a type to the variable 'missing' (line 754)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'missing', object_call_result_17493)
        
        # Assigning a Call to a Name (line 755):
        
        # Assigning a Call to a Name (line 755):
        
        # Call to get_param(...): (line 755)
        # Processing the call arguments (line 755)
        str_17496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 33), 'str', 'charset')
        # Getting the type of 'missing' (line 755)
        missing_17497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 44), 'missing', False)
        # Processing the call keyword arguments (line 755)
        kwargs_17498 = {}
        # Getting the type of 'self' (line 755)
        self_17494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 18), 'self', False)
        # Obtaining the member 'get_param' of a type (line 755)
        get_param_17495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 18), self_17494, 'get_param')
        # Calling get_param(args, kwargs) (line 755)
        get_param_call_result_17499 = invoke(stypy.reporting.localization.Localization(__file__, 755, 18), get_param_17495, *[str_17496, missing_17497], **kwargs_17498)
        
        # Assigning a type to the variable 'charset' (line 755)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'charset', get_param_call_result_17499)
        
        # Getting the type of 'charset' (line 756)
        charset_17500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 11), 'charset')
        # Getting the type of 'missing' (line 756)
        missing_17501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 22), 'missing')
        # Applying the binary operator 'is' (line 756)
        result_is__17502 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 11), 'is', charset_17500, missing_17501)
        
        # Testing if the type of an if condition is none (line 756)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 756, 8), result_is__17502):
            pass
        else:
            
            # Testing the type of an if condition (line 756)
            if_condition_17503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 756, 8), result_is__17502)
            # Assigning a type to the variable 'if_condition_17503' (line 756)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'if_condition_17503', if_condition_17503)
            # SSA begins for if statement (line 756)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'failobj' (line 757)
            failobj_17504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 19), 'failobj')
            # Assigning a type to the variable 'stypy_return_type' (line 757)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 12), 'stypy_return_type', failobj_17504)
            # SSA join for if statement (line 756)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 758)
        # Getting the type of 'tuple' (line 758)
        tuple_17505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 31), 'tuple')
        # Getting the type of 'charset' (line 758)
        charset_17506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 22), 'charset')
        
        (may_be_17507, more_types_in_union_17508) = may_be_subtype(tuple_17505, charset_17506)

        if may_be_17507:

            if more_types_in_union_17508:
                # Runtime conditional SSA (line 758)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'charset' (line 758)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'charset', remove_not_subtype_from_union(charset_17506, tuple))
            
            # Assigning a BoolOp to a Name (line 760):
            
            # Assigning a BoolOp to a Name (line 760):
            
            # Evaluating a boolean operation
            
            # Obtaining the type of the subscript
            int_17509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 31), 'int')
            # Getting the type of 'charset' (line 760)
            charset_17510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 23), 'charset')
            # Obtaining the member '__getitem__' of a type (line 760)
            getitem___17511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 23), charset_17510, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 760)
            subscript_call_result_17512 = invoke(stypy.reporting.localization.Localization(__file__, 760, 23), getitem___17511, int_17509)
            
            str_17513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 37), 'str', 'us-ascii')
            # Applying the binary operator 'or' (line 760)
            result_or_keyword_17514 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 23), 'or', subscript_call_result_17512, str_17513)
            
            # Assigning a type to the variable 'pcharset' (line 760)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'pcharset', result_or_keyword_17514)
            
            
            # SSA begins for try-except statement (line 761)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 765):
            
            # Assigning a Call to a Name (line 765):
            
            # Call to encode(...): (line 765)
            # Processing the call arguments (line 765)
            str_17524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 63), 'str', 'us-ascii')
            # Processing the call keyword arguments (line 765)
            kwargs_17525 = {}
            
            # Call to unicode(...): (line 765)
            # Processing the call arguments (line 765)
            
            # Obtaining the type of the subscript
            int_17516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 42), 'int')
            # Getting the type of 'charset' (line 765)
            charset_17517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 34), 'charset', False)
            # Obtaining the member '__getitem__' of a type (line 765)
            getitem___17518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 34), charset_17517, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 765)
            subscript_call_result_17519 = invoke(stypy.reporting.localization.Localization(__file__, 765, 34), getitem___17518, int_17516)
            
            # Getting the type of 'pcharset' (line 765)
            pcharset_17520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 46), 'pcharset', False)
            # Processing the call keyword arguments (line 765)
            kwargs_17521 = {}
            # Getting the type of 'unicode' (line 765)
            unicode_17515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 26), 'unicode', False)
            # Calling unicode(args, kwargs) (line 765)
            unicode_call_result_17522 = invoke(stypy.reporting.localization.Localization(__file__, 765, 26), unicode_17515, *[subscript_call_result_17519, pcharset_17520], **kwargs_17521)
            
            # Obtaining the member 'encode' of a type (line 765)
            encode_17523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 26), unicode_call_result_17522, 'encode')
            # Calling encode(args, kwargs) (line 765)
            encode_call_result_17526 = invoke(stypy.reporting.localization.Localization(__file__, 765, 26), encode_17523, *[str_17524], **kwargs_17525)
            
            # Assigning a type to the variable 'charset' (line 765)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'charset', encode_call_result_17526)
            # SSA branch for the except part of a try statement (line 761)
            # SSA branch for the except 'Tuple' branch of a try statement (line 761)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Subscript to a Name (line 767):
            
            # Assigning a Subscript to a Name (line 767):
            
            # Obtaining the type of the subscript
            int_17527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 34), 'int')
            # Getting the type of 'charset' (line 767)
            charset_17528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 26), 'charset')
            # Obtaining the member '__getitem__' of a type (line 767)
            getitem___17529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 26), charset_17528, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 767)
            subscript_call_result_17530 = invoke(stypy.reporting.localization.Localization(__file__, 767, 26), getitem___17529, int_17527)
            
            # Assigning a type to the variable 'charset' (line 767)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 16), 'charset', subscript_call_result_17530)
            # SSA join for try-except statement (line 761)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_17508:
                # SSA join for if statement (line 758)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # SSA begins for try-except statement (line 769)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Type idiom detected: calculating its left and rigth part (line 770)
        # Getting the type of 'str' (line 770)
        str_17531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 35), 'str')
        # Getting the type of 'charset' (line 770)
        charset_17532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 26), 'charset')
        
        (may_be_17533, more_types_in_union_17534) = may_be_subtype(str_17531, charset_17532)

        if may_be_17533:

            if more_types_in_union_17534:
                # Runtime conditional SSA (line 770)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'charset' (line 770)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'charset', remove_not_subtype_from_union(charset_17532, str))
            
            # Assigning a Call to a Name (line 771):
            
            # Assigning a Call to a Name (line 771):
            
            # Call to unicode(...): (line 771)
            # Processing the call arguments (line 771)
            # Getting the type of 'charset' (line 771)
            charset_17536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 34), 'charset', False)
            str_17537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 43), 'str', 'us-ascii')
            # Processing the call keyword arguments (line 771)
            kwargs_17538 = {}
            # Getting the type of 'unicode' (line 771)
            unicode_17535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 26), 'unicode', False)
            # Calling unicode(args, kwargs) (line 771)
            unicode_call_result_17539 = invoke(stypy.reporting.localization.Localization(__file__, 771, 26), unicode_17535, *[charset_17536, str_17537], **kwargs_17538)
            
            # Assigning a type to the variable 'charset' (line 771)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 16), 'charset', unicode_call_result_17539)

            if more_types_in_union_17534:
                # SSA join for if statement (line 770)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 772):
        
        # Assigning a Call to a Name (line 772):
        
        # Call to encode(...): (line 772)
        # Processing the call arguments (line 772)
        str_17542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 37), 'str', 'us-ascii')
        # Processing the call keyword arguments (line 772)
        kwargs_17543 = {}
        # Getting the type of 'charset' (line 772)
        charset_17540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 22), 'charset', False)
        # Obtaining the member 'encode' of a type (line 772)
        encode_17541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 22), charset_17540, 'encode')
        # Calling encode(args, kwargs) (line 772)
        encode_call_result_17544 = invoke(stypy.reporting.localization.Localization(__file__, 772, 22), encode_17541, *[str_17542], **kwargs_17543)
        
        # Assigning a type to the variable 'charset' (line 772)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 12), 'charset', encode_call_result_17544)
        # SSA branch for the except part of a try statement (line 769)
        # SSA branch for the except 'UnicodeError' branch of a try statement (line 769)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'failobj' (line 774)
        failobj_17545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 19), 'failobj')
        # Assigning a type to the variable 'stypy_return_type' (line 774)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 12), 'stypy_return_type', failobj_17545)
        # SSA join for try-except statement (line 769)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to lower(...): (line 776)
        # Processing the call keyword arguments (line 776)
        kwargs_17548 = {}
        # Getting the type of 'charset' (line 776)
        charset_17546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 15), 'charset', False)
        # Obtaining the member 'lower' of a type (line 776)
        lower_17547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 15), charset_17546, 'lower')
        # Calling lower(args, kwargs) (line 776)
        lower_call_result_17549 = invoke(stypy.reporting.localization.Localization(__file__, 776, 15), lower_17547, *[], **kwargs_17548)
        
        # Assigning a type to the variable 'stypy_return_type' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'stypy_return_type', lower_call_result_17549)
        
        # ################# End of 'get_content_charset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_content_charset' in the type store
        # Getting the type of 'stypy_return_type' (line 747)
        stypy_return_type_17550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17550)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_content_charset'
        return stypy_return_type_17550


    @norecursion
    def get_charsets(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 778)
        None_17551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 35), 'None')
        defaults = [None_17551]
        # Create a new context for function 'get_charsets'
        module_type_store = module_type_store.open_function_context('get_charsets', 778, 4, False)
        # Assigning a type to the variable 'self' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Message.get_charsets.__dict__.__setitem__('stypy_localization', localization)
        Message.get_charsets.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Message.get_charsets.__dict__.__setitem__('stypy_type_store', module_type_store)
        Message.get_charsets.__dict__.__setitem__('stypy_function_name', 'Message.get_charsets')
        Message.get_charsets.__dict__.__setitem__('stypy_param_names_list', ['failobj'])
        Message.get_charsets.__dict__.__setitem__('stypy_varargs_param_name', None)
        Message.get_charsets.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Message.get_charsets.__dict__.__setitem__('stypy_call_defaults', defaults)
        Message.get_charsets.__dict__.__setitem__('stypy_call_varargs', varargs)
        Message.get_charsets.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Message.get_charsets.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Message.get_charsets', ['failobj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_charsets', localization, ['failobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_charsets(...)' code ##################

        str_17552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, (-1)), 'str', 'Return a list containing the charset(s) used in this message.\n\n        The returned list of items describes the Content-Type headers\'\n        charset parameter for this message and all the subparts in its\n        payload.\n\n        Each item will either be a string (the value of the charset parameter\n        in the Content-Type header of that part) or the value of the\n        \'failobj\' parameter (defaults to None), if the part does not have a\n        main MIME type of "text", or the charset is not defined.\n\n        The list will contain one string for each part of the message, plus\n        one for the container message (i.e. self), so that a non-multipart\n        message will still return a list of length 1.\n        ')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to walk(...): (line 794)
        # Processing the call keyword arguments (line 794)
        kwargs_17560 = {}
        # Getting the type of 'self' (line 794)
        self_17558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 62), 'self', False)
        # Obtaining the member 'walk' of a type (line 794)
        walk_17559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 62), self_17558, 'walk')
        # Calling walk(args, kwargs) (line 794)
        walk_call_result_17561 = invoke(stypy.reporting.localization.Localization(__file__, 794, 62), walk_17559, *[], **kwargs_17560)
        
        comprehension_17562 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 794, 16), walk_call_result_17561)
        # Assigning a type to the variable 'part' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 16), 'part', comprehension_17562)
        
        # Call to get_content_charset(...): (line 794)
        # Processing the call arguments (line 794)
        # Getting the type of 'failobj' (line 794)
        failobj_17555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 41), 'failobj', False)
        # Processing the call keyword arguments (line 794)
        kwargs_17556 = {}
        # Getting the type of 'part' (line 794)
        part_17553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 16), 'part', False)
        # Obtaining the member 'get_content_charset' of a type (line 794)
        get_content_charset_17554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 16), part_17553, 'get_content_charset')
        # Calling get_content_charset(args, kwargs) (line 794)
        get_content_charset_call_result_17557 = invoke(stypy.reporting.localization.Localization(__file__, 794, 16), get_content_charset_17554, *[failobj_17555], **kwargs_17556)
        
        list_17563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 794, 16), list_17563, get_content_charset_call_result_17557)
        # Assigning a type to the variable 'stypy_return_type' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'stypy_return_type', list_17563)
        
        # ################# End of 'get_charsets(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_charsets' in the type store
        # Getting the type of 'stypy_return_type' (line 778)
        stypy_return_type_17564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17564)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_charsets'
        return stypy_return_type_17564

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 797, 4))
    
    # 'from email.iterators import walk' statement (line 797)
    update_path_to_current_file_folder('C:/Python27/lib/email/')
    import_17565 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 797, 4), 'email.iterators')

    if (type(import_17565) is not StypyTypeError):

        if (import_17565 != 'pyd_module'):
            __import__(import_17565)
            sys_modules_17566 = sys.modules[import_17565]
            import_from_module(stypy.reporting.localization.Localization(__file__, 797, 4), 'email.iterators', sys_modules_17566.module_type_store, module_type_store, ['walk'])
            nest_module(stypy.reporting.localization.Localization(__file__, 797, 4), __file__, sys_modules_17566, sys_modules_17566.module_type_store, module_type_store)
        else:
            from email.iterators import walk

            import_from_module(stypy.reporting.localization.Localization(__file__, 797, 4), 'email.iterators', None, module_type_store, ['walk'], [walk])

    else:
        # Assigning a type to the variable 'email.iterators' (line 797)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'email.iterators', import_17565)

    remove_current_file_folder_from_path('C:/Python27/lib/email/')
    

# Assigning a type to the variable 'Message' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'Message', Message)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
