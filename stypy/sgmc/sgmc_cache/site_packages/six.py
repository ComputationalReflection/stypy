
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (c) 2010-2017 Benjamin Peterson
2: #
3: # Permission is hereby granted, free of charge, to any person obtaining a copy
4: # of this software and associated documentation files (the "Software"), to deal
5: # in the Software without restriction, including without limitation the rights
6: # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
7: # copies of the Software, and to permit persons to whom the Software is
8: # furnished to do so, subject to the following conditions:
9: #
10: # The above copyright notice and this permission notice shall be included in all
11: # copies or substantial portions of the Software.
12: #
13: # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
14: # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
15: # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
16: # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
17: # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
18: # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
19: # SOFTWARE.
20: 
21: '''Utilities for writing code that runs on Python 2 and 3'''
22: 
23: from __future__ import absolute_import
24: 
25: import functools
26: import itertools
27: import operator
28: import sys
29: import types
30: 
31: __author__ = "Benjamin Peterson <benjamin@python.org>"
32: __version__ = "1.11.0"
33: 
34: 
35: # Useful for very coarse version differentiation.
36: PY2 = sys.version_info[0] == 2
37: PY3 = sys.version_info[0] == 3
38: PY34 = sys.version_info[0:2] >= (3, 4)
39: 
40: if PY3:
41:     string_types = str,
42:     integer_types = int,
43:     class_types = type,
44:     text_type = str
45:     binary_type = bytes
46: 
47:     MAXSIZE = sys.maxsize
48: else:
49:     string_types = basestring,
50:     integer_types = (int, long)
51:     class_types = (type, types.ClassType)
52:     text_type = unicode
53:     binary_type = str
54: 
55:     if sys.platform.startswith("java"):
56:         # Jython always uses 32 bits.
57:         MAXSIZE = int((1 << 31) - 1)
58:     else:
59:         # It's possible to have sizeof(long) != sizeof(Py_ssize_t).
60:         class X(object):
61: 
62:             def __len__(self):
63:                 return 1 << 31
64:         try:
65:             len(X())
66:         except OverflowError:
67:             # 32-bit
68:             MAXSIZE = int((1 << 31) - 1)
69:         else:
70:             # 64-bit
71:             MAXSIZE = int((1 << 63) - 1)
72:         del X
73: 
74: 
75: def _add_doc(func, doc):
76:     '''Add documentation to a function.'''
77:     func.__doc__ = doc
78: 
79: 
80: def _import_module(name):
81:     '''Import module, returning the module after the last dot.'''
82:     __import__(name)
83:     return sys.modules[name]
84: 
85: 
86: class _LazyDescr(object):
87: 
88:     def __init__(self, name):
89:         self.name = name
90: 
91:     def __get__(self, obj, tp):
92:         result = self._resolve()
93:         setattr(obj, self.name, result)  # Invokes __set__.
94:         try:
95:             # This is a bit ugly, but it avoids running this again by
96:             # removing this descriptor.
97:             delattr(obj.__class__, self.name)
98:         except AttributeError:
99:             pass
100:         return result
101: 
102: 
103: class MovedModule(_LazyDescr):
104: 
105:     def __init__(self, name, old, new=None):
106:         super(MovedModule, self).__init__(name)
107:         if PY3:
108:             if new is None:
109:                 new = name
110:             self.mod = new
111:         else:
112:             self.mod = old
113: 
114:     def _resolve(self):
115:         return _import_module(self.mod)
116: 
117:     def __getattr__(self, attr):
118:         _module = self._resolve()
119:         value = getattr(_module, attr)
120:         setattr(self, attr, value)
121:         return value
122: 
123: 
124: class _LazyModule(types.ModuleType):
125: 
126:     def __init__(self, name):
127:         super(_LazyModule, self).__init__(name)
128:         self.__doc__ = self.__class__.__doc__
129: 
130:     def __dir__(self):
131:         attrs = ["__doc__", "__name__"]
132:         attrs += [attr.name for attr in self._moved_attributes]
133:         return attrs
134: 
135:     # Subclasses should override this
136:     _moved_attributes = []
137: 
138: 
139: class MovedAttribute(_LazyDescr):
140: 
141:     def __init__(self, name, old_mod, new_mod, old_attr=None, new_attr=None):
142:         super(MovedAttribute, self).__init__(name)
143:         if PY3:
144:             if new_mod is None:
145:                 new_mod = name
146:             self.mod = new_mod
147:             if new_attr is None:
148:                 if old_attr is None:
149:                     new_attr = name
150:                 else:
151:                     new_attr = old_attr
152:             self.attr = new_attr
153:         else:
154:             self.mod = old_mod
155:             if old_attr is None:
156:                 old_attr = name
157:             self.attr = old_attr
158: 
159:     def _resolve(self):
160:         module = _import_module(self.mod)
161:         return getattr(module, self.attr)
162: 
163: 
164: class _SixMetaPathImporter(object):
165: 
166:     '''
167:     A meta path importer to import six.moves and its submodules.
168: 
169:     This class implements a PEP302 finder and loader. It should be compatible
170:     with Python 2.5 and all existing versions of Python3
171:     '''
172: 
173:     def __init__(self, six_module_name):
174:         self.name = six_module_name
175:         self.known_modules = {}
176: 
177:     def _add_module(self, mod, *fullnames):
178:         for fullname in fullnames:
179:             self.known_modules[self.name + "." + fullname] = mod
180: 
181:     def _get_module(self, fullname):
182:         return self.known_modules[self.name + "." + fullname]
183: 
184:     def find_module(self, fullname, path=None):
185:         if fullname in self.known_modules:
186:             return self
187:         return None
188: 
189:     def __get_module(self, fullname):
190:         try:
191:             return self.known_modules[fullname]
192:         except KeyError:
193:             raise ImportError("This loader does not know module " + fullname)
194: 
195:     def load_module(self, fullname):
196:         try:
197:             # in case of a reload
198:             return sys.modules[fullname]
199:         except KeyError:
200:             pass
201:         mod = self.__get_module(fullname)
202:         if isinstance(mod, MovedModule):
203:             mod = mod._resolve()
204:         else:
205:             mod.__loader__ = self
206:         sys.modules[fullname] = mod
207:         return mod
208: 
209:     def is_package(self, fullname):
210:         '''
211:         Return true, if the named module is a package.
212: 
213:         We need this method to get correct spec objects with
214:         Python 3.4 (see PEP451)
215:         '''
216:         return hasattr(self.__get_module(fullname), "__path__")
217: 
218:     def get_code(self, fullname):
219:         '''Return None
220: 
221:         Required, if is_package is implemented'''
222:         self.__get_module(fullname)  # eventually raises ImportError
223:         return None
224:     get_source = get_code  # same as get_code
225: 
226: _importer = _SixMetaPathImporter(__name__)
227: 
228: 
229: class _MovedItems(_LazyModule):
230: 
231:     '''Lazy loading of moved objects'''
232:     __path__ = []  # mark as package
233: 
234: 
235: _moved_attributes = [
236:     MovedAttribute("cStringIO", "cStringIO", "io", "StringIO"),
237:     MovedAttribute("filter", "itertools", "builtins", "ifilter", "filter"),
238:     MovedAttribute("filterfalse", "itertools", "itertools", "ifilterfalse", "filterfalse"),
239:     MovedAttribute("input", "__builtin__", "builtins", "raw_input", "input"),
240:     MovedAttribute("intern", "__builtin__", "sys"),
241:     MovedAttribute("map", "itertools", "builtins", "imap", "map"),
242:     MovedAttribute("getcwd", "os", "os", "getcwdu", "getcwd"),
243:     MovedAttribute("getcwdb", "os", "os", "getcwd", "getcwdb"),
244:     MovedAttribute("getoutput", "commands", "subprocess"),
245:     MovedAttribute("range", "__builtin__", "builtins", "xrange", "range"),
246:     MovedAttribute("reload_module", "__builtin__", "importlib" if PY34 else "imp", "reload"),
247:     MovedAttribute("reduce", "__builtin__", "functools"),
248:     MovedAttribute("shlex_quote", "pipes", "shlex", "quote"),
249:     MovedAttribute("StringIO", "StringIO", "io"),
250:     MovedAttribute("UserDict", "UserDict", "collections"),
251:     MovedAttribute("UserList", "UserList", "collections"),
252:     MovedAttribute("UserString", "UserString", "collections"),
253:     MovedAttribute("xrange", "__builtin__", "builtins", "xrange", "range"),
254:     MovedAttribute("zip", "itertools", "builtins", "izip", "zip"),
255:     MovedAttribute("zip_longest", "itertools", "itertools", "izip_longest", "zip_longest"),
256:     MovedModule("builtins", "__builtin__"),
257:     MovedModule("configparser", "ConfigParser"),
258:     MovedModule("copyreg", "copy_reg"),
259:     MovedModule("dbm_gnu", "gdbm", "dbm.gnu"),
260:     MovedModule("_dummy_thread", "dummy_thread", "_dummy_thread"),
261:     MovedModule("http_cookiejar", "cookielib", "http.cookiejar"),
262:     MovedModule("http_cookies", "Cookie", "http.cookies"),
263:     MovedModule("html_entities", "htmlentitydefs", "html.entities"),
264:     MovedModule("html_parser", "HTMLParser", "html.parser"),
265:     MovedModule("http_client", "httplib", "http.client"),
266:     MovedModule("email_mime_base", "email.MIMEBase", "email.mime.base"),
267:     MovedModule("email_mime_image", "email.MIMEImage", "email.mime.image"),
268:     MovedModule("email_mime_multipart", "email.MIMEMultipart", "email.mime.multipart"),
269:     MovedModule("email_mime_nonmultipart", "email.MIMENonMultipart", "email.mime.nonmultipart"),
270:     MovedModule("email_mime_text", "email.MIMEText", "email.mime.text"),
271:     MovedModule("BaseHTTPServer", "BaseHTTPServer", "http.server"),
272:     MovedModule("CGIHTTPServer", "CGIHTTPServer", "http.server"),
273:     MovedModule("SimpleHTTPServer", "SimpleHTTPServer", "http.server"),
274:     MovedModule("cPickle", "cPickle", "pickle"),
275:     MovedModule("queue", "Queue"),
276:     MovedModule("reprlib", "repr"),
277:     MovedModule("socketserver", "SocketServer"),
278:     MovedModule("_thread", "thread", "_thread"),
279:     MovedModule("tkinter", "Tkinter"),
280:     MovedModule("tkinter_dialog", "Dialog", "tkinter.dialog"),
281:     MovedModule("tkinter_filedialog", "FileDialog", "tkinter.filedialog"),
282:     MovedModule("tkinter_scrolledtext", "ScrolledText", "tkinter.scrolledtext"),
283:     MovedModule("tkinter_simpledialog", "SimpleDialog", "tkinter.simpledialog"),
284:     MovedModule("tkinter_tix", "Tix", "tkinter.tix"),
285:     MovedModule("tkinter_ttk", "ttk", "tkinter.ttk"),
286:     MovedModule("tkinter_constants", "Tkconstants", "tkinter.constants"),
287:     MovedModule("tkinter_dnd", "Tkdnd", "tkinter.dnd"),
288:     MovedModule("tkinter_colorchooser", "tkColorChooser",
289:                 "tkinter.colorchooser"),
290:     MovedModule("tkinter_commondialog", "tkCommonDialog",
291:                 "tkinter.commondialog"),
292:     MovedModule("tkinter_tkfiledialog", "tkFileDialog", "tkinter.filedialog"),
293:     MovedModule("tkinter_font", "tkFont", "tkinter.font"),
294:     MovedModule("tkinter_messagebox", "tkMessageBox", "tkinter.messagebox"),
295:     MovedModule("tkinter_tksimpledialog", "tkSimpleDialog",
296:                 "tkinter.simpledialog"),
297:     MovedModule("urllib_parse", __name__ + ".moves.urllib_parse", "urllib.parse"),
298:     MovedModule("urllib_error", __name__ + ".moves.urllib_error", "urllib.error"),
299:     MovedModule("urllib", __name__ + ".moves.urllib", __name__ + ".moves.urllib"),
300:     MovedModule("urllib_robotparser", "robotparser", "urllib.robotparser"),
301:     MovedModule("xmlrpc_client", "xmlrpclib", "xmlrpc.client"),
302:     MovedModule("xmlrpc_server", "SimpleXMLRPCServer", "xmlrpc.server"),
303: ]
304: # Add windows specific modules.
305: if sys.platform == "win32":
306:     _moved_attributes += [
307:         MovedModule("winreg", "_winreg"),
308:     ]
309: 
310: for attr in _moved_attributes:
311:     setattr(_MovedItems, attr.name, attr)
312:     if isinstance(attr, MovedModule):
313:         _importer._add_module(attr, "moves." + attr.name)
314: del attr
315: 
316: _MovedItems._moved_attributes = _moved_attributes
317: 
318: moves = _MovedItems(__name__ + ".moves")
319: _importer._add_module(moves, "moves")
320: 
321: 
322: class Module_six_moves_urllib_parse(_LazyModule):
323: 
324:     '''Lazy loading of moved objects in six.moves.urllib_parse'''
325: 
326: 
327: _urllib_parse_moved_attributes = [
328:     MovedAttribute("ParseResult", "urlparse", "urllib.parse"),
329:     MovedAttribute("SplitResult", "urlparse", "urllib.parse"),
330:     MovedAttribute("parse_qs", "urlparse", "urllib.parse"),
331:     MovedAttribute("parse_qsl", "urlparse", "urllib.parse"),
332:     MovedAttribute("urldefrag", "urlparse", "urllib.parse"),
333:     MovedAttribute("urljoin", "urlparse", "urllib.parse"),
334:     MovedAttribute("urlparse", "urlparse", "urllib.parse"),
335:     MovedAttribute("urlsplit", "urlparse", "urllib.parse"),
336:     MovedAttribute("urlunparse", "urlparse", "urllib.parse"),
337:     MovedAttribute("urlunsplit", "urlparse", "urllib.parse"),
338:     MovedAttribute("quote", "urllib", "urllib.parse"),
339:     MovedAttribute("quote_plus", "urllib", "urllib.parse"),
340:     MovedAttribute("unquote", "urllib", "urllib.parse"),
341:     MovedAttribute("unquote_plus", "urllib", "urllib.parse"),
342:     MovedAttribute("unquote_to_bytes", "urllib", "urllib.parse", "unquote", "unquote_to_bytes"),
343:     MovedAttribute("urlencode", "urllib", "urllib.parse"),
344:     MovedAttribute("splitquery", "urllib", "urllib.parse"),
345:     MovedAttribute("splittag", "urllib", "urllib.parse"),
346:     MovedAttribute("splituser", "urllib", "urllib.parse"),
347:     MovedAttribute("splitvalue", "urllib", "urllib.parse"),
348:     MovedAttribute("uses_fragment", "urlparse", "urllib.parse"),
349:     MovedAttribute("uses_netloc", "urlparse", "urllib.parse"),
350:     MovedAttribute("uses_params", "urlparse", "urllib.parse"),
351:     MovedAttribute("uses_query", "urlparse", "urllib.parse"),
352:     MovedAttribute("uses_relative", "urlparse", "urllib.parse"),
353: ]
354: for attr in _urllib_parse_moved_attributes:
355:     setattr(Module_six_moves_urllib_parse, attr.name, attr)
356: del attr
357: 
358: Module_six_moves_urllib_parse._moved_attributes = _urllib_parse_moved_attributes
359: 
360: _importer._add_module(Module_six_moves_urllib_parse(__name__ + ".moves.urllib_parse"),
361:                       "moves.urllib_parse", "moves.urllib.parse")
362: 
363: 
364: class Module_six_moves_urllib_error(_LazyModule):
365: 
366:     '''Lazy loading of moved objects in six.moves.urllib_error'''
367: 
368: 
369: _urllib_error_moved_attributes = [
370:     MovedAttribute("URLError", "urllib2", "urllib.error"),
371:     MovedAttribute("HTTPError", "urllib2", "urllib.error"),
372:     MovedAttribute("ContentTooShortError", "urllib", "urllib.error"),
373: ]
374: for attr in _urllib_error_moved_attributes:
375:     setattr(Module_six_moves_urllib_error, attr.name, attr)
376: del attr
377: 
378: Module_six_moves_urllib_error._moved_attributes = _urllib_error_moved_attributes
379: 
380: _importer._add_module(Module_six_moves_urllib_error(__name__ + ".moves.urllib.error"),
381:                       "moves.urllib_error", "moves.urllib.error")
382: 
383: 
384: class Module_six_moves_urllib_request(_LazyModule):
385: 
386:     '''Lazy loading of moved objects in six.moves.urllib_request'''
387: 
388: 
389: _urllib_request_moved_attributes = [
390:     MovedAttribute("urlopen", "urllib2", "urllib.request"),
391:     MovedAttribute("install_opener", "urllib2", "urllib.request"),
392:     MovedAttribute("build_opener", "urllib2", "urllib.request"),
393:     MovedAttribute("pathname2url", "urllib", "urllib.request"),
394:     MovedAttribute("url2pathname", "urllib", "urllib.request"),
395:     MovedAttribute("getproxies", "urllib", "urllib.request"),
396:     MovedAttribute("Request", "urllib2", "urllib.request"),
397:     MovedAttribute("OpenerDirector", "urllib2", "urllib.request"),
398:     MovedAttribute("HTTPDefaultErrorHandler", "urllib2", "urllib.request"),
399:     MovedAttribute("HTTPRedirectHandler", "urllib2", "urllib.request"),
400:     MovedAttribute("HTTPCookieProcessor", "urllib2", "urllib.request"),
401:     MovedAttribute("ProxyHandler", "urllib2", "urllib.request"),
402:     MovedAttribute("BaseHandler", "urllib2", "urllib.request"),
403:     MovedAttribute("HTTPPasswordMgr", "urllib2", "urllib.request"),
404:     MovedAttribute("HTTPPasswordMgrWithDefaultRealm", "urllib2", "urllib.request"),
405:     MovedAttribute("AbstractBasicAuthHandler", "urllib2", "urllib.request"),
406:     MovedAttribute("HTTPBasicAuthHandler", "urllib2", "urllib.request"),
407:     MovedAttribute("ProxyBasicAuthHandler", "urllib2", "urllib.request"),
408:     MovedAttribute("AbstractDigestAuthHandler", "urllib2", "urllib.request"),
409:     MovedAttribute("HTTPDigestAuthHandler", "urllib2", "urllib.request"),
410:     MovedAttribute("ProxyDigestAuthHandler", "urllib2", "urllib.request"),
411:     MovedAttribute("HTTPHandler", "urllib2", "urllib.request"),
412:     MovedAttribute("HTTPSHandler", "urllib2", "urllib.request"),
413:     MovedAttribute("FileHandler", "urllib2", "urllib.request"),
414:     MovedAttribute("FTPHandler", "urllib2", "urllib.request"),
415:     MovedAttribute("CacheFTPHandler", "urllib2", "urllib.request"),
416:     MovedAttribute("UnknownHandler", "urllib2", "urllib.request"),
417:     MovedAttribute("HTTPErrorProcessor", "urllib2", "urllib.request"),
418:     MovedAttribute("urlretrieve", "urllib", "urllib.request"),
419:     MovedAttribute("urlcleanup", "urllib", "urllib.request"),
420:     MovedAttribute("URLopener", "urllib", "urllib.request"),
421:     MovedAttribute("FancyURLopener", "urllib", "urllib.request"),
422:     MovedAttribute("proxy_bypass", "urllib", "urllib.request"),
423:     MovedAttribute("parse_http_list", "urllib2", "urllib.request"),
424:     MovedAttribute("parse_keqv_list", "urllib2", "urllib.request"),
425: ]
426: for attr in _urllib_request_moved_attributes:
427:     setattr(Module_six_moves_urllib_request, attr.name, attr)
428: del attr
429: 
430: Module_six_moves_urllib_request._moved_attributes = _urllib_request_moved_attributes
431: 
432: _importer._add_module(Module_six_moves_urllib_request(__name__ + ".moves.urllib.request"),
433:                       "moves.urllib_request", "moves.urllib.request")
434: 
435: 
436: class Module_six_moves_urllib_response(_LazyModule):
437: 
438:     '''Lazy loading of moved objects in six.moves.urllib_response'''
439: 
440: 
441: _urllib_response_moved_attributes = [
442:     MovedAttribute("addbase", "urllib", "urllib.response"),
443:     MovedAttribute("addclosehook", "urllib", "urllib.response"),
444:     MovedAttribute("addinfo", "urllib", "urllib.response"),
445:     MovedAttribute("addinfourl", "urllib", "urllib.response"),
446: ]
447: for attr in _urllib_response_moved_attributes:
448:     setattr(Module_six_moves_urllib_response, attr.name, attr)
449: del attr
450: 
451: Module_six_moves_urllib_response._moved_attributes = _urllib_response_moved_attributes
452: 
453: _importer._add_module(Module_six_moves_urllib_response(__name__ + ".moves.urllib.response"),
454:                       "moves.urllib_response", "moves.urllib.response")
455: 
456: 
457: class Module_six_moves_urllib_robotparser(_LazyModule):
458: 
459:     '''Lazy loading of moved objects in six.moves.urllib_robotparser'''
460: 
461: 
462: _urllib_robotparser_moved_attributes = [
463:     MovedAttribute("RobotFileParser", "robotparser", "urllib.robotparser"),
464: ]
465: for attr in _urllib_robotparser_moved_attributes:
466:     setattr(Module_six_moves_urllib_robotparser, attr.name, attr)
467: del attr
468: 
469: Module_six_moves_urllib_robotparser._moved_attributes = _urllib_robotparser_moved_attributes
470: 
471: _importer._add_module(Module_six_moves_urllib_robotparser(__name__ + ".moves.urllib.robotparser"),
472:                       "moves.urllib_robotparser", "moves.urllib.robotparser")
473: 
474: 
475: class Module_six_moves_urllib(types.ModuleType):
476: 
477:     '''Create a six.moves.urllib namespace that resembles the Python 3 namespace'''
478:     __path__ = []  # mark as package
479:     parse = _importer._get_module("moves.urllib_parse")
480:     error = _importer._get_module("moves.urllib_error")
481:     request = _importer._get_module("moves.urllib_request")
482:     response = _importer._get_module("moves.urllib_response")
483:     robotparser = _importer._get_module("moves.urllib_robotparser")
484: 
485:     def __dir__(self):
486:         return ['parse', 'error', 'request', 'response', 'robotparser']
487: 
488: _importer._add_module(Module_six_moves_urllib(__name__ + ".moves.urllib"),
489:                       "moves.urllib")
490: 
491: 
492: def add_move(move):
493:     '''Add an item to six.moves.'''
494:     setattr(_MovedItems, move.name, move)
495: 
496: 
497: def remove_move(name):
498:     '''Remove item from six.moves.'''
499:     try:
500:         delattr(_MovedItems, name)
501:     except AttributeError:
502:         try:
503:             del moves.__dict__[name]
504:         except KeyError:
505:             raise AttributeError("no such move, %r" % (name,))
506: 
507: 
508: if PY3:
509:     _meth_func = "__func__"
510:     _meth_self = "__self__"
511: 
512:     _func_closure = "__closure__"
513:     _func_code = "__code__"
514:     _func_defaults = "__defaults__"
515:     _func_globals = "__globals__"
516: else:
517:     _meth_func = "im_func"
518:     _meth_self = "im_self"
519: 
520:     _func_closure = "func_closure"
521:     _func_code = "func_code"
522:     _func_defaults = "func_defaults"
523:     _func_globals = "func_globals"
524: 
525: 
526: try:
527:     advance_iterator = next
528: except NameError:
529:     def advance_iterator(it):
530:         return it.next()
531: next = advance_iterator
532: 
533: 
534: try:
535:     callable = callable
536: except NameError:
537:     def callable(obj):
538:         return any("__call__" in klass.__dict__ for klass in type(obj).__mro__)
539: 
540: 
541: if PY3:
542:     def get_unbound_function(unbound):
543:         return unbound
544: 
545:     create_bound_method = types.MethodType
546: 
547:     def create_unbound_method(func, cls):
548:         return func
549: 
550:     Iterator = object
551: else:
552:     def get_unbound_function(unbound):
553:         return unbound.im_func
554: 
555:     def create_bound_method(func, obj):
556:         return types.MethodType(func, obj, obj.__class__)
557: 
558:     def create_unbound_method(func, cls):
559:         return types.MethodType(func, None, cls)
560: 
561:     class Iterator(object):
562: 
563:         def next(self):
564:             return type(self).__next__(self)
565: 
566:     callable = callable
567: _add_doc(get_unbound_function,
568:          '''Get the function out of a possibly unbound function''')
569: 
570: 
571: get_method_function = operator.attrgetter(_meth_func)
572: get_method_self = operator.attrgetter(_meth_self)
573: get_function_closure = operator.attrgetter(_func_closure)
574: get_function_code = operator.attrgetter(_func_code)
575: get_function_defaults = operator.attrgetter(_func_defaults)
576: get_function_globals = operator.attrgetter(_func_globals)
577: 
578: 
579: if PY3:
580:     def iterkeys(d, **kw):
581:         return iter(d.keys(**kw))
582: 
583:     def itervalues(d, **kw):
584:         return iter(d.values(**kw))
585: 
586:     def iteritems(d, **kw):
587:         return iter(d.items(**kw))
588: 
589:     def iterlists(d, **kw):
590:         return iter(d.lists(**kw))
591: 
592:     viewkeys = operator.methodcaller("keys")
593: 
594:     viewvalues = operator.methodcaller("values")
595: 
596:     viewitems = operator.methodcaller("items")
597: else:
598:     def iterkeys(d, **kw):
599:         return d.iterkeys(**kw)
600: 
601:     def itervalues(d, **kw):
602:         return d.itervalues(**kw)
603: 
604:     def iteritems(d, **kw):
605:         return d.iteritems(**kw)
606: 
607:     def iterlists(d, **kw):
608:         return d.iterlists(**kw)
609: 
610:     viewkeys = operator.methodcaller("viewkeys")
611: 
612:     viewvalues = operator.methodcaller("viewvalues")
613: 
614:     viewitems = operator.methodcaller("viewitems")
615: 
616: _add_doc(iterkeys, "Return an iterator over the keys of a dictionary.")
617: _add_doc(itervalues, "Return an iterator over the values of a dictionary.")
618: _add_doc(iteritems,
619:          "Return an iterator over the (key, value) pairs of a dictionary.")
620: _add_doc(iterlists,
621:          "Return an iterator over the (key, [values]) pairs of a dictionary.")
622: 
623: 
624: if PY3:
625:     def b(s):
626:         return s.encode("latin-1")
627: 
628:     def u(s):
629:         return s
630:     unichr = chr
631:     import struct
632:     int2byte = struct.Struct(">B").pack
633:     del struct
634:     byte2int = operator.itemgetter(0)
635:     indexbytes = operator.getitem
636:     iterbytes = iter
637:     import io
638:     StringIO = io.StringIO
639:     BytesIO = io.BytesIO
640:     _assertCountEqual = "assertCountEqual"
641:     if sys.version_info[1] <= 1:
642:         _assertRaisesRegex = "assertRaisesRegexp"
643:         _assertRegex = "assertRegexpMatches"
644:     else:
645:         _assertRaisesRegex = "assertRaisesRegex"
646:         _assertRegex = "assertRegex"
647: else:
648:     def b(s):
649:         return s
650:     # Workaround for standalone backslash
651: 
652:     def u(s):
653:         return unicode(s.replace(r'\\', r'\\\\'), "unicode_escape")
654:     unichr = unichr
655:     int2byte = chr
656: 
657:     def byte2int(bs):
658:         return ord(bs[0])
659: 
660:     def indexbytes(buf, i):
661:         return ord(buf[i])
662:     iterbytes = functools.partial(itertools.imap, ord)
663:     import StringIO
664:     StringIO = BytesIO = StringIO.StringIO
665:     _assertCountEqual = "assertItemsEqual"
666:     _assertRaisesRegex = "assertRaisesRegexp"
667:     _assertRegex = "assertRegexpMatches"
668: _add_doc(b, '''Byte literal''')
669: _add_doc(u, '''Text literal''')
670: 
671: 
672: def assertCountEqual(self, *args, **kwargs):
673:     return getattr(self, _assertCountEqual)(*args, **kwargs)
674: 
675: 
676: def assertRaisesRegex(self, *args, **kwargs):
677:     return getattr(self, _assertRaisesRegex)(*args, **kwargs)
678: 
679: 
680: def assertRegex(self, *args, **kwargs):
681:     return getattr(self, _assertRegex)(*args, **kwargs)
682: 
683: 
684: if PY3:
685:     exec_ = getattr(moves.builtins, "exec")
686: 
687:     def reraise(tp, value, tb=None):
688:         try:
689:             if value is None:
690:                 value = tp()
691:             if value.__traceback__ is not tb:
692:                 raise value.with_traceback(tb)
693:             raise value
694:         finally:
695:             value = None
696:             tb = None
697: 
698: else:
699:     def exec_(_code_, _globs_=None, _locs_=None):
700:         '''Execute code in a namespace.'''
701:         if _globs_ is None:
702:             frame = sys._getframe(1)
703:             _globs_ = frame.f_globals
704:             if _locs_ is None:
705:                 _locs_ = frame.f_locals
706:             del frame
707:         elif _locs_ is None:
708:             _locs_ = _globs_
709:         exec('''exec _code_ in _globs_, _locs_''')
710: 
711:     exec_('''def reraise(tp, value, tb=None):
712:     try:
713:         raise tp, value, tb
714:     finally:
715:         tb = None
716: ''')
717: 
718: 
719: if sys.version_info[:2] == (3, 2):
720:     exec_('''def raise_from(value, from_value):
721:     try:
722:         if from_value is None:
723:             raise value
724:         raise value from from_value
725:     finally:
726:         value = None
727: ''')
728: elif sys.version_info[:2] > (3, 2):
729:     exec_('''def raise_from(value, from_value):
730:     try:
731:         raise value from from_value
732:     finally:
733:         value = None
734: ''')
735: else:
736:     def raise_from(value, from_value):
737:         raise value
738: 
739: 
740: print_ = getattr(moves.builtins, "print", None)
741: if print_ is None:
742:     def print_(*args, **kwargs):
743:         '''The new-style print function for Python 2.4 and 2.5.'''
744:         fp = kwargs.pop("file", sys.stdout)
745:         if fp is None:
746:             return
747: 
748:         def write(data):
749:             if not isinstance(data, basestring):
750:                 data = str(data)
751:             # If the file has an encoding, encode unicode with it.
752:             if (isinstance(fp, file) and
753:                     isinstance(data, unicode) and
754:                     fp.encoding is not None):
755:                 errors = getattr(fp, "errors", None)
756:                 if errors is None:
757:                     errors = "strict"
758:                 data = data.encode(fp.encoding, errors)
759:             fp.write(data)
760:         want_unicode = False
761:         sep = kwargs.pop("sep", None)
762:         if sep is not None:
763:             if isinstance(sep, unicode):
764:                 want_unicode = True
765:             elif not isinstance(sep, str):
766:                 raise TypeError("sep must be None or a string")
767:         end = kwargs.pop("end", None)
768:         if end is not None:
769:             if isinstance(end, unicode):
770:                 want_unicode = True
771:             elif not isinstance(end, str):
772:                 raise TypeError("end must be None or a string")
773:         if kwargs:
774:             raise TypeError("invalid keyword arguments to print()")
775:         if not want_unicode:
776:             for arg in args:
777:                 if isinstance(arg, unicode):
778:                     want_unicode = True
779:                     break
780:         if want_unicode:
781:             newline = unicode("\n")
782:             space = unicode(" ")
783:         else:
784:             newline = "\n"
785:             space = " "
786:         if sep is None:
787:             sep = space
788:         if end is None:
789:             end = newline
790:         for i, arg in enumerate(args):
791:             if i:
792:                 write(sep)
793:             write(arg)
794:         write(end)
795: if sys.version_info[:2] < (3, 3):
796:     _print = print_
797: 
798:     def print_(*args, **kwargs):
799:         fp = kwargs.get("file", sys.stdout)
800:         flush = kwargs.pop("flush", False)
801:         _print(*args, **kwargs)
802:         if flush and fp is not None:
803:             fp.flush()
804: 
805: _add_doc(reraise, '''Reraise an exception.''')
806: 
807: if sys.version_info[0:2] < (3, 4):
808:     def wraps(wrapped, assigned=functools.WRAPPER_ASSIGNMENTS,
809:               updated=functools.WRAPPER_UPDATES):
810:         def wrapper(f):
811:             f = functools.wraps(wrapped, assigned, updated)(f)
812:             f.__wrapped__ = wrapped
813:             return f
814:         return wrapper
815: else:
816:     wraps = functools.wraps
817: 
818: 
819: def with_metaclass(meta, *bases):
820:     '''Create a base class with a metaclass.'''
821:     # This requires a bit of explanation: the basic idea is to make a dummy
822:     # metaclass for one level of class instantiation that replaces itself with
823:     # the actual metaclass.
824:     class metaclass(type):
825: 
826:         def __new__(cls, name, this_bases, d):
827:             return meta(name, bases, d)
828: 
829:         @classmethod
830:         def __prepare__(cls, name, this_bases):
831:             return meta.__prepare__(name, bases)
832:     return type.__new__(metaclass, 'temporary_class', (), {})
833: 
834: 
835: def add_metaclass(metaclass):
836:     '''Class decorator for creating a class with a metaclass.'''
837:     def wrapper(cls):
838:         orig_vars = cls.__dict__.copy()
839:         slots = orig_vars.get('__slots__')
840:         if slots is not None:
841:             if isinstance(slots, str):
842:                 slots = [slots]
843:             for slots_var in slots:
844:                 orig_vars.pop(slots_var)
845:         orig_vars.pop('__dict__', None)
846:         orig_vars.pop('__weakref__', None)
847:         return metaclass(cls.__name__, cls.__bases__, orig_vars)
848:     return wrapper
849: 
850: 
851: def python_2_unicode_compatible(klass):
852:     '''
853:     A decorator that defines __unicode__ and __str__ methods under Python 2.
854:     Under Python 3 it does nothing.
855: 
856:     To support Python 2 and 3 with a single code base, define a __str__ method
857:     returning text and apply this decorator to the class.
858:     '''
859:     if PY2:
860:         if '__str__' not in klass.__dict__:
861:             raise ValueError("@python_2_unicode_compatible cannot be applied "
862:                              "to %s because it doesn't define __str__()." %
863:                              klass.__name__)
864:         klass.__unicode__ = klass.__str__
865:         klass.__str__ = lambda self: self.__unicode__().encode('utf-8')
866:     return klass
867: 
868: 
869: # Complete the moves implementation.
870: # This code is at the end of this module to speed up module loading.
871: # Turn this module into a package.
872: __path__ = []  # required for PEP 302 and PEP 451
873: __package__ = __name__  # see PEP 366 @ReservedAssignment
874: if globals().get("__spec__") is not None:
875:     __spec__.submodule_search_locations = []  # PEP 451 @UndefinedVariable
876: # Remove other six meta path importers, since they cause problems. This can
877: # happen if six is removed from sys.modules and then reloaded. (Setuptools does
878: # this for some reason.)
879: if sys.meta_path:
880:     for i, importer in enumerate(sys.meta_path):
881:         # Here's some real nastiness: Another "instance" of the six module might
882:         # be floating around. Therefore, we can't use isinstance() to check for
883:         # the six meta path importer, since the other six instance will have
884:         # inserted an importer with different class.
885:         if (type(importer).__name__ == "_SixMetaPathImporter" and
886:                 importer.name == __name__):
887:             del sys.meta_path[i]
888:             break
889:     del i, importer
890: # Finally, add the importer to the meta path import hook.
891: sys.meta_path.append(_importer)
892: 

"""



# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)


@norecursion
def iteritems(localization, *varargs, **kwargs):
    return None

# Assigning a type to the variable 'iteritems' (line 586)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'iteritems', iteritems)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
