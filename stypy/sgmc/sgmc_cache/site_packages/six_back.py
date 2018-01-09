
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

# ################# Begin of the type inference program ##################

str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 0), 'str', 'Utilities for writing code that runs on Python 2 and 3')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import functools' statement (line 25)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'functools', functools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'import itertools' statement (line 26)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'import operator' statement (line 27)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import sys' statement (line 28)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'import types' statement (line 29)
import types

import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'types', types, module_type_store)


# Assigning a Str to a Name (line 31):
str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 13), 'str', 'Benjamin Peterson <benjamin@python.org>')
# Assigning a type to the variable '__author__' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), '__author__', str_2)

# Assigning a Str to a Name (line 32):
str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'str', '1.11.0')
# Assigning a type to the variable '__version__' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '__version__', str_3)

# Assigning a Compare to a Name (line 36):


# Obtaining the type of the subscript
int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'int')
# Getting the type of 'sys' (line 36)
sys_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 6), 'sys')
# Obtaining the member 'version_info' of a type (line 36)
version_info_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 6), sys_5, 'version_info')
# Obtaining the member '__getitem__' of a type (line 36)
getitem___7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 6), version_info_6, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 36)
subscript_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 36, 6), getitem___7, int_4)

int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 29), 'int')
# Applying the binary operator '==' (line 36)
result_eq_10 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 6), '==', subscript_call_result_8, int_9)

# Assigning a type to the variable 'PY2' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'PY2', result_eq_10)

# Assigning a Compare to a Name (line 37):


# Obtaining the type of the subscript
int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'int')
# Getting the type of 'sys' (line 37)
sys_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 6), 'sys')
# Obtaining the member 'version_info' of a type (line 37)
version_info_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 6), sys_12, 'version_info')
# Obtaining the member '__getitem__' of a type (line 37)
getitem___14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 6), version_info_13, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 37)
subscript_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 37, 6), getitem___14, int_11)

int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'int')
# Applying the binary operator '==' (line 37)
result_eq_17 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 6), '==', subscript_call_result_15, int_16)

# Assigning a type to the variable 'PY3' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'PY3', result_eq_17)

# Assigning a Compare to a Name (line 38):


# Obtaining the type of the subscript
int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'int')
int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'int')
slice_20 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 38, 7), int_18, int_19, None)
# Getting the type of 'sys' (line 38)
sys_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 'sys')
# Obtaining the member 'version_info' of a type (line 38)
version_info_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 7), sys_21, 'version_info')
# Obtaining the member '__getitem__' of a type (line 38)
getitem___23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 7), version_info_22, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 38)
subscript_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 38, 7), getitem___23, slice_20)


# Obtaining an instance of the builtin type 'tuple' (line 38)
tuple_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 38)
# Adding element type (line 38)
int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 33), tuple_25, int_26)
# Adding element type (line 38)
int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 33), tuple_25, int_27)

# Applying the binary operator '>=' (line 38)
result_ge_28 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 7), '>=', subscript_call_result_24, tuple_25)

# Assigning a type to the variable 'PY34' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'PY34', result_ge_28)

# Getting the type of 'PY3' (line 40)
PY3_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 3), 'PY3')
# Testing the type of an if condition (line 40)
if_condition_30 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 0), PY3_29)
# Assigning a type to the variable 'if_condition_30' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'if_condition_30', if_condition_30)
# SSA begins for if statement (line 40)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Tuple to a Name (line 41):

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
# Getting the type of 'str' (line 41)
str_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 19), tuple_31, str_32)

# Assigning a type to the variable 'string_types' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'string_types', tuple_31)

# Assigning a Tuple to a Name (line 42):

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
# Getting the type of 'int' (line 42)
int_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 20), tuple_33, int_34)

# Assigning a type to the variable 'integer_types' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'integer_types', tuple_33)

# Assigning a Tuple to a Name (line 43):

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)
# Getting the type of 'type' (line 43)
type_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 18), tuple_35, type_36)

# Assigning a type to the variable 'class_types' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'class_types', tuple_35)

# Assigning a Name to a Name (line 44):
# Getting the type of 'str' (line 44)
str_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'str')
# Assigning a type to the variable 'text_type' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'text_type', str_37)

# Assigning a Name to a Name (line 45):
# Getting the type of 'bytes' (line 45)
bytes_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'bytes')
# Assigning a type to the variable 'binary_type' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'binary_type', bytes_38)

# Assigning a Attribute to a Name (line 47):
# Getting the type of 'sys' (line 47)
sys_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'sys')
# Obtaining the member 'maxsize' of a type (line 47)
maxsize_40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 14), sys_39, 'maxsize')
# Assigning a type to the variable 'MAXSIZE' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'MAXSIZE', maxsize_40)
# SSA branch for the else part of an if statement (line 40)
module_type_store.open_ssa_branch('else')

# Assigning a Tuple to a Name (line 49):

# Obtaining an instance of the builtin type 'tuple' (line 49)
tuple_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 49)
# Adding element type (line 49)
# Getting the type of 'basestring' (line 49)
basestring_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'basestring')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), tuple_41, basestring_42)

# Assigning a type to the variable 'string_types' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'string_types', tuple_41)

# Assigning a Tuple to a Name (line 50):

# Obtaining an instance of the builtin type 'tuple' (line 50)
tuple_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 50)
# Adding element type (line 50)
# Getting the type of 'int' (line 50)
int_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 21), tuple_43, int_44)
# Adding element type (line 50)
# Getting the type of 'long' (line 50)
long_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 21), tuple_43, long_45)

# Assigning a type to the variable 'integer_types' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'integer_types', tuple_43)

# Assigning a Tuple to a Name (line 51):

# Obtaining an instance of the builtin type 'tuple' (line 51)
tuple_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 51)
# Adding element type (line 51)
# Getting the type of 'type' (line 51)
type_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 19), tuple_46, type_47)
# Adding element type (line 51)
# Getting the type of 'types' (line 51)
types_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'types')
# Obtaining the member 'ClassType' of a type (line 51)
ClassType_49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), types_48, 'ClassType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 19), tuple_46, ClassType_49)

# Assigning a type to the variable 'class_types' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'class_types', tuple_46)

# Assigning a Name to a Name (line 52):
# Getting the type of 'unicode' (line 52)
unicode_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'unicode')
# Assigning a type to the variable 'text_type' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'text_type', unicode_50)

# Assigning a Name to a Name (line 53):
# Getting the type of 'str' (line 53)
str_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 18), 'str')
# Assigning a type to the variable 'binary_type' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'binary_type', str_51)


# Call to startswith(...): (line 55)
# Processing the call arguments (line 55)
str_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 31), 'str', 'java')
# Processing the call keyword arguments (line 55)
kwargs_56 = {}
# Getting the type of 'sys' (line 55)
sys_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 7), 'sys', False)
# Obtaining the member 'platform' of a type (line 55)
platform_53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 7), sys_52, 'platform')
# Obtaining the member 'startswith' of a type (line 55)
startswith_54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 7), platform_53, 'startswith')
# Calling startswith(args, kwargs) (line 55)
startswith_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 55, 7), startswith_54, *[str_55], **kwargs_56)

# Testing the type of an if condition (line 55)
if_condition_58 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 4), startswith_call_result_57)
# Assigning a type to the variable 'if_condition_58' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'if_condition_58', if_condition_58)
# SSA begins for if statement (line 55)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 57):

# Call to int(...): (line 57)
# Processing the call arguments (line 57)
int_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'int')
int_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'int')
# Applying the binary operator '<<' (line 57)
result_lshift_62 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 23), '<<', int_60, int_61)

int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 34), 'int')
# Applying the binary operator '-' (line 57)
result_sub_64 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 22), '-', result_lshift_62, int_63)

# Processing the call keyword arguments (line 57)
kwargs_65 = {}
# Getting the type of 'int' (line 57)
int_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'int', False)
# Calling int(args, kwargs) (line 57)
int_call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 57, 18), int_59, *[result_sub_64], **kwargs_65)

# Assigning a type to the variable 'MAXSIZE' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'MAXSIZE', int_call_result_66)
# SSA branch for the else part of an if statement (line 55)
module_type_store.open_ssa_branch('else')
# Declaration of the 'X' class

class X(object, ):

    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 62, 12, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'self', type_of_self)
        
        # Passed parameters checking function
        X.__len__.__dict__.__setitem__('stypy_localization', localization)
        X.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        X.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        X.__len__.__dict__.__setitem__('stypy_function_name', 'X.__len__')
        X.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        X.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        X.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        X.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        X.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        X.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        X.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'X.__len__', [], None, None, defaults, varargs, kwargs)

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

        int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'int')
        int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 28), 'int')
        # Applying the binary operator '<<' (line 63)
        result_lshift_69 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 23), '<<', int_67, int_68)
        
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'stypy_return_type', result_lshift_69)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_70)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_70


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 60, 8, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'X.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'X' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'X', X)


# SSA begins for try-except statement (line 64)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to len(...): (line 65)
# Processing the call arguments (line 65)

# Call to X(...): (line 65)
# Processing the call keyword arguments (line 65)
kwargs_73 = {}
# Getting the type of 'X' (line 65)
X_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'X', False)
# Calling X(args, kwargs) (line 65)
X_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), X_72, *[], **kwargs_73)

# Processing the call keyword arguments (line 65)
kwargs_75 = {}
# Getting the type of 'len' (line 65)
len_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'len', False)
# Calling len(args, kwargs) (line 65)
len_call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), len_71, *[X_call_result_74], **kwargs_75)

# SSA branch for the except part of a try statement (line 64)
# SSA branch for the except 'OverflowError' branch of a try statement (line 64)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 68):

# Call to int(...): (line 68)
# Processing the call arguments (line 68)
int_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'int')
int_79 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 32), 'int')
# Applying the binary operator '<<' (line 68)
result_lshift_80 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 27), '<<', int_78, int_79)

int_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 38), 'int')
# Applying the binary operator '-' (line 68)
result_sub_82 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 26), '-', result_lshift_80, int_81)

# Processing the call keyword arguments (line 68)
kwargs_83 = {}
# Getting the type of 'int' (line 68)
int_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'int', False)
# Calling int(args, kwargs) (line 68)
int_call_result_84 = invoke(stypy.reporting.localization.Localization(__file__, 68, 22), int_77, *[result_sub_82], **kwargs_83)

# Assigning a type to the variable 'MAXSIZE' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'MAXSIZE', int_call_result_84)
# SSA branch for the else branch of a try statement (line 64)
module_type_store.open_ssa_branch('except else')

# Assigning a Call to a Name (line 71):

# Call to int(...): (line 71)
# Processing the call arguments (line 71)
int_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'int')
int_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 32), 'int')
# Applying the binary operator '<<' (line 71)
result_lshift_88 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 27), '<<', int_86, int_87)

int_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 38), 'int')
# Applying the binary operator '-' (line 71)
result_sub_90 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 26), '-', result_lshift_88, int_89)

# Processing the call keyword arguments (line 71)
kwargs_91 = {}
# Getting the type of 'int' (line 71)
int_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'int', False)
# Calling int(args, kwargs) (line 71)
int_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 71, 22), int_85, *[result_sub_90], **kwargs_91)

# Assigning a type to the variable 'MAXSIZE' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'MAXSIZE', int_call_result_92)
# SSA join for try-except statement (line 64)
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 72, 8), module_type_store, 'X')
# SSA join for if statement (line 55)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 40)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _add_doc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_add_doc'
    module_type_store = module_type_store.open_function_context('_add_doc', 75, 0, False)
    
    # Passed parameters checking function
    _add_doc.stypy_localization = localization
    _add_doc.stypy_type_of_self = None
    _add_doc.stypy_type_store = module_type_store
    _add_doc.stypy_function_name = '_add_doc'
    _add_doc.stypy_param_names_list = ['func', 'doc']
    _add_doc.stypy_varargs_param_name = None
    _add_doc.stypy_kwargs_param_name = None
    _add_doc.stypy_call_defaults = defaults
    _add_doc.stypy_call_varargs = varargs
    _add_doc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_add_doc', ['func', 'doc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_add_doc', localization, ['func', 'doc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_add_doc(...)' code ##################

    str_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'str', 'Add documentation to a function.')
    
    # Assigning a Name to a Attribute (line 77):
    # Getting the type of 'doc' (line 77)
    doc_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'doc')
    # Getting the type of 'func' (line 77)
    func_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'func')
    # Setting the type of the member '__doc__' of a type (line 77)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 4), func_95, '__doc__', doc_94)
    
    # ################# End of '_add_doc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_add_doc' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_96)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_add_doc'
    return stypy_return_type_96

# Assigning a type to the variable '_add_doc' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), '_add_doc', _add_doc)

@norecursion
def _import_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_import_module'
    module_type_store = module_type_store.open_function_context('_import_module', 80, 0, False)
    
    # Passed parameters checking function
    _import_module.stypy_localization = localization
    _import_module.stypy_type_of_self = None
    _import_module.stypy_type_store = module_type_store
    _import_module.stypy_function_name = '_import_module'
    _import_module.stypy_param_names_list = ['name']
    _import_module.stypy_varargs_param_name = None
    _import_module.stypy_kwargs_param_name = None
    _import_module.stypy_call_defaults = defaults
    _import_module.stypy_call_varargs = varargs
    _import_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_import_module', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_import_module', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_import_module(...)' code ##################

    str_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 4), 'str', 'Import module, returning the module after the last dot.')
    
    # Call to __import__(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'name' (line 82)
    name_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'name', False)
    # Processing the call keyword arguments (line 82)
    kwargs_100 = {}
    # Getting the type of '__import__' (line 82)
    import___98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), '__import__', False)
    # Calling __import__(args, kwargs) (line 82)
    import___call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 82, 4), import___98, *[name_99], **kwargs_100)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 83)
    name_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'name')
    # Getting the type of 'sys' (line 83)
    sys_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'sys')
    # Obtaining the member 'modules' of a type (line 83)
    modules_104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 11), sys_103, 'modules')
    # Obtaining the member '__getitem__' of a type (line 83)
    getitem___105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 11), modules_104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 83)
    subscript_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), getitem___105, name_102)
    
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type', subscript_call_result_106)
    
    # ################# End of '_import_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_import_module' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_107)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_import_module'
    return stypy_return_type_107

# Assigning a type to the variable '_import_module' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), '_import_module', _import_module)
# Declaration of the '_LazyDescr' class

class _LazyDescr(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_LazyDescr.__init__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 89):
        # Getting the type of 'name' (line 89)
        name_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'name')
        # Getting the type of 'self' (line 89)
        self_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
        # Setting the type of the member 'name' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_109, 'name', name_108)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __get__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get__'
        module_type_store = module_type_store.open_function_context('__get__', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _LazyDescr.__get__.__dict__.__setitem__('stypy_localization', localization)
        _LazyDescr.__get__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _LazyDescr.__get__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _LazyDescr.__get__.__dict__.__setitem__('stypy_function_name', '_LazyDescr.__get__')
        _LazyDescr.__get__.__dict__.__setitem__('stypy_param_names_list', ['obj', 'tp'])
        _LazyDescr.__get__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _LazyDescr.__get__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _LazyDescr.__get__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _LazyDescr.__get__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _LazyDescr.__get__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _LazyDescr.__get__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_LazyDescr.__get__', ['obj', 'tp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get__', localization, ['obj', 'tp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get__(...)' code ##################

        
        # Assigning a Call to a Name (line 92):
        
        # Call to _resolve(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_112 = {}
        # Getting the type of 'self' (line 92)
        self_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'self', False)
        # Obtaining the member '_resolve' of a type (line 92)
        _resolve_111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), self_110, '_resolve')
        # Calling _resolve(args, kwargs) (line 92)
        _resolve_call_result_113 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), _resolve_111, *[], **kwargs_112)
        
        # Assigning a type to the variable 'result' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'result', _resolve_call_result_113)
        
        # Call to setattr(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'obj' (line 93)
        obj_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'obj', False)
        # Getting the type of 'self' (line 93)
        self_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 21), 'self', False)
        # Obtaining the member 'name' of a type (line 93)
        name_117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 21), self_116, 'name')
        # Getting the type of 'result' (line 93)
        result_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 32), 'result', False)
        # Processing the call keyword arguments (line 93)
        kwargs_119 = {}
        # Getting the type of 'setattr' (line 93)
        setattr_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 93)
        setattr_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), setattr_114, *[obj_115, name_117, result_118], **kwargs_119)
        
        
        
        # SSA begins for try-except statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to delattr(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'obj' (line 97)
        obj_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'obj', False)
        # Obtaining the member '__class__' of a type (line 97)
        class___123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 20), obj_122, '__class__')
        # Getting the type of 'self' (line 97)
        self_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 35), 'self', False)
        # Obtaining the member 'name' of a type (line 97)
        name_125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 35), self_124, 'name')
        # Processing the call keyword arguments (line 97)
        kwargs_126 = {}
        # Getting the type of 'delattr' (line 97)
        delattr_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'delattr', False)
        # Calling delattr(args, kwargs) (line 97)
        delattr_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), delattr_121, *[class___123, name_125], **kwargs_126)
        
        # SSA branch for the except part of a try statement (line 94)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 94)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 100)
        result_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'stypy_return_type', result_128)
        
        # ################# End of '__get__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get__' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get__'
        return stypy_return_type_129


# Assigning a type to the variable '_LazyDescr' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), '_LazyDescr', _LazyDescr)
# Declaration of the 'MovedModule' class
# Getting the type of '_LazyDescr' (line 103)
_LazyDescr_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 18), '_LazyDescr')

class MovedModule(_LazyDescr_130, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 105)
        None_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 38), 'None')
        defaults = [None_131]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MovedModule.__init__', ['name', 'old', 'new'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'old', 'new'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'name' (line 106)
        name_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), 'name', False)
        # Processing the call keyword arguments (line 106)
        kwargs_139 = {}
        
        # Call to super(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'MovedModule' (line 106)
        MovedModule_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 'MovedModule', False)
        # Getting the type of 'self' (line 106)
        self_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'self', False)
        # Processing the call keyword arguments (line 106)
        kwargs_135 = {}
        # Getting the type of 'super' (line 106)
        super_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'super', False)
        # Calling super(args, kwargs) (line 106)
        super_call_result_136 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), super_132, *[MovedModule_133, self_134], **kwargs_135)
        
        # Obtaining the member '__init__' of a type (line 106)
        init___137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), super_call_result_136, '__init__')
        # Calling __init__(args, kwargs) (line 106)
        init___call_result_140 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), init___137, *[name_138], **kwargs_139)
        
        
        # Getting the type of 'PY3' (line 107)
        PY3_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'PY3')
        # Testing the type of an if condition (line 107)
        if_condition_142 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), PY3_141)
        # Assigning a type to the variable 'if_condition_142' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_142', if_condition_142)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 108)
        # Getting the type of 'new' (line 108)
        new_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'new')
        # Getting the type of 'None' (line 108)
        None_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'None')
        
        (may_be_145, more_types_in_union_146) = may_be_none(new_143, None_144)

        if may_be_145:

            if more_types_in_union_146:
                # Runtime conditional SSA (line 108)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 109):
            # Getting the type of 'name' (line 109)
            name_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 22), 'name')
            # Assigning a type to the variable 'new' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'new', name_147)

            if more_types_in_union_146:
                # SSA join for if statement (line 108)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'new' (line 110)
        new_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'new')
        # Getting the type of 'self' (line 110)
        self_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'self')
        # Setting the type of the member 'mod' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), self_149, 'mod', new_148)
        # SSA branch for the else part of an if statement (line 107)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'old' (line 112)
        old_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'old')
        # Getting the type of 'self' (line 112)
        self_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'self')
        # Setting the type of the member 'mod' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), self_151, 'mod', old_150)
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _resolve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_resolve'
        module_type_store = module_type_store.open_function_context('_resolve', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MovedModule._resolve.__dict__.__setitem__('stypy_localization', localization)
        MovedModule._resolve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MovedModule._resolve.__dict__.__setitem__('stypy_type_store', module_type_store)
        MovedModule._resolve.__dict__.__setitem__('stypy_function_name', 'MovedModule._resolve')
        MovedModule._resolve.__dict__.__setitem__('stypy_param_names_list', [])
        MovedModule._resolve.__dict__.__setitem__('stypy_varargs_param_name', None)
        MovedModule._resolve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MovedModule._resolve.__dict__.__setitem__('stypy_call_defaults', defaults)
        MovedModule._resolve.__dict__.__setitem__('stypy_call_varargs', varargs)
        MovedModule._resolve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MovedModule._resolve.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MovedModule._resolve', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_resolve', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_resolve(...)' code ##################

        
        # Call to _import_module(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'self' (line 115)
        self_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 30), 'self', False)
        # Obtaining the member 'mod' of a type (line 115)
        mod_154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 30), self_153, 'mod')
        # Processing the call keyword arguments (line 115)
        kwargs_155 = {}
        # Getting the type of '_import_module' (line 115)
        _import_module_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), '_import_module', False)
        # Calling _import_module(args, kwargs) (line 115)
        _import_module_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), _import_module_152, *[mod_154], **kwargs_155)
        
        # Assigning a type to the variable 'stypy_return_type' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'stypy_return_type', _import_module_call_result_156)
        
        # ################# End of '_resolve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_resolve' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_157)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_resolve'
        return stypy_return_type_157


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MovedModule.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        MovedModule.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MovedModule.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MovedModule.__getattr__.__dict__.__setitem__('stypy_function_name', 'MovedModule.__getattr__')
        MovedModule.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        MovedModule.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MovedModule.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MovedModule.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MovedModule.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MovedModule.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MovedModule.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MovedModule.__getattr__', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        
        # Assigning a Call to a Name (line 118):
        
        # Call to _resolve(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_160 = {}
        # Getting the type of 'self' (line 118)
        self_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'self', False)
        # Obtaining the member '_resolve' of a type (line 118)
        _resolve_159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 18), self_158, '_resolve')
        # Calling _resolve(args, kwargs) (line 118)
        _resolve_call_result_161 = invoke(stypy.reporting.localization.Localization(__file__, 118, 18), _resolve_159, *[], **kwargs_160)
        
        # Assigning a type to the variable '_module' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), '_module', _resolve_call_result_161)
        
        # Assigning a Call to a Name (line 119):
        
        # Call to getattr(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of '_module' (line 119)
        _module_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), '_module', False)
        # Getting the type of 'attr' (line 119)
        attr_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 33), 'attr', False)
        # Processing the call keyword arguments (line 119)
        kwargs_165 = {}
        # Getting the type of 'getattr' (line 119)
        getattr_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'getattr', False)
        # Calling getattr(args, kwargs) (line 119)
        getattr_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), getattr_162, *[_module_163, attr_164], **kwargs_165)
        
        # Assigning a type to the variable 'value' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'value', getattr_call_result_166)
        
        # Call to setattr(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'self', False)
        # Getting the type of 'attr' (line 120)
        attr_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'attr', False)
        # Getting the type of 'value' (line 120)
        value_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'value', False)
        # Processing the call keyword arguments (line 120)
        kwargs_171 = {}
        # Getting the type of 'setattr' (line 120)
        setattr_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 120)
        setattr_call_result_172 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), setattr_167, *[self_168, attr_169, value_170], **kwargs_171)
        
        # Getting the type of 'value' (line 121)
        value_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', value_173)
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_174)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_174


# Assigning a type to the variable 'MovedModule' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'MovedModule', MovedModule)
# Declaration of the '_LazyModule' class
# Getting the type of 'types' (line 124)
types_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'types')
# Obtaining the member 'ModuleType' of a type (line 124)
ModuleType_176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 18), types_175, 'ModuleType')

class _LazyModule(ModuleType_176, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_LazyModule.__init__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'name' (line 127)
        name_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 42), 'name', False)
        # Processing the call keyword arguments (line 127)
        kwargs_184 = {}
        
        # Call to super(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of '_LazyModule' (line 127)
        _LazyModule_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 14), '_LazyModule', False)
        # Getting the type of 'self' (line 127)
        self_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'self', False)
        # Processing the call keyword arguments (line 127)
        kwargs_180 = {}
        # Getting the type of 'super' (line 127)
        super_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'super', False)
        # Calling super(args, kwargs) (line 127)
        super_call_result_181 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), super_177, *[_LazyModule_178, self_179], **kwargs_180)
        
        # Obtaining the member '__init__' of a type (line 127)
        init___182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), super_call_result_181, '__init__')
        # Calling __init__(args, kwargs) (line 127)
        init___call_result_185 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), init___182, *[name_183], **kwargs_184)
        
        
        # Assigning a Attribute to a Attribute (line 128):
        # Getting the type of 'self' (line 128)
        self_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'self')
        # Obtaining the member '__class__' of a type (line 128)
        class___187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 23), self_186, '__class__')
        # Obtaining the member '__doc__' of a type (line 128)
        doc___188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 23), class___187, '__doc__')
        # Getting the type of 'self' (line 128)
        self_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member '__doc__' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_189, '__doc__', doc___188)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __dir__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__dir__'
        module_type_store = module_type_store.open_function_context('__dir__', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _LazyModule.__dir__.__dict__.__setitem__('stypy_localization', localization)
        _LazyModule.__dir__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _LazyModule.__dir__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _LazyModule.__dir__.__dict__.__setitem__('stypy_function_name', '_LazyModule.__dir__')
        _LazyModule.__dir__.__dict__.__setitem__('stypy_param_names_list', [])
        _LazyModule.__dir__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _LazyModule.__dir__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _LazyModule.__dir__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _LazyModule.__dir__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _LazyModule.__dir__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _LazyModule.__dir__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_LazyModule.__dir__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__dir__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__dir__(...)' code ##################

        
        # Assigning a List to a Name (line 131):
        
        # Obtaining an instance of the builtin type 'list' (line 131)
        list_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 131)
        # Adding element type (line 131)
        str_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 17), 'str', '__doc__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 16), list_190, str_191)
        # Adding element type (line 131)
        str_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 28), 'str', '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 16), list_190, str_192)
        
        # Assigning a type to the variable 'attrs' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'attrs', list_190)
        
        # Getting the type of 'attrs' (line 132)
        attrs_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'attrs')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 132)
        self_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 40), 'self')
        # Obtaining the member '_moved_attributes' of a type (line 132)
        _moved_attributes_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 40), self_196, '_moved_attributes')
        comprehension_198 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 18), _moved_attributes_197)
        # Assigning a type to the variable 'attr' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'attr', comprehension_198)
        # Getting the type of 'attr' (line 132)
        attr_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'attr')
        # Obtaining the member 'name' of a type (line 132)
        name_195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 18), attr_194, 'name')
        list_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 18), list_199, name_195)
        # Applying the binary operator '+=' (line 132)
        result_iadd_200 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 8), '+=', attrs_193, list_199)
        # Assigning a type to the variable 'attrs' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'attrs', result_iadd_200)
        
        # Getting the type of 'attrs' (line 133)
        attrs_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'attrs')
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', attrs_201)
        
        # ################# End of '__dir__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__dir__' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_202)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__dir__'
        return stypy_return_type_202


# Assigning a type to the variable '_LazyModule' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), '_LazyModule', _LazyModule)

# Assigning a List to a Name (line 136):

# Obtaining an instance of the builtin type 'list' (line 136)
list_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 24), 'list')
# Adding type elements to the builtin type 'list' instance (line 136)

# Getting the type of '_LazyModule'
_LazyModule_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_LazyModule')
# Setting the type of the member '_moved_attributes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _LazyModule_204, '_moved_attributes', list_203)
# Declaration of the 'MovedAttribute' class
# Getting the type of '_LazyDescr' (line 139)
_LazyDescr_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), '_LazyDescr')

class MovedAttribute(_LazyDescr_205, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 141)
        None_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 56), 'None')
        # Getting the type of 'None' (line 141)
        None_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 71), 'None')
        defaults = [None_206, None_207]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MovedAttribute.__init__', ['name', 'old_mod', 'new_mod', 'old_attr', 'new_attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'old_mod', 'new_mod', 'old_attr', 'new_attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'name' (line 142)
        name_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'name', False)
        # Processing the call keyword arguments (line 142)
        kwargs_215 = {}
        
        # Call to super(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'MovedAttribute' (line 142)
        MovedAttribute_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'MovedAttribute', False)
        # Getting the type of 'self' (line 142)
        self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'self', False)
        # Processing the call keyword arguments (line 142)
        kwargs_211 = {}
        # Getting the type of 'super' (line 142)
        super_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'super', False)
        # Calling super(args, kwargs) (line 142)
        super_call_result_212 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), super_208, *[MovedAttribute_209, self_210], **kwargs_211)
        
        # Obtaining the member '__init__' of a type (line 142)
        init___213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), super_call_result_212, '__init__')
        # Calling __init__(args, kwargs) (line 142)
        init___call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), init___213, *[name_214], **kwargs_215)
        
        
        # Getting the type of 'PY3' (line 143)
        PY3_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'PY3')
        # Testing the type of an if condition (line 143)
        if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 8), PY3_217)
        # Assigning a type to the variable 'if_condition_218' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'if_condition_218', if_condition_218)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 144)
        # Getting the type of 'new_mod' (line 144)
        new_mod_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'new_mod')
        # Getting the type of 'None' (line 144)
        None_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'None')
        
        (may_be_221, more_types_in_union_222) = may_be_none(new_mod_219, None_220)

        if may_be_221:

            if more_types_in_union_222:
                # Runtime conditional SSA (line 144)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 145):
            # Getting the type of 'name' (line 145)
            name_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 26), 'name')
            # Assigning a type to the variable 'new_mod' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'new_mod', name_223)

            if more_types_in_union_222:
                # SSA join for if statement (line 144)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 146):
        # Getting the type of 'new_mod' (line 146)
        new_mod_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'new_mod')
        # Getting the type of 'self' (line 146)
        self_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'self')
        # Setting the type of the member 'mod' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), self_225, 'mod', new_mod_224)
        
        # Type idiom detected: calculating its left and rigth part (line 147)
        # Getting the type of 'new_attr' (line 147)
        new_attr_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'new_attr')
        # Getting the type of 'None' (line 147)
        None_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'None')
        
        (may_be_228, more_types_in_union_229) = may_be_none(new_attr_226, None_227)

        if may_be_228:

            if more_types_in_union_229:
                # Runtime conditional SSA (line 147)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 148)
            # Getting the type of 'old_attr' (line 148)
            old_attr_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'old_attr')
            # Getting the type of 'None' (line 148)
            None_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'None')
            
            (may_be_232, more_types_in_union_233) = may_be_none(old_attr_230, None_231)

            if may_be_232:

                if more_types_in_union_233:
                    # Runtime conditional SSA (line 148)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Name to a Name (line 149):
                # Getting the type of 'name' (line 149)
                name_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 31), 'name')
                # Assigning a type to the variable 'new_attr' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'new_attr', name_234)

                if more_types_in_union_233:
                    # Runtime conditional SSA for else branch (line 148)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_232) or more_types_in_union_233):
                
                # Assigning a Name to a Name (line 151):
                # Getting the type of 'old_attr' (line 151)
                old_attr_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 31), 'old_attr')
                # Assigning a type to the variable 'new_attr' (line 151)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'new_attr', old_attr_235)

                if (may_be_232 and more_types_in_union_233):
                    # SSA join for if statement (line 148)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_229:
                # SSA join for if statement (line 147)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 152):
        # Getting the type of 'new_attr' (line 152)
        new_attr_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'new_attr')
        # Getting the type of 'self' (line 152)
        self_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'self')
        # Setting the type of the member 'attr' of a type (line 152)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), self_237, 'attr', new_attr_236)
        # SSA branch for the else part of an if statement (line 143)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 154):
        # Getting the type of 'old_mod' (line 154)
        old_mod_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'old_mod')
        # Getting the type of 'self' (line 154)
        self_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'self')
        # Setting the type of the member 'mod' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), self_239, 'mod', old_mod_238)
        
        # Type idiom detected: calculating its left and rigth part (line 155)
        # Getting the type of 'old_attr' (line 155)
        old_attr_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'old_attr')
        # Getting the type of 'None' (line 155)
        None_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'None')
        
        (may_be_242, more_types_in_union_243) = may_be_none(old_attr_240, None_241)

        if may_be_242:

            if more_types_in_union_243:
                # Runtime conditional SSA (line 155)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 156):
            # Getting the type of 'name' (line 156)
            name_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'name')
            # Assigning a type to the variable 'old_attr' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'old_attr', name_244)

            if more_types_in_union_243:
                # SSA join for if statement (line 155)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 157):
        # Getting the type of 'old_attr' (line 157)
        old_attr_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'old_attr')
        # Getting the type of 'self' (line 157)
        self_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'self')
        # Setting the type of the member 'attr' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), self_246, 'attr', old_attr_245)
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _resolve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_resolve'
        module_type_store = module_type_store.open_function_context('_resolve', 159, 4, False)
        # Assigning a type to the variable 'self' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MovedAttribute._resolve.__dict__.__setitem__('stypy_localization', localization)
        MovedAttribute._resolve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MovedAttribute._resolve.__dict__.__setitem__('stypy_type_store', module_type_store)
        MovedAttribute._resolve.__dict__.__setitem__('stypy_function_name', 'MovedAttribute._resolve')
        MovedAttribute._resolve.__dict__.__setitem__('stypy_param_names_list', [])
        MovedAttribute._resolve.__dict__.__setitem__('stypy_varargs_param_name', None)
        MovedAttribute._resolve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MovedAttribute._resolve.__dict__.__setitem__('stypy_call_defaults', defaults)
        MovedAttribute._resolve.__dict__.__setitem__('stypy_call_varargs', varargs)
        MovedAttribute._resolve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MovedAttribute._resolve.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MovedAttribute._resolve', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_resolve', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_resolve(...)' code ##################

        
        # Assigning a Call to a Name (line 160):
        
        # Call to _import_module(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'self' (line 160)
        self_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'self', False)
        # Obtaining the member 'mod' of a type (line 160)
        mod_249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 32), self_248, 'mod')
        # Processing the call keyword arguments (line 160)
        kwargs_250 = {}
        # Getting the type of '_import_module' (line 160)
        _import_module_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), '_import_module', False)
        # Calling _import_module(args, kwargs) (line 160)
        _import_module_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 160, 17), _import_module_247, *[mod_249], **kwargs_250)
        
        # Assigning a type to the variable 'module' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'module', _import_module_call_result_251)
        
        # Call to getattr(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'module' (line 161)
        module_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'module', False)
        # Getting the type of 'self' (line 161)
        self_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'self', False)
        # Obtaining the member 'attr' of a type (line 161)
        attr_255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 31), self_254, 'attr')
        # Processing the call keyword arguments (line 161)
        kwargs_256 = {}
        # Getting the type of 'getattr' (line 161)
        getattr_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 161)
        getattr_call_result_257 = invoke(stypy.reporting.localization.Localization(__file__, 161, 15), getattr_252, *[module_253, attr_255], **kwargs_256)
        
        # Assigning a type to the variable 'stypy_return_type' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'stypy_return_type', getattr_call_result_257)
        
        # ################# End of '_resolve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_resolve' in the type store
        # Getting the type of 'stypy_return_type' (line 159)
        stypy_return_type_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_258)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_resolve'
        return stypy_return_type_258


# Assigning a type to the variable 'MovedAttribute' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'MovedAttribute', MovedAttribute)
# Declaration of the '_SixMetaPathImporter' class

class _SixMetaPathImporter(object, ):
    str_259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, (-1)), 'str', '\n    A meta path importer to import six.moves and its submodules.\n\n    This class implements a PEP302 finder and loader. It should be compatible\n    with Python 2.5 and all existing versions of Python3\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 173, 4, False)
        # Assigning a type to the variable 'self' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SixMetaPathImporter.__init__', ['six_module_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['six_module_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'six_module_name' (line 174)
        six_module_name_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'six_module_name')
        # Getting the type of 'self' (line 174)
        self_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member 'name' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_261, 'name', six_module_name_260)
        
        # Assigning a Dict to a Attribute (line 175):
        
        # Obtaining an instance of the builtin type 'dict' (line 175)
        dict_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 29), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 175)
        
        # Getting the type of 'self' (line 175)
        self_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Setting the type of the member 'known_modules' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_263, 'known_modules', dict_262)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _add_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_module'
        module_type_store = module_type_store.open_function_context('_add_module', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_localization', localization)
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_function_name', '_SixMetaPathImporter._add_module')
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_param_names_list', ['mod'])
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_varargs_param_name', 'fullnames')
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SixMetaPathImporter._add_module.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SixMetaPathImporter._add_module', ['mod'], 'fullnames', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_module', localization, ['mod'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_module(...)' code ##################

        
        # Getting the type of 'fullnames' (line 178)
        fullnames_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'fullnames')
        # Testing the type of a for loop iterable (line 178)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 178, 8), fullnames_264)
        # Getting the type of the for loop variable (line 178)
        for_loop_var_265 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 178, 8), fullnames_264)
        # Assigning a type to the variable 'fullname' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'fullname', for_loop_var_265)
        # SSA begins for a for statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 179):
        # Getting the type of 'mod' (line 179)
        mod_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 61), 'mod')
        # Getting the type of 'self' (line 179)
        self_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'self')
        # Obtaining the member 'known_modules' of a type (line 179)
        known_modules_268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), self_267, 'known_modules')
        # Getting the type of 'self' (line 179)
        self_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'self')
        # Obtaining the member 'name' of a type (line 179)
        name_270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 31), self_269, 'name')
        str_271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 43), 'str', '.')
        # Applying the binary operator '+' (line 179)
        result_add_272 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 31), '+', name_270, str_271)
        
        # Getting the type of 'fullname' (line 179)
        fullname_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 49), 'fullname')
        # Applying the binary operator '+' (line 179)
        result_add_274 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 47), '+', result_add_272, fullname_273)
        
        # Storing an element on a container (line 179)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), known_modules_268, (result_add_274, mod_266))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_add_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_module' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_275)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_module'
        return stypy_return_type_275


    @norecursion
    def _get_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_module'
        module_type_store = module_type_store.open_function_context('_get_module', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_localization', localization)
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_function_name', '_SixMetaPathImporter._get_module')
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_param_names_list', ['fullname'])
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SixMetaPathImporter._get_module.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SixMetaPathImporter._get_module', ['fullname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_module', localization, ['fullname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_module(...)' code ##################

        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 182)
        self_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 34), 'self')
        # Obtaining the member 'name' of a type (line 182)
        name_277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 34), self_276, 'name')
        str_278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 46), 'str', '.')
        # Applying the binary operator '+' (line 182)
        result_add_279 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 34), '+', name_277, str_278)
        
        # Getting the type of 'fullname' (line 182)
        fullname_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 52), 'fullname')
        # Applying the binary operator '+' (line 182)
        result_add_281 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 50), '+', result_add_279, fullname_280)
        
        # Getting the type of 'self' (line 182)
        self_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'self')
        # Obtaining the member 'known_modules' of a type (line 182)
        known_modules_283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), self_282, 'known_modules')
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), known_modules_283, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_285 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), getitem___284, result_add_281)
        
        # Assigning a type to the variable 'stypy_return_type' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type', subscript_call_result_285)
        
        # ################# End of '_get_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_module' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_286)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_module'
        return stypy_return_type_286


    @norecursion
    def find_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 184)
        None_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 41), 'None')
        defaults = [None_287]
        # Create a new context for function 'find_module'
        module_type_store = module_type_store.open_function_context('find_module', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_localization', localization)
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_function_name', '_SixMetaPathImporter.find_module')
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_param_names_list', ['fullname', 'path'])
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SixMetaPathImporter.find_module.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SixMetaPathImporter.find_module', ['fullname', 'path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_module', localization, ['fullname', 'path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_module(...)' code ##################

        
        
        # Getting the type of 'fullname' (line 185)
        fullname_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'fullname')
        # Getting the type of 'self' (line 185)
        self_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'self')
        # Obtaining the member 'known_modules' of a type (line 185)
        known_modules_290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 23), self_289, 'known_modules')
        # Applying the binary operator 'in' (line 185)
        result_contains_291 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 11), 'in', fullname_288, known_modules_290)
        
        # Testing the type of an if condition (line 185)
        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 8), result_contains_291)
        # Assigning a type to the variable 'if_condition_292' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'if_condition_292', if_condition_292)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 186)
        self_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'stypy_return_type', self_293)
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 187)
        None_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', None_294)
        
        # ################# End of 'find_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_module' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_module'
        return stypy_return_type_295


    @norecursion
    def __get_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_module'
        module_type_store = module_type_store.open_function_context('__get_module', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_localization', localization)
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_function_name', '_SixMetaPathImporter.__get_module')
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_param_names_list', ['fullname'])
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SixMetaPathImporter.__get_module.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SixMetaPathImporter.__get_module', ['fullname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_module', localization, ['fullname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_module(...)' code ##################

        
        
        # SSA begins for try-except statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'fullname' (line 191)
        fullname_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 38), 'fullname')
        # Getting the type of 'self' (line 191)
        self_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), 'self')
        # Obtaining the member 'known_modules' of a type (line 191)
        known_modules_298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 19), self_297, 'known_modules')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 19), known_modules_298, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_300 = invoke(stypy.reporting.localization.Localization(__file__, 191, 19), getitem___299, fullname_296)
        
        # Assigning a type to the variable 'stypy_return_type' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'stypy_return_type', subscript_call_result_300)
        # SSA branch for the except part of a try statement (line 190)
        # SSA branch for the except 'KeyError' branch of a try statement (line 190)
        module_type_store.open_ssa_branch('except')
        
        # Call to ImportError(...): (line 193)
        # Processing the call arguments (line 193)
        str_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 30), 'str', 'This loader does not know module ')
        # Getting the type of 'fullname' (line 193)
        fullname_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 68), 'fullname', False)
        # Applying the binary operator '+' (line 193)
        result_add_304 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 30), '+', str_302, fullname_303)
        
        # Processing the call keyword arguments (line 193)
        kwargs_305 = {}
        # Getting the type of 'ImportError' (line 193)
        ImportError_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'ImportError', False)
        # Calling ImportError(args, kwargs) (line 193)
        ImportError_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 193, 18), ImportError_301, *[result_add_304], **kwargs_305)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 193, 12), ImportError_call_result_306, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 190)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__get_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_module' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_307)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_module'
        return stypy_return_type_307


    @norecursion
    def load_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'load_module'
        module_type_store = module_type_store.open_function_context('load_module', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_localization', localization)
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_function_name', '_SixMetaPathImporter.load_module')
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_param_names_list', ['fullname'])
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SixMetaPathImporter.load_module.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SixMetaPathImporter.load_module', ['fullname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'load_module', localization, ['fullname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'load_module(...)' code ##################

        
        
        # SSA begins for try-except statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'fullname' (line 198)
        fullname_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 31), 'fullname')
        # Getting the type of 'sys' (line 198)
        sys_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'sys')
        # Obtaining the member 'modules' of a type (line 198)
        modules_310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), sys_309, 'modules')
        # Obtaining the member '__getitem__' of a type (line 198)
        getitem___311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), modules_310, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 198)
        subscript_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 198, 19), getitem___311, fullname_308)
        
        # Assigning a type to the variable 'stypy_return_type' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'stypy_return_type', subscript_call_result_312)
        # SSA branch for the except part of a try statement (line 196)
        # SSA branch for the except 'KeyError' branch of a try statement (line 196)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 201):
        
        # Call to __get_module(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'fullname' (line 201)
        fullname_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 32), 'fullname', False)
        # Processing the call keyword arguments (line 201)
        kwargs_316 = {}
        # Getting the type of 'self' (line 201)
        self_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 14), 'self', False)
        # Obtaining the member '__get_module' of a type (line 201)
        get_module_314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 14), self_313, '__get_module')
        # Calling __get_module(args, kwargs) (line 201)
        get_module_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 201, 14), get_module_314, *[fullname_315], **kwargs_316)
        
        # Assigning a type to the variable 'mod' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'mod', get_module_call_result_317)
        
        
        # Call to isinstance(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'mod' (line 202)
        mod_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'mod', False)
        # Getting the type of 'MovedModule' (line 202)
        MovedModule_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'MovedModule', False)
        # Processing the call keyword arguments (line 202)
        kwargs_321 = {}
        # Getting the type of 'isinstance' (line 202)
        isinstance_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 202)
        isinstance_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 202, 11), isinstance_318, *[mod_319, MovedModule_320], **kwargs_321)
        
        # Testing the type of an if condition (line 202)
        if_condition_323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), isinstance_call_result_322)
        # Assigning a type to the variable 'if_condition_323' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_323', if_condition_323)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 203):
        
        # Call to _resolve(...): (line 203)
        # Processing the call keyword arguments (line 203)
        kwargs_326 = {}
        # Getting the type of 'mod' (line 203)
        mod_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'mod', False)
        # Obtaining the member '_resolve' of a type (line 203)
        _resolve_325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 18), mod_324, '_resolve')
        # Calling _resolve(args, kwargs) (line 203)
        _resolve_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 203, 18), _resolve_325, *[], **kwargs_326)
        
        # Assigning a type to the variable 'mod' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'mod', _resolve_call_result_327)
        # SSA branch for the else part of an if statement (line 202)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 205):
        # Getting the type of 'self' (line 205)
        self_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'self')
        # Getting the type of 'mod' (line 205)
        mod_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'mod')
        # Setting the type of the member '__loader__' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), mod_329, '__loader__', self_328)
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 206):
        # Getting the type of 'mod' (line 206)
        mod_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 32), 'mod')
        # Getting the type of 'sys' (line 206)
        sys_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 206)
        modules_332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), sys_331, 'modules')
        # Getting the type of 'fullname' (line 206)
        fullname_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'fullname')
        # Storing an element on a container (line 206)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 8), modules_332, (fullname_333, mod_330))
        # Getting the type of 'mod' (line 207)
        mod_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'mod')
        # Assigning a type to the variable 'stypy_return_type' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', mod_334)
        
        # ################# End of 'load_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'load_module' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'load_module'
        return stypy_return_type_335


    @norecursion
    def is_package(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_package'
        module_type_store = module_type_store.open_function_context('is_package', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_localization', localization)
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_function_name', '_SixMetaPathImporter.is_package')
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_param_names_list', ['fullname'])
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SixMetaPathImporter.is_package.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SixMetaPathImporter.is_package', ['fullname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_package', localization, ['fullname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_package(...)' code ##################

        str_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, (-1)), 'str', '\n        Return true, if the named module is a package.\n\n        We need this method to get correct spec objects with\n        Python 3.4 (see PEP451)\n        ')
        
        # Call to hasattr(...): (line 216)
        # Processing the call arguments (line 216)
        
        # Call to __get_module(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'fullname' (line 216)
        fullname_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 41), 'fullname', False)
        # Processing the call keyword arguments (line 216)
        kwargs_341 = {}
        # Getting the type of 'self' (line 216)
        self_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 23), 'self', False)
        # Obtaining the member '__get_module' of a type (line 216)
        get_module_339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 23), self_338, '__get_module')
        # Calling __get_module(args, kwargs) (line 216)
        get_module_call_result_342 = invoke(stypy.reporting.localization.Localization(__file__, 216, 23), get_module_339, *[fullname_340], **kwargs_341)
        
        str_343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 52), 'str', '__path__')
        # Processing the call keyword arguments (line 216)
        kwargs_344 = {}
        # Getting the type of 'hasattr' (line 216)
        hasattr_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 216)
        hasattr_call_result_345 = invoke(stypy.reporting.localization.Localization(__file__, 216, 15), hasattr_337, *[get_module_call_result_342, str_343], **kwargs_344)
        
        # Assigning a type to the variable 'stypy_return_type' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', hasattr_call_result_345)
        
        # ################# End of 'is_package(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_package' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_346)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_package'
        return stypy_return_type_346


    @norecursion
    def get_code(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_code'
        module_type_store = module_type_store.open_function_context('get_code', 218, 4, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_localization', localization)
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_function_name', '_SixMetaPathImporter.get_code')
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_param_names_list', ['fullname'])
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SixMetaPathImporter.get_code.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SixMetaPathImporter.get_code', ['fullname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_code', localization, ['fullname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_code(...)' code ##################

        str_347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, (-1)), 'str', 'Return None\n\n        Required, if is_package is implemented')
        
        # Call to __get_module(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'fullname' (line 222)
        fullname_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 26), 'fullname', False)
        # Processing the call keyword arguments (line 222)
        kwargs_351 = {}
        # Getting the type of 'self' (line 222)
        self_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self', False)
        # Obtaining the member '__get_module' of a type (line 222)
        get_module_349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_348, '__get_module')
        # Calling __get_module(args, kwargs) (line 222)
        get_module_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), get_module_349, *[fullname_350], **kwargs_351)
        
        # Getting the type of 'None' (line 223)
        None_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', None_353)
        
        # ################# End of 'get_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_code' in the type store
        # Getting the type of 'stypy_return_type' (line 218)
        stypy_return_type_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_code'
        return stypy_return_type_354


# Assigning a type to the variable '_SixMetaPathImporter' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), '_SixMetaPathImporter', _SixMetaPathImporter)

# Assigning a Name to a Name (line 224):
# Getting the type of '_SixMetaPathImporter'
_SixMetaPathImporter_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_SixMetaPathImporter')
# Obtaining the member 'get_code' of a type
get_code_356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _SixMetaPathImporter_355, 'get_code')
# Getting the type of '_SixMetaPathImporter'
_SixMetaPathImporter_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_SixMetaPathImporter')
# Setting the type of the member 'get_source' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _SixMetaPathImporter_357, 'get_source', get_code_356)

# Assigning a Call to a Name (line 226):

# Call to _SixMetaPathImporter(...): (line 226)
# Processing the call arguments (line 226)
# Getting the type of '__name__' (line 226)
name___359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 33), '__name__', False)
# Processing the call keyword arguments (line 226)
kwargs_360 = {}
# Getting the type of '_SixMetaPathImporter' (line 226)
_SixMetaPathImporter_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), '_SixMetaPathImporter', False)
# Calling _SixMetaPathImporter(args, kwargs) (line 226)
_SixMetaPathImporter_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), _SixMetaPathImporter_358, *[name___359], **kwargs_360)

# Assigning a type to the variable '_importer' (line 226)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 0), '_importer', _SixMetaPathImporter_call_result_361)
# Declaration of the '_MovedItems' class
# Getting the type of '_LazyModule' (line 229)
_LazyModule_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 18), '_LazyModule')

class _MovedItems(_LazyModule_362, ):
    str_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 4), 'str', 'Lazy loading of moved objects')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 229, 0, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_MovedItems.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable '_MovedItems' (line 229)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 0), '_MovedItems', _MovedItems)

# Assigning a List to a Name (line 232):

# Obtaining an instance of the builtin type 'list' (line 232)
list_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 232)

# Getting the type of '_MovedItems'
_MovedItems_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_MovedItems')
# Setting the type of the member '__path__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _MovedItems_365, '__path__', list_364)

# Assigning a List to a Name (line 235):

# Obtaining an instance of the builtin type 'list' (line 235)
list_366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 235)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 236)
# Processing the call arguments (line 236)
str_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 19), 'str', 'cStringIO')
str_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 32), 'str', 'cStringIO')
str_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 45), 'str', 'io')
str_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 51), 'str', 'StringIO')
# Processing the call keyword arguments (line 236)
kwargs_372 = {}
# Getting the type of 'MovedAttribute' (line 236)
MovedAttribute_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 236)
MovedAttribute_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 236, 4), MovedAttribute_367, *[str_368, str_369, str_370, str_371], **kwargs_372)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_373)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 237)
# Processing the call arguments (line 237)
str_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 19), 'str', 'filter')
str_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 29), 'str', 'itertools')
str_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 42), 'str', 'builtins')
str_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 54), 'str', 'ifilter')
str_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 65), 'str', 'filter')
# Processing the call keyword arguments (line 237)
kwargs_380 = {}
# Getting the type of 'MovedAttribute' (line 237)
MovedAttribute_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 237)
MovedAttribute_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 237, 4), MovedAttribute_374, *[str_375, str_376, str_377, str_378, str_379], **kwargs_380)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_381)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 238)
# Processing the call arguments (line 238)
str_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 19), 'str', 'filterfalse')
str_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 34), 'str', 'itertools')
str_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 47), 'str', 'itertools')
str_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 60), 'str', 'ifilterfalse')
str_387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 76), 'str', 'filterfalse')
# Processing the call keyword arguments (line 238)
kwargs_388 = {}
# Getting the type of 'MovedAttribute' (line 238)
MovedAttribute_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 238)
MovedAttribute_call_result_389 = invoke(stypy.reporting.localization.Localization(__file__, 238, 4), MovedAttribute_382, *[str_383, str_384, str_385, str_386, str_387], **kwargs_388)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_389)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 239)
# Processing the call arguments (line 239)
str_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 19), 'str', 'input')
str_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 28), 'str', '__builtin__')
str_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 43), 'str', 'builtins')
str_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 55), 'str', 'raw_input')
str_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 68), 'str', 'input')
# Processing the call keyword arguments (line 239)
kwargs_396 = {}
# Getting the type of 'MovedAttribute' (line 239)
MovedAttribute_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 239)
MovedAttribute_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 239, 4), MovedAttribute_390, *[str_391, str_392, str_393, str_394, str_395], **kwargs_396)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_397)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 240)
# Processing the call arguments (line 240)
str_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 19), 'str', 'intern')
str_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 29), 'str', '__builtin__')
str_401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 44), 'str', 'sys')
# Processing the call keyword arguments (line 240)
kwargs_402 = {}
# Getting the type of 'MovedAttribute' (line 240)
MovedAttribute_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 240)
MovedAttribute_call_result_403 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), MovedAttribute_398, *[str_399, str_400, str_401], **kwargs_402)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_403)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 241)
# Processing the call arguments (line 241)
str_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 19), 'str', 'map')
str_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 26), 'str', 'itertools')
str_407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 39), 'str', 'builtins')
str_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 51), 'str', 'imap')
str_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 59), 'str', 'map')
# Processing the call keyword arguments (line 241)
kwargs_410 = {}
# Getting the type of 'MovedAttribute' (line 241)
MovedAttribute_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 241)
MovedAttribute_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), MovedAttribute_404, *[str_405, str_406, str_407, str_408, str_409], **kwargs_410)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_411)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 242)
# Processing the call arguments (line 242)
str_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 19), 'str', 'getcwd')
str_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 29), 'str', 'os')
str_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 35), 'str', 'os')
str_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 41), 'str', 'getcwdu')
str_417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 52), 'str', 'getcwd')
# Processing the call keyword arguments (line 242)
kwargs_418 = {}
# Getting the type of 'MovedAttribute' (line 242)
MovedAttribute_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 242)
MovedAttribute_call_result_419 = invoke(stypy.reporting.localization.Localization(__file__, 242, 4), MovedAttribute_412, *[str_413, str_414, str_415, str_416, str_417], **kwargs_418)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_419)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 243)
# Processing the call arguments (line 243)
str_421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 19), 'str', 'getcwdb')
str_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 30), 'str', 'os')
str_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 36), 'str', 'os')
str_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 42), 'str', 'getcwd')
str_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 52), 'str', 'getcwdb')
# Processing the call keyword arguments (line 243)
kwargs_426 = {}
# Getting the type of 'MovedAttribute' (line 243)
MovedAttribute_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 243)
MovedAttribute_call_result_427 = invoke(stypy.reporting.localization.Localization(__file__, 243, 4), MovedAttribute_420, *[str_421, str_422, str_423, str_424, str_425], **kwargs_426)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_427)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 244)
# Processing the call arguments (line 244)
str_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 19), 'str', 'getoutput')
str_430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 32), 'str', 'commands')
str_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 44), 'str', 'subprocess')
# Processing the call keyword arguments (line 244)
kwargs_432 = {}
# Getting the type of 'MovedAttribute' (line 244)
MovedAttribute_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 244)
MovedAttribute_call_result_433 = invoke(stypy.reporting.localization.Localization(__file__, 244, 4), MovedAttribute_428, *[str_429, str_430, str_431], **kwargs_432)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_433)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 245)
# Processing the call arguments (line 245)
str_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 19), 'str', 'range')
str_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 28), 'str', '__builtin__')
str_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 43), 'str', 'builtins')
str_438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 55), 'str', 'xrange')
str_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 65), 'str', 'range')
# Processing the call keyword arguments (line 245)
kwargs_440 = {}
# Getting the type of 'MovedAttribute' (line 245)
MovedAttribute_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 245)
MovedAttribute_call_result_441 = invoke(stypy.reporting.localization.Localization(__file__, 245, 4), MovedAttribute_434, *[str_435, str_436, str_437, str_438, str_439], **kwargs_440)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_441)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 246)
# Processing the call arguments (line 246)
str_443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 19), 'str', 'reload_module')
str_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 36), 'str', '__builtin__')

# Getting the type of 'PY34' (line 246)
PY34_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 66), 'PY34', False)
# Testing the type of an if expression (line 246)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 51), PY34_445)
# SSA begins for if expression (line 246)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
str_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 51), 'str', 'importlib')
# SSA branch for the else part of an if expression (line 246)
module_type_store.open_ssa_branch('if expression else')
str_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 76), 'str', 'imp')
# SSA join for if expression (line 246)
module_type_store = module_type_store.join_ssa_context()
if_exp_448 = union_type.UnionType.add(str_446, str_447)

str_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 83), 'str', 'reload')
# Processing the call keyword arguments (line 246)
kwargs_450 = {}
# Getting the type of 'MovedAttribute' (line 246)
MovedAttribute_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 246)
MovedAttribute_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 246, 4), MovedAttribute_442, *[str_443, str_444, if_exp_448, str_449], **kwargs_450)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_451)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 247)
# Processing the call arguments (line 247)
str_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 19), 'str', 'reduce')
str_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 29), 'str', '__builtin__')
str_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 44), 'str', 'functools')
# Processing the call keyword arguments (line 247)
kwargs_456 = {}
# Getting the type of 'MovedAttribute' (line 247)
MovedAttribute_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 247)
MovedAttribute_call_result_457 = invoke(stypy.reporting.localization.Localization(__file__, 247, 4), MovedAttribute_452, *[str_453, str_454, str_455], **kwargs_456)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_457)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 248)
# Processing the call arguments (line 248)
str_459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 19), 'str', 'shlex_quote')
str_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 34), 'str', 'pipes')
str_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 43), 'str', 'shlex')
str_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 52), 'str', 'quote')
# Processing the call keyword arguments (line 248)
kwargs_463 = {}
# Getting the type of 'MovedAttribute' (line 248)
MovedAttribute_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 248)
MovedAttribute_call_result_464 = invoke(stypy.reporting.localization.Localization(__file__, 248, 4), MovedAttribute_458, *[str_459, str_460, str_461, str_462], **kwargs_463)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_464)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 249)
# Processing the call arguments (line 249)
str_466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 19), 'str', 'StringIO')
str_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 31), 'str', 'StringIO')
str_468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 43), 'str', 'io')
# Processing the call keyword arguments (line 249)
kwargs_469 = {}
# Getting the type of 'MovedAttribute' (line 249)
MovedAttribute_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 249)
MovedAttribute_call_result_470 = invoke(stypy.reporting.localization.Localization(__file__, 249, 4), MovedAttribute_465, *[str_466, str_467, str_468], **kwargs_469)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_470)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 250)
# Processing the call arguments (line 250)
str_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 19), 'str', 'UserDict')
str_473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 31), 'str', 'UserDict')
str_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 43), 'str', 'collections')
# Processing the call keyword arguments (line 250)
kwargs_475 = {}
# Getting the type of 'MovedAttribute' (line 250)
MovedAttribute_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 250)
MovedAttribute_call_result_476 = invoke(stypy.reporting.localization.Localization(__file__, 250, 4), MovedAttribute_471, *[str_472, str_473, str_474], **kwargs_475)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_476)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 251)
# Processing the call arguments (line 251)
str_478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 19), 'str', 'UserList')
str_479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 31), 'str', 'UserList')
str_480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 43), 'str', 'collections')
# Processing the call keyword arguments (line 251)
kwargs_481 = {}
# Getting the type of 'MovedAttribute' (line 251)
MovedAttribute_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 251)
MovedAttribute_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 251, 4), MovedAttribute_477, *[str_478, str_479, str_480], **kwargs_481)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_482)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 252)
# Processing the call arguments (line 252)
str_484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 19), 'str', 'UserString')
str_485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 33), 'str', 'UserString')
str_486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 47), 'str', 'collections')
# Processing the call keyword arguments (line 252)
kwargs_487 = {}
# Getting the type of 'MovedAttribute' (line 252)
MovedAttribute_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 252)
MovedAttribute_call_result_488 = invoke(stypy.reporting.localization.Localization(__file__, 252, 4), MovedAttribute_483, *[str_484, str_485, str_486], **kwargs_487)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_488)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 253)
# Processing the call arguments (line 253)
str_490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 19), 'str', 'xrange')
str_491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 29), 'str', '__builtin__')
str_492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 44), 'str', 'builtins')
str_493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 56), 'str', 'xrange')
str_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 66), 'str', 'range')
# Processing the call keyword arguments (line 253)
kwargs_495 = {}
# Getting the type of 'MovedAttribute' (line 253)
MovedAttribute_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 253)
MovedAttribute_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 253, 4), MovedAttribute_489, *[str_490, str_491, str_492, str_493, str_494], **kwargs_495)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_496)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 254)
# Processing the call arguments (line 254)
str_498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 19), 'str', 'zip')
str_499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 26), 'str', 'itertools')
str_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 39), 'str', 'builtins')
str_501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 51), 'str', 'izip')
str_502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 59), 'str', 'zip')
# Processing the call keyword arguments (line 254)
kwargs_503 = {}
# Getting the type of 'MovedAttribute' (line 254)
MovedAttribute_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 254)
MovedAttribute_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 254, 4), MovedAttribute_497, *[str_498, str_499, str_500, str_501, str_502], **kwargs_503)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_504)
# Adding element type (line 235)

# Call to MovedAttribute(...): (line 255)
# Processing the call arguments (line 255)
str_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 19), 'str', 'zip_longest')
str_507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 34), 'str', 'itertools')
str_508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 47), 'str', 'itertools')
str_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 60), 'str', 'izip_longest')
str_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 76), 'str', 'zip_longest')
# Processing the call keyword arguments (line 255)
kwargs_511 = {}
# Getting the type of 'MovedAttribute' (line 255)
MovedAttribute_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 255)
MovedAttribute_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 255, 4), MovedAttribute_505, *[str_506, str_507, str_508, str_509, str_510], **kwargs_511)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedAttribute_call_result_512)
# Adding element type (line 235)

# Call to MovedModule(...): (line 256)
# Processing the call arguments (line 256)
str_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 16), 'str', 'builtins')
str_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 28), 'str', '__builtin__')
# Processing the call keyword arguments (line 256)
kwargs_516 = {}
# Getting the type of 'MovedModule' (line 256)
MovedModule_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 256)
MovedModule_call_result_517 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), MovedModule_513, *[str_514, str_515], **kwargs_516)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_517)
# Adding element type (line 235)

# Call to MovedModule(...): (line 257)
# Processing the call arguments (line 257)
str_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 16), 'str', 'configparser')
str_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 32), 'str', 'ConfigParser')
# Processing the call keyword arguments (line 257)
kwargs_521 = {}
# Getting the type of 'MovedModule' (line 257)
MovedModule_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 257)
MovedModule_call_result_522 = invoke(stypy.reporting.localization.Localization(__file__, 257, 4), MovedModule_518, *[str_519, str_520], **kwargs_521)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_522)
# Adding element type (line 235)

# Call to MovedModule(...): (line 258)
# Processing the call arguments (line 258)
str_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 16), 'str', 'copyreg')
str_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 27), 'str', 'copy_reg')
# Processing the call keyword arguments (line 258)
kwargs_526 = {}
# Getting the type of 'MovedModule' (line 258)
MovedModule_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 258)
MovedModule_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 258, 4), MovedModule_523, *[str_524, str_525], **kwargs_526)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_527)
# Adding element type (line 235)

# Call to MovedModule(...): (line 259)
# Processing the call arguments (line 259)
str_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 16), 'str', 'dbm_gnu')
str_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'str', 'gdbm')
str_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 35), 'str', 'dbm.gnu')
# Processing the call keyword arguments (line 259)
kwargs_532 = {}
# Getting the type of 'MovedModule' (line 259)
MovedModule_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 259)
MovedModule_call_result_533 = invoke(stypy.reporting.localization.Localization(__file__, 259, 4), MovedModule_528, *[str_529, str_530, str_531], **kwargs_532)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_533)
# Adding element type (line 235)

# Call to MovedModule(...): (line 260)
# Processing the call arguments (line 260)
str_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 16), 'str', '_dummy_thread')
str_536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 33), 'str', 'dummy_thread')
str_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 49), 'str', '_dummy_thread')
# Processing the call keyword arguments (line 260)
kwargs_538 = {}
# Getting the type of 'MovedModule' (line 260)
MovedModule_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 260)
MovedModule_call_result_539 = invoke(stypy.reporting.localization.Localization(__file__, 260, 4), MovedModule_534, *[str_535, str_536, str_537], **kwargs_538)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_539)
# Adding element type (line 235)

# Call to MovedModule(...): (line 261)
# Processing the call arguments (line 261)
str_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 16), 'str', 'http_cookiejar')
str_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 34), 'str', 'cookielib')
str_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 47), 'str', 'http.cookiejar')
# Processing the call keyword arguments (line 261)
kwargs_544 = {}
# Getting the type of 'MovedModule' (line 261)
MovedModule_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 261)
MovedModule_call_result_545 = invoke(stypy.reporting.localization.Localization(__file__, 261, 4), MovedModule_540, *[str_541, str_542, str_543], **kwargs_544)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_545)
# Adding element type (line 235)

# Call to MovedModule(...): (line 262)
# Processing the call arguments (line 262)
str_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 16), 'str', 'http_cookies')
str_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 32), 'str', 'Cookie')
str_549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 42), 'str', 'http.cookies')
# Processing the call keyword arguments (line 262)
kwargs_550 = {}
# Getting the type of 'MovedModule' (line 262)
MovedModule_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 262)
MovedModule_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 262, 4), MovedModule_546, *[str_547, str_548, str_549], **kwargs_550)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_551)
# Adding element type (line 235)

# Call to MovedModule(...): (line 263)
# Processing the call arguments (line 263)
str_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 16), 'str', 'html_entities')
str_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 33), 'str', 'htmlentitydefs')
str_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 51), 'str', 'html.entities')
# Processing the call keyword arguments (line 263)
kwargs_556 = {}
# Getting the type of 'MovedModule' (line 263)
MovedModule_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 263)
MovedModule_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 263, 4), MovedModule_552, *[str_553, str_554, str_555], **kwargs_556)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_557)
# Adding element type (line 235)

# Call to MovedModule(...): (line 264)
# Processing the call arguments (line 264)
str_559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 16), 'str', 'html_parser')
str_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 31), 'str', 'HTMLParser')
str_561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 45), 'str', 'html.parser')
# Processing the call keyword arguments (line 264)
kwargs_562 = {}
# Getting the type of 'MovedModule' (line 264)
MovedModule_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 264)
MovedModule_call_result_563 = invoke(stypy.reporting.localization.Localization(__file__, 264, 4), MovedModule_558, *[str_559, str_560, str_561], **kwargs_562)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_563)
# Adding element type (line 235)

# Call to MovedModule(...): (line 265)
# Processing the call arguments (line 265)
str_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 16), 'str', 'http_client')
str_566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 31), 'str', 'httplib')
str_567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 42), 'str', 'http.client')
# Processing the call keyword arguments (line 265)
kwargs_568 = {}
# Getting the type of 'MovedModule' (line 265)
MovedModule_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 265)
MovedModule_call_result_569 = invoke(stypy.reporting.localization.Localization(__file__, 265, 4), MovedModule_564, *[str_565, str_566, str_567], **kwargs_568)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_569)
# Adding element type (line 235)

# Call to MovedModule(...): (line 266)
# Processing the call arguments (line 266)
str_571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 16), 'str', 'email_mime_base')
str_572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 35), 'str', 'email.MIMEBase')
str_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 53), 'str', 'email.mime.base')
# Processing the call keyword arguments (line 266)
kwargs_574 = {}
# Getting the type of 'MovedModule' (line 266)
MovedModule_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 266)
MovedModule_call_result_575 = invoke(stypy.reporting.localization.Localization(__file__, 266, 4), MovedModule_570, *[str_571, str_572, str_573], **kwargs_574)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_575)
# Adding element type (line 235)

# Call to MovedModule(...): (line 267)
# Processing the call arguments (line 267)
str_577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 16), 'str', 'email_mime_image')
str_578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 36), 'str', 'email.MIMEImage')
str_579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 55), 'str', 'email.mime.image')
# Processing the call keyword arguments (line 267)
kwargs_580 = {}
# Getting the type of 'MovedModule' (line 267)
MovedModule_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 267)
MovedModule_call_result_581 = invoke(stypy.reporting.localization.Localization(__file__, 267, 4), MovedModule_576, *[str_577, str_578, str_579], **kwargs_580)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_581)
# Adding element type (line 235)

# Call to MovedModule(...): (line 268)
# Processing the call arguments (line 268)
str_583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 16), 'str', 'email_mime_multipart')
str_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 40), 'str', 'email.MIMEMultipart')
str_585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 63), 'str', 'email.mime.multipart')
# Processing the call keyword arguments (line 268)
kwargs_586 = {}
# Getting the type of 'MovedModule' (line 268)
MovedModule_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 268)
MovedModule_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 268, 4), MovedModule_582, *[str_583, str_584, str_585], **kwargs_586)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_587)
# Adding element type (line 235)

# Call to MovedModule(...): (line 269)
# Processing the call arguments (line 269)
str_589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 16), 'str', 'email_mime_nonmultipart')
str_590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 43), 'str', 'email.MIMENonMultipart')
str_591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 69), 'str', 'email.mime.nonmultipart')
# Processing the call keyword arguments (line 269)
kwargs_592 = {}
# Getting the type of 'MovedModule' (line 269)
MovedModule_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 269)
MovedModule_call_result_593 = invoke(stypy.reporting.localization.Localization(__file__, 269, 4), MovedModule_588, *[str_589, str_590, str_591], **kwargs_592)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_593)
# Adding element type (line 235)

# Call to MovedModule(...): (line 270)
# Processing the call arguments (line 270)
str_595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 16), 'str', 'email_mime_text')
str_596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 35), 'str', 'email.MIMEText')
str_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 53), 'str', 'email.mime.text')
# Processing the call keyword arguments (line 270)
kwargs_598 = {}
# Getting the type of 'MovedModule' (line 270)
MovedModule_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 270)
MovedModule_call_result_599 = invoke(stypy.reporting.localization.Localization(__file__, 270, 4), MovedModule_594, *[str_595, str_596, str_597], **kwargs_598)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_599)
# Adding element type (line 235)

# Call to MovedModule(...): (line 271)
# Processing the call arguments (line 271)
str_601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 16), 'str', 'BaseHTTPServer')
str_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 34), 'str', 'BaseHTTPServer')
str_603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 52), 'str', 'http.server')
# Processing the call keyword arguments (line 271)
kwargs_604 = {}
# Getting the type of 'MovedModule' (line 271)
MovedModule_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 271)
MovedModule_call_result_605 = invoke(stypy.reporting.localization.Localization(__file__, 271, 4), MovedModule_600, *[str_601, str_602, str_603], **kwargs_604)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_605)
# Adding element type (line 235)

# Call to MovedModule(...): (line 272)
# Processing the call arguments (line 272)
str_607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 16), 'str', 'CGIHTTPServer')
str_608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 33), 'str', 'CGIHTTPServer')
str_609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 50), 'str', 'http.server')
# Processing the call keyword arguments (line 272)
kwargs_610 = {}
# Getting the type of 'MovedModule' (line 272)
MovedModule_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 272)
MovedModule_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 272, 4), MovedModule_606, *[str_607, str_608, str_609], **kwargs_610)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_611)
# Adding element type (line 235)

# Call to MovedModule(...): (line 273)
# Processing the call arguments (line 273)
str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 16), 'str', 'SimpleHTTPServer')
str_614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 36), 'str', 'SimpleHTTPServer')
str_615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 56), 'str', 'http.server')
# Processing the call keyword arguments (line 273)
kwargs_616 = {}
# Getting the type of 'MovedModule' (line 273)
MovedModule_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 273)
MovedModule_call_result_617 = invoke(stypy.reporting.localization.Localization(__file__, 273, 4), MovedModule_612, *[str_613, str_614, str_615], **kwargs_616)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_617)
# Adding element type (line 235)

# Call to MovedModule(...): (line 274)
# Processing the call arguments (line 274)
str_619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 16), 'str', 'cPickle')
str_620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 27), 'str', 'cPickle')
str_621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 38), 'str', 'pickle')
# Processing the call keyword arguments (line 274)
kwargs_622 = {}
# Getting the type of 'MovedModule' (line 274)
MovedModule_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 274)
MovedModule_call_result_623 = invoke(stypy.reporting.localization.Localization(__file__, 274, 4), MovedModule_618, *[str_619, str_620, str_621], **kwargs_622)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_623)
# Adding element type (line 235)

# Call to MovedModule(...): (line 275)
# Processing the call arguments (line 275)
str_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 16), 'str', 'queue')
str_626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 25), 'str', 'Queue')
# Processing the call keyword arguments (line 275)
kwargs_627 = {}
# Getting the type of 'MovedModule' (line 275)
MovedModule_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 275)
MovedModule_call_result_628 = invoke(stypy.reporting.localization.Localization(__file__, 275, 4), MovedModule_624, *[str_625, str_626], **kwargs_627)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_628)
# Adding element type (line 235)

# Call to MovedModule(...): (line 276)
# Processing the call arguments (line 276)
str_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 16), 'str', 'reprlib')
str_631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 27), 'str', 'repr')
# Processing the call keyword arguments (line 276)
kwargs_632 = {}
# Getting the type of 'MovedModule' (line 276)
MovedModule_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 276)
MovedModule_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 276, 4), MovedModule_629, *[str_630, str_631], **kwargs_632)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_633)
# Adding element type (line 235)

# Call to MovedModule(...): (line 277)
# Processing the call arguments (line 277)
str_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 16), 'str', 'socketserver')
str_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 32), 'str', 'SocketServer')
# Processing the call keyword arguments (line 277)
kwargs_637 = {}
# Getting the type of 'MovedModule' (line 277)
MovedModule_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 277)
MovedModule_call_result_638 = invoke(stypy.reporting.localization.Localization(__file__, 277, 4), MovedModule_634, *[str_635, str_636], **kwargs_637)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_638)
# Adding element type (line 235)

# Call to MovedModule(...): (line 278)
# Processing the call arguments (line 278)
str_640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 16), 'str', '_thread')
str_641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 27), 'str', 'thread')
str_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 37), 'str', '_thread')
# Processing the call keyword arguments (line 278)
kwargs_643 = {}
# Getting the type of 'MovedModule' (line 278)
MovedModule_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 278)
MovedModule_call_result_644 = invoke(stypy.reporting.localization.Localization(__file__, 278, 4), MovedModule_639, *[str_640, str_641, str_642], **kwargs_643)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_644)
# Adding element type (line 235)

# Call to MovedModule(...): (line 279)
# Processing the call arguments (line 279)
str_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 16), 'str', 'tkinter')
str_647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 27), 'str', 'Tkinter')
# Processing the call keyword arguments (line 279)
kwargs_648 = {}
# Getting the type of 'MovedModule' (line 279)
MovedModule_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 279)
MovedModule_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 279, 4), MovedModule_645, *[str_646, str_647], **kwargs_648)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_649)
# Adding element type (line 235)

# Call to MovedModule(...): (line 280)
# Processing the call arguments (line 280)
str_651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 16), 'str', 'tkinter_dialog')
str_652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 34), 'str', 'Dialog')
str_653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 44), 'str', 'tkinter.dialog')
# Processing the call keyword arguments (line 280)
kwargs_654 = {}
# Getting the type of 'MovedModule' (line 280)
MovedModule_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 280)
MovedModule_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 280, 4), MovedModule_650, *[str_651, str_652, str_653], **kwargs_654)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_655)
# Adding element type (line 235)

# Call to MovedModule(...): (line 281)
# Processing the call arguments (line 281)
str_657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 16), 'str', 'tkinter_filedialog')
str_658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 38), 'str', 'FileDialog')
str_659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 52), 'str', 'tkinter.filedialog')
# Processing the call keyword arguments (line 281)
kwargs_660 = {}
# Getting the type of 'MovedModule' (line 281)
MovedModule_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 281)
MovedModule_call_result_661 = invoke(stypy.reporting.localization.Localization(__file__, 281, 4), MovedModule_656, *[str_657, str_658, str_659], **kwargs_660)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_661)
# Adding element type (line 235)

# Call to MovedModule(...): (line 282)
# Processing the call arguments (line 282)
str_663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 16), 'str', 'tkinter_scrolledtext')
str_664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 40), 'str', 'ScrolledText')
str_665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 56), 'str', 'tkinter.scrolledtext')
# Processing the call keyword arguments (line 282)
kwargs_666 = {}
# Getting the type of 'MovedModule' (line 282)
MovedModule_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 282)
MovedModule_call_result_667 = invoke(stypy.reporting.localization.Localization(__file__, 282, 4), MovedModule_662, *[str_663, str_664, str_665], **kwargs_666)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_667)
# Adding element type (line 235)

# Call to MovedModule(...): (line 283)
# Processing the call arguments (line 283)
str_669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 16), 'str', 'tkinter_simpledialog')
str_670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 40), 'str', 'SimpleDialog')
str_671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 56), 'str', 'tkinter.simpledialog')
# Processing the call keyword arguments (line 283)
kwargs_672 = {}
# Getting the type of 'MovedModule' (line 283)
MovedModule_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 283)
MovedModule_call_result_673 = invoke(stypy.reporting.localization.Localization(__file__, 283, 4), MovedModule_668, *[str_669, str_670, str_671], **kwargs_672)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_673)
# Adding element type (line 235)

# Call to MovedModule(...): (line 284)
# Processing the call arguments (line 284)
str_675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 16), 'str', 'tkinter_tix')
str_676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 31), 'str', 'Tix')
str_677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 38), 'str', 'tkinter.tix')
# Processing the call keyword arguments (line 284)
kwargs_678 = {}
# Getting the type of 'MovedModule' (line 284)
MovedModule_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 284)
MovedModule_call_result_679 = invoke(stypy.reporting.localization.Localization(__file__, 284, 4), MovedModule_674, *[str_675, str_676, str_677], **kwargs_678)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_679)
# Adding element type (line 235)

# Call to MovedModule(...): (line 285)
# Processing the call arguments (line 285)
str_681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 16), 'str', 'tkinter_ttk')
str_682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 31), 'str', 'ttk')
str_683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 38), 'str', 'tkinter.ttk')
# Processing the call keyword arguments (line 285)
kwargs_684 = {}
# Getting the type of 'MovedModule' (line 285)
MovedModule_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 285)
MovedModule_call_result_685 = invoke(stypy.reporting.localization.Localization(__file__, 285, 4), MovedModule_680, *[str_681, str_682, str_683], **kwargs_684)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_685)
# Adding element type (line 235)

# Call to MovedModule(...): (line 286)
# Processing the call arguments (line 286)
str_687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'str', 'tkinter_constants')
str_688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 37), 'str', 'Tkconstants')
str_689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 52), 'str', 'tkinter.constants')
# Processing the call keyword arguments (line 286)
kwargs_690 = {}
# Getting the type of 'MovedModule' (line 286)
MovedModule_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 286)
MovedModule_call_result_691 = invoke(stypy.reporting.localization.Localization(__file__, 286, 4), MovedModule_686, *[str_687, str_688, str_689], **kwargs_690)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_691)
# Adding element type (line 235)

# Call to MovedModule(...): (line 287)
# Processing the call arguments (line 287)
str_693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 16), 'str', 'tkinter_dnd')
str_694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 31), 'str', 'Tkdnd')
str_695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 40), 'str', 'tkinter.dnd')
# Processing the call keyword arguments (line 287)
kwargs_696 = {}
# Getting the type of 'MovedModule' (line 287)
MovedModule_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 287)
MovedModule_call_result_697 = invoke(stypy.reporting.localization.Localization(__file__, 287, 4), MovedModule_692, *[str_693, str_694, str_695], **kwargs_696)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_697)
# Adding element type (line 235)

# Call to MovedModule(...): (line 288)
# Processing the call arguments (line 288)
str_699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 16), 'str', 'tkinter_colorchooser')
str_700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 40), 'str', 'tkColorChooser')
str_701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 16), 'str', 'tkinter.colorchooser')
# Processing the call keyword arguments (line 288)
kwargs_702 = {}
# Getting the type of 'MovedModule' (line 288)
MovedModule_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 288)
MovedModule_call_result_703 = invoke(stypy.reporting.localization.Localization(__file__, 288, 4), MovedModule_698, *[str_699, str_700, str_701], **kwargs_702)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_703)
# Adding element type (line 235)

# Call to MovedModule(...): (line 290)
# Processing the call arguments (line 290)
str_705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 16), 'str', 'tkinter_commondialog')
str_706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 40), 'str', 'tkCommonDialog')
str_707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 16), 'str', 'tkinter.commondialog')
# Processing the call keyword arguments (line 290)
kwargs_708 = {}
# Getting the type of 'MovedModule' (line 290)
MovedModule_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 290)
MovedModule_call_result_709 = invoke(stypy.reporting.localization.Localization(__file__, 290, 4), MovedModule_704, *[str_705, str_706, str_707], **kwargs_708)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_709)
# Adding element type (line 235)

# Call to MovedModule(...): (line 292)
# Processing the call arguments (line 292)
str_711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 16), 'str', 'tkinter_tkfiledialog')
str_712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 40), 'str', 'tkFileDialog')
str_713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 56), 'str', 'tkinter.filedialog')
# Processing the call keyword arguments (line 292)
kwargs_714 = {}
# Getting the type of 'MovedModule' (line 292)
MovedModule_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 292)
MovedModule_call_result_715 = invoke(stypy.reporting.localization.Localization(__file__, 292, 4), MovedModule_710, *[str_711, str_712, str_713], **kwargs_714)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_715)
# Adding element type (line 235)

# Call to MovedModule(...): (line 293)
# Processing the call arguments (line 293)
str_717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 16), 'str', 'tkinter_font')
str_718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 32), 'str', 'tkFont')
str_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 42), 'str', 'tkinter.font')
# Processing the call keyword arguments (line 293)
kwargs_720 = {}
# Getting the type of 'MovedModule' (line 293)
MovedModule_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 293)
MovedModule_call_result_721 = invoke(stypy.reporting.localization.Localization(__file__, 293, 4), MovedModule_716, *[str_717, str_718, str_719], **kwargs_720)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_721)
# Adding element type (line 235)

# Call to MovedModule(...): (line 294)
# Processing the call arguments (line 294)
str_723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 16), 'str', 'tkinter_messagebox')
str_724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 38), 'str', 'tkMessageBox')
str_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 54), 'str', 'tkinter.messagebox')
# Processing the call keyword arguments (line 294)
kwargs_726 = {}
# Getting the type of 'MovedModule' (line 294)
MovedModule_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 294)
MovedModule_call_result_727 = invoke(stypy.reporting.localization.Localization(__file__, 294, 4), MovedModule_722, *[str_723, str_724, str_725], **kwargs_726)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_727)
# Adding element type (line 235)

# Call to MovedModule(...): (line 295)
# Processing the call arguments (line 295)
str_729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 16), 'str', 'tkinter_tksimpledialog')
str_730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 42), 'str', 'tkSimpleDialog')
str_731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 16), 'str', 'tkinter.simpledialog')
# Processing the call keyword arguments (line 295)
kwargs_732 = {}
# Getting the type of 'MovedModule' (line 295)
MovedModule_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 295)
MovedModule_call_result_733 = invoke(stypy.reporting.localization.Localization(__file__, 295, 4), MovedModule_728, *[str_729, str_730, str_731], **kwargs_732)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_733)
# Adding element type (line 235)

# Call to MovedModule(...): (line 297)
# Processing the call arguments (line 297)
str_735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 16), 'str', 'urllib_parse')
# Getting the type of '__name__' (line 297)
name___736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 32), '__name__', False)
str_737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 43), 'str', '.moves.urllib_parse')
# Applying the binary operator '+' (line 297)
result_add_738 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 32), '+', name___736, str_737)

str_739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 66), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 297)
kwargs_740 = {}
# Getting the type of 'MovedModule' (line 297)
MovedModule_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 297)
MovedModule_call_result_741 = invoke(stypy.reporting.localization.Localization(__file__, 297, 4), MovedModule_734, *[str_735, result_add_738, str_739], **kwargs_740)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_741)
# Adding element type (line 235)

# Call to MovedModule(...): (line 298)
# Processing the call arguments (line 298)
str_743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 16), 'str', 'urllib_error')
# Getting the type of '__name__' (line 298)
name___744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 32), '__name__', False)
str_745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 43), 'str', '.moves.urllib_error')
# Applying the binary operator '+' (line 298)
result_add_746 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 32), '+', name___744, str_745)

str_747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 66), 'str', 'urllib.error')
# Processing the call keyword arguments (line 298)
kwargs_748 = {}
# Getting the type of 'MovedModule' (line 298)
MovedModule_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 298)
MovedModule_call_result_749 = invoke(stypy.reporting.localization.Localization(__file__, 298, 4), MovedModule_742, *[str_743, result_add_746, str_747], **kwargs_748)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_749)
# Adding element type (line 235)

# Call to MovedModule(...): (line 299)
# Processing the call arguments (line 299)
str_751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 16), 'str', 'urllib')
# Getting the type of '__name__' (line 299)
name___752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), '__name__', False)
str_753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 37), 'str', '.moves.urllib')
# Applying the binary operator '+' (line 299)
result_add_754 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 26), '+', name___752, str_753)

# Getting the type of '__name__' (line 299)
name___755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 54), '__name__', False)
str_756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 65), 'str', '.moves.urllib')
# Applying the binary operator '+' (line 299)
result_add_757 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 54), '+', name___755, str_756)

# Processing the call keyword arguments (line 299)
kwargs_758 = {}
# Getting the type of 'MovedModule' (line 299)
MovedModule_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 299)
MovedModule_call_result_759 = invoke(stypy.reporting.localization.Localization(__file__, 299, 4), MovedModule_750, *[str_751, result_add_754, result_add_757], **kwargs_758)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_759)
# Adding element type (line 235)

# Call to MovedModule(...): (line 300)
# Processing the call arguments (line 300)
str_761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 16), 'str', 'urllib_robotparser')
str_762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 38), 'str', 'robotparser')
str_763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 53), 'str', 'urllib.robotparser')
# Processing the call keyword arguments (line 300)
kwargs_764 = {}
# Getting the type of 'MovedModule' (line 300)
MovedModule_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 300)
MovedModule_call_result_765 = invoke(stypy.reporting.localization.Localization(__file__, 300, 4), MovedModule_760, *[str_761, str_762, str_763], **kwargs_764)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_765)
# Adding element type (line 235)

# Call to MovedModule(...): (line 301)
# Processing the call arguments (line 301)
str_767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 16), 'str', 'xmlrpc_client')
str_768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 33), 'str', 'xmlrpclib')
str_769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 46), 'str', 'xmlrpc.client')
# Processing the call keyword arguments (line 301)
kwargs_770 = {}
# Getting the type of 'MovedModule' (line 301)
MovedModule_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 301)
MovedModule_call_result_771 = invoke(stypy.reporting.localization.Localization(__file__, 301, 4), MovedModule_766, *[str_767, str_768, str_769], **kwargs_770)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_771)
# Adding element type (line 235)

# Call to MovedModule(...): (line 302)
# Processing the call arguments (line 302)
str_773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 16), 'str', 'xmlrpc_server')
str_774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 33), 'str', 'SimpleXMLRPCServer')
str_775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 55), 'str', 'xmlrpc.server')
# Processing the call keyword arguments (line 302)
kwargs_776 = {}
# Getting the type of 'MovedModule' (line 302)
MovedModule_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 302)
MovedModule_call_result_777 = invoke(stypy.reporting.localization.Localization(__file__, 302, 4), MovedModule_772, *[str_773, str_774, str_775], **kwargs_776)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), list_366, MovedModule_call_result_777)

# Assigning a type to the variable '_moved_attributes' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), '_moved_attributes', list_366)


# Getting the type of 'sys' (line 305)
sys_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 3), 'sys')
# Obtaining the member 'platform' of a type (line 305)
platform_779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 3), sys_778, 'platform')
str_780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 19), 'str', 'win32')
# Applying the binary operator '==' (line 305)
result_eq_781 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 3), '==', platform_779, str_780)

# Testing the type of an if condition (line 305)
if_condition_782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 0), result_eq_781)
# Assigning a type to the variable 'if_condition_782' (line 305)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'if_condition_782', if_condition_782)
# SSA begins for if statement (line 305)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Getting the type of '_moved_attributes' (line 306)
_moved_attributes_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), '_moved_attributes')

# Obtaining an instance of the builtin type 'list' (line 306)
list_784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 306)
# Adding element type (line 306)

# Call to MovedModule(...): (line 307)
# Processing the call arguments (line 307)
str_786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 20), 'str', 'winreg')
str_787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 30), 'str', '_winreg')
# Processing the call keyword arguments (line 307)
kwargs_788 = {}
# Getting the type of 'MovedModule' (line 307)
MovedModule_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'MovedModule', False)
# Calling MovedModule(args, kwargs) (line 307)
MovedModule_call_result_789 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), MovedModule_785, *[str_786, str_787], **kwargs_788)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 25), list_784, MovedModule_call_result_789)

# Applying the binary operator '+=' (line 306)
result_iadd_790 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 4), '+=', _moved_attributes_783, list_784)
# Assigning a type to the variable '_moved_attributes' (line 306)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), '_moved_attributes', result_iadd_790)

# SSA join for if statement (line 305)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of '_moved_attributes' (line 310)
_moved_attributes_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), '_moved_attributes')
# Testing the type of a for loop iterable (line 310)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 310, 0), _moved_attributes_791)
# Getting the type of the for loop variable (line 310)
for_loop_var_792 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 310, 0), _moved_attributes_791)
# Assigning a type to the variable 'attr' (line 310)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 0), 'attr', for_loop_var_792)
# SSA begins for a for statement (line 310)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Call to setattr(...): (line 311)
# Processing the call arguments (line 311)
# Getting the type of '_MovedItems' (line 311)
_MovedItems_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), '_MovedItems', False)
# Getting the type of 'attr' (line 311)
attr_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 25), 'attr', False)
# Obtaining the member 'name' of a type (line 311)
name_796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 25), attr_795, 'name')
# Getting the type of 'attr' (line 311)
attr_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 36), 'attr', False)
# Processing the call keyword arguments (line 311)
kwargs_798 = {}
# Getting the type of 'setattr' (line 311)
setattr_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'setattr', False)
# Calling setattr(args, kwargs) (line 311)
setattr_call_result_799 = invoke(stypy.reporting.localization.Localization(__file__, 311, 4), setattr_793, *[_MovedItems_794, name_796, attr_797], **kwargs_798)



# Call to isinstance(...): (line 312)
# Processing the call arguments (line 312)
# Getting the type of 'attr' (line 312)
attr_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 18), 'attr', False)
# Getting the type of 'MovedModule' (line 312)
MovedModule_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), 'MovedModule', False)
# Processing the call keyword arguments (line 312)
kwargs_803 = {}
# Getting the type of 'isinstance' (line 312)
isinstance_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 7), 'isinstance', False)
# Calling isinstance(args, kwargs) (line 312)
isinstance_call_result_804 = invoke(stypy.reporting.localization.Localization(__file__, 312, 7), isinstance_800, *[attr_801, MovedModule_802], **kwargs_803)

# Testing the type of an if condition (line 312)
if_condition_805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 4), isinstance_call_result_804)
# Assigning a type to the variable 'if_condition_805' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'if_condition_805', if_condition_805)
# SSA begins for if statement (line 312)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to _add_module(...): (line 313)
# Processing the call arguments (line 313)
# Getting the type of 'attr' (line 313)
attr_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 30), 'attr', False)
str_809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 36), 'str', 'moves.')
# Getting the type of 'attr' (line 313)
attr_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 47), 'attr', False)
# Obtaining the member 'name' of a type (line 313)
name_811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 47), attr_810, 'name')
# Applying the binary operator '+' (line 313)
result_add_812 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 36), '+', str_809, name_811)

# Processing the call keyword arguments (line 313)
kwargs_813 = {}
# Getting the type of '_importer' (line 313)
_importer_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), '_importer', False)
# Obtaining the member '_add_module' of a type (line 313)
_add_module_807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), _importer_806, '_add_module')
# Calling _add_module(args, kwargs) (line 313)
_add_module_call_result_814 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), _add_module_807, *[attr_808, result_add_812], **kwargs_813)

# SSA join for if statement (line 312)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 314, 0), module_type_store, 'attr')

# Assigning a Name to a Attribute (line 316):
# Getting the type of '_moved_attributes' (line 316)
_moved_attributes_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 32), '_moved_attributes')
# Getting the type of '_MovedItems' (line 316)
_MovedItems_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 0), '_MovedItems')
# Setting the type of the member '_moved_attributes' of a type (line 316)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 0), _MovedItems_816, '_moved_attributes', _moved_attributes_815)

# Assigning a Call to a Name (line 318):

# Call to _MovedItems(...): (line 318)
# Processing the call arguments (line 318)
# Getting the type of '__name__' (line 318)
name___818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), '__name__', False)
str_819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 31), 'str', '.moves')
# Applying the binary operator '+' (line 318)
result_add_820 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 20), '+', name___818, str_819)

# Processing the call keyword arguments (line 318)
kwargs_821 = {}
# Getting the type of '_MovedItems' (line 318)
_MovedItems_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), '_MovedItems', False)
# Calling _MovedItems(args, kwargs) (line 318)
_MovedItems_call_result_822 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), _MovedItems_817, *[result_add_820], **kwargs_821)

# Assigning a type to the variable 'moves' (line 318)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'moves', _MovedItems_call_result_822)

# Call to _add_module(...): (line 319)
# Processing the call arguments (line 319)
# Getting the type of 'moves' (line 319)
moves_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 22), 'moves', False)
str_826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 29), 'str', 'moves')
# Processing the call keyword arguments (line 319)
kwargs_827 = {}
# Getting the type of '_importer' (line 319)
_importer_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 0), '_importer', False)
# Obtaining the member '_add_module' of a type (line 319)
_add_module_824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 0), _importer_823, '_add_module')
# Calling _add_module(args, kwargs) (line 319)
_add_module_call_result_828 = invoke(stypy.reporting.localization.Localization(__file__, 319, 0), _add_module_824, *[moves_825, str_826], **kwargs_827)

# Declaration of the 'Module_six_moves_urllib_parse' class
# Getting the type of '_LazyModule' (line 322)
_LazyModule_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 36), '_LazyModule')

class Module_six_moves_urllib_parse(_LazyModule_829, ):
    str_830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 4), 'str', 'Lazy loading of moved objects in six.moves.urllib_parse')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 322, 0, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Module_six_moves_urllib_parse.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Module_six_moves_urllib_parse' (line 322)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 0), 'Module_six_moves_urllib_parse', Module_six_moves_urllib_parse)

# Assigning a List to a Name (line 327):

# Obtaining an instance of the builtin type 'list' (line 327)
list_831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 327)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 328)
# Processing the call arguments (line 328)
str_833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 19), 'str', 'ParseResult')
str_834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 34), 'str', 'urlparse')
str_835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 46), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 328)
kwargs_836 = {}
# Getting the type of 'MovedAttribute' (line 328)
MovedAttribute_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 328)
MovedAttribute_call_result_837 = invoke(stypy.reporting.localization.Localization(__file__, 328, 4), MovedAttribute_832, *[str_833, str_834, str_835], **kwargs_836)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_837)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 329)
# Processing the call arguments (line 329)
str_839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 19), 'str', 'SplitResult')
str_840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 34), 'str', 'urlparse')
str_841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 46), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 329)
kwargs_842 = {}
# Getting the type of 'MovedAttribute' (line 329)
MovedAttribute_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 329)
MovedAttribute_call_result_843 = invoke(stypy.reporting.localization.Localization(__file__, 329, 4), MovedAttribute_838, *[str_839, str_840, str_841], **kwargs_842)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_843)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 330)
# Processing the call arguments (line 330)
str_845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 19), 'str', 'parse_qs')
str_846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 31), 'str', 'urlparse')
str_847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 43), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 330)
kwargs_848 = {}
# Getting the type of 'MovedAttribute' (line 330)
MovedAttribute_844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 330)
MovedAttribute_call_result_849 = invoke(stypy.reporting.localization.Localization(__file__, 330, 4), MovedAttribute_844, *[str_845, str_846, str_847], **kwargs_848)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_849)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 331)
# Processing the call arguments (line 331)
str_851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 19), 'str', 'parse_qsl')
str_852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 32), 'str', 'urlparse')
str_853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 44), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 331)
kwargs_854 = {}
# Getting the type of 'MovedAttribute' (line 331)
MovedAttribute_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 331)
MovedAttribute_call_result_855 = invoke(stypy.reporting.localization.Localization(__file__, 331, 4), MovedAttribute_850, *[str_851, str_852, str_853], **kwargs_854)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_855)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 332)
# Processing the call arguments (line 332)
str_857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 19), 'str', 'urldefrag')
str_858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 32), 'str', 'urlparse')
str_859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 44), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 332)
kwargs_860 = {}
# Getting the type of 'MovedAttribute' (line 332)
MovedAttribute_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 332)
MovedAttribute_call_result_861 = invoke(stypy.reporting.localization.Localization(__file__, 332, 4), MovedAttribute_856, *[str_857, str_858, str_859], **kwargs_860)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_861)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 333)
# Processing the call arguments (line 333)
str_863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 19), 'str', 'urljoin')
str_864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 30), 'str', 'urlparse')
str_865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 42), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 333)
kwargs_866 = {}
# Getting the type of 'MovedAttribute' (line 333)
MovedAttribute_862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 333)
MovedAttribute_call_result_867 = invoke(stypy.reporting.localization.Localization(__file__, 333, 4), MovedAttribute_862, *[str_863, str_864, str_865], **kwargs_866)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_867)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 334)
# Processing the call arguments (line 334)
str_869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 19), 'str', 'urlparse')
str_870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 31), 'str', 'urlparse')
str_871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 43), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 334)
kwargs_872 = {}
# Getting the type of 'MovedAttribute' (line 334)
MovedAttribute_868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 334)
MovedAttribute_call_result_873 = invoke(stypy.reporting.localization.Localization(__file__, 334, 4), MovedAttribute_868, *[str_869, str_870, str_871], **kwargs_872)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_873)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 335)
# Processing the call arguments (line 335)
str_875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 19), 'str', 'urlsplit')
str_876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 31), 'str', 'urlparse')
str_877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 43), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 335)
kwargs_878 = {}
# Getting the type of 'MovedAttribute' (line 335)
MovedAttribute_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 335)
MovedAttribute_call_result_879 = invoke(stypy.reporting.localization.Localization(__file__, 335, 4), MovedAttribute_874, *[str_875, str_876, str_877], **kwargs_878)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_879)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 336)
# Processing the call arguments (line 336)
str_881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 19), 'str', 'urlunparse')
str_882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 33), 'str', 'urlparse')
str_883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 45), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 336)
kwargs_884 = {}
# Getting the type of 'MovedAttribute' (line 336)
MovedAttribute_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 336)
MovedAttribute_call_result_885 = invoke(stypy.reporting.localization.Localization(__file__, 336, 4), MovedAttribute_880, *[str_881, str_882, str_883], **kwargs_884)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_885)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 337)
# Processing the call arguments (line 337)
str_887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 19), 'str', 'urlunsplit')
str_888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 33), 'str', 'urlparse')
str_889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 45), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 337)
kwargs_890 = {}
# Getting the type of 'MovedAttribute' (line 337)
MovedAttribute_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 337)
MovedAttribute_call_result_891 = invoke(stypy.reporting.localization.Localization(__file__, 337, 4), MovedAttribute_886, *[str_887, str_888, str_889], **kwargs_890)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_891)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 338)
# Processing the call arguments (line 338)
str_893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 19), 'str', 'quote')
str_894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 28), 'str', 'urllib')
str_895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 38), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 338)
kwargs_896 = {}
# Getting the type of 'MovedAttribute' (line 338)
MovedAttribute_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 338)
MovedAttribute_call_result_897 = invoke(stypy.reporting.localization.Localization(__file__, 338, 4), MovedAttribute_892, *[str_893, str_894, str_895], **kwargs_896)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_897)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 339)
# Processing the call arguments (line 339)
str_899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 19), 'str', 'quote_plus')
str_900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 33), 'str', 'urllib')
str_901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 43), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 339)
kwargs_902 = {}
# Getting the type of 'MovedAttribute' (line 339)
MovedAttribute_898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 339)
MovedAttribute_call_result_903 = invoke(stypy.reporting.localization.Localization(__file__, 339, 4), MovedAttribute_898, *[str_899, str_900, str_901], **kwargs_902)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_903)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 340)
# Processing the call arguments (line 340)
str_905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 19), 'str', 'unquote')
str_906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 30), 'str', 'urllib')
str_907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 40), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 340)
kwargs_908 = {}
# Getting the type of 'MovedAttribute' (line 340)
MovedAttribute_904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 340)
MovedAttribute_call_result_909 = invoke(stypy.reporting.localization.Localization(__file__, 340, 4), MovedAttribute_904, *[str_905, str_906, str_907], **kwargs_908)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_909)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 341)
# Processing the call arguments (line 341)
str_911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 19), 'str', 'unquote_plus')
str_912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 35), 'str', 'urllib')
str_913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 45), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 341)
kwargs_914 = {}
# Getting the type of 'MovedAttribute' (line 341)
MovedAttribute_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 341)
MovedAttribute_call_result_915 = invoke(stypy.reporting.localization.Localization(__file__, 341, 4), MovedAttribute_910, *[str_911, str_912, str_913], **kwargs_914)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_915)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 342)
# Processing the call arguments (line 342)
str_917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 19), 'str', 'unquote_to_bytes')
str_918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 39), 'str', 'urllib')
str_919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 49), 'str', 'urllib.parse')
str_920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 65), 'str', 'unquote')
str_921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 76), 'str', 'unquote_to_bytes')
# Processing the call keyword arguments (line 342)
kwargs_922 = {}
# Getting the type of 'MovedAttribute' (line 342)
MovedAttribute_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 342)
MovedAttribute_call_result_923 = invoke(stypy.reporting.localization.Localization(__file__, 342, 4), MovedAttribute_916, *[str_917, str_918, str_919, str_920, str_921], **kwargs_922)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_923)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 343)
# Processing the call arguments (line 343)
str_925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 19), 'str', 'urlencode')
str_926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 32), 'str', 'urllib')
str_927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 42), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 343)
kwargs_928 = {}
# Getting the type of 'MovedAttribute' (line 343)
MovedAttribute_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 343)
MovedAttribute_call_result_929 = invoke(stypy.reporting.localization.Localization(__file__, 343, 4), MovedAttribute_924, *[str_925, str_926, str_927], **kwargs_928)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_929)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 344)
# Processing the call arguments (line 344)
str_931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 19), 'str', 'splitquery')
str_932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 33), 'str', 'urllib')
str_933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 43), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 344)
kwargs_934 = {}
# Getting the type of 'MovedAttribute' (line 344)
MovedAttribute_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 344)
MovedAttribute_call_result_935 = invoke(stypy.reporting.localization.Localization(__file__, 344, 4), MovedAttribute_930, *[str_931, str_932, str_933], **kwargs_934)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_935)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 345)
# Processing the call arguments (line 345)
str_937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 19), 'str', 'splittag')
str_938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 31), 'str', 'urllib')
str_939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 41), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 345)
kwargs_940 = {}
# Getting the type of 'MovedAttribute' (line 345)
MovedAttribute_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 345)
MovedAttribute_call_result_941 = invoke(stypy.reporting.localization.Localization(__file__, 345, 4), MovedAttribute_936, *[str_937, str_938, str_939], **kwargs_940)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_941)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 346)
# Processing the call arguments (line 346)
str_943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 19), 'str', 'splituser')
str_944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 32), 'str', 'urllib')
str_945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 42), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 346)
kwargs_946 = {}
# Getting the type of 'MovedAttribute' (line 346)
MovedAttribute_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 346)
MovedAttribute_call_result_947 = invoke(stypy.reporting.localization.Localization(__file__, 346, 4), MovedAttribute_942, *[str_943, str_944, str_945], **kwargs_946)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_947)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 347)
# Processing the call arguments (line 347)
str_949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 19), 'str', 'splitvalue')
str_950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 33), 'str', 'urllib')
str_951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 43), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 347)
kwargs_952 = {}
# Getting the type of 'MovedAttribute' (line 347)
MovedAttribute_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 347)
MovedAttribute_call_result_953 = invoke(stypy.reporting.localization.Localization(__file__, 347, 4), MovedAttribute_948, *[str_949, str_950, str_951], **kwargs_952)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_953)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 348)
# Processing the call arguments (line 348)
str_955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 19), 'str', 'uses_fragment')
str_956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 36), 'str', 'urlparse')
str_957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 48), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 348)
kwargs_958 = {}
# Getting the type of 'MovedAttribute' (line 348)
MovedAttribute_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 348)
MovedAttribute_call_result_959 = invoke(stypy.reporting.localization.Localization(__file__, 348, 4), MovedAttribute_954, *[str_955, str_956, str_957], **kwargs_958)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_959)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 349)
# Processing the call arguments (line 349)
str_961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 19), 'str', 'uses_netloc')
str_962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 34), 'str', 'urlparse')
str_963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 46), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 349)
kwargs_964 = {}
# Getting the type of 'MovedAttribute' (line 349)
MovedAttribute_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 349)
MovedAttribute_call_result_965 = invoke(stypy.reporting.localization.Localization(__file__, 349, 4), MovedAttribute_960, *[str_961, str_962, str_963], **kwargs_964)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_965)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 350)
# Processing the call arguments (line 350)
str_967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 19), 'str', 'uses_params')
str_968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 34), 'str', 'urlparse')
str_969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 46), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 350)
kwargs_970 = {}
# Getting the type of 'MovedAttribute' (line 350)
MovedAttribute_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 350)
MovedAttribute_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 350, 4), MovedAttribute_966, *[str_967, str_968, str_969], **kwargs_970)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_971)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 351)
# Processing the call arguments (line 351)
str_973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 19), 'str', 'uses_query')
str_974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 33), 'str', 'urlparse')
str_975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 45), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 351)
kwargs_976 = {}
# Getting the type of 'MovedAttribute' (line 351)
MovedAttribute_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 351)
MovedAttribute_call_result_977 = invoke(stypy.reporting.localization.Localization(__file__, 351, 4), MovedAttribute_972, *[str_973, str_974, str_975], **kwargs_976)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_977)
# Adding element type (line 327)

# Call to MovedAttribute(...): (line 352)
# Processing the call arguments (line 352)
str_979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 19), 'str', 'uses_relative')
str_980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 36), 'str', 'urlparse')
str_981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 48), 'str', 'urllib.parse')
# Processing the call keyword arguments (line 352)
kwargs_982 = {}
# Getting the type of 'MovedAttribute' (line 352)
MovedAttribute_978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 352)
MovedAttribute_call_result_983 = invoke(stypy.reporting.localization.Localization(__file__, 352, 4), MovedAttribute_978, *[str_979, str_980, str_981], **kwargs_982)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 33), list_831, MovedAttribute_call_result_983)

# Assigning a type to the variable '_urllib_parse_moved_attributes' (line 327)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 0), '_urllib_parse_moved_attributes', list_831)

# Getting the type of '_urllib_parse_moved_attributes' (line 354)
_urllib_parse_moved_attributes_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), '_urllib_parse_moved_attributes')
# Testing the type of a for loop iterable (line 354)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 354, 0), _urllib_parse_moved_attributes_984)
# Getting the type of the for loop variable (line 354)
for_loop_var_985 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 354, 0), _urllib_parse_moved_attributes_984)
# Assigning a type to the variable 'attr' (line 354)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'attr', for_loop_var_985)
# SSA begins for a for statement (line 354)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Call to setattr(...): (line 355)
# Processing the call arguments (line 355)
# Getting the type of 'Module_six_moves_urllib_parse' (line 355)
Module_six_moves_urllib_parse_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'Module_six_moves_urllib_parse', False)
# Getting the type of 'attr' (line 355)
attr_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 43), 'attr', False)
# Obtaining the member 'name' of a type (line 355)
name_989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 43), attr_988, 'name')
# Getting the type of 'attr' (line 355)
attr_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 54), 'attr', False)
# Processing the call keyword arguments (line 355)
kwargs_991 = {}
# Getting the type of 'setattr' (line 355)
setattr_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'setattr', False)
# Calling setattr(args, kwargs) (line 355)
setattr_call_result_992 = invoke(stypy.reporting.localization.Localization(__file__, 355, 4), setattr_986, *[Module_six_moves_urllib_parse_987, name_989, attr_990], **kwargs_991)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 356, 0), module_type_store, 'attr')

# Assigning a Name to a Attribute (line 358):
# Getting the type of '_urllib_parse_moved_attributes' (line 358)
_urllib_parse_moved_attributes_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 50), '_urllib_parse_moved_attributes')
# Getting the type of 'Module_six_moves_urllib_parse' (line 358)
Module_six_moves_urllib_parse_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), 'Module_six_moves_urllib_parse')
# Setting the type of the member '_moved_attributes' of a type (line 358)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 0), Module_six_moves_urllib_parse_994, '_moved_attributes', _urllib_parse_moved_attributes_993)

# Call to _add_module(...): (line 360)
# Processing the call arguments (line 360)

# Call to Module_six_moves_urllib_parse(...): (line 360)
# Processing the call arguments (line 360)
# Getting the type of '__name__' (line 360)
name___998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 52), '__name__', False)
str_999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 63), 'str', '.moves.urllib_parse')
# Applying the binary operator '+' (line 360)
result_add_1000 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 52), '+', name___998, str_999)

# Processing the call keyword arguments (line 360)
kwargs_1001 = {}
# Getting the type of 'Module_six_moves_urllib_parse' (line 360)
Module_six_moves_urllib_parse_997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 22), 'Module_six_moves_urllib_parse', False)
# Calling Module_six_moves_urllib_parse(args, kwargs) (line 360)
Module_six_moves_urllib_parse_call_result_1002 = invoke(stypy.reporting.localization.Localization(__file__, 360, 22), Module_six_moves_urllib_parse_997, *[result_add_1000], **kwargs_1001)

str_1003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 22), 'str', 'moves.urllib_parse')
str_1004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 44), 'str', 'moves.urllib.parse')
# Processing the call keyword arguments (line 360)
kwargs_1005 = {}
# Getting the type of '_importer' (line 360)
_importer_995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 0), '_importer', False)
# Obtaining the member '_add_module' of a type (line 360)
_add_module_996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 0), _importer_995, '_add_module')
# Calling _add_module(args, kwargs) (line 360)
_add_module_call_result_1006 = invoke(stypy.reporting.localization.Localization(__file__, 360, 0), _add_module_996, *[Module_six_moves_urllib_parse_call_result_1002, str_1003, str_1004], **kwargs_1005)

# Declaration of the 'Module_six_moves_urllib_error' class
# Getting the type of '_LazyModule' (line 364)
_LazyModule_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 36), '_LazyModule')

class Module_six_moves_urllib_error(_LazyModule_1007, ):
    str_1008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 4), 'str', 'Lazy loading of moved objects in six.moves.urllib_error')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 364, 0, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Module_six_moves_urllib_error.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Module_six_moves_urllib_error' (line 364)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 0), 'Module_six_moves_urllib_error', Module_six_moves_urllib_error)

# Assigning a List to a Name (line 369):

# Obtaining an instance of the builtin type 'list' (line 369)
list_1009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 369)
# Adding element type (line 369)

# Call to MovedAttribute(...): (line 370)
# Processing the call arguments (line 370)
str_1011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 19), 'str', 'URLError')
str_1012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 31), 'str', 'urllib2')
str_1013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 42), 'str', 'urllib.error')
# Processing the call keyword arguments (line 370)
kwargs_1014 = {}
# Getting the type of 'MovedAttribute' (line 370)
MovedAttribute_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 370)
MovedAttribute_call_result_1015 = invoke(stypy.reporting.localization.Localization(__file__, 370, 4), MovedAttribute_1010, *[str_1011, str_1012, str_1013], **kwargs_1014)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 33), list_1009, MovedAttribute_call_result_1015)
# Adding element type (line 369)

# Call to MovedAttribute(...): (line 371)
# Processing the call arguments (line 371)
str_1017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 19), 'str', 'HTTPError')
str_1018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 32), 'str', 'urllib2')
str_1019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 43), 'str', 'urllib.error')
# Processing the call keyword arguments (line 371)
kwargs_1020 = {}
# Getting the type of 'MovedAttribute' (line 371)
MovedAttribute_1016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 371)
MovedAttribute_call_result_1021 = invoke(stypy.reporting.localization.Localization(__file__, 371, 4), MovedAttribute_1016, *[str_1017, str_1018, str_1019], **kwargs_1020)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 33), list_1009, MovedAttribute_call_result_1021)
# Adding element type (line 369)

# Call to MovedAttribute(...): (line 372)
# Processing the call arguments (line 372)
str_1023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 19), 'str', 'ContentTooShortError')
str_1024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 43), 'str', 'urllib')
str_1025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 53), 'str', 'urllib.error')
# Processing the call keyword arguments (line 372)
kwargs_1026 = {}
# Getting the type of 'MovedAttribute' (line 372)
MovedAttribute_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 372)
MovedAttribute_call_result_1027 = invoke(stypy.reporting.localization.Localization(__file__, 372, 4), MovedAttribute_1022, *[str_1023, str_1024, str_1025], **kwargs_1026)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 33), list_1009, MovedAttribute_call_result_1027)

# Assigning a type to the variable '_urllib_error_moved_attributes' (line 369)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 0), '_urllib_error_moved_attributes', list_1009)

# Getting the type of '_urllib_error_moved_attributes' (line 374)
_urllib_error_moved_attributes_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), '_urllib_error_moved_attributes')
# Testing the type of a for loop iterable (line 374)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 374, 0), _urllib_error_moved_attributes_1028)
# Getting the type of the for loop variable (line 374)
for_loop_var_1029 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 374, 0), _urllib_error_moved_attributes_1028)
# Assigning a type to the variable 'attr' (line 374)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'attr', for_loop_var_1029)
# SSA begins for a for statement (line 374)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Call to setattr(...): (line 375)
# Processing the call arguments (line 375)
# Getting the type of 'Module_six_moves_urllib_error' (line 375)
Module_six_moves_urllib_error_1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'Module_six_moves_urllib_error', False)
# Getting the type of 'attr' (line 375)
attr_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 43), 'attr', False)
# Obtaining the member 'name' of a type (line 375)
name_1033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 43), attr_1032, 'name')
# Getting the type of 'attr' (line 375)
attr_1034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 54), 'attr', False)
# Processing the call keyword arguments (line 375)
kwargs_1035 = {}
# Getting the type of 'setattr' (line 375)
setattr_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'setattr', False)
# Calling setattr(args, kwargs) (line 375)
setattr_call_result_1036 = invoke(stypy.reporting.localization.Localization(__file__, 375, 4), setattr_1030, *[Module_six_moves_urllib_error_1031, name_1033, attr_1034], **kwargs_1035)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 376, 0), module_type_store, 'attr')

# Assigning a Name to a Attribute (line 378):
# Getting the type of '_urllib_error_moved_attributes' (line 378)
_urllib_error_moved_attributes_1037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 50), '_urllib_error_moved_attributes')
# Getting the type of 'Module_six_moves_urllib_error' (line 378)
Module_six_moves_urllib_error_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 0), 'Module_six_moves_urllib_error')
# Setting the type of the member '_moved_attributes' of a type (line 378)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 0), Module_six_moves_urllib_error_1038, '_moved_attributes', _urllib_error_moved_attributes_1037)

# Call to _add_module(...): (line 380)
# Processing the call arguments (line 380)

# Call to Module_six_moves_urllib_error(...): (line 380)
# Processing the call arguments (line 380)
# Getting the type of '__name__' (line 380)
name___1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 52), '__name__', False)
str_1043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 63), 'str', '.moves.urllib.error')
# Applying the binary operator '+' (line 380)
result_add_1044 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 52), '+', name___1042, str_1043)

# Processing the call keyword arguments (line 380)
kwargs_1045 = {}
# Getting the type of 'Module_six_moves_urllib_error' (line 380)
Module_six_moves_urllib_error_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 22), 'Module_six_moves_urllib_error', False)
# Calling Module_six_moves_urllib_error(args, kwargs) (line 380)
Module_six_moves_urllib_error_call_result_1046 = invoke(stypy.reporting.localization.Localization(__file__, 380, 22), Module_six_moves_urllib_error_1041, *[result_add_1044], **kwargs_1045)

str_1047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 22), 'str', 'moves.urllib_error')
str_1048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 44), 'str', 'moves.urllib.error')
# Processing the call keyword arguments (line 380)
kwargs_1049 = {}
# Getting the type of '_importer' (line 380)
_importer_1039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), '_importer', False)
# Obtaining the member '_add_module' of a type (line 380)
_add_module_1040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 0), _importer_1039, '_add_module')
# Calling _add_module(args, kwargs) (line 380)
_add_module_call_result_1050 = invoke(stypy.reporting.localization.Localization(__file__, 380, 0), _add_module_1040, *[Module_six_moves_urllib_error_call_result_1046, str_1047, str_1048], **kwargs_1049)

# Declaration of the 'Module_six_moves_urllib_request' class
# Getting the type of '_LazyModule' (line 384)
_LazyModule_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 38), '_LazyModule')

class Module_six_moves_urllib_request(_LazyModule_1051, ):
    str_1052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 4), 'str', 'Lazy loading of moved objects in six.moves.urllib_request')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 384, 0, False)
        # Assigning a type to the variable 'self' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Module_six_moves_urllib_request.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Module_six_moves_urllib_request' (line 384)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 0), 'Module_six_moves_urllib_request', Module_six_moves_urllib_request)

# Assigning a List to a Name (line 389):

# Obtaining an instance of the builtin type 'list' (line 389)
list_1053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 35), 'list')
# Adding type elements to the builtin type 'list' instance (line 389)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 390)
# Processing the call arguments (line 390)
str_1055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 19), 'str', 'urlopen')
str_1056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 30), 'str', 'urllib2')
str_1057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 41), 'str', 'urllib.request')
# Processing the call keyword arguments (line 390)
kwargs_1058 = {}
# Getting the type of 'MovedAttribute' (line 390)
MovedAttribute_1054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 390)
MovedAttribute_call_result_1059 = invoke(stypy.reporting.localization.Localization(__file__, 390, 4), MovedAttribute_1054, *[str_1055, str_1056, str_1057], **kwargs_1058)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1059)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 391)
# Processing the call arguments (line 391)
str_1061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 19), 'str', 'install_opener')
str_1062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 37), 'str', 'urllib2')
str_1063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 48), 'str', 'urllib.request')
# Processing the call keyword arguments (line 391)
kwargs_1064 = {}
# Getting the type of 'MovedAttribute' (line 391)
MovedAttribute_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 391)
MovedAttribute_call_result_1065 = invoke(stypy.reporting.localization.Localization(__file__, 391, 4), MovedAttribute_1060, *[str_1061, str_1062, str_1063], **kwargs_1064)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1065)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 392)
# Processing the call arguments (line 392)
str_1067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 19), 'str', 'build_opener')
str_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 35), 'str', 'urllib2')
str_1069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 46), 'str', 'urllib.request')
# Processing the call keyword arguments (line 392)
kwargs_1070 = {}
# Getting the type of 'MovedAttribute' (line 392)
MovedAttribute_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 392)
MovedAttribute_call_result_1071 = invoke(stypy.reporting.localization.Localization(__file__, 392, 4), MovedAttribute_1066, *[str_1067, str_1068, str_1069], **kwargs_1070)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1071)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 393)
# Processing the call arguments (line 393)
str_1073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 19), 'str', 'pathname2url')
str_1074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 35), 'str', 'urllib')
str_1075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 45), 'str', 'urllib.request')
# Processing the call keyword arguments (line 393)
kwargs_1076 = {}
# Getting the type of 'MovedAttribute' (line 393)
MovedAttribute_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 393)
MovedAttribute_call_result_1077 = invoke(stypy.reporting.localization.Localization(__file__, 393, 4), MovedAttribute_1072, *[str_1073, str_1074, str_1075], **kwargs_1076)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1077)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 394)
# Processing the call arguments (line 394)
str_1079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 19), 'str', 'url2pathname')
str_1080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 35), 'str', 'urllib')
str_1081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 45), 'str', 'urllib.request')
# Processing the call keyword arguments (line 394)
kwargs_1082 = {}
# Getting the type of 'MovedAttribute' (line 394)
MovedAttribute_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 394)
MovedAttribute_call_result_1083 = invoke(stypy.reporting.localization.Localization(__file__, 394, 4), MovedAttribute_1078, *[str_1079, str_1080, str_1081], **kwargs_1082)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1083)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 395)
# Processing the call arguments (line 395)
str_1085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 19), 'str', 'getproxies')
str_1086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 33), 'str', 'urllib')
str_1087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 43), 'str', 'urllib.request')
# Processing the call keyword arguments (line 395)
kwargs_1088 = {}
# Getting the type of 'MovedAttribute' (line 395)
MovedAttribute_1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 395)
MovedAttribute_call_result_1089 = invoke(stypy.reporting.localization.Localization(__file__, 395, 4), MovedAttribute_1084, *[str_1085, str_1086, str_1087], **kwargs_1088)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1089)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 396)
# Processing the call arguments (line 396)
str_1091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 19), 'str', 'Request')
str_1092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 30), 'str', 'urllib2')
str_1093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 41), 'str', 'urllib.request')
# Processing the call keyword arguments (line 396)
kwargs_1094 = {}
# Getting the type of 'MovedAttribute' (line 396)
MovedAttribute_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 396)
MovedAttribute_call_result_1095 = invoke(stypy.reporting.localization.Localization(__file__, 396, 4), MovedAttribute_1090, *[str_1091, str_1092, str_1093], **kwargs_1094)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1095)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 397)
# Processing the call arguments (line 397)
str_1097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 19), 'str', 'OpenerDirector')
str_1098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 37), 'str', 'urllib2')
str_1099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 48), 'str', 'urllib.request')
# Processing the call keyword arguments (line 397)
kwargs_1100 = {}
# Getting the type of 'MovedAttribute' (line 397)
MovedAttribute_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 397)
MovedAttribute_call_result_1101 = invoke(stypy.reporting.localization.Localization(__file__, 397, 4), MovedAttribute_1096, *[str_1097, str_1098, str_1099], **kwargs_1100)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1101)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 398)
# Processing the call arguments (line 398)
str_1103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 19), 'str', 'HTTPDefaultErrorHandler')
str_1104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 46), 'str', 'urllib2')
str_1105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 57), 'str', 'urllib.request')
# Processing the call keyword arguments (line 398)
kwargs_1106 = {}
# Getting the type of 'MovedAttribute' (line 398)
MovedAttribute_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 398)
MovedAttribute_call_result_1107 = invoke(stypy.reporting.localization.Localization(__file__, 398, 4), MovedAttribute_1102, *[str_1103, str_1104, str_1105], **kwargs_1106)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1107)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 399)
# Processing the call arguments (line 399)
str_1109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 19), 'str', 'HTTPRedirectHandler')
str_1110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 42), 'str', 'urllib2')
str_1111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 53), 'str', 'urllib.request')
# Processing the call keyword arguments (line 399)
kwargs_1112 = {}
# Getting the type of 'MovedAttribute' (line 399)
MovedAttribute_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 399)
MovedAttribute_call_result_1113 = invoke(stypy.reporting.localization.Localization(__file__, 399, 4), MovedAttribute_1108, *[str_1109, str_1110, str_1111], **kwargs_1112)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1113)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 400)
# Processing the call arguments (line 400)
str_1115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 19), 'str', 'HTTPCookieProcessor')
str_1116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 42), 'str', 'urllib2')
str_1117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 53), 'str', 'urllib.request')
# Processing the call keyword arguments (line 400)
kwargs_1118 = {}
# Getting the type of 'MovedAttribute' (line 400)
MovedAttribute_1114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 400)
MovedAttribute_call_result_1119 = invoke(stypy.reporting.localization.Localization(__file__, 400, 4), MovedAttribute_1114, *[str_1115, str_1116, str_1117], **kwargs_1118)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1119)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 401)
# Processing the call arguments (line 401)
str_1121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 19), 'str', 'ProxyHandler')
str_1122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 35), 'str', 'urllib2')
str_1123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 46), 'str', 'urllib.request')
# Processing the call keyword arguments (line 401)
kwargs_1124 = {}
# Getting the type of 'MovedAttribute' (line 401)
MovedAttribute_1120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 401)
MovedAttribute_call_result_1125 = invoke(stypy.reporting.localization.Localization(__file__, 401, 4), MovedAttribute_1120, *[str_1121, str_1122, str_1123], **kwargs_1124)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1125)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 402)
# Processing the call arguments (line 402)
str_1127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 19), 'str', 'BaseHandler')
str_1128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 34), 'str', 'urllib2')
str_1129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 45), 'str', 'urllib.request')
# Processing the call keyword arguments (line 402)
kwargs_1130 = {}
# Getting the type of 'MovedAttribute' (line 402)
MovedAttribute_1126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 402)
MovedAttribute_call_result_1131 = invoke(stypy.reporting.localization.Localization(__file__, 402, 4), MovedAttribute_1126, *[str_1127, str_1128, str_1129], **kwargs_1130)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1131)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 403)
# Processing the call arguments (line 403)
str_1133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 19), 'str', 'HTTPPasswordMgr')
str_1134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 38), 'str', 'urllib2')
str_1135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 49), 'str', 'urllib.request')
# Processing the call keyword arguments (line 403)
kwargs_1136 = {}
# Getting the type of 'MovedAttribute' (line 403)
MovedAttribute_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 403)
MovedAttribute_call_result_1137 = invoke(stypy.reporting.localization.Localization(__file__, 403, 4), MovedAttribute_1132, *[str_1133, str_1134, str_1135], **kwargs_1136)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1137)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 404)
# Processing the call arguments (line 404)
str_1139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 19), 'str', 'HTTPPasswordMgrWithDefaultRealm')
str_1140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 54), 'str', 'urllib2')
str_1141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 65), 'str', 'urllib.request')
# Processing the call keyword arguments (line 404)
kwargs_1142 = {}
# Getting the type of 'MovedAttribute' (line 404)
MovedAttribute_1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 404)
MovedAttribute_call_result_1143 = invoke(stypy.reporting.localization.Localization(__file__, 404, 4), MovedAttribute_1138, *[str_1139, str_1140, str_1141], **kwargs_1142)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1143)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 405)
# Processing the call arguments (line 405)
str_1145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 19), 'str', 'AbstractBasicAuthHandler')
str_1146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 47), 'str', 'urllib2')
str_1147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 58), 'str', 'urllib.request')
# Processing the call keyword arguments (line 405)
kwargs_1148 = {}
# Getting the type of 'MovedAttribute' (line 405)
MovedAttribute_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 405)
MovedAttribute_call_result_1149 = invoke(stypy.reporting.localization.Localization(__file__, 405, 4), MovedAttribute_1144, *[str_1145, str_1146, str_1147], **kwargs_1148)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1149)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 406)
# Processing the call arguments (line 406)
str_1151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 19), 'str', 'HTTPBasicAuthHandler')
str_1152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 43), 'str', 'urllib2')
str_1153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 54), 'str', 'urllib.request')
# Processing the call keyword arguments (line 406)
kwargs_1154 = {}
# Getting the type of 'MovedAttribute' (line 406)
MovedAttribute_1150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 406)
MovedAttribute_call_result_1155 = invoke(stypy.reporting.localization.Localization(__file__, 406, 4), MovedAttribute_1150, *[str_1151, str_1152, str_1153], **kwargs_1154)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1155)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 407)
# Processing the call arguments (line 407)
str_1157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 19), 'str', 'ProxyBasicAuthHandler')
str_1158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 44), 'str', 'urllib2')
str_1159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 55), 'str', 'urllib.request')
# Processing the call keyword arguments (line 407)
kwargs_1160 = {}
# Getting the type of 'MovedAttribute' (line 407)
MovedAttribute_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 407)
MovedAttribute_call_result_1161 = invoke(stypy.reporting.localization.Localization(__file__, 407, 4), MovedAttribute_1156, *[str_1157, str_1158, str_1159], **kwargs_1160)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1161)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 408)
# Processing the call arguments (line 408)
str_1163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 19), 'str', 'AbstractDigestAuthHandler')
str_1164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 48), 'str', 'urllib2')
str_1165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 59), 'str', 'urllib.request')
# Processing the call keyword arguments (line 408)
kwargs_1166 = {}
# Getting the type of 'MovedAttribute' (line 408)
MovedAttribute_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 408)
MovedAttribute_call_result_1167 = invoke(stypy.reporting.localization.Localization(__file__, 408, 4), MovedAttribute_1162, *[str_1163, str_1164, str_1165], **kwargs_1166)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1167)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 409)
# Processing the call arguments (line 409)
str_1169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 19), 'str', 'HTTPDigestAuthHandler')
str_1170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 44), 'str', 'urllib2')
str_1171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 55), 'str', 'urllib.request')
# Processing the call keyword arguments (line 409)
kwargs_1172 = {}
# Getting the type of 'MovedAttribute' (line 409)
MovedAttribute_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 409)
MovedAttribute_call_result_1173 = invoke(stypy.reporting.localization.Localization(__file__, 409, 4), MovedAttribute_1168, *[str_1169, str_1170, str_1171], **kwargs_1172)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1173)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 410)
# Processing the call arguments (line 410)
str_1175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 19), 'str', 'ProxyDigestAuthHandler')
str_1176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 45), 'str', 'urllib2')
str_1177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 56), 'str', 'urllib.request')
# Processing the call keyword arguments (line 410)
kwargs_1178 = {}
# Getting the type of 'MovedAttribute' (line 410)
MovedAttribute_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 410)
MovedAttribute_call_result_1179 = invoke(stypy.reporting.localization.Localization(__file__, 410, 4), MovedAttribute_1174, *[str_1175, str_1176, str_1177], **kwargs_1178)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1179)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 411)
# Processing the call arguments (line 411)
str_1181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 19), 'str', 'HTTPHandler')
str_1182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 34), 'str', 'urllib2')
str_1183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 45), 'str', 'urllib.request')
# Processing the call keyword arguments (line 411)
kwargs_1184 = {}
# Getting the type of 'MovedAttribute' (line 411)
MovedAttribute_1180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 411)
MovedAttribute_call_result_1185 = invoke(stypy.reporting.localization.Localization(__file__, 411, 4), MovedAttribute_1180, *[str_1181, str_1182, str_1183], **kwargs_1184)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1185)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 412)
# Processing the call arguments (line 412)
str_1187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 19), 'str', 'HTTPSHandler')
str_1188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 35), 'str', 'urllib2')
str_1189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 46), 'str', 'urllib.request')
# Processing the call keyword arguments (line 412)
kwargs_1190 = {}
# Getting the type of 'MovedAttribute' (line 412)
MovedAttribute_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 412)
MovedAttribute_call_result_1191 = invoke(stypy.reporting.localization.Localization(__file__, 412, 4), MovedAttribute_1186, *[str_1187, str_1188, str_1189], **kwargs_1190)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1191)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 413)
# Processing the call arguments (line 413)
str_1193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 19), 'str', 'FileHandler')
str_1194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 34), 'str', 'urllib2')
str_1195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 45), 'str', 'urllib.request')
# Processing the call keyword arguments (line 413)
kwargs_1196 = {}
# Getting the type of 'MovedAttribute' (line 413)
MovedAttribute_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 413)
MovedAttribute_call_result_1197 = invoke(stypy.reporting.localization.Localization(__file__, 413, 4), MovedAttribute_1192, *[str_1193, str_1194, str_1195], **kwargs_1196)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1197)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 414)
# Processing the call arguments (line 414)
str_1199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 19), 'str', 'FTPHandler')
str_1200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 33), 'str', 'urllib2')
str_1201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 44), 'str', 'urllib.request')
# Processing the call keyword arguments (line 414)
kwargs_1202 = {}
# Getting the type of 'MovedAttribute' (line 414)
MovedAttribute_1198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 414)
MovedAttribute_call_result_1203 = invoke(stypy.reporting.localization.Localization(__file__, 414, 4), MovedAttribute_1198, *[str_1199, str_1200, str_1201], **kwargs_1202)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1203)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 415)
# Processing the call arguments (line 415)
str_1205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 19), 'str', 'CacheFTPHandler')
str_1206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 38), 'str', 'urllib2')
str_1207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 49), 'str', 'urllib.request')
# Processing the call keyword arguments (line 415)
kwargs_1208 = {}
# Getting the type of 'MovedAttribute' (line 415)
MovedAttribute_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 415)
MovedAttribute_call_result_1209 = invoke(stypy.reporting.localization.Localization(__file__, 415, 4), MovedAttribute_1204, *[str_1205, str_1206, str_1207], **kwargs_1208)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1209)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 416)
# Processing the call arguments (line 416)
str_1211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 19), 'str', 'UnknownHandler')
str_1212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 37), 'str', 'urllib2')
str_1213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 48), 'str', 'urllib.request')
# Processing the call keyword arguments (line 416)
kwargs_1214 = {}
# Getting the type of 'MovedAttribute' (line 416)
MovedAttribute_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 416)
MovedAttribute_call_result_1215 = invoke(stypy.reporting.localization.Localization(__file__, 416, 4), MovedAttribute_1210, *[str_1211, str_1212, str_1213], **kwargs_1214)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1215)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 417)
# Processing the call arguments (line 417)
str_1217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 19), 'str', 'HTTPErrorProcessor')
str_1218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 41), 'str', 'urllib2')
str_1219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 52), 'str', 'urllib.request')
# Processing the call keyword arguments (line 417)
kwargs_1220 = {}
# Getting the type of 'MovedAttribute' (line 417)
MovedAttribute_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 417)
MovedAttribute_call_result_1221 = invoke(stypy.reporting.localization.Localization(__file__, 417, 4), MovedAttribute_1216, *[str_1217, str_1218, str_1219], **kwargs_1220)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1221)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 418)
# Processing the call arguments (line 418)
str_1223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 19), 'str', 'urlretrieve')
str_1224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 34), 'str', 'urllib')
str_1225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 44), 'str', 'urllib.request')
# Processing the call keyword arguments (line 418)
kwargs_1226 = {}
# Getting the type of 'MovedAttribute' (line 418)
MovedAttribute_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 418)
MovedAttribute_call_result_1227 = invoke(stypy.reporting.localization.Localization(__file__, 418, 4), MovedAttribute_1222, *[str_1223, str_1224, str_1225], **kwargs_1226)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1227)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 419)
# Processing the call arguments (line 419)
str_1229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 19), 'str', 'urlcleanup')
str_1230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 33), 'str', 'urllib')
str_1231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 43), 'str', 'urllib.request')
# Processing the call keyword arguments (line 419)
kwargs_1232 = {}
# Getting the type of 'MovedAttribute' (line 419)
MovedAttribute_1228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 419)
MovedAttribute_call_result_1233 = invoke(stypy.reporting.localization.Localization(__file__, 419, 4), MovedAttribute_1228, *[str_1229, str_1230, str_1231], **kwargs_1232)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1233)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 420)
# Processing the call arguments (line 420)
str_1235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 19), 'str', 'URLopener')
str_1236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 32), 'str', 'urllib')
str_1237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 42), 'str', 'urllib.request')
# Processing the call keyword arguments (line 420)
kwargs_1238 = {}
# Getting the type of 'MovedAttribute' (line 420)
MovedAttribute_1234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 420)
MovedAttribute_call_result_1239 = invoke(stypy.reporting.localization.Localization(__file__, 420, 4), MovedAttribute_1234, *[str_1235, str_1236, str_1237], **kwargs_1238)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1239)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 421)
# Processing the call arguments (line 421)
str_1241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 19), 'str', 'FancyURLopener')
str_1242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 37), 'str', 'urllib')
str_1243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 47), 'str', 'urllib.request')
# Processing the call keyword arguments (line 421)
kwargs_1244 = {}
# Getting the type of 'MovedAttribute' (line 421)
MovedAttribute_1240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 421)
MovedAttribute_call_result_1245 = invoke(stypy.reporting.localization.Localization(__file__, 421, 4), MovedAttribute_1240, *[str_1241, str_1242, str_1243], **kwargs_1244)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1245)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 422)
# Processing the call arguments (line 422)
str_1247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 19), 'str', 'proxy_bypass')
str_1248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 35), 'str', 'urllib')
str_1249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 45), 'str', 'urllib.request')
# Processing the call keyword arguments (line 422)
kwargs_1250 = {}
# Getting the type of 'MovedAttribute' (line 422)
MovedAttribute_1246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 422)
MovedAttribute_call_result_1251 = invoke(stypy.reporting.localization.Localization(__file__, 422, 4), MovedAttribute_1246, *[str_1247, str_1248, str_1249], **kwargs_1250)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1251)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 423)
# Processing the call arguments (line 423)
str_1253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 19), 'str', 'parse_http_list')
str_1254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 38), 'str', 'urllib2')
str_1255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 49), 'str', 'urllib.request')
# Processing the call keyword arguments (line 423)
kwargs_1256 = {}
# Getting the type of 'MovedAttribute' (line 423)
MovedAttribute_1252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 423)
MovedAttribute_call_result_1257 = invoke(stypy.reporting.localization.Localization(__file__, 423, 4), MovedAttribute_1252, *[str_1253, str_1254, str_1255], **kwargs_1256)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1257)
# Adding element type (line 389)

# Call to MovedAttribute(...): (line 424)
# Processing the call arguments (line 424)
str_1259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 19), 'str', 'parse_keqv_list')
str_1260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 38), 'str', 'urllib2')
str_1261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 49), 'str', 'urllib.request')
# Processing the call keyword arguments (line 424)
kwargs_1262 = {}
# Getting the type of 'MovedAttribute' (line 424)
MovedAttribute_1258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 424)
MovedAttribute_call_result_1263 = invoke(stypy.reporting.localization.Localization(__file__, 424, 4), MovedAttribute_1258, *[str_1259, str_1260, str_1261], **kwargs_1262)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 35), list_1053, MovedAttribute_call_result_1263)

# Assigning a type to the variable '_urllib_request_moved_attributes' (line 389)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), '_urllib_request_moved_attributes', list_1053)

# Getting the type of '_urllib_request_moved_attributes' (line 426)
_urllib_request_moved_attributes_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), '_urllib_request_moved_attributes')
# Testing the type of a for loop iterable (line 426)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 426, 0), _urllib_request_moved_attributes_1264)
# Getting the type of the for loop variable (line 426)
for_loop_var_1265 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 426, 0), _urllib_request_moved_attributes_1264)
# Assigning a type to the variable 'attr' (line 426)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 0), 'attr', for_loop_var_1265)
# SSA begins for a for statement (line 426)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Call to setattr(...): (line 427)
# Processing the call arguments (line 427)
# Getting the type of 'Module_six_moves_urllib_request' (line 427)
Module_six_moves_urllib_request_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'Module_six_moves_urllib_request', False)
# Getting the type of 'attr' (line 427)
attr_1268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 45), 'attr', False)
# Obtaining the member 'name' of a type (line 427)
name_1269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 45), attr_1268, 'name')
# Getting the type of 'attr' (line 427)
attr_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 56), 'attr', False)
# Processing the call keyword arguments (line 427)
kwargs_1271 = {}
# Getting the type of 'setattr' (line 427)
setattr_1266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'setattr', False)
# Calling setattr(args, kwargs) (line 427)
setattr_call_result_1272 = invoke(stypy.reporting.localization.Localization(__file__, 427, 4), setattr_1266, *[Module_six_moves_urllib_request_1267, name_1269, attr_1270], **kwargs_1271)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 428, 0), module_type_store, 'attr')

# Assigning a Name to a Attribute (line 430):
# Getting the type of '_urllib_request_moved_attributes' (line 430)
_urllib_request_moved_attributes_1273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 52), '_urllib_request_moved_attributes')
# Getting the type of 'Module_six_moves_urllib_request' (line 430)
Module_six_moves_urllib_request_1274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 0), 'Module_six_moves_urllib_request')
# Setting the type of the member '_moved_attributes' of a type (line 430)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 0), Module_six_moves_urllib_request_1274, '_moved_attributes', _urllib_request_moved_attributes_1273)

# Call to _add_module(...): (line 432)
# Processing the call arguments (line 432)

# Call to Module_six_moves_urllib_request(...): (line 432)
# Processing the call arguments (line 432)
# Getting the type of '__name__' (line 432)
name___1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 54), '__name__', False)
str_1279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 65), 'str', '.moves.urllib.request')
# Applying the binary operator '+' (line 432)
result_add_1280 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 54), '+', name___1278, str_1279)

# Processing the call keyword arguments (line 432)
kwargs_1281 = {}
# Getting the type of 'Module_six_moves_urllib_request' (line 432)
Module_six_moves_urllib_request_1277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 22), 'Module_six_moves_urllib_request', False)
# Calling Module_six_moves_urllib_request(args, kwargs) (line 432)
Module_six_moves_urllib_request_call_result_1282 = invoke(stypy.reporting.localization.Localization(__file__, 432, 22), Module_six_moves_urllib_request_1277, *[result_add_1280], **kwargs_1281)

str_1283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 22), 'str', 'moves.urllib_request')
str_1284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 46), 'str', 'moves.urllib.request')
# Processing the call keyword arguments (line 432)
kwargs_1285 = {}
# Getting the type of '_importer' (line 432)
_importer_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 0), '_importer', False)
# Obtaining the member '_add_module' of a type (line 432)
_add_module_1276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 0), _importer_1275, '_add_module')
# Calling _add_module(args, kwargs) (line 432)
_add_module_call_result_1286 = invoke(stypy.reporting.localization.Localization(__file__, 432, 0), _add_module_1276, *[Module_six_moves_urllib_request_call_result_1282, str_1283, str_1284], **kwargs_1285)

# Declaration of the 'Module_six_moves_urllib_response' class
# Getting the type of '_LazyModule' (line 436)
_LazyModule_1287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 39), '_LazyModule')

class Module_six_moves_urllib_response(_LazyModule_1287, ):
    str_1288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 4), 'str', 'Lazy loading of moved objects in six.moves.urllib_response')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 436, 0, False)
        # Assigning a type to the variable 'self' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Module_six_moves_urllib_response.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Module_six_moves_urllib_response' (line 436)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 0), 'Module_six_moves_urllib_response', Module_six_moves_urllib_response)

# Assigning a List to a Name (line 441):

# Obtaining an instance of the builtin type 'list' (line 441)
list_1289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 441)
# Adding element type (line 441)

# Call to MovedAttribute(...): (line 442)
# Processing the call arguments (line 442)
str_1291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 19), 'str', 'addbase')
str_1292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 30), 'str', 'urllib')
str_1293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 40), 'str', 'urllib.response')
# Processing the call keyword arguments (line 442)
kwargs_1294 = {}
# Getting the type of 'MovedAttribute' (line 442)
MovedAttribute_1290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 442)
MovedAttribute_call_result_1295 = invoke(stypy.reporting.localization.Localization(__file__, 442, 4), MovedAttribute_1290, *[str_1291, str_1292, str_1293], **kwargs_1294)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 36), list_1289, MovedAttribute_call_result_1295)
# Adding element type (line 441)

# Call to MovedAttribute(...): (line 443)
# Processing the call arguments (line 443)
str_1297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 19), 'str', 'addclosehook')
str_1298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 35), 'str', 'urllib')
str_1299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 45), 'str', 'urllib.response')
# Processing the call keyword arguments (line 443)
kwargs_1300 = {}
# Getting the type of 'MovedAttribute' (line 443)
MovedAttribute_1296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 443)
MovedAttribute_call_result_1301 = invoke(stypy.reporting.localization.Localization(__file__, 443, 4), MovedAttribute_1296, *[str_1297, str_1298, str_1299], **kwargs_1300)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 36), list_1289, MovedAttribute_call_result_1301)
# Adding element type (line 441)

# Call to MovedAttribute(...): (line 444)
# Processing the call arguments (line 444)
str_1303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 19), 'str', 'addinfo')
str_1304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 30), 'str', 'urllib')
str_1305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 40), 'str', 'urllib.response')
# Processing the call keyword arguments (line 444)
kwargs_1306 = {}
# Getting the type of 'MovedAttribute' (line 444)
MovedAttribute_1302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 444)
MovedAttribute_call_result_1307 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), MovedAttribute_1302, *[str_1303, str_1304, str_1305], **kwargs_1306)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 36), list_1289, MovedAttribute_call_result_1307)
# Adding element type (line 441)

# Call to MovedAttribute(...): (line 445)
# Processing the call arguments (line 445)
str_1309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 19), 'str', 'addinfourl')
str_1310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 33), 'str', 'urllib')
str_1311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 43), 'str', 'urllib.response')
# Processing the call keyword arguments (line 445)
kwargs_1312 = {}
# Getting the type of 'MovedAttribute' (line 445)
MovedAttribute_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 445)
MovedAttribute_call_result_1313 = invoke(stypy.reporting.localization.Localization(__file__, 445, 4), MovedAttribute_1308, *[str_1309, str_1310, str_1311], **kwargs_1312)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 36), list_1289, MovedAttribute_call_result_1313)

# Assigning a type to the variable '_urllib_response_moved_attributes' (line 441)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 0), '_urllib_response_moved_attributes', list_1289)

# Getting the type of '_urllib_response_moved_attributes' (line 447)
_urllib_response_moved_attributes_1314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), '_urllib_response_moved_attributes')
# Testing the type of a for loop iterable (line 447)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 447, 0), _urllib_response_moved_attributes_1314)
# Getting the type of the for loop variable (line 447)
for_loop_var_1315 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 447, 0), _urllib_response_moved_attributes_1314)
# Assigning a type to the variable 'attr' (line 447)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 0), 'attr', for_loop_var_1315)
# SSA begins for a for statement (line 447)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Call to setattr(...): (line 448)
# Processing the call arguments (line 448)
# Getting the type of 'Module_six_moves_urllib_response' (line 448)
Module_six_moves_urllib_response_1317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'Module_six_moves_urllib_response', False)
# Getting the type of 'attr' (line 448)
attr_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 46), 'attr', False)
# Obtaining the member 'name' of a type (line 448)
name_1319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 46), attr_1318, 'name')
# Getting the type of 'attr' (line 448)
attr_1320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 57), 'attr', False)
# Processing the call keyword arguments (line 448)
kwargs_1321 = {}
# Getting the type of 'setattr' (line 448)
setattr_1316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'setattr', False)
# Calling setattr(args, kwargs) (line 448)
setattr_call_result_1322 = invoke(stypy.reporting.localization.Localization(__file__, 448, 4), setattr_1316, *[Module_six_moves_urllib_response_1317, name_1319, attr_1320], **kwargs_1321)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 449, 0), module_type_store, 'attr')

# Assigning a Name to a Attribute (line 451):
# Getting the type of '_urllib_response_moved_attributes' (line 451)
_urllib_response_moved_attributes_1323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 53), '_urllib_response_moved_attributes')
# Getting the type of 'Module_six_moves_urllib_response' (line 451)
Module_six_moves_urllib_response_1324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), 'Module_six_moves_urllib_response')
# Setting the type of the member '_moved_attributes' of a type (line 451)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 0), Module_six_moves_urllib_response_1324, '_moved_attributes', _urllib_response_moved_attributes_1323)

# Call to _add_module(...): (line 453)
# Processing the call arguments (line 453)

# Call to Module_six_moves_urllib_response(...): (line 453)
# Processing the call arguments (line 453)
# Getting the type of '__name__' (line 453)
name___1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 55), '__name__', False)
str_1329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 66), 'str', '.moves.urllib.response')
# Applying the binary operator '+' (line 453)
result_add_1330 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 55), '+', name___1328, str_1329)

# Processing the call keyword arguments (line 453)
kwargs_1331 = {}
# Getting the type of 'Module_six_moves_urllib_response' (line 453)
Module_six_moves_urllib_response_1327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 22), 'Module_six_moves_urllib_response', False)
# Calling Module_six_moves_urllib_response(args, kwargs) (line 453)
Module_six_moves_urllib_response_call_result_1332 = invoke(stypy.reporting.localization.Localization(__file__, 453, 22), Module_six_moves_urllib_response_1327, *[result_add_1330], **kwargs_1331)

str_1333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 22), 'str', 'moves.urllib_response')
str_1334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 47), 'str', 'moves.urllib.response')
# Processing the call keyword arguments (line 453)
kwargs_1335 = {}
# Getting the type of '_importer' (line 453)
_importer_1325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 0), '_importer', False)
# Obtaining the member '_add_module' of a type (line 453)
_add_module_1326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 0), _importer_1325, '_add_module')
# Calling _add_module(args, kwargs) (line 453)
_add_module_call_result_1336 = invoke(stypy.reporting.localization.Localization(__file__, 453, 0), _add_module_1326, *[Module_six_moves_urllib_response_call_result_1332, str_1333, str_1334], **kwargs_1335)

# Declaration of the 'Module_six_moves_urllib_robotparser' class
# Getting the type of '_LazyModule' (line 457)
_LazyModule_1337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 42), '_LazyModule')

class Module_six_moves_urllib_robotparser(_LazyModule_1337, ):
    str_1338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 4), 'str', 'Lazy loading of moved objects in six.moves.urllib_robotparser')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 457, 0, False)
        # Assigning a type to the variable 'self' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Module_six_moves_urllib_robotparser.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Module_six_moves_urllib_robotparser' (line 457)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 0), 'Module_six_moves_urllib_robotparser', Module_six_moves_urllib_robotparser)

# Assigning a List to a Name (line 462):

# Obtaining an instance of the builtin type 'list' (line 462)
list_1339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 462)
# Adding element type (line 462)

# Call to MovedAttribute(...): (line 463)
# Processing the call arguments (line 463)
str_1341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 19), 'str', 'RobotFileParser')
str_1342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 38), 'str', 'robotparser')
str_1343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 53), 'str', 'urllib.robotparser')
# Processing the call keyword arguments (line 463)
kwargs_1344 = {}
# Getting the type of 'MovedAttribute' (line 463)
MovedAttribute_1340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'MovedAttribute', False)
# Calling MovedAttribute(args, kwargs) (line 463)
MovedAttribute_call_result_1345 = invoke(stypy.reporting.localization.Localization(__file__, 463, 4), MovedAttribute_1340, *[str_1341, str_1342, str_1343], **kwargs_1344)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 39), list_1339, MovedAttribute_call_result_1345)

# Assigning a type to the variable '_urllib_robotparser_moved_attributes' (line 462)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), '_urllib_robotparser_moved_attributes', list_1339)

# Getting the type of '_urllib_robotparser_moved_attributes' (line 465)
_urllib_robotparser_moved_attributes_1346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), '_urllib_robotparser_moved_attributes')
# Testing the type of a for loop iterable (line 465)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 465, 0), _urllib_robotparser_moved_attributes_1346)
# Getting the type of the for loop variable (line 465)
for_loop_var_1347 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 465, 0), _urllib_robotparser_moved_attributes_1346)
# Assigning a type to the variable 'attr' (line 465)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 0), 'attr', for_loop_var_1347)
# SSA begins for a for statement (line 465)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Call to setattr(...): (line 466)
# Processing the call arguments (line 466)
# Getting the type of 'Module_six_moves_urllib_robotparser' (line 466)
Module_six_moves_urllib_robotparser_1349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'Module_six_moves_urllib_robotparser', False)
# Getting the type of 'attr' (line 466)
attr_1350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 49), 'attr', False)
# Obtaining the member 'name' of a type (line 466)
name_1351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 49), attr_1350, 'name')
# Getting the type of 'attr' (line 466)
attr_1352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 60), 'attr', False)
# Processing the call keyword arguments (line 466)
kwargs_1353 = {}
# Getting the type of 'setattr' (line 466)
setattr_1348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'setattr', False)
# Calling setattr(args, kwargs) (line 466)
setattr_call_result_1354 = invoke(stypy.reporting.localization.Localization(__file__, 466, 4), setattr_1348, *[Module_six_moves_urllib_robotparser_1349, name_1351, attr_1352], **kwargs_1353)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 467, 0), module_type_store, 'attr')

# Assigning a Name to a Attribute (line 469):
# Getting the type of '_urllib_robotparser_moved_attributes' (line 469)
_urllib_robotparser_moved_attributes_1355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 56), '_urllib_robotparser_moved_attributes')
# Getting the type of 'Module_six_moves_urllib_robotparser' (line 469)
Module_six_moves_urllib_robotparser_1356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 0), 'Module_six_moves_urllib_robotparser')
# Setting the type of the member '_moved_attributes' of a type (line 469)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 0), Module_six_moves_urllib_robotparser_1356, '_moved_attributes', _urllib_robotparser_moved_attributes_1355)

# Call to _add_module(...): (line 471)
# Processing the call arguments (line 471)

# Call to Module_six_moves_urllib_robotparser(...): (line 471)
# Processing the call arguments (line 471)
# Getting the type of '__name__' (line 471)
name___1360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 58), '__name__', False)
str_1361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 69), 'str', '.moves.urllib.robotparser')
# Applying the binary operator '+' (line 471)
result_add_1362 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 58), '+', name___1360, str_1361)

# Processing the call keyword arguments (line 471)
kwargs_1363 = {}
# Getting the type of 'Module_six_moves_urllib_robotparser' (line 471)
Module_six_moves_urllib_robotparser_1359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 22), 'Module_six_moves_urllib_robotparser', False)
# Calling Module_six_moves_urllib_robotparser(args, kwargs) (line 471)
Module_six_moves_urllib_robotparser_call_result_1364 = invoke(stypy.reporting.localization.Localization(__file__, 471, 22), Module_six_moves_urllib_robotparser_1359, *[result_add_1362], **kwargs_1363)

str_1365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 22), 'str', 'moves.urllib_robotparser')
str_1366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 50), 'str', 'moves.urllib.robotparser')
# Processing the call keyword arguments (line 471)
kwargs_1367 = {}
# Getting the type of '_importer' (line 471)
_importer_1357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 0), '_importer', False)
# Obtaining the member '_add_module' of a type (line 471)
_add_module_1358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 0), _importer_1357, '_add_module')
# Calling _add_module(args, kwargs) (line 471)
_add_module_call_result_1368 = invoke(stypy.reporting.localization.Localization(__file__, 471, 0), _add_module_1358, *[Module_six_moves_urllib_robotparser_call_result_1364, str_1365, str_1366], **kwargs_1367)

# Declaration of the 'Module_six_moves_urllib' class
# Getting the type of 'types' (line 475)
types_1369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 30), 'types')
# Obtaining the member 'ModuleType' of a type (line 475)
ModuleType_1370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 30), types_1369, 'ModuleType')

class Module_six_moves_urllib(ModuleType_1370, ):
    str_1371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 4), 'str', 'Create a six.moves.urllib namespace that resembles the Python 3 namespace')

    @norecursion
    def __dir__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__dir__'
        module_type_store = module_type_store.open_function_context('__dir__', 485, 4, False)
        # Assigning a type to the variable 'self' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_localization', localization)
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_function_name', 'Module_six_moves_urllib.__dir__')
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_param_names_list', [])
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Module_six_moves_urllib.__dir__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Module_six_moves_urllib.__dir__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__dir__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__dir__(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 486)
        list_1372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 486)
        # Adding element type (line 486)
        str_1373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 16), 'str', 'parse')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 15), list_1372, str_1373)
        # Adding element type (line 486)
        str_1374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 25), 'str', 'error')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 15), list_1372, str_1374)
        # Adding element type (line 486)
        str_1375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 34), 'str', 'request')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 15), list_1372, str_1375)
        # Adding element type (line 486)
        str_1376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 45), 'str', 'response')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 15), list_1372, str_1376)
        # Adding element type (line 486)
        str_1377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 57), 'str', 'robotparser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 15), list_1372, str_1377)
        
        # Assigning a type to the variable 'stypy_return_type' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'stypy_return_type', list_1372)
        
        # ################# End of '__dir__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__dir__' in the type store
        # Getting the type of 'stypy_return_type' (line 485)
        stypy_return_type_1378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1378)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__dir__'
        return stypy_return_type_1378


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 475, 0, False)
        # Assigning a type to the variable 'self' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Module_six_moves_urllib.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Module_six_moves_urllib' (line 475)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'Module_six_moves_urllib', Module_six_moves_urllib)

# Assigning a List to a Name (line 478):

# Obtaining an instance of the builtin type 'list' (line 478)
list_1379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 478)

# Getting the type of 'Module_six_moves_urllib'
Module_six_moves_urllib_1380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Module_six_moves_urllib')
# Setting the type of the member '__path__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Module_six_moves_urllib_1380, '__path__', list_1379)

# Assigning a Call to a Name (line 479):

# Call to _get_module(...): (line 479)
# Processing the call arguments (line 479)
str_1383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 34), 'str', 'moves.urllib_parse')
# Processing the call keyword arguments (line 479)
kwargs_1384 = {}
# Getting the type of '_importer' (line 479)
_importer_1381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), '_importer', False)
# Obtaining the member '_get_module' of a type (line 479)
_get_module_1382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 12), _importer_1381, '_get_module')
# Calling _get_module(args, kwargs) (line 479)
_get_module_call_result_1385 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), _get_module_1382, *[str_1383], **kwargs_1384)

# Getting the type of 'Module_six_moves_urllib'
Module_six_moves_urllib_1386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Module_six_moves_urllib')
# Setting the type of the member 'parse' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Module_six_moves_urllib_1386, 'parse', _get_module_call_result_1385)

# Assigning a Call to a Name (line 480):

# Call to _get_module(...): (line 480)
# Processing the call arguments (line 480)
str_1389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 34), 'str', 'moves.urllib_error')
# Processing the call keyword arguments (line 480)
kwargs_1390 = {}
# Getting the type of '_importer' (line 480)
_importer_1387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), '_importer', False)
# Obtaining the member '_get_module' of a type (line 480)
_get_module_1388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 12), _importer_1387, '_get_module')
# Calling _get_module(args, kwargs) (line 480)
_get_module_call_result_1391 = invoke(stypy.reporting.localization.Localization(__file__, 480, 12), _get_module_1388, *[str_1389], **kwargs_1390)

# Getting the type of 'Module_six_moves_urllib'
Module_six_moves_urllib_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Module_six_moves_urllib')
# Setting the type of the member 'error' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Module_six_moves_urllib_1392, 'error', _get_module_call_result_1391)

# Assigning a Call to a Name (line 481):

# Call to _get_module(...): (line 481)
# Processing the call arguments (line 481)
str_1395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 36), 'str', 'moves.urllib_request')
# Processing the call keyword arguments (line 481)
kwargs_1396 = {}
# Getting the type of '_importer' (line 481)
_importer_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 14), '_importer', False)
# Obtaining the member '_get_module' of a type (line 481)
_get_module_1394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 14), _importer_1393, '_get_module')
# Calling _get_module(args, kwargs) (line 481)
_get_module_call_result_1397 = invoke(stypy.reporting.localization.Localization(__file__, 481, 14), _get_module_1394, *[str_1395], **kwargs_1396)

# Getting the type of 'Module_six_moves_urllib'
Module_six_moves_urllib_1398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Module_six_moves_urllib')
# Setting the type of the member 'request' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Module_six_moves_urllib_1398, 'request', _get_module_call_result_1397)

# Assigning a Call to a Name (line 482):

# Call to _get_module(...): (line 482)
# Processing the call arguments (line 482)
str_1401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 37), 'str', 'moves.urllib_response')
# Processing the call keyword arguments (line 482)
kwargs_1402 = {}
# Getting the type of '_importer' (line 482)
_importer_1399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 15), '_importer', False)
# Obtaining the member '_get_module' of a type (line 482)
_get_module_1400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 15), _importer_1399, '_get_module')
# Calling _get_module(args, kwargs) (line 482)
_get_module_call_result_1403 = invoke(stypy.reporting.localization.Localization(__file__, 482, 15), _get_module_1400, *[str_1401], **kwargs_1402)

# Getting the type of 'Module_six_moves_urllib'
Module_six_moves_urllib_1404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Module_six_moves_urllib')
# Setting the type of the member 'response' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Module_six_moves_urllib_1404, 'response', _get_module_call_result_1403)

# Assigning a Call to a Name (line 483):

# Call to _get_module(...): (line 483)
# Processing the call arguments (line 483)
str_1407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 40), 'str', 'moves.urllib_robotparser')
# Processing the call keyword arguments (line 483)
kwargs_1408 = {}
# Getting the type of '_importer' (line 483)
_importer_1405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 18), '_importer', False)
# Obtaining the member '_get_module' of a type (line 483)
_get_module_1406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 18), _importer_1405, '_get_module')
# Calling _get_module(args, kwargs) (line 483)
_get_module_call_result_1409 = invoke(stypy.reporting.localization.Localization(__file__, 483, 18), _get_module_1406, *[str_1407], **kwargs_1408)

# Getting the type of 'Module_six_moves_urllib'
Module_six_moves_urllib_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Module_six_moves_urllib')
# Setting the type of the member 'robotparser' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Module_six_moves_urllib_1410, 'robotparser', _get_module_call_result_1409)

# Call to _add_module(...): (line 488)
# Processing the call arguments (line 488)

# Call to Module_six_moves_urllib(...): (line 488)
# Processing the call arguments (line 488)
# Getting the type of '__name__' (line 488)
name___1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 46), '__name__', False)
str_1415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 57), 'str', '.moves.urllib')
# Applying the binary operator '+' (line 488)
result_add_1416 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 46), '+', name___1414, str_1415)

# Processing the call keyword arguments (line 488)
kwargs_1417 = {}
# Getting the type of 'Module_six_moves_urllib' (line 488)
Module_six_moves_urllib_1413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 22), 'Module_six_moves_urllib', False)
# Calling Module_six_moves_urllib(args, kwargs) (line 488)
Module_six_moves_urllib_call_result_1418 = invoke(stypy.reporting.localization.Localization(__file__, 488, 22), Module_six_moves_urllib_1413, *[result_add_1416], **kwargs_1417)

str_1419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 22), 'str', 'moves.urllib')
# Processing the call keyword arguments (line 488)
kwargs_1420 = {}
# Getting the type of '_importer' (line 488)
_importer_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 0), '_importer', False)
# Obtaining the member '_add_module' of a type (line 488)
_add_module_1412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 0), _importer_1411, '_add_module')
# Calling _add_module(args, kwargs) (line 488)
_add_module_call_result_1421 = invoke(stypy.reporting.localization.Localization(__file__, 488, 0), _add_module_1412, *[Module_six_moves_urllib_call_result_1418, str_1419], **kwargs_1420)


@norecursion
def add_move(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'add_move'
    module_type_store = module_type_store.open_function_context('add_move', 492, 0, False)
    
    # Passed parameters checking function
    add_move.stypy_localization = localization
    add_move.stypy_type_of_self = None
    add_move.stypy_type_store = module_type_store
    add_move.stypy_function_name = 'add_move'
    add_move.stypy_param_names_list = ['move']
    add_move.stypy_varargs_param_name = None
    add_move.stypy_kwargs_param_name = None
    add_move.stypy_call_defaults = defaults
    add_move.stypy_call_varargs = varargs
    add_move.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_move', ['move'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_move', localization, ['move'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_move(...)' code ##################

    str_1422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 4), 'str', 'Add an item to six.moves.')
    
    # Call to setattr(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of '_MovedItems' (line 494)
    _MovedItems_1424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), '_MovedItems', False)
    # Getting the type of 'move' (line 494)
    move_1425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 25), 'move', False)
    # Obtaining the member 'name' of a type (line 494)
    name_1426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 25), move_1425, 'name')
    # Getting the type of 'move' (line 494)
    move_1427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 36), 'move', False)
    # Processing the call keyword arguments (line 494)
    kwargs_1428 = {}
    # Getting the type of 'setattr' (line 494)
    setattr_1423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 494)
    setattr_call_result_1429 = invoke(stypy.reporting.localization.Localization(__file__, 494, 4), setattr_1423, *[_MovedItems_1424, name_1426, move_1427], **kwargs_1428)
    
    
    # ################# End of 'add_move(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_move' in the type store
    # Getting the type of 'stypy_return_type' (line 492)
    stypy_return_type_1430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1430)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_move'
    return stypy_return_type_1430

# Assigning a type to the variable 'add_move' (line 492)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 0), 'add_move', add_move)

@norecursion
def remove_move(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'remove_move'
    module_type_store = module_type_store.open_function_context('remove_move', 497, 0, False)
    
    # Passed parameters checking function
    remove_move.stypy_localization = localization
    remove_move.stypy_type_of_self = None
    remove_move.stypy_type_store = module_type_store
    remove_move.stypy_function_name = 'remove_move'
    remove_move.stypy_param_names_list = ['name']
    remove_move.stypy_varargs_param_name = None
    remove_move.stypy_kwargs_param_name = None
    remove_move.stypy_call_defaults = defaults
    remove_move.stypy_call_varargs = varargs
    remove_move.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'remove_move', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'remove_move', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'remove_move(...)' code ##################

    str_1431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 4), 'str', 'Remove item from six.moves.')
    
    
    # SSA begins for try-except statement (line 499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to delattr(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of '_MovedItems' (line 500)
    _MovedItems_1433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 16), '_MovedItems', False)
    # Getting the type of 'name' (line 500)
    name_1434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 29), 'name', False)
    # Processing the call keyword arguments (line 500)
    kwargs_1435 = {}
    # Getting the type of 'delattr' (line 500)
    delattr_1432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'delattr', False)
    # Calling delattr(args, kwargs) (line 500)
    delattr_call_result_1436 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), delattr_1432, *[_MovedItems_1433, name_1434], **kwargs_1435)
    
    # SSA branch for the except part of a try statement (line 499)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 499)
    module_type_store.open_ssa_branch('except')
    
    
    # SSA begins for try-except statement (line 502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    # Deleting a member
    # Getting the type of 'moves' (line 503)
    moves_1437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 16), 'moves')
    # Obtaining the member '__dict__' of a type (line 503)
    dict___1438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 16), moves_1437, '__dict__')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 503)
    name_1439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 31), 'name')
    # Getting the type of 'moves' (line 503)
    moves_1440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 16), 'moves')
    # Obtaining the member '__dict__' of a type (line 503)
    dict___1441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 16), moves_1440, '__dict__')
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___1442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 16), dict___1441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_1443 = invoke(stypy.reporting.localization.Localization(__file__, 503, 16), getitem___1442, name_1439)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 12), dict___1438, subscript_call_result_1443)
    # SSA branch for the except part of a try statement (line 502)
    # SSA branch for the except 'KeyError' branch of a try statement (line 502)
    module_type_store.open_ssa_branch('except')
    
    # Call to AttributeError(...): (line 505)
    # Processing the call arguments (line 505)
    str_1445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 33), 'str', 'no such move, %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 505)
    tuple_1446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 505)
    # Adding element type (line 505)
    # Getting the type of 'name' (line 505)
    name_1447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 55), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 55), tuple_1446, name_1447)
    
    # Applying the binary operator '%' (line 505)
    result_mod_1448 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 33), '%', str_1445, tuple_1446)
    
    # Processing the call keyword arguments (line 505)
    kwargs_1449 = {}
    # Getting the type of 'AttributeError' (line 505)
    AttributeError_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 18), 'AttributeError', False)
    # Calling AttributeError(args, kwargs) (line 505)
    AttributeError_call_result_1450 = invoke(stypy.reporting.localization.Localization(__file__, 505, 18), AttributeError_1444, *[result_mod_1448], **kwargs_1449)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 505, 12), AttributeError_call_result_1450, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 502)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 499)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'remove_move(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'remove_move' in the type store
    # Getting the type of 'stypy_return_type' (line 497)
    stypy_return_type_1451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1451)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'remove_move'
    return stypy_return_type_1451

# Assigning a type to the variable 'remove_move' (line 497)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 0), 'remove_move', remove_move)

# Getting the type of 'PY3' (line 508)
PY3_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 3), 'PY3')
# Testing the type of an if condition (line 508)
if_condition_1453 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 508, 0), PY3_1452)
# Assigning a type to the variable 'if_condition_1453' (line 508)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 0), 'if_condition_1453', if_condition_1453)
# SSA begins for if statement (line 508)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 509):
str_1454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 17), 'str', '__func__')
# Assigning a type to the variable '_meth_func' (line 509)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), '_meth_func', str_1454)

# Assigning a Str to a Name (line 510):
str_1455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 17), 'str', '__self__')
# Assigning a type to the variable '_meth_self' (line 510)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), '_meth_self', str_1455)

# Assigning a Str to a Name (line 512):
str_1456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 20), 'str', '__closure__')
# Assigning a type to the variable '_func_closure' (line 512)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), '_func_closure', str_1456)

# Assigning a Str to a Name (line 513):
str_1457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 17), 'str', '__code__')
# Assigning a type to the variable '_func_code' (line 513)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), '_func_code', str_1457)

# Assigning a Str to a Name (line 514):
str_1458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 21), 'str', '__defaults__')
# Assigning a type to the variable '_func_defaults' (line 514)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), '_func_defaults', str_1458)

# Assigning a Str to a Name (line 515):
str_1459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 20), 'str', '__globals__')
# Assigning a type to the variable '_func_globals' (line 515)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), '_func_globals', str_1459)
# SSA branch for the else part of an if statement (line 508)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 517):
str_1460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 17), 'str', 'im_func')
# Assigning a type to the variable '_meth_func' (line 517)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), '_meth_func', str_1460)

# Assigning a Str to a Name (line 518):
str_1461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 17), 'str', 'im_self')
# Assigning a type to the variable '_meth_self' (line 518)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), '_meth_self', str_1461)

# Assigning a Str to a Name (line 520):
str_1462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 20), 'str', 'func_closure')
# Assigning a type to the variable '_func_closure' (line 520)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), '_func_closure', str_1462)

# Assigning a Str to a Name (line 521):
str_1463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 17), 'str', 'func_code')
# Assigning a type to the variable '_func_code' (line 521)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), '_func_code', str_1463)

# Assigning a Str to a Name (line 522):
str_1464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 21), 'str', 'func_defaults')
# Assigning a type to the variable '_func_defaults' (line 522)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), '_func_defaults', str_1464)

# Assigning a Str to a Name (line 523):
str_1465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 20), 'str', 'func_globals')
# Assigning a type to the variable '_func_globals' (line 523)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), '_func_globals', str_1465)
# SSA join for if statement (line 508)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 526)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Name to a Name (line 527):
# Getting the type of 'next' (line 527)
next_1466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 23), 'next')
# Assigning a type to the variable 'advance_iterator' (line 527)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'advance_iterator', next_1466)
# SSA branch for the except part of a try statement (line 526)
# SSA branch for the except 'NameError' branch of a try statement (line 526)
module_type_store.open_ssa_branch('except')

@norecursion
def advance_iterator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'advance_iterator'
    module_type_store = module_type_store.open_function_context('advance_iterator', 529, 4, False)
    
    # Passed parameters checking function
    advance_iterator.stypy_localization = localization
    advance_iterator.stypy_type_of_self = None
    advance_iterator.stypy_type_store = module_type_store
    advance_iterator.stypy_function_name = 'advance_iterator'
    advance_iterator.stypy_param_names_list = ['it']
    advance_iterator.stypy_varargs_param_name = None
    advance_iterator.stypy_kwargs_param_name = None
    advance_iterator.stypy_call_defaults = defaults
    advance_iterator.stypy_call_varargs = varargs
    advance_iterator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'advance_iterator', ['it'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'advance_iterator', localization, ['it'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'advance_iterator(...)' code ##################

    
    # Call to next(...): (line 530)
    # Processing the call keyword arguments (line 530)
    kwargs_1469 = {}
    # Getting the type of 'it' (line 530)
    it_1467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 15), 'it', False)
    # Obtaining the member 'next' of a type (line 530)
    next_1468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 15), it_1467, 'next')
    # Calling next(args, kwargs) (line 530)
    next_call_result_1470 = invoke(stypy.reporting.localization.Localization(__file__, 530, 15), next_1468, *[], **kwargs_1469)
    
    # Assigning a type to the variable 'stypy_return_type' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'stypy_return_type', next_call_result_1470)
    
    # ################# End of 'advance_iterator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'advance_iterator' in the type store
    # Getting the type of 'stypy_return_type' (line 529)
    stypy_return_type_1471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1471)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'advance_iterator'
    return stypy_return_type_1471

# Assigning a type to the variable 'advance_iterator' (line 529)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'advance_iterator', advance_iterator)
# SSA join for try-except statement (line 526)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 531):
# Getting the type of 'advance_iterator' (line 531)
advance_iterator_1472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 7), 'advance_iterator')
# Assigning a type to the variable 'next' (line 531)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 0), 'next', advance_iterator_1472)


# SSA begins for try-except statement (line 534)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Name to a Name (line 535):
# Getting the type of 'callable' (line 535)
callable_1473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), 'callable')
# Assigning a type to the variable 'callable' (line 535)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'callable', callable_1473)
# SSA branch for the except part of a try statement (line 534)
# SSA branch for the except 'NameError' branch of a try statement (line 534)
module_type_store.open_ssa_branch('except')

@norecursion
def callable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'callable'
    module_type_store = module_type_store.open_function_context('callable', 537, 4, False)
    
    # Passed parameters checking function
    callable.stypy_localization = localization
    callable.stypy_type_of_self = None
    callable.stypy_type_store = module_type_store
    callable.stypy_function_name = 'callable'
    callable.stypy_param_names_list = ['obj']
    callable.stypy_varargs_param_name = None
    callable.stypy_kwargs_param_name = None
    callable.stypy_call_defaults = defaults
    callable.stypy_call_varargs = varargs
    callable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'callable', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'callable', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'callable(...)' code ##################

    
    # Call to any(...): (line 538)
    # Processing the call arguments (line 538)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 538, 19, True)
    # Calculating comprehension expression
    
    # Call to type(...): (line 538)
    # Processing the call arguments (line 538)
    # Getting the type of 'obj' (line 538)
    obj_1480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 66), 'obj', False)
    # Processing the call keyword arguments (line 538)
    kwargs_1481 = {}
    # Getting the type of 'type' (line 538)
    type_1479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 61), 'type', False)
    # Calling type(args, kwargs) (line 538)
    type_call_result_1482 = invoke(stypy.reporting.localization.Localization(__file__, 538, 61), type_1479, *[obj_1480], **kwargs_1481)
    
    # Obtaining the member '__mro__' of a type (line 538)
    mro___1483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 61), type_call_result_1482, '__mro__')
    comprehension_1484 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 19), mro___1483)
    # Assigning a type to the variable 'klass' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 19), 'klass', comprehension_1484)
    
    str_1475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 19), 'str', '__call__')
    # Getting the type of 'klass' (line 538)
    klass_1476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 33), 'klass', False)
    # Obtaining the member '__dict__' of a type (line 538)
    dict___1477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 33), klass_1476, '__dict__')
    # Applying the binary operator 'in' (line 538)
    result_contains_1478 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 19), 'in', str_1475, dict___1477)
    
    list_1485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 19), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 19), list_1485, result_contains_1478)
    # Processing the call keyword arguments (line 538)
    kwargs_1486 = {}
    # Getting the type of 'any' (line 538)
    any_1474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 15), 'any', False)
    # Calling any(args, kwargs) (line 538)
    any_call_result_1487 = invoke(stypy.reporting.localization.Localization(__file__, 538, 15), any_1474, *[list_1485], **kwargs_1486)
    
    # Assigning a type to the variable 'stypy_return_type' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'stypy_return_type', any_call_result_1487)
    
    # ################# End of 'callable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'callable' in the type store
    # Getting the type of 'stypy_return_type' (line 537)
    stypy_return_type_1488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1488)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'callable'
    return stypy_return_type_1488

# Assigning a type to the variable 'callable' (line 537)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'callable', callable)
# SSA join for try-except statement (line 534)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'PY3' (line 541)
PY3_1489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 3), 'PY3')
# Testing the type of an if condition (line 541)
if_condition_1490 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 541, 0), PY3_1489)
# Assigning a type to the variable 'if_condition_1490' (line 541)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 0), 'if_condition_1490', if_condition_1490)
# SSA begins for if statement (line 541)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def get_unbound_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_unbound_function'
    module_type_store = module_type_store.open_function_context('get_unbound_function', 542, 4, False)
    
    # Passed parameters checking function
    get_unbound_function.stypy_localization = localization
    get_unbound_function.stypy_type_of_self = None
    get_unbound_function.stypy_type_store = module_type_store
    get_unbound_function.stypy_function_name = 'get_unbound_function'
    get_unbound_function.stypy_param_names_list = ['unbound']
    get_unbound_function.stypy_varargs_param_name = None
    get_unbound_function.stypy_kwargs_param_name = None
    get_unbound_function.stypy_call_defaults = defaults
    get_unbound_function.stypy_call_varargs = varargs
    get_unbound_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_unbound_function', ['unbound'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_unbound_function', localization, ['unbound'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_unbound_function(...)' code ##################

    # Getting the type of 'unbound' (line 543)
    unbound_1491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 15), 'unbound')
    # Assigning a type to the variable 'stypy_return_type' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'stypy_return_type', unbound_1491)
    
    # ################# End of 'get_unbound_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_unbound_function' in the type store
    # Getting the type of 'stypy_return_type' (line 542)
    stypy_return_type_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1492)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_unbound_function'
    return stypy_return_type_1492

# Assigning a type to the variable 'get_unbound_function' (line 542)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'get_unbound_function', get_unbound_function)

# Assigning a Attribute to a Name (line 545):
# Getting the type of 'types' (line 545)
types_1493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 26), 'types')
# Obtaining the member 'MethodType' of a type (line 545)
MethodType_1494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 26), types_1493, 'MethodType')
# Assigning a type to the variable 'create_bound_method' (line 545)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'create_bound_method', MethodType_1494)

@norecursion
def create_unbound_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_unbound_method'
    module_type_store = module_type_store.open_function_context('create_unbound_method', 547, 4, False)
    
    # Passed parameters checking function
    create_unbound_method.stypy_localization = localization
    create_unbound_method.stypy_type_of_self = None
    create_unbound_method.stypy_type_store = module_type_store
    create_unbound_method.stypy_function_name = 'create_unbound_method'
    create_unbound_method.stypy_param_names_list = ['func', 'cls']
    create_unbound_method.stypy_varargs_param_name = None
    create_unbound_method.stypy_kwargs_param_name = None
    create_unbound_method.stypy_call_defaults = defaults
    create_unbound_method.stypy_call_varargs = varargs
    create_unbound_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_unbound_method', ['func', 'cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_unbound_method', localization, ['func', 'cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_unbound_method(...)' code ##################

    # Getting the type of 'func' (line 548)
    func_1495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'func')
    # Assigning a type to the variable 'stypy_return_type' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'stypy_return_type', func_1495)
    
    # ################# End of 'create_unbound_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_unbound_method' in the type store
    # Getting the type of 'stypy_return_type' (line 547)
    stypy_return_type_1496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1496)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_unbound_method'
    return stypy_return_type_1496

# Assigning a type to the variable 'create_unbound_method' (line 547)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'create_unbound_method', create_unbound_method)

# Assigning a Name to a Name (line 550):
# Getting the type of 'object' (line 550)
object_1497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 15), 'object')
# Assigning a type to the variable 'Iterator' (line 550)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'Iterator', object_1497)
# SSA branch for the else part of an if statement (line 541)
module_type_store.open_ssa_branch('else')

@norecursion
def get_unbound_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_unbound_function'
    module_type_store = module_type_store.open_function_context('get_unbound_function', 552, 4, False)
    
    # Passed parameters checking function
    get_unbound_function.stypy_localization = localization
    get_unbound_function.stypy_type_of_self = None
    get_unbound_function.stypy_type_store = module_type_store
    get_unbound_function.stypy_function_name = 'get_unbound_function'
    get_unbound_function.stypy_param_names_list = ['unbound']
    get_unbound_function.stypy_varargs_param_name = None
    get_unbound_function.stypy_kwargs_param_name = None
    get_unbound_function.stypy_call_defaults = defaults
    get_unbound_function.stypy_call_varargs = varargs
    get_unbound_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_unbound_function', ['unbound'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_unbound_function', localization, ['unbound'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_unbound_function(...)' code ##################

    # Getting the type of 'unbound' (line 553)
    unbound_1498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 15), 'unbound')
    # Obtaining the member 'im_func' of a type (line 553)
    im_func_1499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 15), unbound_1498, 'im_func')
    # Assigning a type to the variable 'stypy_return_type' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'stypy_return_type', im_func_1499)
    
    # ################# End of 'get_unbound_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_unbound_function' in the type store
    # Getting the type of 'stypy_return_type' (line 552)
    stypy_return_type_1500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1500)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_unbound_function'
    return stypy_return_type_1500

# Assigning a type to the variable 'get_unbound_function' (line 552)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'get_unbound_function', get_unbound_function)

@norecursion
def create_bound_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_bound_method'
    module_type_store = module_type_store.open_function_context('create_bound_method', 555, 4, False)
    
    # Passed parameters checking function
    create_bound_method.stypy_localization = localization
    create_bound_method.stypy_type_of_self = None
    create_bound_method.stypy_type_store = module_type_store
    create_bound_method.stypy_function_name = 'create_bound_method'
    create_bound_method.stypy_param_names_list = ['func', 'obj']
    create_bound_method.stypy_varargs_param_name = None
    create_bound_method.stypy_kwargs_param_name = None
    create_bound_method.stypy_call_defaults = defaults
    create_bound_method.stypy_call_varargs = varargs
    create_bound_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_bound_method', ['func', 'obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_bound_method', localization, ['func', 'obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_bound_method(...)' code ##################

    
    # Call to MethodType(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'func' (line 556)
    func_1503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 32), 'func', False)
    # Getting the type of 'obj' (line 556)
    obj_1504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 38), 'obj', False)
    # Getting the type of 'obj' (line 556)
    obj_1505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 43), 'obj', False)
    # Obtaining the member '__class__' of a type (line 556)
    class___1506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 43), obj_1505, '__class__')
    # Processing the call keyword arguments (line 556)
    kwargs_1507 = {}
    # Getting the type of 'types' (line 556)
    types_1501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 15), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 556)
    MethodType_1502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 15), types_1501, 'MethodType')
    # Calling MethodType(args, kwargs) (line 556)
    MethodType_call_result_1508 = invoke(stypy.reporting.localization.Localization(__file__, 556, 15), MethodType_1502, *[func_1503, obj_1504, class___1506], **kwargs_1507)
    
    # Assigning a type to the variable 'stypy_return_type' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'stypy_return_type', MethodType_call_result_1508)
    
    # ################# End of 'create_bound_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_bound_method' in the type store
    # Getting the type of 'stypy_return_type' (line 555)
    stypy_return_type_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1509)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_bound_method'
    return stypy_return_type_1509

# Assigning a type to the variable 'create_bound_method' (line 555)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'create_bound_method', create_bound_method)

@norecursion
def create_unbound_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_unbound_method'
    module_type_store = module_type_store.open_function_context('create_unbound_method', 558, 4, False)
    
    # Passed parameters checking function
    create_unbound_method.stypy_localization = localization
    create_unbound_method.stypy_type_of_self = None
    create_unbound_method.stypy_type_store = module_type_store
    create_unbound_method.stypy_function_name = 'create_unbound_method'
    create_unbound_method.stypy_param_names_list = ['func', 'cls']
    create_unbound_method.stypy_varargs_param_name = None
    create_unbound_method.stypy_kwargs_param_name = None
    create_unbound_method.stypy_call_defaults = defaults
    create_unbound_method.stypy_call_varargs = varargs
    create_unbound_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_unbound_method', ['func', 'cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_unbound_method', localization, ['func', 'cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_unbound_method(...)' code ##################

    
    # Call to MethodType(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'func' (line 559)
    func_1512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 32), 'func', False)
    # Getting the type of 'None' (line 559)
    None_1513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 38), 'None', False)
    # Getting the type of 'cls' (line 559)
    cls_1514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 44), 'cls', False)
    # Processing the call keyword arguments (line 559)
    kwargs_1515 = {}
    # Getting the type of 'types' (line 559)
    types_1510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 15), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 559)
    MethodType_1511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 15), types_1510, 'MethodType')
    # Calling MethodType(args, kwargs) (line 559)
    MethodType_call_result_1516 = invoke(stypy.reporting.localization.Localization(__file__, 559, 15), MethodType_1511, *[func_1512, None_1513, cls_1514], **kwargs_1515)
    
    # Assigning a type to the variable 'stypy_return_type' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'stypy_return_type', MethodType_call_result_1516)
    
    # ################# End of 'create_unbound_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_unbound_method' in the type store
    # Getting the type of 'stypy_return_type' (line 558)
    stypy_return_type_1517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1517)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_unbound_method'
    return stypy_return_type_1517

# Assigning a type to the variable 'create_unbound_method' (line 558)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'create_unbound_method', create_unbound_method)
# Declaration of the 'Iterator' class

class Iterator(object, ):

    @norecursion
    def next(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'next'
        module_type_store = module_type_store.open_function_context('next', 563, 8, False)
        # Assigning a type to the variable 'self' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        Iterator.next.__dict__.__setitem__('stypy_localization', localization)
        Iterator.next.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Iterator.next.__dict__.__setitem__('stypy_type_store', module_type_store)
        Iterator.next.__dict__.__setitem__('stypy_function_name', 'Iterator.next')
        Iterator.next.__dict__.__setitem__('stypy_param_names_list', [])
        Iterator.next.__dict__.__setitem__('stypy_varargs_param_name', None)
        Iterator.next.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Iterator.next.__dict__.__setitem__('stypy_call_defaults', defaults)
        Iterator.next.__dict__.__setitem__('stypy_call_varargs', varargs)
        Iterator.next.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Iterator.next.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Iterator.next', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __next__(...): (line 564)
        # Processing the call arguments (line 564)
        # Getting the type of 'self' (line 564)
        self_1523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 39), 'self', False)
        # Processing the call keyword arguments (line 564)
        kwargs_1524 = {}
        
        # Call to type(...): (line 564)
        # Processing the call arguments (line 564)
        # Getting the type of 'self' (line 564)
        self_1519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 24), 'self', False)
        # Processing the call keyword arguments (line 564)
        kwargs_1520 = {}
        # Getting the type of 'type' (line 564)
        type_1518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 19), 'type', False)
        # Calling type(args, kwargs) (line 564)
        type_call_result_1521 = invoke(stypy.reporting.localization.Localization(__file__, 564, 19), type_1518, *[self_1519], **kwargs_1520)
        
        # Obtaining the member '__next__' of a type (line 564)
        next___1522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 19), type_call_result_1521, '__next__')
        # Calling __next__(args, kwargs) (line 564)
        next___call_result_1525 = invoke(stypy.reporting.localization.Localization(__file__, 564, 19), next___1522, *[self_1523], **kwargs_1524)
        
        # Assigning a type to the variable 'stypy_return_type' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'stypy_return_type', next___call_result_1525)
        
        # ################# End of 'next(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'next' in the type store
        # Getting the type of 'stypy_return_type' (line 563)
        stypy_return_type_1526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1526)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'next'
        return stypy_return_type_1526


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 561, 4, False)
        # Assigning a type to the variable 'self' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Iterator.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Iterator' (line 561)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'Iterator', Iterator)

# Assigning a Name to a Name (line 566):
# Getting the type of 'callable' (line 566)
callable_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 15), 'callable')
# Assigning a type to the variable 'callable' (line 566)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'callable', callable_1527)
# SSA join for if statement (line 541)
module_type_store = module_type_store.join_ssa_context()


# Call to _add_doc(...): (line 567)
# Processing the call arguments (line 567)
# Getting the type of 'get_unbound_function' (line 567)
get_unbound_function_1529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 9), 'get_unbound_function', False)
str_1530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 9), 'str', 'Get the function out of a possibly unbound function')
# Processing the call keyword arguments (line 567)
kwargs_1531 = {}
# Getting the type of '_add_doc' (line 567)
_add_doc_1528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 567)
_add_doc_call_result_1532 = invoke(stypy.reporting.localization.Localization(__file__, 567, 0), _add_doc_1528, *[get_unbound_function_1529, str_1530], **kwargs_1531)


# Assigning a Call to a Name (line 571):

# Call to attrgetter(...): (line 571)
# Processing the call arguments (line 571)
# Getting the type of '_meth_func' (line 571)
_meth_func_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 42), '_meth_func', False)
# Processing the call keyword arguments (line 571)
kwargs_1536 = {}
# Getting the type of 'operator' (line 571)
operator_1533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 22), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 571)
attrgetter_1534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 22), operator_1533, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 571)
attrgetter_call_result_1537 = invoke(stypy.reporting.localization.Localization(__file__, 571, 22), attrgetter_1534, *[_meth_func_1535], **kwargs_1536)

# Assigning a type to the variable 'get_method_function' (line 571)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 0), 'get_method_function', attrgetter_call_result_1537)

# Assigning a Call to a Name (line 572):

# Call to attrgetter(...): (line 572)
# Processing the call arguments (line 572)
# Getting the type of '_meth_self' (line 572)
_meth_self_1540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 38), '_meth_self', False)
# Processing the call keyword arguments (line 572)
kwargs_1541 = {}
# Getting the type of 'operator' (line 572)
operator_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 18), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 572)
attrgetter_1539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 18), operator_1538, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 572)
attrgetter_call_result_1542 = invoke(stypy.reporting.localization.Localization(__file__, 572, 18), attrgetter_1539, *[_meth_self_1540], **kwargs_1541)

# Assigning a type to the variable 'get_method_self' (line 572)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 0), 'get_method_self', attrgetter_call_result_1542)

# Assigning a Call to a Name (line 573):

# Call to attrgetter(...): (line 573)
# Processing the call arguments (line 573)
# Getting the type of '_func_closure' (line 573)
_func_closure_1545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 43), '_func_closure', False)
# Processing the call keyword arguments (line 573)
kwargs_1546 = {}
# Getting the type of 'operator' (line 573)
operator_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 23), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 573)
attrgetter_1544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 23), operator_1543, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 573)
attrgetter_call_result_1547 = invoke(stypy.reporting.localization.Localization(__file__, 573, 23), attrgetter_1544, *[_func_closure_1545], **kwargs_1546)

# Assigning a type to the variable 'get_function_closure' (line 573)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 0), 'get_function_closure', attrgetter_call_result_1547)

# Assigning a Call to a Name (line 574):

# Call to attrgetter(...): (line 574)
# Processing the call arguments (line 574)
# Getting the type of '_func_code' (line 574)
_func_code_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 40), '_func_code', False)
# Processing the call keyword arguments (line 574)
kwargs_1551 = {}
# Getting the type of 'operator' (line 574)
operator_1548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 20), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 574)
attrgetter_1549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 20), operator_1548, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 574)
attrgetter_call_result_1552 = invoke(stypy.reporting.localization.Localization(__file__, 574, 20), attrgetter_1549, *[_func_code_1550], **kwargs_1551)

# Assigning a type to the variable 'get_function_code' (line 574)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 0), 'get_function_code', attrgetter_call_result_1552)

# Assigning a Call to a Name (line 575):

# Call to attrgetter(...): (line 575)
# Processing the call arguments (line 575)
# Getting the type of '_func_defaults' (line 575)
_func_defaults_1555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 44), '_func_defaults', False)
# Processing the call keyword arguments (line 575)
kwargs_1556 = {}
# Getting the type of 'operator' (line 575)
operator_1553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 24), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 575)
attrgetter_1554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 24), operator_1553, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 575)
attrgetter_call_result_1557 = invoke(stypy.reporting.localization.Localization(__file__, 575, 24), attrgetter_1554, *[_func_defaults_1555], **kwargs_1556)

# Assigning a type to the variable 'get_function_defaults' (line 575)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 0), 'get_function_defaults', attrgetter_call_result_1557)

# Assigning a Call to a Name (line 576):

# Call to attrgetter(...): (line 576)
# Processing the call arguments (line 576)
# Getting the type of '_func_globals' (line 576)
_func_globals_1560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 43), '_func_globals', False)
# Processing the call keyword arguments (line 576)
kwargs_1561 = {}
# Getting the type of 'operator' (line 576)
operator_1558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 23), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 576)
attrgetter_1559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 23), operator_1558, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 576)
attrgetter_call_result_1562 = invoke(stypy.reporting.localization.Localization(__file__, 576, 23), attrgetter_1559, *[_func_globals_1560], **kwargs_1561)

# Assigning a type to the variable 'get_function_globals' (line 576)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 0), 'get_function_globals', attrgetter_call_result_1562)

# Getting the type of 'PY3' (line 579)
PY3_1563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 3), 'PY3')
# Testing the type of an if condition (line 579)
if_condition_1564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 579, 0), PY3_1563)
# Assigning a type to the variable 'if_condition_1564' (line 579)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 0), 'if_condition_1564', if_condition_1564)
# SSA begins for if statement (line 579)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def iterkeys(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iterkeys'
    module_type_store = module_type_store.open_function_context('iterkeys', 580, 4, False)
    
    # Passed parameters checking function
    iterkeys.stypy_localization = localization
    iterkeys.stypy_type_of_self = None
    iterkeys.stypy_type_store = module_type_store
    iterkeys.stypy_function_name = 'iterkeys'
    iterkeys.stypy_param_names_list = ['d']
    iterkeys.stypy_varargs_param_name = None
    iterkeys.stypy_kwargs_param_name = 'kw'
    iterkeys.stypy_call_defaults = defaults
    iterkeys.stypy_call_varargs = varargs
    iterkeys.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iterkeys', ['d'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iterkeys', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iterkeys(...)' code ##################

    
    # Call to iter(...): (line 581)
    # Processing the call arguments (line 581)
    
    # Call to keys(...): (line 581)
    # Processing the call keyword arguments (line 581)
    # Getting the type of 'kw' (line 581)
    kw_1568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 29), 'kw', False)
    kwargs_1569 = {'kw_1568': kw_1568}
    # Getting the type of 'd' (line 581)
    d_1566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 20), 'd', False)
    # Obtaining the member 'keys' of a type (line 581)
    keys_1567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 20), d_1566, 'keys')
    # Calling keys(args, kwargs) (line 581)
    keys_call_result_1570 = invoke(stypy.reporting.localization.Localization(__file__, 581, 20), keys_1567, *[], **kwargs_1569)
    
    # Processing the call keyword arguments (line 581)
    kwargs_1571 = {}
    # Getting the type of 'iter' (line 581)
    iter_1565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), 'iter', False)
    # Calling iter(args, kwargs) (line 581)
    iter_call_result_1572 = invoke(stypy.reporting.localization.Localization(__file__, 581, 15), iter_1565, *[keys_call_result_1570], **kwargs_1571)
    
    # Assigning a type to the variable 'stypy_return_type' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'stypy_return_type', iter_call_result_1572)
    
    # ################# End of 'iterkeys(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iterkeys' in the type store
    # Getting the type of 'stypy_return_type' (line 580)
    stypy_return_type_1573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1573)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iterkeys'
    return stypy_return_type_1573

# Assigning a type to the variable 'iterkeys' (line 580)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'iterkeys', iterkeys)

@norecursion
def itervalues(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'itervalues'
    module_type_store = module_type_store.open_function_context('itervalues', 583, 4, False)
    
    # Passed parameters checking function
    itervalues.stypy_localization = localization
    itervalues.stypy_type_of_self = None
    itervalues.stypy_type_store = module_type_store
    itervalues.stypy_function_name = 'itervalues'
    itervalues.stypy_param_names_list = ['d']
    itervalues.stypy_varargs_param_name = None
    itervalues.stypy_kwargs_param_name = 'kw'
    itervalues.stypy_call_defaults = defaults
    itervalues.stypy_call_varargs = varargs
    itervalues.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'itervalues', ['d'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'itervalues', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'itervalues(...)' code ##################

    
    # Call to iter(...): (line 584)
    # Processing the call arguments (line 584)
    
    # Call to values(...): (line 584)
    # Processing the call keyword arguments (line 584)
    # Getting the type of 'kw' (line 584)
    kw_1577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 31), 'kw', False)
    kwargs_1578 = {'kw_1577': kw_1577}
    # Getting the type of 'd' (line 584)
    d_1575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'd', False)
    # Obtaining the member 'values' of a type (line 584)
    values_1576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 20), d_1575, 'values')
    # Calling values(args, kwargs) (line 584)
    values_call_result_1579 = invoke(stypy.reporting.localization.Localization(__file__, 584, 20), values_1576, *[], **kwargs_1578)
    
    # Processing the call keyword arguments (line 584)
    kwargs_1580 = {}
    # Getting the type of 'iter' (line 584)
    iter_1574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 15), 'iter', False)
    # Calling iter(args, kwargs) (line 584)
    iter_call_result_1581 = invoke(stypy.reporting.localization.Localization(__file__, 584, 15), iter_1574, *[values_call_result_1579], **kwargs_1580)
    
    # Assigning a type to the variable 'stypy_return_type' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'stypy_return_type', iter_call_result_1581)
    
    # ################# End of 'itervalues(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'itervalues' in the type store
    # Getting the type of 'stypy_return_type' (line 583)
    stypy_return_type_1582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1582)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'itervalues'
    return stypy_return_type_1582

# Assigning a type to the variable 'itervalues' (line 583)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'itervalues', itervalues)

@norecursion
def iteritems(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iteritems'
    module_type_store = module_type_store.open_function_context('iteritems', 586, 4, False)
    
    # Passed parameters checking function
    iteritems.stypy_localization = localization
    iteritems.stypy_type_of_self = None
    iteritems.stypy_type_store = module_type_store
    iteritems.stypy_function_name = 'iteritems'
    iteritems.stypy_param_names_list = ['d']
    iteritems.stypy_varargs_param_name = None
    iteritems.stypy_kwargs_param_name = 'kw'
    iteritems.stypy_call_defaults = defaults
    iteritems.stypy_call_varargs = varargs
    iteritems.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iteritems', ['d'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iteritems', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iteritems(...)' code ##################

    
    # Call to iter(...): (line 587)
    # Processing the call arguments (line 587)
    
    # Call to items(...): (line 587)
    # Processing the call keyword arguments (line 587)
    # Getting the type of 'kw' (line 587)
    kw_1586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'kw', False)
    kwargs_1587 = {'kw_1586': kw_1586}
    # Getting the type of 'd' (line 587)
    d_1584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 20), 'd', False)
    # Obtaining the member 'items' of a type (line 587)
    items_1585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 20), d_1584, 'items')
    # Calling items(args, kwargs) (line 587)
    items_call_result_1588 = invoke(stypy.reporting.localization.Localization(__file__, 587, 20), items_1585, *[], **kwargs_1587)
    
    # Processing the call keyword arguments (line 587)
    kwargs_1589 = {}
    # Getting the type of 'iter' (line 587)
    iter_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 15), 'iter', False)
    # Calling iter(args, kwargs) (line 587)
    iter_call_result_1590 = invoke(stypy.reporting.localization.Localization(__file__, 587, 15), iter_1583, *[items_call_result_1588], **kwargs_1589)
    
    # Assigning a type to the variable 'stypy_return_type' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'stypy_return_type', iter_call_result_1590)
    
    # ################# End of 'iteritems(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iteritems' in the type store
    # Getting the type of 'stypy_return_type' (line 586)
    stypy_return_type_1591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1591)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iteritems'
    return stypy_return_type_1591

# Assigning a type to the variable 'iteritems' (line 586)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'iteritems', iteritems)

@norecursion
def iterlists(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iterlists'
    module_type_store = module_type_store.open_function_context('iterlists', 589, 4, False)
    
    # Passed parameters checking function
    iterlists.stypy_localization = localization
    iterlists.stypy_type_of_self = None
    iterlists.stypy_type_store = module_type_store
    iterlists.stypy_function_name = 'iterlists'
    iterlists.stypy_param_names_list = ['d']
    iterlists.stypy_varargs_param_name = None
    iterlists.stypy_kwargs_param_name = 'kw'
    iterlists.stypy_call_defaults = defaults
    iterlists.stypy_call_varargs = varargs
    iterlists.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iterlists', ['d'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iterlists', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iterlists(...)' code ##################

    
    # Call to iter(...): (line 590)
    # Processing the call arguments (line 590)
    
    # Call to lists(...): (line 590)
    # Processing the call keyword arguments (line 590)
    # Getting the type of 'kw' (line 590)
    kw_1595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 30), 'kw', False)
    kwargs_1596 = {'kw_1595': kw_1595}
    # Getting the type of 'd' (line 590)
    d_1593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 20), 'd', False)
    # Obtaining the member 'lists' of a type (line 590)
    lists_1594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 20), d_1593, 'lists')
    # Calling lists(args, kwargs) (line 590)
    lists_call_result_1597 = invoke(stypy.reporting.localization.Localization(__file__, 590, 20), lists_1594, *[], **kwargs_1596)
    
    # Processing the call keyword arguments (line 590)
    kwargs_1598 = {}
    # Getting the type of 'iter' (line 590)
    iter_1592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 15), 'iter', False)
    # Calling iter(args, kwargs) (line 590)
    iter_call_result_1599 = invoke(stypy.reporting.localization.Localization(__file__, 590, 15), iter_1592, *[lists_call_result_1597], **kwargs_1598)
    
    # Assigning a type to the variable 'stypy_return_type' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'stypy_return_type', iter_call_result_1599)
    
    # ################# End of 'iterlists(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iterlists' in the type store
    # Getting the type of 'stypy_return_type' (line 589)
    stypy_return_type_1600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1600)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iterlists'
    return stypy_return_type_1600

# Assigning a type to the variable 'iterlists' (line 589)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'iterlists', iterlists)

# Assigning a Call to a Name (line 592):

# Call to methodcaller(...): (line 592)
# Processing the call arguments (line 592)
str_1603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 37), 'str', 'keys')
# Processing the call keyword arguments (line 592)
kwargs_1604 = {}
# Getting the type of 'operator' (line 592)
operator_1601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 15), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 592)
methodcaller_1602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 15), operator_1601, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 592)
methodcaller_call_result_1605 = invoke(stypy.reporting.localization.Localization(__file__, 592, 15), methodcaller_1602, *[str_1603], **kwargs_1604)

# Assigning a type to the variable 'viewkeys' (line 592)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'viewkeys', methodcaller_call_result_1605)

# Assigning a Call to a Name (line 594):

# Call to methodcaller(...): (line 594)
# Processing the call arguments (line 594)
str_1608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 39), 'str', 'values')
# Processing the call keyword arguments (line 594)
kwargs_1609 = {}
# Getting the type of 'operator' (line 594)
operator_1606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 17), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 594)
methodcaller_1607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 17), operator_1606, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 594)
methodcaller_call_result_1610 = invoke(stypy.reporting.localization.Localization(__file__, 594, 17), methodcaller_1607, *[str_1608], **kwargs_1609)

# Assigning a type to the variable 'viewvalues' (line 594)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'viewvalues', methodcaller_call_result_1610)

# Assigning a Call to a Name (line 596):

# Call to methodcaller(...): (line 596)
# Processing the call arguments (line 596)
str_1613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 38), 'str', 'items')
# Processing the call keyword arguments (line 596)
kwargs_1614 = {}
# Getting the type of 'operator' (line 596)
operator_1611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 596)
methodcaller_1612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 16), operator_1611, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 596)
methodcaller_call_result_1615 = invoke(stypy.reporting.localization.Localization(__file__, 596, 16), methodcaller_1612, *[str_1613], **kwargs_1614)

# Assigning a type to the variable 'viewitems' (line 596)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'viewitems', methodcaller_call_result_1615)
# SSA branch for the else part of an if statement (line 579)
module_type_store.open_ssa_branch('else')

@norecursion
def iterkeys(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iterkeys'
    module_type_store = module_type_store.open_function_context('iterkeys', 598, 4, False)
    
    # Passed parameters checking function
    iterkeys.stypy_localization = localization
    iterkeys.stypy_type_of_self = None
    iterkeys.stypy_type_store = module_type_store
    iterkeys.stypy_function_name = 'iterkeys'
    iterkeys.stypy_param_names_list = ['d']
    iterkeys.stypy_varargs_param_name = None
    iterkeys.stypy_kwargs_param_name = 'kw'
    iterkeys.stypy_call_defaults = defaults
    iterkeys.stypy_call_varargs = varargs
    iterkeys.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iterkeys', ['d'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iterkeys', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iterkeys(...)' code ##################

    
    # Call to iterkeys(...): (line 599)
    # Processing the call keyword arguments (line 599)
    # Getting the type of 'kw' (line 599)
    kw_1618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 28), 'kw', False)
    kwargs_1619 = {'kw_1618': kw_1618}
    # Getting the type of 'd' (line 599)
    d_1616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 15), 'd', False)
    # Obtaining the member 'iterkeys' of a type (line 599)
    iterkeys_1617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 15), d_1616, 'iterkeys')
    # Calling iterkeys(args, kwargs) (line 599)
    iterkeys_call_result_1620 = invoke(stypy.reporting.localization.Localization(__file__, 599, 15), iterkeys_1617, *[], **kwargs_1619)
    
    # Assigning a type to the variable 'stypy_return_type' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'stypy_return_type', iterkeys_call_result_1620)
    
    # ################# End of 'iterkeys(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iterkeys' in the type store
    # Getting the type of 'stypy_return_type' (line 598)
    stypy_return_type_1621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1621)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iterkeys'
    return stypy_return_type_1621

# Assigning a type to the variable 'iterkeys' (line 598)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'iterkeys', iterkeys)

@norecursion
def itervalues(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'itervalues'
    module_type_store = module_type_store.open_function_context('itervalues', 601, 4, False)
    
    # Passed parameters checking function
    itervalues.stypy_localization = localization
    itervalues.stypy_type_of_self = None
    itervalues.stypy_type_store = module_type_store
    itervalues.stypy_function_name = 'itervalues'
    itervalues.stypy_param_names_list = ['d']
    itervalues.stypy_varargs_param_name = None
    itervalues.stypy_kwargs_param_name = 'kw'
    itervalues.stypy_call_defaults = defaults
    itervalues.stypy_call_varargs = varargs
    itervalues.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'itervalues', ['d'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'itervalues', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'itervalues(...)' code ##################

    
    # Call to itervalues(...): (line 602)
    # Processing the call keyword arguments (line 602)
    # Getting the type of 'kw' (line 602)
    kw_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 30), 'kw', False)
    kwargs_1625 = {'kw_1624': kw_1624}
    # Getting the type of 'd' (line 602)
    d_1622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 15), 'd', False)
    # Obtaining the member 'itervalues' of a type (line 602)
    itervalues_1623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 15), d_1622, 'itervalues')
    # Calling itervalues(args, kwargs) (line 602)
    itervalues_call_result_1626 = invoke(stypy.reporting.localization.Localization(__file__, 602, 15), itervalues_1623, *[], **kwargs_1625)
    
    # Assigning a type to the variable 'stypy_return_type' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'stypy_return_type', itervalues_call_result_1626)
    
    # ################# End of 'itervalues(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'itervalues' in the type store
    # Getting the type of 'stypy_return_type' (line 601)
    stypy_return_type_1627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1627)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'itervalues'
    return stypy_return_type_1627

# Assigning a type to the variable 'itervalues' (line 601)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'itervalues', itervalues)

@norecursion
def iteritems(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iteritems'
    module_type_store = module_type_store.open_function_context('iteritems', 604, 4, False)
    
    # Passed parameters checking function
    iteritems.stypy_localization = localization
    iteritems.stypy_type_of_self = None
    iteritems.stypy_type_store = module_type_store
    iteritems.stypy_function_name = 'iteritems'
    iteritems.stypy_param_names_list = ['d']
    iteritems.stypy_varargs_param_name = None
    iteritems.stypy_kwargs_param_name = 'kw'
    iteritems.stypy_call_defaults = defaults
    iteritems.stypy_call_varargs = varargs
    iteritems.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iteritems', ['d'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iteritems', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iteritems(...)' code ##################

    
    # Call to iteritems(...): (line 605)
    # Processing the call keyword arguments (line 605)
    # Getting the type of 'kw' (line 605)
    kw_1630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 29), 'kw', False)
    kwargs_1631 = {'kw_1630': kw_1630}
    # Getting the type of 'd' (line 605)
    d_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 15), 'd', False)
    # Obtaining the member 'iteritems' of a type (line 605)
    iteritems_1629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 15), d_1628, 'iteritems')
    # Calling iteritems(args, kwargs) (line 605)
    iteritems_call_result_1632 = invoke(stypy.reporting.localization.Localization(__file__, 605, 15), iteritems_1629, *[], **kwargs_1631)
    
    # Assigning a type to the variable 'stypy_return_type' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'stypy_return_type', iteritems_call_result_1632)
    
    # ################# End of 'iteritems(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iteritems' in the type store
    # Getting the type of 'stypy_return_type' (line 604)
    stypy_return_type_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1633)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iteritems'
    return stypy_return_type_1633

# Assigning a type to the variable 'iteritems' (line 604)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'iteritems', iteritems)

@norecursion
def iterlists(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iterlists'
    module_type_store = module_type_store.open_function_context('iterlists', 607, 4, False)
    
    # Passed parameters checking function
    iterlists.stypy_localization = localization
    iterlists.stypy_type_of_self = None
    iterlists.stypy_type_store = module_type_store
    iterlists.stypy_function_name = 'iterlists'
    iterlists.stypy_param_names_list = ['d']
    iterlists.stypy_varargs_param_name = None
    iterlists.stypy_kwargs_param_name = 'kw'
    iterlists.stypy_call_defaults = defaults
    iterlists.stypy_call_varargs = varargs
    iterlists.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iterlists', ['d'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iterlists', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iterlists(...)' code ##################

    
    # Call to iterlists(...): (line 608)
    # Processing the call keyword arguments (line 608)
    # Getting the type of 'kw' (line 608)
    kw_1636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 29), 'kw', False)
    kwargs_1637 = {'kw_1636': kw_1636}
    # Getting the type of 'd' (line 608)
    d_1634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 15), 'd', False)
    # Obtaining the member 'iterlists' of a type (line 608)
    iterlists_1635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 15), d_1634, 'iterlists')
    # Calling iterlists(args, kwargs) (line 608)
    iterlists_call_result_1638 = invoke(stypy.reporting.localization.Localization(__file__, 608, 15), iterlists_1635, *[], **kwargs_1637)
    
    # Assigning a type to the variable 'stypy_return_type' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'stypy_return_type', iterlists_call_result_1638)
    
    # ################# End of 'iterlists(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iterlists' in the type store
    # Getting the type of 'stypy_return_type' (line 607)
    stypy_return_type_1639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1639)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iterlists'
    return stypy_return_type_1639

# Assigning a type to the variable 'iterlists' (line 607)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'iterlists', iterlists)

# Assigning a Call to a Name (line 610):

# Call to methodcaller(...): (line 610)
# Processing the call arguments (line 610)
str_1642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 37), 'str', 'viewkeys')
# Processing the call keyword arguments (line 610)
kwargs_1643 = {}
# Getting the type of 'operator' (line 610)
operator_1640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 15), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 610)
methodcaller_1641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 15), operator_1640, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 610)
methodcaller_call_result_1644 = invoke(stypy.reporting.localization.Localization(__file__, 610, 15), methodcaller_1641, *[str_1642], **kwargs_1643)

# Assigning a type to the variable 'viewkeys' (line 610)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'viewkeys', methodcaller_call_result_1644)

# Assigning a Call to a Name (line 612):

# Call to methodcaller(...): (line 612)
# Processing the call arguments (line 612)
str_1647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 39), 'str', 'viewvalues')
# Processing the call keyword arguments (line 612)
kwargs_1648 = {}
# Getting the type of 'operator' (line 612)
operator_1645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 17), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 612)
methodcaller_1646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 17), operator_1645, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 612)
methodcaller_call_result_1649 = invoke(stypy.reporting.localization.Localization(__file__, 612, 17), methodcaller_1646, *[str_1647], **kwargs_1648)

# Assigning a type to the variable 'viewvalues' (line 612)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'viewvalues', methodcaller_call_result_1649)

# Assigning a Call to a Name (line 614):

# Call to methodcaller(...): (line 614)
# Processing the call arguments (line 614)
str_1652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 38), 'str', 'viewitems')
# Processing the call keyword arguments (line 614)
kwargs_1653 = {}
# Getting the type of 'operator' (line 614)
operator_1650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 16), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 614)
methodcaller_1651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 16), operator_1650, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 614)
methodcaller_call_result_1654 = invoke(stypy.reporting.localization.Localization(__file__, 614, 16), methodcaller_1651, *[str_1652], **kwargs_1653)

# Assigning a type to the variable 'viewitems' (line 614)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'viewitems', methodcaller_call_result_1654)
# SSA join for if statement (line 579)
module_type_store = module_type_store.join_ssa_context()


# Call to _add_doc(...): (line 616)
# Processing the call arguments (line 616)
# Getting the type of 'iterkeys' (line 616)
iterkeys_1656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 9), 'iterkeys', False)
str_1657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 19), 'str', 'Return an iterator over the keys of a dictionary.')
# Processing the call keyword arguments (line 616)
kwargs_1658 = {}
# Getting the type of '_add_doc' (line 616)
_add_doc_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 616)
_add_doc_call_result_1659 = invoke(stypy.reporting.localization.Localization(__file__, 616, 0), _add_doc_1655, *[iterkeys_1656, str_1657], **kwargs_1658)


# Call to _add_doc(...): (line 617)
# Processing the call arguments (line 617)
# Getting the type of 'itervalues' (line 617)
itervalues_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 9), 'itervalues', False)
str_1662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 21), 'str', 'Return an iterator over the values of a dictionary.')
# Processing the call keyword arguments (line 617)
kwargs_1663 = {}
# Getting the type of '_add_doc' (line 617)
_add_doc_1660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 617)
_add_doc_call_result_1664 = invoke(stypy.reporting.localization.Localization(__file__, 617, 0), _add_doc_1660, *[itervalues_1661, str_1662], **kwargs_1663)


# Call to _add_doc(...): (line 618)
# Processing the call arguments (line 618)
# Getting the type of 'iteritems' (line 618)
iteritems_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 9), 'iteritems', False)
str_1667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 9), 'str', 'Return an iterator over the (key, value) pairs of a dictionary.')
# Processing the call keyword arguments (line 618)
kwargs_1668 = {}
# Getting the type of '_add_doc' (line 618)
_add_doc_1665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 618)
_add_doc_call_result_1669 = invoke(stypy.reporting.localization.Localization(__file__, 618, 0), _add_doc_1665, *[iteritems_1666, str_1667], **kwargs_1668)


# Call to _add_doc(...): (line 620)
# Processing the call arguments (line 620)
# Getting the type of 'iterlists' (line 620)
iterlists_1671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 9), 'iterlists', False)
str_1672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 9), 'str', 'Return an iterator over the (key, [values]) pairs of a dictionary.')
# Processing the call keyword arguments (line 620)
kwargs_1673 = {}
# Getting the type of '_add_doc' (line 620)
_add_doc_1670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 620)
_add_doc_call_result_1674 = invoke(stypy.reporting.localization.Localization(__file__, 620, 0), _add_doc_1670, *[iterlists_1671, str_1672], **kwargs_1673)


# Getting the type of 'PY3' (line 624)
PY3_1675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 3), 'PY3')
# Testing the type of an if condition (line 624)
if_condition_1676 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 624, 0), PY3_1675)
# Assigning a type to the variable 'if_condition_1676' (line 624)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 0), 'if_condition_1676', if_condition_1676)
# SSA begins for if statement (line 624)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def b(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'b'
    module_type_store = module_type_store.open_function_context('b', 625, 4, False)
    
    # Passed parameters checking function
    b.stypy_localization = localization
    b.stypy_type_of_self = None
    b.stypy_type_store = module_type_store
    b.stypy_function_name = 'b'
    b.stypy_param_names_list = ['s']
    b.stypy_varargs_param_name = None
    b.stypy_kwargs_param_name = None
    b.stypy_call_defaults = defaults
    b.stypy_call_varargs = varargs
    b.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'b', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'b', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'b(...)' code ##################

    
    # Call to encode(...): (line 626)
    # Processing the call arguments (line 626)
    str_1679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 24), 'str', 'latin-1')
    # Processing the call keyword arguments (line 626)
    kwargs_1680 = {}
    # Getting the type of 's' (line 626)
    s_1677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 15), 's', False)
    # Obtaining the member 'encode' of a type (line 626)
    encode_1678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 15), s_1677, 'encode')
    # Calling encode(args, kwargs) (line 626)
    encode_call_result_1681 = invoke(stypy.reporting.localization.Localization(__file__, 626, 15), encode_1678, *[str_1679], **kwargs_1680)
    
    # Assigning a type to the variable 'stypy_return_type' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'stypy_return_type', encode_call_result_1681)
    
    # ################# End of 'b(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'b' in the type store
    # Getting the type of 'stypy_return_type' (line 625)
    stypy_return_type_1682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1682)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'b'
    return stypy_return_type_1682

# Assigning a type to the variable 'b' (line 625)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'b', b)

@norecursion
def u(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'u'
    module_type_store = module_type_store.open_function_context('u', 628, 4, False)
    
    # Passed parameters checking function
    u.stypy_localization = localization
    u.stypy_type_of_self = None
    u.stypy_type_store = module_type_store
    u.stypy_function_name = 'u'
    u.stypy_param_names_list = ['s']
    u.stypy_varargs_param_name = None
    u.stypy_kwargs_param_name = None
    u.stypy_call_defaults = defaults
    u.stypy_call_varargs = varargs
    u.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'u', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'u', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'u(...)' code ##################

    # Getting the type of 's' (line 629)
    s_1683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 15), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'stypy_return_type', s_1683)
    
    # ################# End of 'u(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'u' in the type store
    # Getting the type of 'stypy_return_type' (line 628)
    stypy_return_type_1684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1684)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'u'
    return stypy_return_type_1684

# Assigning a type to the variable 'u' (line 628)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'u', u)

# Assigning a Name to a Name (line 630):
# Getting the type of 'chr' (line 630)
chr_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 13), 'chr')
# Assigning a type to the variable 'unichr' (line 630)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'unichr', chr_1685)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 631, 4))

# 'import struct' statement (line 631)
import struct

import_module(stypy.reporting.localization.Localization(__file__, 631, 4), 'struct', struct, module_type_store)


# Assigning a Attribute to a Name (line 632):

# Call to Struct(...): (line 632)
# Processing the call arguments (line 632)
str_1688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 29), 'str', '>B')
# Processing the call keyword arguments (line 632)
kwargs_1689 = {}
# Getting the type of 'struct' (line 632)
struct_1686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 15), 'struct', False)
# Obtaining the member 'Struct' of a type (line 632)
Struct_1687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 15), struct_1686, 'Struct')
# Calling Struct(args, kwargs) (line 632)
Struct_call_result_1690 = invoke(stypy.reporting.localization.Localization(__file__, 632, 15), Struct_1687, *[str_1688], **kwargs_1689)

# Obtaining the member 'pack' of a type (line 632)
pack_1691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 15), Struct_call_result_1690, 'pack')
# Assigning a type to the variable 'int2byte' (line 632)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 4), 'int2byte', pack_1691)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 633, 4), module_type_store, 'struct')

# Assigning a Call to a Name (line 634):

# Call to itemgetter(...): (line 634)
# Processing the call arguments (line 634)
int_1694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 35), 'int')
# Processing the call keyword arguments (line 634)
kwargs_1695 = {}
# Getting the type of 'operator' (line 634)
operator_1692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 15), 'operator', False)
# Obtaining the member 'itemgetter' of a type (line 634)
itemgetter_1693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 15), operator_1692, 'itemgetter')
# Calling itemgetter(args, kwargs) (line 634)
itemgetter_call_result_1696 = invoke(stypy.reporting.localization.Localization(__file__, 634, 15), itemgetter_1693, *[int_1694], **kwargs_1695)

# Assigning a type to the variable 'byte2int' (line 634)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 4), 'byte2int', itemgetter_call_result_1696)

# Assigning a Attribute to a Name (line 635):
# Getting the type of 'operator' (line 635)
operator_1697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 17), 'operator')
# Obtaining the member 'getitem' of a type (line 635)
getitem_1698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 17), operator_1697, 'getitem')
# Assigning a type to the variable 'indexbytes' (line 635)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'indexbytes', getitem_1698)

# Assigning a Name to a Name (line 636):
# Getting the type of 'iter' (line 636)
iter_1699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'iter')
# Assigning a type to the variable 'iterbytes' (line 636)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'iterbytes', iter_1699)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 637, 4))

# 'import io' statement (line 637)
import io

import_module(stypy.reporting.localization.Localization(__file__, 637, 4), 'io', io, module_type_store)


# Assigning a Attribute to a Name (line 638):
# Getting the type of 'io' (line 638)
io_1700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 15), 'io')
# Obtaining the member 'StringIO' of a type (line 638)
StringIO_1701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 15), io_1700, 'StringIO')
# Assigning a type to the variable 'StringIO' (line 638)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'StringIO', StringIO_1701)

# Assigning a Attribute to a Name (line 639):
# Getting the type of 'io' (line 639)
io_1702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 14), 'io')
# Obtaining the member 'BytesIO' of a type (line 639)
BytesIO_1703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 14), io_1702, 'BytesIO')
# Assigning a type to the variable 'BytesIO' (line 639)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'BytesIO', BytesIO_1703)

# Assigning a Str to a Name (line 640):
str_1704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 24), 'str', 'assertCountEqual')
# Assigning a type to the variable '_assertCountEqual' (line 640)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), '_assertCountEqual', str_1704)



# Obtaining the type of the subscript
int_1705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 24), 'int')
# Getting the type of 'sys' (line 641)
sys_1706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 7), 'sys')
# Obtaining the member 'version_info' of a type (line 641)
version_info_1707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 7), sys_1706, 'version_info')
# Obtaining the member '__getitem__' of a type (line 641)
getitem___1708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 7), version_info_1707, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 641)
subscript_call_result_1709 = invoke(stypy.reporting.localization.Localization(__file__, 641, 7), getitem___1708, int_1705)

int_1710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 30), 'int')
# Applying the binary operator '<=' (line 641)
result_le_1711 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 7), '<=', subscript_call_result_1709, int_1710)

# Testing the type of an if condition (line 641)
if_condition_1712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 641, 4), result_le_1711)
# Assigning a type to the variable 'if_condition_1712' (line 641)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'if_condition_1712', if_condition_1712)
# SSA begins for if statement (line 641)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 642):
str_1713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 29), 'str', 'assertRaisesRegexp')
# Assigning a type to the variable '_assertRaisesRegex' (line 642)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), '_assertRaisesRegex', str_1713)

# Assigning a Str to a Name (line 643):
str_1714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 23), 'str', 'assertRegexpMatches')
# Assigning a type to the variable '_assertRegex' (line 643)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), '_assertRegex', str_1714)
# SSA branch for the else part of an if statement (line 641)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 645):
str_1715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 29), 'str', 'assertRaisesRegex')
# Assigning a type to the variable '_assertRaisesRegex' (line 645)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), '_assertRaisesRegex', str_1715)

# Assigning a Str to a Name (line 646):
str_1716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 23), 'str', 'assertRegex')
# Assigning a type to the variable '_assertRegex' (line 646)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), '_assertRegex', str_1716)
# SSA join for if statement (line 641)
module_type_store = module_type_store.join_ssa_context()

# SSA branch for the else part of an if statement (line 624)
module_type_store.open_ssa_branch('else')

@norecursion
def b(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'b'
    module_type_store = module_type_store.open_function_context('b', 648, 4, False)
    
    # Passed parameters checking function
    b.stypy_localization = localization
    b.stypy_type_of_self = None
    b.stypy_type_store = module_type_store
    b.stypy_function_name = 'b'
    b.stypy_param_names_list = ['s']
    b.stypy_varargs_param_name = None
    b.stypy_kwargs_param_name = None
    b.stypy_call_defaults = defaults
    b.stypy_call_varargs = varargs
    b.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'b', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'b', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'b(...)' code ##################

    # Getting the type of 's' (line 649)
    s_1717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 15), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'stypy_return_type', s_1717)
    
    # ################# End of 'b(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'b' in the type store
    # Getting the type of 'stypy_return_type' (line 648)
    stypy_return_type_1718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1718)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'b'
    return stypy_return_type_1718

# Assigning a type to the variable 'b' (line 648)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'b', b)

@norecursion
def u(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'u'
    module_type_store = module_type_store.open_function_context('u', 652, 4, False)
    
    # Passed parameters checking function
    u.stypy_localization = localization
    u.stypy_type_of_self = None
    u.stypy_type_store = module_type_store
    u.stypy_function_name = 'u'
    u.stypy_param_names_list = ['s']
    u.stypy_varargs_param_name = None
    u.stypy_kwargs_param_name = None
    u.stypy_call_defaults = defaults
    u.stypy_call_varargs = varargs
    u.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'u', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'u', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'u(...)' code ##################

    
    # Call to unicode(...): (line 653)
    # Processing the call arguments (line 653)
    
    # Call to replace(...): (line 653)
    # Processing the call arguments (line 653)
    str_1722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 33), 'str', '\\\\')
    str_1723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 40), 'str', '\\\\\\\\')
    # Processing the call keyword arguments (line 653)
    kwargs_1724 = {}
    # Getting the type of 's' (line 653)
    s_1720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 23), 's', False)
    # Obtaining the member 'replace' of a type (line 653)
    replace_1721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 23), s_1720, 'replace')
    # Calling replace(args, kwargs) (line 653)
    replace_call_result_1725 = invoke(stypy.reporting.localization.Localization(__file__, 653, 23), replace_1721, *[str_1722, str_1723], **kwargs_1724)
    
    str_1726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 50), 'str', 'unicode_escape')
    # Processing the call keyword arguments (line 653)
    kwargs_1727 = {}
    # Getting the type of 'unicode' (line 653)
    unicode_1719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 15), 'unicode', False)
    # Calling unicode(args, kwargs) (line 653)
    unicode_call_result_1728 = invoke(stypy.reporting.localization.Localization(__file__, 653, 15), unicode_1719, *[replace_call_result_1725, str_1726], **kwargs_1727)
    
    # Assigning a type to the variable 'stypy_return_type' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'stypy_return_type', unicode_call_result_1728)
    
    # ################# End of 'u(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'u' in the type store
    # Getting the type of 'stypy_return_type' (line 652)
    stypy_return_type_1729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1729)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'u'
    return stypy_return_type_1729

# Assigning a type to the variable 'u' (line 652)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'u', u)

# Assigning a Name to a Name (line 654):
# Getting the type of 'unichr' (line 654)
unichr_1730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 13), 'unichr')
# Assigning a type to the variable 'unichr' (line 654)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'unichr', unichr_1730)

# Assigning a Name to a Name (line 655):
# Getting the type of 'chr' (line 655)
chr_1731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'chr')
# Assigning a type to the variable 'int2byte' (line 655)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'int2byte', chr_1731)

@norecursion
def byte2int(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'byte2int'
    module_type_store = module_type_store.open_function_context('byte2int', 657, 4, False)
    
    # Passed parameters checking function
    byte2int.stypy_localization = localization
    byte2int.stypy_type_of_self = None
    byte2int.stypy_type_store = module_type_store
    byte2int.stypy_function_name = 'byte2int'
    byte2int.stypy_param_names_list = ['bs']
    byte2int.stypy_varargs_param_name = None
    byte2int.stypy_kwargs_param_name = None
    byte2int.stypy_call_defaults = defaults
    byte2int.stypy_call_varargs = varargs
    byte2int.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'byte2int', ['bs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'byte2int', localization, ['bs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'byte2int(...)' code ##################

    
    # Call to ord(...): (line 658)
    # Processing the call arguments (line 658)
    
    # Obtaining the type of the subscript
    int_1733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 22), 'int')
    # Getting the type of 'bs' (line 658)
    bs_1734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 19), 'bs', False)
    # Obtaining the member '__getitem__' of a type (line 658)
    getitem___1735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 19), bs_1734, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 658)
    subscript_call_result_1736 = invoke(stypy.reporting.localization.Localization(__file__, 658, 19), getitem___1735, int_1733)
    
    # Processing the call keyword arguments (line 658)
    kwargs_1737 = {}
    # Getting the type of 'ord' (line 658)
    ord_1732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 15), 'ord', False)
    # Calling ord(args, kwargs) (line 658)
    ord_call_result_1738 = invoke(stypy.reporting.localization.Localization(__file__, 658, 15), ord_1732, *[subscript_call_result_1736], **kwargs_1737)
    
    # Assigning a type to the variable 'stypy_return_type' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'stypy_return_type', ord_call_result_1738)
    
    # ################# End of 'byte2int(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'byte2int' in the type store
    # Getting the type of 'stypy_return_type' (line 657)
    stypy_return_type_1739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1739)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'byte2int'
    return stypy_return_type_1739

# Assigning a type to the variable 'byte2int' (line 657)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'byte2int', byte2int)

@norecursion
def indexbytes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'indexbytes'
    module_type_store = module_type_store.open_function_context('indexbytes', 660, 4, False)
    
    # Passed parameters checking function
    indexbytes.stypy_localization = localization
    indexbytes.stypy_type_of_self = None
    indexbytes.stypy_type_store = module_type_store
    indexbytes.stypy_function_name = 'indexbytes'
    indexbytes.stypy_param_names_list = ['buf', 'i']
    indexbytes.stypy_varargs_param_name = None
    indexbytes.stypy_kwargs_param_name = None
    indexbytes.stypy_call_defaults = defaults
    indexbytes.stypy_call_varargs = varargs
    indexbytes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'indexbytes', ['buf', 'i'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'indexbytes', localization, ['buf', 'i'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'indexbytes(...)' code ##################

    
    # Call to ord(...): (line 661)
    # Processing the call arguments (line 661)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 661)
    i_1741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 23), 'i', False)
    # Getting the type of 'buf' (line 661)
    buf_1742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 19), 'buf', False)
    # Obtaining the member '__getitem__' of a type (line 661)
    getitem___1743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 19), buf_1742, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 661)
    subscript_call_result_1744 = invoke(stypy.reporting.localization.Localization(__file__, 661, 19), getitem___1743, i_1741)
    
    # Processing the call keyword arguments (line 661)
    kwargs_1745 = {}
    # Getting the type of 'ord' (line 661)
    ord_1740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 15), 'ord', False)
    # Calling ord(args, kwargs) (line 661)
    ord_call_result_1746 = invoke(stypy.reporting.localization.Localization(__file__, 661, 15), ord_1740, *[subscript_call_result_1744], **kwargs_1745)
    
    # Assigning a type to the variable 'stypy_return_type' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 8), 'stypy_return_type', ord_call_result_1746)
    
    # ################# End of 'indexbytes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'indexbytes' in the type store
    # Getting the type of 'stypy_return_type' (line 660)
    stypy_return_type_1747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1747)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'indexbytes'
    return stypy_return_type_1747

# Assigning a type to the variable 'indexbytes' (line 660)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'indexbytes', indexbytes)

# Assigning a Call to a Name (line 662):

# Call to partial(...): (line 662)
# Processing the call arguments (line 662)
# Getting the type of 'itertools' (line 662)
itertools_1750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 34), 'itertools', False)
# Obtaining the member 'imap' of a type (line 662)
imap_1751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 34), itertools_1750, 'imap')
# Getting the type of 'ord' (line 662)
ord_1752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 50), 'ord', False)
# Processing the call keyword arguments (line 662)
kwargs_1753 = {}
# Getting the type of 'functools' (line 662)
functools_1748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'functools', False)
# Obtaining the member 'partial' of a type (line 662)
partial_1749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), functools_1748, 'partial')
# Calling partial(args, kwargs) (line 662)
partial_call_result_1754 = invoke(stypy.reporting.localization.Localization(__file__, 662, 16), partial_1749, *[imap_1751, ord_1752], **kwargs_1753)

# Assigning a type to the variable 'iterbytes' (line 662)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'iterbytes', partial_call_result_1754)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 663, 4))

# 'import StringIO' statement (line 663)
import StringIO

import_module(stypy.reporting.localization.Localization(__file__, 663, 4), 'StringIO', StringIO, module_type_store)


# Multiple assignment of 2 elements.
# Getting the type of 'StringIO' (line 664)
StringIO_1755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 25), 'StringIO')
# Obtaining the member 'StringIO' of a type (line 664)
StringIO_1756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 25), StringIO_1755, 'StringIO')
# Assigning a type to the variable 'BytesIO' (line 664)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 15), 'BytesIO', StringIO_1756)
# Getting the type of 'BytesIO' (line 664)
BytesIO_1757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 15), 'BytesIO')
# Assigning a type to the variable 'StringIO' (line 664)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'StringIO', BytesIO_1757)

# Assigning a Str to a Name (line 665):
str_1758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 24), 'str', 'assertItemsEqual')
# Assigning a type to the variable '_assertCountEqual' (line 665)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), '_assertCountEqual', str_1758)

# Assigning a Str to a Name (line 666):
str_1759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 25), 'str', 'assertRaisesRegexp')
# Assigning a type to the variable '_assertRaisesRegex' (line 666)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), '_assertRaisesRegex', str_1759)

# Assigning a Str to a Name (line 667):
str_1760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 19), 'str', 'assertRegexpMatches')
# Assigning a type to the variable '_assertRegex' (line 667)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 4), '_assertRegex', str_1760)
# SSA join for if statement (line 624)
module_type_store = module_type_store.join_ssa_context()


# Call to _add_doc(...): (line 668)
# Processing the call arguments (line 668)
# Getting the type of 'b' (line 668)
b_1762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 9), 'b', False)
str_1763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 12), 'str', 'Byte literal')
# Processing the call keyword arguments (line 668)
kwargs_1764 = {}
# Getting the type of '_add_doc' (line 668)
_add_doc_1761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 668)
_add_doc_call_result_1765 = invoke(stypy.reporting.localization.Localization(__file__, 668, 0), _add_doc_1761, *[b_1762, str_1763], **kwargs_1764)


# Call to _add_doc(...): (line 669)
# Processing the call arguments (line 669)
# Getting the type of 'u' (line 669)
u_1767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 9), 'u', False)
str_1768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 12), 'str', 'Text literal')
# Processing the call keyword arguments (line 669)
kwargs_1769 = {}
# Getting the type of '_add_doc' (line 669)
_add_doc_1766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 669)
_add_doc_call_result_1770 = invoke(stypy.reporting.localization.Localization(__file__, 669, 0), _add_doc_1766, *[u_1767, str_1768], **kwargs_1769)


@norecursion
def assertCountEqual(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assertCountEqual'
    module_type_store = module_type_store.open_function_context('assertCountEqual', 672, 0, False)
    
    # Passed parameters checking function
    assertCountEqual.stypy_localization = localization
    assertCountEqual.stypy_type_of_self = None
    assertCountEqual.stypy_type_store = module_type_store
    assertCountEqual.stypy_function_name = 'assertCountEqual'
    assertCountEqual.stypy_param_names_list = ['self']
    assertCountEqual.stypy_varargs_param_name = 'args'
    assertCountEqual.stypy_kwargs_param_name = 'kwargs'
    assertCountEqual.stypy_call_defaults = defaults
    assertCountEqual.stypy_call_varargs = varargs
    assertCountEqual.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assertCountEqual', ['self'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assertCountEqual', localization, ['self'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assertCountEqual(...)' code ##################

    
    # Call to (...): (line 673)
    # Getting the type of 'args' (line 673)
    args_1776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 45), 'args', False)
    # Processing the call keyword arguments (line 673)
    # Getting the type of 'kwargs' (line 673)
    kwargs_1777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 53), 'kwargs', False)
    kwargs_1778 = {'kwargs_1777': kwargs_1777}
    
    # Call to getattr(...): (line 673)
    # Processing the call arguments (line 673)
    # Getting the type of 'self' (line 673)
    self_1772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 19), 'self', False)
    # Getting the type of '_assertCountEqual' (line 673)
    _assertCountEqual_1773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 25), '_assertCountEqual', False)
    # Processing the call keyword arguments (line 673)
    kwargs_1774 = {}
    # Getting the type of 'getattr' (line 673)
    getattr_1771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 11), 'getattr', False)
    # Calling getattr(args, kwargs) (line 673)
    getattr_call_result_1775 = invoke(stypy.reporting.localization.Localization(__file__, 673, 11), getattr_1771, *[self_1772, _assertCountEqual_1773], **kwargs_1774)
    
    # Calling (args, kwargs) (line 673)
    _call_result_1779 = invoke(stypy.reporting.localization.Localization(__file__, 673, 11), getattr_call_result_1775, *[args_1776], **kwargs_1778)
    
    # Assigning a type to the variable 'stypy_return_type' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'stypy_return_type', _call_result_1779)
    
    # ################# End of 'assertCountEqual(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assertCountEqual' in the type store
    # Getting the type of 'stypy_return_type' (line 672)
    stypy_return_type_1780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assertCountEqual'
    return stypy_return_type_1780

# Assigning a type to the variable 'assertCountEqual' (line 672)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 0), 'assertCountEqual', assertCountEqual)

@norecursion
def assertRaisesRegex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assertRaisesRegex'
    module_type_store = module_type_store.open_function_context('assertRaisesRegex', 676, 0, False)
    
    # Passed parameters checking function
    assertRaisesRegex.stypy_localization = localization
    assertRaisesRegex.stypy_type_of_self = None
    assertRaisesRegex.stypy_type_store = module_type_store
    assertRaisesRegex.stypy_function_name = 'assertRaisesRegex'
    assertRaisesRegex.stypy_param_names_list = ['self']
    assertRaisesRegex.stypy_varargs_param_name = 'args'
    assertRaisesRegex.stypy_kwargs_param_name = 'kwargs'
    assertRaisesRegex.stypy_call_defaults = defaults
    assertRaisesRegex.stypy_call_varargs = varargs
    assertRaisesRegex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assertRaisesRegex', ['self'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assertRaisesRegex', localization, ['self'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assertRaisesRegex(...)' code ##################

    
    # Call to (...): (line 677)
    # Getting the type of 'args' (line 677)
    args_1786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 46), 'args', False)
    # Processing the call keyword arguments (line 677)
    # Getting the type of 'kwargs' (line 677)
    kwargs_1787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 54), 'kwargs', False)
    kwargs_1788 = {'kwargs_1787': kwargs_1787}
    
    # Call to getattr(...): (line 677)
    # Processing the call arguments (line 677)
    # Getting the type of 'self' (line 677)
    self_1782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 19), 'self', False)
    # Getting the type of '_assertRaisesRegex' (line 677)
    _assertRaisesRegex_1783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 25), '_assertRaisesRegex', False)
    # Processing the call keyword arguments (line 677)
    kwargs_1784 = {}
    # Getting the type of 'getattr' (line 677)
    getattr_1781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 11), 'getattr', False)
    # Calling getattr(args, kwargs) (line 677)
    getattr_call_result_1785 = invoke(stypy.reporting.localization.Localization(__file__, 677, 11), getattr_1781, *[self_1782, _assertRaisesRegex_1783], **kwargs_1784)
    
    # Calling (args, kwargs) (line 677)
    _call_result_1789 = invoke(stypy.reporting.localization.Localization(__file__, 677, 11), getattr_call_result_1785, *[args_1786], **kwargs_1788)
    
    # Assigning a type to the variable 'stypy_return_type' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 4), 'stypy_return_type', _call_result_1789)
    
    # ################# End of 'assertRaisesRegex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assertRaisesRegex' in the type store
    # Getting the type of 'stypy_return_type' (line 676)
    stypy_return_type_1790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1790)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assertRaisesRegex'
    return stypy_return_type_1790

# Assigning a type to the variable 'assertRaisesRegex' (line 676)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 0), 'assertRaisesRegex', assertRaisesRegex)

@norecursion
def assertRegex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assertRegex'
    module_type_store = module_type_store.open_function_context('assertRegex', 680, 0, False)
    
    # Passed parameters checking function
    assertRegex.stypy_localization = localization
    assertRegex.stypy_type_of_self = None
    assertRegex.stypy_type_store = module_type_store
    assertRegex.stypy_function_name = 'assertRegex'
    assertRegex.stypy_param_names_list = ['self']
    assertRegex.stypy_varargs_param_name = 'args'
    assertRegex.stypy_kwargs_param_name = 'kwargs'
    assertRegex.stypy_call_defaults = defaults
    assertRegex.stypy_call_varargs = varargs
    assertRegex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assertRegex', ['self'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assertRegex', localization, ['self'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assertRegex(...)' code ##################

    
    # Call to (...): (line 681)
    # Getting the type of 'args' (line 681)
    args_1796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 40), 'args', False)
    # Processing the call keyword arguments (line 681)
    # Getting the type of 'kwargs' (line 681)
    kwargs_1797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 48), 'kwargs', False)
    kwargs_1798 = {'kwargs_1797': kwargs_1797}
    
    # Call to getattr(...): (line 681)
    # Processing the call arguments (line 681)
    # Getting the type of 'self' (line 681)
    self_1792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 19), 'self', False)
    # Getting the type of '_assertRegex' (line 681)
    _assertRegex_1793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 25), '_assertRegex', False)
    # Processing the call keyword arguments (line 681)
    kwargs_1794 = {}
    # Getting the type of 'getattr' (line 681)
    getattr_1791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 11), 'getattr', False)
    # Calling getattr(args, kwargs) (line 681)
    getattr_call_result_1795 = invoke(stypy.reporting.localization.Localization(__file__, 681, 11), getattr_1791, *[self_1792, _assertRegex_1793], **kwargs_1794)
    
    # Calling (args, kwargs) (line 681)
    _call_result_1799 = invoke(stypy.reporting.localization.Localization(__file__, 681, 11), getattr_call_result_1795, *[args_1796], **kwargs_1798)
    
    # Assigning a type to the variable 'stypy_return_type' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'stypy_return_type', _call_result_1799)
    
    # ################# End of 'assertRegex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assertRegex' in the type store
    # Getting the type of 'stypy_return_type' (line 680)
    stypy_return_type_1800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1800)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assertRegex'
    return stypy_return_type_1800

# Assigning a type to the variable 'assertRegex' (line 680)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 0), 'assertRegex', assertRegex)

# Getting the type of 'PY3' (line 684)
PY3_1801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 3), 'PY3')
# Testing the type of an if condition (line 684)
if_condition_1802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 684, 0), PY3_1801)
# Assigning a type to the variable 'if_condition_1802' (line 684)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 0), 'if_condition_1802', if_condition_1802)
# SSA begins for if statement (line 684)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 685):

# Call to getattr(...): (line 685)
# Processing the call arguments (line 685)
# Getting the type of 'moves' (line 685)
moves_1804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 20), 'moves', False)
# Obtaining the member 'builtins' of a type (line 685)
builtins_1805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 20), moves_1804, 'builtins')
str_1806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 36), 'str', 'exec')
# Processing the call keyword arguments (line 685)
kwargs_1807 = {}
# Getting the type of 'getattr' (line 685)
getattr_1803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 12), 'getattr', False)
# Calling getattr(args, kwargs) (line 685)
getattr_call_result_1808 = invoke(stypy.reporting.localization.Localization(__file__, 685, 12), getattr_1803, *[builtins_1805, str_1806], **kwargs_1807)

# Assigning a type to the variable 'exec_' (line 685)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'exec_', getattr_call_result_1808)

@norecursion
def reraise(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 687)
    None_1809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 30), 'None')
    defaults = [None_1809]
    # Create a new context for function 'reraise'
    module_type_store = module_type_store.open_function_context('reraise', 687, 4, False)
    
    # Passed parameters checking function
    reraise.stypy_localization = localization
    reraise.stypy_type_of_self = None
    reraise.stypy_type_store = module_type_store
    reraise.stypy_function_name = 'reraise'
    reraise.stypy_param_names_list = ['tp', 'value', 'tb']
    reraise.stypy_varargs_param_name = None
    reraise.stypy_kwargs_param_name = None
    reraise.stypy_call_defaults = defaults
    reraise.stypy_call_varargs = varargs
    reraise.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'reraise', ['tp', 'value', 'tb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'reraise', localization, ['tp', 'value', 'tb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'reraise(...)' code ##################

    
    # Try-finally block (line 688)
    
    # Type idiom detected: calculating its left and rigth part (line 689)
    # Getting the type of 'value' (line 689)
    value_1810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 15), 'value')
    # Getting the type of 'None' (line 689)
    None_1811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 24), 'None')
    
    (may_be_1812, more_types_in_union_1813) = may_be_none(value_1810, None_1811)

    if may_be_1812:

        if more_types_in_union_1813:
            # Runtime conditional SSA (line 689)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 690):
        
        # Call to tp(...): (line 690)
        # Processing the call keyword arguments (line 690)
        kwargs_1815 = {}
        # Getting the type of 'tp' (line 690)
        tp_1814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 24), 'tp', False)
        # Calling tp(args, kwargs) (line 690)
        tp_call_result_1816 = invoke(stypy.reporting.localization.Localization(__file__, 690, 24), tp_1814, *[], **kwargs_1815)
        
        # Assigning a type to the variable 'value' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'value', tp_call_result_1816)

        if more_types_in_union_1813:
            # SSA join for if statement (line 689)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'value' (line 691)
    value_1817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 15), 'value')
    # Obtaining the member '__traceback__' of a type (line 691)
    traceback___1818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 15), value_1817, '__traceback__')
    # Getting the type of 'tb' (line 691)
    tb_1819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 42), 'tb')
    # Applying the binary operator 'isnot' (line 691)
    result_is_not_1820 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 15), 'isnot', traceback___1818, tb_1819)
    
    # Testing the type of an if condition (line 691)
    if_condition_1821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 691, 12), result_is_not_1820)
    # Assigning a type to the variable 'if_condition_1821' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 12), 'if_condition_1821', if_condition_1821)
    # SSA begins for if statement (line 691)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to with_traceback(...): (line 692)
    # Processing the call arguments (line 692)
    # Getting the type of 'tb' (line 692)
    tb_1824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 43), 'tb', False)
    # Processing the call keyword arguments (line 692)
    kwargs_1825 = {}
    # Getting the type of 'value' (line 692)
    value_1822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 22), 'value', False)
    # Obtaining the member 'with_traceback' of a type (line 692)
    with_traceback_1823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 22), value_1822, 'with_traceback')
    # Calling with_traceback(args, kwargs) (line 692)
    with_traceback_call_result_1826 = invoke(stypy.reporting.localization.Localization(__file__, 692, 22), with_traceback_1823, *[tb_1824], **kwargs_1825)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 692, 16), with_traceback_call_result_1826, 'raise parameter', BaseException)
    # SSA join for if statement (line 691)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'value' (line 693)
    value_1827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 18), 'value')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 693, 12), value_1827, 'raise parameter', BaseException)
    
    # finally branch of the try-finally block (line 688)
    
    # Assigning a Name to a Name (line 695):
    # Getting the type of 'None' (line 695)
    None_1828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 20), 'None')
    # Assigning a type to the variable 'value' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 12), 'value', None_1828)
    
    # Assigning a Name to a Name (line 696):
    # Getting the type of 'None' (line 696)
    None_1829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'None')
    # Assigning a type to the variable 'tb' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 12), 'tb', None_1829)
    
    
    # ################# End of 'reraise(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reraise' in the type store
    # Getting the type of 'stypy_return_type' (line 687)
    stypy_return_type_1830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1830)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reraise'
    return stypy_return_type_1830

# Assigning a type to the variable 'reraise' (line 687)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'reraise', reraise)
# SSA branch for the else part of an if statement (line 684)
module_type_store.open_ssa_branch('else')

@norecursion
def exec_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 699)
    None_1831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 30), 'None')
    # Getting the type of 'None' (line 699)
    None_1832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 43), 'None')
    defaults = [None_1831, None_1832]
    # Create a new context for function 'exec_'
    module_type_store = module_type_store.open_function_context('exec_', 699, 4, False)
    
    # Passed parameters checking function
    exec_.stypy_localization = localization
    exec_.stypy_type_of_self = None
    exec_.stypy_type_store = module_type_store
    exec_.stypy_function_name = 'exec_'
    exec_.stypy_param_names_list = ['_code_', '_globs_', '_locs_']
    exec_.stypy_varargs_param_name = None
    exec_.stypy_kwargs_param_name = None
    exec_.stypy_call_defaults = defaults
    exec_.stypy_call_varargs = varargs
    exec_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exec_', ['_code_', '_globs_', '_locs_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exec_', localization, ['_code_', '_globs_', '_locs_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exec_(...)' code ##################

    str_1833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 8), 'str', 'Execute code in a namespace.')
    
    # Type idiom detected: calculating its left and rigth part (line 701)
    # Getting the type of '_globs_' (line 701)
    _globs__1834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 11), '_globs_')
    # Getting the type of 'None' (line 701)
    None_1835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 22), 'None')
    
    (may_be_1836, more_types_in_union_1837) = may_be_none(_globs__1834, None_1835)

    if may_be_1836:

        if more_types_in_union_1837:
            # Runtime conditional SSA (line 701)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 702):
        
        # Call to _getframe(...): (line 702)
        # Processing the call arguments (line 702)
        int_1840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 34), 'int')
        # Processing the call keyword arguments (line 702)
        kwargs_1841 = {}
        # Getting the type of 'sys' (line 702)
        sys_1838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 20), 'sys', False)
        # Obtaining the member '_getframe' of a type (line 702)
        _getframe_1839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 20), sys_1838, '_getframe')
        # Calling _getframe(args, kwargs) (line 702)
        _getframe_call_result_1842 = invoke(stypy.reporting.localization.Localization(__file__, 702, 20), _getframe_1839, *[int_1840], **kwargs_1841)
        
        # Assigning a type to the variable 'frame' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'frame', _getframe_call_result_1842)
        
        # Assigning a Attribute to a Name (line 703):
        # Getting the type of 'frame' (line 703)
        frame_1843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 22), 'frame')
        # Obtaining the member 'f_globals' of a type (line 703)
        f_globals_1844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 22), frame_1843, 'f_globals')
        # Assigning a type to the variable '_globs_' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 12), '_globs_', f_globals_1844)
        
        # Type idiom detected: calculating its left and rigth part (line 704)
        # Getting the type of '_locs_' (line 704)
        _locs__1845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 15), '_locs_')
        # Getting the type of 'None' (line 704)
        None_1846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 25), 'None')
        
        (may_be_1847, more_types_in_union_1848) = may_be_none(_locs__1845, None_1846)

        if may_be_1847:

            if more_types_in_union_1848:
                # Runtime conditional SSA (line 704)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 705):
            # Getting the type of 'frame' (line 705)
            frame_1849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 25), 'frame')
            # Obtaining the member 'f_locals' of a type (line 705)
            f_locals_1850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 25), frame_1849, 'f_locals')
            # Assigning a type to the variable '_locs_' (line 705)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 16), '_locs_', f_locals_1850)

            if more_types_in_union_1848:
                # SSA join for if statement (line 704)
                module_type_store = module_type_store.join_ssa_context()


        
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 706, 12), module_type_store, 'frame')

        if more_types_in_union_1837:
            # Runtime conditional SSA for else branch (line 701)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_1836) or more_types_in_union_1837):
        
        # Type idiom detected: calculating its left and rigth part (line 707)
        # Getting the type of '_locs_' (line 707)
        _locs__1851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 13), '_locs_')
        # Getting the type of 'None' (line 707)
        None_1852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 23), 'None')
        
        (may_be_1853, more_types_in_union_1854) = may_be_none(_locs__1851, None_1852)

        if may_be_1853:

            if more_types_in_union_1854:
                # Runtime conditional SSA (line 707)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 708):
            # Getting the type of '_globs_' (line 708)
            _globs__1855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 21), '_globs_')
            # Assigning a type to the variable '_locs_' (line 708)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 12), '_locs_', _globs__1855)

            if more_types_in_union_1854:
                # SSA join for if statement (line 707)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_1836 and more_types_in_union_1837):
            # SSA join for if statement (line 701)
            module_type_store = module_type_store.join_ssa_context()


    
    # Dynamic code evaluation using an exec statement
    str_1856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 13), 'str', 'exec _code_ in _globs_, _locs_')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 709, 8), str_1856, 'exec parameter', 'StringType', 'FileType', 'CodeType')
    enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 709, 8))
    
    # ################# End of 'exec_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exec_' in the type store
    # Getting the type of 'stypy_return_type' (line 699)
    stypy_return_type_1857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1857)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exec_'
    return stypy_return_type_1857

# Assigning a type to the variable 'exec_' (line 699)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'exec_', exec_)

# Call to exec_(...): (line 711)
# Processing the call arguments (line 711)
str_1859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, (-1)), 'str', 'def reraise(tp, value, tb=None):\n    try:\n        raise tp, value, tb\n    finally:\n        tb = None\n')
# Processing the call keyword arguments (line 711)
kwargs_1860 = {}
# Getting the type of 'exec_' (line 711)
exec__1858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'exec_', False)
# Calling exec_(args, kwargs) (line 711)
exec__call_result_1861 = invoke(stypy.reporting.localization.Localization(__file__, 711, 4), exec__1858, *[str_1859], **kwargs_1860)

# SSA join for if statement (line 684)
module_type_store = module_type_store.join_ssa_context()




# Obtaining the type of the subscript
int_1862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 21), 'int')
slice_1863 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 719, 3), None, int_1862, None)
# Getting the type of 'sys' (line 719)
sys_1864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 719)
version_info_1865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 3), sys_1864, 'version_info')
# Obtaining the member '__getitem__' of a type (line 719)
getitem___1866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 3), version_info_1865, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 719)
subscript_call_result_1867 = invoke(stypy.reporting.localization.Localization(__file__, 719, 3), getitem___1866, slice_1863)


# Obtaining an instance of the builtin type 'tuple' (line 719)
tuple_1868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 719)
# Adding element type (line 719)
int_1869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 28), tuple_1868, int_1869)
# Adding element type (line 719)
int_1870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 28), tuple_1868, int_1870)

# Applying the binary operator '==' (line 719)
result_eq_1871 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 3), '==', subscript_call_result_1867, tuple_1868)

# Testing the type of an if condition (line 719)
if_condition_1872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 719, 0), result_eq_1871)
# Assigning a type to the variable 'if_condition_1872' (line 719)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 0), 'if_condition_1872', if_condition_1872)
# SSA begins for if statement (line 719)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to exec_(...): (line 720)
# Processing the call arguments (line 720)
str_1874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, (-1)), 'str', 'def raise_from(value, from_value):\n    try:\n        if from_value is None:\n            raise value\n        raise value from from_value\n    finally:\n        value = None\n')
# Processing the call keyword arguments (line 720)
kwargs_1875 = {}
# Getting the type of 'exec_' (line 720)
exec__1873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 4), 'exec_', False)
# Calling exec_(args, kwargs) (line 720)
exec__call_result_1876 = invoke(stypy.reporting.localization.Localization(__file__, 720, 4), exec__1873, *[str_1874], **kwargs_1875)

# SSA branch for the else part of an if statement (line 719)
module_type_store.open_ssa_branch('else')



# Obtaining the type of the subscript
int_1877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 23), 'int')
slice_1878 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 728, 5), None, int_1877, None)
# Getting the type of 'sys' (line 728)
sys_1879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 5), 'sys')
# Obtaining the member 'version_info' of a type (line 728)
version_info_1880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 5), sys_1879, 'version_info')
# Obtaining the member '__getitem__' of a type (line 728)
getitem___1881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 5), version_info_1880, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 728)
subscript_call_result_1882 = invoke(stypy.reporting.localization.Localization(__file__, 728, 5), getitem___1881, slice_1878)


# Obtaining an instance of the builtin type 'tuple' (line 728)
tuple_1883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 728)
# Adding element type (line 728)
int_1884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 728, 29), tuple_1883, int_1884)
# Adding element type (line 728)
int_1885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 728, 29), tuple_1883, int_1885)

# Applying the binary operator '>' (line 728)
result_gt_1886 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 5), '>', subscript_call_result_1882, tuple_1883)

# Testing the type of an if condition (line 728)
if_condition_1887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 728, 5), result_gt_1886)
# Assigning a type to the variable 'if_condition_1887' (line 728)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 5), 'if_condition_1887', if_condition_1887)
# SSA begins for if statement (line 728)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to exec_(...): (line 729)
# Processing the call arguments (line 729)
str_1889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, (-1)), 'str', 'def raise_from(value, from_value):\n    try:\n        raise value from from_value\n    finally:\n        value = None\n')
# Processing the call keyword arguments (line 729)
kwargs_1890 = {}
# Getting the type of 'exec_' (line 729)
exec__1888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'exec_', False)
# Calling exec_(args, kwargs) (line 729)
exec__call_result_1891 = invoke(stypy.reporting.localization.Localization(__file__, 729, 4), exec__1888, *[str_1889], **kwargs_1890)

# SSA branch for the else part of an if statement (line 728)
module_type_store.open_ssa_branch('else')

@norecursion
def raise_from(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'raise_from'
    module_type_store = module_type_store.open_function_context('raise_from', 736, 4, False)
    
    # Passed parameters checking function
    raise_from.stypy_localization = localization
    raise_from.stypy_type_of_self = None
    raise_from.stypy_type_store = module_type_store
    raise_from.stypy_function_name = 'raise_from'
    raise_from.stypy_param_names_list = ['value', 'from_value']
    raise_from.stypy_varargs_param_name = None
    raise_from.stypy_kwargs_param_name = None
    raise_from.stypy_call_defaults = defaults
    raise_from.stypy_call_varargs = varargs
    raise_from.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'raise_from', ['value', 'from_value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'raise_from', localization, ['value', 'from_value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'raise_from(...)' code ##################

    # Getting the type of 'value' (line 737)
    value_1892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 14), 'value')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 737, 8), value_1892, 'raise parameter', BaseException)
    
    # ################# End of 'raise_from(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'raise_from' in the type store
    # Getting the type of 'stypy_return_type' (line 736)
    stypy_return_type_1893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1893)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'raise_from'
    return stypy_return_type_1893

# Assigning a type to the variable 'raise_from' (line 736)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'raise_from', raise_from)
# SSA join for if statement (line 728)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 719)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 740):

# Call to getattr(...): (line 740)
# Processing the call arguments (line 740)
# Getting the type of 'moves' (line 740)
moves_1895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 17), 'moves', False)
# Obtaining the member 'builtins' of a type (line 740)
builtins_1896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 17), moves_1895, 'builtins')
str_1897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 33), 'str', 'print')
# Getting the type of 'None' (line 740)
None_1898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 42), 'None', False)
# Processing the call keyword arguments (line 740)
kwargs_1899 = {}
# Getting the type of 'getattr' (line 740)
getattr_1894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 9), 'getattr', False)
# Calling getattr(args, kwargs) (line 740)
getattr_call_result_1900 = invoke(stypy.reporting.localization.Localization(__file__, 740, 9), getattr_1894, *[builtins_1896, str_1897, None_1898], **kwargs_1899)

# Assigning a type to the variable 'print_' (line 740)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 0), 'print_', getattr_call_result_1900)

# Type idiom detected: calculating its left and rigth part (line 741)
# Getting the type of 'print_' (line 741)
print__1901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 3), 'print_')
# Getting the type of 'None' (line 741)
None_1902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 13), 'None')

(may_be_1903, more_types_in_union_1904) = may_be_none(print__1901, None_1902)

if may_be_1903:

    if more_types_in_union_1904:
        # Runtime conditional SSA (line 741)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store


    @norecursion
    def print_(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_'
        module_type_store = module_type_store.open_function_context('print_', 742, 4, False)
        
        # Passed parameters checking function
        print_.stypy_localization = localization
        print_.stypy_type_of_self = None
        print_.stypy_type_store = module_type_store
        print_.stypy_function_name = 'print_'
        print_.stypy_param_names_list = []
        print_.stypy_varargs_param_name = 'args'
        print_.stypy_kwargs_param_name = 'kwargs'
        print_.stypy_call_defaults = defaults
        print_.stypy_call_varargs = varargs
        print_.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'print_', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_(...)' code ##################

        str_1905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 8), 'str', 'The new-style print function for Python 2.4 and 2.5.')
        
        # Assigning a Call to a Name (line 744):
        
        # Call to pop(...): (line 744)
        # Processing the call arguments (line 744)
        str_1908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 24), 'str', 'file')
        # Getting the type of 'sys' (line 744)
        sys_1909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 32), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 744)
        stdout_1910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 32), sys_1909, 'stdout')
        # Processing the call keyword arguments (line 744)
        kwargs_1911 = {}
        # Getting the type of 'kwargs' (line 744)
        kwargs_1906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 13), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 744)
        pop_1907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 13), kwargs_1906, 'pop')
        # Calling pop(args, kwargs) (line 744)
        pop_call_result_1912 = invoke(stypy.reporting.localization.Localization(__file__, 744, 13), pop_1907, *[str_1908, stdout_1910], **kwargs_1911)
        
        # Assigning a type to the variable 'fp' (line 744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'fp', pop_call_result_1912)
        
        # Type idiom detected: calculating its left and rigth part (line 745)
        # Getting the type of 'fp' (line 745)
        fp_1913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 11), 'fp')
        # Getting the type of 'None' (line 745)
        None_1914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 17), 'None')
        
        (may_be_1915, more_types_in_union_1916) = may_be_none(fp_1913, None_1914)

        if may_be_1915:

            if more_types_in_union_1916:
                # Runtime conditional SSA (line 745)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 746)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_1916:
                # SSA join for if statement (line 745)
                module_type_store = module_type_store.join_ssa_context()


        

        @norecursion
        def write(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'write'
            module_type_store = module_type_store.open_function_context('write', 748, 8, False)
            
            # Passed parameters checking function
            write.stypy_localization = localization
            write.stypy_type_of_self = None
            write.stypy_type_store = module_type_store
            write.stypy_function_name = 'write'
            write.stypy_param_names_list = ['data']
            write.stypy_varargs_param_name = None
            write.stypy_kwargs_param_name = None
            write.stypy_call_defaults = defaults
            write.stypy_call_varargs = varargs
            write.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'write', ['data'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'write', localization, ['data'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'write(...)' code ##################

            
            # Type idiom detected: calculating its left and rigth part (line 749)
            # Getting the type of 'basestring' (line 749)
            basestring_1917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 36), 'basestring')
            # Getting the type of 'data' (line 749)
            data_1918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 30), 'data')
            
            (may_be_1919, more_types_in_union_1920) = may_not_be_subtype(basestring_1917, data_1918)

            if may_be_1919:

                if more_types_in_union_1920:
                    # Runtime conditional SSA (line 749)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'data' (line 749)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 12), 'data', remove_subtype_from_union(data_1918, basestring))
                
                # Assigning a Call to a Name (line 750):
                
                # Call to str(...): (line 750)
                # Processing the call arguments (line 750)
                # Getting the type of 'data' (line 750)
                data_1922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 27), 'data', False)
                # Processing the call keyword arguments (line 750)
                kwargs_1923 = {}
                # Getting the type of 'str' (line 750)
                str_1921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 23), 'str', False)
                # Calling str(args, kwargs) (line 750)
                str_call_result_1924 = invoke(stypy.reporting.localization.Localization(__file__, 750, 23), str_1921, *[data_1922], **kwargs_1923)
                
                # Assigning a type to the variable 'data' (line 750)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 16), 'data', str_call_result_1924)

                if more_types_in_union_1920:
                    # SSA join for if statement (line 749)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Evaluating a boolean operation
            
            # Call to isinstance(...): (line 752)
            # Processing the call arguments (line 752)
            # Getting the type of 'fp' (line 752)
            fp_1926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 27), 'fp', False)
            # Getting the type of 'file' (line 752)
            file_1927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 31), 'file', False)
            # Processing the call keyword arguments (line 752)
            kwargs_1928 = {}
            # Getting the type of 'isinstance' (line 752)
            isinstance_1925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 16), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 752)
            isinstance_call_result_1929 = invoke(stypy.reporting.localization.Localization(__file__, 752, 16), isinstance_1925, *[fp_1926, file_1927], **kwargs_1928)
            
            
            # Call to isinstance(...): (line 753)
            # Processing the call arguments (line 753)
            # Getting the type of 'data' (line 753)
            data_1931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 31), 'data', False)
            # Getting the type of 'unicode' (line 753)
            unicode_1932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 37), 'unicode', False)
            # Processing the call keyword arguments (line 753)
            kwargs_1933 = {}
            # Getting the type of 'isinstance' (line 753)
            isinstance_1930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 20), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 753)
            isinstance_call_result_1934 = invoke(stypy.reporting.localization.Localization(__file__, 753, 20), isinstance_1930, *[data_1931, unicode_1932], **kwargs_1933)
            
            # Applying the binary operator 'and' (line 752)
            result_and_keyword_1935 = python_operator(stypy.reporting.localization.Localization(__file__, 752, 16), 'and', isinstance_call_result_1929, isinstance_call_result_1934)
            
            # Getting the type of 'fp' (line 754)
            fp_1936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 20), 'fp')
            # Obtaining the member 'encoding' of a type (line 754)
            encoding_1937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 20), fp_1936, 'encoding')
            # Getting the type of 'None' (line 754)
            None_1938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 39), 'None')
            # Applying the binary operator 'isnot' (line 754)
            result_is_not_1939 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 20), 'isnot', encoding_1937, None_1938)
            
            # Applying the binary operator 'and' (line 752)
            result_and_keyword_1940 = python_operator(stypy.reporting.localization.Localization(__file__, 752, 16), 'and', result_and_keyword_1935, result_is_not_1939)
            
            # Testing the type of an if condition (line 752)
            if_condition_1941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 752, 12), result_and_keyword_1940)
            # Assigning a type to the variable 'if_condition_1941' (line 752)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 12), 'if_condition_1941', if_condition_1941)
            # SSA begins for if statement (line 752)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 755):
            
            # Call to getattr(...): (line 755)
            # Processing the call arguments (line 755)
            # Getting the type of 'fp' (line 755)
            fp_1943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 33), 'fp', False)
            str_1944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 37), 'str', 'errors')
            # Getting the type of 'None' (line 755)
            None_1945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 47), 'None', False)
            # Processing the call keyword arguments (line 755)
            kwargs_1946 = {}
            # Getting the type of 'getattr' (line 755)
            getattr_1942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 25), 'getattr', False)
            # Calling getattr(args, kwargs) (line 755)
            getattr_call_result_1947 = invoke(stypy.reporting.localization.Localization(__file__, 755, 25), getattr_1942, *[fp_1943, str_1944, None_1945], **kwargs_1946)
            
            # Assigning a type to the variable 'errors' (line 755)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 16), 'errors', getattr_call_result_1947)
            
            # Type idiom detected: calculating its left and rigth part (line 756)
            # Getting the type of 'errors' (line 756)
            errors_1948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 19), 'errors')
            # Getting the type of 'None' (line 756)
            None_1949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 29), 'None')
            
            (may_be_1950, more_types_in_union_1951) = may_be_none(errors_1948, None_1949)

            if may_be_1950:

                if more_types_in_union_1951:
                    # Runtime conditional SSA (line 756)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Str to a Name (line 757):
                str_1952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 29), 'str', 'strict')
                # Assigning a type to the variable 'errors' (line 757)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 20), 'errors', str_1952)

                if more_types_in_union_1951:
                    # SSA join for if statement (line 756)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Name (line 758):
            
            # Call to encode(...): (line 758)
            # Processing the call arguments (line 758)
            # Getting the type of 'fp' (line 758)
            fp_1955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 35), 'fp', False)
            # Obtaining the member 'encoding' of a type (line 758)
            encoding_1956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 35), fp_1955, 'encoding')
            # Getting the type of 'errors' (line 758)
            errors_1957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 48), 'errors', False)
            # Processing the call keyword arguments (line 758)
            kwargs_1958 = {}
            # Getting the type of 'data' (line 758)
            data_1953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 23), 'data', False)
            # Obtaining the member 'encode' of a type (line 758)
            encode_1954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 23), data_1953, 'encode')
            # Calling encode(args, kwargs) (line 758)
            encode_call_result_1959 = invoke(stypy.reporting.localization.Localization(__file__, 758, 23), encode_1954, *[encoding_1956, errors_1957], **kwargs_1958)
            
            # Assigning a type to the variable 'data' (line 758)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'data', encode_call_result_1959)
            # SSA join for if statement (line 752)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to write(...): (line 759)
            # Processing the call arguments (line 759)
            # Getting the type of 'data' (line 759)
            data_1962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 21), 'data', False)
            # Processing the call keyword arguments (line 759)
            kwargs_1963 = {}
            # Getting the type of 'fp' (line 759)
            fp_1960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 12), 'fp', False)
            # Obtaining the member 'write' of a type (line 759)
            write_1961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 12), fp_1960, 'write')
            # Calling write(args, kwargs) (line 759)
            write_call_result_1964 = invoke(stypy.reporting.localization.Localization(__file__, 759, 12), write_1961, *[data_1962], **kwargs_1963)
            
            
            # ################# End of 'write(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'write' in the type store
            # Getting the type of 'stypy_return_type' (line 748)
            stypy_return_type_1965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1965)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'write'
            return stypy_return_type_1965

        # Assigning a type to the variable 'write' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'write', write)
        
        # Assigning a Name to a Name (line 760):
        # Getting the type of 'False' (line 760)
        False_1966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 23), 'False')
        # Assigning a type to the variable 'want_unicode' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'want_unicode', False_1966)
        
        # Assigning a Call to a Name (line 761):
        
        # Call to pop(...): (line 761)
        # Processing the call arguments (line 761)
        str_1969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 25), 'str', 'sep')
        # Getting the type of 'None' (line 761)
        None_1970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 32), 'None', False)
        # Processing the call keyword arguments (line 761)
        kwargs_1971 = {}
        # Getting the type of 'kwargs' (line 761)
        kwargs_1967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 14), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 761)
        pop_1968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 14), kwargs_1967, 'pop')
        # Calling pop(args, kwargs) (line 761)
        pop_call_result_1972 = invoke(stypy.reporting.localization.Localization(__file__, 761, 14), pop_1968, *[str_1969, None_1970], **kwargs_1971)
        
        # Assigning a type to the variable 'sep' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'sep', pop_call_result_1972)
        
        # Type idiom detected: calculating its left and rigth part (line 762)
        # Getting the type of 'sep' (line 762)
        sep_1973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'sep')
        # Getting the type of 'None' (line 762)
        None_1974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 22), 'None')
        
        (may_be_1975, more_types_in_union_1976) = may_not_be_none(sep_1973, None_1974)

        if may_be_1975:

            if more_types_in_union_1976:
                # Runtime conditional SSA (line 762)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 763)
            # Getting the type of 'unicode' (line 763)
            unicode_1977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 31), 'unicode')
            # Getting the type of 'sep' (line 763)
            sep_1978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 26), 'sep')
            
            (may_be_1979, more_types_in_union_1980) = may_be_subtype(unicode_1977, sep_1978)

            if may_be_1979:

                if more_types_in_union_1980:
                    # Runtime conditional SSA (line 763)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'sep' (line 763)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'sep', remove_not_subtype_from_union(sep_1978, unicode))
                
                # Assigning a Name to a Name (line 764):
                # Getting the type of 'True' (line 764)
                True_1981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 31), 'True')
                # Assigning a type to the variable 'want_unicode' (line 764)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 16), 'want_unicode', True_1981)

                if more_types_in_union_1980:
                    # Runtime conditional SSA for else branch (line 763)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_1979) or more_types_in_union_1980):
                # Assigning a type to the variable 'sep' (line 763)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'sep', remove_subtype_from_union(sep_1978, unicode))
                
                # Type idiom detected: calculating its left and rigth part (line 765)
                # Getting the type of 'str' (line 765)
                str_1982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 37), 'str')
                # Getting the type of 'sep' (line 765)
                sep_1983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 32), 'sep')
                
                (may_be_1984, more_types_in_union_1985) = may_not_be_subtype(str_1982, sep_1983)

                if may_be_1984:

                    if more_types_in_union_1985:
                        # Runtime conditional SSA (line 765)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'sep' (line 765)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 17), 'sep', remove_subtype_from_union(sep_1983, str))
                    
                    # Call to TypeError(...): (line 766)
                    # Processing the call arguments (line 766)
                    str_1987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 32), 'str', 'sep must be None or a string')
                    # Processing the call keyword arguments (line 766)
                    kwargs_1988 = {}
                    # Getting the type of 'TypeError' (line 766)
                    TypeError_1986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 22), 'TypeError', False)
                    # Calling TypeError(args, kwargs) (line 766)
                    TypeError_call_result_1989 = invoke(stypy.reporting.localization.Localization(__file__, 766, 22), TypeError_1986, *[str_1987], **kwargs_1988)
                    
                    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 766, 16), TypeError_call_result_1989, 'raise parameter', BaseException)

                    if more_types_in_union_1985:
                        # SSA join for if statement (line 765)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_1979 and more_types_in_union_1980):
                    # SSA join for if statement (line 763)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_1976:
                # SSA join for if statement (line 762)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 767):
        
        # Call to pop(...): (line 767)
        # Processing the call arguments (line 767)
        str_1992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 25), 'str', 'end')
        # Getting the type of 'None' (line 767)
        None_1993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 32), 'None', False)
        # Processing the call keyword arguments (line 767)
        kwargs_1994 = {}
        # Getting the type of 'kwargs' (line 767)
        kwargs_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 14), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 767)
        pop_1991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 14), kwargs_1990, 'pop')
        # Calling pop(args, kwargs) (line 767)
        pop_call_result_1995 = invoke(stypy.reporting.localization.Localization(__file__, 767, 14), pop_1991, *[str_1992, None_1993], **kwargs_1994)
        
        # Assigning a type to the variable 'end' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'end', pop_call_result_1995)
        
        # Type idiom detected: calculating its left and rigth part (line 768)
        # Getting the type of 'end' (line 768)
        end_1996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'end')
        # Getting the type of 'None' (line 768)
        None_1997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 22), 'None')
        
        (may_be_1998, more_types_in_union_1999) = may_not_be_none(end_1996, None_1997)

        if may_be_1998:

            if more_types_in_union_1999:
                # Runtime conditional SSA (line 768)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 769)
            # Getting the type of 'unicode' (line 769)
            unicode_2000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 31), 'unicode')
            # Getting the type of 'end' (line 769)
            end_2001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 26), 'end')
            
            (may_be_2002, more_types_in_union_2003) = may_be_subtype(unicode_2000, end_2001)

            if may_be_2002:

                if more_types_in_union_2003:
                    # Runtime conditional SSA (line 769)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'end' (line 769)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'end', remove_not_subtype_from_union(end_2001, unicode))
                
                # Assigning a Name to a Name (line 770):
                # Getting the type of 'True' (line 770)
                True_2004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 31), 'True')
                # Assigning a type to the variable 'want_unicode' (line 770)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 16), 'want_unicode', True_2004)

                if more_types_in_union_2003:
                    # Runtime conditional SSA for else branch (line 769)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_2002) or more_types_in_union_2003):
                # Assigning a type to the variable 'end' (line 769)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'end', remove_subtype_from_union(end_2001, unicode))
                
                # Type idiom detected: calculating its left and rigth part (line 771)
                # Getting the type of 'str' (line 771)
                str_2005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 37), 'str')
                # Getting the type of 'end' (line 771)
                end_2006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 32), 'end')
                
                (may_be_2007, more_types_in_union_2008) = may_not_be_subtype(str_2005, end_2006)

                if may_be_2007:

                    if more_types_in_union_2008:
                        # Runtime conditional SSA (line 771)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'end' (line 771)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 17), 'end', remove_subtype_from_union(end_2006, str))
                    
                    # Call to TypeError(...): (line 772)
                    # Processing the call arguments (line 772)
                    str_2010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 32), 'str', 'end must be None or a string')
                    # Processing the call keyword arguments (line 772)
                    kwargs_2011 = {}
                    # Getting the type of 'TypeError' (line 772)
                    TypeError_2009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 22), 'TypeError', False)
                    # Calling TypeError(args, kwargs) (line 772)
                    TypeError_call_result_2012 = invoke(stypy.reporting.localization.Localization(__file__, 772, 22), TypeError_2009, *[str_2010], **kwargs_2011)
                    
                    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 772, 16), TypeError_call_result_2012, 'raise parameter', BaseException)

                    if more_types_in_union_2008:
                        # SSA join for if statement (line 771)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_2002 and more_types_in_union_2003):
                    # SSA join for if statement (line 769)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_1999:
                # SSA join for if statement (line 768)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'kwargs' (line 773)
        kwargs_2013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 11), 'kwargs')
        # Testing the type of an if condition (line 773)
        if_condition_2014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 773, 8), kwargs_2013)
        # Assigning a type to the variable 'if_condition_2014' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'if_condition_2014', if_condition_2014)
        # SSA begins for if statement (line 773)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 774)
        # Processing the call arguments (line 774)
        str_2016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 28), 'str', 'invalid keyword arguments to print()')
        # Processing the call keyword arguments (line 774)
        kwargs_2017 = {}
        # Getting the type of 'TypeError' (line 774)
        TypeError_2015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 774)
        TypeError_call_result_2018 = invoke(stypy.reporting.localization.Localization(__file__, 774, 18), TypeError_2015, *[str_2016], **kwargs_2017)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 774, 12), TypeError_call_result_2018, 'raise parameter', BaseException)
        # SSA join for if statement (line 773)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'want_unicode' (line 775)
        want_unicode_2019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 15), 'want_unicode')
        # Applying the 'not' unary operator (line 775)
        result_not__2020 = python_operator(stypy.reporting.localization.Localization(__file__, 775, 11), 'not', want_unicode_2019)
        
        # Testing the type of an if condition (line 775)
        if_condition_2021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 775, 8), result_not__2020)
        # Assigning a type to the variable 'if_condition_2021' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'if_condition_2021', if_condition_2021)
        # SSA begins for if statement (line 775)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'args' (line 776)
        args_2022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 23), 'args')
        # Testing the type of a for loop iterable (line 776)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 776, 12), args_2022)
        # Getting the type of the for loop variable (line 776)
        for_loop_var_2023 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 776, 12), args_2022)
        # Assigning a type to the variable 'arg' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 12), 'arg', for_loop_var_2023)
        # SSA begins for a for statement (line 776)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 777)
        # Getting the type of 'unicode' (line 777)
        unicode_2024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 35), 'unicode')
        # Getting the type of 'arg' (line 777)
        arg_2025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 30), 'arg')
        
        (may_be_2026, more_types_in_union_2027) = may_be_subtype(unicode_2024, arg_2025)

        if may_be_2026:

            if more_types_in_union_2027:
                # Runtime conditional SSA (line 777)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'arg' (line 777)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 16), 'arg', remove_not_subtype_from_union(arg_2025, unicode))
            
            # Assigning a Name to a Name (line 778):
            # Getting the type of 'True' (line 778)
            True_2028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 35), 'True')
            # Assigning a type to the variable 'want_unicode' (line 778)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 20), 'want_unicode', True_2028)

            if more_types_in_union_2027:
                # SSA join for if statement (line 777)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 775)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'want_unicode' (line 780)
        want_unicode_2029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 11), 'want_unicode')
        # Testing the type of an if condition (line 780)
        if_condition_2030 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 780, 8), want_unicode_2029)
        # Assigning a type to the variable 'if_condition_2030' (line 780)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'if_condition_2030', if_condition_2030)
        # SSA begins for if statement (line 780)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 781):
        
        # Call to unicode(...): (line 781)
        # Processing the call arguments (line 781)
        str_2032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 30), 'str', '\n')
        # Processing the call keyword arguments (line 781)
        kwargs_2033 = {}
        # Getting the type of 'unicode' (line 781)
        unicode_2031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 22), 'unicode', False)
        # Calling unicode(args, kwargs) (line 781)
        unicode_call_result_2034 = invoke(stypy.reporting.localization.Localization(__file__, 781, 22), unicode_2031, *[str_2032], **kwargs_2033)
        
        # Assigning a type to the variable 'newline' (line 781)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'newline', unicode_call_result_2034)
        
        # Assigning a Call to a Name (line 782):
        
        # Call to unicode(...): (line 782)
        # Processing the call arguments (line 782)
        str_2036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 28), 'str', ' ')
        # Processing the call keyword arguments (line 782)
        kwargs_2037 = {}
        # Getting the type of 'unicode' (line 782)
        unicode_2035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 20), 'unicode', False)
        # Calling unicode(args, kwargs) (line 782)
        unicode_call_result_2038 = invoke(stypy.reporting.localization.Localization(__file__, 782, 20), unicode_2035, *[str_2036], **kwargs_2037)
        
        # Assigning a type to the variable 'space' (line 782)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 12), 'space', unicode_call_result_2038)
        # SSA branch for the else part of an if statement (line 780)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 784):
        str_2039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 22), 'str', '\n')
        # Assigning a type to the variable 'newline' (line 784)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 12), 'newline', str_2039)
        
        # Assigning a Str to a Name (line 785):
        str_2040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 20), 'str', ' ')
        # Assigning a type to the variable 'space' (line 785)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'space', str_2040)
        # SSA join for if statement (line 780)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 786)
        # Getting the type of 'sep' (line 786)
        sep_2041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 11), 'sep')
        # Getting the type of 'None' (line 786)
        None_2042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 18), 'None')
        
        (may_be_2043, more_types_in_union_2044) = may_be_none(sep_2041, None_2042)

        if may_be_2043:

            if more_types_in_union_2044:
                # Runtime conditional SSA (line 786)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 787):
            # Getting the type of 'space' (line 787)
            space_2045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 18), 'space')
            # Assigning a type to the variable 'sep' (line 787)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 12), 'sep', space_2045)

            if more_types_in_union_2044:
                # SSA join for if statement (line 786)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 788)
        # Getting the type of 'end' (line 788)
        end_2046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 11), 'end')
        # Getting the type of 'None' (line 788)
        None_2047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 18), 'None')
        
        (may_be_2048, more_types_in_union_2049) = may_be_none(end_2046, None_2047)

        if may_be_2048:

            if more_types_in_union_2049:
                # Runtime conditional SSA (line 788)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 789):
            # Getting the type of 'newline' (line 789)
            newline_2050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 18), 'newline')
            # Assigning a type to the variable 'end' (line 789)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 12), 'end', newline_2050)

            if more_types_in_union_2049:
                # SSA join for if statement (line 788)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to enumerate(...): (line 790)
        # Processing the call arguments (line 790)
        # Getting the type of 'args' (line 790)
        args_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 32), 'args', False)
        # Processing the call keyword arguments (line 790)
        kwargs_2053 = {}
        # Getting the type of 'enumerate' (line 790)
        enumerate_2051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 22), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 790)
        enumerate_call_result_2054 = invoke(stypy.reporting.localization.Localization(__file__, 790, 22), enumerate_2051, *[args_2052], **kwargs_2053)
        
        # Testing the type of a for loop iterable (line 790)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 790, 8), enumerate_call_result_2054)
        # Getting the type of the for loop variable (line 790)
        for_loop_var_2055 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 790, 8), enumerate_call_result_2054)
        # Assigning a type to the variable 'i' (line 790)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 8), for_loop_var_2055))
        # Assigning a type to the variable 'arg' (line 790)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 8), for_loop_var_2055))
        # SSA begins for a for statement (line 790)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'i' (line 791)
        i_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 15), 'i')
        # Testing the type of an if condition (line 791)
        if_condition_2057 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 791, 12), i_2056)
        # Assigning a type to the variable 'if_condition_2057' (line 791)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'if_condition_2057', if_condition_2057)
        # SSA begins for if statement (line 791)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 792)
        # Processing the call arguments (line 792)
        # Getting the type of 'sep' (line 792)
        sep_2059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 22), 'sep', False)
        # Processing the call keyword arguments (line 792)
        kwargs_2060 = {}
        # Getting the type of 'write' (line 792)
        write_2058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 16), 'write', False)
        # Calling write(args, kwargs) (line 792)
        write_call_result_2061 = invoke(stypy.reporting.localization.Localization(__file__, 792, 16), write_2058, *[sep_2059], **kwargs_2060)
        
        # SSA join for if statement (line 791)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 793)
        # Processing the call arguments (line 793)
        # Getting the type of 'arg' (line 793)
        arg_2063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 18), 'arg', False)
        # Processing the call keyword arguments (line 793)
        kwargs_2064 = {}
        # Getting the type of 'write' (line 793)
        write_2062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 12), 'write', False)
        # Calling write(args, kwargs) (line 793)
        write_call_result_2065 = invoke(stypy.reporting.localization.Localization(__file__, 793, 12), write_2062, *[arg_2063], **kwargs_2064)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 794)
        # Processing the call arguments (line 794)
        # Getting the type of 'end' (line 794)
        end_2067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 14), 'end', False)
        # Processing the call keyword arguments (line 794)
        kwargs_2068 = {}
        # Getting the type of 'write' (line 794)
        write_2066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'write', False)
        # Calling write(args, kwargs) (line 794)
        write_call_result_2069 = invoke(stypy.reporting.localization.Localization(__file__, 794, 8), write_2066, *[end_2067], **kwargs_2068)
        
        
        # ################# End of 'print_(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_' in the type store
        # Getting the type of 'stypy_return_type' (line 742)
        stypy_return_type_2070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2070)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_'
        return stypy_return_type_2070

    # Assigning a type to the variable 'print_' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'print_', print_)

    if more_types_in_union_1904:
        # SSA join for if statement (line 741)
        module_type_store = module_type_store.join_ssa_context()






# Obtaining the type of the subscript
int_2071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 21), 'int')
slice_2072 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 795, 3), None, int_2071, None)
# Getting the type of 'sys' (line 795)
sys_2073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 795)
version_info_2074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 3), sys_2073, 'version_info')
# Obtaining the member '__getitem__' of a type (line 795)
getitem___2075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 3), version_info_2074, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 795)
subscript_call_result_2076 = invoke(stypy.reporting.localization.Localization(__file__, 795, 3), getitem___2075, slice_2072)


# Obtaining an instance of the builtin type 'tuple' (line 795)
tuple_2077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 27), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 795)
# Adding element type (line 795)
int_2078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 795, 27), tuple_2077, int_2078)
# Adding element type (line 795)
int_2079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 795, 27), tuple_2077, int_2079)

# Applying the binary operator '<' (line 795)
result_lt_2080 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 3), '<', subscript_call_result_2076, tuple_2077)

# Testing the type of an if condition (line 795)
if_condition_2081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 795, 0), result_lt_2080)
# Assigning a type to the variable 'if_condition_2081' (line 795)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 0), 'if_condition_2081', if_condition_2081)
# SSA begins for if statement (line 795)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 796):
# Getting the type of 'print_' (line 796)
print__2082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 13), 'print_')
# Assigning a type to the variable '_print' (line 796)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), '_print', print__2082)

@norecursion
def print_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_'
    module_type_store = module_type_store.open_function_context('print_', 798, 4, False)
    
    # Passed parameters checking function
    print_.stypy_localization = localization
    print_.stypy_type_of_self = None
    print_.stypy_type_store = module_type_store
    print_.stypy_function_name = 'print_'
    print_.stypy_param_names_list = []
    print_.stypy_varargs_param_name = 'args'
    print_.stypy_kwargs_param_name = 'kwargs'
    print_.stypy_call_defaults = defaults
    print_.stypy_call_varargs = varargs
    print_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_', [], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_(...)' code ##################

    
    # Assigning a Call to a Name (line 799):
    
    # Call to get(...): (line 799)
    # Processing the call arguments (line 799)
    str_2085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 24), 'str', 'file')
    # Getting the type of 'sys' (line 799)
    sys_2086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 32), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 799)
    stdout_2087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 32), sys_2086, 'stdout')
    # Processing the call keyword arguments (line 799)
    kwargs_2088 = {}
    # Getting the type of 'kwargs' (line 799)
    kwargs_2083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 13), 'kwargs', False)
    # Obtaining the member 'get' of a type (line 799)
    get_2084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 13), kwargs_2083, 'get')
    # Calling get(args, kwargs) (line 799)
    get_call_result_2089 = invoke(stypy.reporting.localization.Localization(__file__, 799, 13), get_2084, *[str_2085, stdout_2087], **kwargs_2088)
    
    # Assigning a type to the variable 'fp' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 8), 'fp', get_call_result_2089)
    
    # Assigning a Call to a Name (line 800):
    
    # Call to pop(...): (line 800)
    # Processing the call arguments (line 800)
    str_2092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 27), 'str', 'flush')
    # Getting the type of 'False' (line 800)
    False_2093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 36), 'False', False)
    # Processing the call keyword arguments (line 800)
    kwargs_2094 = {}
    # Getting the type of 'kwargs' (line 800)
    kwargs_2090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 16), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 800)
    pop_2091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 16), kwargs_2090, 'pop')
    # Calling pop(args, kwargs) (line 800)
    pop_call_result_2095 = invoke(stypy.reporting.localization.Localization(__file__, 800, 16), pop_2091, *[str_2092, False_2093], **kwargs_2094)
    
    # Assigning a type to the variable 'flush' (line 800)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 8), 'flush', pop_call_result_2095)
    
    # Call to _print(...): (line 801)
    # Getting the type of 'args' (line 801)
    args_2097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 16), 'args', False)
    # Processing the call keyword arguments (line 801)
    # Getting the type of 'kwargs' (line 801)
    kwargs_2098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 24), 'kwargs', False)
    kwargs_2099 = {'kwargs_2098': kwargs_2098}
    # Getting the type of '_print' (line 801)
    _print_2096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 8), '_print', False)
    # Calling _print(args, kwargs) (line 801)
    _print_call_result_2100 = invoke(stypy.reporting.localization.Localization(__file__, 801, 8), _print_2096, *[args_2097], **kwargs_2099)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'flush' (line 802)
    flush_2101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 11), 'flush')
    
    # Getting the type of 'fp' (line 802)
    fp_2102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 21), 'fp')
    # Getting the type of 'None' (line 802)
    None_2103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 31), 'None')
    # Applying the binary operator 'isnot' (line 802)
    result_is_not_2104 = python_operator(stypy.reporting.localization.Localization(__file__, 802, 21), 'isnot', fp_2102, None_2103)
    
    # Applying the binary operator 'and' (line 802)
    result_and_keyword_2105 = python_operator(stypy.reporting.localization.Localization(__file__, 802, 11), 'and', flush_2101, result_is_not_2104)
    
    # Testing the type of an if condition (line 802)
    if_condition_2106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 802, 8), result_and_keyword_2105)
    # Assigning a type to the variable 'if_condition_2106' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 8), 'if_condition_2106', if_condition_2106)
    # SSA begins for if statement (line 802)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to flush(...): (line 803)
    # Processing the call keyword arguments (line 803)
    kwargs_2109 = {}
    # Getting the type of 'fp' (line 803)
    fp_2107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 12), 'fp', False)
    # Obtaining the member 'flush' of a type (line 803)
    flush_2108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 12), fp_2107, 'flush')
    # Calling flush(args, kwargs) (line 803)
    flush_call_result_2110 = invoke(stypy.reporting.localization.Localization(__file__, 803, 12), flush_2108, *[], **kwargs_2109)
    
    # SSA join for if statement (line 802)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'print_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_' in the type store
    # Getting the type of 'stypy_return_type' (line 798)
    stypy_return_type_2111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2111)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_'
    return stypy_return_type_2111

# Assigning a type to the variable 'print_' (line 798)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'print_', print_)
# SSA join for if statement (line 795)
module_type_store = module_type_store.join_ssa_context()


# Call to _add_doc(...): (line 805)
# Processing the call arguments (line 805)
# Getting the type of 'reraise' (line 805)
reraise_2113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 9), 'reraise', False)
str_2114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 18), 'str', 'Reraise an exception.')
# Processing the call keyword arguments (line 805)
kwargs_2115 = {}
# Getting the type of '_add_doc' (line 805)
_add_doc_2112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 805)
_add_doc_call_result_2116 = invoke(stypy.reporting.localization.Localization(__file__, 805, 0), _add_doc_2112, *[reraise_2113, str_2114], **kwargs_2115)




# Obtaining the type of the subscript
int_2117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 20), 'int')
int_2118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 22), 'int')
slice_2119 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 807, 3), int_2117, int_2118, None)
# Getting the type of 'sys' (line 807)
sys_2120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 807)
version_info_2121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 3), sys_2120, 'version_info')
# Obtaining the member '__getitem__' of a type (line 807)
getitem___2122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 3), version_info_2121, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 807)
subscript_call_result_2123 = invoke(stypy.reporting.localization.Localization(__file__, 807, 3), getitem___2122, slice_2119)


# Obtaining an instance of the builtin type 'tuple' (line 807)
tuple_2124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 807)
# Adding element type (line 807)
int_2125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 28), tuple_2124, int_2125)
# Adding element type (line 807)
int_2126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 28), tuple_2124, int_2126)

# Applying the binary operator '<' (line 807)
result_lt_2127 = python_operator(stypy.reporting.localization.Localization(__file__, 807, 3), '<', subscript_call_result_2123, tuple_2124)

# Testing the type of an if condition (line 807)
if_condition_2128 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 807, 0), result_lt_2127)
# Assigning a type to the variable 'if_condition_2128' (line 807)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 0), 'if_condition_2128', if_condition_2128)
# SSA begins for if statement (line 807)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def wraps(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'functools' (line 808)
    functools_2129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 32), 'functools')
    # Obtaining the member 'WRAPPER_ASSIGNMENTS' of a type (line 808)
    WRAPPER_ASSIGNMENTS_2130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 32), functools_2129, 'WRAPPER_ASSIGNMENTS')
    # Getting the type of 'functools' (line 809)
    functools_2131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 22), 'functools')
    # Obtaining the member 'WRAPPER_UPDATES' of a type (line 809)
    WRAPPER_UPDATES_2132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 22), functools_2131, 'WRAPPER_UPDATES')
    defaults = [WRAPPER_ASSIGNMENTS_2130, WRAPPER_UPDATES_2132]
    # Create a new context for function 'wraps'
    module_type_store = module_type_store.open_function_context('wraps', 808, 4, False)
    
    # Passed parameters checking function
    wraps.stypy_localization = localization
    wraps.stypy_type_of_self = None
    wraps.stypy_type_store = module_type_store
    wraps.stypy_function_name = 'wraps'
    wraps.stypy_param_names_list = ['wrapped', 'assigned', 'updated']
    wraps.stypy_varargs_param_name = None
    wraps.stypy_kwargs_param_name = None
    wraps.stypy_call_defaults = defaults
    wraps.stypy_call_varargs = varargs
    wraps.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'wraps', ['wrapped', 'assigned', 'updated'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'wraps', localization, ['wrapped', 'assigned', 'updated'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'wraps(...)' code ##################


    @norecursion
    def wrapper(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrapper'
        module_type_store = module_type_store.open_function_context('wrapper', 810, 8, False)
        
        # Passed parameters checking function
        wrapper.stypy_localization = localization
        wrapper.stypy_type_of_self = None
        wrapper.stypy_type_store = module_type_store
        wrapper.stypy_function_name = 'wrapper'
        wrapper.stypy_param_names_list = ['f']
        wrapper.stypy_varargs_param_name = None
        wrapper.stypy_kwargs_param_name = None
        wrapper.stypy_call_defaults = defaults
        wrapper.stypy_call_varargs = varargs
        wrapper.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrapper', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrapper', localization, ['f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrapper(...)' code ##################

        
        # Assigning a Call to a Name (line 811):
        
        # Call to (...): (line 811)
        # Processing the call arguments (line 811)
        # Getting the type of 'f' (line 811)
        f_2140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 60), 'f', False)
        # Processing the call keyword arguments (line 811)
        kwargs_2141 = {}
        
        # Call to wraps(...): (line 811)
        # Processing the call arguments (line 811)
        # Getting the type of 'wrapped' (line 811)
        wrapped_2135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 32), 'wrapped', False)
        # Getting the type of 'assigned' (line 811)
        assigned_2136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 41), 'assigned', False)
        # Getting the type of 'updated' (line 811)
        updated_2137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 51), 'updated', False)
        # Processing the call keyword arguments (line 811)
        kwargs_2138 = {}
        # Getting the type of 'functools' (line 811)
        functools_2133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 16), 'functools', False)
        # Obtaining the member 'wraps' of a type (line 811)
        wraps_2134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 16), functools_2133, 'wraps')
        # Calling wraps(args, kwargs) (line 811)
        wraps_call_result_2139 = invoke(stypy.reporting.localization.Localization(__file__, 811, 16), wraps_2134, *[wrapped_2135, assigned_2136, updated_2137], **kwargs_2138)
        
        # Calling (args, kwargs) (line 811)
        _call_result_2142 = invoke(stypy.reporting.localization.Localization(__file__, 811, 16), wraps_call_result_2139, *[f_2140], **kwargs_2141)
        
        # Assigning a type to the variable 'f' (line 811)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 12), 'f', _call_result_2142)
        
        # Assigning a Name to a Attribute (line 812):
        # Getting the type of 'wrapped' (line 812)
        wrapped_2143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 28), 'wrapped')
        # Getting the type of 'f' (line 812)
        f_2144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'f')
        # Setting the type of the member '__wrapped__' of a type (line 812)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 12), f_2144, '__wrapped__', wrapped_2143)
        # Getting the type of 'f' (line 813)
        f_2145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 19), 'f')
        # Assigning a type to the variable 'stypy_return_type' (line 813)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 12), 'stypy_return_type', f_2145)
        
        # ################# End of 'wrapper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrapper' in the type store
        # Getting the type of 'stypy_return_type' (line 810)
        stypy_return_type_2146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2146)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrapper'
        return stypy_return_type_2146

    # Assigning a type to the variable 'wrapper' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'wrapper', wrapper)
    # Getting the type of 'wrapper' (line 814)
    wrapper_2147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 15), 'wrapper')
    # Assigning a type to the variable 'stypy_return_type' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 8), 'stypy_return_type', wrapper_2147)
    
    # ################# End of 'wraps(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'wraps' in the type store
    # Getting the type of 'stypy_return_type' (line 808)
    stypy_return_type_2148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2148)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'wraps'
    return stypy_return_type_2148

# Assigning a type to the variable 'wraps' (line 808)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'wraps', wraps)
# SSA branch for the else part of an if statement (line 807)
module_type_store.open_ssa_branch('else')

# Assigning a Attribute to a Name (line 816):
# Getting the type of 'functools' (line 816)
functools_2149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 12), 'functools')
# Obtaining the member 'wraps' of a type (line 816)
wraps_2150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 12), functools_2149, 'wraps')
# Assigning a type to the variable 'wraps' (line 816)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 4), 'wraps', wraps_2150)
# SSA join for if statement (line 807)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def with_metaclass(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'with_metaclass'
    module_type_store = module_type_store.open_function_context('with_metaclass', 819, 0, False)
    
    # Passed parameters checking function
    with_metaclass.stypy_localization = localization
    with_metaclass.stypy_type_of_self = None
    with_metaclass.stypy_type_store = module_type_store
    with_metaclass.stypy_function_name = 'with_metaclass'
    with_metaclass.stypy_param_names_list = ['meta']
    with_metaclass.stypy_varargs_param_name = 'bases'
    with_metaclass.stypy_kwargs_param_name = None
    with_metaclass.stypy_call_defaults = defaults
    with_metaclass.stypy_call_varargs = varargs
    with_metaclass.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'with_metaclass', ['meta'], 'bases', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'with_metaclass', localization, ['meta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'with_metaclass(...)' code ##################

    str_2151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 4), 'str', 'Create a base class with a metaclass.')
    # Declaration of the 'metaclass' class
    # Getting the type of 'type' (line 824)
    type_2152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 20), 'type')

    class metaclass(type_2152, ):

        @norecursion
        def __new__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__new__'
            module_type_store = module_type_store.open_function_context('__new__', 826, 8, False)
            # Assigning a type to the variable 'self' (line 827)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            metaclass.__new__.__dict__.__setitem__('stypy_localization', localization)
            metaclass.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            metaclass.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
            metaclass.__new__.__dict__.__setitem__('stypy_function_name', 'metaclass.__new__')
            metaclass.__new__.__dict__.__setitem__('stypy_param_names_list', ['name', 'this_bases', 'd'])
            metaclass.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
            metaclass.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            metaclass.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
            metaclass.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
            metaclass.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            metaclass.__new__.__dict__.__setitem__('stypy_declared_arg_number', 4)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'metaclass.__new__', ['name', 'this_bases', 'd'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__new__', localization, ['name', 'this_bases', 'd'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__new__(...)' code ##################

            
            # Call to meta(...): (line 827)
            # Processing the call arguments (line 827)
            # Getting the type of 'name' (line 827)
            name_2154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 24), 'name', False)
            # Getting the type of 'bases' (line 827)
            bases_2155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 30), 'bases', False)
            # Getting the type of 'd' (line 827)
            d_2156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 37), 'd', False)
            # Processing the call keyword arguments (line 827)
            kwargs_2157 = {}
            # Getting the type of 'meta' (line 827)
            meta_2153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 19), 'meta', False)
            # Calling meta(args, kwargs) (line 827)
            meta_call_result_2158 = invoke(stypy.reporting.localization.Localization(__file__, 827, 19), meta_2153, *[name_2154, bases_2155, d_2156], **kwargs_2157)
            
            # Assigning a type to the variable 'stypy_return_type' (line 827)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 12), 'stypy_return_type', meta_call_result_2158)
            
            # ################# End of '__new__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__new__' in the type store
            # Getting the type of 'stypy_return_type' (line 826)
            stypy_return_type_2159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2159)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__new__'
            return stypy_return_type_2159


        @norecursion
        def __prepare__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__prepare__'
            module_type_store = module_type_store.open_function_context('__prepare__', 829, 8, False)
            # Assigning a type to the variable 'self' (line 830)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            metaclass.__prepare__.__dict__.__setitem__('stypy_localization', localization)
            metaclass.__prepare__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            metaclass.__prepare__.__dict__.__setitem__('stypy_type_store', module_type_store)
            metaclass.__prepare__.__dict__.__setitem__('stypy_function_name', 'metaclass.__prepare__')
            metaclass.__prepare__.__dict__.__setitem__('stypy_param_names_list', ['name', 'this_bases'])
            metaclass.__prepare__.__dict__.__setitem__('stypy_varargs_param_name', None)
            metaclass.__prepare__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            metaclass.__prepare__.__dict__.__setitem__('stypy_call_defaults', defaults)
            metaclass.__prepare__.__dict__.__setitem__('stypy_call_varargs', varargs)
            metaclass.__prepare__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            metaclass.__prepare__.__dict__.__setitem__('stypy_declared_arg_number', 3)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'metaclass.__prepare__', ['name', 'this_bases'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__prepare__', localization, ['name', 'this_bases'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__prepare__(...)' code ##################

            
            # Call to __prepare__(...): (line 831)
            # Processing the call arguments (line 831)
            # Getting the type of 'name' (line 831)
            name_2162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 36), 'name', False)
            # Getting the type of 'bases' (line 831)
            bases_2163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 42), 'bases', False)
            # Processing the call keyword arguments (line 831)
            kwargs_2164 = {}
            # Getting the type of 'meta' (line 831)
            meta_2160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 19), 'meta', False)
            # Obtaining the member '__prepare__' of a type (line 831)
            prepare___2161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 19), meta_2160, '__prepare__')
            # Calling __prepare__(args, kwargs) (line 831)
            prepare___call_result_2165 = invoke(stypy.reporting.localization.Localization(__file__, 831, 19), prepare___2161, *[name_2162, bases_2163], **kwargs_2164)
            
            # Assigning a type to the variable 'stypy_return_type' (line 831)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 12), 'stypy_return_type', prepare___call_result_2165)
            
            # ################# End of '__prepare__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__prepare__' in the type store
            # Getting the type of 'stypy_return_type' (line 829)
            stypy_return_type_2166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2166)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__prepare__'
            return stypy_return_type_2166


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 824, 4, False)
            # Assigning a type to the variable 'self' (line 825)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'metaclass.__init__', [], None, None, defaults, varargs, kwargs)

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

            pass
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'metaclass' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'metaclass', metaclass)
    
    # Call to __new__(...): (line 832)
    # Processing the call arguments (line 832)
    # Getting the type of 'metaclass' (line 832)
    metaclass_2169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 24), 'metaclass', False)
    str_2170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 35), 'str', 'temporary_class')
    
    # Obtaining an instance of the builtin type 'tuple' (line 832)
    tuple_2171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 832)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 832)
    dict_2172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 58), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 832)
    
    # Processing the call keyword arguments (line 832)
    kwargs_2173 = {}
    # Getting the type of 'type' (line 832)
    type_2167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 11), 'type', False)
    # Obtaining the member '__new__' of a type (line 832)
    new___2168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 11), type_2167, '__new__')
    # Calling __new__(args, kwargs) (line 832)
    new___call_result_2174 = invoke(stypy.reporting.localization.Localization(__file__, 832, 11), new___2168, *[metaclass_2169, str_2170, tuple_2171, dict_2172], **kwargs_2173)
    
    # Assigning a type to the variable 'stypy_return_type' (line 832)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 4), 'stypy_return_type', new___call_result_2174)
    
    # ################# End of 'with_metaclass(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'with_metaclass' in the type store
    # Getting the type of 'stypy_return_type' (line 819)
    stypy_return_type_2175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2175)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'with_metaclass'
    return stypy_return_type_2175

# Assigning a type to the variable 'with_metaclass' (line 819)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 0), 'with_metaclass', with_metaclass)

@norecursion
def add_metaclass(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'add_metaclass'
    module_type_store = module_type_store.open_function_context('add_metaclass', 835, 0, False)
    
    # Passed parameters checking function
    add_metaclass.stypy_localization = localization
    add_metaclass.stypy_type_of_self = None
    add_metaclass.stypy_type_store = module_type_store
    add_metaclass.stypy_function_name = 'add_metaclass'
    add_metaclass.stypy_param_names_list = ['metaclass']
    add_metaclass.stypy_varargs_param_name = None
    add_metaclass.stypy_kwargs_param_name = None
    add_metaclass.stypy_call_defaults = defaults
    add_metaclass.stypy_call_varargs = varargs
    add_metaclass.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_metaclass', ['metaclass'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_metaclass', localization, ['metaclass'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_metaclass(...)' code ##################

    str_2176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 4), 'str', 'Class decorator for creating a class with a metaclass.')

    @norecursion
    def wrapper(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrapper'
        module_type_store = module_type_store.open_function_context('wrapper', 837, 4, False)
        
        # Passed parameters checking function
        wrapper.stypy_localization = localization
        wrapper.stypy_type_of_self = None
        wrapper.stypy_type_store = module_type_store
        wrapper.stypy_function_name = 'wrapper'
        wrapper.stypy_param_names_list = ['cls']
        wrapper.stypy_varargs_param_name = None
        wrapper.stypy_kwargs_param_name = None
        wrapper.stypy_call_defaults = defaults
        wrapper.stypy_call_varargs = varargs
        wrapper.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrapper', ['cls'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrapper', localization, ['cls'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrapper(...)' code ##################

        
        # Assigning a Call to a Name (line 838):
        
        # Call to copy(...): (line 838)
        # Processing the call keyword arguments (line 838)
        kwargs_2180 = {}
        # Getting the type of 'cls' (line 838)
        cls_2177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 20), 'cls', False)
        # Obtaining the member '__dict__' of a type (line 838)
        dict___2178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 20), cls_2177, '__dict__')
        # Obtaining the member 'copy' of a type (line 838)
        copy_2179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 20), dict___2178, 'copy')
        # Calling copy(args, kwargs) (line 838)
        copy_call_result_2181 = invoke(stypy.reporting.localization.Localization(__file__, 838, 20), copy_2179, *[], **kwargs_2180)
        
        # Assigning a type to the variable 'orig_vars' (line 838)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'orig_vars', copy_call_result_2181)
        
        # Assigning a Call to a Name (line 839):
        
        # Call to get(...): (line 839)
        # Processing the call arguments (line 839)
        str_2184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 30), 'str', '__slots__')
        # Processing the call keyword arguments (line 839)
        kwargs_2185 = {}
        # Getting the type of 'orig_vars' (line 839)
        orig_vars_2182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 16), 'orig_vars', False)
        # Obtaining the member 'get' of a type (line 839)
        get_2183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 16), orig_vars_2182, 'get')
        # Calling get(args, kwargs) (line 839)
        get_call_result_2186 = invoke(stypy.reporting.localization.Localization(__file__, 839, 16), get_2183, *[str_2184], **kwargs_2185)
        
        # Assigning a type to the variable 'slots' (line 839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'slots', get_call_result_2186)
        
        # Type idiom detected: calculating its left and rigth part (line 840)
        # Getting the type of 'slots' (line 840)
        slots_2187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'slots')
        # Getting the type of 'None' (line 840)
        None_2188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 24), 'None')
        
        (may_be_2189, more_types_in_union_2190) = may_not_be_none(slots_2187, None_2188)

        if may_be_2189:

            if more_types_in_union_2190:
                # Runtime conditional SSA (line 840)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 841)
            # Getting the type of 'str' (line 841)
            str_2191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 33), 'str')
            # Getting the type of 'slots' (line 841)
            slots_2192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 26), 'slots')
            
            (may_be_2193, more_types_in_union_2194) = may_be_subtype(str_2191, slots_2192)

            if may_be_2193:

                if more_types_in_union_2194:
                    # Runtime conditional SSA (line 841)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'slots' (line 841)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 12), 'slots', remove_not_subtype_from_union(slots_2192, str))
                
                # Assigning a List to a Name (line 842):
                
                # Obtaining an instance of the builtin type 'list' (line 842)
                list_2195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 24), 'list')
                # Adding type elements to the builtin type 'list' instance (line 842)
                # Adding element type (line 842)
                # Getting the type of 'slots' (line 842)
                slots_2196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 25), 'slots')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 24), list_2195, slots_2196)
                
                # Assigning a type to the variable 'slots' (line 842)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 16), 'slots', list_2195)

                if more_types_in_union_2194:
                    # SSA join for if statement (line 841)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Getting the type of 'slots' (line 843)
            slots_2197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 29), 'slots')
            # Testing the type of a for loop iterable (line 843)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 843, 12), slots_2197)
            # Getting the type of the for loop variable (line 843)
            for_loop_var_2198 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 843, 12), slots_2197)
            # Assigning a type to the variable 'slots_var' (line 843)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'slots_var', for_loop_var_2198)
            # SSA begins for a for statement (line 843)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to pop(...): (line 844)
            # Processing the call arguments (line 844)
            # Getting the type of 'slots_var' (line 844)
            slots_var_2201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 30), 'slots_var', False)
            # Processing the call keyword arguments (line 844)
            kwargs_2202 = {}
            # Getting the type of 'orig_vars' (line 844)
            orig_vars_2199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 16), 'orig_vars', False)
            # Obtaining the member 'pop' of a type (line 844)
            pop_2200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 16), orig_vars_2199, 'pop')
            # Calling pop(args, kwargs) (line 844)
            pop_call_result_2203 = invoke(stypy.reporting.localization.Localization(__file__, 844, 16), pop_2200, *[slots_var_2201], **kwargs_2202)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_2190:
                # SSA join for if statement (line 840)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to pop(...): (line 845)
        # Processing the call arguments (line 845)
        str_2206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 22), 'str', '__dict__')
        # Getting the type of 'None' (line 845)
        None_2207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 34), 'None', False)
        # Processing the call keyword arguments (line 845)
        kwargs_2208 = {}
        # Getting the type of 'orig_vars' (line 845)
        orig_vars_2204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 8), 'orig_vars', False)
        # Obtaining the member 'pop' of a type (line 845)
        pop_2205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 8), orig_vars_2204, 'pop')
        # Calling pop(args, kwargs) (line 845)
        pop_call_result_2209 = invoke(stypy.reporting.localization.Localization(__file__, 845, 8), pop_2205, *[str_2206, None_2207], **kwargs_2208)
        
        
        # Call to pop(...): (line 846)
        # Processing the call arguments (line 846)
        str_2212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 22), 'str', '__weakref__')
        # Getting the type of 'None' (line 846)
        None_2213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 37), 'None', False)
        # Processing the call keyword arguments (line 846)
        kwargs_2214 = {}
        # Getting the type of 'orig_vars' (line 846)
        orig_vars_2210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'orig_vars', False)
        # Obtaining the member 'pop' of a type (line 846)
        pop_2211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 8), orig_vars_2210, 'pop')
        # Calling pop(args, kwargs) (line 846)
        pop_call_result_2215 = invoke(stypy.reporting.localization.Localization(__file__, 846, 8), pop_2211, *[str_2212, None_2213], **kwargs_2214)
        
        
        # Call to metaclass(...): (line 847)
        # Processing the call arguments (line 847)
        # Getting the type of 'cls' (line 847)
        cls_2217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 25), 'cls', False)
        # Obtaining the member '__name__' of a type (line 847)
        name___2218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 25), cls_2217, '__name__')
        # Getting the type of 'cls' (line 847)
        cls_2219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 39), 'cls', False)
        # Obtaining the member '__bases__' of a type (line 847)
        bases___2220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 39), cls_2219, '__bases__')
        # Getting the type of 'orig_vars' (line 847)
        orig_vars_2221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 54), 'orig_vars', False)
        # Processing the call keyword arguments (line 847)
        kwargs_2222 = {}
        # Getting the type of 'metaclass' (line 847)
        metaclass_2216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 15), 'metaclass', False)
        # Calling metaclass(args, kwargs) (line 847)
        metaclass_call_result_2223 = invoke(stypy.reporting.localization.Localization(__file__, 847, 15), metaclass_2216, *[name___2218, bases___2220, orig_vars_2221], **kwargs_2222)
        
        # Assigning a type to the variable 'stypy_return_type' (line 847)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 8), 'stypy_return_type', metaclass_call_result_2223)
        
        # ################# End of 'wrapper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrapper' in the type store
        # Getting the type of 'stypy_return_type' (line 837)
        stypy_return_type_2224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2224)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrapper'
        return stypy_return_type_2224

    # Assigning a type to the variable 'wrapper' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'wrapper', wrapper)
    # Getting the type of 'wrapper' (line 848)
    wrapper_2225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 11), 'wrapper')
    # Assigning a type to the variable 'stypy_return_type' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 4), 'stypy_return_type', wrapper_2225)
    
    # ################# End of 'add_metaclass(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_metaclass' in the type store
    # Getting the type of 'stypy_return_type' (line 835)
    stypy_return_type_2226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2226)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_metaclass'
    return stypy_return_type_2226

# Assigning a type to the variable 'add_metaclass' (line 835)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 0), 'add_metaclass', add_metaclass)

@norecursion
def python_2_unicode_compatible(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'python_2_unicode_compatible'
    module_type_store = module_type_store.open_function_context('python_2_unicode_compatible', 851, 0, False)
    
    # Passed parameters checking function
    python_2_unicode_compatible.stypy_localization = localization
    python_2_unicode_compatible.stypy_type_of_self = None
    python_2_unicode_compatible.stypy_type_store = module_type_store
    python_2_unicode_compatible.stypy_function_name = 'python_2_unicode_compatible'
    python_2_unicode_compatible.stypy_param_names_list = ['klass']
    python_2_unicode_compatible.stypy_varargs_param_name = None
    python_2_unicode_compatible.stypy_kwargs_param_name = None
    python_2_unicode_compatible.stypy_call_defaults = defaults
    python_2_unicode_compatible.stypy_call_varargs = varargs
    python_2_unicode_compatible.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'python_2_unicode_compatible', ['klass'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'python_2_unicode_compatible', localization, ['klass'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'python_2_unicode_compatible(...)' code ##################

    str_2227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, (-1)), 'str', '\n    A decorator that defines __unicode__ and __str__ methods under Python 2.\n    Under Python 3 it does nothing.\n\n    To support Python 2 and 3 with a single code base, define a __str__ method\n    returning text and apply this decorator to the class.\n    ')
    
    # Getting the type of 'PY2' (line 859)
    PY2_2228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 7), 'PY2')
    # Testing the type of an if condition (line 859)
    if_condition_2229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 859, 4), PY2_2228)
    # Assigning a type to the variable 'if_condition_2229' (line 859)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'if_condition_2229', if_condition_2229)
    # SSA begins for if statement (line 859)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_2230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 11), 'str', '__str__')
    # Getting the type of 'klass' (line 860)
    klass_2231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 28), 'klass')
    # Obtaining the member '__dict__' of a type (line 860)
    dict___2232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 28), klass_2231, '__dict__')
    # Applying the binary operator 'notin' (line 860)
    result_contains_2233 = python_operator(stypy.reporting.localization.Localization(__file__, 860, 11), 'notin', str_2230, dict___2232)
    
    # Testing the type of an if condition (line 860)
    if_condition_2234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 860, 8), result_contains_2233)
    # Assigning a type to the variable 'if_condition_2234' (line 860)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'if_condition_2234', if_condition_2234)
    # SSA begins for if statement (line 860)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 861)
    # Processing the call arguments (line 861)
    str_2236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 29), 'str', "@python_2_unicode_compatible cannot be applied to %s because it doesn't define __str__().")
    # Getting the type of 'klass' (line 863)
    klass_2237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 29), 'klass', False)
    # Obtaining the member '__name__' of a type (line 863)
    name___2238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 29), klass_2237, '__name__')
    # Applying the binary operator '%' (line 861)
    result_mod_2239 = python_operator(stypy.reporting.localization.Localization(__file__, 861, 29), '%', str_2236, name___2238)
    
    # Processing the call keyword arguments (line 861)
    kwargs_2240 = {}
    # Getting the type of 'ValueError' (line 861)
    ValueError_2235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 861)
    ValueError_call_result_2241 = invoke(stypy.reporting.localization.Localization(__file__, 861, 18), ValueError_2235, *[result_mod_2239], **kwargs_2240)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 861, 12), ValueError_call_result_2241, 'raise parameter', BaseException)
    # SSA join for if statement (line 860)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Attribute (line 864):
    # Getting the type of 'klass' (line 864)
    klass_2242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 28), 'klass')
    # Obtaining the member '__str__' of a type (line 864)
    str___2243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 28), klass_2242, '__str__')
    # Getting the type of 'klass' (line 864)
    klass_2244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 8), 'klass')
    # Setting the type of the member '__unicode__' of a type (line 864)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 8), klass_2244, '__unicode__', str___2243)
    
    # Assigning a Lambda to a Attribute (line 865):

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 865, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_1.stypy_localization = localization
        _stypy_temp_lambda_1.stypy_type_of_self = None
        _stypy_temp_lambda_1.stypy_type_store = module_type_store
        _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
        _stypy_temp_lambda_1.stypy_param_names_list = ['self']
        _stypy_temp_lambda_1.stypy_varargs_param_name = None
        _stypy_temp_lambda_1.stypy_kwargs_param_name = None
        _stypy_temp_lambda_1.stypy_call_defaults = defaults
        _stypy_temp_lambda_1.stypy_call_varargs = varargs
        _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_1', ['self'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to encode(...): (line 865)
        # Processing the call arguments (line 865)
        str_2250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 63), 'str', 'utf-8')
        # Processing the call keyword arguments (line 865)
        kwargs_2251 = {}
        
        # Call to __unicode__(...): (line 865)
        # Processing the call keyword arguments (line 865)
        kwargs_2247 = {}
        # Getting the type of 'self' (line 865)
        self_2245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 37), 'self', False)
        # Obtaining the member '__unicode__' of a type (line 865)
        unicode___2246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 37), self_2245, '__unicode__')
        # Calling __unicode__(args, kwargs) (line 865)
        unicode___call_result_2248 = invoke(stypy.reporting.localization.Localization(__file__, 865, 37), unicode___2246, *[], **kwargs_2247)
        
        # Obtaining the member 'encode' of a type (line 865)
        encode_2249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 37), unicode___call_result_2248, 'encode')
        # Calling encode(args, kwargs) (line 865)
        encode_call_result_2252 = invoke(stypy.reporting.localization.Localization(__file__, 865, 37), encode_2249, *[str_2250], **kwargs_2251)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 865)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 24), 'stypy_return_type', encode_call_result_2252)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 865)
        stypy_return_type_2253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2253)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_2253

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 865)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 24), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 865)
    _stypy_temp_lambda_1_2254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 24), '_stypy_temp_lambda_1')
    # Getting the type of 'klass' (line 865)
    klass_2255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 8), 'klass')
    # Setting the type of the member '__str__' of a type (line 865)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 8), klass_2255, '__str__', _stypy_temp_lambda_1_2254)
    # SSA join for if statement (line 859)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'klass' (line 866)
    klass_2256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 11), 'klass')
    # Assigning a type to the variable 'stypy_return_type' (line 866)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 4), 'stypy_return_type', klass_2256)
    
    # ################# End of 'python_2_unicode_compatible(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'python_2_unicode_compatible' in the type store
    # Getting the type of 'stypy_return_type' (line 851)
    stypy_return_type_2257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2257)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'python_2_unicode_compatible'
    return stypy_return_type_2257

# Assigning a type to the variable 'python_2_unicode_compatible' (line 851)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 0), 'python_2_unicode_compatible', python_2_unicode_compatible)

# Assigning a List to a Name (line 872):

# Obtaining an instance of the builtin type 'list' (line 872)
list_2258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 872)

# Assigning a type to the variable '__path__' (line 872)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 0), '__path__', list_2258)

# Assigning a Name to a Name (line 873):
# Getting the type of '__name__' (line 873)
name___2259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 14), '__name__')
# Assigning a type to the variable '__package__' (line 873)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 0), '__package__', name___2259)



# Call to get(...): (line 874)
# Processing the call arguments (line 874)
str_2264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 17), 'str', '__spec__')
# Processing the call keyword arguments (line 874)
kwargs_2265 = {}

# Call to globals(...): (line 874)
# Processing the call keyword arguments (line 874)
kwargs_2261 = {}
# Getting the type of 'globals' (line 874)
globals_2260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 3), 'globals', False)
# Calling globals(args, kwargs) (line 874)
globals_call_result_2262 = invoke(stypy.reporting.localization.Localization(__file__, 874, 3), globals_2260, *[], **kwargs_2261)

# Obtaining the member 'get' of a type (line 874)
get_2263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 3), globals_call_result_2262, 'get')
# Calling get(args, kwargs) (line 874)
get_call_result_2266 = invoke(stypy.reporting.localization.Localization(__file__, 874, 3), get_2263, *[str_2264], **kwargs_2265)

# Getting the type of 'None' (line 874)
None_2267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 36), 'None')
# Applying the binary operator 'isnot' (line 874)
result_is_not_2268 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 3), 'isnot', get_call_result_2266, None_2267)

# Testing the type of an if condition (line 874)
if_condition_2269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 874, 0), result_is_not_2268)
# Assigning a type to the variable 'if_condition_2269' (line 874)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 0), 'if_condition_2269', if_condition_2269)
# SSA begins for if statement (line 874)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a List to a Attribute (line 875):

# Obtaining an instance of the builtin type 'list' (line 875)
list_2270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 42), 'list')
# Adding type elements to the builtin type 'list' instance (line 875)

# Getting the type of '__spec__' (line 875)
spec___2271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 4), '__spec__')
# Setting the type of the member 'submodule_search_locations' of a type (line 875)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 4), spec___2271, 'submodule_search_locations', list_2270)
# SSA join for if statement (line 874)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'sys' (line 879)
sys_2272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 3), 'sys')
# Obtaining the member 'meta_path' of a type (line 879)
meta_path_2273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 3), sys_2272, 'meta_path')
# Testing the type of an if condition (line 879)
if_condition_2274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 879, 0), meta_path_2273)
# Assigning a type to the variable 'if_condition_2274' (line 879)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 0), 'if_condition_2274', if_condition_2274)
# SSA begins for if statement (line 879)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# Call to enumerate(...): (line 880)
# Processing the call arguments (line 880)
# Getting the type of 'sys' (line 880)
sys_2276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 33), 'sys', False)
# Obtaining the member 'meta_path' of a type (line 880)
meta_path_2277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 33), sys_2276, 'meta_path')
# Processing the call keyword arguments (line 880)
kwargs_2278 = {}
# Getting the type of 'enumerate' (line 880)
enumerate_2275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 23), 'enumerate', False)
# Calling enumerate(args, kwargs) (line 880)
enumerate_call_result_2279 = invoke(stypy.reporting.localization.Localization(__file__, 880, 23), enumerate_2275, *[meta_path_2277], **kwargs_2278)

# Testing the type of a for loop iterable (line 880)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 880, 4), enumerate_call_result_2279)
# Getting the type of the for loop variable (line 880)
for_loop_var_2280 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 880, 4), enumerate_call_result_2279)
# Assigning a type to the variable 'i' (line 880)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 880, 4), for_loop_var_2280))
# Assigning a type to the variable 'importer' (line 880)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 4), 'importer', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 880, 4), for_loop_var_2280))
# SSA begins for a for statement (line 880)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# Evaluating a boolean operation


# Call to type(...): (line 885)
# Processing the call arguments (line 885)
# Getting the type of 'importer' (line 885)
importer_2282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 17), 'importer', False)
# Processing the call keyword arguments (line 885)
kwargs_2283 = {}
# Getting the type of 'type' (line 885)
type_2281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 12), 'type', False)
# Calling type(args, kwargs) (line 885)
type_call_result_2284 = invoke(stypy.reporting.localization.Localization(__file__, 885, 12), type_2281, *[importer_2282], **kwargs_2283)

# Obtaining the member '__name__' of a type (line 885)
name___2285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 12), type_call_result_2284, '__name__')
str_2286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 39), 'str', '_SixMetaPathImporter')
# Applying the binary operator '==' (line 885)
result_eq_2287 = python_operator(stypy.reporting.localization.Localization(__file__, 885, 12), '==', name___2285, str_2286)


# Getting the type of 'importer' (line 886)
importer_2288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 16), 'importer')
# Obtaining the member 'name' of a type (line 886)
name_2289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 16), importer_2288, 'name')
# Getting the type of '__name__' (line 886)
name___2290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 33), '__name__')
# Applying the binary operator '==' (line 886)
result_eq_2291 = python_operator(stypy.reporting.localization.Localization(__file__, 886, 16), '==', name_2289, name___2290)

# Applying the binary operator 'and' (line 885)
result_and_keyword_2292 = python_operator(stypy.reporting.localization.Localization(__file__, 885, 12), 'and', result_eq_2287, result_eq_2291)

# Testing the type of an if condition (line 885)
if_condition_2293 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 885, 8), result_and_keyword_2292)
# Assigning a type to the variable 'if_condition_2293' (line 885)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 8), 'if_condition_2293', if_condition_2293)
# SSA begins for if statement (line 885)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
# Deleting a member
# Getting the type of 'sys' (line 887)
sys_2294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 16), 'sys')
# Obtaining the member 'meta_path' of a type (line 887)
meta_path_2295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 16), sys_2294, 'meta_path')

# Obtaining the type of the subscript
# Getting the type of 'i' (line 887)
i_2296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 30), 'i')
# Getting the type of 'sys' (line 887)
sys_2297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 16), 'sys')
# Obtaining the member 'meta_path' of a type (line 887)
meta_path_2298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 16), sys_2297, 'meta_path')
# Obtaining the member '__getitem__' of a type (line 887)
getitem___2299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 16), meta_path_2298, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 887)
subscript_call_result_2300 = invoke(stypy.reporting.localization.Localization(__file__, 887, 16), getitem___2299, i_2296)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 887, 12), meta_path_2295, subscript_call_result_2300)
# SSA join for if statement (line 885)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 889, 4), module_type_store, 'i')
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 889, 4), module_type_store, 'importer')
# SSA join for if statement (line 879)
module_type_store = module_type_store.join_ssa_context()


# Call to append(...): (line 891)
# Processing the call arguments (line 891)
# Getting the type of '_importer' (line 891)
_importer_2304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 21), '_importer', False)
# Processing the call keyword arguments (line 891)
kwargs_2305 = {}
# Getting the type of 'sys' (line 891)
sys_2301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 0), 'sys', False)
# Obtaining the member 'meta_path' of a type (line 891)
meta_path_2302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 0), sys_2301, 'meta_path')
# Obtaining the member 'append' of a type (line 891)
append_2303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 0), meta_path_2302, 'append')
# Calling append(args, kwargs) (line 891)
append_call_result_2306 = invoke(stypy.reporting.localization.Localization(__file__, 891, 0), append_2303, *[_importer_2304], **kwargs_2305)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
