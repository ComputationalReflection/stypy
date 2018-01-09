
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''A file interface for handling local and remote data files.
2: 
3: The goal of datasource is to abstract some of the file system operations
4: when dealing with data files so the researcher doesn't have to know all the
5: low-level details.  Through datasource, a researcher can obtain and use a
6: file with one function call, regardless of location of the file.
7: 
8: DataSource is meant to augment standard python libraries, not replace them.
9: It should work seemlessly with standard file IO operations and the os
10: module.
11: 
12: DataSource files can originate locally or remotely:
13: 
14: - local files : '/home/guido/src/local/data.txt'
15: - URLs (http, ftp, ...) : 'http://www.scipy.org/not/real/data.txt'
16: 
17: DataSource files can also be compressed or uncompressed.  Currently only
18: gzip and bz2 are supported.
19: 
20: Example::
21: 
22:     >>> # Create a DataSource, use os.curdir (default) for local storage.
23:     >>> ds = datasource.DataSource()
24:     >>>
25:     >>> # Open a remote file.
26:     >>> # DataSource downloads the file, stores it locally in:
27:     >>> #     './www.google.com/index.html'
28:     >>> # opens the file and returns a file object.
29:     >>> fp = ds.open('http://www.google.com/index.html')
30:     >>>
31:     >>> # Use the file as you normally would
32:     >>> fp.read()
33:     >>> fp.close()
34: 
35: '''
36: from __future__ import division, absolute_import, print_function
37: 
38: import os
39: import sys
40: import shutil
41: 
42: _open = open
43: 
44: 
45: # Using a class instead of a module-level dictionary
46: # to reduce the inital 'import numpy' overhead by
47: # deferring the import of bz2 and gzip until needed
48: 
49: # TODO: .zip support, .tar support?
50: class _FileOpeners(object):
51:     '''
52:     Container for different methods to open (un-)compressed files.
53: 
54:     `_FileOpeners` contains a dictionary that holds one method for each
55:     supported file format. Attribute lookup is implemented in such a way
56:     that an instance of `_FileOpeners` itself can be indexed with the keys
57:     of that dictionary. Currently uncompressed files as well as files
58:     compressed with ``gzip`` or ``bz2`` compression are supported.
59: 
60:     Notes
61:     -----
62:     `_file_openers`, an instance of `_FileOpeners`, is made available for
63:     use in the `_datasource` module.
64: 
65:     Examples
66:     --------
67:     >>> np.lib._datasource._file_openers.keys()
68:     [None, '.bz2', '.gz']
69:     >>> np.lib._datasource._file_openers['.gz'] is gzip.open
70:     True
71: 
72:     '''
73: 
74:     def __init__(self):
75:         self._loaded = False
76:         self._file_openers = {None: open}
77: 
78:     def _load(self):
79:         if self._loaded:
80:             return
81:         try:
82:             import bz2
83:             self._file_openers[".bz2"] = bz2.BZ2File
84:         except ImportError:
85:             pass
86:         try:
87:             import gzip
88:             self._file_openers[".gz"] = gzip.open
89:         except ImportError:
90:             pass
91:         self._loaded = True
92: 
93:     def keys(self):
94:         '''
95:         Return the keys of currently supported file openers.
96: 
97:         Parameters
98:         ----------
99:         None
100: 
101:         Returns
102:         -------
103:         keys : list
104:             The keys are None for uncompressed files and the file extension
105:             strings (i.e. ``'.gz'``, ``'.bz2'``) for supported compression
106:             methods.
107: 
108:         '''
109:         self._load()
110:         return list(self._file_openers.keys())
111: 
112:     def __getitem__(self, key):
113:         self._load()
114:         return self._file_openers[key]
115: 
116: _file_openers = _FileOpeners()
117: 
118: def open(path, mode='r', destpath=os.curdir):
119:     '''
120:     Open `path` with `mode` and return the file object.
121: 
122:     If ``path`` is an URL, it will be downloaded, stored in the
123:     `DataSource` `destpath` directory and opened from there.
124: 
125:     Parameters
126:     ----------
127:     path : str
128:         Local file path or URL to open.
129:     mode : str, optional
130:         Mode to open `path`. Mode 'r' for reading, 'w' for writing, 'a' to
131:         append. Available modes depend on the type of object specified by
132:         path.  Default is 'r'.
133:     destpath : str, optional
134:         Path to the directory where the source file gets downloaded to for
135:         use.  If `destpath` is None, a temporary directory will be created.
136:         The default path is the current directory.
137: 
138:     Returns
139:     -------
140:     out : file object
141:         The opened file.
142: 
143:     Notes
144:     -----
145:     This is a convenience function that instantiates a `DataSource` and
146:     returns the file object from ``DataSource.open(path)``.
147: 
148:     '''
149: 
150:     ds = DataSource(destpath)
151:     return ds.open(path, mode)
152: 
153: 
154: class DataSource (object):
155:     '''
156:     DataSource(destpath='.')
157: 
158:     A generic data source file (file, http, ftp, ...).
159: 
160:     DataSources can be local files or remote files/URLs.  The files may
161:     also be compressed or uncompressed. DataSource hides some of the
162:     low-level details of downloading the file, allowing you to simply pass
163:     in a valid file path (or URL) and obtain a file object.
164: 
165:     Parameters
166:     ----------
167:     destpath : str or None, optional
168:         Path to the directory where the source file gets downloaded to for
169:         use.  If `destpath` is None, a temporary directory will be created.
170:         The default path is the current directory.
171: 
172:     Notes
173:     -----
174:     URLs require a scheme string (``http://``) to be used, without it they
175:     will fail::
176: 
177:         >>> repos = DataSource()
178:         >>> repos.exists('www.google.com/index.html')
179:         False
180:         >>> repos.exists('http://www.google.com/index.html')
181:         True
182: 
183:     Temporary directories are deleted when the DataSource is deleted.
184: 
185:     Examples
186:     --------
187:     ::
188: 
189:         >>> ds = DataSource('/home/guido')
190:         >>> urlname = 'http://www.google.com/index.html'
191:         >>> gfile = ds.open('http://www.google.com/index.html')  # remote file
192:         >>> ds.abspath(urlname)
193:         '/home/guido/www.google.com/site/index.html'
194: 
195:         >>> ds = DataSource(None)  # use with temporary file
196:         >>> ds.open('/home/guido/foobar.txt')
197:         <open file '/home/guido.foobar.txt', mode 'r' at 0x91d4430>
198:         >>> ds.abspath('/home/guido/foobar.txt')
199:         '/tmp/tmpy4pgsP/home/guido/foobar.txt'
200: 
201:     '''
202: 
203:     def __init__(self, destpath=os.curdir):
204:         '''Create a DataSource with a local path at destpath.'''
205:         if destpath:
206:             self._destpath = os.path.abspath(destpath)
207:             self._istmpdest = False
208:         else:
209:             import tempfile  # deferring import to improve startup time
210:             self._destpath = tempfile.mkdtemp()
211:             self._istmpdest = True
212: 
213:     def __del__(self):
214:         # Remove temp directories
215:         if self._istmpdest:
216:             shutil.rmtree(self._destpath)
217: 
218:     def _iszip(self, filename):
219:         '''Test if the filename is a zip file by looking at the file extension.
220: 
221:         '''
222:         fname, ext = os.path.splitext(filename)
223:         return ext in _file_openers.keys()
224: 
225:     def _iswritemode(self, mode):
226:         '''Test if the given mode will open a file for writing.'''
227: 
228:         # Currently only used to test the bz2 files.
229:         _writemodes = ("w", "+")
230:         for c in mode:
231:             if c in _writemodes:
232:                 return True
233:         return False
234: 
235:     def _splitzipext(self, filename):
236:         '''Split zip extension from filename and return filename.
237: 
238:         *Returns*:
239:             base, zip_ext : {tuple}
240: 
241:         '''
242: 
243:         if self._iszip(filename):
244:             return os.path.splitext(filename)
245:         else:
246:             return filename, None
247: 
248:     def _possible_names(self, filename):
249:         '''Return a tuple containing compressed filename variations.'''
250:         names = [filename]
251:         if not self._iszip(filename):
252:             for zipext in _file_openers.keys():
253:                 if zipext:
254:                     names.append(filename+zipext)
255:         return names
256: 
257:     def _isurl(self, path):
258:         '''Test if path is a net location.  Tests the scheme and netloc.'''
259: 
260:         # We do this here to reduce the 'import numpy' initial import time.
261:         if sys.version_info[0] >= 3:
262:             from urllib.parse import urlparse
263:         else:
264:             from urlparse import urlparse
265: 
266:         # BUG : URLs require a scheme string ('http://') to be used.
267:         #       www.google.com will fail.
268:         #       Should we prepend the scheme for those that don't have it and
269:         #       test that also?  Similar to the way we append .gz and test for
270:         #       for compressed versions of files.
271: 
272:         scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
273:         return bool(scheme and netloc)
274: 
275:     def _cache(self, path):
276:         '''Cache the file specified by path.
277: 
278:         Creates a copy of the file in the datasource cache.
279: 
280:         '''
281:         # We import these here because importing urllib2 is slow and
282:         # a significant fraction of numpy's total import time.
283:         if sys.version_info[0] >= 3:
284:             from urllib.request import urlopen
285:             from urllib.error import URLError
286:         else:
287:             from urllib2 import urlopen
288:             from urllib2 import URLError
289: 
290:         upath = self.abspath(path)
291: 
292:         # ensure directory exists
293:         if not os.path.exists(os.path.dirname(upath)):
294:             os.makedirs(os.path.dirname(upath))
295: 
296:         # TODO: Doesn't handle compressed files!
297:         if self._isurl(path):
298:             try:
299:                 openedurl = urlopen(path)
300:                 f = _open(upath, 'wb')
301:                 try:
302:                     shutil.copyfileobj(openedurl, f)
303:                 finally:
304:                     f.close()
305:                     openedurl.close()
306:             except URLError:
307:                 raise URLError("URL not found: %s" % path)
308:         else:
309:             shutil.copyfile(path, upath)
310:         return upath
311: 
312:     def _findfile(self, path):
313:         '''Searches for ``path`` and returns full path if found.
314: 
315:         If path is an URL, _findfile will cache a local copy and return the
316:         path to the cached file.  If path is a local file, _findfile will
317:         return a path to that local file.
318: 
319:         The search will include possible compressed versions of the file
320:         and return the first occurence found.
321: 
322:         '''
323: 
324:         # Build list of possible local file paths
325:         if not self._isurl(path):
326:             # Valid local paths
327:             filelist = self._possible_names(path)
328:             # Paths in self._destpath
329:             filelist += self._possible_names(self.abspath(path))
330:         else:
331:             # Cached URLs in self._destpath
332:             filelist = self._possible_names(self.abspath(path))
333:             # Remote URLs
334:             filelist = filelist + self._possible_names(path)
335: 
336:         for name in filelist:
337:             if self.exists(name):
338:                 if self._isurl(name):
339:                     name = self._cache(name)
340:                 return name
341:         return None
342: 
343:     def abspath(self, path):
344:         '''
345:         Return absolute path of file in the DataSource directory.
346: 
347:         If `path` is an URL, then `abspath` will return either the location
348:         the file exists locally or the location it would exist when opened
349:         using the `open` method.
350: 
351:         Parameters
352:         ----------
353:         path : str
354:             Can be a local file or a remote URL.
355: 
356:         Returns
357:         -------
358:         out : str
359:             Complete path, including the `DataSource` destination directory.
360: 
361:         Notes
362:         -----
363:         The functionality is based on `os.path.abspath`.
364: 
365:         '''
366:         # We do this here to reduce the 'import numpy' initial import time.
367:         if sys.version_info[0] >= 3:
368:             from urllib.parse import urlparse
369:         else:
370:             from urlparse import urlparse
371: 
372:         # TODO:  This should be more robust.  Handles case where path includes
373:         #        the destpath, but not other sub-paths. Failing case:
374:         #        path = /home/guido/datafile.txt
375:         #        destpath = /home/alex/
376:         #        upath = self.abspath(path)
377:         #        upath == '/home/alex/home/guido/datafile.txt'
378: 
379:         # handle case where path includes self._destpath
380:         splitpath = path.split(self._destpath, 2)
381:         if len(splitpath) > 1:
382:             path = splitpath[1]
383:         scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
384:         netloc = self._sanitize_relative_path(netloc)
385:         upath = self._sanitize_relative_path(upath)
386:         return os.path.join(self._destpath, netloc, upath)
387: 
388:     def _sanitize_relative_path(self, path):
389:         '''Return a sanitised relative path for which
390:         os.path.abspath(os.path.join(base, path)).startswith(base)
391:         '''
392:         last = None
393:         path = os.path.normpath(path)
394:         while path != last:
395:             last = path
396:             # Note: os.path.join treats '/' as os.sep on Windows
397:             path = path.lstrip(os.sep).lstrip('/')
398:             path = path.lstrip(os.pardir).lstrip('..')
399:             drive, path = os.path.splitdrive(path)  # for Windows
400:         return path
401: 
402:     def exists(self, path):
403:         '''
404:         Test if path exists.
405: 
406:         Test if `path` exists as (and in this order):
407: 
408:         - a local file.
409:         - a remote URL that has been downloaded and stored locally in the
410:           `DataSource` directory.
411:         - a remote URL that has not been downloaded, but is valid and
412:           accessible.
413: 
414:         Parameters
415:         ----------
416:         path : str
417:             Can be a local file or a remote URL.
418: 
419:         Returns
420:         -------
421:         out : bool
422:             True if `path` exists.
423: 
424:         Notes
425:         -----
426:         When `path` is an URL, `exists` will return True if it's either
427:         stored locally in the `DataSource` directory, or is a valid remote
428:         URL.  `DataSource` does not discriminate between the two, the file
429:         is accessible if it exists in either location.
430: 
431:         '''
432:         # We import this here because importing urllib2 is slow and
433:         # a significant fraction of numpy's total import time.
434:         if sys.version_info[0] >= 3:
435:             from urllib.request import urlopen
436:             from urllib.error import URLError
437:         else:
438:             from urllib2 import urlopen
439:             from urllib2 import URLError
440: 
441:         # Test local path
442:         if os.path.exists(path):
443:             return True
444: 
445:         # Test cached url
446:         upath = self.abspath(path)
447:         if os.path.exists(upath):
448:             return True
449: 
450:         # Test remote url
451:         if self._isurl(path):
452:             try:
453:                 netfile = urlopen(path)
454:                 netfile.close()
455:                 del(netfile)
456:                 return True
457:             except URLError:
458:                 return False
459:         return False
460: 
461:     def open(self, path, mode='r'):
462:         '''
463:         Open and return file-like object.
464: 
465:         If `path` is an URL, it will be downloaded, stored in the
466:         `DataSource` directory and opened from there.
467: 
468:         Parameters
469:         ----------
470:         path : str
471:             Local file path or URL to open.
472:         mode : {'r', 'w', 'a'}, optional
473:             Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
474:             'a' to append. Available modes depend on the type of object
475:             specified by `path`. Default is 'r'.
476: 
477:         Returns
478:         -------
479:         out : file object
480:             File object.
481: 
482:         '''
483: 
484:         # TODO: There is no support for opening a file for writing which
485:         #       doesn't exist yet (creating a file).  Should there be?
486: 
487:         # TODO: Add a ``subdir`` parameter for specifying the subdirectory
488:         #       used to store URLs in self._destpath.
489: 
490:         if self._isurl(path) and self._iswritemode(mode):
491:             raise ValueError("URLs are not writeable")
492: 
493:         # NOTE: _findfile will fail on a new file opened for writing.
494:         found = self._findfile(path)
495:         if found:
496:             _fname, ext = self._splitzipext(found)
497:             if ext == 'bz2':
498:                 mode.replace("+", "")
499:             return _file_openers[ext](found, mode=mode)
500:         else:
501:             raise IOError("%s not found." % path)
502: 
503: 
504: class Repository (DataSource):
505:     '''
506:     Repository(baseurl, destpath='.')
507: 
508:     A data repository where multiple DataSource's share a base
509:     URL/directory.
510: 
511:     `Repository` extends `DataSource` by prepending a base URL (or
512:     directory) to all the files it handles. Use `Repository` when you will
513:     be working with multiple files from one base URL.  Initialize
514:     `Repository` with the base URL, then refer to each file by its filename
515:     only.
516: 
517:     Parameters
518:     ----------
519:     baseurl : str
520:         Path to the local directory or remote location that contains the
521:         data files.
522:     destpath : str or None, optional
523:         Path to the directory where the source file gets downloaded to for
524:         use.  If `destpath` is None, a temporary directory will be created.
525:         The default path is the current directory.
526: 
527:     Examples
528:     --------
529:     To analyze all files in the repository, do something like this
530:     (note: this is not self-contained code)::
531: 
532:         >>> repos = np.lib._datasource.Repository('/home/user/data/dir/')
533:         >>> for filename in filelist:
534:         ...     fp = repos.open(filename)
535:         ...     fp.analyze()
536:         ...     fp.close()
537: 
538:     Similarly you could use a URL for a repository::
539: 
540:         >>> repos = np.lib._datasource.Repository('http://www.xyz.edu/data')
541: 
542:     '''
543: 
544:     def __init__(self, baseurl, destpath=os.curdir):
545:         '''Create a Repository with a shared url or directory of baseurl.'''
546:         DataSource.__init__(self, destpath=destpath)
547:         self._baseurl = baseurl
548: 
549:     def __del__(self):
550:         DataSource.__del__(self)
551: 
552:     def _fullpath(self, path):
553:         '''Return complete path for path.  Prepends baseurl if necessary.'''
554:         splitpath = path.split(self._baseurl, 2)
555:         if len(splitpath) == 1:
556:             result = os.path.join(self._baseurl, path)
557:         else:
558:             result = path    # path contains baseurl already
559:         return result
560: 
561:     def _findfile(self, path):
562:         '''Extend DataSource method to prepend baseurl to ``path``.'''
563:         return DataSource._findfile(self, self._fullpath(path))
564: 
565:     def abspath(self, path):
566:         '''
567:         Return absolute path of file in the Repository directory.
568: 
569:         If `path` is an URL, then `abspath` will return either the location
570:         the file exists locally or the location it would exist when opened
571:         using the `open` method.
572: 
573:         Parameters
574:         ----------
575:         path : str
576:             Can be a local file or a remote URL. This may, but does not
577:             have to, include the `baseurl` with which the `Repository` was
578:             initialized.
579: 
580:         Returns
581:         -------
582:         out : str
583:             Complete path, including the `DataSource` destination directory.
584: 
585:         '''
586:         return DataSource.abspath(self, self._fullpath(path))
587: 
588:     def exists(self, path):
589:         '''
590:         Test if path exists prepending Repository base URL to path.
591: 
592:         Test if `path` exists as (and in this order):
593: 
594:         - a local file.
595:         - a remote URL that has been downloaded and stored locally in the
596:           `DataSource` directory.
597:         - a remote URL that has not been downloaded, but is valid and
598:           accessible.
599: 
600:         Parameters
601:         ----------
602:         path : str
603:             Can be a local file or a remote URL. This may, but does not
604:             have to, include the `baseurl` with which the `Repository` was
605:             initialized.
606: 
607:         Returns
608:         -------
609:         out : bool
610:             True if `path` exists.
611: 
612:         Notes
613:         -----
614:         When `path` is an URL, `exists` will return True if it's either
615:         stored locally in the `DataSource` directory, or is a valid remote
616:         URL.  `DataSource` does not discriminate between the two, the file
617:         is accessible if it exists in either location.
618: 
619:         '''
620:         return DataSource.exists(self, self._fullpath(path))
621: 
622:     def open(self, path, mode='r'):
623:         '''
624:         Open and return file-like object prepending Repository base URL.
625: 
626:         If `path` is an URL, it will be downloaded, stored in the
627:         DataSource directory and opened from there.
628: 
629:         Parameters
630:         ----------
631:         path : str
632:             Local file path or URL to open. This may, but does not have to,
633:             include the `baseurl` with which the `Repository` was
634:             initialized.
635:         mode : {'r', 'w', 'a'}, optional
636:             Mode to open `path`.  Mode 'r' for reading, 'w' for writing,
637:             'a' to append. Available modes depend on the type of object
638:             specified by `path`. Default is 'r'.
639: 
640:         Returns
641:         -------
642:         out : file object
643:             File object.
644: 
645:         '''
646:         return DataSource.open(self, self._fullpath(path), mode)
647: 
648:     def listdir(self):
649:         '''
650:         List files in the source Repository.
651: 
652:         Returns
653:         -------
654:         files : list of str
655:             List of file names (not containing a directory part).
656: 
657:         Notes
658:         -----
659:         Does not currently work for remote repositories.
660: 
661:         '''
662:         if self._isurl(self._baseurl):
663:             raise NotImplementedError(
664:                   "Directory listing of URLs, not supported yet.")
665:         else:
666:             return os.listdir(self._baseurl)
667: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_131369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', "A file interface for handling local and remote data files.\n\nThe goal of datasource is to abstract some of the file system operations\nwhen dealing with data files so the researcher doesn't have to know all the\nlow-level details.  Through datasource, a researcher can obtain and use a\nfile with one function call, regardless of location of the file.\n\nDataSource is meant to augment standard python libraries, not replace them.\nIt should work seemlessly with standard file IO operations and the os\nmodule.\n\nDataSource files can originate locally or remotely:\n\n- local files : '/home/guido/src/local/data.txt'\n- URLs (http, ftp, ...) : 'http://www.scipy.org/not/real/data.txt'\n\nDataSource files can also be compressed or uncompressed.  Currently only\ngzip and bz2 are supported.\n\nExample::\n\n    >>> # Create a DataSource, use os.curdir (default) for local storage.\n    >>> ds = datasource.DataSource()\n    >>>\n    >>> # Open a remote file.\n    >>> # DataSource downloads the file, stores it locally in:\n    >>> #     './www.google.com/index.html'\n    >>> # opens the file and returns a file object.\n    >>> fp = ds.open('http://www.google.com/index.html')\n    >>>\n    >>> # Use the file as you normally would\n    >>> fp.read()\n    >>> fp.close()\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'import os' statement (line 38)
import os

import_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'import sys' statement (line 39)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'import shutil' statement (line 40)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'shutil', shutil, module_type_store)


# Assigning a Name to a Name (line 42):

# Assigning a Name to a Name (line 42):
# Getting the type of 'open' (line 42)
open_131370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'open')
# Assigning a type to the variable '_open' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), '_open', open_131370)
# Declaration of the '_FileOpeners' class

class _FileOpeners(object, ):
    str_131371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', "\n    Container for different methods to open (un-)compressed files.\n\n    `_FileOpeners` contains a dictionary that holds one method for each\n    supported file format. Attribute lookup is implemented in such a way\n    that an instance of `_FileOpeners` itself can be indexed with the keys\n    of that dictionary. Currently uncompressed files as well as files\n    compressed with ``gzip`` or ``bz2`` compression are supported.\n\n    Notes\n    -----\n    `_file_openers`, an instance of `_FileOpeners`, is made available for\n    use in the `_datasource` module.\n\n    Examples\n    --------\n    >>> np.lib._datasource._file_openers.keys()\n    [None, '.bz2', '.gz']\n    >>> np.lib._datasource._file_openers['.gz'] is gzip.open\n    True\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_FileOpeners.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 75):
        
        # Assigning a Name to a Attribute (line 75):
        # Getting the type of 'False' (line 75)
        False_131372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'False')
        # Getting the type of 'self' (line 75)
        self_131373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member '_loaded' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_131373, '_loaded', False_131372)
        
        # Assigning a Dict to a Attribute (line 76):
        
        # Assigning a Dict to a Attribute (line 76):
        
        # Obtaining an instance of the builtin type 'dict' (line 76)
        dict_131374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 29), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 76)
        # Adding element type (key, value) (line 76)
        # Getting the type of 'None' (line 76)
        None_131375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'None')
        # Getting the type of 'open' (line 76)
        open_131376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'open')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), dict_131374, (None_131375, open_131376))
        
        # Getting the type of 'self' (line 76)
        self_131377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member '_file_openers' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_131377, '_file_openers', dict_131374)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _load(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_load'
        module_type_store = module_type_store.open_function_context('_load', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _FileOpeners._load.__dict__.__setitem__('stypy_localization', localization)
        _FileOpeners._load.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _FileOpeners._load.__dict__.__setitem__('stypy_type_store', module_type_store)
        _FileOpeners._load.__dict__.__setitem__('stypy_function_name', '_FileOpeners._load')
        _FileOpeners._load.__dict__.__setitem__('stypy_param_names_list', [])
        _FileOpeners._load.__dict__.__setitem__('stypy_varargs_param_name', None)
        _FileOpeners._load.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _FileOpeners._load.__dict__.__setitem__('stypy_call_defaults', defaults)
        _FileOpeners._load.__dict__.__setitem__('stypy_call_varargs', varargs)
        _FileOpeners._load.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _FileOpeners._load.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_FileOpeners._load', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_load', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_load(...)' code ##################

        
        # Getting the type of 'self' (line 79)
        self_131378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'self')
        # Obtaining the member '_loaded' of a type (line 79)
        _loaded_131379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), self_131378, '_loaded')
        # Testing the type of an if condition (line 79)
        if_condition_131380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), _loaded_131379)
        # Assigning a type to the variable 'if_condition_131380' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_131380', if_condition_131380)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 82, 12))
        
        # 'import bz2' statement (line 82)
        import bz2

        import_module(stypy.reporting.localization.Localization(__file__, 82, 12), 'bz2', bz2, module_type_store)
        
        
        # Assigning a Attribute to a Subscript (line 83):
        
        # Assigning a Attribute to a Subscript (line 83):
        # Getting the type of 'bz2' (line 83)
        bz2_131381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'bz2')
        # Obtaining the member 'BZ2File' of a type (line 83)
        BZ2File_131382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 41), bz2_131381, 'BZ2File')
        # Getting the type of 'self' (line 83)
        self_131383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'self')
        # Obtaining the member '_file_openers' of a type (line 83)
        _file_openers_131384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), self_131383, '_file_openers')
        str_131385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 31), 'str', '.bz2')
        # Storing an element on a container (line 83)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 12), _file_openers_131384, (str_131385, BZ2File_131382))
        # SSA branch for the except part of a try statement (line 81)
        # SSA branch for the except 'ImportError' branch of a try statement (line 81)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 87, 12))
        
        # 'import gzip' statement (line 87)
        import gzip

        import_module(stypy.reporting.localization.Localization(__file__, 87, 12), 'gzip', gzip, module_type_store)
        
        
        # Assigning a Attribute to a Subscript (line 88):
        
        # Assigning a Attribute to a Subscript (line 88):
        # Getting the type of 'gzip' (line 88)
        gzip_131386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 40), 'gzip')
        # Obtaining the member 'open' of a type (line 88)
        open_131387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 40), gzip_131386, 'open')
        # Getting the type of 'self' (line 88)
        self_131388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'self')
        # Obtaining the member '_file_openers' of a type (line 88)
        _file_openers_131389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), self_131388, '_file_openers')
        str_131390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 31), 'str', '.gz')
        # Storing an element on a container (line 88)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 12), _file_openers_131389, (str_131390, open_131387))
        # SSA branch for the except part of a try statement (line 86)
        # SSA branch for the except 'ImportError' branch of a try statement (line 86)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 91):
        
        # Assigning a Name to a Attribute (line 91):
        # Getting the type of 'True' (line 91)
        True_131391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'True')
        # Getting the type of 'self' (line 91)
        self_131392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member '_loaded' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_131392, '_loaded', True_131391)
        
        # ################# End of '_load(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_load' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_131393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131393)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_load'
        return stypy_return_type_131393


    @norecursion
    def keys(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'keys'
        module_type_store = module_type_store.open_function_context('keys', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _FileOpeners.keys.__dict__.__setitem__('stypy_localization', localization)
        _FileOpeners.keys.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _FileOpeners.keys.__dict__.__setitem__('stypy_type_store', module_type_store)
        _FileOpeners.keys.__dict__.__setitem__('stypy_function_name', '_FileOpeners.keys')
        _FileOpeners.keys.__dict__.__setitem__('stypy_param_names_list', [])
        _FileOpeners.keys.__dict__.__setitem__('stypy_varargs_param_name', None)
        _FileOpeners.keys.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _FileOpeners.keys.__dict__.__setitem__('stypy_call_defaults', defaults)
        _FileOpeners.keys.__dict__.__setitem__('stypy_call_varargs', varargs)
        _FileOpeners.keys.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _FileOpeners.keys.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_FileOpeners.keys', [], None, None, defaults, varargs, kwargs)

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

        str_131394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', "\n        Return the keys of currently supported file openers.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        keys : list\n            The keys are None for uncompressed files and the file extension\n            strings (i.e. ``'.gz'``, ``'.bz2'``) for supported compression\n            methods.\n\n        ")
        
        # Call to _load(...): (line 109)
        # Processing the call keyword arguments (line 109)
        kwargs_131397 = {}
        # Getting the type of 'self' (line 109)
        self_131395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self', False)
        # Obtaining the member '_load' of a type (line 109)
        _load_131396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_131395, '_load')
        # Calling _load(args, kwargs) (line 109)
        _load_call_result_131398 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), _load_131396, *[], **kwargs_131397)
        
        
        # Call to list(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to keys(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_131403 = {}
        # Getting the type of 'self' (line 110)
        self_131400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'self', False)
        # Obtaining the member '_file_openers' of a type (line 110)
        _file_openers_131401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 20), self_131400, '_file_openers')
        # Obtaining the member 'keys' of a type (line 110)
        keys_131402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 20), _file_openers_131401, 'keys')
        # Calling keys(args, kwargs) (line 110)
        keys_call_result_131404 = invoke(stypy.reporting.localization.Localization(__file__, 110, 20), keys_131402, *[], **kwargs_131403)
        
        # Processing the call keyword arguments (line 110)
        kwargs_131405 = {}
        # Getting the type of 'list' (line 110)
        list_131399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'list', False)
        # Calling list(args, kwargs) (line 110)
        list_call_result_131406 = invoke(stypy.reporting.localization.Localization(__file__, 110, 15), list_131399, *[keys_call_result_131404], **kwargs_131405)
        
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'stypy_return_type', list_call_result_131406)
        
        # ################# End of 'keys(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'keys' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_131407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131407)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'keys'
        return stypy_return_type_131407


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_function_name', '_FileOpeners.__getitem__')
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _FileOpeners.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_FileOpeners.__getitem__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Call to _load(...): (line 113)
        # Processing the call keyword arguments (line 113)
        kwargs_131410 = {}
        # Getting the type of 'self' (line 113)
        self_131408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self', False)
        # Obtaining the member '_load' of a type (line 113)
        _load_131409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_131408, '_load')
        # Calling _load(args, kwargs) (line 113)
        _load_call_result_131411 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), _load_131409, *[], **kwargs_131410)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 114)
        key_131412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), 'key')
        # Getting the type of 'self' (line 114)
        self_131413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'self')
        # Obtaining the member '_file_openers' of a type (line 114)
        _file_openers_131414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 15), self_131413, '_file_openers')
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___131415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 15), _file_openers_131414, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_131416 = invoke(stypy.reporting.localization.Localization(__file__, 114, 15), getitem___131415, key_131412)
        
        # Assigning a type to the variable 'stypy_return_type' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'stypy_return_type', subscript_call_result_131416)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_131417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131417)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_131417


# Assigning a type to the variable '_FileOpeners' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '_FileOpeners', _FileOpeners)

# Assigning a Call to a Name (line 116):

# Assigning a Call to a Name (line 116):

# Call to _FileOpeners(...): (line 116)
# Processing the call keyword arguments (line 116)
kwargs_131419 = {}
# Getting the type of '_FileOpeners' (line 116)
_FileOpeners_131418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), '_FileOpeners', False)
# Calling _FileOpeners(args, kwargs) (line 116)
_FileOpeners_call_result_131420 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), _FileOpeners_131418, *[], **kwargs_131419)

# Assigning a type to the variable '_file_openers' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), '_file_openers', _FileOpeners_call_result_131420)

@norecursion
def open(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_131421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 20), 'str', 'r')
    # Getting the type of 'os' (line 118)
    os_131422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 34), 'os')
    # Obtaining the member 'curdir' of a type (line 118)
    curdir_131423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 34), os_131422, 'curdir')
    defaults = [str_131421, curdir_131423]
    # Create a new context for function 'open'
    module_type_store = module_type_store.open_function_context('open', 118, 0, False)
    
    # Passed parameters checking function
    open.stypy_localization = localization
    open.stypy_type_of_self = None
    open.stypy_type_store = module_type_store
    open.stypy_function_name = 'open'
    open.stypy_param_names_list = ['path', 'mode', 'destpath']
    open.stypy_varargs_param_name = None
    open.stypy_kwargs_param_name = None
    open.stypy_call_defaults = defaults
    open.stypy_call_varargs = varargs
    open.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'open', ['path', 'mode', 'destpath'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'open', localization, ['path', 'mode', 'destpath'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'open(...)' code ##################

    str_131424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, (-1)), 'str', "\n    Open `path` with `mode` and return the file object.\n\n    If ``path`` is an URL, it will be downloaded, stored in the\n    `DataSource` `destpath` directory and opened from there.\n\n    Parameters\n    ----------\n    path : str\n        Local file path or URL to open.\n    mode : str, optional\n        Mode to open `path`. Mode 'r' for reading, 'w' for writing, 'a' to\n        append. Available modes depend on the type of object specified by\n        path.  Default is 'r'.\n    destpath : str, optional\n        Path to the directory where the source file gets downloaded to for\n        use.  If `destpath` is None, a temporary directory will be created.\n        The default path is the current directory.\n\n    Returns\n    -------\n    out : file object\n        The opened file.\n\n    Notes\n    -----\n    This is a convenience function that instantiates a `DataSource` and\n    returns the file object from ``DataSource.open(path)``.\n\n    ")
    
    # Assigning a Call to a Name (line 150):
    
    # Assigning a Call to a Name (line 150):
    
    # Call to DataSource(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'destpath' (line 150)
    destpath_131426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'destpath', False)
    # Processing the call keyword arguments (line 150)
    kwargs_131427 = {}
    # Getting the type of 'DataSource' (line 150)
    DataSource_131425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 9), 'DataSource', False)
    # Calling DataSource(args, kwargs) (line 150)
    DataSource_call_result_131428 = invoke(stypy.reporting.localization.Localization(__file__, 150, 9), DataSource_131425, *[destpath_131426], **kwargs_131427)
    
    # Assigning a type to the variable 'ds' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'ds', DataSource_call_result_131428)
    
    # Call to open(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'path' (line 151)
    path_131431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'path', False)
    # Getting the type of 'mode' (line 151)
    mode_131432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'mode', False)
    # Processing the call keyword arguments (line 151)
    kwargs_131433 = {}
    # Getting the type of 'ds' (line 151)
    ds_131429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'ds', False)
    # Obtaining the member 'open' of a type (line 151)
    open_131430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), ds_131429, 'open')
    # Calling open(args, kwargs) (line 151)
    open_call_result_131434 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), open_131430, *[path_131431, mode_131432], **kwargs_131433)
    
    # Assigning a type to the variable 'stypy_return_type' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type', open_call_result_131434)
    
    # ################# End of 'open(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'open' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_131435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131435)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'open'
    return stypy_return_type_131435

# Assigning a type to the variable 'open' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'open', open)
# Declaration of the 'DataSource' class

class DataSource(object, ):
    str_131436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'str', "\n    DataSource(destpath='.')\n\n    A generic data source file (file, http, ftp, ...).\n\n    DataSources can be local files or remote files/URLs.  The files may\n    also be compressed or uncompressed. DataSource hides some of the\n    low-level details of downloading the file, allowing you to simply pass\n    in a valid file path (or URL) and obtain a file object.\n\n    Parameters\n    ----------\n    destpath : str or None, optional\n        Path to the directory where the source file gets downloaded to for\n        use.  If `destpath` is None, a temporary directory will be created.\n        The default path is the current directory.\n\n    Notes\n    -----\n    URLs require a scheme string (``http://``) to be used, without it they\n    will fail::\n\n        >>> repos = DataSource()\n        >>> repos.exists('www.google.com/index.html')\n        False\n        >>> repos.exists('http://www.google.com/index.html')\n        True\n\n    Temporary directories are deleted when the DataSource is deleted.\n\n    Examples\n    --------\n    ::\n\n        >>> ds = DataSource('/home/guido')\n        >>> urlname = 'http://www.google.com/index.html'\n        >>> gfile = ds.open('http://www.google.com/index.html')  # remote file\n        >>> ds.abspath(urlname)\n        '/home/guido/www.google.com/site/index.html'\n\n        >>> ds = DataSource(None)  # use with temporary file\n        >>> ds.open('/home/guido/foobar.txt')\n        <open file '/home/guido.foobar.txt', mode 'r' at 0x91d4430>\n        >>> ds.abspath('/home/guido/foobar.txt')\n        '/tmp/tmpy4pgsP/home/guido/foobar.txt'\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'os' (line 203)
        os_131437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 32), 'os')
        # Obtaining the member 'curdir' of a type (line 203)
        curdir_131438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 32), os_131437, 'curdir')
        defaults = [curdir_131438]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource.__init__', ['destpath'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['destpath'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_131439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 8), 'str', 'Create a DataSource with a local path at destpath.')
        
        # Getting the type of 'destpath' (line 205)
        destpath_131440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'destpath')
        # Testing the type of an if condition (line 205)
        if_condition_131441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), destpath_131440)
        # Assigning a type to the variable 'if_condition_131441' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_131441', if_condition_131441)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 206):
        
        # Assigning a Call to a Attribute (line 206):
        
        # Call to abspath(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'destpath' (line 206)
        destpath_131445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 45), 'destpath', False)
        # Processing the call keyword arguments (line 206)
        kwargs_131446 = {}
        # Getting the type of 'os' (line 206)
        os_131442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 206)
        path_131443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 29), os_131442, 'path')
        # Obtaining the member 'abspath' of a type (line 206)
        abspath_131444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 29), path_131443, 'abspath')
        # Calling abspath(args, kwargs) (line 206)
        abspath_call_result_131447 = invoke(stypy.reporting.localization.Localization(__file__, 206, 29), abspath_131444, *[destpath_131445], **kwargs_131446)
        
        # Getting the type of 'self' (line 206)
        self_131448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'self')
        # Setting the type of the member '_destpath' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), self_131448, '_destpath', abspath_call_result_131447)
        
        # Assigning a Name to a Attribute (line 207):
        
        # Assigning a Name to a Attribute (line 207):
        # Getting the type of 'False' (line 207)
        False_131449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 30), 'False')
        # Getting the type of 'self' (line 207)
        self_131450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'self')
        # Setting the type of the member '_istmpdest' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), self_131450, '_istmpdest', False_131449)
        # SSA branch for the else part of an if statement (line 205)
        module_type_store.open_ssa_branch('else')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 209, 12))
        
        # 'import tempfile' statement (line 209)
        import tempfile

        import_module(stypy.reporting.localization.Localization(__file__, 209, 12), 'tempfile', tempfile, module_type_store)
        
        
        # Assigning a Call to a Attribute (line 210):
        
        # Assigning a Call to a Attribute (line 210):
        
        # Call to mkdtemp(...): (line 210)
        # Processing the call keyword arguments (line 210)
        kwargs_131453 = {}
        # Getting the type of 'tempfile' (line 210)
        tempfile_131451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 29), 'tempfile', False)
        # Obtaining the member 'mkdtemp' of a type (line 210)
        mkdtemp_131452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 29), tempfile_131451, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 210)
        mkdtemp_call_result_131454 = invoke(stypy.reporting.localization.Localization(__file__, 210, 29), mkdtemp_131452, *[], **kwargs_131453)
        
        # Getting the type of 'self' (line 210)
        self_131455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'self')
        # Setting the type of the member '_destpath' of a type (line 210)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), self_131455, '_destpath', mkdtemp_call_result_131454)
        
        # Assigning a Name to a Attribute (line 211):
        
        # Assigning a Name to a Attribute (line 211):
        # Getting the type of 'True' (line 211)
        True_131456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'True')
        # Getting the type of 'self' (line 211)
        self_131457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'self')
        # Setting the type of the member '_istmpdest' of a type (line 211)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), self_131457, '_istmpdest', True_131456)
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __del__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__del__'
        module_type_store = module_type_store.open_function_context('__del__', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource.__del__.__dict__.__setitem__('stypy_localization', localization)
        DataSource.__del__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource.__del__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource.__del__.__dict__.__setitem__('stypy_function_name', 'DataSource.__del__')
        DataSource.__del__.__dict__.__setitem__('stypy_param_names_list', [])
        DataSource.__del__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource.__del__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource.__del__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource.__del__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource.__del__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource.__del__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource.__del__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__del__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__del__(...)' code ##################

        
        # Getting the type of 'self' (line 215)
        self_131458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'self')
        # Obtaining the member '_istmpdest' of a type (line 215)
        _istmpdest_131459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 11), self_131458, '_istmpdest')
        # Testing the type of an if condition (line 215)
        if_condition_131460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), _istmpdest_131459)
        # Assigning a type to the variable 'if_condition_131460' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_131460', if_condition_131460)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to rmtree(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'self' (line 216)
        self_131463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 26), 'self', False)
        # Obtaining the member '_destpath' of a type (line 216)
        _destpath_131464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 26), self_131463, '_destpath')
        # Processing the call keyword arguments (line 216)
        kwargs_131465 = {}
        # Getting the type of 'shutil' (line 216)
        shutil_131461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'shutil', False)
        # Obtaining the member 'rmtree' of a type (line 216)
        rmtree_131462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), shutil_131461, 'rmtree')
        # Calling rmtree(args, kwargs) (line 216)
        rmtree_call_result_131466 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), rmtree_131462, *[_destpath_131464], **kwargs_131465)
        
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__del__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__del__' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_131467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__del__'
        return stypy_return_type_131467


    @norecursion
    def _iszip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_iszip'
        module_type_store = module_type_store.open_function_context('_iszip', 218, 4, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource._iszip.__dict__.__setitem__('stypy_localization', localization)
        DataSource._iszip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource._iszip.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource._iszip.__dict__.__setitem__('stypy_function_name', 'DataSource._iszip')
        DataSource._iszip.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        DataSource._iszip.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource._iszip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource._iszip.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource._iszip.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource._iszip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource._iszip.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource._iszip', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_iszip', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_iszip(...)' code ##################

        str_131468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, (-1)), 'str', 'Test if the filename is a zip file by looking at the file extension.\n\n        ')
        
        # Assigning a Call to a Tuple (line 222):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'filename' (line 222)
        filename_131472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 38), 'filename', False)
        # Processing the call keyword arguments (line 222)
        kwargs_131473 = {}
        # Getting the type of 'os' (line 222)
        os_131469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 222)
        path_131470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 21), os_131469, 'path')
        # Obtaining the member 'splitext' of a type (line 222)
        splitext_131471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 21), path_131470, 'splitext')
        # Calling splitext(args, kwargs) (line 222)
        splitext_call_result_131474 = invoke(stypy.reporting.localization.Localization(__file__, 222, 21), splitext_131471, *[filename_131472], **kwargs_131473)
        
        # Assigning a type to the variable 'call_assignment_131346' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'call_assignment_131346', splitext_call_result_131474)
        
        # Assigning a Call to a Name (line 222):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131478 = {}
        # Getting the type of 'call_assignment_131346' (line 222)
        call_assignment_131346_131475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'call_assignment_131346', False)
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___131476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), call_assignment_131346_131475, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131479 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131476, *[int_131477], **kwargs_131478)
        
        # Assigning a type to the variable 'call_assignment_131347' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'call_assignment_131347', getitem___call_result_131479)
        
        # Assigning a Name to a Name (line 222):
        # Getting the type of 'call_assignment_131347' (line 222)
        call_assignment_131347_131480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'call_assignment_131347')
        # Assigning a type to the variable 'fname' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'fname', call_assignment_131347_131480)
        
        # Assigning a Call to a Name (line 222):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131484 = {}
        # Getting the type of 'call_assignment_131346' (line 222)
        call_assignment_131346_131481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'call_assignment_131346', False)
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___131482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), call_assignment_131346_131481, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131485 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131482, *[int_131483], **kwargs_131484)
        
        # Assigning a type to the variable 'call_assignment_131348' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'call_assignment_131348', getitem___call_result_131485)
        
        # Assigning a Name to a Name (line 222):
        # Getting the type of 'call_assignment_131348' (line 222)
        call_assignment_131348_131486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'call_assignment_131348')
        # Assigning a type to the variable 'ext' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'ext', call_assignment_131348_131486)
        
        # Getting the type of 'ext' (line 223)
        ext_131487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'ext')
        
        # Call to keys(...): (line 223)
        # Processing the call keyword arguments (line 223)
        kwargs_131490 = {}
        # Getting the type of '_file_openers' (line 223)
        _file_openers_131488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 22), '_file_openers', False)
        # Obtaining the member 'keys' of a type (line 223)
        keys_131489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 22), _file_openers_131488, 'keys')
        # Calling keys(args, kwargs) (line 223)
        keys_call_result_131491 = invoke(stypy.reporting.localization.Localization(__file__, 223, 22), keys_131489, *[], **kwargs_131490)
        
        # Applying the binary operator 'in' (line 223)
        result_contains_131492 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 15), 'in', ext_131487, keys_call_result_131491)
        
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', result_contains_131492)
        
        # ################# End of '_iszip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_iszip' in the type store
        # Getting the type of 'stypy_return_type' (line 218)
        stypy_return_type_131493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131493)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_iszip'
        return stypy_return_type_131493


    @norecursion
    def _iswritemode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_iswritemode'
        module_type_store = module_type_store.open_function_context('_iswritemode', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource._iswritemode.__dict__.__setitem__('stypy_localization', localization)
        DataSource._iswritemode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource._iswritemode.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource._iswritemode.__dict__.__setitem__('stypy_function_name', 'DataSource._iswritemode')
        DataSource._iswritemode.__dict__.__setitem__('stypy_param_names_list', ['mode'])
        DataSource._iswritemode.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource._iswritemode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource._iswritemode.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource._iswritemode.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource._iswritemode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource._iswritemode.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource._iswritemode', ['mode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_iswritemode', localization, ['mode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_iswritemode(...)' code ##################

        str_131494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'str', 'Test if the given mode will open a file for writing.')
        
        # Assigning a Tuple to a Name (line 229):
        
        # Assigning a Tuple to a Name (line 229):
        
        # Obtaining an instance of the builtin type 'tuple' (line 229)
        tuple_131495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 229)
        # Adding element type (line 229)
        str_131496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 23), 'str', 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 23), tuple_131495, str_131496)
        # Adding element type (line 229)
        str_131497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 28), 'str', '+')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 23), tuple_131495, str_131497)
        
        # Assigning a type to the variable '_writemodes' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), '_writemodes', tuple_131495)
        
        # Getting the type of 'mode' (line 230)
        mode_131498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 17), 'mode')
        # Testing the type of a for loop iterable (line 230)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 230, 8), mode_131498)
        # Getting the type of the for loop variable (line 230)
        for_loop_var_131499 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 230, 8), mode_131498)
        # Assigning a type to the variable 'c' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'c', for_loop_var_131499)
        # SSA begins for a for statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'c' (line 231)
        c_131500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'c')
        # Getting the type of '_writemodes' (line 231)
        _writemodes_131501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), '_writemodes')
        # Applying the binary operator 'in' (line 231)
        result_contains_131502 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 15), 'in', c_131500, _writemodes_131501)
        
        # Testing the type of an if condition (line 231)
        if_condition_131503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 12), result_contains_131502)
        # Assigning a type to the variable 'if_condition_131503' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'if_condition_131503', if_condition_131503)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 232)
        True_131504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'stypy_return_type', True_131504)
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 233)
        False_131505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'stypy_return_type', False_131505)
        
        # ################# End of '_iswritemode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_iswritemode' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_131506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131506)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_iswritemode'
        return stypy_return_type_131506


    @norecursion
    def _splitzipext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_splitzipext'
        module_type_store = module_type_store.open_function_context('_splitzipext', 235, 4, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource._splitzipext.__dict__.__setitem__('stypy_localization', localization)
        DataSource._splitzipext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource._splitzipext.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource._splitzipext.__dict__.__setitem__('stypy_function_name', 'DataSource._splitzipext')
        DataSource._splitzipext.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        DataSource._splitzipext.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource._splitzipext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource._splitzipext.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource._splitzipext.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource._splitzipext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource._splitzipext.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource._splitzipext', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_splitzipext', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_splitzipext(...)' code ##################

        str_131507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, (-1)), 'str', 'Split zip extension from filename and return filename.\n\n        *Returns*:\n            base, zip_ext : {tuple}\n\n        ')
        
        
        # Call to _iszip(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 'filename' (line 243)
        filename_131510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 23), 'filename', False)
        # Processing the call keyword arguments (line 243)
        kwargs_131511 = {}
        # Getting the type of 'self' (line 243)
        self_131508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'self', False)
        # Obtaining the member '_iszip' of a type (line 243)
        _iszip_131509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 11), self_131508, '_iszip')
        # Calling _iszip(args, kwargs) (line 243)
        _iszip_call_result_131512 = invoke(stypy.reporting.localization.Localization(__file__, 243, 11), _iszip_131509, *[filename_131510], **kwargs_131511)
        
        # Testing the type of an if condition (line 243)
        if_condition_131513 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), _iszip_call_result_131512)
        # Assigning a type to the variable 'if_condition_131513' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_131513', if_condition_131513)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to splitext(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'filename' (line 244)
        filename_131517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 36), 'filename', False)
        # Processing the call keyword arguments (line 244)
        kwargs_131518 = {}
        # Getting the type of 'os' (line 244)
        os_131514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 244)
        path_131515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 19), os_131514, 'path')
        # Obtaining the member 'splitext' of a type (line 244)
        splitext_131516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 19), path_131515, 'splitext')
        # Calling splitext(args, kwargs) (line 244)
        splitext_call_result_131519 = invoke(stypy.reporting.localization.Localization(__file__, 244, 19), splitext_131516, *[filename_131517], **kwargs_131518)
        
        # Assigning a type to the variable 'stypy_return_type' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'stypy_return_type', splitext_call_result_131519)
        # SSA branch for the else part of an if statement (line 243)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'tuple' (line 246)
        tuple_131520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 246)
        # Adding element type (line 246)
        # Getting the type of 'filename' (line 246)
        filename_131521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), tuple_131520, filename_131521)
        # Adding element type (line 246)
        # Getting the type of 'None' (line 246)
        None_131522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), tuple_131520, None_131522)
        
        # Assigning a type to the variable 'stypy_return_type' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'stypy_return_type', tuple_131520)
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_splitzipext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_splitzipext' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_131523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131523)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_splitzipext'
        return stypy_return_type_131523


    @norecursion
    def _possible_names(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_possible_names'
        module_type_store = module_type_store.open_function_context('_possible_names', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource._possible_names.__dict__.__setitem__('stypy_localization', localization)
        DataSource._possible_names.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource._possible_names.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource._possible_names.__dict__.__setitem__('stypy_function_name', 'DataSource._possible_names')
        DataSource._possible_names.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        DataSource._possible_names.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource._possible_names.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource._possible_names.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource._possible_names.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource._possible_names.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource._possible_names.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource._possible_names', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_possible_names', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_possible_names(...)' code ##################

        str_131524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 8), 'str', 'Return a tuple containing compressed filename variations.')
        
        # Assigning a List to a Name (line 250):
        
        # Assigning a List to a Name (line 250):
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_131525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        # Getting the type of 'filename' (line 250)
        filename_131526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 17), 'filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 16), list_131525, filename_131526)
        
        # Assigning a type to the variable 'names' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'names', list_131525)
        
        
        
        # Call to _iszip(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'filename' (line 251)
        filename_131529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 27), 'filename', False)
        # Processing the call keyword arguments (line 251)
        kwargs_131530 = {}
        # Getting the type of 'self' (line 251)
        self_131527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'self', False)
        # Obtaining the member '_iszip' of a type (line 251)
        _iszip_131528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), self_131527, '_iszip')
        # Calling _iszip(args, kwargs) (line 251)
        _iszip_call_result_131531 = invoke(stypy.reporting.localization.Localization(__file__, 251, 15), _iszip_131528, *[filename_131529], **kwargs_131530)
        
        # Applying the 'not' unary operator (line 251)
        result_not__131532 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 11), 'not', _iszip_call_result_131531)
        
        # Testing the type of an if condition (line 251)
        if_condition_131533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), result_not__131532)
        # Assigning a type to the variable 'if_condition_131533' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'if_condition_131533', if_condition_131533)
        # SSA begins for if statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to keys(...): (line 252)
        # Processing the call keyword arguments (line 252)
        kwargs_131536 = {}
        # Getting the type of '_file_openers' (line 252)
        _file_openers_131534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 26), '_file_openers', False)
        # Obtaining the member 'keys' of a type (line 252)
        keys_131535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 26), _file_openers_131534, 'keys')
        # Calling keys(args, kwargs) (line 252)
        keys_call_result_131537 = invoke(stypy.reporting.localization.Localization(__file__, 252, 26), keys_131535, *[], **kwargs_131536)
        
        # Testing the type of a for loop iterable (line 252)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 252, 12), keys_call_result_131537)
        # Getting the type of the for loop variable (line 252)
        for_loop_var_131538 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 252, 12), keys_call_result_131537)
        # Assigning a type to the variable 'zipext' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'zipext', for_loop_var_131538)
        # SSA begins for a for statement (line 252)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'zipext' (line 253)
        zipext_131539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'zipext')
        # Testing the type of an if condition (line 253)
        if_condition_131540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 16), zipext_131539)
        # Assigning a type to the variable 'if_condition_131540' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'if_condition_131540', if_condition_131540)
        # SSA begins for if statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'filename' (line 254)
        filename_131543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 33), 'filename', False)
        # Getting the type of 'zipext' (line 254)
        zipext_131544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 42), 'zipext', False)
        # Applying the binary operator '+' (line 254)
        result_add_131545 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 33), '+', filename_131543, zipext_131544)
        
        # Processing the call keyword arguments (line 254)
        kwargs_131546 = {}
        # Getting the type of 'names' (line 254)
        names_131541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'names', False)
        # Obtaining the member 'append' of a type (line 254)
        append_131542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 20), names_131541, 'append')
        # Calling append(args, kwargs) (line 254)
        append_call_result_131547 = invoke(stypy.reporting.localization.Localization(__file__, 254, 20), append_131542, *[result_add_131545], **kwargs_131546)
        
        # SSA join for if statement (line 253)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 251)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'names' (line 255)
        names_131548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'names')
        # Assigning a type to the variable 'stypy_return_type' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'stypy_return_type', names_131548)
        
        # ################# End of '_possible_names(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_possible_names' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_131549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131549)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_possible_names'
        return stypy_return_type_131549


    @norecursion
    def _isurl(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_isurl'
        module_type_store = module_type_store.open_function_context('_isurl', 257, 4, False)
        # Assigning a type to the variable 'self' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource._isurl.__dict__.__setitem__('stypy_localization', localization)
        DataSource._isurl.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource._isurl.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource._isurl.__dict__.__setitem__('stypy_function_name', 'DataSource._isurl')
        DataSource._isurl.__dict__.__setitem__('stypy_param_names_list', ['path'])
        DataSource._isurl.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource._isurl.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource._isurl.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource._isurl.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource._isurl.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource._isurl.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource._isurl', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_isurl', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_isurl(...)' code ##################

        str_131550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 8), 'str', 'Test if path is a net location.  Tests the scheme and netloc.')
        
        
        
        # Obtaining the type of the subscript
        int_131551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 28), 'int')
        # Getting the type of 'sys' (line 261)
        sys_131552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'sys')
        # Obtaining the member 'version_info' of a type (line 261)
        version_info_131553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 11), sys_131552, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___131554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 11), version_info_131553, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_131555 = invoke(stypy.reporting.localization.Localization(__file__, 261, 11), getitem___131554, int_131551)
        
        int_131556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 34), 'int')
        # Applying the binary operator '>=' (line 261)
        result_ge_131557 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), '>=', subscript_call_result_131555, int_131556)
        
        # Testing the type of an if condition (line 261)
        if_condition_131558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), result_ge_131557)
        # Assigning a type to the variable 'if_condition_131558' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_131558', if_condition_131558)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 262, 12))
        
        # 'from urllib.parse import urlparse' statement (line 262)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
        import_131559 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 262, 12), 'urllib.parse')

        if (type(import_131559) is not StypyTypeError):

            if (import_131559 != 'pyd_module'):
                __import__(import_131559)
                sys_modules_131560 = sys.modules[import_131559]
                import_from_module(stypy.reporting.localization.Localization(__file__, 262, 12), 'urllib.parse', sys_modules_131560.module_type_store, module_type_store, ['urlparse'])
                nest_module(stypy.reporting.localization.Localization(__file__, 262, 12), __file__, sys_modules_131560, sys_modules_131560.module_type_store, module_type_store)
            else:
                from urllib.parse import urlparse

                import_from_module(stypy.reporting.localization.Localization(__file__, 262, 12), 'urllib.parse', None, module_type_store, ['urlparse'], [urlparse])

        else:
            # Assigning a type to the variable 'urllib.parse' (line 262)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'urllib.parse', import_131559)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
        
        # SSA branch for the else part of an if statement (line 261)
        module_type_store.open_ssa_branch('else')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 264, 12))
        
        # 'from urlparse import urlparse' statement (line 264)
        from urlparse import urlparse

        import_from_module(stypy.reporting.localization.Localization(__file__, 264, 12), 'urlparse', None, module_type_store, ['urlparse'], [urlparse])
        
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 272):
        
        # Assigning a Call to a Name:
        
        # Call to urlparse(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'path' (line 272)
        path_131562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 65), 'path', False)
        # Processing the call keyword arguments (line 272)
        kwargs_131563 = {}
        # Getting the type of 'urlparse' (line 272)
        urlparse_131561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 56), 'urlparse', False)
        # Calling urlparse(args, kwargs) (line 272)
        urlparse_call_result_131564 = invoke(stypy.reporting.localization.Localization(__file__, 272, 56), urlparse_131561, *[path_131562], **kwargs_131563)
        
        # Assigning a type to the variable 'call_assignment_131349' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131349', urlparse_call_result_131564)
        
        # Assigning a Call to a Name (line 272):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131568 = {}
        # Getting the type of 'call_assignment_131349' (line 272)
        call_assignment_131349_131565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131349', False)
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___131566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), call_assignment_131349_131565, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131569 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131566, *[int_131567], **kwargs_131568)
        
        # Assigning a type to the variable 'call_assignment_131350' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131350', getitem___call_result_131569)
        
        # Assigning a Name to a Name (line 272):
        # Getting the type of 'call_assignment_131350' (line 272)
        call_assignment_131350_131570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131350')
        # Assigning a type to the variable 'scheme' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'scheme', call_assignment_131350_131570)
        
        # Assigning a Call to a Name (line 272):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131574 = {}
        # Getting the type of 'call_assignment_131349' (line 272)
        call_assignment_131349_131571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131349', False)
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___131572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), call_assignment_131349_131571, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131575 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131572, *[int_131573], **kwargs_131574)
        
        # Assigning a type to the variable 'call_assignment_131351' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131351', getitem___call_result_131575)
        
        # Assigning a Name to a Name (line 272):
        # Getting the type of 'call_assignment_131351' (line 272)
        call_assignment_131351_131576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131351')
        # Assigning a type to the variable 'netloc' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'netloc', call_assignment_131351_131576)
        
        # Assigning a Call to a Name (line 272):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131580 = {}
        # Getting the type of 'call_assignment_131349' (line 272)
        call_assignment_131349_131577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131349', False)
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___131578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), call_assignment_131349_131577, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131581 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131578, *[int_131579], **kwargs_131580)
        
        # Assigning a type to the variable 'call_assignment_131352' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131352', getitem___call_result_131581)
        
        # Assigning a Name to a Name (line 272):
        # Getting the type of 'call_assignment_131352' (line 272)
        call_assignment_131352_131582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131352')
        # Assigning a type to the variable 'upath' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 24), 'upath', call_assignment_131352_131582)
        
        # Assigning a Call to a Name (line 272):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131586 = {}
        # Getting the type of 'call_assignment_131349' (line 272)
        call_assignment_131349_131583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131349', False)
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___131584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), call_assignment_131349_131583, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131587 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131584, *[int_131585], **kwargs_131586)
        
        # Assigning a type to the variable 'call_assignment_131353' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131353', getitem___call_result_131587)
        
        # Assigning a Name to a Name (line 272):
        # Getting the type of 'call_assignment_131353' (line 272)
        call_assignment_131353_131588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131353')
        # Assigning a type to the variable 'uparams' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 31), 'uparams', call_assignment_131353_131588)
        
        # Assigning a Call to a Name (line 272):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131592 = {}
        # Getting the type of 'call_assignment_131349' (line 272)
        call_assignment_131349_131589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131349', False)
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___131590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), call_assignment_131349_131589, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131593 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131590, *[int_131591], **kwargs_131592)
        
        # Assigning a type to the variable 'call_assignment_131354' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131354', getitem___call_result_131593)
        
        # Assigning a Name to a Name (line 272):
        # Getting the type of 'call_assignment_131354' (line 272)
        call_assignment_131354_131594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131354')
        # Assigning a type to the variable 'uquery' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 40), 'uquery', call_assignment_131354_131594)
        
        # Assigning a Call to a Name (line 272):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131598 = {}
        # Getting the type of 'call_assignment_131349' (line 272)
        call_assignment_131349_131595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131349', False)
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___131596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), call_assignment_131349_131595, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131599 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131596, *[int_131597], **kwargs_131598)
        
        # Assigning a type to the variable 'call_assignment_131355' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131355', getitem___call_result_131599)
        
        # Assigning a Name to a Name (line 272):
        # Getting the type of 'call_assignment_131355' (line 272)
        call_assignment_131355_131600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'call_assignment_131355')
        # Assigning a type to the variable 'ufrag' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 48), 'ufrag', call_assignment_131355_131600)
        
        # Call to bool(...): (line 273)
        # Processing the call arguments (line 273)
        
        # Evaluating a boolean operation
        # Getting the type of 'scheme' (line 273)
        scheme_131602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'scheme', False)
        # Getting the type of 'netloc' (line 273)
        netloc_131603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 31), 'netloc', False)
        # Applying the binary operator 'and' (line 273)
        result_and_keyword_131604 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 20), 'and', scheme_131602, netloc_131603)
        
        # Processing the call keyword arguments (line 273)
        kwargs_131605 = {}
        # Getting the type of 'bool' (line 273)
        bool_131601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'bool', False)
        # Calling bool(args, kwargs) (line 273)
        bool_call_result_131606 = invoke(stypy.reporting.localization.Localization(__file__, 273, 15), bool_131601, *[result_and_keyword_131604], **kwargs_131605)
        
        # Assigning a type to the variable 'stypy_return_type' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'stypy_return_type', bool_call_result_131606)
        
        # ################# End of '_isurl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_isurl' in the type store
        # Getting the type of 'stypy_return_type' (line 257)
        stypy_return_type_131607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131607)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_isurl'
        return stypy_return_type_131607


    @norecursion
    def _cache(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cache'
        module_type_store = module_type_store.open_function_context('_cache', 275, 4, False)
        # Assigning a type to the variable 'self' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource._cache.__dict__.__setitem__('stypy_localization', localization)
        DataSource._cache.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource._cache.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource._cache.__dict__.__setitem__('stypy_function_name', 'DataSource._cache')
        DataSource._cache.__dict__.__setitem__('stypy_param_names_list', ['path'])
        DataSource._cache.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource._cache.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource._cache.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource._cache.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource._cache.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource._cache.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource._cache', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cache', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cache(...)' code ##################

        str_131608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, (-1)), 'str', 'Cache the file specified by path.\n\n        Creates a copy of the file in the datasource cache.\n\n        ')
        
        
        
        # Obtaining the type of the subscript
        int_131609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 28), 'int')
        # Getting the type of 'sys' (line 283)
        sys_131610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'sys')
        # Obtaining the member 'version_info' of a type (line 283)
        version_info_131611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 11), sys_131610, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 283)
        getitem___131612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 11), version_info_131611, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 283)
        subscript_call_result_131613 = invoke(stypy.reporting.localization.Localization(__file__, 283, 11), getitem___131612, int_131609)
        
        int_131614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 34), 'int')
        # Applying the binary operator '>=' (line 283)
        result_ge_131615 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 11), '>=', subscript_call_result_131613, int_131614)
        
        # Testing the type of an if condition (line 283)
        if_condition_131616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 8), result_ge_131615)
        # Assigning a type to the variable 'if_condition_131616' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'if_condition_131616', if_condition_131616)
        # SSA begins for if statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 284, 12))
        
        # 'from urllib.request import urlopen' statement (line 284)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
        import_131617 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 284, 12), 'urllib.request')

        if (type(import_131617) is not StypyTypeError):

            if (import_131617 != 'pyd_module'):
                __import__(import_131617)
                sys_modules_131618 = sys.modules[import_131617]
                import_from_module(stypy.reporting.localization.Localization(__file__, 284, 12), 'urllib.request', sys_modules_131618.module_type_store, module_type_store, ['urlopen'])
                nest_module(stypy.reporting.localization.Localization(__file__, 284, 12), __file__, sys_modules_131618, sys_modules_131618.module_type_store, module_type_store)
            else:
                from urllib.request import urlopen

                import_from_module(stypy.reporting.localization.Localization(__file__, 284, 12), 'urllib.request', None, module_type_store, ['urlopen'], [urlopen])

        else:
            # Assigning a type to the variable 'urllib.request' (line 284)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'urllib.request', import_131617)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 285, 12))
        
        # 'from urllib.error import URLError' statement (line 285)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
        import_131619 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 285, 12), 'urllib.error')

        if (type(import_131619) is not StypyTypeError):

            if (import_131619 != 'pyd_module'):
                __import__(import_131619)
                sys_modules_131620 = sys.modules[import_131619]
                import_from_module(stypy.reporting.localization.Localization(__file__, 285, 12), 'urllib.error', sys_modules_131620.module_type_store, module_type_store, ['URLError'])
                nest_module(stypy.reporting.localization.Localization(__file__, 285, 12), __file__, sys_modules_131620, sys_modules_131620.module_type_store, module_type_store)
            else:
                from urllib.error import URLError

                import_from_module(stypy.reporting.localization.Localization(__file__, 285, 12), 'urllib.error', None, module_type_store, ['URLError'], [URLError])

        else:
            # Assigning a type to the variable 'urllib.error' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'urllib.error', import_131619)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
        
        # SSA branch for the else part of an if statement (line 283)
        module_type_store.open_ssa_branch('else')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 287, 12))
        
        # 'from urllib2 import urlopen' statement (line 287)
        from urllib2 import urlopen

        import_from_module(stypy.reporting.localization.Localization(__file__, 287, 12), 'urllib2', None, module_type_store, ['urlopen'], [urlopen])
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 288, 12))
        
        # 'from urllib2 import URLError' statement (line 288)
        from urllib2 import URLError

        import_from_module(stypy.reporting.localization.Localization(__file__, 288, 12), 'urllib2', None, module_type_store, ['URLError'], [URLError])
        
        # SSA join for if statement (line 283)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to abspath(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'path' (line 290)
        path_131623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 29), 'path', False)
        # Processing the call keyword arguments (line 290)
        kwargs_131624 = {}
        # Getting the type of 'self' (line 290)
        self_131621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'self', False)
        # Obtaining the member 'abspath' of a type (line 290)
        abspath_131622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 16), self_131621, 'abspath')
        # Calling abspath(args, kwargs) (line 290)
        abspath_call_result_131625 = invoke(stypy.reporting.localization.Localization(__file__, 290, 16), abspath_131622, *[path_131623], **kwargs_131624)
        
        # Assigning a type to the variable 'upath' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'upath', abspath_call_result_131625)
        
        
        
        # Call to exists(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Call to dirname(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'upath' (line 293)
        upath_131632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 46), 'upath', False)
        # Processing the call keyword arguments (line 293)
        kwargs_131633 = {}
        # Getting the type of 'os' (line 293)
        os_131629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 293)
        path_131630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 30), os_131629, 'path')
        # Obtaining the member 'dirname' of a type (line 293)
        dirname_131631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 30), path_131630, 'dirname')
        # Calling dirname(args, kwargs) (line 293)
        dirname_call_result_131634 = invoke(stypy.reporting.localization.Localization(__file__, 293, 30), dirname_131631, *[upath_131632], **kwargs_131633)
        
        # Processing the call keyword arguments (line 293)
        kwargs_131635 = {}
        # Getting the type of 'os' (line 293)
        os_131626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 293)
        path_131627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 15), os_131626, 'path')
        # Obtaining the member 'exists' of a type (line 293)
        exists_131628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 15), path_131627, 'exists')
        # Calling exists(args, kwargs) (line 293)
        exists_call_result_131636 = invoke(stypy.reporting.localization.Localization(__file__, 293, 15), exists_131628, *[dirname_call_result_131634], **kwargs_131635)
        
        # Applying the 'not' unary operator (line 293)
        result_not__131637 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 11), 'not', exists_call_result_131636)
        
        # Testing the type of an if condition (line 293)
        if_condition_131638 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 8), result_not__131637)
        # Assigning a type to the variable 'if_condition_131638' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'if_condition_131638', if_condition_131638)
        # SSA begins for if statement (line 293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to makedirs(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Call to dirname(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'upath' (line 294)
        upath_131644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 40), 'upath', False)
        # Processing the call keyword arguments (line 294)
        kwargs_131645 = {}
        # Getting the type of 'os' (line 294)
        os_131641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 294)
        path_131642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 24), os_131641, 'path')
        # Obtaining the member 'dirname' of a type (line 294)
        dirname_131643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 24), path_131642, 'dirname')
        # Calling dirname(args, kwargs) (line 294)
        dirname_call_result_131646 = invoke(stypy.reporting.localization.Localization(__file__, 294, 24), dirname_131643, *[upath_131644], **kwargs_131645)
        
        # Processing the call keyword arguments (line 294)
        kwargs_131647 = {}
        # Getting the type of 'os' (line 294)
        os_131639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'os', False)
        # Obtaining the member 'makedirs' of a type (line 294)
        makedirs_131640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), os_131639, 'makedirs')
        # Calling makedirs(args, kwargs) (line 294)
        makedirs_call_result_131648 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), makedirs_131640, *[dirname_call_result_131646], **kwargs_131647)
        
        # SSA join for if statement (line 293)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to _isurl(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'path' (line 297)
        path_131651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 23), 'path', False)
        # Processing the call keyword arguments (line 297)
        kwargs_131652 = {}
        # Getting the type of 'self' (line 297)
        self_131649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'self', False)
        # Obtaining the member '_isurl' of a type (line 297)
        _isurl_131650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 11), self_131649, '_isurl')
        # Calling _isurl(args, kwargs) (line 297)
        _isurl_call_result_131653 = invoke(stypy.reporting.localization.Localization(__file__, 297, 11), _isurl_131650, *[path_131651], **kwargs_131652)
        
        # Testing the type of an if condition (line 297)
        if_condition_131654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 8), _isurl_call_result_131653)
        # Assigning a type to the variable 'if_condition_131654' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'if_condition_131654', if_condition_131654)
        # SSA begins for if statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to urlopen(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'path' (line 299)
        path_131656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 36), 'path', False)
        # Processing the call keyword arguments (line 299)
        kwargs_131657 = {}
        # Getting the type of 'urlopen' (line 299)
        urlopen_131655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 28), 'urlopen', False)
        # Calling urlopen(args, kwargs) (line 299)
        urlopen_call_result_131658 = invoke(stypy.reporting.localization.Localization(__file__, 299, 28), urlopen_131655, *[path_131656], **kwargs_131657)
        
        # Assigning a type to the variable 'openedurl' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'openedurl', urlopen_call_result_131658)
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to _open(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'upath' (line 300)
        upath_131660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 26), 'upath', False)
        str_131661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 33), 'str', 'wb')
        # Processing the call keyword arguments (line 300)
        kwargs_131662 = {}
        # Getting the type of '_open' (line 300)
        _open_131659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), '_open', False)
        # Calling _open(args, kwargs) (line 300)
        _open_call_result_131663 = invoke(stypy.reporting.localization.Localization(__file__, 300, 20), _open_131659, *[upath_131660, str_131661], **kwargs_131662)
        
        # Assigning a type to the variable 'f' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'f', _open_call_result_131663)
        
        # Try-finally block (line 301)
        
        # Call to copyfileobj(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'openedurl' (line 302)
        openedurl_131666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 39), 'openedurl', False)
        # Getting the type of 'f' (line 302)
        f_131667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 50), 'f', False)
        # Processing the call keyword arguments (line 302)
        kwargs_131668 = {}
        # Getting the type of 'shutil' (line 302)
        shutil_131664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 20), 'shutil', False)
        # Obtaining the member 'copyfileobj' of a type (line 302)
        copyfileobj_131665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 20), shutil_131664, 'copyfileobj')
        # Calling copyfileobj(args, kwargs) (line 302)
        copyfileobj_call_result_131669 = invoke(stypy.reporting.localization.Localization(__file__, 302, 20), copyfileobj_131665, *[openedurl_131666, f_131667], **kwargs_131668)
        
        
        # finally branch of the try-finally block (line 301)
        
        # Call to close(...): (line 304)
        # Processing the call keyword arguments (line 304)
        kwargs_131672 = {}
        # Getting the type of 'f' (line 304)
        f_131670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'f', False)
        # Obtaining the member 'close' of a type (line 304)
        close_131671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 20), f_131670, 'close')
        # Calling close(args, kwargs) (line 304)
        close_call_result_131673 = invoke(stypy.reporting.localization.Localization(__file__, 304, 20), close_131671, *[], **kwargs_131672)
        
        
        # Call to close(...): (line 305)
        # Processing the call keyword arguments (line 305)
        kwargs_131676 = {}
        # Getting the type of 'openedurl' (line 305)
        openedurl_131674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'openedurl', False)
        # Obtaining the member 'close' of a type (line 305)
        close_131675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 20), openedurl_131674, 'close')
        # Calling close(args, kwargs) (line 305)
        close_call_result_131677 = invoke(stypy.reporting.localization.Localization(__file__, 305, 20), close_131675, *[], **kwargs_131676)
        
        
        # SSA branch for the except part of a try statement (line 298)
        # SSA branch for the except 'URLError' branch of a try statement (line 298)
        module_type_store.open_ssa_branch('except')
        
        # Call to URLError(...): (line 307)
        # Processing the call arguments (line 307)
        str_131679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 31), 'str', 'URL not found: %s')
        # Getting the type of 'path' (line 307)
        path_131680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 53), 'path', False)
        # Applying the binary operator '%' (line 307)
        result_mod_131681 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 31), '%', str_131679, path_131680)
        
        # Processing the call keyword arguments (line 307)
        kwargs_131682 = {}
        # Getting the type of 'URLError' (line 307)
        URLError_131678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'URLError', False)
        # Calling URLError(args, kwargs) (line 307)
        URLError_call_result_131683 = invoke(stypy.reporting.localization.Localization(__file__, 307, 22), URLError_131678, *[result_mod_131681], **kwargs_131682)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 307, 16), URLError_call_result_131683, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 298)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 297)
        module_type_store.open_ssa_branch('else')
        
        # Call to copyfile(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'path' (line 309)
        path_131686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'path', False)
        # Getting the type of 'upath' (line 309)
        upath_131687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 34), 'upath', False)
        # Processing the call keyword arguments (line 309)
        kwargs_131688 = {}
        # Getting the type of 'shutil' (line 309)
        shutil_131684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'shutil', False)
        # Obtaining the member 'copyfile' of a type (line 309)
        copyfile_131685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), shutil_131684, 'copyfile')
        # Calling copyfile(args, kwargs) (line 309)
        copyfile_call_result_131689 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), copyfile_131685, *[path_131686, upath_131687], **kwargs_131688)
        
        # SSA join for if statement (line 297)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'upath' (line 310)
        upath_131690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'upath')
        # Assigning a type to the variable 'stypy_return_type' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'stypy_return_type', upath_131690)
        
        # ################# End of '_cache(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cache' in the type store
        # Getting the type of 'stypy_return_type' (line 275)
        stypy_return_type_131691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131691)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cache'
        return stypy_return_type_131691


    @norecursion
    def _findfile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_findfile'
        module_type_store = module_type_store.open_function_context('_findfile', 312, 4, False)
        # Assigning a type to the variable 'self' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource._findfile.__dict__.__setitem__('stypy_localization', localization)
        DataSource._findfile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource._findfile.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource._findfile.__dict__.__setitem__('stypy_function_name', 'DataSource._findfile')
        DataSource._findfile.__dict__.__setitem__('stypy_param_names_list', ['path'])
        DataSource._findfile.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource._findfile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource._findfile.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource._findfile.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource._findfile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource._findfile.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource._findfile', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_findfile', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_findfile(...)' code ##################

        str_131692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, (-1)), 'str', 'Searches for ``path`` and returns full path if found.\n\n        If path is an URL, _findfile will cache a local copy and return the\n        path to the cached file.  If path is a local file, _findfile will\n        return a path to that local file.\n\n        The search will include possible compressed versions of the file\n        and return the first occurence found.\n\n        ')
        
        
        
        # Call to _isurl(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'path' (line 325)
        path_131695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 27), 'path', False)
        # Processing the call keyword arguments (line 325)
        kwargs_131696 = {}
        # Getting the type of 'self' (line 325)
        self_131693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'self', False)
        # Obtaining the member '_isurl' of a type (line 325)
        _isurl_131694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 15), self_131693, '_isurl')
        # Calling _isurl(args, kwargs) (line 325)
        _isurl_call_result_131697 = invoke(stypy.reporting.localization.Localization(__file__, 325, 15), _isurl_131694, *[path_131695], **kwargs_131696)
        
        # Applying the 'not' unary operator (line 325)
        result_not__131698 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 11), 'not', _isurl_call_result_131697)
        
        # Testing the type of an if condition (line 325)
        if_condition_131699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 8), result_not__131698)
        # Assigning a type to the variable 'if_condition_131699' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'if_condition_131699', if_condition_131699)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 327):
        
        # Assigning a Call to a Name (line 327):
        
        # Call to _possible_names(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'path' (line 327)
        path_131702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 44), 'path', False)
        # Processing the call keyword arguments (line 327)
        kwargs_131703 = {}
        # Getting the type of 'self' (line 327)
        self_131700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 23), 'self', False)
        # Obtaining the member '_possible_names' of a type (line 327)
        _possible_names_131701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 23), self_131700, '_possible_names')
        # Calling _possible_names(args, kwargs) (line 327)
        _possible_names_call_result_131704 = invoke(stypy.reporting.localization.Localization(__file__, 327, 23), _possible_names_131701, *[path_131702], **kwargs_131703)
        
        # Assigning a type to the variable 'filelist' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'filelist', _possible_names_call_result_131704)
        
        # Getting the type of 'filelist' (line 329)
        filelist_131705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'filelist')
        
        # Call to _possible_names(...): (line 329)
        # Processing the call arguments (line 329)
        
        # Call to abspath(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'path' (line 329)
        path_131710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 58), 'path', False)
        # Processing the call keyword arguments (line 329)
        kwargs_131711 = {}
        # Getting the type of 'self' (line 329)
        self_131708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 45), 'self', False)
        # Obtaining the member 'abspath' of a type (line 329)
        abspath_131709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 45), self_131708, 'abspath')
        # Calling abspath(args, kwargs) (line 329)
        abspath_call_result_131712 = invoke(stypy.reporting.localization.Localization(__file__, 329, 45), abspath_131709, *[path_131710], **kwargs_131711)
        
        # Processing the call keyword arguments (line 329)
        kwargs_131713 = {}
        # Getting the type of 'self' (line 329)
        self_131706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 24), 'self', False)
        # Obtaining the member '_possible_names' of a type (line 329)
        _possible_names_131707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 24), self_131706, '_possible_names')
        # Calling _possible_names(args, kwargs) (line 329)
        _possible_names_call_result_131714 = invoke(stypy.reporting.localization.Localization(__file__, 329, 24), _possible_names_131707, *[abspath_call_result_131712], **kwargs_131713)
        
        # Applying the binary operator '+=' (line 329)
        result_iadd_131715 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 12), '+=', filelist_131705, _possible_names_call_result_131714)
        # Assigning a type to the variable 'filelist' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'filelist', result_iadd_131715)
        
        # SSA branch for the else part of an if statement (line 325)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 332):
        
        # Assigning a Call to a Name (line 332):
        
        # Call to _possible_names(...): (line 332)
        # Processing the call arguments (line 332)
        
        # Call to abspath(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'path' (line 332)
        path_131720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 57), 'path', False)
        # Processing the call keyword arguments (line 332)
        kwargs_131721 = {}
        # Getting the type of 'self' (line 332)
        self_131718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 44), 'self', False)
        # Obtaining the member 'abspath' of a type (line 332)
        abspath_131719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 44), self_131718, 'abspath')
        # Calling abspath(args, kwargs) (line 332)
        abspath_call_result_131722 = invoke(stypy.reporting.localization.Localization(__file__, 332, 44), abspath_131719, *[path_131720], **kwargs_131721)
        
        # Processing the call keyword arguments (line 332)
        kwargs_131723 = {}
        # Getting the type of 'self' (line 332)
        self_131716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 23), 'self', False)
        # Obtaining the member '_possible_names' of a type (line 332)
        _possible_names_131717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 23), self_131716, '_possible_names')
        # Calling _possible_names(args, kwargs) (line 332)
        _possible_names_call_result_131724 = invoke(stypy.reporting.localization.Localization(__file__, 332, 23), _possible_names_131717, *[abspath_call_result_131722], **kwargs_131723)
        
        # Assigning a type to the variable 'filelist' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'filelist', _possible_names_call_result_131724)
        
        # Assigning a BinOp to a Name (line 334):
        
        # Assigning a BinOp to a Name (line 334):
        # Getting the type of 'filelist' (line 334)
        filelist_131725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 23), 'filelist')
        
        # Call to _possible_names(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'path' (line 334)
        path_131728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 55), 'path', False)
        # Processing the call keyword arguments (line 334)
        kwargs_131729 = {}
        # Getting the type of 'self' (line 334)
        self_131726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 34), 'self', False)
        # Obtaining the member '_possible_names' of a type (line 334)
        _possible_names_131727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 34), self_131726, '_possible_names')
        # Calling _possible_names(args, kwargs) (line 334)
        _possible_names_call_result_131730 = invoke(stypy.reporting.localization.Localization(__file__, 334, 34), _possible_names_131727, *[path_131728], **kwargs_131729)
        
        # Applying the binary operator '+' (line 334)
        result_add_131731 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 23), '+', filelist_131725, _possible_names_call_result_131730)
        
        # Assigning a type to the variable 'filelist' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'filelist', result_add_131731)
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'filelist' (line 336)
        filelist_131732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'filelist')
        # Testing the type of a for loop iterable (line 336)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 336, 8), filelist_131732)
        # Getting the type of the for loop variable (line 336)
        for_loop_var_131733 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 336, 8), filelist_131732)
        # Assigning a type to the variable 'name' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'name', for_loop_var_131733)
        # SSA begins for a for statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to exists(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'name' (line 337)
        name_131736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 27), 'name', False)
        # Processing the call keyword arguments (line 337)
        kwargs_131737 = {}
        # Getting the type of 'self' (line 337)
        self_131734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'self', False)
        # Obtaining the member 'exists' of a type (line 337)
        exists_131735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 15), self_131734, 'exists')
        # Calling exists(args, kwargs) (line 337)
        exists_call_result_131738 = invoke(stypy.reporting.localization.Localization(__file__, 337, 15), exists_131735, *[name_131736], **kwargs_131737)
        
        # Testing the type of an if condition (line 337)
        if_condition_131739 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 337, 12), exists_call_result_131738)
        # Assigning a type to the variable 'if_condition_131739' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'if_condition_131739', if_condition_131739)
        # SSA begins for if statement (line 337)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to _isurl(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'name' (line 338)
        name_131742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 31), 'name', False)
        # Processing the call keyword arguments (line 338)
        kwargs_131743 = {}
        # Getting the type of 'self' (line 338)
        self_131740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 19), 'self', False)
        # Obtaining the member '_isurl' of a type (line 338)
        _isurl_131741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 19), self_131740, '_isurl')
        # Calling _isurl(args, kwargs) (line 338)
        _isurl_call_result_131744 = invoke(stypy.reporting.localization.Localization(__file__, 338, 19), _isurl_131741, *[name_131742], **kwargs_131743)
        
        # Testing the type of an if condition (line 338)
        if_condition_131745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 16), _isurl_call_result_131744)
        # Assigning a type to the variable 'if_condition_131745' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'if_condition_131745', if_condition_131745)
        # SSA begins for if statement (line 338)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 339):
        
        # Assigning a Call to a Name (line 339):
        
        # Call to _cache(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'name' (line 339)
        name_131748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 39), 'name', False)
        # Processing the call keyword arguments (line 339)
        kwargs_131749 = {}
        # Getting the type of 'self' (line 339)
        self_131746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 27), 'self', False)
        # Obtaining the member '_cache' of a type (line 339)
        _cache_131747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 27), self_131746, '_cache')
        # Calling _cache(args, kwargs) (line 339)
        _cache_call_result_131750 = invoke(stypy.reporting.localization.Localization(__file__, 339, 27), _cache_131747, *[name_131748], **kwargs_131749)
        
        # Assigning a type to the variable 'name' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 20), 'name', _cache_call_result_131750)
        # SSA join for if statement (line 338)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'name' (line 340)
        name_131751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 23), 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'stypy_return_type', name_131751)
        # SSA join for if statement (line 337)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 341)
        None_131752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'stypy_return_type', None_131752)
        
        # ################# End of '_findfile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_findfile' in the type store
        # Getting the type of 'stypy_return_type' (line 312)
        stypy_return_type_131753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131753)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_findfile'
        return stypy_return_type_131753


    @norecursion
    def abspath(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'abspath'
        module_type_store = module_type_store.open_function_context('abspath', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource.abspath.__dict__.__setitem__('stypy_localization', localization)
        DataSource.abspath.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource.abspath.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource.abspath.__dict__.__setitem__('stypy_function_name', 'DataSource.abspath')
        DataSource.abspath.__dict__.__setitem__('stypy_param_names_list', ['path'])
        DataSource.abspath.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource.abspath.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource.abspath.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource.abspath.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource.abspath.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource.abspath.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource.abspath', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'abspath', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'abspath(...)' code ##################

        str_131754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, (-1)), 'str', '\n        Return absolute path of file in the DataSource directory.\n\n        If `path` is an URL, then `abspath` will return either the location\n        the file exists locally or the location it would exist when opened\n        using the `open` method.\n\n        Parameters\n        ----------\n        path : str\n            Can be a local file or a remote URL.\n\n        Returns\n        -------\n        out : str\n            Complete path, including the `DataSource` destination directory.\n\n        Notes\n        -----\n        The functionality is based on `os.path.abspath`.\n\n        ')
        
        
        
        # Obtaining the type of the subscript
        int_131755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 28), 'int')
        # Getting the type of 'sys' (line 367)
        sys_131756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 11), 'sys')
        # Obtaining the member 'version_info' of a type (line 367)
        version_info_131757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 11), sys_131756, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___131758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 11), version_info_131757, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 367)
        subscript_call_result_131759 = invoke(stypy.reporting.localization.Localization(__file__, 367, 11), getitem___131758, int_131755)
        
        int_131760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 34), 'int')
        # Applying the binary operator '>=' (line 367)
        result_ge_131761 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 11), '>=', subscript_call_result_131759, int_131760)
        
        # Testing the type of an if condition (line 367)
        if_condition_131762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 8), result_ge_131761)
        # Assigning a type to the variable 'if_condition_131762' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'if_condition_131762', if_condition_131762)
        # SSA begins for if statement (line 367)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 368, 12))
        
        # 'from urllib.parse import urlparse' statement (line 368)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
        import_131763 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 368, 12), 'urllib.parse')

        if (type(import_131763) is not StypyTypeError):

            if (import_131763 != 'pyd_module'):
                __import__(import_131763)
                sys_modules_131764 = sys.modules[import_131763]
                import_from_module(stypy.reporting.localization.Localization(__file__, 368, 12), 'urllib.parse', sys_modules_131764.module_type_store, module_type_store, ['urlparse'])
                nest_module(stypy.reporting.localization.Localization(__file__, 368, 12), __file__, sys_modules_131764, sys_modules_131764.module_type_store, module_type_store)
            else:
                from urllib.parse import urlparse

                import_from_module(stypy.reporting.localization.Localization(__file__, 368, 12), 'urllib.parse', None, module_type_store, ['urlparse'], [urlparse])

        else:
            # Assigning a type to the variable 'urllib.parse' (line 368)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'urllib.parse', import_131763)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
        
        # SSA branch for the else part of an if statement (line 367)
        module_type_store.open_ssa_branch('else')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 370, 12))
        
        # 'from urlparse import urlparse' statement (line 370)
        from urlparse import urlparse

        import_from_module(stypy.reporting.localization.Localization(__file__, 370, 12), 'urlparse', None, module_type_store, ['urlparse'], [urlparse])
        
        # SSA join for if statement (line 367)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 380):
        
        # Assigning a Call to a Name (line 380):
        
        # Call to split(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'self' (line 380)
        self_131767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 31), 'self', False)
        # Obtaining the member '_destpath' of a type (line 380)
        _destpath_131768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 31), self_131767, '_destpath')
        int_131769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 47), 'int')
        # Processing the call keyword arguments (line 380)
        kwargs_131770 = {}
        # Getting the type of 'path' (line 380)
        path_131765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 20), 'path', False)
        # Obtaining the member 'split' of a type (line 380)
        split_131766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 20), path_131765, 'split')
        # Calling split(args, kwargs) (line 380)
        split_call_result_131771 = invoke(stypy.reporting.localization.Localization(__file__, 380, 20), split_131766, *[_destpath_131768, int_131769], **kwargs_131770)
        
        # Assigning a type to the variable 'splitpath' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'splitpath', split_call_result_131771)
        
        
        
        # Call to len(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'splitpath' (line 381)
        splitpath_131773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 15), 'splitpath', False)
        # Processing the call keyword arguments (line 381)
        kwargs_131774 = {}
        # Getting the type of 'len' (line 381)
        len_131772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'len', False)
        # Calling len(args, kwargs) (line 381)
        len_call_result_131775 = invoke(stypy.reporting.localization.Localization(__file__, 381, 11), len_131772, *[splitpath_131773], **kwargs_131774)
        
        int_131776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 28), 'int')
        # Applying the binary operator '>' (line 381)
        result_gt_131777 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 11), '>', len_call_result_131775, int_131776)
        
        # Testing the type of an if condition (line 381)
        if_condition_131778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 8), result_gt_131777)
        # Assigning a type to the variable 'if_condition_131778' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'if_condition_131778', if_condition_131778)
        # SSA begins for if statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 382):
        
        # Assigning a Subscript to a Name (line 382):
        
        # Obtaining the type of the subscript
        int_131779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 29), 'int')
        # Getting the type of 'splitpath' (line 382)
        splitpath_131780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'splitpath')
        # Obtaining the member '__getitem__' of a type (line 382)
        getitem___131781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 19), splitpath_131780, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 382)
        subscript_call_result_131782 = invoke(stypy.reporting.localization.Localization(__file__, 382, 19), getitem___131781, int_131779)
        
        # Assigning a type to the variable 'path' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'path', subscript_call_result_131782)
        # SSA join for if statement (line 381)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 383):
        
        # Assigning a Call to a Name:
        
        # Call to urlparse(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'path' (line 383)
        path_131784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 65), 'path', False)
        # Processing the call keyword arguments (line 383)
        kwargs_131785 = {}
        # Getting the type of 'urlparse' (line 383)
        urlparse_131783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 56), 'urlparse', False)
        # Calling urlparse(args, kwargs) (line 383)
        urlparse_call_result_131786 = invoke(stypy.reporting.localization.Localization(__file__, 383, 56), urlparse_131783, *[path_131784], **kwargs_131785)
        
        # Assigning a type to the variable 'call_assignment_131356' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131356', urlparse_call_result_131786)
        
        # Assigning a Call to a Name (line 383):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131790 = {}
        # Getting the type of 'call_assignment_131356' (line 383)
        call_assignment_131356_131787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131356', False)
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___131788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), call_assignment_131356_131787, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131791 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131788, *[int_131789], **kwargs_131790)
        
        # Assigning a type to the variable 'call_assignment_131357' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131357', getitem___call_result_131791)
        
        # Assigning a Name to a Name (line 383):
        # Getting the type of 'call_assignment_131357' (line 383)
        call_assignment_131357_131792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131357')
        # Assigning a type to the variable 'scheme' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'scheme', call_assignment_131357_131792)
        
        # Assigning a Call to a Name (line 383):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131796 = {}
        # Getting the type of 'call_assignment_131356' (line 383)
        call_assignment_131356_131793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131356', False)
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___131794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), call_assignment_131356_131793, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131797 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131794, *[int_131795], **kwargs_131796)
        
        # Assigning a type to the variable 'call_assignment_131358' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131358', getitem___call_result_131797)
        
        # Assigning a Name to a Name (line 383):
        # Getting the type of 'call_assignment_131358' (line 383)
        call_assignment_131358_131798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131358')
        # Assigning a type to the variable 'netloc' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'netloc', call_assignment_131358_131798)
        
        # Assigning a Call to a Name (line 383):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131802 = {}
        # Getting the type of 'call_assignment_131356' (line 383)
        call_assignment_131356_131799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131356', False)
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___131800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), call_assignment_131356_131799, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131803 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131800, *[int_131801], **kwargs_131802)
        
        # Assigning a type to the variable 'call_assignment_131359' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131359', getitem___call_result_131803)
        
        # Assigning a Name to a Name (line 383):
        # Getting the type of 'call_assignment_131359' (line 383)
        call_assignment_131359_131804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131359')
        # Assigning a type to the variable 'upath' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 24), 'upath', call_assignment_131359_131804)
        
        # Assigning a Call to a Name (line 383):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131808 = {}
        # Getting the type of 'call_assignment_131356' (line 383)
        call_assignment_131356_131805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131356', False)
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___131806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), call_assignment_131356_131805, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131809 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131806, *[int_131807], **kwargs_131808)
        
        # Assigning a type to the variable 'call_assignment_131360' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131360', getitem___call_result_131809)
        
        # Assigning a Name to a Name (line 383):
        # Getting the type of 'call_assignment_131360' (line 383)
        call_assignment_131360_131810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131360')
        # Assigning a type to the variable 'uparams' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 31), 'uparams', call_assignment_131360_131810)
        
        # Assigning a Call to a Name (line 383):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131814 = {}
        # Getting the type of 'call_assignment_131356' (line 383)
        call_assignment_131356_131811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131356', False)
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___131812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), call_assignment_131356_131811, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131815 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131812, *[int_131813], **kwargs_131814)
        
        # Assigning a type to the variable 'call_assignment_131361' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131361', getitem___call_result_131815)
        
        # Assigning a Name to a Name (line 383):
        # Getting the type of 'call_assignment_131361' (line 383)
        call_assignment_131361_131816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131361')
        # Assigning a type to the variable 'uquery' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 40), 'uquery', call_assignment_131361_131816)
        
        # Assigning a Call to a Name (line 383):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 8), 'int')
        # Processing the call keyword arguments
        kwargs_131820 = {}
        # Getting the type of 'call_assignment_131356' (line 383)
        call_assignment_131356_131817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131356', False)
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___131818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), call_assignment_131356_131817, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131821 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131818, *[int_131819], **kwargs_131820)
        
        # Assigning a type to the variable 'call_assignment_131362' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131362', getitem___call_result_131821)
        
        # Assigning a Name to a Name (line 383):
        # Getting the type of 'call_assignment_131362' (line 383)
        call_assignment_131362_131822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'call_assignment_131362')
        # Assigning a type to the variable 'ufrag' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 48), 'ufrag', call_assignment_131362_131822)
        
        # Assigning a Call to a Name (line 384):
        
        # Assigning a Call to a Name (line 384):
        
        # Call to _sanitize_relative_path(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'netloc' (line 384)
        netloc_131825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 46), 'netloc', False)
        # Processing the call keyword arguments (line 384)
        kwargs_131826 = {}
        # Getting the type of 'self' (line 384)
        self_131823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 17), 'self', False)
        # Obtaining the member '_sanitize_relative_path' of a type (line 384)
        _sanitize_relative_path_131824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 17), self_131823, '_sanitize_relative_path')
        # Calling _sanitize_relative_path(args, kwargs) (line 384)
        _sanitize_relative_path_call_result_131827 = invoke(stypy.reporting.localization.Localization(__file__, 384, 17), _sanitize_relative_path_131824, *[netloc_131825], **kwargs_131826)
        
        # Assigning a type to the variable 'netloc' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'netloc', _sanitize_relative_path_call_result_131827)
        
        # Assigning a Call to a Name (line 385):
        
        # Assigning a Call to a Name (line 385):
        
        # Call to _sanitize_relative_path(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'upath' (line 385)
        upath_131830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 45), 'upath', False)
        # Processing the call keyword arguments (line 385)
        kwargs_131831 = {}
        # Getting the type of 'self' (line 385)
        self_131828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'self', False)
        # Obtaining the member '_sanitize_relative_path' of a type (line 385)
        _sanitize_relative_path_131829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 16), self_131828, '_sanitize_relative_path')
        # Calling _sanitize_relative_path(args, kwargs) (line 385)
        _sanitize_relative_path_call_result_131832 = invoke(stypy.reporting.localization.Localization(__file__, 385, 16), _sanitize_relative_path_131829, *[upath_131830], **kwargs_131831)
        
        # Assigning a type to the variable 'upath' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'upath', _sanitize_relative_path_call_result_131832)
        
        # Call to join(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'self' (line 386)
        self_131836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 28), 'self', False)
        # Obtaining the member '_destpath' of a type (line 386)
        _destpath_131837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 28), self_131836, '_destpath')
        # Getting the type of 'netloc' (line 386)
        netloc_131838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 44), 'netloc', False)
        # Getting the type of 'upath' (line 386)
        upath_131839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 52), 'upath', False)
        # Processing the call keyword arguments (line 386)
        kwargs_131840 = {}
        # Getting the type of 'os' (line 386)
        os_131833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 386)
        path_131834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 15), os_131833, 'path')
        # Obtaining the member 'join' of a type (line 386)
        join_131835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 15), path_131834, 'join')
        # Calling join(args, kwargs) (line 386)
        join_call_result_131841 = invoke(stypy.reporting.localization.Localization(__file__, 386, 15), join_131835, *[_destpath_131837, netloc_131838, upath_131839], **kwargs_131840)
        
        # Assigning a type to the variable 'stypy_return_type' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'stypy_return_type', join_call_result_131841)
        
        # ################# End of 'abspath(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'abspath' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_131842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131842)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'abspath'
        return stypy_return_type_131842


    @norecursion
    def _sanitize_relative_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sanitize_relative_path'
        module_type_store = module_type_store.open_function_context('_sanitize_relative_path', 388, 4, False)
        # Assigning a type to the variable 'self' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_localization', localization)
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_function_name', 'DataSource._sanitize_relative_path')
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_param_names_list', ['path'])
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource._sanitize_relative_path.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource._sanitize_relative_path', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sanitize_relative_path', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sanitize_relative_path(...)' code ##################

        str_131843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, (-1)), 'str', 'Return a sanitised relative path for which\n        os.path.abspath(os.path.join(base, path)).startswith(base)\n        ')
        
        # Assigning a Name to a Name (line 392):
        
        # Assigning a Name to a Name (line 392):
        # Getting the type of 'None' (line 392)
        None_131844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'None')
        # Assigning a type to the variable 'last' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'last', None_131844)
        
        # Assigning a Call to a Name (line 393):
        
        # Assigning a Call to a Name (line 393):
        
        # Call to normpath(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'path' (line 393)
        path_131848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 32), 'path', False)
        # Processing the call keyword arguments (line 393)
        kwargs_131849 = {}
        # Getting the type of 'os' (line 393)
        os_131845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 393)
        path_131846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 15), os_131845, 'path')
        # Obtaining the member 'normpath' of a type (line 393)
        normpath_131847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 15), path_131846, 'normpath')
        # Calling normpath(args, kwargs) (line 393)
        normpath_call_result_131850 = invoke(stypy.reporting.localization.Localization(__file__, 393, 15), normpath_131847, *[path_131848], **kwargs_131849)
        
        # Assigning a type to the variable 'path' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'path', normpath_call_result_131850)
        
        
        # Getting the type of 'path' (line 394)
        path_131851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 14), 'path')
        # Getting the type of 'last' (line 394)
        last_131852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 22), 'last')
        # Applying the binary operator '!=' (line 394)
        result_ne_131853 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 14), '!=', path_131851, last_131852)
        
        # Testing the type of an if condition (line 394)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 8), result_ne_131853)
        # SSA begins for while statement (line 394)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Name to a Name (line 395):
        
        # Assigning a Name to a Name (line 395):
        # Getting the type of 'path' (line 395)
        path_131854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'path')
        # Assigning a type to the variable 'last' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'last', path_131854)
        
        # Assigning a Call to a Name (line 397):
        
        # Assigning a Call to a Name (line 397):
        
        # Call to lstrip(...): (line 397)
        # Processing the call arguments (line 397)
        str_131862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 46), 'str', '/')
        # Processing the call keyword arguments (line 397)
        kwargs_131863 = {}
        
        # Call to lstrip(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'os' (line 397)
        os_131857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 31), 'os', False)
        # Obtaining the member 'sep' of a type (line 397)
        sep_131858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 31), os_131857, 'sep')
        # Processing the call keyword arguments (line 397)
        kwargs_131859 = {}
        # Getting the type of 'path' (line 397)
        path_131855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 'path', False)
        # Obtaining the member 'lstrip' of a type (line 397)
        lstrip_131856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 19), path_131855, 'lstrip')
        # Calling lstrip(args, kwargs) (line 397)
        lstrip_call_result_131860 = invoke(stypy.reporting.localization.Localization(__file__, 397, 19), lstrip_131856, *[sep_131858], **kwargs_131859)
        
        # Obtaining the member 'lstrip' of a type (line 397)
        lstrip_131861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 19), lstrip_call_result_131860, 'lstrip')
        # Calling lstrip(args, kwargs) (line 397)
        lstrip_call_result_131864 = invoke(stypy.reporting.localization.Localization(__file__, 397, 19), lstrip_131861, *[str_131862], **kwargs_131863)
        
        # Assigning a type to the variable 'path' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'path', lstrip_call_result_131864)
        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Call to lstrip(...): (line 398)
        # Processing the call arguments (line 398)
        str_131872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 49), 'str', '..')
        # Processing the call keyword arguments (line 398)
        kwargs_131873 = {}
        
        # Call to lstrip(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'os' (line 398)
        os_131867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 31), 'os', False)
        # Obtaining the member 'pardir' of a type (line 398)
        pardir_131868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 31), os_131867, 'pardir')
        # Processing the call keyword arguments (line 398)
        kwargs_131869 = {}
        # Getting the type of 'path' (line 398)
        path_131865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'path', False)
        # Obtaining the member 'lstrip' of a type (line 398)
        lstrip_131866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 19), path_131865, 'lstrip')
        # Calling lstrip(args, kwargs) (line 398)
        lstrip_call_result_131870 = invoke(stypy.reporting.localization.Localization(__file__, 398, 19), lstrip_131866, *[pardir_131868], **kwargs_131869)
        
        # Obtaining the member 'lstrip' of a type (line 398)
        lstrip_131871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 19), lstrip_call_result_131870, 'lstrip')
        # Calling lstrip(args, kwargs) (line 398)
        lstrip_call_result_131874 = invoke(stypy.reporting.localization.Localization(__file__, 398, 19), lstrip_131871, *[str_131872], **kwargs_131873)
        
        # Assigning a type to the variable 'path' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'path', lstrip_call_result_131874)
        
        # Assigning a Call to a Tuple (line 399):
        
        # Assigning a Call to a Name:
        
        # Call to splitdrive(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'path' (line 399)
        path_131878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 45), 'path', False)
        # Processing the call keyword arguments (line 399)
        kwargs_131879 = {}
        # Getting the type of 'os' (line 399)
        os_131875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 399)
        path_131876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 26), os_131875, 'path')
        # Obtaining the member 'splitdrive' of a type (line 399)
        splitdrive_131877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 26), path_131876, 'splitdrive')
        # Calling splitdrive(args, kwargs) (line 399)
        splitdrive_call_result_131880 = invoke(stypy.reporting.localization.Localization(__file__, 399, 26), splitdrive_131877, *[path_131878], **kwargs_131879)
        
        # Assigning a type to the variable 'call_assignment_131363' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'call_assignment_131363', splitdrive_call_result_131880)
        
        # Assigning a Call to a Name (line 399):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 12), 'int')
        # Processing the call keyword arguments
        kwargs_131884 = {}
        # Getting the type of 'call_assignment_131363' (line 399)
        call_assignment_131363_131881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'call_assignment_131363', False)
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___131882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), call_assignment_131363_131881, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131885 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131882, *[int_131883], **kwargs_131884)
        
        # Assigning a type to the variable 'call_assignment_131364' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'call_assignment_131364', getitem___call_result_131885)
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'call_assignment_131364' (line 399)
        call_assignment_131364_131886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'call_assignment_131364')
        # Assigning a type to the variable 'drive' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'drive', call_assignment_131364_131886)
        
        # Assigning a Call to a Name (line 399):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 12), 'int')
        # Processing the call keyword arguments
        kwargs_131890 = {}
        # Getting the type of 'call_assignment_131363' (line 399)
        call_assignment_131363_131887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'call_assignment_131363', False)
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___131888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), call_assignment_131363_131887, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131891 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131888, *[int_131889], **kwargs_131890)
        
        # Assigning a type to the variable 'call_assignment_131365' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'call_assignment_131365', getitem___call_result_131891)
        
        # Assigning a Name to a Name (line 399):
        # Getting the type of 'call_assignment_131365' (line 399)
        call_assignment_131365_131892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'call_assignment_131365')
        # Assigning a type to the variable 'path' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 19), 'path', call_assignment_131365_131892)
        # SSA join for while statement (line 394)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'path' (line 400)
        path_131893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 15), 'path')
        # Assigning a type to the variable 'stypy_return_type' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'stypy_return_type', path_131893)
        
        # ################# End of '_sanitize_relative_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sanitize_relative_path' in the type store
        # Getting the type of 'stypy_return_type' (line 388)
        stypy_return_type_131894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131894)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sanitize_relative_path'
        return stypy_return_type_131894


    @norecursion
    def exists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'exists'
        module_type_store = module_type_store.open_function_context('exists', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource.exists.__dict__.__setitem__('stypy_localization', localization)
        DataSource.exists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource.exists.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource.exists.__dict__.__setitem__('stypy_function_name', 'DataSource.exists')
        DataSource.exists.__dict__.__setitem__('stypy_param_names_list', ['path'])
        DataSource.exists.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource.exists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource.exists.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource.exists.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource.exists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource.exists.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource.exists', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'exists', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'exists(...)' code ##################

        str_131895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, (-1)), 'str', "\n        Test if path exists.\n\n        Test if `path` exists as (and in this order):\n\n        - a local file.\n        - a remote URL that has been downloaded and stored locally in the\n          `DataSource` directory.\n        - a remote URL that has not been downloaded, but is valid and\n          accessible.\n\n        Parameters\n        ----------\n        path : str\n            Can be a local file or a remote URL.\n\n        Returns\n        -------\n        out : bool\n            True if `path` exists.\n\n        Notes\n        -----\n        When `path` is an URL, `exists` will return True if it's either\n        stored locally in the `DataSource` directory, or is a valid remote\n        URL.  `DataSource` does not discriminate between the two, the file\n        is accessible if it exists in either location.\n\n        ")
        
        
        
        # Obtaining the type of the subscript
        int_131896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 28), 'int')
        # Getting the type of 'sys' (line 434)
        sys_131897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 11), 'sys')
        # Obtaining the member 'version_info' of a type (line 434)
        version_info_131898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 11), sys_131897, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 434)
        getitem___131899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 11), version_info_131898, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
        subscript_call_result_131900 = invoke(stypy.reporting.localization.Localization(__file__, 434, 11), getitem___131899, int_131896)
        
        int_131901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 34), 'int')
        # Applying the binary operator '>=' (line 434)
        result_ge_131902 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 11), '>=', subscript_call_result_131900, int_131901)
        
        # Testing the type of an if condition (line 434)
        if_condition_131903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 8), result_ge_131902)
        # Assigning a type to the variable 'if_condition_131903' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'if_condition_131903', if_condition_131903)
        # SSA begins for if statement (line 434)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 435, 12))
        
        # 'from urllib.request import urlopen' statement (line 435)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
        import_131904 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 435, 12), 'urllib.request')

        if (type(import_131904) is not StypyTypeError):

            if (import_131904 != 'pyd_module'):
                __import__(import_131904)
                sys_modules_131905 = sys.modules[import_131904]
                import_from_module(stypy.reporting.localization.Localization(__file__, 435, 12), 'urllib.request', sys_modules_131905.module_type_store, module_type_store, ['urlopen'])
                nest_module(stypy.reporting.localization.Localization(__file__, 435, 12), __file__, sys_modules_131905, sys_modules_131905.module_type_store, module_type_store)
            else:
                from urllib.request import urlopen

                import_from_module(stypy.reporting.localization.Localization(__file__, 435, 12), 'urllib.request', None, module_type_store, ['urlopen'], [urlopen])

        else:
            # Assigning a type to the variable 'urllib.request' (line 435)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'urllib.request', import_131904)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 436, 12))
        
        # 'from urllib.error import URLError' statement (line 436)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
        import_131906 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 436, 12), 'urllib.error')

        if (type(import_131906) is not StypyTypeError):

            if (import_131906 != 'pyd_module'):
                __import__(import_131906)
                sys_modules_131907 = sys.modules[import_131906]
                import_from_module(stypy.reporting.localization.Localization(__file__, 436, 12), 'urllib.error', sys_modules_131907.module_type_store, module_type_store, ['URLError'])
                nest_module(stypy.reporting.localization.Localization(__file__, 436, 12), __file__, sys_modules_131907, sys_modules_131907.module_type_store, module_type_store)
            else:
                from urllib.error import URLError

                import_from_module(stypy.reporting.localization.Localization(__file__, 436, 12), 'urllib.error', None, module_type_store, ['URLError'], [URLError])

        else:
            # Assigning a type to the variable 'urllib.error' (line 436)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'urllib.error', import_131906)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
        
        # SSA branch for the else part of an if statement (line 434)
        module_type_store.open_ssa_branch('else')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 438, 12))
        
        # 'from urllib2 import urlopen' statement (line 438)
        from urllib2 import urlopen

        import_from_module(stypy.reporting.localization.Localization(__file__, 438, 12), 'urllib2', None, module_type_store, ['urlopen'], [urlopen])
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 439, 12))
        
        # 'from urllib2 import URLError' statement (line 439)
        from urllib2 import URLError

        import_from_module(stypy.reporting.localization.Localization(__file__, 439, 12), 'urllib2', None, module_type_store, ['URLError'], [URLError])
        
        # SSA join for if statement (line 434)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to exists(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'path' (line 442)
        path_131911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 26), 'path', False)
        # Processing the call keyword arguments (line 442)
        kwargs_131912 = {}
        # Getting the type of 'os' (line 442)
        os_131908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 442)
        path_131909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 11), os_131908, 'path')
        # Obtaining the member 'exists' of a type (line 442)
        exists_131910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 11), path_131909, 'exists')
        # Calling exists(args, kwargs) (line 442)
        exists_call_result_131913 = invoke(stypy.reporting.localization.Localization(__file__, 442, 11), exists_131910, *[path_131911], **kwargs_131912)
        
        # Testing the type of an if condition (line 442)
        if_condition_131914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 8), exists_call_result_131913)
        # Assigning a type to the variable 'if_condition_131914' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'if_condition_131914', if_condition_131914)
        # SSA begins for if statement (line 442)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 443)
        True_131915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'stypy_return_type', True_131915)
        # SSA join for if statement (line 442)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 446):
        
        # Assigning a Call to a Name (line 446):
        
        # Call to abspath(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'path' (line 446)
        path_131918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 29), 'path', False)
        # Processing the call keyword arguments (line 446)
        kwargs_131919 = {}
        # Getting the type of 'self' (line 446)
        self_131916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'self', False)
        # Obtaining the member 'abspath' of a type (line 446)
        abspath_131917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 16), self_131916, 'abspath')
        # Calling abspath(args, kwargs) (line 446)
        abspath_call_result_131920 = invoke(stypy.reporting.localization.Localization(__file__, 446, 16), abspath_131917, *[path_131918], **kwargs_131919)
        
        # Assigning a type to the variable 'upath' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'upath', abspath_call_result_131920)
        
        
        # Call to exists(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'upath' (line 447)
        upath_131924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 26), 'upath', False)
        # Processing the call keyword arguments (line 447)
        kwargs_131925 = {}
        # Getting the type of 'os' (line 447)
        os_131921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 447)
        path_131922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 11), os_131921, 'path')
        # Obtaining the member 'exists' of a type (line 447)
        exists_131923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 11), path_131922, 'exists')
        # Calling exists(args, kwargs) (line 447)
        exists_call_result_131926 = invoke(stypy.reporting.localization.Localization(__file__, 447, 11), exists_131923, *[upath_131924], **kwargs_131925)
        
        # Testing the type of an if condition (line 447)
        if_condition_131927 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 8), exists_call_result_131926)
        # Assigning a type to the variable 'if_condition_131927' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'if_condition_131927', if_condition_131927)
        # SSA begins for if statement (line 447)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 448)
        True_131928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'stypy_return_type', True_131928)
        # SSA join for if statement (line 447)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to _isurl(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'path' (line 451)
        path_131931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'path', False)
        # Processing the call keyword arguments (line 451)
        kwargs_131932 = {}
        # Getting the type of 'self' (line 451)
        self_131929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 11), 'self', False)
        # Obtaining the member '_isurl' of a type (line 451)
        _isurl_131930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 11), self_131929, '_isurl')
        # Calling _isurl(args, kwargs) (line 451)
        _isurl_call_result_131933 = invoke(stypy.reporting.localization.Localization(__file__, 451, 11), _isurl_131930, *[path_131931], **kwargs_131932)
        
        # Testing the type of an if condition (line 451)
        if_condition_131934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 8), _isurl_call_result_131933)
        # Assigning a type to the variable 'if_condition_131934' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'if_condition_131934', if_condition_131934)
        # SSA begins for if statement (line 451)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 452)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 453):
        
        # Assigning a Call to a Name (line 453):
        
        # Call to urlopen(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'path' (line 453)
        path_131936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'path', False)
        # Processing the call keyword arguments (line 453)
        kwargs_131937 = {}
        # Getting the type of 'urlopen' (line 453)
        urlopen_131935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 26), 'urlopen', False)
        # Calling urlopen(args, kwargs) (line 453)
        urlopen_call_result_131938 = invoke(stypy.reporting.localization.Localization(__file__, 453, 26), urlopen_131935, *[path_131936], **kwargs_131937)
        
        # Assigning a type to the variable 'netfile' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 16), 'netfile', urlopen_call_result_131938)
        
        # Call to close(...): (line 454)
        # Processing the call keyword arguments (line 454)
        kwargs_131941 = {}
        # Getting the type of 'netfile' (line 454)
        netfile_131939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 16), 'netfile', False)
        # Obtaining the member 'close' of a type (line 454)
        close_131940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 16), netfile_131939, 'close')
        # Calling close(args, kwargs) (line 454)
        close_call_result_131942 = invoke(stypy.reporting.localization.Localization(__file__, 454, 16), close_131940, *[], **kwargs_131941)
        
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 455, 16), module_type_store, 'netfile')
        # Getting the type of 'True' (line 456)
        True_131943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'stypy_return_type', True_131943)
        # SSA branch for the except part of a try statement (line 452)
        # SSA branch for the except 'URLError' branch of a try statement (line 452)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 458)
        False_131944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'stypy_return_type', False_131944)
        # SSA join for try-except statement (line 452)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 451)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 459)
        False_131945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'stypy_return_type', False_131945)
        
        # ################# End of 'exists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'exists' in the type store
        # Getting the type of 'stypy_return_type' (line 402)
        stypy_return_type_131946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131946)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'exists'
        return stypy_return_type_131946


    @norecursion
    def open(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_131947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 30), 'str', 'r')
        defaults = [str_131947]
        # Create a new context for function 'open'
        module_type_store = module_type_store.open_function_context('open', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DataSource.open.__dict__.__setitem__('stypy_localization', localization)
        DataSource.open.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DataSource.open.__dict__.__setitem__('stypy_type_store', module_type_store)
        DataSource.open.__dict__.__setitem__('stypy_function_name', 'DataSource.open')
        DataSource.open.__dict__.__setitem__('stypy_param_names_list', ['path', 'mode'])
        DataSource.open.__dict__.__setitem__('stypy_varargs_param_name', None)
        DataSource.open.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DataSource.open.__dict__.__setitem__('stypy_call_defaults', defaults)
        DataSource.open.__dict__.__setitem__('stypy_call_varargs', varargs)
        DataSource.open.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DataSource.open.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DataSource.open', ['path', 'mode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'open', localization, ['path', 'mode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'open(...)' code ##################

        str_131948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, (-1)), 'str', "\n        Open and return file-like object.\n\n        If `path` is an URL, it will be downloaded, stored in the\n        `DataSource` directory and opened from there.\n\n        Parameters\n        ----------\n        path : str\n            Local file path or URL to open.\n        mode : {'r', 'w', 'a'}, optional\n            Mode to open `path`.  Mode 'r' for reading, 'w' for writing,\n            'a' to append. Available modes depend on the type of object\n            specified by `path`. Default is 'r'.\n\n        Returns\n        -------\n        out : file object\n            File object.\n\n        ")
        
        
        # Evaluating a boolean operation
        
        # Call to _isurl(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'path' (line 490)
        path_131951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'path', False)
        # Processing the call keyword arguments (line 490)
        kwargs_131952 = {}
        # Getting the type of 'self' (line 490)
        self_131949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 11), 'self', False)
        # Obtaining the member '_isurl' of a type (line 490)
        _isurl_131950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 11), self_131949, '_isurl')
        # Calling _isurl(args, kwargs) (line 490)
        _isurl_call_result_131953 = invoke(stypy.reporting.localization.Localization(__file__, 490, 11), _isurl_131950, *[path_131951], **kwargs_131952)
        
        
        # Call to _iswritemode(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'mode' (line 490)
        mode_131956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 51), 'mode', False)
        # Processing the call keyword arguments (line 490)
        kwargs_131957 = {}
        # Getting the type of 'self' (line 490)
        self_131954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 33), 'self', False)
        # Obtaining the member '_iswritemode' of a type (line 490)
        _iswritemode_131955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 33), self_131954, '_iswritemode')
        # Calling _iswritemode(args, kwargs) (line 490)
        _iswritemode_call_result_131958 = invoke(stypy.reporting.localization.Localization(__file__, 490, 33), _iswritemode_131955, *[mode_131956], **kwargs_131957)
        
        # Applying the binary operator 'and' (line 490)
        result_and_keyword_131959 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), 'and', _isurl_call_result_131953, _iswritemode_call_result_131958)
        
        # Testing the type of an if condition (line 490)
        if_condition_131960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 8), result_and_keyword_131959)
        # Assigning a type to the variable 'if_condition_131960' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'if_condition_131960', if_condition_131960)
        # SSA begins for if statement (line 490)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 491)
        # Processing the call arguments (line 491)
        str_131962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 29), 'str', 'URLs are not writeable')
        # Processing the call keyword arguments (line 491)
        kwargs_131963 = {}
        # Getting the type of 'ValueError' (line 491)
        ValueError_131961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 491)
        ValueError_call_result_131964 = invoke(stypy.reporting.localization.Localization(__file__, 491, 18), ValueError_131961, *[str_131962], **kwargs_131963)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 491, 12), ValueError_call_result_131964, 'raise parameter', BaseException)
        # SSA join for if statement (line 490)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 494):
        
        # Assigning a Call to a Name (line 494):
        
        # Call to _findfile(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 'path' (line 494)
        path_131967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 31), 'path', False)
        # Processing the call keyword arguments (line 494)
        kwargs_131968 = {}
        # Getting the type of 'self' (line 494)
        self_131965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 16), 'self', False)
        # Obtaining the member '_findfile' of a type (line 494)
        _findfile_131966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 16), self_131965, '_findfile')
        # Calling _findfile(args, kwargs) (line 494)
        _findfile_call_result_131969 = invoke(stypy.reporting.localization.Localization(__file__, 494, 16), _findfile_131966, *[path_131967], **kwargs_131968)
        
        # Assigning a type to the variable 'found' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'found', _findfile_call_result_131969)
        
        # Getting the type of 'found' (line 495)
        found_131970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 11), 'found')
        # Testing the type of an if condition (line 495)
        if_condition_131971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 495, 8), found_131970)
        # Assigning a type to the variable 'if_condition_131971' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'if_condition_131971', if_condition_131971)
        # SSA begins for if statement (line 495)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 496):
        
        # Assigning a Call to a Name:
        
        # Call to _splitzipext(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'found' (line 496)
        found_131974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 44), 'found', False)
        # Processing the call keyword arguments (line 496)
        kwargs_131975 = {}
        # Getting the type of 'self' (line 496)
        self_131972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 26), 'self', False)
        # Obtaining the member '_splitzipext' of a type (line 496)
        _splitzipext_131973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 26), self_131972, '_splitzipext')
        # Calling _splitzipext(args, kwargs) (line 496)
        _splitzipext_call_result_131976 = invoke(stypy.reporting.localization.Localization(__file__, 496, 26), _splitzipext_131973, *[found_131974], **kwargs_131975)
        
        # Assigning a type to the variable 'call_assignment_131366' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'call_assignment_131366', _splitzipext_call_result_131976)
        
        # Assigning a Call to a Name (line 496):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 12), 'int')
        # Processing the call keyword arguments
        kwargs_131980 = {}
        # Getting the type of 'call_assignment_131366' (line 496)
        call_assignment_131366_131977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'call_assignment_131366', False)
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___131978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 12), call_assignment_131366_131977, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131981 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131978, *[int_131979], **kwargs_131980)
        
        # Assigning a type to the variable 'call_assignment_131367' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'call_assignment_131367', getitem___call_result_131981)
        
        # Assigning a Name to a Name (line 496):
        # Getting the type of 'call_assignment_131367' (line 496)
        call_assignment_131367_131982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'call_assignment_131367')
        # Assigning a type to the variable '_fname' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), '_fname', call_assignment_131367_131982)
        
        # Assigning a Call to a Name (line 496):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_131985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 12), 'int')
        # Processing the call keyword arguments
        kwargs_131986 = {}
        # Getting the type of 'call_assignment_131366' (line 496)
        call_assignment_131366_131983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'call_assignment_131366', False)
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___131984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 12), call_assignment_131366_131983, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_131987 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___131984, *[int_131985], **kwargs_131986)
        
        # Assigning a type to the variable 'call_assignment_131368' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'call_assignment_131368', getitem___call_result_131987)
        
        # Assigning a Name to a Name (line 496):
        # Getting the type of 'call_assignment_131368' (line 496)
        call_assignment_131368_131988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'call_assignment_131368')
        # Assigning a type to the variable 'ext' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 20), 'ext', call_assignment_131368_131988)
        
        
        # Getting the type of 'ext' (line 497)
        ext_131989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 15), 'ext')
        str_131990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 22), 'str', 'bz2')
        # Applying the binary operator '==' (line 497)
        result_eq_131991 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 15), '==', ext_131989, str_131990)
        
        # Testing the type of an if condition (line 497)
        if_condition_131992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 12), result_eq_131991)
        # Assigning a type to the variable 'if_condition_131992' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'if_condition_131992', if_condition_131992)
        # SSA begins for if statement (line 497)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to replace(...): (line 498)
        # Processing the call arguments (line 498)
        str_131995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 29), 'str', '+')
        str_131996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 34), 'str', '')
        # Processing the call keyword arguments (line 498)
        kwargs_131997 = {}
        # Getting the type of 'mode' (line 498)
        mode_131993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'mode', False)
        # Obtaining the member 'replace' of a type (line 498)
        replace_131994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 16), mode_131993, 'replace')
        # Calling replace(args, kwargs) (line 498)
        replace_call_result_131998 = invoke(stypy.reporting.localization.Localization(__file__, 498, 16), replace_131994, *[str_131995, str_131996], **kwargs_131997)
        
        # SSA join for if statement (line 497)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to (...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 'found' (line 499)
        found_132003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 38), 'found', False)
        # Processing the call keyword arguments (line 499)
        # Getting the type of 'mode' (line 499)
        mode_132004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 50), 'mode', False)
        keyword_132005 = mode_132004
        kwargs_132006 = {'mode': keyword_132005}
        
        # Obtaining the type of the subscript
        # Getting the type of 'ext' (line 499)
        ext_131999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 33), 'ext', False)
        # Getting the type of '_file_openers' (line 499)
        _file_openers_132000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 19), '_file_openers', False)
        # Obtaining the member '__getitem__' of a type (line 499)
        getitem___132001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 19), _file_openers_132000, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 499)
        subscript_call_result_132002 = invoke(stypy.reporting.localization.Localization(__file__, 499, 19), getitem___132001, ext_131999)
        
        # Calling (args, kwargs) (line 499)
        _call_result_132007 = invoke(stypy.reporting.localization.Localization(__file__, 499, 19), subscript_call_result_132002, *[found_132003], **kwargs_132006)
        
        # Assigning a type to the variable 'stypy_return_type' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 12), 'stypy_return_type', _call_result_132007)
        # SSA branch for the else part of an if statement (line 495)
        module_type_store.open_ssa_branch('else')
        
        # Call to IOError(...): (line 501)
        # Processing the call arguments (line 501)
        str_132009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 26), 'str', '%s not found.')
        # Getting the type of 'path' (line 501)
        path_132010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 44), 'path', False)
        # Applying the binary operator '%' (line 501)
        result_mod_132011 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 26), '%', str_132009, path_132010)
        
        # Processing the call keyword arguments (line 501)
        kwargs_132012 = {}
        # Getting the type of 'IOError' (line 501)
        IOError_132008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 18), 'IOError', False)
        # Calling IOError(args, kwargs) (line 501)
        IOError_call_result_132013 = invoke(stypy.reporting.localization.Localization(__file__, 501, 18), IOError_132008, *[result_mod_132011], **kwargs_132012)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 501, 12), IOError_call_result_132013, 'raise parameter', BaseException)
        # SSA join for if statement (line 495)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'open(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'open' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_132014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132014)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'open'
        return stypy_return_type_132014


# Assigning a type to the variable 'DataSource' (line 154)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'DataSource', DataSource)
# Declaration of the 'Repository' class
# Getting the type of 'DataSource' (line 504)
DataSource_132015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 18), 'DataSource')

class Repository(DataSource_132015, ):
    str_132016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, (-1)), 'str', "\n    Repository(baseurl, destpath='.')\n\n    A data repository where multiple DataSource's share a base\n    URL/directory.\n\n    `Repository` extends `DataSource` by prepending a base URL (or\n    directory) to all the files it handles. Use `Repository` when you will\n    be working with multiple files from one base URL.  Initialize\n    `Repository` with the base URL, then refer to each file by its filename\n    only.\n\n    Parameters\n    ----------\n    baseurl : str\n        Path to the local directory or remote location that contains the\n        data files.\n    destpath : str or None, optional\n        Path to the directory where the source file gets downloaded to for\n        use.  If `destpath` is None, a temporary directory will be created.\n        The default path is the current directory.\n\n    Examples\n    --------\n    To analyze all files in the repository, do something like this\n    (note: this is not self-contained code)::\n\n        >>> repos = np.lib._datasource.Repository('/home/user/data/dir/')\n        >>> for filename in filelist:\n        ...     fp = repos.open(filename)\n        ...     fp.analyze()\n        ...     fp.close()\n\n    Similarly you could use a URL for a repository::\n\n        >>> repos = np.lib._datasource.Repository('http://www.xyz.edu/data')\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'os' (line 544)
        os_132017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 41), 'os')
        # Obtaining the member 'curdir' of a type (line 544)
        curdir_132018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 41), os_132017, 'curdir')
        defaults = [curdir_132018]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 544, 4, False)
        # Assigning a type to the variable 'self' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Repository.__init__', ['baseurl', 'destpath'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['baseurl', 'destpath'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_132019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 8), 'str', 'Create a Repository with a shared url or directory of baseurl.')
        
        # Call to __init__(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'self' (line 546)
        self_132022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 28), 'self', False)
        # Processing the call keyword arguments (line 546)
        # Getting the type of 'destpath' (line 546)
        destpath_132023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 43), 'destpath', False)
        keyword_132024 = destpath_132023
        kwargs_132025 = {'destpath': keyword_132024}
        # Getting the type of 'DataSource' (line 546)
        DataSource_132020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'DataSource', False)
        # Obtaining the member '__init__' of a type (line 546)
        init___132021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 8), DataSource_132020, '__init__')
        # Calling __init__(args, kwargs) (line 546)
        init___call_result_132026 = invoke(stypy.reporting.localization.Localization(__file__, 546, 8), init___132021, *[self_132022], **kwargs_132025)
        
        
        # Assigning a Name to a Attribute (line 547):
        
        # Assigning a Name to a Attribute (line 547):
        # Getting the type of 'baseurl' (line 547)
        baseurl_132027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 24), 'baseurl')
        # Getting the type of 'self' (line 547)
        self_132028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'self')
        # Setting the type of the member '_baseurl' of a type (line 547)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 8), self_132028, '_baseurl', baseurl_132027)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __del__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__del__'
        module_type_store = module_type_store.open_function_context('__del__', 549, 4, False)
        # Assigning a type to the variable 'self' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Repository.__del__.__dict__.__setitem__('stypy_localization', localization)
        Repository.__del__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Repository.__del__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Repository.__del__.__dict__.__setitem__('stypy_function_name', 'Repository.__del__')
        Repository.__del__.__dict__.__setitem__('stypy_param_names_list', [])
        Repository.__del__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Repository.__del__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Repository.__del__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Repository.__del__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Repository.__del__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Repository.__del__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Repository.__del__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__del__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__del__(...)' code ##################

        
        # Call to __del__(...): (line 550)
        # Processing the call arguments (line 550)
        # Getting the type of 'self' (line 550)
        self_132031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 27), 'self', False)
        # Processing the call keyword arguments (line 550)
        kwargs_132032 = {}
        # Getting the type of 'DataSource' (line 550)
        DataSource_132029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'DataSource', False)
        # Obtaining the member '__del__' of a type (line 550)
        del___132030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), DataSource_132029, '__del__')
        # Calling __del__(args, kwargs) (line 550)
        del___call_result_132033 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), del___132030, *[self_132031], **kwargs_132032)
        
        
        # ################# End of '__del__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__del__' in the type store
        # Getting the type of 'stypy_return_type' (line 549)
        stypy_return_type_132034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132034)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__del__'
        return stypy_return_type_132034


    @norecursion
    def _fullpath(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fullpath'
        module_type_store = module_type_store.open_function_context('_fullpath', 552, 4, False)
        # Assigning a type to the variable 'self' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Repository._fullpath.__dict__.__setitem__('stypy_localization', localization)
        Repository._fullpath.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Repository._fullpath.__dict__.__setitem__('stypy_type_store', module_type_store)
        Repository._fullpath.__dict__.__setitem__('stypy_function_name', 'Repository._fullpath')
        Repository._fullpath.__dict__.__setitem__('stypy_param_names_list', ['path'])
        Repository._fullpath.__dict__.__setitem__('stypy_varargs_param_name', None)
        Repository._fullpath.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Repository._fullpath.__dict__.__setitem__('stypy_call_defaults', defaults)
        Repository._fullpath.__dict__.__setitem__('stypy_call_varargs', varargs)
        Repository._fullpath.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Repository._fullpath.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Repository._fullpath', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fullpath', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fullpath(...)' code ##################

        str_132035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 8), 'str', 'Return complete path for path.  Prepends baseurl if necessary.')
        
        # Assigning a Call to a Name (line 554):
        
        # Assigning a Call to a Name (line 554):
        
        # Call to split(...): (line 554)
        # Processing the call arguments (line 554)
        # Getting the type of 'self' (line 554)
        self_132038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 31), 'self', False)
        # Obtaining the member '_baseurl' of a type (line 554)
        _baseurl_132039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 31), self_132038, '_baseurl')
        int_132040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 46), 'int')
        # Processing the call keyword arguments (line 554)
        kwargs_132041 = {}
        # Getting the type of 'path' (line 554)
        path_132036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'path', False)
        # Obtaining the member 'split' of a type (line 554)
        split_132037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 20), path_132036, 'split')
        # Calling split(args, kwargs) (line 554)
        split_call_result_132042 = invoke(stypy.reporting.localization.Localization(__file__, 554, 20), split_132037, *[_baseurl_132039, int_132040], **kwargs_132041)
        
        # Assigning a type to the variable 'splitpath' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'splitpath', split_call_result_132042)
        
        
        
        # Call to len(...): (line 555)
        # Processing the call arguments (line 555)
        # Getting the type of 'splitpath' (line 555)
        splitpath_132044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 15), 'splitpath', False)
        # Processing the call keyword arguments (line 555)
        kwargs_132045 = {}
        # Getting the type of 'len' (line 555)
        len_132043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 11), 'len', False)
        # Calling len(args, kwargs) (line 555)
        len_call_result_132046 = invoke(stypy.reporting.localization.Localization(__file__, 555, 11), len_132043, *[splitpath_132044], **kwargs_132045)
        
        int_132047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 29), 'int')
        # Applying the binary operator '==' (line 555)
        result_eq_132048 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 11), '==', len_call_result_132046, int_132047)
        
        # Testing the type of an if condition (line 555)
        if_condition_132049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 8), result_eq_132048)
        # Assigning a type to the variable 'if_condition_132049' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'if_condition_132049', if_condition_132049)
        # SSA begins for if statement (line 555)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 556):
        
        # Assigning a Call to a Name (line 556):
        
        # Call to join(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'self' (line 556)
        self_132053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 34), 'self', False)
        # Obtaining the member '_baseurl' of a type (line 556)
        _baseurl_132054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 34), self_132053, '_baseurl')
        # Getting the type of 'path' (line 556)
        path_132055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 49), 'path', False)
        # Processing the call keyword arguments (line 556)
        kwargs_132056 = {}
        # Getting the type of 'os' (line 556)
        os_132050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 556)
        path_132051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 21), os_132050, 'path')
        # Obtaining the member 'join' of a type (line 556)
        join_132052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 21), path_132051, 'join')
        # Calling join(args, kwargs) (line 556)
        join_call_result_132057 = invoke(stypy.reporting.localization.Localization(__file__, 556, 21), join_132052, *[_baseurl_132054, path_132055], **kwargs_132056)
        
        # Assigning a type to the variable 'result' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'result', join_call_result_132057)
        # SSA branch for the else part of an if statement (line 555)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 558):
        
        # Assigning a Name to a Name (line 558):
        # Getting the type of 'path' (line 558)
        path_132058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 21), 'path')
        # Assigning a type to the variable 'result' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'result', path_132058)
        # SSA join for if statement (line 555)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 559)
        result_132059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'stypy_return_type', result_132059)
        
        # ################# End of '_fullpath(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fullpath' in the type store
        # Getting the type of 'stypy_return_type' (line 552)
        stypy_return_type_132060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132060)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fullpath'
        return stypy_return_type_132060


    @norecursion
    def _findfile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_findfile'
        module_type_store = module_type_store.open_function_context('_findfile', 561, 4, False)
        # Assigning a type to the variable 'self' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Repository._findfile.__dict__.__setitem__('stypy_localization', localization)
        Repository._findfile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Repository._findfile.__dict__.__setitem__('stypy_type_store', module_type_store)
        Repository._findfile.__dict__.__setitem__('stypy_function_name', 'Repository._findfile')
        Repository._findfile.__dict__.__setitem__('stypy_param_names_list', ['path'])
        Repository._findfile.__dict__.__setitem__('stypy_varargs_param_name', None)
        Repository._findfile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Repository._findfile.__dict__.__setitem__('stypy_call_defaults', defaults)
        Repository._findfile.__dict__.__setitem__('stypy_call_varargs', varargs)
        Repository._findfile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Repository._findfile.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Repository._findfile', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_findfile', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_findfile(...)' code ##################

        str_132061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 8), 'str', 'Extend DataSource method to prepend baseurl to ``path``.')
        
        # Call to _findfile(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'self' (line 563)
        self_132064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 36), 'self', False)
        
        # Call to _fullpath(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'path' (line 563)
        path_132067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 57), 'path', False)
        # Processing the call keyword arguments (line 563)
        kwargs_132068 = {}
        # Getting the type of 'self' (line 563)
        self_132065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 42), 'self', False)
        # Obtaining the member '_fullpath' of a type (line 563)
        _fullpath_132066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 42), self_132065, '_fullpath')
        # Calling _fullpath(args, kwargs) (line 563)
        _fullpath_call_result_132069 = invoke(stypy.reporting.localization.Localization(__file__, 563, 42), _fullpath_132066, *[path_132067], **kwargs_132068)
        
        # Processing the call keyword arguments (line 563)
        kwargs_132070 = {}
        # Getting the type of 'DataSource' (line 563)
        DataSource_132062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 15), 'DataSource', False)
        # Obtaining the member '_findfile' of a type (line 563)
        _findfile_132063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 15), DataSource_132062, '_findfile')
        # Calling _findfile(args, kwargs) (line 563)
        _findfile_call_result_132071 = invoke(stypy.reporting.localization.Localization(__file__, 563, 15), _findfile_132063, *[self_132064, _fullpath_call_result_132069], **kwargs_132070)
        
        # Assigning a type to the variable 'stypy_return_type' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'stypy_return_type', _findfile_call_result_132071)
        
        # ################# End of '_findfile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_findfile' in the type store
        # Getting the type of 'stypy_return_type' (line 561)
        stypy_return_type_132072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132072)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_findfile'
        return stypy_return_type_132072


    @norecursion
    def abspath(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'abspath'
        module_type_store = module_type_store.open_function_context('abspath', 565, 4, False)
        # Assigning a type to the variable 'self' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Repository.abspath.__dict__.__setitem__('stypy_localization', localization)
        Repository.abspath.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Repository.abspath.__dict__.__setitem__('stypy_type_store', module_type_store)
        Repository.abspath.__dict__.__setitem__('stypy_function_name', 'Repository.abspath')
        Repository.abspath.__dict__.__setitem__('stypy_param_names_list', ['path'])
        Repository.abspath.__dict__.__setitem__('stypy_varargs_param_name', None)
        Repository.abspath.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Repository.abspath.__dict__.__setitem__('stypy_call_defaults', defaults)
        Repository.abspath.__dict__.__setitem__('stypy_call_varargs', varargs)
        Repository.abspath.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Repository.abspath.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Repository.abspath', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'abspath', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'abspath(...)' code ##################

        str_132073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, (-1)), 'str', '\n        Return absolute path of file in the Repository directory.\n\n        If `path` is an URL, then `abspath` will return either the location\n        the file exists locally or the location it would exist when opened\n        using the `open` method.\n\n        Parameters\n        ----------\n        path : str\n            Can be a local file or a remote URL. This may, but does not\n            have to, include the `baseurl` with which the `Repository` was\n            initialized.\n\n        Returns\n        -------\n        out : str\n            Complete path, including the `DataSource` destination directory.\n\n        ')
        
        # Call to abspath(...): (line 586)
        # Processing the call arguments (line 586)
        # Getting the type of 'self' (line 586)
        self_132076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 34), 'self', False)
        
        # Call to _fullpath(...): (line 586)
        # Processing the call arguments (line 586)
        # Getting the type of 'path' (line 586)
        path_132079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 55), 'path', False)
        # Processing the call keyword arguments (line 586)
        kwargs_132080 = {}
        # Getting the type of 'self' (line 586)
        self_132077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 40), 'self', False)
        # Obtaining the member '_fullpath' of a type (line 586)
        _fullpath_132078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 40), self_132077, '_fullpath')
        # Calling _fullpath(args, kwargs) (line 586)
        _fullpath_call_result_132081 = invoke(stypy.reporting.localization.Localization(__file__, 586, 40), _fullpath_132078, *[path_132079], **kwargs_132080)
        
        # Processing the call keyword arguments (line 586)
        kwargs_132082 = {}
        # Getting the type of 'DataSource' (line 586)
        DataSource_132074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 15), 'DataSource', False)
        # Obtaining the member 'abspath' of a type (line 586)
        abspath_132075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 15), DataSource_132074, 'abspath')
        # Calling abspath(args, kwargs) (line 586)
        abspath_call_result_132083 = invoke(stypy.reporting.localization.Localization(__file__, 586, 15), abspath_132075, *[self_132076, _fullpath_call_result_132081], **kwargs_132082)
        
        # Assigning a type to the variable 'stypy_return_type' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'stypy_return_type', abspath_call_result_132083)
        
        # ################# End of 'abspath(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'abspath' in the type store
        # Getting the type of 'stypy_return_type' (line 565)
        stypy_return_type_132084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'abspath'
        return stypy_return_type_132084


    @norecursion
    def exists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'exists'
        module_type_store = module_type_store.open_function_context('exists', 588, 4, False)
        # Assigning a type to the variable 'self' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Repository.exists.__dict__.__setitem__('stypy_localization', localization)
        Repository.exists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Repository.exists.__dict__.__setitem__('stypy_type_store', module_type_store)
        Repository.exists.__dict__.__setitem__('stypy_function_name', 'Repository.exists')
        Repository.exists.__dict__.__setitem__('stypy_param_names_list', ['path'])
        Repository.exists.__dict__.__setitem__('stypy_varargs_param_name', None)
        Repository.exists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Repository.exists.__dict__.__setitem__('stypy_call_defaults', defaults)
        Repository.exists.__dict__.__setitem__('stypy_call_varargs', varargs)
        Repository.exists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Repository.exists.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Repository.exists', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'exists', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'exists(...)' code ##################

        str_132085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, (-1)), 'str', "\n        Test if path exists prepending Repository base URL to path.\n\n        Test if `path` exists as (and in this order):\n\n        - a local file.\n        - a remote URL that has been downloaded and stored locally in the\n          `DataSource` directory.\n        - a remote URL that has not been downloaded, but is valid and\n          accessible.\n\n        Parameters\n        ----------\n        path : str\n            Can be a local file or a remote URL. This may, but does not\n            have to, include the `baseurl` with which the `Repository` was\n            initialized.\n\n        Returns\n        -------\n        out : bool\n            True if `path` exists.\n\n        Notes\n        -----\n        When `path` is an URL, `exists` will return True if it's either\n        stored locally in the `DataSource` directory, or is a valid remote\n        URL.  `DataSource` does not discriminate between the two, the file\n        is accessible if it exists in either location.\n\n        ")
        
        # Call to exists(...): (line 620)
        # Processing the call arguments (line 620)
        # Getting the type of 'self' (line 620)
        self_132088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 33), 'self', False)
        
        # Call to _fullpath(...): (line 620)
        # Processing the call arguments (line 620)
        # Getting the type of 'path' (line 620)
        path_132091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 54), 'path', False)
        # Processing the call keyword arguments (line 620)
        kwargs_132092 = {}
        # Getting the type of 'self' (line 620)
        self_132089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 39), 'self', False)
        # Obtaining the member '_fullpath' of a type (line 620)
        _fullpath_132090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 39), self_132089, '_fullpath')
        # Calling _fullpath(args, kwargs) (line 620)
        _fullpath_call_result_132093 = invoke(stypy.reporting.localization.Localization(__file__, 620, 39), _fullpath_132090, *[path_132091], **kwargs_132092)
        
        # Processing the call keyword arguments (line 620)
        kwargs_132094 = {}
        # Getting the type of 'DataSource' (line 620)
        DataSource_132086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'DataSource', False)
        # Obtaining the member 'exists' of a type (line 620)
        exists_132087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 15), DataSource_132086, 'exists')
        # Calling exists(args, kwargs) (line 620)
        exists_call_result_132095 = invoke(stypy.reporting.localization.Localization(__file__, 620, 15), exists_132087, *[self_132088, _fullpath_call_result_132093], **kwargs_132094)
        
        # Assigning a type to the variable 'stypy_return_type' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'stypy_return_type', exists_call_result_132095)
        
        # ################# End of 'exists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'exists' in the type store
        # Getting the type of 'stypy_return_type' (line 588)
        stypy_return_type_132096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132096)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'exists'
        return stypy_return_type_132096


    @norecursion
    def open(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_132097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 30), 'str', 'r')
        defaults = [str_132097]
        # Create a new context for function 'open'
        module_type_store = module_type_store.open_function_context('open', 622, 4, False)
        # Assigning a type to the variable 'self' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Repository.open.__dict__.__setitem__('stypy_localization', localization)
        Repository.open.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Repository.open.__dict__.__setitem__('stypy_type_store', module_type_store)
        Repository.open.__dict__.__setitem__('stypy_function_name', 'Repository.open')
        Repository.open.__dict__.__setitem__('stypy_param_names_list', ['path', 'mode'])
        Repository.open.__dict__.__setitem__('stypy_varargs_param_name', None)
        Repository.open.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Repository.open.__dict__.__setitem__('stypy_call_defaults', defaults)
        Repository.open.__dict__.__setitem__('stypy_call_varargs', varargs)
        Repository.open.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Repository.open.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Repository.open', ['path', 'mode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'open', localization, ['path', 'mode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'open(...)' code ##################

        str_132098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, (-1)), 'str', "\n        Open and return file-like object prepending Repository base URL.\n\n        If `path` is an URL, it will be downloaded, stored in the\n        DataSource directory and opened from there.\n\n        Parameters\n        ----------\n        path : str\n            Local file path or URL to open. This may, but does not have to,\n            include the `baseurl` with which the `Repository` was\n            initialized.\n        mode : {'r', 'w', 'a'}, optional\n            Mode to open `path`.  Mode 'r' for reading, 'w' for writing,\n            'a' to append. Available modes depend on the type of object\n            specified by `path`. Default is 'r'.\n\n        Returns\n        -------\n        out : file object\n            File object.\n\n        ")
        
        # Call to open(...): (line 646)
        # Processing the call arguments (line 646)
        # Getting the type of 'self' (line 646)
        self_132101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 31), 'self', False)
        
        # Call to _fullpath(...): (line 646)
        # Processing the call arguments (line 646)
        # Getting the type of 'path' (line 646)
        path_132104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 52), 'path', False)
        # Processing the call keyword arguments (line 646)
        kwargs_132105 = {}
        # Getting the type of 'self' (line 646)
        self_132102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 37), 'self', False)
        # Obtaining the member '_fullpath' of a type (line 646)
        _fullpath_132103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 37), self_132102, '_fullpath')
        # Calling _fullpath(args, kwargs) (line 646)
        _fullpath_call_result_132106 = invoke(stypy.reporting.localization.Localization(__file__, 646, 37), _fullpath_132103, *[path_132104], **kwargs_132105)
        
        # Getting the type of 'mode' (line 646)
        mode_132107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 59), 'mode', False)
        # Processing the call keyword arguments (line 646)
        kwargs_132108 = {}
        # Getting the type of 'DataSource' (line 646)
        DataSource_132099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 15), 'DataSource', False)
        # Obtaining the member 'open' of a type (line 646)
        open_132100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 15), DataSource_132099, 'open')
        # Calling open(args, kwargs) (line 646)
        open_call_result_132109 = invoke(stypy.reporting.localization.Localization(__file__, 646, 15), open_132100, *[self_132101, _fullpath_call_result_132106, mode_132107], **kwargs_132108)
        
        # Assigning a type to the variable 'stypy_return_type' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'stypy_return_type', open_call_result_132109)
        
        # ################# End of 'open(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'open' in the type store
        # Getting the type of 'stypy_return_type' (line 622)
        stypy_return_type_132110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132110)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'open'
        return stypy_return_type_132110


    @norecursion
    def listdir(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'listdir'
        module_type_store = module_type_store.open_function_context('listdir', 648, 4, False)
        # Assigning a type to the variable 'self' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Repository.listdir.__dict__.__setitem__('stypy_localization', localization)
        Repository.listdir.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Repository.listdir.__dict__.__setitem__('stypy_type_store', module_type_store)
        Repository.listdir.__dict__.__setitem__('stypy_function_name', 'Repository.listdir')
        Repository.listdir.__dict__.__setitem__('stypy_param_names_list', [])
        Repository.listdir.__dict__.__setitem__('stypy_varargs_param_name', None)
        Repository.listdir.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Repository.listdir.__dict__.__setitem__('stypy_call_defaults', defaults)
        Repository.listdir.__dict__.__setitem__('stypy_call_varargs', varargs)
        Repository.listdir.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Repository.listdir.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Repository.listdir', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'listdir', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'listdir(...)' code ##################

        str_132111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, (-1)), 'str', '\n        List files in the source Repository.\n\n        Returns\n        -------\n        files : list of str\n            List of file names (not containing a directory part).\n\n        Notes\n        -----\n        Does not currently work for remote repositories.\n\n        ')
        
        
        # Call to _isurl(...): (line 662)
        # Processing the call arguments (line 662)
        # Getting the type of 'self' (line 662)
        self_132114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 23), 'self', False)
        # Obtaining the member '_baseurl' of a type (line 662)
        _baseurl_132115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 23), self_132114, '_baseurl')
        # Processing the call keyword arguments (line 662)
        kwargs_132116 = {}
        # Getting the type of 'self' (line 662)
        self_132112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 11), 'self', False)
        # Obtaining the member '_isurl' of a type (line 662)
        _isurl_132113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 11), self_132112, '_isurl')
        # Calling _isurl(args, kwargs) (line 662)
        _isurl_call_result_132117 = invoke(stypy.reporting.localization.Localization(__file__, 662, 11), _isurl_132113, *[_baseurl_132115], **kwargs_132116)
        
        # Testing the type of an if condition (line 662)
        if_condition_132118 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 8), _isurl_call_result_132117)
        # Assigning a type to the variable 'if_condition_132118' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'if_condition_132118', if_condition_132118)
        # SSA begins for if statement (line 662)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to NotImplementedError(...): (line 663)
        # Processing the call arguments (line 663)
        str_132120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 18), 'str', 'Directory listing of URLs, not supported yet.')
        # Processing the call keyword arguments (line 663)
        kwargs_132121 = {}
        # Getting the type of 'NotImplementedError' (line 663)
        NotImplementedError_132119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 663)
        NotImplementedError_call_result_132122 = invoke(stypy.reporting.localization.Localization(__file__, 663, 18), NotImplementedError_132119, *[str_132120], **kwargs_132121)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 663, 12), NotImplementedError_call_result_132122, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 662)
        module_type_store.open_ssa_branch('else')
        
        # Call to listdir(...): (line 666)
        # Processing the call arguments (line 666)
        # Getting the type of 'self' (line 666)
        self_132125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 30), 'self', False)
        # Obtaining the member '_baseurl' of a type (line 666)
        _baseurl_132126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 30), self_132125, '_baseurl')
        # Processing the call keyword arguments (line 666)
        kwargs_132127 = {}
        # Getting the type of 'os' (line 666)
        os_132123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 19), 'os', False)
        # Obtaining the member 'listdir' of a type (line 666)
        listdir_132124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 19), os_132123, 'listdir')
        # Calling listdir(args, kwargs) (line 666)
        listdir_call_result_132128 = invoke(stypy.reporting.localization.Localization(__file__, 666, 19), listdir_132124, *[_baseurl_132126], **kwargs_132127)
        
        # Assigning a type to the variable 'stypy_return_type' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'stypy_return_type', listdir_call_result_132128)
        # SSA join for if statement (line 662)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'listdir(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'listdir' in the type store
        # Getting the type of 'stypy_return_type' (line 648)
        stypy_return_type_132129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132129)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'listdir'
        return stypy_return_type_132129


# Assigning a type to the variable 'Repository' (line 504)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 0), 'Repository', Repository)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
