
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Provides a collection of utilities for comparing (image) results.
3: 
4: '''
5: from __future__ import (absolute_import, division, print_function,
6:                         unicode_literals)
7: 
8: import six
9: 
10: import atexit
11: import functools
12: import hashlib
13: import itertools
14: import os
15: import re
16: import shutil
17: import sys
18: from tempfile import TemporaryFile
19: 
20: import numpy as np
21: 
22: import matplotlib
23: from matplotlib.compat import subprocess
24: from matplotlib.testing.exceptions import ImageComparisonFailure
25: from matplotlib import _png
26: from matplotlib import _get_cachedir
27: from matplotlib import cbook
28: 
29: __all__ = ['compare_float', 'compare_images', 'comparable_formats']
30: 
31: 
32: def make_test_filename(fname, purpose):
33:     '''
34:     Make a new filename by inserting `purpose` before the file's
35:     extension.
36:     '''
37:     base, ext = os.path.splitext(fname)
38:     return '%s-%s%s' % (base, purpose, ext)
39: 
40: 
41: def compare_float(expected, actual, relTol=None, absTol=None):
42:     '''
43:     Fail if the floating point values are not close enough, with
44:     the given message.
45: 
46:     You can specify a relative tolerance, absolute tolerance, or both.
47: 
48:     '''
49:     if relTol is None and absTol is None:
50:         raise ValueError("You haven't specified a 'relTol' relative "
51:                          "tolerance or a 'absTol' absolute tolerance "
52:                          "function argument. You must specify one.")
53:     msg = ""
54: 
55:     if absTol is not None:
56:         absDiff = abs(expected - actual)
57:         if absTol < absDiff:
58:             template = ['',
59:                         'Expected: {expected}',
60:                         'Actual:   {actual}',
61:                         'Abs diff: {absDiff}',
62:                         'Abs tol:  {absTol}']
63:             msg += '\n  '.join([line.format(**locals()) for line in template])
64: 
65:     if relTol is not None:
66:         # The relative difference of the two values.  If the expected value is
67:         # zero, then return the absolute value of the difference.
68:         relDiff = abs(expected - actual)
69:         if expected:
70:             relDiff = relDiff / abs(expected)
71: 
72:         if relTol < relDiff:
73:             # The relative difference is a ratio, so it's always unit-less.
74:             template = ['',
75:                         'Expected: {expected}',
76:                         'Actual:   {actual}',
77:                         'Rel diff: {relDiff}',
78:                         'Rel tol:  {relTol}']
79:             msg += '\n  '.join([line.format(**locals()) for line in template])
80: 
81:     return msg or None
82: 
83: 
84: def get_cache_dir():
85:     cachedir = _get_cachedir()
86:     if cachedir is None:
87:         raise RuntimeError('Could not find a suitable configuration directory')
88:     cache_dir = os.path.join(cachedir, 'test_cache')
89:     if not os.path.exists(cache_dir):
90:         try:
91:             cbook.mkdirs(cache_dir)
92:         except IOError:
93:             return None
94:     if not os.access(cache_dir, os.W_OK):
95:         return None
96:     return cache_dir
97: 
98: 
99: def get_file_hash(path, block_size=2 ** 20):
100:     md5 = hashlib.md5()
101:     with open(path, 'rb') as fd:
102:         while True:
103:             data = fd.read(block_size)
104:             if not data:
105:                 break
106:             md5.update(data)
107: 
108:     if path.endswith('.pdf'):
109:         from matplotlib import checkdep_ghostscript
110:         md5.update(checkdep_ghostscript()[1].encode('utf-8'))
111:     elif path.endswith('.svg'):
112:         from matplotlib import checkdep_inkscape
113:         md5.update(checkdep_inkscape().encode('utf-8'))
114: 
115:     return md5.hexdigest()
116: 
117: 
118: def make_external_conversion_command(cmd):
119:     def convert(old, new):
120:         cmdline = cmd(old, new)
121:         pipe = subprocess.Popen(cmdline, universal_newlines=True,
122:                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
123:         stdout, stderr = pipe.communicate()
124:         errcode = pipe.wait()
125:         if not os.path.exists(new) or errcode:
126:             msg = "Conversion command failed:\n%s\n" % ' '.join(cmdline)
127:             if stdout:
128:                 msg += "Standard output:\n%s\n" % stdout
129:             if stderr:
130:                 msg += "Standard error:\n%s\n" % stderr
131:             raise IOError(msg)
132: 
133:     return convert
134: 
135: 
136: # Modified from https://bugs.python.org/issue25567.
137: _find_unsafe_bytes = re.compile(br'[^a-zA-Z0-9_@%+=:,./-]').search
138: 
139: 
140: def _shlex_quote_bytes(b):
141:     return (b if _find_unsafe_bytes(b) is None
142:             else b"'" + b.replace(b"'", b"'\"'\"'") + b"'")
143: 
144: 
145: class _SVGConverter(object):
146:     def __init__(self):
147:         self._proc = None
148:         # We cannot rely on the GC to trigger `__del__` at exit because
149:         # other modules (e.g. `subprocess`) may already have their globals
150:         # set to `None`, which make `proc.communicate` or `proc.terminate`
151:         # fail.  By relying on `atexit` we ensure the destructor runs before
152:         # `None`-setting occurs.
153:         atexit.register(self.__del__)
154: 
155:     def _read_to_prompt(self):
156:         '''Did Inkscape reach the prompt without crashing?
157:         '''
158:         stream = iter(functools.partial(self._proc.stdout.read, 1), b"")
159:         prompt = (b"\n", b">")
160:         n = len(prompt)
161:         its = itertools.tee(stream, n)
162:         for i, it in enumerate(its):
163:             next(itertools.islice(it, i, i), None)  # Advance `it` by `i`.
164:         while True:
165:             window = tuple(map(next, its))
166:             if len(window) != n:
167:                 # Ran out of data -- one of the `next(it)` raised
168:                 # StopIteration, so the tuple is shorter.
169:                 return False
170:             if self._proc.poll() is not None:
171:                 # Inkscape exited.
172:                 return False
173:             if window == prompt:
174:                 # Successfully read until prompt.
175:                 return True
176: 
177:     def __call__(self, orig, dest):
178:         if (not self._proc  # First run.
179:                 or self._proc.poll() is not None):  # Inkscape terminated.
180:             env = os.environ.copy()
181:             # If one passes e.g. a png file to Inkscape, it will try to
182:             # query the user for conversion options via a GUI (even with
183:             # `--without-gui`).  Unsetting `DISPLAY` prevents this (and causes
184:             # GTK to crash and Inkscape to terminate, but that'll just be
185:             # reported as a regular exception below).
186:             env.pop("DISPLAY", None)  # May already be unset.
187:             # Do not load any user options.
188:             # `os.environ` needs native strings on Py2+Windows.
189:             env[str("INKSCAPE_PROFILE_DIR")] = os.devnull
190:             # Old versions of Inkscape (0.48.3.1, used on Travis as of now)
191:             # seem to sometimes deadlock when stderr is redirected to a pipe,
192:             # so we redirect it to a temporary file instead.  This is not
193:             # necessary anymore as of Inkscape 0.92.1.
194:             self._stderr = TemporaryFile()
195:             self._proc = subprocess.Popen(
196:                 [str("inkscape"), "--without-gui", "--shell"],
197:                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
198:                 stderr=self._stderr, env=env)
199:             if not self._read_to_prompt():
200:                 raise OSError("Failed to start Inkscape")
201: 
202:         try:
203:             fsencode = os.fsencode
204:         except AttributeError:  # Py2.
205:             def fsencode(s):
206:                 return s.encode(sys.getfilesystemencoding())
207: 
208:         # Inkscape uses glib's `g_shell_parse_argv`, which has a consistent
209:         # behavior across platforms, so we can just use `shlex.quote`.
210:         orig_b, dest_b = map(_shlex_quote_bytes, map(fsencode, [orig, dest]))
211:         if b"\n" in orig_b or b"\n" in dest_b:
212:             # Who knows whether the current folder name has a newline, or if
213:             # our encoding is even ASCII compatible...  Just fall back on the
214:             # slow solution (Inkscape uses `fgets` so it will always stop at a
215:             # newline).
216:             return make_external_conversion_command(lambda old, new: [
217:                 str('inkscape'), '-z', old, '--export-png', new])(orig, dest)
218:         self._proc.stdin.write(orig_b + b" --export-png=" + dest_b + b"\n")
219:         self._proc.stdin.flush()
220:         if not self._read_to_prompt():
221:             # Inkscape's output is not localized but gtk's is, so the
222:             # output stream probably has a mixed encoding.  Using
223:             # `getfilesystemencoding` should at least get the filenames
224:             # right...
225:             self._stderr.seek(0)
226:             raise ImageComparisonFailure(
227:                 self._stderr.read().decode(
228:                     sys.getfilesystemencoding(), "replace"))
229: 
230:     def __del__(self):
231:         if self._proc:
232:             if self._proc.poll() is None:  # Not exited yet.
233:                 self._proc.communicate(b"quit\n")
234:                 self._proc.wait()
235:             self._proc.stdin.close()
236:             self._proc.stdout.close()
237:             self._stderr.close()
238: 
239: 
240: def _update_converter():
241:     gs, gs_v = matplotlib.checkdep_ghostscript()
242:     if gs_v is not None:
243:         def cmd(old, new):
244:             return [str(gs), '-q', '-sDEVICE=png16m', '-dNOPAUSE', '-dBATCH',
245:              '-sOutputFile=' + new, old]
246:         converter['pdf'] = make_external_conversion_command(cmd)
247:         converter['eps'] = make_external_conversion_command(cmd)
248: 
249:     if matplotlib.checkdep_inkscape() is not None:
250:         converter['svg'] = _SVGConverter()
251: 
252: 
253: #: A dictionary that maps filename extensions to functions which
254: #: themselves map arguments `old` and `new` (filenames) to a list of strings.
255: #: The list can then be passed to Popen to convert files with that
256: #: extension to png format.
257: converter = {}
258: _update_converter()
259: 
260: 
261: def comparable_formats():
262:     '''
263:     Returns the list of file formats that compare_images can compare
264:     on this system.
265: 
266:     '''
267:     return ['png'] + list(converter)
268: 
269: 
270: def convert(filename, cache):
271:     '''
272:     Convert the named file into a png file.  Returns the name of the
273:     created file.
274: 
275:     If *cache* is True, the result of the conversion is cached in
276:     `matplotlib._get_cachedir() + '/test_cache/'`.  The caching is based
277:     on a hash of the exact contents of the input file.  The is no limit
278:     on the size of the cache, so it may need to be manually cleared
279:     periodically.
280: 
281:     '''
282:     base, extension = filename.rsplit('.', 1)
283:     if extension not in converter:
284:         reason = "Don't know how to convert %s files to png" % extension
285:         from . import is_called_from_pytest
286:         if is_called_from_pytest():
287:             import pytest
288:             pytest.skip(reason)
289:         else:
290:             from nose import SkipTest
291:             raise SkipTest(reason)
292:     newname = base + '_' + extension + '.png'
293:     if not os.path.exists(filename):
294:         raise IOError("'%s' does not exist" % filename)
295: 
296:     # Only convert the file if the destination doesn't already exist or
297:     # is out of date.
298:     if (not os.path.exists(newname) or
299:             os.stat(newname).st_mtime < os.stat(filename).st_mtime):
300:         if cache:
301:             cache_dir = get_cache_dir()
302:         else:
303:             cache_dir = None
304: 
305:         if cache_dir is not None:
306:             hash_value = get_file_hash(filename)
307:             new_ext = os.path.splitext(newname)[1]
308:             cached_file = os.path.join(cache_dir, hash_value + new_ext)
309:             if os.path.exists(cached_file):
310:                 shutil.copyfile(cached_file, newname)
311:                 return newname
312: 
313:         converter[extension](filename, newname)
314: 
315:         if cache_dir is not None:
316:             shutil.copyfile(newname, cached_file)
317: 
318:     return newname
319: 
320: #: Maps file extensions to a function which takes a filename as its
321: #: only argument to return a list suitable for execution with Popen.
322: #: The purpose of this is so that the result file (with the given
323: #: extension) can be verified with tools such as xmllint for svg.
324: verifiers = {}
325: 
326: # Turning this off, because it seems to cause multiprocessing issues
327: if False and matplotlib.checkdep_xmllint():
328:     verifiers['svg'] = lambda filename: [
329:         'xmllint', '--valid', '--nowarning', '--noout', filename]
330: 
331: 
332: @cbook.deprecated("2.1")
333: def verify(filename):
334:     '''Verify the file through some sort of verification tool.'''
335:     if not os.path.exists(filename):
336:         raise IOError("'%s' does not exist" % filename)
337:     base, extension = filename.rsplit('.', 1)
338:     verifier = verifiers.get(extension, None)
339:     if verifier is not None:
340:         cmd = verifier(filename)
341:         pipe = subprocess.Popen(cmd, universal_newlines=True,
342:                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
343:         stdout, stderr = pipe.communicate()
344:         errcode = pipe.wait()
345:         if errcode != 0:
346:             msg = "File verification command failed:\n%s\n" % ' '.join(cmd)
347:             if stdout:
348:                 msg += "Standard output:\n%s\n" % stdout
349:             if stderr:
350:                 msg += "Standard error:\n%s\n" % stderr
351:             raise IOError(msg)
352: 
353: 
354: def crop_to_same(actual_path, actual_image, expected_path, expected_image):
355:     # clip the images to the same size -- this is useful only when
356:     # comparing eps to pdf
357:     if actual_path[-7:-4] == 'eps' and expected_path[-7:-4] == 'pdf':
358:         aw, ah, ad = actual_image.shape
359:         ew, eh, ed = expected_image.shape
360:         actual_image = actual_image[int(aw / 2 - ew / 2):int(
361:             aw / 2 + ew / 2), int(ah / 2 - eh / 2):int(ah / 2 + eh / 2)]
362:     return actual_image, expected_image
363: 
364: 
365: def calculate_rms(expectedImage, actualImage):
366:     "Calculate the per-pixel errors, then compute the root mean square error."
367:     if expectedImage.shape != actualImage.shape:
368:         raise ImageComparisonFailure(
369:             "Image sizes do not match expected size: {0} "
370:             "actual size {1}".format(expectedImage.shape, actualImage.shape))
371:     num_values = expectedImage.size
372:     abs_diff_image = abs(expectedImage - actualImage)
373:     histogram = np.bincount(abs_diff_image.ravel(), minlength=256)
374:     sum_of_squares = np.sum(histogram * np.arange(len(histogram)) ** 2)
375:     rms = np.sqrt(float(sum_of_squares) / num_values)
376:     return rms
377: 
378: 
379: def compare_images(expected, actual, tol, in_decorator=False):
380:     '''
381:     Compare two "image" files checking differences within a tolerance.
382: 
383:     The two given filenames may point to files which are convertible to
384:     PNG via the `.converter` dictionary. The underlying RMS is calculated
385:     with the `.calculate_rms` function.
386: 
387:     Parameters
388:     ----------
389:     expected : str
390:         The filename of the expected image.
391:     actual :str
392:         The filename of the actual image.
393:     tol : float
394:         The tolerance (a color value difference, where 255 is the
395:         maximal difference).  The test fails if the average pixel
396:         difference is greater than this value.
397:     in_decorator : bool
398:         If called from image_comparison decorator, this should be
399:         True. (default=False)
400: 
401:     Examples
402:     --------
403:     img1 = "./baseline/plot.png"
404:     img2 = "./output/plot.png"
405:     compare_images( img1, img2, 0.001 ):
406: 
407:     '''
408:     if not os.path.exists(actual):
409:         raise Exception("Output image %s does not exist." % actual)
410: 
411:     if os.stat(actual).st_size == 0:
412:         raise Exception("Output image file %s is empty." % actual)
413: 
414:     # Convert the image to png
415:     extension = expected.split('.')[-1]
416: 
417:     if not os.path.exists(expected):
418:         raise IOError('Baseline image %r does not exist.' % expected)
419: 
420:     if extension != 'png':
421:         actual = convert(actual, False)
422:         expected = convert(expected, True)
423: 
424:     # open the image files and remove the alpha channel (if it exists)
425:     expectedImage = _png.read_png_int(expected)
426:     actualImage = _png.read_png_int(actual)
427:     expectedImage = expectedImage[:, :, :3]
428:     actualImage = actualImage[:, :, :3]
429: 
430:     actualImage, expectedImage = crop_to_same(
431:         actual, actualImage, expected, expectedImage)
432: 
433:     diff_image = make_test_filename(actual, 'failed-diff')
434: 
435:     if tol <= 0.0:
436:         if np.array_equal(expectedImage, actualImage):
437:             return None
438: 
439:     # convert to signed integers, so that the images can be subtracted without
440:     # overflow
441:     expectedImage = expectedImage.astype(np.int16)
442:     actualImage = actualImage.astype(np.int16)
443: 
444:     rms = calculate_rms(expectedImage, actualImage)
445: 
446:     if rms <= tol:
447:         return None
448: 
449:     save_diff_image(expected, actual, diff_image)
450: 
451:     results = dict(rms=rms, expected=str(expected),
452:                    actual=str(actual), diff=str(diff_image), tol=tol)
453: 
454:     if not in_decorator:
455:         # Then the results should be a string suitable for stdout.
456:         template = ['Error: Image files did not match.',
457:                     'RMS Value: {rms}',
458:                     'Expected:  \n    {expected}',
459:                     'Actual:    \n    {actual}',
460:                     'Difference:\n    {diff}',
461:                     'Tolerance: \n    {tol}', ]
462:         results = '\n  '.join([line.format(**results) for line in template])
463:     return results
464: 
465: 
466: def save_diff_image(expected, actual, output):
467:     expectedImage = _png.read_png(expected)
468:     actualImage = _png.read_png(actual)
469:     actualImage, expectedImage = crop_to_same(
470:         actual, actualImage, expected, expectedImage)
471:     expectedImage = np.array(expectedImage).astype(float)
472:     actualImage = np.array(actualImage).astype(float)
473:     if expectedImage.shape != actualImage.shape:
474:         raise ImageComparisonFailure(
475:             "Image sizes do not match expected size: {0} "
476:             "actual size {1}".format(expectedImage.shape, actualImage.shape))
477:     absDiffImage = np.abs(expectedImage - actualImage)
478: 
479:     # expand differences in luminance domain
480:     absDiffImage *= 255 * 10
481:     save_image_np = np.clip(absDiffImage, 0, 255).astype(np.uint8)
482:     height, width, depth = save_image_np.shape
483: 
484:     # The PDF renderer doesn't produce an alpha channel, but the
485:     # matplotlib PNG writer requires one, so expand the array
486:     if depth == 3:
487:         with_alpha = np.empty((height, width, 4), dtype=np.uint8)
488:         with_alpha[:, :, 0:3] = save_image_np
489:         save_image_np = with_alpha
490: 
491:     # Hard-code the alpha channel to fully solid
492:     save_image_np[:, :, 3] = 255
493: 
494:     _png.write_png(save_image_np, output)
495: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_288361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'unicode', u'\nProvides a collection of utilities for comparing (image) results.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import six' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_288362 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six')

if (type(import_288362) is not StypyTypeError):

    if (import_288362 != 'pyd_module'):
        __import__(import_288362)
        sys_modules_288363 = sys.modules[import_288362]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', sys_modules_288363.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', import_288362)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import atexit' statement (line 10)
import atexit

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'atexit', atexit, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import functools' statement (line 11)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'functools', functools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import hashlib' statement (line 12)
import hashlib

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'hashlib', hashlib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import itertools' statement (line 13)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import os' statement (line 14)
import os

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import re' statement (line 15)
import re

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import shutil' statement (line 16)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import sys' statement (line 17)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from tempfile import TemporaryFile' statement (line 18)
try:
    from tempfile import TemporaryFile

except:
    TemporaryFile = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'tempfile', None, module_type_store, ['TemporaryFile'], [TemporaryFile])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import numpy' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_288364 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy')

if (type(import_288364) is not StypyTypeError):

    if (import_288364 != 'pyd_module'):
        __import__(import_288364)
        sys_modules_288365 = sys.modules[import_288364]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'np', sys_modules_288365.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy', import_288364)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import matplotlib' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_288366 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib')

if (type(import_288366) is not StypyTypeError):

    if (import_288366 != 'pyd_module'):
        __import__(import_288366)
        sys_modules_288367 = sys.modules[import_288366]
        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib', sys_modules_288367.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib', import_288366)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from matplotlib.compat import subprocess' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_288368 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib.compat')

if (type(import_288368) is not StypyTypeError):

    if (import_288368 != 'pyd_module'):
        __import__(import_288368)
        sys_modules_288369 = sys.modules[import_288368]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib.compat', sys_modules_288369.module_type_store, module_type_store, ['subprocess'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_288369, sys_modules_288369.module_type_store, module_type_store)
    else:
        from matplotlib.compat import subprocess

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib.compat', None, module_type_store, ['subprocess'], [subprocess])

else:
    # Assigning a type to the variable 'matplotlib.compat' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib.compat', import_288368)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from matplotlib.testing.exceptions import ImageComparisonFailure' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_288370 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.testing.exceptions')

if (type(import_288370) is not StypyTypeError):

    if (import_288370 != 'pyd_module'):
        __import__(import_288370)
        sys_modules_288371 = sys.modules[import_288370]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.testing.exceptions', sys_modules_288371.module_type_store, module_type_store, ['ImageComparisonFailure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_288371, sys_modules_288371.module_type_store, module_type_store)
    else:
        from matplotlib.testing.exceptions import ImageComparisonFailure

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.testing.exceptions', None, module_type_store, ['ImageComparisonFailure'], [ImageComparisonFailure])

else:
    # Assigning a type to the variable 'matplotlib.testing.exceptions' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.testing.exceptions', import_288370)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from matplotlib import _png' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_288372 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib')

if (type(import_288372) is not StypyTypeError):

    if (import_288372 != 'pyd_module'):
        __import__(import_288372)
        sys_modules_288373 = sys.modules[import_288372]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', sys_modules_288373.module_type_store, module_type_store, ['_png'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_288373, sys_modules_288373.module_type_store, module_type_store)
    else:
        from matplotlib import _png

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', None, module_type_store, ['_png'], [_png])

else:
    # Assigning a type to the variable 'matplotlib' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', import_288372)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from matplotlib import _get_cachedir' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_288374 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib')

if (type(import_288374) is not StypyTypeError):

    if (import_288374 != 'pyd_module'):
        __import__(import_288374)
        sys_modules_288375 = sys.modules[import_288374]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib', sys_modules_288375.module_type_store, module_type_store, ['_get_cachedir'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_288375, sys_modules_288375.module_type_store, module_type_store)
    else:
        from matplotlib import _get_cachedir

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib', None, module_type_store, ['_get_cachedir'], [_get_cachedir])

else:
    # Assigning a type to the variable 'matplotlib' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib', import_288374)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from matplotlib import cbook' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_288376 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib')

if (type(import_288376) is not StypyTypeError):

    if (import_288376 != 'pyd_module'):
        __import__(import_288376)
        sys_modules_288377 = sys.modules[import_288376]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib', sys_modules_288377.module_type_store, module_type_store, ['cbook'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_288377, sys_modules_288377.module_type_store, module_type_store)
    else:
        from matplotlib import cbook

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib', None, module_type_store, ['cbook'], [cbook])

else:
    # Assigning a type to the variable 'matplotlib' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib', import_288376)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')


# Assigning a List to a Name (line 29):

# Assigning a List to a Name (line 29):
__all__ = [u'compare_float', u'compare_images', u'comparable_formats']
module_type_store.set_exportable_members([u'compare_float', u'compare_images', u'comparable_formats'])

# Obtaining an instance of the builtin type 'list' (line 29)
list_288378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
unicode_288379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'unicode', u'compare_float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_288378, unicode_288379)
# Adding element type (line 29)
unicode_288380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'unicode', u'compare_images')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_288378, unicode_288380)
# Adding element type (line 29)
unicode_288381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 46), 'unicode', u'comparable_formats')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_288378, unicode_288381)

# Assigning a type to the variable '__all__' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '__all__', list_288378)

@norecursion
def make_test_filename(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'make_test_filename'
    module_type_store = module_type_store.open_function_context('make_test_filename', 32, 0, False)
    
    # Passed parameters checking function
    make_test_filename.stypy_localization = localization
    make_test_filename.stypy_type_of_self = None
    make_test_filename.stypy_type_store = module_type_store
    make_test_filename.stypy_function_name = 'make_test_filename'
    make_test_filename.stypy_param_names_list = ['fname', 'purpose']
    make_test_filename.stypy_varargs_param_name = None
    make_test_filename.stypy_kwargs_param_name = None
    make_test_filename.stypy_call_defaults = defaults
    make_test_filename.stypy_call_varargs = varargs
    make_test_filename.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_test_filename', ['fname', 'purpose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_test_filename', localization, ['fname', 'purpose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_test_filename(...)' code ##################

    unicode_288382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'unicode', u"\n    Make a new filename by inserting `purpose` before the file's\n    extension.\n    ")
    
    # Assigning a Call to a Tuple (line 37):
    
    # Assigning a Call to a Name:
    
    # Call to splitext(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'fname' (line 37)
    fname_288386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 33), 'fname', False)
    # Processing the call keyword arguments (line 37)
    kwargs_288387 = {}
    # Getting the type of 'os' (line 37)
    os_288383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 37)
    path_288384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), os_288383, 'path')
    # Obtaining the member 'splitext' of a type (line 37)
    splitext_288385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), path_288384, 'splitext')
    # Calling splitext(args, kwargs) (line 37)
    splitext_call_result_288388 = invoke(stypy.reporting.localization.Localization(__file__, 37, 16), splitext_288385, *[fname_288386], **kwargs_288387)
    
    # Assigning a type to the variable 'call_assignment_288325' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'call_assignment_288325', splitext_call_result_288388)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_288391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'int')
    # Processing the call keyword arguments
    kwargs_288392 = {}
    # Getting the type of 'call_assignment_288325' (line 37)
    call_assignment_288325_288389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'call_assignment_288325', False)
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___288390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), call_assignment_288325_288389, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_288393 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___288390, *[int_288391], **kwargs_288392)
    
    # Assigning a type to the variable 'call_assignment_288326' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'call_assignment_288326', getitem___call_result_288393)
    
    # Assigning a Name to a Name (line 37):
    # Getting the type of 'call_assignment_288326' (line 37)
    call_assignment_288326_288394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'call_assignment_288326')
    # Assigning a type to the variable 'base' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'base', call_assignment_288326_288394)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_288397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'int')
    # Processing the call keyword arguments
    kwargs_288398 = {}
    # Getting the type of 'call_assignment_288325' (line 37)
    call_assignment_288325_288395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'call_assignment_288325', False)
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___288396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), call_assignment_288325_288395, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_288399 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___288396, *[int_288397], **kwargs_288398)
    
    # Assigning a type to the variable 'call_assignment_288327' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'call_assignment_288327', getitem___call_result_288399)
    
    # Assigning a Name to a Name (line 37):
    # Getting the type of 'call_assignment_288327' (line 37)
    call_assignment_288327_288400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'call_assignment_288327')
    # Assigning a type to the variable 'ext' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'ext', call_assignment_288327_288400)
    unicode_288401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 11), 'unicode', u'%s-%s%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_288402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    # Getting the type of 'base' (line 38)
    base_288403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'base')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), tuple_288402, base_288403)
    # Adding element type (line 38)
    # Getting the type of 'purpose' (line 38)
    purpose_288404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'purpose')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), tuple_288402, purpose_288404)
    # Adding element type (line 38)
    # Getting the type of 'ext' (line 38)
    ext_288405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 39), 'ext')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), tuple_288402, ext_288405)
    
    # Applying the binary operator '%' (line 38)
    result_mod_288406 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 11), '%', unicode_288401, tuple_288402)
    
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type', result_mod_288406)
    
    # ################# End of 'make_test_filename(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_test_filename' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_288407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288407)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_test_filename'
    return stypy_return_type_288407

# Assigning a type to the variable 'make_test_filename' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'make_test_filename', make_test_filename)

@norecursion
def compare_float(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 41)
    None_288408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 43), 'None')
    # Getting the type of 'None' (line 41)
    None_288409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 56), 'None')
    defaults = [None_288408, None_288409]
    # Create a new context for function 'compare_float'
    module_type_store = module_type_store.open_function_context('compare_float', 41, 0, False)
    
    # Passed parameters checking function
    compare_float.stypy_localization = localization
    compare_float.stypy_type_of_self = None
    compare_float.stypy_type_store = module_type_store
    compare_float.stypy_function_name = 'compare_float'
    compare_float.stypy_param_names_list = ['expected', 'actual', 'relTol', 'absTol']
    compare_float.stypy_varargs_param_name = None
    compare_float.stypy_kwargs_param_name = None
    compare_float.stypy_call_defaults = defaults
    compare_float.stypy_call_varargs = varargs
    compare_float.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compare_float', ['expected', 'actual', 'relTol', 'absTol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compare_float', localization, ['expected', 'actual', 'relTol', 'absTol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compare_float(...)' code ##################

    unicode_288410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'unicode', u'\n    Fail if the floating point values are not close enough, with\n    the given message.\n\n    You can specify a relative tolerance, absolute tolerance, or both.\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'relTol' (line 49)
    relTol_288411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 7), 'relTol')
    # Getting the type of 'None' (line 49)
    None_288412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'None')
    # Applying the binary operator 'is' (line 49)
    result_is__288413 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 7), 'is', relTol_288411, None_288412)
    
    
    # Getting the type of 'absTol' (line 49)
    absTol_288414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'absTol')
    # Getting the type of 'None' (line 49)
    None_288415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'None')
    # Applying the binary operator 'is' (line 49)
    result_is__288416 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 26), 'is', absTol_288414, None_288415)
    
    # Applying the binary operator 'and' (line 49)
    result_and_keyword_288417 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 7), 'and', result_is__288413, result_is__288416)
    
    # Testing the type of an if condition (line 49)
    if_condition_288418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 4), result_and_keyword_288417)
    # Assigning a type to the variable 'if_condition_288418' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'if_condition_288418', if_condition_288418)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 50)
    # Processing the call arguments (line 50)
    unicode_288420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 25), 'unicode', u"You haven't specified a 'relTol' relative tolerance or a 'absTol' absolute tolerance function argument. You must specify one.")
    # Processing the call keyword arguments (line 50)
    kwargs_288421 = {}
    # Getting the type of 'ValueError' (line 50)
    ValueError_288419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 50)
    ValueError_call_result_288422 = invoke(stypy.reporting.localization.Localization(__file__, 50, 14), ValueError_288419, *[unicode_288420], **kwargs_288421)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 50, 8), ValueError_call_result_288422, 'raise parameter', BaseException)
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 53):
    
    # Assigning a Str to a Name (line 53):
    unicode_288423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 10), 'unicode', u'')
    # Assigning a type to the variable 'msg' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'msg', unicode_288423)
    
    # Type idiom detected: calculating its left and rigth part (line 55)
    # Getting the type of 'absTol' (line 55)
    absTol_288424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'absTol')
    # Getting the type of 'None' (line 55)
    None_288425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'None')
    
    (may_be_288426, more_types_in_union_288427) = may_not_be_none(absTol_288424, None_288425)

    if may_be_288426:

        if more_types_in_union_288427:
            # Runtime conditional SSA (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to abs(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'expected' (line 56)
        expected_288429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'expected', False)
        # Getting the type of 'actual' (line 56)
        actual_288430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'actual', False)
        # Applying the binary operator '-' (line 56)
        result_sub_288431 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 22), '-', expected_288429, actual_288430)
        
        # Processing the call keyword arguments (line 56)
        kwargs_288432 = {}
        # Getting the type of 'abs' (line 56)
        abs_288428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'abs', False)
        # Calling abs(args, kwargs) (line 56)
        abs_call_result_288433 = invoke(stypy.reporting.localization.Localization(__file__, 56, 18), abs_288428, *[result_sub_288431], **kwargs_288432)
        
        # Assigning a type to the variable 'absDiff' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'absDiff', abs_call_result_288433)
        
        
        # Getting the type of 'absTol' (line 57)
        absTol_288434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'absTol')
        # Getting the type of 'absDiff' (line 57)
        absDiff_288435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'absDiff')
        # Applying the binary operator '<' (line 57)
        result_lt_288436 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 11), '<', absTol_288434, absDiff_288435)
        
        # Testing the type of an if condition (line 57)
        if_condition_288437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 8), result_lt_288436)
        # Assigning a type to the variable 'if_condition_288437' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'if_condition_288437', if_condition_288437)
        # SSA begins for if statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 58):
        
        # Assigning a List to a Name (line 58):
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_288438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        unicode_288439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 24), 'unicode', u'')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), list_288438, unicode_288439)
        # Adding element type (line 58)
        unicode_288440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'unicode', u'Expected: {expected}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), list_288438, unicode_288440)
        # Adding element type (line 58)
        unicode_288441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 24), 'unicode', u'Actual:   {actual}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), list_288438, unicode_288441)
        # Adding element type (line 58)
        unicode_288442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'unicode', u'Abs diff: {absDiff}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), list_288438, unicode_288442)
        # Adding element type (line 58)
        unicode_288443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'unicode', u'Abs tol:  {absTol}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), list_288438, unicode_288443)
        
        # Assigning a type to the variable 'template' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'template', list_288438)
        
        # Getting the type of 'msg' (line 63)
        msg_288444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'msg')
        
        # Call to join(...): (line 63)
        # Processing the call arguments (line 63)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'template' (line 63)
        template_288454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 68), 'template', False)
        comprehension_288455 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 32), template_288454)
        # Assigning a type to the variable 'line' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'line', comprehension_288455)
        
        # Call to format(...): (line 63)
        # Processing the call keyword arguments (line 63)
        
        # Call to locals(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_288450 = {}
        # Getting the type of 'locals' (line 63)
        locals_288449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 46), 'locals', False)
        # Calling locals(args, kwargs) (line 63)
        locals_call_result_288451 = invoke(stypy.reporting.localization.Localization(__file__, 63, 46), locals_288449, *[], **kwargs_288450)
        
        kwargs_288452 = {'locals_call_result_288451': locals_call_result_288451}
        # Getting the type of 'line' (line 63)
        line_288447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'line', False)
        # Obtaining the member 'format' of a type (line 63)
        format_288448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 32), line_288447, 'format')
        # Calling format(args, kwargs) (line 63)
        format_call_result_288453 = invoke(stypy.reporting.localization.Localization(__file__, 63, 32), format_288448, *[], **kwargs_288452)
        
        list_288456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 32), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 32), list_288456, format_call_result_288453)
        # Processing the call keyword arguments (line 63)
        kwargs_288457 = {}
        unicode_288445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'unicode', u'\n  ')
        # Obtaining the member 'join' of a type (line 63)
        join_288446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), unicode_288445, 'join')
        # Calling join(args, kwargs) (line 63)
        join_call_result_288458 = invoke(stypy.reporting.localization.Localization(__file__, 63, 19), join_288446, *[list_288456], **kwargs_288457)
        
        # Applying the binary operator '+=' (line 63)
        result_iadd_288459 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 12), '+=', msg_288444, join_call_result_288458)
        # Assigning a type to the variable 'msg' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'msg', result_iadd_288459)
        
        # SSA join for if statement (line 57)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_288427:
            # SSA join for if statement (line 55)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 65)
    # Getting the type of 'relTol' (line 65)
    relTol_288460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'relTol')
    # Getting the type of 'None' (line 65)
    None_288461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'None')
    
    (may_be_288462, more_types_in_union_288463) = may_not_be_none(relTol_288460, None_288461)

    if may_be_288462:

        if more_types_in_union_288463:
            # Runtime conditional SSA (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to abs(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'expected' (line 68)
        expected_288465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'expected', False)
        # Getting the type of 'actual' (line 68)
        actual_288466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 33), 'actual', False)
        # Applying the binary operator '-' (line 68)
        result_sub_288467 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 22), '-', expected_288465, actual_288466)
        
        # Processing the call keyword arguments (line 68)
        kwargs_288468 = {}
        # Getting the type of 'abs' (line 68)
        abs_288464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), 'abs', False)
        # Calling abs(args, kwargs) (line 68)
        abs_call_result_288469 = invoke(stypy.reporting.localization.Localization(__file__, 68, 18), abs_288464, *[result_sub_288467], **kwargs_288468)
        
        # Assigning a type to the variable 'relDiff' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'relDiff', abs_call_result_288469)
        
        # Getting the type of 'expected' (line 69)
        expected_288470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'expected')
        # Testing the type of an if condition (line 69)
        if_condition_288471 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), expected_288470)
        # Assigning a type to the variable 'if_condition_288471' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_288471', if_condition_288471)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 70):
        
        # Assigning a BinOp to a Name (line 70):
        # Getting the type of 'relDiff' (line 70)
        relDiff_288472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 22), 'relDiff')
        
        # Call to abs(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'expected' (line 70)
        expected_288474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 36), 'expected', False)
        # Processing the call keyword arguments (line 70)
        kwargs_288475 = {}
        # Getting the type of 'abs' (line 70)
        abs_288473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'abs', False)
        # Calling abs(args, kwargs) (line 70)
        abs_call_result_288476 = invoke(stypy.reporting.localization.Localization(__file__, 70, 32), abs_288473, *[expected_288474], **kwargs_288475)
        
        # Applying the binary operator 'div' (line 70)
        result_div_288477 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 22), 'div', relDiff_288472, abs_call_result_288476)
        
        # Assigning a type to the variable 'relDiff' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'relDiff', result_div_288477)
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'relTol' (line 72)
        relTol_288478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'relTol')
        # Getting the type of 'relDiff' (line 72)
        relDiff_288479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'relDiff')
        # Applying the binary operator '<' (line 72)
        result_lt_288480 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 11), '<', relTol_288478, relDiff_288479)
        
        # Testing the type of an if condition (line 72)
        if_condition_288481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 8), result_lt_288480)
        # Assigning a type to the variable 'if_condition_288481' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'if_condition_288481', if_condition_288481)
        # SSA begins for if statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 74):
        
        # Assigning a List to a Name (line 74):
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_288482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        unicode_288483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'unicode', u'')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), list_288482, unicode_288483)
        # Adding element type (line 74)
        unicode_288484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 24), 'unicode', u'Expected: {expected}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), list_288482, unicode_288484)
        # Adding element type (line 74)
        unicode_288485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 24), 'unicode', u'Actual:   {actual}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), list_288482, unicode_288485)
        # Adding element type (line 74)
        unicode_288486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'unicode', u'Rel diff: {relDiff}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), list_288482, unicode_288486)
        # Adding element type (line 74)
        unicode_288487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 24), 'unicode', u'Rel tol:  {relTol}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), list_288482, unicode_288487)
        
        # Assigning a type to the variable 'template' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'template', list_288482)
        
        # Getting the type of 'msg' (line 79)
        msg_288488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'msg')
        
        # Call to join(...): (line 79)
        # Processing the call arguments (line 79)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'template' (line 79)
        template_288498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 68), 'template', False)
        comprehension_288499 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 32), template_288498)
        # Assigning a type to the variable 'line' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 32), 'line', comprehension_288499)
        
        # Call to format(...): (line 79)
        # Processing the call keyword arguments (line 79)
        
        # Call to locals(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_288494 = {}
        # Getting the type of 'locals' (line 79)
        locals_288493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 46), 'locals', False)
        # Calling locals(args, kwargs) (line 79)
        locals_call_result_288495 = invoke(stypy.reporting.localization.Localization(__file__, 79, 46), locals_288493, *[], **kwargs_288494)
        
        kwargs_288496 = {'locals_call_result_288495': locals_call_result_288495}
        # Getting the type of 'line' (line 79)
        line_288491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 32), 'line', False)
        # Obtaining the member 'format' of a type (line 79)
        format_288492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 32), line_288491, 'format')
        # Calling format(args, kwargs) (line 79)
        format_call_result_288497 = invoke(stypy.reporting.localization.Localization(__file__, 79, 32), format_288492, *[], **kwargs_288496)
        
        list_288500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 32), list_288500, format_call_result_288497)
        # Processing the call keyword arguments (line 79)
        kwargs_288501 = {}
        unicode_288489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'unicode', u'\n  ')
        # Obtaining the member 'join' of a type (line 79)
        join_288490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 19), unicode_288489, 'join')
        # Calling join(args, kwargs) (line 79)
        join_call_result_288502 = invoke(stypy.reporting.localization.Localization(__file__, 79, 19), join_288490, *[list_288500], **kwargs_288501)
        
        # Applying the binary operator '+=' (line 79)
        result_iadd_288503 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 12), '+=', msg_288488, join_call_result_288502)
        # Assigning a type to the variable 'msg' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'msg', result_iadd_288503)
        
        # SSA join for if statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_288463:
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Evaluating a boolean operation
    # Getting the type of 'msg' (line 81)
    msg_288504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'msg')
    # Getting the type of 'None' (line 81)
    None_288505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 18), 'None')
    # Applying the binary operator 'or' (line 81)
    result_or_keyword_288506 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 11), 'or', msg_288504, None_288505)
    
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', result_or_keyword_288506)
    
    # ################# End of 'compare_float(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compare_float' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_288507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288507)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compare_float'
    return stypy_return_type_288507

# Assigning a type to the variable 'compare_float' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'compare_float', compare_float)

@norecursion
def get_cache_dir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_cache_dir'
    module_type_store = module_type_store.open_function_context('get_cache_dir', 84, 0, False)
    
    # Passed parameters checking function
    get_cache_dir.stypy_localization = localization
    get_cache_dir.stypy_type_of_self = None
    get_cache_dir.stypy_type_store = module_type_store
    get_cache_dir.stypy_function_name = 'get_cache_dir'
    get_cache_dir.stypy_param_names_list = []
    get_cache_dir.stypy_varargs_param_name = None
    get_cache_dir.stypy_kwargs_param_name = None
    get_cache_dir.stypy_call_defaults = defaults
    get_cache_dir.stypy_call_varargs = varargs
    get_cache_dir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_cache_dir', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_cache_dir', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_cache_dir(...)' code ##################

    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to _get_cachedir(...): (line 85)
    # Processing the call keyword arguments (line 85)
    kwargs_288509 = {}
    # Getting the type of '_get_cachedir' (line 85)
    _get_cachedir_288508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), '_get_cachedir', False)
    # Calling _get_cachedir(args, kwargs) (line 85)
    _get_cachedir_call_result_288510 = invoke(stypy.reporting.localization.Localization(__file__, 85, 15), _get_cachedir_288508, *[], **kwargs_288509)
    
    # Assigning a type to the variable 'cachedir' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'cachedir', _get_cachedir_call_result_288510)
    
    # Type idiom detected: calculating its left and rigth part (line 86)
    # Getting the type of 'cachedir' (line 86)
    cachedir_288511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 7), 'cachedir')
    # Getting the type of 'None' (line 86)
    None_288512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'None')
    
    (may_be_288513, more_types_in_union_288514) = may_be_none(cachedir_288511, None_288512)

    if may_be_288513:

        if more_types_in_union_288514:
            # Runtime conditional SSA (line 86)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to RuntimeError(...): (line 87)
        # Processing the call arguments (line 87)
        unicode_288516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'unicode', u'Could not find a suitable configuration directory')
        # Processing the call keyword arguments (line 87)
        kwargs_288517 = {}
        # Getting the type of 'RuntimeError' (line 87)
        RuntimeError_288515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 14), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 87)
        RuntimeError_call_result_288518 = invoke(stypy.reporting.localization.Localization(__file__, 87, 14), RuntimeError_288515, *[unicode_288516], **kwargs_288517)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 87, 8), RuntimeError_call_result_288518, 'raise parameter', BaseException)

        if more_types_in_union_288514:
            # SSA join for if statement (line 86)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 88):
    
    # Assigning a Call to a Name (line 88):
    
    # Call to join(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'cachedir' (line 88)
    cachedir_288522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'cachedir', False)
    unicode_288523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 39), 'unicode', u'test_cache')
    # Processing the call keyword arguments (line 88)
    kwargs_288524 = {}
    # Getting the type of 'os' (line 88)
    os_288519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 88)
    path_288520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 16), os_288519, 'path')
    # Obtaining the member 'join' of a type (line 88)
    join_288521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 16), path_288520, 'join')
    # Calling join(args, kwargs) (line 88)
    join_call_result_288525 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), join_288521, *[cachedir_288522, unicode_288523], **kwargs_288524)
    
    # Assigning a type to the variable 'cache_dir' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'cache_dir', join_call_result_288525)
    
    
    
    # Call to exists(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'cache_dir' (line 89)
    cache_dir_288529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'cache_dir', False)
    # Processing the call keyword arguments (line 89)
    kwargs_288530 = {}
    # Getting the type of 'os' (line 89)
    os_288526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 89)
    path_288527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 11), os_288526, 'path')
    # Obtaining the member 'exists' of a type (line 89)
    exists_288528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 11), path_288527, 'exists')
    # Calling exists(args, kwargs) (line 89)
    exists_call_result_288531 = invoke(stypy.reporting.localization.Localization(__file__, 89, 11), exists_288528, *[cache_dir_288529], **kwargs_288530)
    
    # Applying the 'not' unary operator (line 89)
    result_not__288532 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 7), 'not', exists_call_result_288531)
    
    # Testing the type of an if condition (line 89)
    if_condition_288533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 4), result_not__288532)
    # Assigning a type to the variable 'if_condition_288533' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'if_condition_288533', if_condition_288533)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to mkdirs(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'cache_dir' (line 91)
    cache_dir_288536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'cache_dir', False)
    # Processing the call keyword arguments (line 91)
    kwargs_288537 = {}
    # Getting the type of 'cbook' (line 91)
    cbook_288534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'cbook', False)
    # Obtaining the member 'mkdirs' of a type (line 91)
    mkdirs_288535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), cbook_288534, 'mkdirs')
    # Calling mkdirs(args, kwargs) (line 91)
    mkdirs_call_result_288538 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), mkdirs_288535, *[cache_dir_288536], **kwargs_288537)
    
    # SSA branch for the except part of a try statement (line 90)
    # SSA branch for the except 'IOError' branch of a try statement (line 90)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'None' (line 93)
    None_288539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'stypy_return_type', None_288539)
    # SSA join for try-except statement (line 90)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to access(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'cache_dir' (line 94)
    cache_dir_288542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'cache_dir', False)
    # Getting the type of 'os' (line 94)
    os_288543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 32), 'os', False)
    # Obtaining the member 'W_OK' of a type (line 94)
    W_OK_288544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 32), os_288543, 'W_OK')
    # Processing the call keyword arguments (line 94)
    kwargs_288545 = {}
    # Getting the type of 'os' (line 94)
    os_288540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'os', False)
    # Obtaining the member 'access' of a type (line 94)
    access_288541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 11), os_288540, 'access')
    # Calling access(args, kwargs) (line 94)
    access_call_result_288546 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), access_288541, *[cache_dir_288542, W_OK_288544], **kwargs_288545)
    
    # Applying the 'not' unary operator (line 94)
    result_not__288547 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), 'not', access_call_result_288546)
    
    # Testing the type of an if condition (line 94)
    if_condition_288548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_not__288547)
    # Assigning a type to the variable 'if_condition_288548' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_288548', if_condition_288548)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 95)
    None_288549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', None_288549)
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'cache_dir' (line 96)
    cache_dir_288550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'cache_dir')
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type', cache_dir_288550)
    
    # ################# End of 'get_cache_dir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_cache_dir' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_288551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288551)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_cache_dir'
    return stypy_return_type_288551

# Assigning a type to the variable 'get_cache_dir' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'get_cache_dir', get_cache_dir)

@norecursion
def get_file_hash(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_288552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 35), 'int')
    int_288553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 40), 'int')
    # Applying the binary operator '**' (line 99)
    result_pow_288554 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 35), '**', int_288552, int_288553)
    
    defaults = [result_pow_288554]
    # Create a new context for function 'get_file_hash'
    module_type_store = module_type_store.open_function_context('get_file_hash', 99, 0, False)
    
    # Passed parameters checking function
    get_file_hash.stypy_localization = localization
    get_file_hash.stypy_type_of_self = None
    get_file_hash.stypy_type_store = module_type_store
    get_file_hash.stypy_function_name = 'get_file_hash'
    get_file_hash.stypy_param_names_list = ['path', 'block_size']
    get_file_hash.stypy_varargs_param_name = None
    get_file_hash.stypy_kwargs_param_name = None
    get_file_hash.stypy_call_defaults = defaults
    get_file_hash.stypy_call_varargs = varargs
    get_file_hash.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_file_hash', ['path', 'block_size'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_file_hash', localization, ['path', 'block_size'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_file_hash(...)' code ##################

    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to md5(...): (line 100)
    # Processing the call keyword arguments (line 100)
    kwargs_288557 = {}
    # Getting the type of 'hashlib' (line 100)
    hashlib_288555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 10), 'hashlib', False)
    # Obtaining the member 'md5' of a type (line 100)
    md5_288556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 10), hashlib_288555, 'md5')
    # Calling md5(args, kwargs) (line 100)
    md5_call_result_288558 = invoke(stypy.reporting.localization.Localization(__file__, 100, 10), md5_288556, *[], **kwargs_288557)
    
    # Assigning a type to the variable 'md5' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'md5', md5_call_result_288558)
    
    # Call to open(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'path' (line 101)
    path_288560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), 'path', False)
    unicode_288561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'unicode', u'rb')
    # Processing the call keyword arguments (line 101)
    kwargs_288562 = {}
    # Getting the type of 'open' (line 101)
    open_288559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 9), 'open', False)
    # Calling open(args, kwargs) (line 101)
    open_call_result_288563 = invoke(stypy.reporting.localization.Localization(__file__, 101, 9), open_288559, *[path_288560, unicode_288561], **kwargs_288562)
    
    with_288564 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 101, 9), open_call_result_288563, 'with parameter', '__enter__', '__exit__')

    if with_288564:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 101)
        enter___288565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 9), open_call_result_288563, '__enter__')
        with_enter_288566 = invoke(stypy.reporting.localization.Localization(__file__, 101, 9), enter___288565)
        # Assigning a type to the variable 'fd' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 9), 'fd', with_enter_288566)
        
        # Getting the type of 'True' (line 102)
        True_288567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'True')
        # Testing the type of an if condition (line 102)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), True_288567)
        # SSA begins for while statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to read(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'block_size' (line 103)
        block_size_288570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'block_size', False)
        # Processing the call keyword arguments (line 103)
        kwargs_288571 = {}
        # Getting the type of 'fd' (line 103)
        fd_288568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'fd', False)
        # Obtaining the member 'read' of a type (line 103)
        read_288569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 19), fd_288568, 'read')
        # Calling read(args, kwargs) (line 103)
        read_call_result_288572 = invoke(stypy.reporting.localization.Localization(__file__, 103, 19), read_288569, *[block_size_288570], **kwargs_288571)
        
        # Assigning a type to the variable 'data' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'data', read_call_result_288572)
        
        
        # Getting the type of 'data' (line 104)
        data_288573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'data')
        # Applying the 'not' unary operator (line 104)
        result_not__288574 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 15), 'not', data_288573)
        
        # Testing the type of an if condition (line 104)
        if_condition_288575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 12), result_not__288574)
        # Assigning a type to the variable 'if_condition_288575' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'if_condition_288575', if_condition_288575)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to update(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'data' (line 106)
        data_288578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'data', False)
        # Processing the call keyword arguments (line 106)
        kwargs_288579 = {}
        # Getting the type of 'md5' (line 106)
        md5_288576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'md5', False)
        # Obtaining the member 'update' of a type (line 106)
        update_288577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), md5_288576, 'update')
        # Calling update(args, kwargs) (line 106)
        update_call_result_288580 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), update_288577, *[data_288578], **kwargs_288579)
        
        # SSA join for while statement (line 102)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 101)
        exit___288581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 9), open_call_result_288563, '__exit__')
        with_exit_288582 = invoke(stypy.reporting.localization.Localization(__file__, 101, 9), exit___288581, None, None, None)

    
    
    # Call to endswith(...): (line 108)
    # Processing the call arguments (line 108)
    unicode_288585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'unicode', u'.pdf')
    # Processing the call keyword arguments (line 108)
    kwargs_288586 = {}
    # Getting the type of 'path' (line 108)
    path_288583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'path', False)
    # Obtaining the member 'endswith' of a type (line 108)
    endswith_288584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 7), path_288583, 'endswith')
    # Calling endswith(args, kwargs) (line 108)
    endswith_call_result_288587 = invoke(stypy.reporting.localization.Localization(__file__, 108, 7), endswith_288584, *[unicode_288585], **kwargs_288586)
    
    # Testing the type of an if condition (line 108)
    if_condition_288588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), endswith_call_result_288587)
    # Assigning a type to the variable 'if_condition_288588' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_288588', if_condition_288588)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 109, 8))
    
    # 'from matplotlib import checkdep_ghostscript' statement (line 109)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_288589 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 109, 8), 'matplotlib')

    if (type(import_288589) is not StypyTypeError):

        if (import_288589 != 'pyd_module'):
            __import__(import_288589)
            sys_modules_288590 = sys.modules[import_288589]
            import_from_module(stypy.reporting.localization.Localization(__file__, 109, 8), 'matplotlib', sys_modules_288590.module_type_store, module_type_store, ['checkdep_ghostscript'])
            nest_module(stypy.reporting.localization.Localization(__file__, 109, 8), __file__, sys_modules_288590, sys_modules_288590.module_type_store, module_type_store)
        else:
            from matplotlib import checkdep_ghostscript

            import_from_module(stypy.reporting.localization.Localization(__file__, 109, 8), 'matplotlib', None, module_type_store, ['checkdep_ghostscript'], [checkdep_ghostscript])

    else:
        # Assigning a type to the variable 'matplotlib' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'matplotlib', import_288589)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    # Call to update(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Call to encode(...): (line 110)
    # Processing the call arguments (line 110)
    unicode_288600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 52), 'unicode', u'utf-8')
    # Processing the call keyword arguments (line 110)
    kwargs_288601 = {}
    
    # Obtaining the type of the subscript
    int_288593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 42), 'int')
    
    # Call to checkdep_ghostscript(...): (line 110)
    # Processing the call keyword arguments (line 110)
    kwargs_288595 = {}
    # Getting the type of 'checkdep_ghostscript' (line 110)
    checkdep_ghostscript_288594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'checkdep_ghostscript', False)
    # Calling checkdep_ghostscript(args, kwargs) (line 110)
    checkdep_ghostscript_call_result_288596 = invoke(stypy.reporting.localization.Localization(__file__, 110, 19), checkdep_ghostscript_288594, *[], **kwargs_288595)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___288597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 19), checkdep_ghostscript_call_result_288596, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_288598 = invoke(stypy.reporting.localization.Localization(__file__, 110, 19), getitem___288597, int_288593)
    
    # Obtaining the member 'encode' of a type (line 110)
    encode_288599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 19), subscript_call_result_288598, 'encode')
    # Calling encode(args, kwargs) (line 110)
    encode_call_result_288602 = invoke(stypy.reporting.localization.Localization(__file__, 110, 19), encode_288599, *[unicode_288600], **kwargs_288601)
    
    # Processing the call keyword arguments (line 110)
    kwargs_288603 = {}
    # Getting the type of 'md5' (line 110)
    md5_288591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'md5', False)
    # Obtaining the member 'update' of a type (line 110)
    update_288592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), md5_288591, 'update')
    # Calling update(args, kwargs) (line 110)
    update_call_result_288604 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), update_288592, *[encode_call_result_288602], **kwargs_288603)
    
    # SSA branch for the else part of an if statement (line 108)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to endswith(...): (line 111)
    # Processing the call arguments (line 111)
    unicode_288607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 23), 'unicode', u'.svg')
    # Processing the call keyword arguments (line 111)
    kwargs_288608 = {}
    # Getting the type of 'path' (line 111)
    path_288605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 9), 'path', False)
    # Obtaining the member 'endswith' of a type (line 111)
    endswith_288606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 9), path_288605, 'endswith')
    # Calling endswith(args, kwargs) (line 111)
    endswith_call_result_288609 = invoke(stypy.reporting.localization.Localization(__file__, 111, 9), endswith_288606, *[unicode_288607], **kwargs_288608)
    
    # Testing the type of an if condition (line 111)
    if_condition_288610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 9), endswith_call_result_288609)
    # Assigning a type to the variable 'if_condition_288610' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 9), 'if_condition_288610', if_condition_288610)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 112, 8))
    
    # 'from matplotlib import checkdep_inkscape' statement (line 112)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_288611 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 112, 8), 'matplotlib')

    if (type(import_288611) is not StypyTypeError):

        if (import_288611 != 'pyd_module'):
            __import__(import_288611)
            sys_modules_288612 = sys.modules[import_288611]
            import_from_module(stypy.reporting.localization.Localization(__file__, 112, 8), 'matplotlib', sys_modules_288612.module_type_store, module_type_store, ['checkdep_inkscape'])
            nest_module(stypy.reporting.localization.Localization(__file__, 112, 8), __file__, sys_modules_288612, sys_modules_288612.module_type_store, module_type_store)
        else:
            from matplotlib import checkdep_inkscape

            import_from_module(stypy.reporting.localization.Localization(__file__, 112, 8), 'matplotlib', None, module_type_store, ['checkdep_inkscape'], [checkdep_inkscape])

    else:
        # Assigning a type to the variable 'matplotlib' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'matplotlib', import_288611)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    # Call to update(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Call to encode(...): (line 113)
    # Processing the call arguments (line 113)
    unicode_288619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 46), 'unicode', u'utf-8')
    # Processing the call keyword arguments (line 113)
    kwargs_288620 = {}
    
    # Call to checkdep_inkscape(...): (line 113)
    # Processing the call keyword arguments (line 113)
    kwargs_288616 = {}
    # Getting the type of 'checkdep_inkscape' (line 113)
    checkdep_inkscape_288615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'checkdep_inkscape', False)
    # Calling checkdep_inkscape(args, kwargs) (line 113)
    checkdep_inkscape_call_result_288617 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), checkdep_inkscape_288615, *[], **kwargs_288616)
    
    # Obtaining the member 'encode' of a type (line 113)
    encode_288618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), checkdep_inkscape_call_result_288617, 'encode')
    # Calling encode(args, kwargs) (line 113)
    encode_call_result_288621 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), encode_288618, *[unicode_288619], **kwargs_288620)
    
    # Processing the call keyword arguments (line 113)
    kwargs_288622 = {}
    # Getting the type of 'md5' (line 113)
    md5_288613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'md5', False)
    # Obtaining the member 'update' of a type (line 113)
    update_288614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), md5_288613, 'update')
    # Calling update(args, kwargs) (line 113)
    update_call_result_288623 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), update_288614, *[encode_call_result_288621], **kwargs_288622)
    
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to hexdigest(...): (line 115)
    # Processing the call keyword arguments (line 115)
    kwargs_288626 = {}
    # Getting the type of 'md5' (line 115)
    md5_288624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'md5', False)
    # Obtaining the member 'hexdigest' of a type (line 115)
    hexdigest_288625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 11), md5_288624, 'hexdigest')
    # Calling hexdigest(args, kwargs) (line 115)
    hexdigest_call_result_288627 = invoke(stypy.reporting.localization.Localization(__file__, 115, 11), hexdigest_288625, *[], **kwargs_288626)
    
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type', hexdigest_call_result_288627)
    
    # ################# End of 'get_file_hash(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_file_hash' in the type store
    # Getting the type of 'stypy_return_type' (line 99)
    stypy_return_type_288628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288628)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_file_hash'
    return stypy_return_type_288628

# Assigning a type to the variable 'get_file_hash' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'get_file_hash', get_file_hash)

@norecursion
def make_external_conversion_command(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'make_external_conversion_command'
    module_type_store = module_type_store.open_function_context('make_external_conversion_command', 118, 0, False)
    
    # Passed parameters checking function
    make_external_conversion_command.stypy_localization = localization
    make_external_conversion_command.stypy_type_of_self = None
    make_external_conversion_command.stypy_type_store = module_type_store
    make_external_conversion_command.stypy_function_name = 'make_external_conversion_command'
    make_external_conversion_command.stypy_param_names_list = ['cmd']
    make_external_conversion_command.stypy_varargs_param_name = None
    make_external_conversion_command.stypy_kwargs_param_name = None
    make_external_conversion_command.stypy_call_defaults = defaults
    make_external_conversion_command.stypy_call_varargs = varargs
    make_external_conversion_command.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_external_conversion_command', ['cmd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_external_conversion_command', localization, ['cmd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_external_conversion_command(...)' code ##################


    @norecursion
    def convert(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert'
        module_type_store = module_type_store.open_function_context('convert', 119, 4, False)
        
        # Passed parameters checking function
        convert.stypy_localization = localization
        convert.stypy_type_of_self = None
        convert.stypy_type_store = module_type_store
        convert.stypy_function_name = 'convert'
        convert.stypy_param_names_list = ['old', 'new']
        convert.stypy_varargs_param_name = None
        convert.stypy_kwargs_param_name = None
        convert.stypy_call_defaults = defaults
        convert.stypy_call_varargs = varargs
        convert.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'convert', ['old', 'new'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert', localization, ['old', 'new'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert(...)' code ##################

        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to cmd(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'old' (line 120)
        old_288630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'old', False)
        # Getting the type of 'new' (line 120)
        new_288631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'new', False)
        # Processing the call keyword arguments (line 120)
        kwargs_288632 = {}
        # Getting the type of 'cmd' (line 120)
        cmd_288629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 18), 'cmd', False)
        # Calling cmd(args, kwargs) (line 120)
        cmd_call_result_288633 = invoke(stypy.reporting.localization.Localization(__file__, 120, 18), cmd_288629, *[old_288630, new_288631], **kwargs_288632)
        
        # Assigning a type to the variable 'cmdline' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'cmdline', cmd_call_result_288633)
        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to Popen(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'cmdline' (line 121)
        cmdline_288636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'cmdline', False)
        # Processing the call keyword arguments (line 121)
        # Getting the type of 'True' (line 121)
        True_288637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 60), 'True', False)
        keyword_288638 = True_288637
        # Getting the type of 'subprocess' (line 122)
        subprocess_288639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'subprocess', False)
        # Obtaining the member 'PIPE' of a type (line 122)
        PIPE_288640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 39), subprocess_288639, 'PIPE')
        keyword_288641 = PIPE_288640
        # Getting the type of 'subprocess' (line 122)
        subprocess_288642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 63), 'subprocess', False)
        # Obtaining the member 'PIPE' of a type (line 122)
        PIPE_288643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 63), subprocess_288642, 'PIPE')
        keyword_288644 = PIPE_288643
        kwargs_288645 = {'stdout': keyword_288641, 'stderr': keyword_288644, 'universal_newlines': keyword_288638}
        # Getting the type of 'subprocess' (line 121)
        subprocess_288634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'subprocess', False)
        # Obtaining the member 'Popen' of a type (line 121)
        Popen_288635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), subprocess_288634, 'Popen')
        # Calling Popen(args, kwargs) (line 121)
        Popen_call_result_288646 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), Popen_288635, *[cmdline_288636], **kwargs_288645)
        
        # Assigning a type to the variable 'pipe' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'pipe', Popen_call_result_288646)
        
        # Assigning a Call to a Tuple (line 123):
        
        # Assigning a Call to a Name:
        
        # Call to communicate(...): (line 123)
        # Processing the call keyword arguments (line 123)
        kwargs_288649 = {}
        # Getting the type of 'pipe' (line 123)
        pipe_288647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'pipe', False)
        # Obtaining the member 'communicate' of a type (line 123)
        communicate_288648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 25), pipe_288647, 'communicate')
        # Calling communicate(args, kwargs) (line 123)
        communicate_call_result_288650 = invoke(stypy.reporting.localization.Localization(__file__, 123, 25), communicate_288648, *[], **kwargs_288649)
        
        # Assigning a type to the variable 'call_assignment_288328' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'call_assignment_288328', communicate_call_result_288650)
        
        # Assigning a Call to a Name (line 123):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_288653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 8), 'int')
        # Processing the call keyword arguments
        kwargs_288654 = {}
        # Getting the type of 'call_assignment_288328' (line 123)
        call_assignment_288328_288651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'call_assignment_288328', False)
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___288652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), call_assignment_288328_288651, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_288655 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___288652, *[int_288653], **kwargs_288654)
        
        # Assigning a type to the variable 'call_assignment_288329' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'call_assignment_288329', getitem___call_result_288655)
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'call_assignment_288329' (line 123)
        call_assignment_288329_288656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'call_assignment_288329')
        # Assigning a type to the variable 'stdout' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stdout', call_assignment_288329_288656)
        
        # Assigning a Call to a Name (line 123):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_288659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 8), 'int')
        # Processing the call keyword arguments
        kwargs_288660 = {}
        # Getting the type of 'call_assignment_288328' (line 123)
        call_assignment_288328_288657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'call_assignment_288328', False)
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___288658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), call_assignment_288328_288657, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_288661 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___288658, *[int_288659], **kwargs_288660)
        
        # Assigning a type to the variable 'call_assignment_288330' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'call_assignment_288330', getitem___call_result_288661)
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'call_assignment_288330' (line 123)
        call_assignment_288330_288662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'call_assignment_288330')
        # Assigning a type to the variable 'stderr' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'stderr', call_assignment_288330_288662)
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to wait(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_288665 = {}
        # Getting the type of 'pipe' (line 124)
        pipe_288663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'pipe', False)
        # Obtaining the member 'wait' of a type (line 124)
        wait_288664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 18), pipe_288663, 'wait')
        # Calling wait(args, kwargs) (line 124)
        wait_call_result_288666 = invoke(stypy.reporting.localization.Localization(__file__, 124, 18), wait_288664, *[], **kwargs_288665)
        
        # Assigning a type to the variable 'errcode' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'errcode', wait_call_result_288666)
        
        
        # Evaluating a boolean operation
        
        
        # Call to exists(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'new' (line 125)
        new_288670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 30), 'new', False)
        # Processing the call keyword arguments (line 125)
        kwargs_288671 = {}
        # Getting the type of 'os' (line 125)
        os_288667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 125)
        path_288668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 15), os_288667, 'path')
        # Obtaining the member 'exists' of a type (line 125)
        exists_288669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 15), path_288668, 'exists')
        # Calling exists(args, kwargs) (line 125)
        exists_call_result_288672 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), exists_288669, *[new_288670], **kwargs_288671)
        
        # Applying the 'not' unary operator (line 125)
        result_not__288673 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), 'not', exists_call_result_288672)
        
        # Getting the type of 'errcode' (line 125)
        errcode_288674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 38), 'errcode')
        # Applying the binary operator 'or' (line 125)
        result_or_keyword_288675 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), 'or', result_not__288673, errcode_288674)
        
        # Testing the type of an if condition (line 125)
        if_condition_288676 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_or_keyword_288675)
        # Assigning a type to the variable 'if_condition_288676' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_288676', if_condition_288676)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 126):
        
        # Assigning a BinOp to a Name (line 126):
        unicode_288677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 18), 'unicode', u'Conversion command failed:\n%s\n')
        
        # Call to join(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'cmdline' (line 126)
        cmdline_288680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 64), 'cmdline', False)
        # Processing the call keyword arguments (line 126)
        kwargs_288681 = {}
        unicode_288678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 55), 'unicode', u' ')
        # Obtaining the member 'join' of a type (line 126)
        join_288679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 55), unicode_288678, 'join')
        # Calling join(args, kwargs) (line 126)
        join_call_result_288682 = invoke(stypy.reporting.localization.Localization(__file__, 126, 55), join_288679, *[cmdline_288680], **kwargs_288681)
        
        # Applying the binary operator '%' (line 126)
        result_mod_288683 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 18), '%', unicode_288677, join_call_result_288682)
        
        # Assigning a type to the variable 'msg' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'msg', result_mod_288683)
        
        # Getting the type of 'stdout' (line 127)
        stdout_288684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'stdout')
        # Testing the type of an if condition (line 127)
        if_condition_288685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 12), stdout_288684)
        # Assigning a type to the variable 'if_condition_288685' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'if_condition_288685', if_condition_288685)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'msg' (line 128)
        msg_288686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'msg')
        unicode_288687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 23), 'unicode', u'Standard output:\n%s\n')
        # Getting the type of 'stdout' (line 128)
        stdout_288688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'stdout')
        # Applying the binary operator '%' (line 128)
        result_mod_288689 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 23), '%', unicode_288687, stdout_288688)
        
        # Applying the binary operator '+=' (line 128)
        result_iadd_288690 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 16), '+=', msg_288686, result_mod_288689)
        # Assigning a type to the variable 'msg' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'msg', result_iadd_288690)
        
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'stderr' (line 129)
        stderr_288691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'stderr')
        # Testing the type of an if condition (line 129)
        if_condition_288692 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 12), stderr_288691)
        # Assigning a type to the variable 'if_condition_288692' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'if_condition_288692', if_condition_288692)
        # SSA begins for if statement (line 129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'msg' (line 130)
        msg_288693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'msg')
        unicode_288694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 23), 'unicode', u'Standard error:\n%s\n')
        # Getting the type of 'stderr' (line 130)
        stderr_288695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 49), 'stderr')
        # Applying the binary operator '%' (line 130)
        result_mod_288696 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 23), '%', unicode_288694, stderr_288695)
        
        # Applying the binary operator '+=' (line 130)
        result_iadd_288697 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 16), '+=', msg_288693, result_mod_288696)
        # Assigning a type to the variable 'msg' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'msg', result_iadd_288697)
        
        # SSA join for if statement (line 129)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to IOError(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'msg' (line 131)
        msg_288699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'msg', False)
        # Processing the call keyword arguments (line 131)
        kwargs_288700 = {}
        # Getting the type of 'IOError' (line 131)
        IOError_288698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 18), 'IOError', False)
        # Calling IOError(args, kwargs) (line 131)
        IOError_call_result_288701 = invoke(stypy.reporting.localization.Localization(__file__, 131, 18), IOError_288698, *[msg_288699], **kwargs_288700)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 131, 12), IOError_call_result_288701, 'raise parameter', BaseException)
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_288702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_288702)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_288702

    # Assigning a type to the variable 'convert' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'convert', convert)
    # Getting the type of 'convert' (line 133)
    convert_288703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'convert')
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type', convert_288703)
    
    # ################# End of 'make_external_conversion_command(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_external_conversion_command' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_288704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288704)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_external_conversion_command'
    return stypy_return_type_288704

# Assigning a type to the variable 'make_external_conversion_command' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'make_external_conversion_command', make_external_conversion_command)

# Assigning a Attribute to a Name (line 137):

# Assigning a Attribute to a Name (line 137):

# Call to compile(...): (line 137)
# Processing the call arguments (line 137)
str_288707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 32), 'str', '[^a-zA-Z0-9_@%+=:,./-]')
# Processing the call keyword arguments (line 137)
kwargs_288708 = {}
# Getting the type of 're' (line 137)
re_288705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 're', False)
# Obtaining the member 'compile' of a type (line 137)
compile_288706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 21), re_288705, 'compile')
# Calling compile(args, kwargs) (line 137)
compile_call_result_288709 = invoke(stypy.reporting.localization.Localization(__file__, 137, 21), compile_288706, *[str_288707], **kwargs_288708)

# Obtaining the member 'search' of a type (line 137)
search_288710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 21), compile_call_result_288709, 'search')
# Assigning a type to the variable '_find_unsafe_bytes' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), '_find_unsafe_bytes', search_288710)

@norecursion
def _shlex_quote_bytes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_shlex_quote_bytes'
    module_type_store = module_type_store.open_function_context('_shlex_quote_bytes', 140, 0, False)
    
    # Passed parameters checking function
    _shlex_quote_bytes.stypy_localization = localization
    _shlex_quote_bytes.stypy_type_of_self = None
    _shlex_quote_bytes.stypy_type_store = module_type_store
    _shlex_quote_bytes.stypy_function_name = '_shlex_quote_bytes'
    _shlex_quote_bytes.stypy_param_names_list = ['b']
    _shlex_quote_bytes.stypy_varargs_param_name = None
    _shlex_quote_bytes.stypy_kwargs_param_name = None
    _shlex_quote_bytes.stypy_call_defaults = defaults
    _shlex_quote_bytes.stypy_call_varargs = varargs
    _shlex_quote_bytes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_shlex_quote_bytes', ['b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_shlex_quote_bytes', localization, ['b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_shlex_quote_bytes(...)' code ##################

    
    
    
    # Call to _find_unsafe_bytes(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'b' (line 141)
    b_288712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 36), 'b', False)
    # Processing the call keyword arguments (line 141)
    kwargs_288713 = {}
    # Getting the type of '_find_unsafe_bytes' (line 141)
    _find_unsafe_bytes_288711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 17), '_find_unsafe_bytes', False)
    # Calling _find_unsafe_bytes(args, kwargs) (line 141)
    _find_unsafe_bytes_call_result_288714 = invoke(stypy.reporting.localization.Localization(__file__, 141, 17), _find_unsafe_bytes_288711, *[b_288712], **kwargs_288713)
    
    # Getting the type of 'None' (line 141)
    None_288715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 42), 'None')
    # Applying the binary operator 'is' (line 141)
    result_is__288716 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 17), 'is', _find_unsafe_bytes_call_result_288714, None_288715)
    
    # Testing the type of an if expression (line 141)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 12), result_is__288716)
    # SSA begins for if expression (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'b' (line 141)
    b_288717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'b')
    # SSA branch for the else part of an if expression (line 141)
    module_type_store.open_ssa_branch('if expression else')
    str_288718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 17), 'str', "'")
    
    # Call to replace(...): (line 142)
    # Processing the call arguments (line 142)
    str_288721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 34), 'str', "'")
    str_288722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 40), 'str', '\'"\'"\'')
    # Processing the call keyword arguments (line 142)
    kwargs_288723 = {}
    # Getting the type of 'b' (line 142)
    b_288719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'b', False)
    # Obtaining the member 'replace' of a type (line 142)
    replace_288720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 24), b_288719, 'replace')
    # Calling replace(args, kwargs) (line 142)
    replace_call_result_288724 = invoke(stypy.reporting.localization.Localization(__file__, 142, 24), replace_288720, *[str_288721, str_288722], **kwargs_288723)
    
    # Applying the binary operator '+' (line 142)
    result_add_288725 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 17), '+', str_288718, replace_call_result_288724)
    
    str_288726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 54), 'str', "'")
    # Applying the binary operator '+' (line 142)
    result_add_288727 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 52), '+', result_add_288725, str_288726)
    
    # SSA join for if expression (line 141)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_288728 = union_type.UnionType.add(b_288717, result_add_288727)
    
    # Assigning a type to the variable 'stypy_return_type' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type', if_exp_288728)
    
    # ################# End of '_shlex_quote_bytes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_shlex_quote_bytes' in the type store
    # Getting the type of 'stypy_return_type' (line 140)
    stypy_return_type_288729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288729)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_shlex_quote_bytes'
    return stypy_return_type_288729

# Assigning a type to the variable '_shlex_quote_bytes' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), '_shlex_quote_bytes', _shlex_quote_bytes)
# Declaration of the '_SVGConverter' class

class _SVGConverter(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SVGConverter.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 147):
        
        # Assigning a Name to a Attribute (line 147):
        # Getting the type of 'None' (line 147)
        None_288730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'None')
        # Getting the type of 'self' (line 147)
        self_288731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member '_proc' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_288731, '_proc', None_288730)
        
        # Call to register(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'self' (line 153)
        self_288734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'self', False)
        # Obtaining the member '__del__' of a type (line 153)
        del___288735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 24), self_288734, '__del__')
        # Processing the call keyword arguments (line 153)
        kwargs_288736 = {}
        # Getting the type of 'atexit' (line 153)
        atexit_288732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'atexit', False)
        # Obtaining the member 'register' of a type (line 153)
        register_288733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), atexit_288732, 'register')
        # Calling register(args, kwargs) (line 153)
        register_call_result_288737 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), register_288733, *[del___288735], **kwargs_288736)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _read_to_prompt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_to_prompt'
        module_type_store = module_type_store.open_function_context('_read_to_prompt', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_localization', localization)
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_function_name', '_SVGConverter._read_to_prompt')
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_param_names_list', [])
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SVGConverter._read_to_prompt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SVGConverter._read_to_prompt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_to_prompt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_to_prompt(...)' code ##################

        unicode_288738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, (-1)), 'unicode', u'Did Inkscape reach the prompt without crashing?\n        ')
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to iter(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Call to partial(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'self' (line 158)
        self_288742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 40), 'self', False)
        # Obtaining the member '_proc' of a type (line 158)
        _proc_288743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 40), self_288742, '_proc')
        # Obtaining the member 'stdout' of a type (line 158)
        stdout_288744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 40), _proc_288743, 'stdout')
        # Obtaining the member 'read' of a type (line 158)
        read_288745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 40), stdout_288744, 'read')
        int_288746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 64), 'int')
        # Processing the call keyword arguments (line 158)
        kwargs_288747 = {}
        # Getting the type of 'functools' (line 158)
        functools_288740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'functools', False)
        # Obtaining the member 'partial' of a type (line 158)
        partial_288741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 22), functools_288740, 'partial')
        # Calling partial(args, kwargs) (line 158)
        partial_call_result_288748 = invoke(stypy.reporting.localization.Localization(__file__, 158, 22), partial_288741, *[read_288745, int_288746], **kwargs_288747)
        
        str_288749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 68), 'str', '')
        # Processing the call keyword arguments (line 158)
        kwargs_288750 = {}
        # Getting the type of 'iter' (line 158)
        iter_288739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), 'iter', False)
        # Calling iter(args, kwargs) (line 158)
        iter_call_result_288751 = invoke(stypy.reporting.localization.Localization(__file__, 158, 17), iter_288739, *[partial_call_result_288748, str_288749], **kwargs_288750)
        
        # Assigning a type to the variable 'stream' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stream', iter_call_result_288751)
        
        # Assigning a Tuple to a Name (line 159):
        
        # Assigning a Tuple to a Name (line 159):
        
        # Obtaining an instance of the builtin type 'tuple' (line 159)
        tuple_288752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 159)
        # Adding element type (line 159)
        str_288753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 18), 'str', '\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 18), tuple_288752, str_288753)
        # Adding element type (line 159)
        str_288754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'str', '>')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 18), tuple_288752, str_288754)
        
        # Assigning a type to the variable 'prompt' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'prompt', tuple_288752)
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to len(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'prompt' (line 160)
        prompt_288756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'prompt', False)
        # Processing the call keyword arguments (line 160)
        kwargs_288757 = {}
        # Getting the type of 'len' (line 160)
        len_288755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'len', False)
        # Calling len(args, kwargs) (line 160)
        len_call_result_288758 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), len_288755, *[prompt_288756], **kwargs_288757)
        
        # Assigning a type to the variable 'n' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'n', len_call_result_288758)
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to tee(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'stream' (line 161)
        stream_288761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'stream', False)
        # Getting the type of 'n' (line 161)
        n_288762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 36), 'n', False)
        # Processing the call keyword arguments (line 161)
        kwargs_288763 = {}
        # Getting the type of 'itertools' (line 161)
        itertools_288759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 14), 'itertools', False)
        # Obtaining the member 'tee' of a type (line 161)
        tee_288760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 14), itertools_288759, 'tee')
        # Calling tee(args, kwargs) (line 161)
        tee_call_result_288764 = invoke(stypy.reporting.localization.Localization(__file__, 161, 14), tee_288760, *[stream_288761, n_288762], **kwargs_288763)
        
        # Assigning a type to the variable 'its' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'its', tee_call_result_288764)
        
        
        # Call to enumerate(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'its' (line 162)
        its_288766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 31), 'its', False)
        # Processing the call keyword arguments (line 162)
        kwargs_288767 = {}
        # Getting the type of 'enumerate' (line 162)
        enumerate_288765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 162)
        enumerate_call_result_288768 = invoke(stypy.reporting.localization.Localization(__file__, 162, 21), enumerate_288765, *[its_288766], **kwargs_288767)
        
        # Testing the type of a for loop iterable (line 162)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 162, 8), enumerate_call_result_288768)
        # Getting the type of the for loop variable (line 162)
        for_loop_var_288769 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 162, 8), enumerate_call_result_288768)
        # Assigning a type to the variable 'i' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), for_loop_var_288769))
        # Assigning a type to the variable 'it' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'it', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), for_loop_var_288769))
        # SSA begins for a for statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to next(...): (line 163)
        # Processing the call arguments (line 163)
        
        # Call to islice(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'it' (line 163)
        it_288773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), 'it', False)
        # Getting the type of 'i' (line 163)
        i_288774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 38), 'i', False)
        # Getting the type of 'i' (line 163)
        i_288775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 41), 'i', False)
        # Processing the call keyword arguments (line 163)
        kwargs_288776 = {}
        # Getting the type of 'itertools' (line 163)
        itertools_288771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 17), 'itertools', False)
        # Obtaining the member 'islice' of a type (line 163)
        islice_288772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 17), itertools_288771, 'islice')
        # Calling islice(args, kwargs) (line 163)
        islice_call_result_288777 = invoke(stypy.reporting.localization.Localization(__file__, 163, 17), islice_288772, *[it_288773, i_288774, i_288775], **kwargs_288776)
        
        # Getting the type of 'None' (line 163)
        None_288778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 45), 'None', False)
        # Processing the call keyword arguments (line 163)
        kwargs_288779 = {}
        # Getting the type of 'next' (line 163)
        next_288770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'next', False)
        # Calling next(args, kwargs) (line 163)
        next_call_result_288780 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), next_288770, *[islice_call_result_288777, None_288778], **kwargs_288779)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'True' (line 164)
        True_288781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 14), 'True')
        # Testing the type of an if condition (line 164)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 8), True_288781)
        # SSA begins for while statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 165):
        
        # Assigning a Call to a Name (line 165):
        
        # Call to tuple(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Call to map(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'next' (line 165)
        next_288784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 31), 'next', False)
        # Getting the type of 'its' (line 165)
        its_288785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 'its', False)
        # Processing the call keyword arguments (line 165)
        kwargs_288786 = {}
        # Getting the type of 'map' (line 165)
        map_288783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'map', False)
        # Calling map(args, kwargs) (line 165)
        map_call_result_288787 = invoke(stypy.reporting.localization.Localization(__file__, 165, 27), map_288783, *[next_288784, its_288785], **kwargs_288786)
        
        # Processing the call keyword arguments (line 165)
        kwargs_288788 = {}
        # Getting the type of 'tuple' (line 165)
        tuple_288782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'tuple', False)
        # Calling tuple(args, kwargs) (line 165)
        tuple_call_result_288789 = invoke(stypy.reporting.localization.Localization(__file__, 165, 21), tuple_288782, *[map_call_result_288787], **kwargs_288788)
        
        # Assigning a type to the variable 'window' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'window', tuple_call_result_288789)
        
        
        
        # Call to len(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'window' (line 166)
        window_288791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 'window', False)
        # Processing the call keyword arguments (line 166)
        kwargs_288792 = {}
        # Getting the type of 'len' (line 166)
        len_288790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'len', False)
        # Calling len(args, kwargs) (line 166)
        len_call_result_288793 = invoke(stypy.reporting.localization.Localization(__file__, 166, 15), len_288790, *[window_288791], **kwargs_288792)
        
        # Getting the type of 'n' (line 166)
        n_288794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 30), 'n')
        # Applying the binary operator '!=' (line 166)
        result_ne_288795 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 15), '!=', len_call_result_288793, n_288794)
        
        # Testing the type of an if condition (line 166)
        if_condition_288796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 12), result_ne_288795)
        # Assigning a type to the variable 'if_condition_288796' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'if_condition_288796', if_condition_288796)
        # SSA begins for if statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 169)
        False_288797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'stypy_return_type', False_288797)
        # SSA join for if statement (line 166)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to poll(...): (line 170)
        # Processing the call keyword arguments (line 170)
        kwargs_288801 = {}
        # Getting the type of 'self' (line 170)
        self_288798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'self', False)
        # Obtaining the member '_proc' of a type (line 170)
        _proc_288799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 15), self_288798, '_proc')
        # Obtaining the member 'poll' of a type (line 170)
        poll_288800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 15), _proc_288799, 'poll')
        # Calling poll(args, kwargs) (line 170)
        poll_call_result_288802 = invoke(stypy.reporting.localization.Localization(__file__, 170, 15), poll_288800, *[], **kwargs_288801)
        
        # Getting the type of 'None' (line 170)
        None_288803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 40), 'None')
        # Applying the binary operator 'isnot' (line 170)
        result_is_not_288804 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 15), 'isnot', poll_call_result_288802, None_288803)
        
        # Testing the type of an if condition (line 170)
        if_condition_288805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 12), result_is_not_288804)
        # Assigning a type to the variable 'if_condition_288805' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'if_condition_288805', if_condition_288805)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 172)
        False_288806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'stypy_return_type', False_288806)
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'window' (line 173)
        window_288807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'window')
        # Getting the type of 'prompt' (line 173)
        prompt_288808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 25), 'prompt')
        # Applying the binary operator '==' (line 173)
        result_eq_288809 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 15), '==', window_288807, prompt_288808)
        
        # Testing the type of an if condition (line 173)
        if_condition_288810 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 12), result_eq_288809)
        # Assigning a type to the variable 'if_condition_288810' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'if_condition_288810', if_condition_288810)
        # SSA begins for if statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 175)
        True_288811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'stypy_return_type', True_288811)
        # SSA join for if statement (line 173)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 164)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_read_to_prompt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_to_prompt' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_288812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_288812)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_to_prompt'
        return stypy_return_type_288812


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SVGConverter.__call__.__dict__.__setitem__('stypy_localization', localization)
        _SVGConverter.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SVGConverter.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SVGConverter.__call__.__dict__.__setitem__('stypy_function_name', '_SVGConverter.__call__')
        _SVGConverter.__call__.__dict__.__setitem__('stypy_param_names_list', ['orig', 'dest'])
        _SVGConverter.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SVGConverter.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SVGConverter.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SVGConverter.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SVGConverter.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SVGConverter.__call__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SVGConverter.__call__', ['orig', 'dest'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['orig', 'dest'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 178)
        self_288813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'self')
        # Obtaining the member '_proc' of a type (line 178)
        _proc_288814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 16), self_288813, '_proc')
        # Applying the 'not' unary operator (line 178)
        result_not__288815 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 12), 'not', _proc_288814)
        
        
        
        # Call to poll(...): (line 179)
        # Processing the call keyword arguments (line 179)
        kwargs_288819 = {}
        # Getting the type of 'self' (line 179)
        self_288816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'self', False)
        # Obtaining the member '_proc' of a type (line 179)
        _proc_288817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 19), self_288816, '_proc')
        # Obtaining the member 'poll' of a type (line 179)
        poll_288818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 19), _proc_288817, 'poll')
        # Calling poll(args, kwargs) (line 179)
        poll_call_result_288820 = invoke(stypy.reporting.localization.Localization(__file__, 179, 19), poll_288818, *[], **kwargs_288819)
        
        # Getting the type of 'None' (line 179)
        None_288821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 44), 'None')
        # Applying the binary operator 'isnot' (line 179)
        result_is_not_288822 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 19), 'isnot', poll_call_result_288820, None_288821)
        
        # Applying the binary operator 'or' (line 178)
        result_or_keyword_288823 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 12), 'or', result_not__288815, result_is_not_288822)
        
        # Testing the type of an if condition (line 178)
        if_condition_288824 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 8), result_or_keyword_288823)
        # Assigning a type to the variable 'if_condition_288824' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'if_condition_288824', if_condition_288824)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 180):
        
        # Assigning a Call to a Name (line 180):
        
        # Call to copy(...): (line 180)
        # Processing the call keyword arguments (line 180)
        kwargs_288828 = {}
        # Getting the type of 'os' (line 180)
        os_288825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 18), 'os', False)
        # Obtaining the member 'environ' of a type (line 180)
        environ_288826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 18), os_288825, 'environ')
        # Obtaining the member 'copy' of a type (line 180)
        copy_288827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 18), environ_288826, 'copy')
        # Calling copy(args, kwargs) (line 180)
        copy_call_result_288829 = invoke(stypy.reporting.localization.Localization(__file__, 180, 18), copy_288827, *[], **kwargs_288828)
        
        # Assigning a type to the variable 'env' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'env', copy_call_result_288829)
        
        # Call to pop(...): (line 186)
        # Processing the call arguments (line 186)
        unicode_288832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 20), 'unicode', u'DISPLAY')
        # Getting the type of 'None' (line 186)
        None_288833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 31), 'None', False)
        # Processing the call keyword arguments (line 186)
        kwargs_288834 = {}
        # Getting the type of 'env' (line 186)
        env_288830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'env', False)
        # Obtaining the member 'pop' of a type (line 186)
        pop_288831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), env_288830, 'pop')
        # Calling pop(args, kwargs) (line 186)
        pop_call_result_288835 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), pop_288831, *[unicode_288832, None_288833], **kwargs_288834)
        
        
        # Assigning a Attribute to a Subscript (line 189):
        
        # Assigning a Attribute to a Subscript (line 189):
        # Getting the type of 'os' (line 189)
        os_288836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 47), 'os')
        # Obtaining the member 'devnull' of a type (line 189)
        devnull_288837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 47), os_288836, 'devnull')
        # Getting the type of 'env' (line 189)
        env_288838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'env')
        
        # Call to str(...): (line 189)
        # Processing the call arguments (line 189)
        unicode_288840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 20), 'unicode', u'INKSCAPE_PROFILE_DIR')
        # Processing the call keyword arguments (line 189)
        kwargs_288841 = {}
        # Getting the type of 'str' (line 189)
        str_288839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'str', False)
        # Calling str(args, kwargs) (line 189)
        str_call_result_288842 = invoke(stypy.reporting.localization.Localization(__file__, 189, 16), str_288839, *[unicode_288840], **kwargs_288841)
        
        # Storing an element on a container (line 189)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), env_288838, (str_call_result_288842, devnull_288837))
        
        # Assigning a Call to a Attribute (line 194):
        
        # Assigning a Call to a Attribute (line 194):
        
        # Call to TemporaryFile(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_288844 = {}
        # Getting the type of 'TemporaryFile' (line 194)
        TemporaryFile_288843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'TemporaryFile', False)
        # Calling TemporaryFile(args, kwargs) (line 194)
        TemporaryFile_call_result_288845 = invoke(stypy.reporting.localization.Localization(__file__, 194, 27), TemporaryFile_288843, *[], **kwargs_288844)
        
        # Getting the type of 'self' (line 194)
        self_288846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'self')
        # Setting the type of the member '_stderr' of a type (line 194)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), self_288846, '_stderr', TemporaryFile_call_result_288845)
        
        # Assigning a Call to a Attribute (line 195):
        
        # Assigning a Call to a Attribute (line 195):
        
        # Call to Popen(...): (line 195)
        # Processing the call arguments (line 195)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_288849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        
        # Call to str(...): (line 196)
        # Processing the call arguments (line 196)
        unicode_288851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 21), 'unicode', u'inkscape')
        # Processing the call keyword arguments (line 196)
        kwargs_288852 = {}
        # Getting the type of 'str' (line 196)
        str_288850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'str', False)
        # Calling str(args, kwargs) (line 196)
        str_call_result_288853 = invoke(stypy.reporting.localization.Localization(__file__, 196, 17), str_288850, *[unicode_288851], **kwargs_288852)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 16), list_288849, str_call_result_288853)
        # Adding element type (line 196)
        unicode_288854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 34), 'unicode', u'--without-gui')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 16), list_288849, unicode_288854)
        # Adding element type (line 196)
        unicode_288855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 51), 'unicode', u'--shell')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 16), list_288849, unicode_288855)
        
        # Processing the call keyword arguments (line 195)
        # Getting the type of 'subprocess' (line 197)
        subprocess_288856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 22), 'subprocess', False)
        # Obtaining the member 'PIPE' of a type (line 197)
        PIPE_288857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 22), subprocess_288856, 'PIPE')
        keyword_288858 = PIPE_288857
        # Getting the type of 'subprocess' (line 197)
        subprocess_288859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 46), 'subprocess', False)
        # Obtaining the member 'PIPE' of a type (line 197)
        PIPE_288860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 46), subprocess_288859, 'PIPE')
        keyword_288861 = PIPE_288860
        # Getting the type of 'self' (line 198)
        self_288862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'self', False)
        # Obtaining the member '_stderr' of a type (line 198)
        _stderr_288863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 23), self_288862, '_stderr')
        keyword_288864 = _stderr_288863
        # Getting the type of 'env' (line 198)
        env_288865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 41), 'env', False)
        keyword_288866 = env_288865
        kwargs_288867 = {'stdin': keyword_288858, 'env': keyword_288866, 'stderr': keyword_288864, 'stdout': keyword_288861}
        # Getting the type of 'subprocess' (line 195)
        subprocess_288847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 25), 'subprocess', False)
        # Obtaining the member 'Popen' of a type (line 195)
        Popen_288848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 25), subprocess_288847, 'Popen')
        # Calling Popen(args, kwargs) (line 195)
        Popen_call_result_288868 = invoke(stypy.reporting.localization.Localization(__file__, 195, 25), Popen_288848, *[list_288849], **kwargs_288867)
        
        # Getting the type of 'self' (line 195)
        self_288869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'self')
        # Setting the type of the member '_proc' of a type (line 195)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), self_288869, '_proc', Popen_call_result_288868)
        
        
        
        # Call to _read_to_prompt(...): (line 199)
        # Processing the call keyword arguments (line 199)
        kwargs_288872 = {}
        # Getting the type of 'self' (line 199)
        self_288870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'self', False)
        # Obtaining the member '_read_to_prompt' of a type (line 199)
        _read_to_prompt_288871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 19), self_288870, '_read_to_prompt')
        # Calling _read_to_prompt(args, kwargs) (line 199)
        _read_to_prompt_call_result_288873 = invoke(stypy.reporting.localization.Localization(__file__, 199, 19), _read_to_prompt_288871, *[], **kwargs_288872)
        
        # Applying the 'not' unary operator (line 199)
        result_not__288874 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 15), 'not', _read_to_prompt_call_result_288873)
        
        # Testing the type of an if condition (line 199)
        if_condition_288875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 12), result_not__288874)
        # Assigning a type to the variable 'if_condition_288875' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'if_condition_288875', if_condition_288875)
        # SSA begins for if statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to OSError(...): (line 200)
        # Processing the call arguments (line 200)
        unicode_288877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 30), 'unicode', u'Failed to start Inkscape')
        # Processing the call keyword arguments (line 200)
        kwargs_288878 = {}
        # Getting the type of 'OSError' (line 200)
        OSError_288876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'OSError', False)
        # Calling OSError(args, kwargs) (line 200)
        OSError_call_result_288879 = invoke(stypy.reporting.localization.Localization(__file__, 200, 22), OSError_288876, *[unicode_288877], **kwargs_288878)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 200, 16), OSError_call_result_288879, 'raise parameter', BaseException)
        # SSA join for if statement (line 199)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Name (line 203):
        
        # Assigning a Attribute to a Name (line 203):
        # Getting the type of 'os' (line 203)
        os_288880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'os')
        # Obtaining the member 'fsencode' of a type (line 203)
        fsencode_288881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 23), os_288880, 'fsencode')
        # Assigning a type to the variable 'fsencode' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'fsencode', fsencode_288881)
        # SSA branch for the except part of a try statement (line 202)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 202)
        module_type_store.open_ssa_branch('except')

        @norecursion
        def fsencode(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fsencode'
            module_type_store = module_type_store.open_function_context('fsencode', 205, 12, False)
            
            # Passed parameters checking function
            fsencode.stypy_localization = localization
            fsencode.stypy_type_of_self = None
            fsencode.stypy_type_store = module_type_store
            fsencode.stypy_function_name = 'fsencode'
            fsencode.stypy_param_names_list = ['s']
            fsencode.stypy_varargs_param_name = None
            fsencode.stypy_kwargs_param_name = None
            fsencode.stypy_call_defaults = defaults
            fsencode.stypy_call_varargs = varargs
            fsencode.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fsencode', ['s'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fsencode', localization, ['s'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fsencode(...)' code ##################

            
            # Call to encode(...): (line 206)
            # Processing the call arguments (line 206)
            
            # Call to getfilesystemencoding(...): (line 206)
            # Processing the call keyword arguments (line 206)
            kwargs_288886 = {}
            # Getting the type of 'sys' (line 206)
            sys_288884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 32), 'sys', False)
            # Obtaining the member 'getfilesystemencoding' of a type (line 206)
            getfilesystemencoding_288885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 32), sys_288884, 'getfilesystemencoding')
            # Calling getfilesystemencoding(args, kwargs) (line 206)
            getfilesystemencoding_call_result_288887 = invoke(stypy.reporting.localization.Localization(__file__, 206, 32), getfilesystemencoding_288885, *[], **kwargs_288886)
            
            # Processing the call keyword arguments (line 206)
            kwargs_288888 = {}
            # Getting the type of 's' (line 206)
            s_288882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 's', False)
            # Obtaining the member 'encode' of a type (line 206)
            encode_288883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 23), s_288882, 'encode')
            # Calling encode(args, kwargs) (line 206)
            encode_call_result_288889 = invoke(stypy.reporting.localization.Localization(__file__, 206, 23), encode_288883, *[getfilesystemencoding_call_result_288887], **kwargs_288888)
            
            # Assigning a type to the variable 'stypy_return_type' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'stypy_return_type', encode_call_result_288889)
            
            # ################# End of 'fsencode(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fsencode' in the type store
            # Getting the type of 'stypy_return_type' (line 205)
            stypy_return_type_288890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_288890)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fsencode'
            return stypy_return_type_288890

        # Assigning a type to the variable 'fsencode' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'fsencode', fsencode)
        # SSA join for try-except statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 210):
        
        # Assigning a Call to a Name:
        
        # Call to map(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of '_shlex_quote_bytes' (line 210)
        _shlex_quote_bytes_288892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 29), '_shlex_quote_bytes', False)
        
        # Call to map(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'fsencode' (line 210)
        fsencode_288894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 53), 'fsencode', False)
        
        # Obtaining an instance of the builtin type 'list' (line 210)
        list_288895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 63), 'list')
        # Adding type elements to the builtin type 'list' instance (line 210)
        # Adding element type (line 210)
        # Getting the type of 'orig' (line 210)
        orig_288896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 64), 'orig', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 63), list_288895, orig_288896)
        # Adding element type (line 210)
        # Getting the type of 'dest' (line 210)
        dest_288897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 70), 'dest', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 63), list_288895, dest_288897)
        
        # Processing the call keyword arguments (line 210)
        kwargs_288898 = {}
        # Getting the type of 'map' (line 210)
        map_288893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 49), 'map', False)
        # Calling map(args, kwargs) (line 210)
        map_call_result_288899 = invoke(stypy.reporting.localization.Localization(__file__, 210, 49), map_288893, *[fsencode_288894, list_288895], **kwargs_288898)
        
        # Processing the call keyword arguments (line 210)
        kwargs_288900 = {}
        # Getting the type of 'map' (line 210)
        map_288891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 25), 'map', False)
        # Calling map(args, kwargs) (line 210)
        map_call_result_288901 = invoke(stypy.reporting.localization.Localization(__file__, 210, 25), map_288891, *[_shlex_quote_bytes_288892, map_call_result_288899], **kwargs_288900)
        
        # Assigning a type to the variable 'call_assignment_288331' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'call_assignment_288331', map_call_result_288901)
        
        # Assigning a Call to a Name (line 210):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_288904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 8), 'int')
        # Processing the call keyword arguments
        kwargs_288905 = {}
        # Getting the type of 'call_assignment_288331' (line 210)
        call_assignment_288331_288902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'call_assignment_288331', False)
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___288903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), call_assignment_288331_288902, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_288906 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___288903, *[int_288904], **kwargs_288905)
        
        # Assigning a type to the variable 'call_assignment_288332' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'call_assignment_288332', getitem___call_result_288906)
        
        # Assigning a Name to a Name (line 210):
        # Getting the type of 'call_assignment_288332' (line 210)
        call_assignment_288332_288907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'call_assignment_288332')
        # Assigning a type to the variable 'orig_b' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'orig_b', call_assignment_288332_288907)
        
        # Assigning a Call to a Name (line 210):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_288910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 8), 'int')
        # Processing the call keyword arguments
        kwargs_288911 = {}
        # Getting the type of 'call_assignment_288331' (line 210)
        call_assignment_288331_288908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'call_assignment_288331', False)
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___288909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), call_assignment_288331_288908, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_288912 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___288909, *[int_288910], **kwargs_288911)
        
        # Assigning a type to the variable 'call_assignment_288333' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'call_assignment_288333', getitem___call_result_288912)
        
        # Assigning a Name to a Name (line 210):
        # Getting the type of 'call_assignment_288333' (line 210)
        call_assignment_288333_288913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'call_assignment_288333')
        # Assigning a type to the variable 'dest_b' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'dest_b', call_assignment_288333_288913)
        
        
        # Evaluating a boolean operation
        
        str_288914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 11), 'str', '\n')
        # Getting the type of 'orig_b' (line 211)
        orig_b_288915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'orig_b')
        # Applying the binary operator 'in' (line 211)
        result_contains_288916 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 11), 'in', str_288914, orig_b_288915)
        
        
        str_288917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 30), 'str', '\n')
        # Getting the type of 'dest_b' (line 211)
        dest_b_288918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 39), 'dest_b')
        # Applying the binary operator 'in' (line 211)
        result_contains_288919 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 30), 'in', str_288917, dest_b_288918)
        
        # Applying the binary operator 'or' (line 211)
        result_or_keyword_288920 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 11), 'or', result_contains_288916, result_contains_288919)
        
        # Testing the type of an if condition (line 211)
        if_condition_288921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), result_or_keyword_288920)
        # Assigning a type to the variable 'if_condition_288921' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_288921', if_condition_288921)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to (...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'orig' (line 217)
        orig_288936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 66), 'orig', False)
        # Getting the type of 'dest' (line 217)
        dest_288937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 72), 'dest', False)
        # Processing the call keyword arguments (line 216)
        kwargs_288938 = {}
        
        # Call to make_external_conversion_command(...): (line 216)
        # Processing the call arguments (line 216)

        @norecursion
        def _stypy_temp_lambda_146(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_146'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_146', 216, 52, True)
            # Passed parameters checking function
            _stypy_temp_lambda_146.stypy_localization = localization
            _stypy_temp_lambda_146.stypy_type_of_self = None
            _stypy_temp_lambda_146.stypy_type_store = module_type_store
            _stypy_temp_lambda_146.stypy_function_name = '_stypy_temp_lambda_146'
            _stypy_temp_lambda_146.stypy_param_names_list = ['old', 'new']
            _stypy_temp_lambda_146.stypy_varargs_param_name = None
            _stypy_temp_lambda_146.stypy_kwargs_param_name = None
            _stypy_temp_lambda_146.stypy_call_defaults = defaults
            _stypy_temp_lambda_146.stypy_call_varargs = varargs
            _stypy_temp_lambda_146.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_146', ['old', 'new'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_146', ['old', 'new'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 216)
            list_288923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 69), 'list')
            # Adding type elements to the builtin type 'list' instance (line 216)
            # Adding element type (line 216)
            
            # Call to str(...): (line 217)
            # Processing the call arguments (line 217)
            unicode_288925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 20), 'unicode', u'inkscape')
            # Processing the call keyword arguments (line 217)
            kwargs_288926 = {}
            # Getting the type of 'str' (line 217)
            str_288924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'str', False)
            # Calling str(args, kwargs) (line 217)
            str_call_result_288927 = invoke(stypy.reporting.localization.Localization(__file__, 217, 16), str_288924, *[unicode_288925], **kwargs_288926)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 69), list_288923, str_call_result_288927)
            # Adding element type (line 216)
            unicode_288928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 33), 'unicode', u'-z')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 69), list_288923, unicode_288928)
            # Adding element type (line 216)
            # Getting the type of 'old' (line 217)
            old_288929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 39), 'old', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 69), list_288923, old_288929)
            # Adding element type (line 216)
            unicode_288930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 44), 'unicode', u'--export-png')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 69), list_288923, unicode_288930)
            # Adding element type (line 216)
            # Getting the type of 'new' (line 217)
            new_288931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 60), 'new', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 69), list_288923, new_288931)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 52), 'stypy_return_type', list_288923)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_146' in the type store
            # Getting the type of 'stypy_return_type' (line 216)
            stypy_return_type_288932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 52), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_288932)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_146'
            return stypy_return_type_288932

        # Assigning a type to the variable '_stypy_temp_lambda_146' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 52), '_stypy_temp_lambda_146', _stypy_temp_lambda_146)
        # Getting the type of '_stypy_temp_lambda_146' (line 216)
        _stypy_temp_lambda_146_288933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 52), '_stypy_temp_lambda_146')
        # Processing the call keyword arguments (line 216)
        kwargs_288934 = {}
        # Getting the type of 'make_external_conversion_command' (line 216)
        make_external_conversion_command_288922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 19), 'make_external_conversion_command', False)
        # Calling make_external_conversion_command(args, kwargs) (line 216)
        make_external_conversion_command_call_result_288935 = invoke(stypy.reporting.localization.Localization(__file__, 216, 19), make_external_conversion_command_288922, *[_stypy_temp_lambda_146_288933], **kwargs_288934)
        
        # Calling (args, kwargs) (line 216)
        _call_result_288939 = invoke(stypy.reporting.localization.Localization(__file__, 216, 19), make_external_conversion_command_call_result_288935, *[orig_288936, dest_288937], **kwargs_288938)
        
        # Assigning a type to the variable 'stypy_return_type' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'stypy_return_type', _call_result_288939)
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'orig_b' (line 218)
        orig_b_288944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 31), 'orig_b', False)
        str_288945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 40), 'str', ' --export-png=')
        # Applying the binary operator '+' (line 218)
        result_add_288946 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 31), '+', orig_b_288944, str_288945)
        
        # Getting the type of 'dest_b' (line 218)
        dest_b_288947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 60), 'dest_b', False)
        # Applying the binary operator '+' (line 218)
        result_add_288948 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 58), '+', result_add_288946, dest_b_288947)
        
        str_288949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 69), 'str', '\n')
        # Applying the binary operator '+' (line 218)
        result_add_288950 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 67), '+', result_add_288948, str_288949)
        
        # Processing the call keyword arguments (line 218)
        kwargs_288951 = {}
        # Getting the type of 'self' (line 218)
        self_288940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'self', False)
        # Obtaining the member '_proc' of a type (line 218)
        _proc_288941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), self_288940, '_proc')
        # Obtaining the member 'stdin' of a type (line 218)
        stdin_288942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), _proc_288941, 'stdin')
        # Obtaining the member 'write' of a type (line 218)
        write_288943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), stdin_288942, 'write')
        # Calling write(args, kwargs) (line 218)
        write_call_result_288952 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), write_288943, *[result_add_288950], **kwargs_288951)
        
        
        # Call to flush(...): (line 219)
        # Processing the call keyword arguments (line 219)
        kwargs_288957 = {}
        # Getting the type of 'self' (line 219)
        self_288953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self', False)
        # Obtaining the member '_proc' of a type (line 219)
        _proc_288954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_288953, '_proc')
        # Obtaining the member 'stdin' of a type (line 219)
        stdin_288955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), _proc_288954, 'stdin')
        # Obtaining the member 'flush' of a type (line 219)
        flush_288956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), stdin_288955, 'flush')
        # Calling flush(args, kwargs) (line 219)
        flush_call_result_288958 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), flush_288956, *[], **kwargs_288957)
        
        
        
        
        # Call to _read_to_prompt(...): (line 220)
        # Processing the call keyword arguments (line 220)
        kwargs_288961 = {}
        # Getting the type of 'self' (line 220)
        self_288959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 'self', False)
        # Obtaining the member '_read_to_prompt' of a type (line 220)
        _read_to_prompt_288960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 15), self_288959, '_read_to_prompt')
        # Calling _read_to_prompt(args, kwargs) (line 220)
        _read_to_prompt_call_result_288962 = invoke(stypy.reporting.localization.Localization(__file__, 220, 15), _read_to_prompt_288960, *[], **kwargs_288961)
        
        # Applying the 'not' unary operator (line 220)
        result_not__288963 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 11), 'not', _read_to_prompt_call_result_288962)
        
        # Testing the type of an if condition (line 220)
        if_condition_288964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 8), result_not__288963)
        # Assigning a type to the variable 'if_condition_288964' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'if_condition_288964', if_condition_288964)
        # SSA begins for if statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to seek(...): (line 225)
        # Processing the call arguments (line 225)
        int_288968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 30), 'int')
        # Processing the call keyword arguments (line 225)
        kwargs_288969 = {}
        # Getting the type of 'self' (line 225)
        self_288965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'self', False)
        # Obtaining the member '_stderr' of a type (line 225)
        _stderr_288966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), self_288965, '_stderr')
        # Obtaining the member 'seek' of a type (line 225)
        seek_288967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), _stderr_288966, 'seek')
        # Calling seek(args, kwargs) (line 225)
        seek_call_result_288970 = invoke(stypy.reporting.localization.Localization(__file__, 225, 12), seek_288967, *[int_288968], **kwargs_288969)
        
        
        # Call to ImageComparisonFailure(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Call to decode(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Call to getfilesystemencoding(...): (line 228)
        # Processing the call keyword arguments (line 228)
        kwargs_288980 = {}
        # Getting the type of 'sys' (line 228)
        sys_288978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'sys', False)
        # Obtaining the member 'getfilesystemencoding' of a type (line 228)
        getfilesystemencoding_288979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 20), sys_288978, 'getfilesystemencoding')
        # Calling getfilesystemencoding(args, kwargs) (line 228)
        getfilesystemencoding_call_result_288981 = invoke(stypy.reporting.localization.Localization(__file__, 228, 20), getfilesystemencoding_288979, *[], **kwargs_288980)
        
        unicode_288982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 49), 'unicode', u'replace')
        # Processing the call keyword arguments (line 227)
        kwargs_288983 = {}
        
        # Call to read(...): (line 227)
        # Processing the call keyword arguments (line 227)
        kwargs_288975 = {}
        # Getting the type of 'self' (line 227)
        self_288972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'self', False)
        # Obtaining the member '_stderr' of a type (line 227)
        _stderr_288973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), self_288972, '_stderr')
        # Obtaining the member 'read' of a type (line 227)
        read_288974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), _stderr_288973, 'read')
        # Calling read(args, kwargs) (line 227)
        read_call_result_288976 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), read_288974, *[], **kwargs_288975)
        
        # Obtaining the member 'decode' of a type (line 227)
        decode_288977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), read_call_result_288976, 'decode')
        # Calling decode(args, kwargs) (line 227)
        decode_call_result_288984 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), decode_288977, *[getfilesystemencoding_call_result_288981, unicode_288982], **kwargs_288983)
        
        # Processing the call keyword arguments (line 226)
        kwargs_288985 = {}
        # Getting the type of 'ImageComparisonFailure' (line 226)
        ImageComparisonFailure_288971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 'ImageComparisonFailure', False)
        # Calling ImageComparisonFailure(args, kwargs) (line 226)
        ImageComparisonFailure_call_result_288986 = invoke(stypy.reporting.localization.Localization(__file__, 226, 18), ImageComparisonFailure_288971, *[decode_call_result_288984], **kwargs_288985)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 226, 12), ImageComparisonFailure_call_result_288986, 'raise parameter', BaseException)
        # SSA join for if statement (line 220)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_288987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_288987)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_288987


    @norecursion
    def __del__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__del__'
        module_type_store = module_type_store.open_function_context('__del__', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SVGConverter.__del__.__dict__.__setitem__('stypy_localization', localization)
        _SVGConverter.__del__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SVGConverter.__del__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SVGConverter.__del__.__dict__.__setitem__('stypy_function_name', '_SVGConverter.__del__')
        _SVGConverter.__del__.__dict__.__setitem__('stypy_param_names_list', [])
        _SVGConverter.__del__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SVGConverter.__del__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SVGConverter.__del__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SVGConverter.__del__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SVGConverter.__del__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SVGConverter.__del__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SVGConverter.__del__', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 231)
        self_288988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'self')
        # Obtaining the member '_proc' of a type (line 231)
        _proc_288989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 11), self_288988, '_proc')
        # Testing the type of an if condition (line 231)
        if_condition_288990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 8), _proc_288989)
        # Assigning a type to the variable 'if_condition_288990' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'if_condition_288990', if_condition_288990)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 232)
        
        # Call to poll(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_288994 = {}
        # Getting the type of 'self' (line 232)
        self_288991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'self', False)
        # Obtaining the member '_proc' of a type (line 232)
        _proc_288992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), self_288991, '_proc')
        # Obtaining the member 'poll' of a type (line 232)
        poll_288993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), _proc_288992, 'poll')
        # Calling poll(args, kwargs) (line 232)
        poll_call_result_288995 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), poll_288993, *[], **kwargs_288994)
        
        # Getting the type of 'None' (line 232)
        None_288996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 36), 'None')
        
        (may_be_288997, more_types_in_union_288998) = may_be_none(poll_call_result_288995, None_288996)

        if may_be_288997:

            if more_types_in_union_288998:
                # Runtime conditional SSA (line 232)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to communicate(...): (line 233)
            # Processing the call arguments (line 233)
            str_289002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 39), 'str', 'quit\n')
            # Processing the call keyword arguments (line 233)
            kwargs_289003 = {}
            # Getting the type of 'self' (line 233)
            self_288999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'self', False)
            # Obtaining the member '_proc' of a type (line 233)
            _proc_289000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), self_288999, '_proc')
            # Obtaining the member 'communicate' of a type (line 233)
            communicate_289001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), _proc_289000, 'communicate')
            # Calling communicate(args, kwargs) (line 233)
            communicate_call_result_289004 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), communicate_289001, *[str_289002], **kwargs_289003)
            
            
            # Call to wait(...): (line 234)
            # Processing the call keyword arguments (line 234)
            kwargs_289008 = {}
            # Getting the type of 'self' (line 234)
            self_289005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'self', False)
            # Obtaining the member '_proc' of a type (line 234)
            _proc_289006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 16), self_289005, '_proc')
            # Obtaining the member 'wait' of a type (line 234)
            wait_289007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 16), _proc_289006, 'wait')
            # Calling wait(args, kwargs) (line 234)
            wait_call_result_289009 = invoke(stypy.reporting.localization.Localization(__file__, 234, 16), wait_289007, *[], **kwargs_289008)
            

            if more_types_in_union_288998:
                # SSA join for if statement (line 232)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to close(...): (line 235)
        # Processing the call keyword arguments (line 235)
        kwargs_289014 = {}
        # Getting the type of 'self' (line 235)
        self_289010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'self', False)
        # Obtaining the member '_proc' of a type (line 235)
        _proc_289011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), self_289010, '_proc')
        # Obtaining the member 'stdin' of a type (line 235)
        stdin_289012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), _proc_289011, 'stdin')
        # Obtaining the member 'close' of a type (line 235)
        close_289013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), stdin_289012, 'close')
        # Calling close(args, kwargs) (line 235)
        close_call_result_289015 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), close_289013, *[], **kwargs_289014)
        
        
        # Call to close(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_289020 = {}
        # Getting the type of 'self' (line 236)
        self_289016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'self', False)
        # Obtaining the member '_proc' of a type (line 236)
        _proc_289017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), self_289016, '_proc')
        # Obtaining the member 'stdout' of a type (line 236)
        stdout_289018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), _proc_289017, 'stdout')
        # Obtaining the member 'close' of a type (line 236)
        close_289019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), stdout_289018, 'close')
        # Calling close(args, kwargs) (line 236)
        close_call_result_289021 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), close_289019, *[], **kwargs_289020)
        
        
        # Call to close(...): (line 237)
        # Processing the call keyword arguments (line 237)
        kwargs_289025 = {}
        # Getting the type of 'self' (line 237)
        self_289022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self', False)
        # Obtaining the member '_stderr' of a type (line 237)
        _stderr_289023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), self_289022, '_stderr')
        # Obtaining the member 'close' of a type (line 237)
        close_289024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), _stderr_289023, 'close')
        # Calling close(args, kwargs) (line 237)
        close_call_result_289026 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), close_289024, *[], **kwargs_289025)
        
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__del__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__del__' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_289027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_289027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__del__'
        return stypy_return_type_289027


# Assigning a type to the variable '_SVGConverter' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), '_SVGConverter', _SVGConverter)

@norecursion
def _update_converter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_update_converter'
    module_type_store = module_type_store.open_function_context('_update_converter', 240, 0, False)
    
    # Passed parameters checking function
    _update_converter.stypy_localization = localization
    _update_converter.stypy_type_of_self = None
    _update_converter.stypy_type_store = module_type_store
    _update_converter.stypy_function_name = '_update_converter'
    _update_converter.stypy_param_names_list = []
    _update_converter.stypy_varargs_param_name = None
    _update_converter.stypy_kwargs_param_name = None
    _update_converter.stypy_call_defaults = defaults
    _update_converter.stypy_call_varargs = varargs
    _update_converter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_update_converter', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_update_converter', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_update_converter(...)' code ##################

    
    # Assigning a Call to a Tuple (line 241):
    
    # Assigning a Call to a Name:
    
    # Call to checkdep_ghostscript(...): (line 241)
    # Processing the call keyword arguments (line 241)
    kwargs_289030 = {}
    # Getting the type of 'matplotlib' (line 241)
    matplotlib_289028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'matplotlib', False)
    # Obtaining the member 'checkdep_ghostscript' of a type (line 241)
    checkdep_ghostscript_289029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 15), matplotlib_289028, 'checkdep_ghostscript')
    # Calling checkdep_ghostscript(args, kwargs) (line 241)
    checkdep_ghostscript_call_result_289031 = invoke(stypy.reporting.localization.Localization(__file__, 241, 15), checkdep_ghostscript_289029, *[], **kwargs_289030)
    
    # Assigning a type to the variable 'call_assignment_288334' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'call_assignment_288334', checkdep_ghostscript_call_result_289031)
    
    # Assigning a Call to a Name (line 241):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_289034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 4), 'int')
    # Processing the call keyword arguments
    kwargs_289035 = {}
    # Getting the type of 'call_assignment_288334' (line 241)
    call_assignment_288334_289032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'call_assignment_288334', False)
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___289033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 4), call_assignment_288334_289032, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_289036 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289033, *[int_289034], **kwargs_289035)
    
    # Assigning a type to the variable 'call_assignment_288335' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'call_assignment_288335', getitem___call_result_289036)
    
    # Assigning a Name to a Name (line 241):
    # Getting the type of 'call_assignment_288335' (line 241)
    call_assignment_288335_289037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'call_assignment_288335')
    # Assigning a type to the variable 'gs' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'gs', call_assignment_288335_289037)
    
    # Assigning a Call to a Name (line 241):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_289040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 4), 'int')
    # Processing the call keyword arguments
    kwargs_289041 = {}
    # Getting the type of 'call_assignment_288334' (line 241)
    call_assignment_288334_289038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'call_assignment_288334', False)
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___289039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 4), call_assignment_288334_289038, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_289042 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289039, *[int_289040], **kwargs_289041)
    
    # Assigning a type to the variable 'call_assignment_288336' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'call_assignment_288336', getitem___call_result_289042)
    
    # Assigning a Name to a Name (line 241):
    # Getting the type of 'call_assignment_288336' (line 241)
    call_assignment_288336_289043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'call_assignment_288336')
    # Assigning a type to the variable 'gs_v' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'gs_v', call_assignment_288336_289043)
    
    # Type idiom detected: calculating its left and rigth part (line 242)
    # Getting the type of 'gs_v' (line 242)
    gs_v_289044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'gs_v')
    # Getting the type of 'None' (line 242)
    None_289045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'None')
    
    (may_be_289046, more_types_in_union_289047) = may_not_be_none(gs_v_289044, None_289045)

    if may_be_289046:

        if more_types_in_union_289047:
            # Runtime conditional SSA (line 242)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def cmd(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cmd'
            module_type_store = module_type_store.open_function_context('cmd', 243, 8, False)
            
            # Passed parameters checking function
            cmd.stypy_localization = localization
            cmd.stypy_type_of_self = None
            cmd.stypy_type_store = module_type_store
            cmd.stypy_function_name = 'cmd'
            cmd.stypy_param_names_list = ['old', 'new']
            cmd.stypy_varargs_param_name = None
            cmd.stypy_kwargs_param_name = None
            cmd.stypy_call_defaults = defaults
            cmd.stypy_call_varargs = varargs
            cmd.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cmd', ['old', 'new'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cmd', localization, ['old', 'new'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cmd(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 244)
            list_289048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 244)
            # Adding element type (line 244)
            
            # Call to str(...): (line 244)
            # Processing the call arguments (line 244)
            # Getting the type of 'gs' (line 244)
            gs_289050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'gs', False)
            # Processing the call keyword arguments (line 244)
            kwargs_289051 = {}
            # Getting the type of 'str' (line 244)
            str_289049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'str', False)
            # Calling str(args, kwargs) (line 244)
            str_call_result_289052 = invoke(stypy.reporting.localization.Localization(__file__, 244, 20), str_289049, *[gs_289050], **kwargs_289051)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 19), list_289048, str_call_result_289052)
            # Adding element type (line 244)
            unicode_289053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 29), 'unicode', u'-q')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 19), list_289048, unicode_289053)
            # Adding element type (line 244)
            unicode_289054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 35), 'unicode', u'-sDEVICE=png16m')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 19), list_289048, unicode_289054)
            # Adding element type (line 244)
            unicode_289055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 54), 'unicode', u'-dNOPAUSE')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 19), list_289048, unicode_289055)
            # Adding element type (line 244)
            unicode_289056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 67), 'unicode', u'-dBATCH')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 19), list_289048, unicode_289056)
            # Adding element type (line 244)
            unicode_289057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 13), 'unicode', u'-sOutputFile=')
            # Getting the type of 'new' (line 245)
            new_289058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 31), 'new')
            # Applying the binary operator '+' (line 245)
            result_add_289059 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 13), '+', unicode_289057, new_289058)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 19), list_289048, result_add_289059)
            # Adding element type (line 244)
            # Getting the type of 'old' (line 245)
            old_289060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 36), 'old')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 19), list_289048, old_289060)
            
            # Assigning a type to the variable 'stypy_return_type' (line 244)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'stypy_return_type', list_289048)
            
            # ################# End of 'cmd(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cmd' in the type store
            # Getting the type of 'stypy_return_type' (line 243)
            stypy_return_type_289061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_289061)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cmd'
            return stypy_return_type_289061

        # Assigning a type to the variable 'cmd' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'cmd', cmd)
        
        # Assigning a Call to a Subscript (line 246):
        
        # Assigning a Call to a Subscript (line 246):
        
        # Call to make_external_conversion_command(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'cmd' (line 246)
        cmd_289063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 60), 'cmd', False)
        # Processing the call keyword arguments (line 246)
        kwargs_289064 = {}
        # Getting the type of 'make_external_conversion_command' (line 246)
        make_external_conversion_command_289062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 27), 'make_external_conversion_command', False)
        # Calling make_external_conversion_command(args, kwargs) (line 246)
        make_external_conversion_command_call_result_289065 = invoke(stypy.reporting.localization.Localization(__file__, 246, 27), make_external_conversion_command_289062, *[cmd_289063], **kwargs_289064)
        
        # Getting the type of 'converter' (line 246)
        converter_289066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'converter')
        unicode_289067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 18), 'unicode', u'pdf')
        # Storing an element on a container (line 246)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 8), converter_289066, (unicode_289067, make_external_conversion_command_call_result_289065))
        
        # Assigning a Call to a Subscript (line 247):
        
        # Assigning a Call to a Subscript (line 247):
        
        # Call to make_external_conversion_command(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'cmd' (line 247)
        cmd_289069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 60), 'cmd', False)
        # Processing the call keyword arguments (line 247)
        kwargs_289070 = {}
        # Getting the type of 'make_external_conversion_command' (line 247)
        make_external_conversion_command_289068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 27), 'make_external_conversion_command', False)
        # Calling make_external_conversion_command(args, kwargs) (line 247)
        make_external_conversion_command_call_result_289071 = invoke(stypy.reporting.localization.Localization(__file__, 247, 27), make_external_conversion_command_289068, *[cmd_289069], **kwargs_289070)
        
        # Getting the type of 'converter' (line 247)
        converter_289072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'converter')
        unicode_289073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 18), 'unicode', u'eps')
        # Storing an element on a container (line 247)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 8), converter_289072, (unicode_289073, make_external_conversion_command_call_result_289071))

        if more_types_in_union_289047:
            # SSA join for if statement (line 242)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to checkdep_inkscape(...): (line 249)
    # Processing the call keyword arguments (line 249)
    kwargs_289076 = {}
    # Getting the type of 'matplotlib' (line 249)
    matplotlib_289074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 7), 'matplotlib', False)
    # Obtaining the member 'checkdep_inkscape' of a type (line 249)
    checkdep_inkscape_289075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 7), matplotlib_289074, 'checkdep_inkscape')
    # Calling checkdep_inkscape(args, kwargs) (line 249)
    checkdep_inkscape_call_result_289077 = invoke(stypy.reporting.localization.Localization(__file__, 249, 7), checkdep_inkscape_289075, *[], **kwargs_289076)
    
    # Getting the type of 'None' (line 249)
    None_289078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 45), 'None')
    # Applying the binary operator 'isnot' (line 249)
    result_is_not_289079 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 7), 'isnot', checkdep_inkscape_call_result_289077, None_289078)
    
    # Testing the type of an if condition (line 249)
    if_condition_289080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 4), result_is_not_289079)
    # Assigning a type to the variable 'if_condition_289080' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'if_condition_289080', if_condition_289080)
    # SSA begins for if statement (line 249)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 250):
    
    # Assigning a Call to a Subscript (line 250):
    
    # Call to _SVGConverter(...): (line 250)
    # Processing the call keyword arguments (line 250)
    kwargs_289082 = {}
    # Getting the type of '_SVGConverter' (line 250)
    _SVGConverter_289081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), '_SVGConverter', False)
    # Calling _SVGConverter(args, kwargs) (line 250)
    _SVGConverter_call_result_289083 = invoke(stypy.reporting.localization.Localization(__file__, 250, 27), _SVGConverter_289081, *[], **kwargs_289082)
    
    # Getting the type of 'converter' (line 250)
    converter_289084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'converter')
    unicode_289085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 18), 'unicode', u'svg')
    # Storing an element on a container (line 250)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 8), converter_289084, (unicode_289085, _SVGConverter_call_result_289083))
    # SSA join for if statement (line 249)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_update_converter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_update_converter' in the type store
    # Getting the type of 'stypy_return_type' (line 240)
    stypy_return_type_289086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289086)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_update_converter'
    return stypy_return_type_289086

# Assigning a type to the variable '_update_converter' (line 240)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), '_update_converter', _update_converter)

# Assigning a Dict to a Name (line 257):

# Assigning a Dict to a Name (line 257):

# Obtaining an instance of the builtin type 'dict' (line 257)
dict_289087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 257)

# Assigning a type to the variable 'converter' (line 257)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'converter', dict_289087)

# Call to _update_converter(...): (line 258)
# Processing the call keyword arguments (line 258)
kwargs_289089 = {}
# Getting the type of '_update_converter' (line 258)
_update_converter_289088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 0), '_update_converter', False)
# Calling _update_converter(args, kwargs) (line 258)
_update_converter_call_result_289090 = invoke(stypy.reporting.localization.Localization(__file__, 258, 0), _update_converter_289088, *[], **kwargs_289089)


@norecursion
def comparable_formats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'comparable_formats'
    module_type_store = module_type_store.open_function_context('comparable_formats', 261, 0, False)
    
    # Passed parameters checking function
    comparable_formats.stypy_localization = localization
    comparable_formats.stypy_type_of_self = None
    comparable_formats.stypy_type_store = module_type_store
    comparable_formats.stypy_function_name = 'comparable_formats'
    comparable_formats.stypy_param_names_list = []
    comparable_formats.stypy_varargs_param_name = None
    comparable_formats.stypy_kwargs_param_name = None
    comparable_formats.stypy_call_defaults = defaults
    comparable_formats.stypy_call_varargs = varargs
    comparable_formats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'comparable_formats', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'comparable_formats', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'comparable_formats(...)' code ##################

    unicode_289091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, (-1)), 'unicode', u'\n    Returns the list of file formats that compare_images can compare\n    on this system.\n\n    ')
    
    # Obtaining an instance of the builtin type 'list' (line 267)
    list_289092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 267)
    # Adding element type (line 267)
    unicode_289093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 12), 'unicode', u'png')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 11), list_289092, unicode_289093)
    
    
    # Call to list(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'converter' (line 267)
    converter_289095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 26), 'converter', False)
    # Processing the call keyword arguments (line 267)
    kwargs_289096 = {}
    # Getting the type of 'list' (line 267)
    list_289094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 21), 'list', False)
    # Calling list(args, kwargs) (line 267)
    list_call_result_289097 = invoke(stypy.reporting.localization.Localization(__file__, 267, 21), list_289094, *[converter_289095], **kwargs_289096)
    
    # Applying the binary operator '+' (line 267)
    result_add_289098 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 11), '+', list_289092, list_call_result_289097)
    
    # Assigning a type to the variable 'stypy_return_type' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type', result_add_289098)
    
    # ################# End of 'comparable_formats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'comparable_formats' in the type store
    # Getting the type of 'stypy_return_type' (line 261)
    stypy_return_type_289099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289099)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'comparable_formats'
    return stypy_return_type_289099

# Assigning a type to the variable 'comparable_formats' (line 261)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 0), 'comparable_formats', comparable_formats)

@norecursion
def convert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'convert'
    module_type_store = module_type_store.open_function_context('convert', 270, 0, False)
    
    # Passed parameters checking function
    convert.stypy_localization = localization
    convert.stypy_type_of_self = None
    convert.stypy_type_store = module_type_store
    convert.stypy_function_name = 'convert'
    convert.stypy_param_names_list = ['filename', 'cache']
    convert.stypy_varargs_param_name = None
    convert.stypy_kwargs_param_name = None
    convert.stypy_call_defaults = defaults
    convert.stypy_call_varargs = varargs
    convert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'convert', ['filename', 'cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'convert', localization, ['filename', 'cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'convert(...)' code ##################

    unicode_289100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, (-1)), 'unicode', u"\n    Convert the named file into a png file.  Returns the name of the\n    created file.\n\n    If *cache* is True, the result of the conversion is cached in\n    `matplotlib._get_cachedir() + '/test_cache/'`.  The caching is based\n    on a hash of the exact contents of the input file.  The is no limit\n    on the size of the cache, so it may need to be manually cleared\n    periodically.\n\n    ")
    
    # Assigning a Call to a Tuple (line 282):
    
    # Assigning a Call to a Name:
    
    # Call to rsplit(...): (line 282)
    # Processing the call arguments (line 282)
    unicode_289103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 38), 'unicode', u'.')
    int_289104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 43), 'int')
    # Processing the call keyword arguments (line 282)
    kwargs_289105 = {}
    # Getting the type of 'filename' (line 282)
    filename_289101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 22), 'filename', False)
    # Obtaining the member 'rsplit' of a type (line 282)
    rsplit_289102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 22), filename_289101, 'rsplit')
    # Calling rsplit(args, kwargs) (line 282)
    rsplit_call_result_289106 = invoke(stypy.reporting.localization.Localization(__file__, 282, 22), rsplit_289102, *[unicode_289103, int_289104], **kwargs_289105)
    
    # Assigning a type to the variable 'call_assignment_288337' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'call_assignment_288337', rsplit_call_result_289106)
    
    # Assigning a Call to a Name (line 282):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_289109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 4), 'int')
    # Processing the call keyword arguments
    kwargs_289110 = {}
    # Getting the type of 'call_assignment_288337' (line 282)
    call_assignment_288337_289107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'call_assignment_288337', False)
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___289108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 4), call_assignment_288337_289107, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_289111 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289108, *[int_289109], **kwargs_289110)
    
    # Assigning a type to the variable 'call_assignment_288338' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'call_assignment_288338', getitem___call_result_289111)
    
    # Assigning a Name to a Name (line 282):
    # Getting the type of 'call_assignment_288338' (line 282)
    call_assignment_288338_289112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'call_assignment_288338')
    # Assigning a type to the variable 'base' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'base', call_assignment_288338_289112)
    
    # Assigning a Call to a Name (line 282):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_289115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 4), 'int')
    # Processing the call keyword arguments
    kwargs_289116 = {}
    # Getting the type of 'call_assignment_288337' (line 282)
    call_assignment_288337_289113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'call_assignment_288337', False)
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___289114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 4), call_assignment_288337_289113, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_289117 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289114, *[int_289115], **kwargs_289116)
    
    # Assigning a type to the variable 'call_assignment_288339' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'call_assignment_288339', getitem___call_result_289117)
    
    # Assigning a Name to a Name (line 282):
    # Getting the type of 'call_assignment_288339' (line 282)
    call_assignment_288339_289118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'call_assignment_288339')
    # Assigning a type to the variable 'extension' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 10), 'extension', call_assignment_288339_289118)
    
    
    # Getting the type of 'extension' (line 283)
    extension_289119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 7), 'extension')
    # Getting the type of 'converter' (line 283)
    converter_289120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'converter')
    # Applying the binary operator 'notin' (line 283)
    result_contains_289121 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 7), 'notin', extension_289119, converter_289120)
    
    # Testing the type of an if condition (line 283)
    if_condition_289122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 4), result_contains_289121)
    # Assigning a type to the variable 'if_condition_289122' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'if_condition_289122', if_condition_289122)
    # SSA begins for if statement (line 283)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 284):
    
    # Assigning a BinOp to a Name (line 284):
    unicode_289123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 17), 'unicode', u"Don't know how to convert %s files to png")
    # Getting the type of 'extension' (line 284)
    extension_289124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 63), 'extension')
    # Applying the binary operator '%' (line 284)
    result_mod_289125 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 17), '%', unicode_289123, extension_289124)
    
    # Assigning a type to the variable 'reason' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'reason', result_mod_289125)
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 285, 8))
    
    # 'from matplotlib.testing import is_called_from_pytest' statement (line 285)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_289126 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 285, 8), 'matplotlib.testing')

    if (type(import_289126) is not StypyTypeError):

        if (import_289126 != 'pyd_module'):
            __import__(import_289126)
            sys_modules_289127 = sys.modules[import_289126]
            import_from_module(stypy.reporting.localization.Localization(__file__, 285, 8), 'matplotlib.testing', sys_modules_289127.module_type_store, module_type_store, ['is_called_from_pytest'])
            nest_module(stypy.reporting.localization.Localization(__file__, 285, 8), __file__, sys_modules_289127, sys_modules_289127.module_type_store, module_type_store)
        else:
            from matplotlib.testing import is_called_from_pytest

            import_from_module(stypy.reporting.localization.Localization(__file__, 285, 8), 'matplotlib.testing', None, module_type_store, ['is_called_from_pytest'], [is_called_from_pytest])

    else:
        # Assigning a type to the variable 'matplotlib.testing' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'matplotlib.testing', import_289126)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    
    # Call to is_called_from_pytest(...): (line 286)
    # Processing the call keyword arguments (line 286)
    kwargs_289129 = {}
    # Getting the type of 'is_called_from_pytest' (line 286)
    is_called_from_pytest_289128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), 'is_called_from_pytest', False)
    # Calling is_called_from_pytest(args, kwargs) (line 286)
    is_called_from_pytest_call_result_289130 = invoke(stypy.reporting.localization.Localization(__file__, 286, 11), is_called_from_pytest_289128, *[], **kwargs_289129)
    
    # Testing the type of an if condition (line 286)
    if_condition_289131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 8), is_called_from_pytest_call_result_289130)
    # Assigning a type to the variable 'if_condition_289131' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'if_condition_289131', if_condition_289131)
    # SSA begins for if statement (line 286)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 287, 12))
    
    # 'import pytest' statement (line 287)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_289132 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 287, 12), 'pytest')

    if (type(import_289132) is not StypyTypeError):

        if (import_289132 != 'pyd_module'):
            __import__(import_289132)
            sys_modules_289133 = sys.modules[import_289132]
            import_module(stypy.reporting.localization.Localization(__file__, 287, 12), 'pytest', sys_modules_289133.module_type_store, module_type_store)
        else:
            import pytest

            import_module(stypy.reporting.localization.Localization(__file__, 287, 12), 'pytest', pytest, module_type_store)

    else:
        # Assigning a type to the variable 'pytest' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'pytest', import_289132)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    # Call to skip(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'reason' (line 288)
    reason_289136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'reason', False)
    # Processing the call keyword arguments (line 288)
    kwargs_289137 = {}
    # Getting the type of 'pytest' (line 288)
    pytest_289134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'pytest', False)
    # Obtaining the member 'skip' of a type (line 288)
    skip_289135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), pytest_289134, 'skip')
    # Calling skip(args, kwargs) (line 288)
    skip_call_result_289138 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), skip_289135, *[reason_289136], **kwargs_289137)
    
    # SSA branch for the else part of an if statement (line 286)
    module_type_store.open_ssa_branch('else')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 290, 12))
    
    # 'from nose import SkipTest' statement (line 290)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_289139 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 290, 12), 'nose')

    if (type(import_289139) is not StypyTypeError):

        if (import_289139 != 'pyd_module'):
            __import__(import_289139)
            sys_modules_289140 = sys.modules[import_289139]
            import_from_module(stypy.reporting.localization.Localization(__file__, 290, 12), 'nose', sys_modules_289140.module_type_store, module_type_store, ['SkipTest'])
            nest_module(stypy.reporting.localization.Localization(__file__, 290, 12), __file__, sys_modules_289140, sys_modules_289140.module_type_store, module_type_store)
        else:
            from nose import SkipTest

            import_from_module(stypy.reporting.localization.Localization(__file__, 290, 12), 'nose', None, module_type_store, ['SkipTest'], [SkipTest])

    else:
        # Assigning a type to the variable 'nose' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'nose', import_289139)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    # Call to SkipTest(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'reason' (line 291)
    reason_289142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 27), 'reason', False)
    # Processing the call keyword arguments (line 291)
    kwargs_289143 = {}
    # Getting the type of 'SkipTest' (line 291)
    SkipTest_289141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), 'SkipTest', False)
    # Calling SkipTest(args, kwargs) (line 291)
    SkipTest_call_result_289144 = invoke(stypy.reporting.localization.Localization(__file__, 291, 18), SkipTest_289141, *[reason_289142], **kwargs_289143)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 291, 12), SkipTest_call_result_289144, 'raise parameter', BaseException)
    # SSA join for if statement (line 286)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 283)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 292):
    
    # Assigning a BinOp to a Name (line 292):
    # Getting the type of 'base' (line 292)
    base_289145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 14), 'base')
    unicode_289146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 21), 'unicode', u'_')
    # Applying the binary operator '+' (line 292)
    result_add_289147 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 14), '+', base_289145, unicode_289146)
    
    # Getting the type of 'extension' (line 292)
    extension_289148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 27), 'extension')
    # Applying the binary operator '+' (line 292)
    result_add_289149 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 25), '+', result_add_289147, extension_289148)
    
    unicode_289150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 39), 'unicode', u'.png')
    # Applying the binary operator '+' (line 292)
    result_add_289151 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 37), '+', result_add_289149, unicode_289150)
    
    # Assigning a type to the variable 'newname' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'newname', result_add_289151)
    
    
    
    # Call to exists(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'filename' (line 293)
    filename_289155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'filename', False)
    # Processing the call keyword arguments (line 293)
    kwargs_289156 = {}
    # Getting the type of 'os' (line 293)
    os_289152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 293)
    path_289153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 11), os_289152, 'path')
    # Obtaining the member 'exists' of a type (line 293)
    exists_289154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 11), path_289153, 'exists')
    # Calling exists(args, kwargs) (line 293)
    exists_call_result_289157 = invoke(stypy.reporting.localization.Localization(__file__, 293, 11), exists_289154, *[filename_289155], **kwargs_289156)
    
    # Applying the 'not' unary operator (line 293)
    result_not__289158 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 7), 'not', exists_call_result_289157)
    
    # Testing the type of an if condition (line 293)
    if_condition_289159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 4), result_not__289158)
    # Assigning a type to the variable 'if_condition_289159' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'if_condition_289159', if_condition_289159)
    # SSA begins for if statement (line 293)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to IOError(...): (line 294)
    # Processing the call arguments (line 294)
    unicode_289161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 22), 'unicode', u"'%s' does not exist")
    # Getting the type of 'filename' (line 294)
    filename_289162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 46), 'filename', False)
    # Applying the binary operator '%' (line 294)
    result_mod_289163 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 22), '%', unicode_289161, filename_289162)
    
    # Processing the call keyword arguments (line 294)
    kwargs_289164 = {}
    # Getting the type of 'IOError' (line 294)
    IOError_289160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 14), 'IOError', False)
    # Calling IOError(args, kwargs) (line 294)
    IOError_call_result_289165 = invoke(stypy.reporting.localization.Localization(__file__, 294, 14), IOError_289160, *[result_mod_289163], **kwargs_289164)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 294, 8), IOError_call_result_289165, 'raise parameter', BaseException)
    # SSA join for if statement (line 293)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to exists(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'newname' (line 298)
    newname_289169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 27), 'newname', False)
    # Processing the call keyword arguments (line 298)
    kwargs_289170 = {}
    # Getting the type of 'os' (line 298)
    os_289166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'os', False)
    # Obtaining the member 'path' of a type (line 298)
    path_289167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 12), os_289166, 'path')
    # Obtaining the member 'exists' of a type (line 298)
    exists_289168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 12), path_289167, 'exists')
    # Calling exists(args, kwargs) (line 298)
    exists_call_result_289171 = invoke(stypy.reporting.localization.Localization(__file__, 298, 12), exists_289168, *[newname_289169], **kwargs_289170)
    
    # Applying the 'not' unary operator (line 298)
    result_not__289172 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 8), 'not', exists_call_result_289171)
    
    
    
    # Call to stat(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'newname' (line 299)
    newname_289175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 20), 'newname', False)
    # Processing the call keyword arguments (line 299)
    kwargs_289176 = {}
    # Getting the type of 'os' (line 299)
    os_289173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'os', False)
    # Obtaining the member 'stat' of a type (line 299)
    stat_289174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), os_289173, 'stat')
    # Calling stat(args, kwargs) (line 299)
    stat_call_result_289177 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), stat_289174, *[newname_289175], **kwargs_289176)
    
    # Obtaining the member 'st_mtime' of a type (line 299)
    st_mtime_289178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), stat_call_result_289177, 'st_mtime')
    
    # Call to stat(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'filename' (line 299)
    filename_289181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 48), 'filename', False)
    # Processing the call keyword arguments (line 299)
    kwargs_289182 = {}
    # Getting the type of 'os' (line 299)
    os_289179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 40), 'os', False)
    # Obtaining the member 'stat' of a type (line 299)
    stat_289180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 40), os_289179, 'stat')
    # Calling stat(args, kwargs) (line 299)
    stat_call_result_289183 = invoke(stypy.reporting.localization.Localization(__file__, 299, 40), stat_289180, *[filename_289181], **kwargs_289182)
    
    # Obtaining the member 'st_mtime' of a type (line 299)
    st_mtime_289184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 40), stat_call_result_289183, 'st_mtime')
    # Applying the binary operator '<' (line 299)
    result_lt_289185 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 12), '<', st_mtime_289178, st_mtime_289184)
    
    # Applying the binary operator 'or' (line 298)
    result_or_keyword_289186 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 8), 'or', result_not__289172, result_lt_289185)
    
    # Testing the type of an if condition (line 298)
    if_condition_289187 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 4), result_or_keyword_289186)
    # Assigning a type to the variable 'if_condition_289187' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'if_condition_289187', if_condition_289187)
    # SSA begins for if statement (line 298)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'cache' (line 300)
    cache_289188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 'cache')
    # Testing the type of an if condition (line 300)
    if_condition_289189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 8), cache_289188)
    # Assigning a type to the variable 'if_condition_289189' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'if_condition_289189', if_condition_289189)
    # SSA begins for if statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 301):
    
    # Assigning a Call to a Name (line 301):
    
    # Call to get_cache_dir(...): (line 301)
    # Processing the call keyword arguments (line 301)
    kwargs_289191 = {}
    # Getting the type of 'get_cache_dir' (line 301)
    get_cache_dir_289190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'get_cache_dir', False)
    # Calling get_cache_dir(args, kwargs) (line 301)
    get_cache_dir_call_result_289192 = invoke(stypy.reporting.localization.Localization(__file__, 301, 24), get_cache_dir_289190, *[], **kwargs_289191)
    
    # Assigning a type to the variable 'cache_dir' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'cache_dir', get_cache_dir_call_result_289192)
    # SSA branch for the else part of an if statement (line 300)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 303):
    
    # Assigning a Name to a Name (line 303):
    # Getting the type of 'None' (line 303)
    None_289193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'None')
    # Assigning a type to the variable 'cache_dir' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'cache_dir', None_289193)
    # SSA join for if statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 305)
    # Getting the type of 'cache_dir' (line 305)
    cache_dir_289194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'cache_dir')
    # Getting the type of 'None' (line 305)
    None_289195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 28), 'None')
    
    (may_be_289196, more_types_in_union_289197) = may_not_be_none(cache_dir_289194, None_289195)

    if may_be_289196:

        if more_types_in_union_289197:
            # Runtime conditional SSA (line 305)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 306):
        
        # Assigning a Call to a Name (line 306):
        
        # Call to get_file_hash(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'filename' (line 306)
        filename_289199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 39), 'filename', False)
        # Processing the call keyword arguments (line 306)
        kwargs_289200 = {}
        # Getting the type of 'get_file_hash' (line 306)
        get_file_hash_289198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 25), 'get_file_hash', False)
        # Calling get_file_hash(args, kwargs) (line 306)
        get_file_hash_call_result_289201 = invoke(stypy.reporting.localization.Localization(__file__, 306, 25), get_file_hash_289198, *[filename_289199], **kwargs_289200)
        
        # Assigning a type to the variable 'hash_value' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'hash_value', get_file_hash_call_result_289201)
        
        # Assigning a Subscript to a Name (line 307):
        
        # Assigning a Subscript to a Name (line 307):
        
        # Obtaining the type of the subscript
        int_289202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 48), 'int')
        
        # Call to splitext(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'newname' (line 307)
        newname_289206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 39), 'newname', False)
        # Processing the call keyword arguments (line 307)
        kwargs_289207 = {}
        # Getting the type of 'os' (line 307)
        os_289203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 307)
        path_289204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 22), os_289203, 'path')
        # Obtaining the member 'splitext' of a type (line 307)
        splitext_289205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 22), path_289204, 'splitext')
        # Calling splitext(args, kwargs) (line 307)
        splitext_call_result_289208 = invoke(stypy.reporting.localization.Localization(__file__, 307, 22), splitext_289205, *[newname_289206], **kwargs_289207)
        
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___289209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 22), splitext_call_result_289208, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_289210 = invoke(stypy.reporting.localization.Localization(__file__, 307, 22), getitem___289209, int_289202)
        
        # Assigning a type to the variable 'new_ext' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'new_ext', subscript_call_result_289210)
        
        # Assigning a Call to a Name (line 308):
        
        # Assigning a Call to a Name (line 308):
        
        # Call to join(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'cache_dir' (line 308)
        cache_dir_289214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 39), 'cache_dir', False)
        # Getting the type of 'hash_value' (line 308)
        hash_value_289215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 50), 'hash_value', False)
        # Getting the type of 'new_ext' (line 308)
        new_ext_289216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 63), 'new_ext', False)
        # Applying the binary operator '+' (line 308)
        result_add_289217 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 50), '+', hash_value_289215, new_ext_289216)
        
        # Processing the call keyword arguments (line 308)
        kwargs_289218 = {}
        # Getting the type of 'os' (line 308)
        os_289211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 308)
        path_289212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 26), os_289211, 'path')
        # Obtaining the member 'join' of a type (line 308)
        join_289213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 26), path_289212, 'join')
        # Calling join(args, kwargs) (line 308)
        join_call_result_289219 = invoke(stypy.reporting.localization.Localization(__file__, 308, 26), join_289213, *[cache_dir_289214, result_add_289217], **kwargs_289218)
        
        # Assigning a type to the variable 'cached_file' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'cached_file', join_call_result_289219)
        
        
        # Call to exists(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'cached_file' (line 309)
        cached_file_289223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 30), 'cached_file', False)
        # Processing the call keyword arguments (line 309)
        kwargs_289224 = {}
        # Getting the type of 'os' (line 309)
        os_289220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 309)
        path_289221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 15), os_289220, 'path')
        # Obtaining the member 'exists' of a type (line 309)
        exists_289222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 15), path_289221, 'exists')
        # Calling exists(args, kwargs) (line 309)
        exists_call_result_289225 = invoke(stypy.reporting.localization.Localization(__file__, 309, 15), exists_289222, *[cached_file_289223], **kwargs_289224)
        
        # Testing the type of an if condition (line 309)
        if_condition_289226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 12), exists_call_result_289225)
        # Assigning a type to the variable 'if_condition_289226' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'if_condition_289226', if_condition_289226)
        # SSA begins for if statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copyfile(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'cached_file' (line 310)
        cached_file_289229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 32), 'cached_file', False)
        # Getting the type of 'newname' (line 310)
        newname_289230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 45), 'newname', False)
        # Processing the call keyword arguments (line 310)
        kwargs_289231 = {}
        # Getting the type of 'shutil' (line 310)
        shutil_289227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'shutil', False)
        # Obtaining the member 'copyfile' of a type (line 310)
        copyfile_289228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 16), shutil_289227, 'copyfile')
        # Calling copyfile(args, kwargs) (line 310)
        copyfile_call_result_289232 = invoke(stypy.reporting.localization.Localization(__file__, 310, 16), copyfile_289228, *[cached_file_289229, newname_289230], **kwargs_289231)
        
        # Getting the type of 'newname' (line 311)
        newname_289233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'newname')
        # Assigning a type to the variable 'stypy_return_type' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'stypy_return_type', newname_289233)
        # SSA join for if statement (line 309)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_289197:
            # SSA join for if statement (line 305)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to (...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'filename' (line 313)
    filename_289238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 29), 'filename', False)
    # Getting the type of 'newname' (line 313)
    newname_289239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 39), 'newname', False)
    # Processing the call keyword arguments (line 313)
    kwargs_289240 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'extension' (line 313)
    extension_289234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 18), 'extension', False)
    # Getting the type of 'converter' (line 313)
    converter_289235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'converter', False)
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___289236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), converter_289235, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_289237 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), getitem___289236, extension_289234)
    
    # Calling (args, kwargs) (line 313)
    _call_result_289241 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), subscript_call_result_289237, *[filename_289238, newname_289239], **kwargs_289240)
    
    
    # Type idiom detected: calculating its left and rigth part (line 315)
    # Getting the type of 'cache_dir' (line 315)
    cache_dir_289242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'cache_dir')
    # Getting the type of 'None' (line 315)
    None_289243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 28), 'None')
    
    (may_be_289244, more_types_in_union_289245) = may_not_be_none(cache_dir_289242, None_289243)

    if may_be_289244:

        if more_types_in_union_289245:
            # Runtime conditional SSA (line 315)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to copyfile(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'newname' (line 316)
        newname_289248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 28), 'newname', False)
        # Getting the type of 'cached_file' (line 316)
        cached_file_289249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 37), 'cached_file', False)
        # Processing the call keyword arguments (line 316)
        kwargs_289250 = {}
        # Getting the type of 'shutil' (line 316)
        shutil_289246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'shutil', False)
        # Obtaining the member 'copyfile' of a type (line 316)
        copyfile_289247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), shutil_289246, 'copyfile')
        # Calling copyfile(args, kwargs) (line 316)
        copyfile_call_result_289251 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), copyfile_289247, *[newname_289248, cached_file_289249], **kwargs_289250)
        

        if more_types_in_union_289245:
            # SSA join for if statement (line 315)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 298)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'newname' (line 318)
    newname_289252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'newname')
    # Assigning a type to the variable 'stypy_return_type' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type', newname_289252)
    
    # ################# End of 'convert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'convert' in the type store
    # Getting the type of 'stypy_return_type' (line 270)
    stypy_return_type_289253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289253)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'convert'
    return stypy_return_type_289253

# Assigning a type to the variable 'convert' (line 270)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'convert', convert)

# Assigning a Dict to a Name (line 324):

# Assigning a Dict to a Name (line 324):

# Obtaining an instance of the builtin type 'dict' (line 324)
dict_289254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 324)

# Assigning a type to the variable 'verifiers' (line 324)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 0), 'verifiers', dict_289254)


# Evaluating a boolean operation
# Getting the type of 'False' (line 327)
False_289255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 3), 'False')

# Call to checkdep_xmllint(...): (line 327)
# Processing the call keyword arguments (line 327)
kwargs_289258 = {}
# Getting the type of 'matplotlib' (line 327)
matplotlib_289256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 13), 'matplotlib', False)
# Obtaining the member 'checkdep_xmllint' of a type (line 327)
checkdep_xmllint_289257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 13), matplotlib_289256, 'checkdep_xmllint')
# Calling checkdep_xmllint(args, kwargs) (line 327)
checkdep_xmllint_call_result_289259 = invoke(stypy.reporting.localization.Localization(__file__, 327, 13), checkdep_xmllint_289257, *[], **kwargs_289258)

# Applying the binary operator 'and' (line 327)
result_and_keyword_289260 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 3), 'and', False_289255, checkdep_xmllint_call_result_289259)

# Testing the type of an if condition (line 327)
if_condition_289261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 0), result_and_keyword_289260)
# Assigning a type to the variable 'if_condition_289261' (line 327)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 0), 'if_condition_289261', if_condition_289261)
# SSA begins for if statement (line 327)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Lambda to a Subscript (line 328):

# Assigning a Lambda to a Subscript (line 328):

@norecursion
def _stypy_temp_lambda_147(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_147'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_147', 328, 23, True)
    # Passed parameters checking function
    _stypy_temp_lambda_147.stypy_localization = localization
    _stypy_temp_lambda_147.stypy_type_of_self = None
    _stypy_temp_lambda_147.stypy_type_store = module_type_store
    _stypy_temp_lambda_147.stypy_function_name = '_stypy_temp_lambda_147'
    _stypy_temp_lambda_147.stypy_param_names_list = ['filename']
    _stypy_temp_lambda_147.stypy_varargs_param_name = None
    _stypy_temp_lambda_147.stypy_kwargs_param_name = None
    _stypy_temp_lambda_147.stypy_call_defaults = defaults
    _stypy_temp_lambda_147.stypy_call_varargs = varargs
    _stypy_temp_lambda_147.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_147', ['filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_147', ['filename'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Obtaining an instance of the builtin type 'list' (line 328)
    list_289262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 328)
    # Adding element type (line 328)
    unicode_289263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 8), 'unicode', u'xmllint')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 40), list_289262, unicode_289263)
    # Adding element type (line 328)
    unicode_289264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 19), 'unicode', u'--valid')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 40), list_289262, unicode_289264)
    # Adding element type (line 328)
    unicode_289265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 30), 'unicode', u'--nowarning')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 40), list_289262, unicode_289265)
    # Adding element type (line 328)
    unicode_289266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 45), 'unicode', u'--noout')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 40), list_289262, unicode_289266)
    # Adding element type (line 328)
    # Getting the type of 'filename' (line 329)
    filename_289267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 56), 'filename')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 40), list_289262, filename_289267)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 23), 'stypy_return_type', list_289262)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_147' in the type store
    # Getting the type of 'stypy_return_type' (line 328)
    stypy_return_type_289268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 23), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289268)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_147'
    return stypy_return_type_289268

# Assigning a type to the variable '_stypy_temp_lambda_147' (line 328)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 23), '_stypy_temp_lambda_147', _stypy_temp_lambda_147)
# Getting the type of '_stypy_temp_lambda_147' (line 328)
_stypy_temp_lambda_147_289269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 23), '_stypy_temp_lambda_147')
# Getting the type of 'verifiers' (line 328)
verifiers_289270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'verifiers')
unicode_289271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 14), 'unicode', u'svg')
# Storing an element on a container (line 328)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 4), verifiers_289270, (unicode_289271, _stypy_temp_lambda_147_289269))
# SSA join for if statement (line 327)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def verify(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'verify'
    module_type_store = module_type_store.open_function_context('verify', 332, 0, False)
    
    # Passed parameters checking function
    verify.stypy_localization = localization
    verify.stypy_type_of_self = None
    verify.stypy_type_store = module_type_store
    verify.stypy_function_name = 'verify'
    verify.stypy_param_names_list = ['filename']
    verify.stypy_varargs_param_name = None
    verify.stypy_kwargs_param_name = None
    verify.stypy_call_defaults = defaults
    verify.stypy_call_varargs = varargs
    verify.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'verify', ['filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'verify', localization, ['filename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'verify(...)' code ##################

    unicode_289272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 4), 'unicode', u'Verify the file through some sort of verification tool.')
    
    
    
    # Call to exists(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'filename' (line 335)
    filename_289276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 26), 'filename', False)
    # Processing the call keyword arguments (line 335)
    kwargs_289277 = {}
    # Getting the type of 'os' (line 335)
    os_289273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 335)
    path_289274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 11), os_289273, 'path')
    # Obtaining the member 'exists' of a type (line 335)
    exists_289275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 11), path_289274, 'exists')
    # Calling exists(args, kwargs) (line 335)
    exists_call_result_289278 = invoke(stypy.reporting.localization.Localization(__file__, 335, 11), exists_289275, *[filename_289276], **kwargs_289277)
    
    # Applying the 'not' unary operator (line 335)
    result_not__289279 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 7), 'not', exists_call_result_289278)
    
    # Testing the type of an if condition (line 335)
    if_condition_289280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 4), result_not__289279)
    # Assigning a type to the variable 'if_condition_289280' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'if_condition_289280', if_condition_289280)
    # SSA begins for if statement (line 335)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to IOError(...): (line 336)
    # Processing the call arguments (line 336)
    unicode_289282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 22), 'unicode', u"'%s' does not exist")
    # Getting the type of 'filename' (line 336)
    filename_289283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 46), 'filename', False)
    # Applying the binary operator '%' (line 336)
    result_mod_289284 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 22), '%', unicode_289282, filename_289283)
    
    # Processing the call keyword arguments (line 336)
    kwargs_289285 = {}
    # Getting the type of 'IOError' (line 336)
    IOError_289281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 14), 'IOError', False)
    # Calling IOError(args, kwargs) (line 336)
    IOError_call_result_289286 = invoke(stypy.reporting.localization.Localization(__file__, 336, 14), IOError_289281, *[result_mod_289284], **kwargs_289285)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 336, 8), IOError_call_result_289286, 'raise parameter', BaseException)
    # SSA join for if statement (line 335)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 337):
    
    # Assigning a Call to a Name:
    
    # Call to rsplit(...): (line 337)
    # Processing the call arguments (line 337)
    unicode_289289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 38), 'unicode', u'.')
    int_289290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 43), 'int')
    # Processing the call keyword arguments (line 337)
    kwargs_289291 = {}
    # Getting the type of 'filename' (line 337)
    filename_289287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 22), 'filename', False)
    # Obtaining the member 'rsplit' of a type (line 337)
    rsplit_289288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 22), filename_289287, 'rsplit')
    # Calling rsplit(args, kwargs) (line 337)
    rsplit_call_result_289292 = invoke(stypy.reporting.localization.Localization(__file__, 337, 22), rsplit_289288, *[unicode_289289, int_289290], **kwargs_289291)
    
    # Assigning a type to the variable 'call_assignment_288340' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'call_assignment_288340', rsplit_call_result_289292)
    
    # Assigning a Call to a Name (line 337):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_289295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 4), 'int')
    # Processing the call keyword arguments
    kwargs_289296 = {}
    # Getting the type of 'call_assignment_288340' (line 337)
    call_assignment_288340_289293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'call_assignment_288340', False)
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___289294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 4), call_assignment_288340_289293, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_289297 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289294, *[int_289295], **kwargs_289296)
    
    # Assigning a type to the variable 'call_assignment_288341' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'call_assignment_288341', getitem___call_result_289297)
    
    # Assigning a Name to a Name (line 337):
    # Getting the type of 'call_assignment_288341' (line 337)
    call_assignment_288341_289298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'call_assignment_288341')
    # Assigning a type to the variable 'base' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'base', call_assignment_288341_289298)
    
    # Assigning a Call to a Name (line 337):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_289301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 4), 'int')
    # Processing the call keyword arguments
    kwargs_289302 = {}
    # Getting the type of 'call_assignment_288340' (line 337)
    call_assignment_288340_289299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'call_assignment_288340', False)
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___289300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 4), call_assignment_288340_289299, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_289303 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289300, *[int_289301], **kwargs_289302)
    
    # Assigning a type to the variable 'call_assignment_288342' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'call_assignment_288342', getitem___call_result_289303)
    
    # Assigning a Name to a Name (line 337):
    # Getting the type of 'call_assignment_288342' (line 337)
    call_assignment_288342_289304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'call_assignment_288342')
    # Assigning a type to the variable 'extension' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 10), 'extension', call_assignment_288342_289304)
    
    # Assigning a Call to a Name (line 338):
    
    # Assigning a Call to a Name (line 338):
    
    # Call to get(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'extension' (line 338)
    extension_289307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 29), 'extension', False)
    # Getting the type of 'None' (line 338)
    None_289308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 40), 'None', False)
    # Processing the call keyword arguments (line 338)
    kwargs_289309 = {}
    # Getting the type of 'verifiers' (line 338)
    verifiers_289305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 15), 'verifiers', False)
    # Obtaining the member 'get' of a type (line 338)
    get_289306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 15), verifiers_289305, 'get')
    # Calling get(args, kwargs) (line 338)
    get_call_result_289310 = invoke(stypy.reporting.localization.Localization(__file__, 338, 15), get_289306, *[extension_289307, None_289308], **kwargs_289309)
    
    # Assigning a type to the variable 'verifier' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'verifier', get_call_result_289310)
    
    # Type idiom detected: calculating its left and rigth part (line 339)
    # Getting the type of 'verifier' (line 339)
    verifier_289311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'verifier')
    # Getting the type of 'None' (line 339)
    None_289312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 23), 'None')
    
    (may_be_289313, more_types_in_union_289314) = may_not_be_none(verifier_289311, None_289312)

    if may_be_289313:

        if more_types_in_union_289314:
            # Runtime conditional SSA (line 339)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 340):
        
        # Assigning a Call to a Name (line 340):
        
        # Call to verifier(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'filename' (line 340)
        filename_289316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 23), 'filename', False)
        # Processing the call keyword arguments (line 340)
        kwargs_289317 = {}
        # Getting the type of 'verifier' (line 340)
        verifier_289315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 14), 'verifier', False)
        # Calling verifier(args, kwargs) (line 340)
        verifier_call_result_289318 = invoke(stypy.reporting.localization.Localization(__file__, 340, 14), verifier_289315, *[filename_289316], **kwargs_289317)
        
        # Assigning a type to the variable 'cmd' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'cmd', verifier_call_result_289318)
        
        # Assigning a Call to a Name (line 341):
        
        # Assigning a Call to a Name (line 341):
        
        # Call to Popen(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'cmd' (line 341)
        cmd_289321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 32), 'cmd', False)
        # Processing the call keyword arguments (line 341)
        # Getting the type of 'True' (line 341)
        True_289322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 56), 'True', False)
        keyword_289323 = True_289322
        # Getting the type of 'subprocess' (line 342)
        subprocess_289324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 39), 'subprocess', False)
        # Obtaining the member 'PIPE' of a type (line 342)
        PIPE_289325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 39), subprocess_289324, 'PIPE')
        keyword_289326 = PIPE_289325
        # Getting the type of 'subprocess' (line 342)
        subprocess_289327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 63), 'subprocess', False)
        # Obtaining the member 'PIPE' of a type (line 342)
        PIPE_289328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 63), subprocess_289327, 'PIPE')
        keyword_289329 = PIPE_289328
        kwargs_289330 = {'stdout': keyword_289326, 'stderr': keyword_289329, 'universal_newlines': keyword_289323}
        # Getting the type of 'subprocess' (line 341)
        subprocess_289319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'subprocess', False)
        # Obtaining the member 'Popen' of a type (line 341)
        Popen_289320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), subprocess_289319, 'Popen')
        # Calling Popen(args, kwargs) (line 341)
        Popen_call_result_289331 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), Popen_289320, *[cmd_289321], **kwargs_289330)
        
        # Assigning a type to the variable 'pipe' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'pipe', Popen_call_result_289331)
        
        # Assigning a Call to a Tuple (line 343):
        
        # Assigning a Call to a Name:
        
        # Call to communicate(...): (line 343)
        # Processing the call keyword arguments (line 343)
        kwargs_289334 = {}
        # Getting the type of 'pipe' (line 343)
        pipe_289332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'pipe', False)
        # Obtaining the member 'communicate' of a type (line 343)
        communicate_289333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 25), pipe_289332, 'communicate')
        # Calling communicate(args, kwargs) (line 343)
        communicate_call_result_289335 = invoke(stypy.reporting.localization.Localization(__file__, 343, 25), communicate_289333, *[], **kwargs_289334)
        
        # Assigning a type to the variable 'call_assignment_288343' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_288343', communicate_call_result_289335)
        
        # Assigning a Call to a Name (line 343):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_289338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'int')
        # Processing the call keyword arguments
        kwargs_289339 = {}
        # Getting the type of 'call_assignment_288343' (line 343)
        call_assignment_288343_289336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_288343', False)
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___289337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), call_assignment_288343_289336, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_289340 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289337, *[int_289338], **kwargs_289339)
        
        # Assigning a type to the variable 'call_assignment_288344' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_288344', getitem___call_result_289340)
        
        # Assigning a Name to a Name (line 343):
        # Getting the type of 'call_assignment_288344' (line 343)
        call_assignment_288344_289341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_288344')
        # Assigning a type to the variable 'stdout' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'stdout', call_assignment_288344_289341)
        
        # Assigning a Call to a Name (line 343):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_289344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'int')
        # Processing the call keyword arguments
        kwargs_289345 = {}
        # Getting the type of 'call_assignment_288343' (line 343)
        call_assignment_288343_289342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_288343', False)
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___289343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), call_assignment_288343_289342, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_289346 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289343, *[int_289344], **kwargs_289345)
        
        # Assigning a type to the variable 'call_assignment_288345' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_288345', getitem___call_result_289346)
        
        # Assigning a Name to a Name (line 343):
        # Getting the type of 'call_assignment_288345' (line 343)
        call_assignment_288345_289347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'call_assignment_288345')
        # Assigning a type to the variable 'stderr' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 16), 'stderr', call_assignment_288345_289347)
        
        # Assigning a Call to a Name (line 344):
        
        # Assigning a Call to a Name (line 344):
        
        # Call to wait(...): (line 344)
        # Processing the call keyword arguments (line 344)
        kwargs_289350 = {}
        # Getting the type of 'pipe' (line 344)
        pipe_289348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 18), 'pipe', False)
        # Obtaining the member 'wait' of a type (line 344)
        wait_289349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 18), pipe_289348, 'wait')
        # Calling wait(args, kwargs) (line 344)
        wait_call_result_289351 = invoke(stypy.reporting.localization.Localization(__file__, 344, 18), wait_289349, *[], **kwargs_289350)
        
        # Assigning a type to the variable 'errcode' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'errcode', wait_call_result_289351)
        
        
        # Getting the type of 'errcode' (line 345)
        errcode_289352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 11), 'errcode')
        int_289353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 22), 'int')
        # Applying the binary operator '!=' (line 345)
        result_ne_289354 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 11), '!=', errcode_289352, int_289353)
        
        # Testing the type of an if condition (line 345)
        if_condition_289355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 345, 8), result_ne_289354)
        # Assigning a type to the variable 'if_condition_289355' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'if_condition_289355', if_condition_289355)
        # SSA begins for if statement (line 345)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 346):
        
        # Assigning a BinOp to a Name (line 346):
        unicode_289356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 18), 'unicode', u'File verification command failed:\n%s\n')
        
        # Call to join(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'cmd' (line 346)
        cmd_289359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 71), 'cmd', False)
        # Processing the call keyword arguments (line 346)
        kwargs_289360 = {}
        unicode_289357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 62), 'unicode', u' ')
        # Obtaining the member 'join' of a type (line 346)
        join_289358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 62), unicode_289357, 'join')
        # Calling join(args, kwargs) (line 346)
        join_call_result_289361 = invoke(stypy.reporting.localization.Localization(__file__, 346, 62), join_289358, *[cmd_289359], **kwargs_289360)
        
        # Applying the binary operator '%' (line 346)
        result_mod_289362 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 18), '%', unicode_289356, join_call_result_289361)
        
        # Assigning a type to the variable 'msg' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'msg', result_mod_289362)
        
        # Getting the type of 'stdout' (line 347)
        stdout_289363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'stdout')
        # Testing the type of an if condition (line 347)
        if_condition_289364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 347, 12), stdout_289363)
        # Assigning a type to the variable 'if_condition_289364' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'if_condition_289364', if_condition_289364)
        # SSA begins for if statement (line 347)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'msg' (line 348)
        msg_289365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'msg')
        unicode_289366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 23), 'unicode', u'Standard output:\n%s\n')
        # Getting the type of 'stdout' (line 348)
        stdout_289367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 50), 'stdout')
        # Applying the binary operator '%' (line 348)
        result_mod_289368 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 23), '%', unicode_289366, stdout_289367)
        
        # Applying the binary operator '+=' (line 348)
        result_iadd_289369 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 16), '+=', msg_289365, result_mod_289368)
        # Assigning a type to the variable 'msg' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'msg', result_iadd_289369)
        
        # SSA join for if statement (line 347)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'stderr' (line 349)
        stderr_289370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'stderr')
        # Testing the type of an if condition (line 349)
        if_condition_289371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 12), stderr_289370)
        # Assigning a type to the variable 'if_condition_289371' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'if_condition_289371', if_condition_289371)
        # SSA begins for if statement (line 349)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'msg' (line 350)
        msg_289372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'msg')
        unicode_289373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 23), 'unicode', u'Standard error:\n%s\n')
        # Getting the type of 'stderr' (line 350)
        stderr_289374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 49), 'stderr')
        # Applying the binary operator '%' (line 350)
        result_mod_289375 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 23), '%', unicode_289373, stderr_289374)
        
        # Applying the binary operator '+=' (line 350)
        result_iadd_289376 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 16), '+=', msg_289372, result_mod_289375)
        # Assigning a type to the variable 'msg' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'msg', result_iadd_289376)
        
        # SSA join for if statement (line 349)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to IOError(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'msg' (line 351)
        msg_289378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'msg', False)
        # Processing the call keyword arguments (line 351)
        kwargs_289379 = {}
        # Getting the type of 'IOError' (line 351)
        IOError_289377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 18), 'IOError', False)
        # Calling IOError(args, kwargs) (line 351)
        IOError_call_result_289380 = invoke(stypy.reporting.localization.Localization(__file__, 351, 18), IOError_289377, *[msg_289378], **kwargs_289379)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 351, 12), IOError_call_result_289380, 'raise parameter', BaseException)
        # SSA join for if statement (line 345)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_289314:
            # SSA join for if statement (line 339)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'verify(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'verify' in the type store
    # Getting the type of 'stypy_return_type' (line 332)
    stypy_return_type_289381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289381)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'verify'
    return stypy_return_type_289381

# Assigning a type to the variable 'verify' (line 332)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'verify', verify)

@norecursion
def crop_to_same(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'crop_to_same'
    module_type_store = module_type_store.open_function_context('crop_to_same', 354, 0, False)
    
    # Passed parameters checking function
    crop_to_same.stypy_localization = localization
    crop_to_same.stypy_type_of_self = None
    crop_to_same.stypy_type_store = module_type_store
    crop_to_same.stypy_function_name = 'crop_to_same'
    crop_to_same.stypy_param_names_list = ['actual_path', 'actual_image', 'expected_path', 'expected_image']
    crop_to_same.stypy_varargs_param_name = None
    crop_to_same.stypy_kwargs_param_name = None
    crop_to_same.stypy_call_defaults = defaults
    crop_to_same.stypy_call_varargs = varargs
    crop_to_same.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'crop_to_same', ['actual_path', 'actual_image', 'expected_path', 'expected_image'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'crop_to_same', localization, ['actual_path', 'actual_image', 'expected_path', 'expected_image'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'crop_to_same(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_289382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 19), 'int')
    int_289383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 22), 'int')
    slice_289384 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 357, 7), int_289382, int_289383, None)
    # Getting the type of 'actual_path' (line 357)
    actual_path_289385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 7), 'actual_path')
    # Obtaining the member '__getitem__' of a type (line 357)
    getitem___289386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 7), actual_path_289385, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 357)
    subscript_call_result_289387 = invoke(stypy.reporting.localization.Localization(__file__, 357, 7), getitem___289386, slice_289384)
    
    unicode_289388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 29), 'unicode', u'eps')
    # Applying the binary operator '==' (line 357)
    result_eq_289389 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 7), '==', subscript_call_result_289387, unicode_289388)
    
    
    
    # Obtaining the type of the subscript
    int_289390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 53), 'int')
    int_289391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 56), 'int')
    slice_289392 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 357, 39), int_289390, int_289391, None)
    # Getting the type of 'expected_path' (line 357)
    expected_path_289393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 39), 'expected_path')
    # Obtaining the member '__getitem__' of a type (line 357)
    getitem___289394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 39), expected_path_289393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 357)
    subscript_call_result_289395 = invoke(stypy.reporting.localization.Localization(__file__, 357, 39), getitem___289394, slice_289392)
    
    unicode_289396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 63), 'unicode', u'pdf')
    # Applying the binary operator '==' (line 357)
    result_eq_289397 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 39), '==', subscript_call_result_289395, unicode_289396)
    
    # Applying the binary operator 'and' (line 357)
    result_and_keyword_289398 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 7), 'and', result_eq_289389, result_eq_289397)
    
    # Testing the type of an if condition (line 357)
    if_condition_289399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 4), result_and_keyword_289398)
    # Assigning a type to the variable 'if_condition_289399' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'if_condition_289399', if_condition_289399)
    # SSA begins for if statement (line 357)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Tuple (line 358):
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_289400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 8), 'int')
    # Getting the type of 'actual_image' (line 358)
    actual_image_289401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 21), 'actual_image')
    # Obtaining the member 'shape' of a type (line 358)
    shape_289402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 21), actual_image_289401, 'shape')
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___289403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), shape_289402, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_289404 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), getitem___289403, int_289400)
    
    # Assigning a type to the variable 'tuple_var_assignment_288346' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'tuple_var_assignment_288346', subscript_call_result_289404)
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_289405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 8), 'int')
    # Getting the type of 'actual_image' (line 358)
    actual_image_289406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 21), 'actual_image')
    # Obtaining the member 'shape' of a type (line 358)
    shape_289407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 21), actual_image_289406, 'shape')
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___289408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), shape_289407, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_289409 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), getitem___289408, int_289405)
    
    # Assigning a type to the variable 'tuple_var_assignment_288347' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'tuple_var_assignment_288347', subscript_call_result_289409)
    
    # Assigning a Subscript to a Name (line 358):
    
    # Obtaining the type of the subscript
    int_289410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 8), 'int')
    # Getting the type of 'actual_image' (line 358)
    actual_image_289411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 21), 'actual_image')
    # Obtaining the member 'shape' of a type (line 358)
    shape_289412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 21), actual_image_289411, 'shape')
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___289413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), shape_289412, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_289414 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), getitem___289413, int_289410)
    
    # Assigning a type to the variable 'tuple_var_assignment_288348' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'tuple_var_assignment_288348', subscript_call_result_289414)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_288346' (line 358)
    tuple_var_assignment_288346_289415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'tuple_var_assignment_288346')
    # Assigning a type to the variable 'aw' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'aw', tuple_var_assignment_288346_289415)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_288347' (line 358)
    tuple_var_assignment_288347_289416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'tuple_var_assignment_288347')
    # Assigning a type to the variable 'ah' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'ah', tuple_var_assignment_288347_289416)
    
    # Assigning a Name to a Name (line 358):
    # Getting the type of 'tuple_var_assignment_288348' (line 358)
    tuple_var_assignment_288348_289417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'tuple_var_assignment_288348')
    # Assigning a type to the variable 'ad' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'ad', tuple_var_assignment_288348_289417)
    
    # Assigning a Attribute to a Tuple (line 359):
    
    # Assigning a Subscript to a Name (line 359):
    
    # Obtaining the type of the subscript
    int_289418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 8), 'int')
    # Getting the type of 'expected_image' (line 359)
    expected_image_289419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 21), 'expected_image')
    # Obtaining the member 'shape' of a type (line 359)
    shape_289420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 21), expected_image_289419, 'shape')
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___289421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), shape_289420, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_289422 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), getitem___289421, int_289418)
    
    # Assigning a type to the variable 'tuple_var_assignment_288349' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_var_assignment_288349', subscript_call_result_289422)
    
    # Assigning a Subscript to a Name (line 359):
    
    # Obtaining the type of the subscript
    int_289423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 8), 'int')
    # Getting the type of 'expected_image' (line 359)
    expected_image_289424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 21), 'expected_image')
    # Obtaining the member 'shape' of a type (line 359)
    shape_289425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 21), expected_image_289424, 'shape')
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___289426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), shape_289425, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_289427 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), getitem___289426, int_289423)
    
    # Assigning a type to the variable 'tuple_var_assignment_288350' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_var_assignment_288350', subscript_call_result_289427)
    
    # Assigning a Subscript to a Name (line 359):
    
    # Obtaining the type of the subscript
    int_289428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 8), 'int')
    # Getting the type of 'expected_image' (line 359)
    expected_image_289429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 21), 'expected_image')
    # Obtaining the member 'shape' of a type (line 359)
    shape_289430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 21), expected_image_289429, 'shape')
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___289431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), shape_289430, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_289432 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), getitem___289431, int_289428)
    
    # Assigning a type to the variable 'tuple_var_assignment_288351' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_var_assignment_288351', subscript_call_result_289432)
    
    # Assigning a Name to a Name (line 359):
    # Getting the type of 'tuple_var_assignment_288349' (line 359)
    tuple_var_assignment_288349_289433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_var_assignment_288349')
    # Assigning a type to the variable 'ew' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'ew', tuple_var_assignment_288349_289433)
    
    # Assigning a Name to a Name (line 359):
    # Getting the type of 'tuple_var_assignment_288350' (line 359)
    tuple_var_assignment_288350_289434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_var_assignment_288350')
    # Assigning a type to the variable 'eh' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'eh', tuple_var_assignment_288350_289434)
    
    # Assigning a Name to a Name (line 359):
    # Getting the type of 'tuple_var_assignment_288351' (line 359)
    tuple_var_assignment_288351_289435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_var_assignment_288351')
    # Assigning a type to the variable 'ed' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'ed', tuple_var_assignment_288351_289435)
    
    # Assigning a Subscript to a Name (line 360):
    
    # Assigning a Subscript to a Name (line 360):
    
    # Obtaining the type of the subscript
    
    # Call to int(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'aw' (line 360)
    aw_289437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'aw', False)
    int_289438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 45), 'int')
    # Applying the binary operator 'div' (line 360)
    result_div_289439 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 40), 'div', aw_289437, int_289438)
    
    # Getting the type of 'ew' (line 360)
    ew_289440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 49), 'ew', False)
    int_289441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 54), 'int')
    # Applying the binary operator 'div' (line 360)
    result_div_289442 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 49), 'div', ew_289440, int_289441)
    
    # Applying the binary operator '-' (line 360)
    result_sub_289443 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 40), '-', result_div_289439, result_div_289442)
    
    # Processing the call keyword arguments (line 360)
    kwargs_289444 = {}
    # Getting the type of 'int' (line 360)
    int_289436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 36), 'int', False)
    # Calling int(args, kwargs) (line 360)
    int_call_result_289445 = invoke(stypy.reporting.localization.Localization(__file__, 360, 36), int_289436, *[result_sub_289443], **kwargs_289444)
    
    
    # Call to int(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'aw' (line 361)
    aw_289447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'aw', False)
    int_289448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 17), 'int')
    # Applying the binary operator 'div' (line 361)
    result_div_289449 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 12), 'div', aw_289447, int_289448)
    
    # Getting the type of 'ew' (line 361)
    ew_289450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 'ew', False)
    int_289451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 26), 'int')
    # Applying the binary operator 'div' (line 361)
    result_div_289452 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 21), 'div', ew_289450, int_289451)
    
    # Applying the binary operator '+' (line 361)
    result_add_289453 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 12), '+', result_div_289449, result_div_289452)
    
    # Processing the call keyword arguments (line 360)
    kwargs_289454 = {}
    # Getting the type of 'int' (line 360)
    int_289446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 57), 'int', False)
    # Calling int(args, kwargs) (line 360)
    int_call_result_289455 = invoke(stypy.reporting.localization.Localization(__file__, 360, 57), int_289446, *[result_add_289453], **kwargs_289454)
    
    slice_289456 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 360, 23), int_call_result_289445, int_call_result_289455, None)
    
    # Call to int(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'ah' (line 361)
    ah_289458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 34), 'ah', False)
    int_289459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 39), 'int')
    # Applying the binary operator 'div' (line 361)
    result_div_289460 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 34), 'div', ah_289458, int_289459)
    
    # Getting the type of 'eh' (line 361)
    eh_289461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 43), 'eh', False)
    int_289462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 48), 'int')
    # Applying the binary operator 'div' (line 361)
    result_div_289463 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 43), 'div', eh_289461, int_289462)
    
    # Applying the binary operator '-' (line 361)
    result_sub_289464 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 34), '-', result_div_289460, result_div_289463)
    
    # Processing the call keyword arguments (line 361)
    kwargs_289465 = {}
    # Getting the type of 'int' (line 361)
    int_289457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 30), 'int', False)
    # Calling int(args, kwargs) (line 361)
    int_call_result_289466 = invoke(stypy.reporting.localization.Localization(__file__, 361, 30), int_289457, *[result_sub_289464], **kwargs_289465)
    
    
    # Call to int(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'ah' (line 361)
    ah_289468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 55), 'ah', False)
    int_289469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 60), 'int')
    # Applying the binary operator 'div' (line 361)
    result_div_289470 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 55), 'div', ah_289468, int_289469)
    
    # Getting the type of 'eh' (line 361)
    eh_289471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 64), 'eh', False)
    int_289472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 69), 'int')
    # Applying the binary operator 'div' (line 361)
    result_div_289473 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 64), 'div', eh_289471, int_289472)
    
    # Applying the binary operator '+' (line 361)
    result_add_289474 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 55), '+', result_div_289470, result_div_289473)
    
    # Processing the call keyword arguments (line 361)
    kwargs_289475 = {}
    # Getting the type of 'int' (line 361)
    int_289467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 51), 'int', False)
    # Calling int(args, kwargs) (line 361)
    int_call_result_289476 = invoke(stypy.reporting.localization.Localization(__file__, 361, 51), int_289467, *[result_add_289474], **kwargs_289475)
    
    slice_289477 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 360, 23), int_call_result_289466, int_call_result_289476, None)
    # Getting the type of 'actual_image' (line 360)
    actual_image_289478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'actual_image')
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___289479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 23), actual_image_289478, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_289480 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), getitem___289479, (slice_289456, slice_289477))
    
    # Assigning a type to the variable 'actual_image' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'actual_image', subscript_call_result_289480)
    # SSA join for if statement (line 357)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 362)
    tuple_289481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 362)
    # Adding element type (line 362)
    # Getting the type of 'actual_image' (line 362)
    actual_image_289482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 11), 'actual_image')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 11), tuple_289481, actual_image_289482)
    # Adding element type (line 362)
    # Getting the type of 'expected_image' (line 362)
    expected_image_289483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 25), 'expected_image')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 11), tuple_289481, expected_image_289483)
    
    # Assigning a type to the variable 'stypy_return_type' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'stypy_return_type', tuple_289481)
    
    # ################# End of 'crop_to_same(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'crop_to_same' in the type store
    # Getting the type of 'stypy_return_type' (line 354)
    stypy_return_type_289484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289484)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'crop_to_same'
    return stypy_return_type_289484

# Assigning a type to the variable 'crop_to_same' (line 354)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'crop_to_same', crop_to_same)

@norecursion
def calculate_rms(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'calculate_rms'
    module_type_store = module_type_store.open_function_context('calculate_rms', 365, 0, False)
    
    # Passed parameters checking function
    calculate_rms.stypy_localization = localization
    calculate_rms.stypy_type_of_self = None
    calculate_rms.stypy_type_store = module_type_store
    calculate_rms.stypy_function_name = 'calculate_rms'
    calculate_rms.stypy_param_names_list = ['expectedImage', 'actualImage']
    calculate_rms.stypy_varargs_param_name = None
    calculate_rms.stypy_kwargs_param_name = None
    calculate_rms.stypy_call_defaults = defaults
    calculate_rms.stypy_call_varargs = varargs
    calculate_rms.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'calculate_rms', ['expectedImage', 'actualImage'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'calculate_rms', localization, ['expectedImage', 'actualImage'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'calculate_rms(...)' code ##################

    unicode_289485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 4), 'unicode', u'Calculate the per-pixel errors, then compute the root mean square error.')
    
    
    # Getting the type of 'expectedImage' (line 367)
    expectedImage_289486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 7), 'expectedImage')
    # Obtaining the member 'shape' of a type (line 367)
    shape_289487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 7), expectedImage_289486, 'shape')
    # Getting the type of 'actualImage' (line 367)
    actualImage_289488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 30), 'actualImage')
    # Obtaining the member 'shape' of a type (line 367)
    shape_289489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 30), actualImage_289488, 'shape')
    # Applying the binary operator '!=' (line 367)
    result_ne_289490 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 7), '!=', shape_289487, shape_289489)
    
    # Testing the type of an if condition (line 367)
    if_condition_289491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 4), result_ne_289490)
    # Assigning a type to the variable 'if_condition_289491' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'if_condition_289491', if_condition_289491)
    # SSA begins for if statement (line 367)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ImageComparisonFailure(...): (line 368)
    # Processing the call arguments (line 368)
    
    # Call to format(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'expectedImage' (line 370)
    expectedImage_289495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 37), 'expectedImage', False)
    # Obtaining the member 'shape' of a type (line 370)
    shape_289496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 37), expectedImage_289495, 'shape')
    # Getting the type of 'actualImage' (line 370)
    actualImage_289497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 58), 'actualImage', False)
    # Obtaining the member 'shape' of a type (line 370)
    shape_289498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 58), actualImage_289497, 'shape')
    # Processing the call keyword arguments (line 369)
    kwargs_289499 = {}
    unicode_289493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 12), 'unicode', u'Image sizes do not match expected size: {0} actual size {1}')
    # Obtaining the member 'format' of a type (line 369)
    format_289494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 12), unicode_289493, 'format')
    # Calling format(args, kwargs) (line 369)
    format_call_result_289500 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), format_289494, *[shape_289496, shape_289498], **kwargs_289499)
    
    # Processing the call keyword arguments (line 368)
    kwargs_289501 = {}
    # Getting the type of 'ImageComparisonFailure' (line 368)
    ImageComparisonFailure_289492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 14), 'ImageComparisonFailure', False)
    # Calling ImageComparisonFailure(args, kwargs) (line 368)
    ImageComparisonFailure_call_result_289502 = invoke(stypy.reporting.localization.Localization(__file__, 368, 14), ImageComparisonFailure_289492, *[format_call_result_289500], **kwargs_289501)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 368, 8), ImageComparisonFailure_call_result_289502, 'raise parameter', BaseException)
    # SSA join for if statement (line 367)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 371):
    
    # Assigning a Attribute to a Name (line 371):
    # Getting the type of 'expectedImage' (line 371)
    expectedImage_289503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 17), 'expectedImage')
    # Obtaining the member 'size' of a type (line 371)
    size_289504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 17), expectedImage_289503, 'size')
    # Assigning a type to the variable 'num_values' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'num_values', size_289504)
    
    # Assigning a Call to a Name (line 372):
    
    # Assigning a Call to a Name (line 372):
    
    # Call to abs(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'expectedImage' (line 372)
    expectedImage_289506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 25), 'expectedImage', False)
    # Getting the type of 'actualImage' (line 372)
    actualImage_289507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 41), 'actualImage', False)
    # Applying the binary operator '-' (line 372)
    result_sub_289508 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 25), '-', expectedImage_289506, actualImage_289507)
    
    # Processing the call keyword arguments (line 372)
    kwargs_289509 = {}
    # Getting the type of 'abs' (line 372)
    abs_289505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 21), 'abs', False)
    # Calling abs(args, kwargs) (line 372)
    abs_call_result_289510 = invoke(stypy.reporting.localization.Localization(__file__, 372, 21), abs_289505, *[result_sub_289508], **kwargs_289509)
    
    # Assigning a type to the variable 'abs_diff_image' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'abs_diff_image', abs_call_result_289510)
    
    # Assigning a Call to a Name (line 373):
    
    # Assigning a Call to a Name (line 373):
    
    # Call to bincount(...): (line 373)
    # Processing the call arguments (line 373)
    
    # Call to ravel(...): (line 373)
    # Processing the call keyword arguments (line 373)
    kwargs_289515 = {}
    # Getting the type of 'abs_diff_image' (line 373)
    abs_diff_image_289513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 28), 'abs_diff_image', False)
    # Obtaining the member 'ravel' of a type (line 373)
    ravel_289514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 28), abs_diff_image_289513, 'ravel')
    # Calling ravel(args, kwargs) (line 373)
    ravel_call_result_289516 = invoke(stypy.reporting.localization.Localization(__file__, 373, 28), ravel_289514, *[], **kwargs_289515)
    
    # Processing the call keyword arguments (line 373)
    int_289517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 62), 'int')
    keyword_289518 = int_289517
    kwargs_289519 = {'minlength': keyword_289518}
    # Getting the type of 'np' (line 373)
    np_289511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'np', False)
    # Obtaining the member 'bincount' of a type (line 373)
    bincount_289512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 16), np_289511, 'bincount')
    # Calling bincount(args, kwargs) (line 373)
    bincount_call_result_289520 = invoke(stypy.reporting.localization.Localization(__file__, 373, 16), bincount_289512, *[ravel_call_result_289516], **kwargs_289519)
    
    # Assigning a type to the variable 'histogram' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'histogram', bincount_call_result_289520)
    
    # Assigning a Call to a Name (line 374):
    
    # Assigning a Call to a Name (line 374):
    
    # Call to sum(...): (line 374)
    # Processing the call arguments (line 374)
    # Getting the type of 'histogram' (line 374)
    histogram_289523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'histogram', False)
    
    # Call to arange(...): (line 374)
    # Processing the call arguments (line 374)
    
    # Call to len(...): (line 374)
    # Processing the call arguments (line 374)
    # Getting the type of 'histogram' (line 374)
    histogram_289527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 54), 'histogram', False)
    # Processing the call keyword arguments (line 374)
    kwargs_289528 = {}
    # Getting the type of 'len' (line 374)
    len_289526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 50), 'len', False)
    # Calling len(args, kwargs) (line 374)
    len_call_result_289529 = invoke(stypy.reporting.localization.Localization(__file__, 374, 50), len_289526, *[histogram_289527], **kwargs_289528)
    
    # Processing the call keyword arguments (line 374)
    kwargs_289530 = {}
    # Getting the type of 'np' (line 374)
    np_289524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 40), 'np', False)
    # Obtaining the member 'arange' of a type (line 374)
    arange_289525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 40), np_289524, 'arange')
    # Calling arange(args, kwargs) (line 374)
    arange_call_result_289531 = invoke(stypy.reporting.localization.Localization(__file__, 374, 40), arange_289525, *[len_call_result_289529], **kwargs_289530)
    
    int_289532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 69), 'int')
    # Applying the binary operator '**' (line 374)
    result_pow_289533 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 40), '**', arange_call_result_289531, int_289532)
    
    # Applying the binary operator '*' (line 374)
    result_mul_289534 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 28), '*', histogram_289523, result_pow_289533)
    
    # Processing the call keyword arguments (line 374)
    kwargs_289535 = {}
    # Getting the type of 'np' (line 374)
    np_289521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 21), 'np', False)
    # Obtaining the member 'sum' of a type (line 374)
    sum_289522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 21), np_289521, 'sum')
    # Calling sum(args, kwargs) (line 374)
    sum_call_result_289536 = invoke(stypy.reporting.localization.Localization(__file__, 374, 21), sum_289522, *[result_mul_289534], **kwargs_289535)
    
    # Assigning a type to the variable 'sum_of_squares' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'sum_of_squares', sum_call_result_289536)
    
    # Assigning a Call to a Name (line 375):
    
    # Assigning a Call to a Name (line 375):
    
    # Call to sqrt(...): (line 375)
    # Processing the call arguments (line 375)
    
    # Call to float(...): (line 375)
    # Processing the call arguments (line 375)
    # Getting the type of 'sum_of_squares' (line 375)
    sum_of_squares_289540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 24), 'sum_of_squares', False)
    # Processing the call keyword arguments (line 375)
    kwargs_289541 = {}
    # Getting the type of 'float' (line 375)
    float_289539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 18), 'float', False)
    # Calling float(args, kwargs) (line 375)
    float_call_result_289542 = invoke(stypy.reporting.localization.Localization(__file__, 375, 18), float_289539, *[sum_of_squares_289540], **kwargs_289541)
    
    # Getting the type of 'num_values' (line 375)
    num_values_289543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 42), 'num_values', False)
    # Applying the binary operator 'div' (line 375)
    result_div_289544 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 18), 'div', float_call_result_289542, num_values_289543)
    
    # Processing the call keyword arguments (line 375)
    kwargs_289545 = {}
    # Getting the type of 'np' (line 375)
    np_289537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 10), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 375)
    sqrt_289538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 10), np_289537, 'sqrt')
    # Calling sqrt(args, kwargs) (line 375)
    sqrt_call_result_289546 = invoke(stypy.reporting.localization.Localization(__file__, 375, 10), sqrt_289538, *[result_div_289544], **kwargs_289545)
    
    # Assigning a type to the variable 'rms' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'rms', sqrt_call_result_289546)
    # Getting the type of 'rms' (line 376)
    rms_289547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 11), 'rms')
    # Assigning a type to the variable 'stypy_return_type' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'stypy_return_type', rms_289547)
    
    # ################# End of 'calculate_rms(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'calculate_rms' in the type store
    # Getting the type of 'stypy_return_type' (line 365)
    stypy_return_type_289548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289548)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'calculate_rms'
    return stypy_return_type_289548

# Assigning a type to the variable 'calculate_rms' (line 365)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 0), 'calculate_rms', calculate_rms)

@norecursion
def compare_images(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 379)
    False_289549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 55), 'False')
    defaults = [False_289549]
    # Create a new context for function 'compare_images'
    module_type_store = module_type_store.open_function_context('compare_images', 379, 0, False)
    
    # Passed parameters checking function
    compare_images.stypy_localization = localization
    compare_images.stypy_type_of_self = None
    compare_images.stypy_type_store = module_type_store
    compare_images.stypy_function_name = 'compare_images'
    compare_images.stypy_param_names_list = ['expected', 'actual', 'tol', 'in_decorator']
    compare_images.stypy_varargs_param_name = None
    compare_images.stypy_kwargs_param_name = None
    compare_images.stypy_call_defaults = defaults
    compare_images.stypy_call_varargs = varargs
    compare_images.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compare_images', ['expected', 'actual', 'tol', 'in_decorator'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compare_images', localization, ['expected', 'actual', 'tol', 'in_decorator'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compare_images(...)' code ##################

    unicode_289550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, (-1)), 'unicode', u'\n    Compare two "image" files checking differences within a tolerance.\n\n    The two given filenames may point to files which are convertible to\n    PNG via the `.converter` dictionary. The underlying RMS is calculated\n    with the `.calculate_rms` function.\n\n    Parameters\n    ----------\n    expected : str\n        The filename of the expected image.\n    actual :str\n        The filename of the actual image.\n    tol : float\n        The tolerance (a color value difference, where 255 is the\n        maximal difference).  The test fails if the average pixel\n        difference is greater than this value.\n    in_decorator : bool\n        If called from image_comparison decorator, this should be\n        True. (default=False)\n\n    Examples\n    --------\n    img1 = "./baseline/plot.png"\n    img2 = "./output/plot.png"\n    compare_images( img1, img2, 0.001 ):\n\n    ')
    
    
    
    # Call to exists(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'actual' (line 408)
    actual_289554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 26), 'actual', False)
    # Processing the call keyword arguments (line 408)
    kwargs_289555 = {}
    # Getting the type of 'os' (line 408)
    os_289551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 408)
    path_289552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 11), os_289551, 'path')
    # Obtaining the member 'exists' of a type (line 408)
    exists_289553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 11), path_289552, 'exists')
    # Calling exists(args, kwargs) (line 408)
    exists_call_result_289556 = invoke(stypy.reporting.localization.Localization(__file__, 408, 11), exists_289553, *[actual_289554], **kwargs_289555)
    
    # Applying the 'not' unary operator (line 408)
    result_not__289557 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 7), 'not', exists_call_result_289556)
    
    # Testing the type of an if condition (line 408)
    if_condition_289558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 408, 4), result_not__289557)
    # Assigning a type to the variable 'if_condition_289558' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'if_condition_289558', if_condition_289558)
    # SSA begins for if statement (line 408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 409)
    # Processing the call arguments (line 409)
    unicode_289560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 24), 'unicode', u'Output image %s does not exist.')
    # Getting the type of 'actual' (line 409)
    actual_289561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 60), 'actual', False)
    # Applying the binary operator '%' (line 409)
    result_mod_289562 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 24), '%', unicode_289560, actual_289561)
    
    # Processing the call keyword arguments (line 409)
    kwargs_289563 = {}
    # Getting the type of 'Exception' (line 409)
    Exception_289559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 409)
    Exception_call_result_289564 = invoke(stypy.reporting.localization.Localization(__file__, 409, 14), Exception_289559, *[result_mod_289562], **kwargs_289563)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 409, 8), Exception_call_result_289564, 'raise parameter', BaseException)
    # SSA join for if statement (line 408)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to stat(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'actual' (line 411)
    actual_289567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 15), 'actual', False)
    # Processing the call keyword arguments (line 411)
    kwargs_289568 = {}
    # Getting the type of 'os' (line 411)
    os_289565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 7), 'os', False)
    # Obtaining the member 'stat' of a type (line 411)
    stat_289566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 7), os_289565, 'stat')
    # Calling stat(args, kwargs) (line 411)
    stat_call_result_289569 = invoke(stypy.reporting.localization.Localization(__file__, 411, 7), stat_289566, *[actual_289567], **kwargs_289568)
    
    # Obtaining the member 'st_size' of a type (line 411)
    st_size_289570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 7), stat_call_result_289569, 'st_size')
    int_289571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 34), 'int')
    # Applying the binary operator '==' (line 411)
    result_eq_289572 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 7), '==', st_size_289570, int_289571)
    
    # Testing the type of an if condition (line 411)
    if_condition_289573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 4), result_eq_289572)
    # Assigning a type to the variable 'if_condition_289573' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'if_condition_289573', if_condition_289573)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 412)
    # Processing the call arguments (line 412)
    unicode_289575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 24), 'unicode', u'Output image file %s is empty.')
    # Getting the type of 'actual' (line 412)
    actual_289576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 59), 'actual', False)
    # Applying the binary operator '%' (line 412)
    result_mod_289577 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 24), '%', unicode_289575, actual_289576)
    
    # Processing the call keyword arguments (line 412)
    kwargs_289578 = {}
    # Getting the type of 'Exception' (line 412)
    Exception_289574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 412)
    Exception_call_result_289579 = invoke(stypy.reporting.localization.Localization(__file__, 412, 14), Exception_289574, *[result_mod_289577], **kwargs_289578)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 412, 8), Exception_call_result_289579, 'raise parameter', BaseException)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 415):
    
    # Assigning a Subscript to a Name (line 415):
    
    # Obtaining the type of the subscript
    int_289580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 36), 'int')
    
    # Call to split(...): (line 415)
    # Processing the call arguments (line 415)
    unicode_289583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 31), 'unicode', u'.')
    # Processing the call keyword arguments (line 415)
    kwargs_289584 = {}
    # Getting the type of 'expected' (line 415)
    expected_289581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'expected', False)
    # Obtaining the member 'split' of a type (line 415)
    split_289582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 16), expected_289581, 'split')
    # Calling split(args, kwargs) (line 415)
    split_call_result_289585 = invoke(stypy.reporting.localization.Localization(__file__, 415, 16), split_289582, *[unicode_289583], **kwargs_289584)
    
    # Obtaining the member '__getitem__' of a type (line 415)
    getitem___289586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 16), split_call_result_289585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 415)
    subscript_call_result_289587 = invoke(stypy.reporting.localization.Localization(__file__, 415, 16), getitem___289586, int_289580)
    
    # Assigning a type to the variable 'extension' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'extension', subscript_call_result_289587)
    
    
    
    # Call to exists(...): (line 417)
    # Processing the call arguments (line 417)
    # Getting the type of 'expected' (line 417)
    expected_289591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 26), 'expected', False)
    # Processing the call keyword arguments (line 417)
    kwargs_289592 = {}
    # Getting the type of 'os' (line 417)
    os_289588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 417)
    path_289589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 11), os_289588, 'path')
    # Obtaining the member 'exists' of a type (line 417)
    exists_289590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 11), path_289589, 'exists')
    # Calling exists(args, kwargs) (line 417)
    exists_call_result_289593 = invoke(stypy.reporting.localization.Localization(__file__, 417, 11), exists_289590, *[expected_289591], **kwargs_289592)
    
    # Applying the 'not' unary operator (line 417)
    result_not__289594 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 7), 'not', exists_call_result_289593)
    
    # Testing the type of an if condition (line 417)
    if_condition_289595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 4), result_not__289594)
    # Assigning a type to the variable 'if_condition_289595' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'if_condition_289595', if_condition_289595)
    # SSA begins for if statement (line 417)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to IOError(...): (line 418)
    # Processing the call arguments (line 418)
    unicode_289597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 22), 'unicode', u'Baseline image %r does not exist.')
    # Getting the type of 'expected' (line 418)
    expected_289598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 60), 'expected', False)
    # Applying the binary operator '%' (line 418)
    result_mod_289599 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 22), '%', unicode_289597, expected_289598)
    
    # Processing the call keyword arguments (line 418)
    kwargs_289600 = {}
    # Getting the type of 'IOError' (line 418)
    IOError_289596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 14), 'IOError', False)
    # Calling IOError(args, kwargs) (line 418)
    IOError_call_result_289601 = invoke(stypy.reporting.localization.Localization(__file__, 418, 14), IOError_289596, *[result_mod_289599], **kwargs_289600)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 418, 8), IOError_call_result_289601, 'raise parameter', BaseException)
    # SSA join for if statement (line 417)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'extension' (line 420)
    extension_289602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 7), 'extension')
    unicode_289603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 20), 'unicode', u'png')
    # Applying the binary operator '!=' (line 420)
    result_ne_289604 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 7), '!=', extension_289602, unicode_289603)
    
    # Testing the type of an if condition (line 420)
    if_condition_289605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 4), result_ne_289604)
    # Assigning a type to the variable 'if_condition_289605' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'if_condition_289605', if_condition_289605)
    # SSA begins for if statement (line 420)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 421):
    
    # Assigning a Call to a Name (line 421):
    
    # Call to convert(...): (line 421)
    # Processing the call arguments (line 421)
    # Getting the type of 'actual' (line 421)
    actual_289607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 25), 'actual', False)
    # Getting the type of 'False' (line 421)
    False_289608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 33), 'False', False)
    # Processing the call keyword arguments (line 421)
    kwargs_289609 = {}
    # Getting the type of 'convert' (line 421)
    convert_289606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 17), 'convert', False)
    # Calling convert(args, kwargs) (line 421)
    convert_call_result_289610 = invoke(stypy.reporting.localization.Localization(__file__, 421, 17), convert_289606, *[actual_289607, False_289608], **kwargs_289609)
    
    # Assigning a type to the variable 'actual' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'actual', convert_call_result_289610)
    
    # Assigning a Call to a Name (line 422):
    
    # Assigning a Call to a Name (line 422):
    
    # Call to convert(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'expected' (line 422)
    expected_289612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 27), 'expected', False)
    # Getting the type of 'True' (line 422)
    True_289613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 37), 'True', False)
    # Processing the call keyword arguments (line 422)
    kwargs_289614 = {}
    # Getting the type of 'convert' (line 422)
    convert_289611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 19), 'convert', False)
    # Calling convert(args, kwargs) (line 422)
    convert_call_result_289615 = invoke(stypy.reporting.localization.Localization(__file__, 422, 19), convert_289611, *[expected_289612, True_289613], **kwargs_289614)
    
    # Assigning a type to the variable 'expected' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'expected', convert_call_result_289615)
    # SSA join for if statement (line 420)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 425):
    
    # Assigning a Call to a Name (line 425):
    
    # Call to read_png_int(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'expected' (line 425)
    expected_289618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 38), 'expected', False)
    # Processing the call keyword arguments (line 425)
    kwargs_289619 = {}
    # Getting the type of '_png' (line 425)
    _png_289616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), '_png', False)
    # Obtaining the member 'read_png_int' of a type (line 425)
    read_png_int_289617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 20), _png_289616, 'read_png_int')
    # Calling read_png_int(args, kwargs) (line 425)
    read_png_int_call_result_289620 = invoke(stypy.reporting.localization.Localization(__file__, 425, 20), read_png_int_289617, *[expected_289618], **kwargs_289619)
    
    # Assigning a type to the variable 'expectedImage' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'expectedImage', read_png_int_call_result_289620)
    
    # Assigning a Call to a Name (line 426):
    
    # Assigning a Call to a Name (line 426):
    
    # Call to read_png_int(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'actual' (line 426)
    actual_289623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 36), 'actual', False)
    # Processing the call keyword arguments (line 426)
    kwargs_289624 = {}
    # Getting the type of '_png' (line 426)
    _png_289621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 18), '_png', False)
    # Obtaining the member 'read_png_int' of a type (line 426)
    read_png_int_289622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 18), _png_289621, 'read_png_int')
    # Calling read_png_int(args, kwargs) (line 426)
    read_png_int_call_result_289625 = invoke(stypy.reporting.localization.Localization(__file__, 426, 18), read_png_int_289622, *[actual_289623], **kwargs_289624)
    
    # Assigning a type to the variable 'actualImage' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'actualImage', read_png_int_call_result_289625)
    
    # Assigning a Subscript to a Name (line 427):
    
    # Assigning a Subscript to a Name (line 427):
    
    # Obtaining the type of the subscript
    slice_289626 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 427, 20), None, None, None)
    slice_289627 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 427, 20), None, None, None)
    int_289628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 41), 'int')
    slice_289629 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 427, 20), None, int_289628, None)
    # Getting the type of 'expectedImage' (line 427)
    expectedImage_289630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 20), 'expectedImage')
    # Obtaining the member '__getitem__' of a type (line 427)
    getitem___289631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 20), expectedImage_289630, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 427)
    subscript_call_result_289632 = invoke(stypy.reporting.localization.Localization(__file__, 427, 20), getitem___289631, (slice_289626, slice_289627, slice_289629))
    
    # Assigning a type to the variable 'expectedImage' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'expectedImage', subscript_call_result_289632)
    
    # Assigning a Subscript to a Name (line 428):
    
    # Assigning a Subscript to a Name (line 428):
    
    # Obtaining the type of the subscript
    slice_289633 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 428, 18), None, None, None)
    slice_289634 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 428, 18), None, None, None)
    int_289635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 37), 'int')
    slice_289636 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 428, 18), None, int_289635, None)
    # Getting the type of 'actualImage' (line 428)
    actualImage_289637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 18), 'actualImage')
    # Obtaining the member '__getitem__' of a type (line 428)
    getitem___289638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 18), actualImage_289637, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
    subscript_call_result_289639 = invoke(stypy.reporting.localization.Localization(__file__, 428, 18), getitem___289638, (slice_289633, slice_289634, slice_289636))
    
    # Assigning a type to the variable 'actualImage' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'actualImage', subscript_call_result_289639)
    
    # Assigning a Call to a Tuple (line 430):
    
    # Assigning a Call to a Name:
    
    # Call to crop_to_same(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'actual' (line 431)
    actual_289641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'actual', False)
    # Getting the type of 'actualImage' (line 431)
    actualImage_289642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'actualImage', False)
    # Getting the type of 'expected' (line 431)
    expected_289643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 29), 'expected', False)
    # Getting the type of 'expectedImage' (line 431)
    expectedImage_289644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 39), 'expectedImage', False)
    # Processing the call keyword arguments (line 430)
    kwargs_289645 = {}
    # Getting the type of 'crop_to_same' (line 430)
    crop_to_same_289640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 33), 'crop_to_same', False)
    # Calling crop_to_same(args, kwargs) (line 430)
    crop_to_same_call_result_289646 = invoke(stypy.reporting.localization.Localization(__file__, 430, 33), crop_to_same_289640, *[actual_289641, actualImage_289642, expected_289643, expectedImage_289644], **kwargs_289645)
    
    # Assigning a type to the variable 'call_assignment_288352' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'call_assignment_288352', crop_to_same_call_result_289646)
    
    # Assigning a Call to a Name (line 430):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_289649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 4), 'int')
    # Processing the call keyword arguments
    kwargs_289650 = {}
    # Getting the type of 'call_assignment_288352' (line 430)
    call_assignment_288352_289647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'call_assignment_288352', False)
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___289648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 4), call_assignment_288352_289647, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_289651 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289648, *[int_289649], **kwargs_289650)
    
    # Assigning a type to the variable 'call_assignment_288353' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'call_assignment_288353', getitem___call_result_289651)
    
    # Assigning a Name to a Name (line 430):
    # Getting the type of 'call_assignment_288353' (line 430)
    call_assignment_288353_289652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'call_assignment_288353')
    # Assigning a type to the variable 'actualImage' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'actualImage', call_assignment_288353_289652)
    
    # Assigning a Call to a Name (line 430):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_289655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 4), 'int')
    # Processing the call keyword arguments
    kwargs_289656 = {}
    # Getting the type of 'call_assignment_288352' (line 430)
    call_assignment_288352_289653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'call_assignment_288352', False)
    # Obtaining the member '__getitem__' of a type (line 430)
    getitem___289654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 4), call_assignment_288352_289653, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_289657 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289654, *[int_289655], **kwargs_289656)
    
    # Assigning a type to the variable 'call_assignment_288354' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'call_assignment_288354', getitem___call_result_289657)
    
    # Assigning a Name to a Name (line 430):
    # Getting the type of 'call_assignment_288354' (line 430)
    call_assignment_288354_289658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'call_assignment_288354')
    # Assigning a type to the variable 'expectedImage' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 17), 'expectedImage', call_assignment_288354_289658)
    
    # Assigning a Call to a Name (line 433):
    
    # Assigning a Call to a Name (line 433):
    
    # Call to make_test_filename(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'actual' (line 433)
    actual_289660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 36), 'actual', False)
    unicode_289661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 44), 'unicode', u'failed-diff')
    # Processing the call keyword arguments (line 433)
    kwargs_289662 = {}
    # Getting the type of 'make_test_filename' (line 433)
    make_test_filename_289659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 17), 'make_test_filename', False)
    # Calling make_test_filename(args, kwargs) (line 433)
    make_test_filename_call_result_289663 = invoke(stypy.reporting.localization.Localization(__file__, 433, 17), make_test_filename_289659, *[actual_289660, unicode_289661], **kwargs_289662)
    
    # Assigning a type to the variable 'diff_image' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'diff_image', make_test_filename_call_result_289663)
    
    
    # Getting the type of 'tol' (line 435)
    tol_289664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 7), 'tol')
    float_289665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 14), 'float')
    # Applying the binary operator '<=' (line 435)
    result_le_289666 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 7), '<=', tol_289664, float_289665)
    
    # Testing the type of an if condition (line 435)
    if_condition_289667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 4), result_le_289666)
    # Assigning a type to the variable 'if_condition_289667' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'if_condition_289667', if_condition_289667)
    # SSA begins for if statement (line 435)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to array_equal(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'expectedImage' (line 436)
    expectedImage_289670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 26), 'expectedImage', False)
    # Getting the type of 'actualImage' (line 436)
    actualImage_289671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 41), 'actualImage', False)
    # Processing the call keyword arguments (line 436)
    kwargs_289672 = {}
    # Getting the type of 'np' (line 436)
    np_289668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 436)
    array_equal_289669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 11), np_289668, 'array_equal')
    # Calling array_equal(args, kwargs) (line 436)
    array_equal_call_result_289673 = invoke(stypy.reporting.localization.Localization(__file__, 436, 11), array_equal_289669, *[expectedImage_289670, actualImage_289671], **kwargs_289672)
    
    # Testing the type of an if condition (line 436)
    if_condition_289674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 8), array_equal_call_result_289673)
    # Assigning a type to the variable 'if_condition_289674' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'if_condition_289674', if_condition_289674)
    # SSA begins for if statement (line 436)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 437)
    None_289675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'stypy_return_type', None_289675)
    # SSA join for if statement (line 436)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 435)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to astype(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'np' (line 441)
    np_289678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 41), 'np', False)
    # Obtaining the member 'int16' of a type (line 441)
    int16_289679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 41), np_289678, 'int16')
    # Processing the call keyword arguments (line 441)
    kwargs_289680 = {}
    # Getting the type of 'expectedImage' (line 441)
    expectedImage_289676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'expectedImage', False)
    # Obtaining the member 'astype' of a type (line 441)
    astype_289677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 20), expectedImage_289676, 'astype')
    # Calling astype(args, kwargs) (line 441)
    astype_call_result_289681 = invoke(stypy.reporting.localization.Localization(__file__, 441, 20), astype_289677, *[int16_289679], **kwargs_289680)
    
    # Assigning a type to the variable 'expectedImage' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'expectedImage', astype_call_result_289681)
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to astype(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'np' (line 442)
    np_289684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 37), 'np', False)
    # Obtaining the member 'int16' of a type (line 442)
    int16_289685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 37), np_289684, 'int16')
    # Processing the call keyword arguments (line 442)
    kwargs_289686 = {}
    # Getting the type of 'actualImage' (line 442)
    actualImage_289682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 18), 'actualImage', False)
    # Obtaining the member 'astype' of a type (line 442)
    astype_289683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 18), actualImage_289682, 'astype')
    # Calling astype(args, kwargs) (line 442)
    astype_call_result_289687 = invoke(stypy.reporting.localization.Localization(__file__, 442, 18), astype_289683, *[int16_289685], **kwargs_289686)
    
    # Assigning a type to the variable 'actualImage' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'actualImage', astype_call_result_289687)
    
    # Assigning a Call to a Name (line 444):
    
    # Assigning a Call to a Name (line 444):
    
    # Call to calculate_rms(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'expectedImage' (line 444)
    expectedImage_289689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 24), 'expectedImage', False)
    # Getting the type of 'actualImage' (line 444)
    actualImage_289690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 39), 'actualImage', False)
    # Processing the call keyword arguments (line 444)
    kwargs_289691 = {}
    # Getting the type of 'calculate_rms' (line 444)
    calculate_rms_289688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 10), 'calculate_rms', False)
    # Calling calculate_rms(args, kwargs) (line 444)
    calculate_rms_call_result_289692 = invoke(stypy.reporting.localization.Localization(__file__, 444, 10), calculate_rms_289688, *[expectedImage_289689, actualImage_289690], **kwargs_289691)
    
    # Assigning a type to the variable 'rms' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'rms', calculate_rms_call_result_289692)
    
    
    # Getting the type of 'rms' (line 446)
    rms_289693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 7), 'rms')
    # Getting the type of 'tol' (line 446)
    tol_289694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 14), 'tol')
    # Applying the binary operator '<=' (line 446)
    result_le_289695 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 7), '<=', rms_289693, tol_289694)
    
    # Testing the type of an if condition (line 446)
    if_condition_289696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 446, 4), result_le_289695)
    # Assigning a type to the variable 'if_condition_289696' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'if_condition_289696', if_condition_289696)
    # SSA begins for if statement (line 446)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 447)
    None_289697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'stypy_return_type', None_289697)
    # SSA join for if statement (line 446)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to save_diff_image(...): (line 449)
    # Processing the call arguments (line 449)
    # Getting the type of 'expected' (line 449)
    expected_289699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 20), 'expected', False)
    # Getting the type of 'actual' (line 449)
    actual_289700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 30), 'actual', False)
    # Getting the type of 'diff_image' (line 449)
    diff_image_289701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 38), 'diff_image', False)
    # Processing the call keyword arguments (line 449)
    kwargs_289702 = {}
    # Getting the type of 'save_diff_image' (line 449)
    save_diff_image_289698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'save_diff_image', False)
    # Calling save_diff_image(args, kwargs) (line 449)
    save_diff_image_call_result_289703 = invoke(stypy.reporting.localization.Localization(__file__, 449, 4), save_diff_image_289698, *[expected_289699, actual_289700, diff_image_289701], **kwargs_289702)
    
    
    # Assigning a Call to a Name (line 451):
    
    # Assigning a Call to a Name (line 451):
    
    # Call to dict(...): (line 451)
    # Processing the call keyword arguments (line 451)
    # Getting the type of 'rms' (line 451)
    rms_289705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'rms', False)
    keyword_289706 = rms_289705
    
    # Call to str(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'expected' (line 451)
    expected_289708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 41), 'expected', False)
    # Processing the call keyword arguments (line 451)
    kwargs_289709 = {}
    # Getting the type of 'str' (line 451)
    str_289707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 37), 'str', False)
    # Calling str(args, kwargs) (line 451)
    str_call_result_289710 = invoke(stypy.reporting.localization.Localization(__file__, 451, 37), str_289707, *[expected_289708], **kwargs_289709)
    
    keyword_289711 = str_call_result_289710
    
    # Call to str(...): (line 452)
    # Processing the call arguments (line 452)
    # Getting the type of 'actual' (line 452)
    actual_289713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 30), 'actual', False)
    # Processing the call keyword arguments (line 452)
    kwargs_289714 = {}
    # Getting the type of 'str' (line 452)
    str_289712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 26), 'str', False)
    # Calling str(args, kwargs) (line 452)
    str_call_result_289715 = invoke(stypy.reporting.localization.Localization(__file__, 452, 26), str_289712, *[actual_289713], **kwargs_289714)
    
    keyword_289716 = str_call_result_289715
    
    # Call to str(...): (line 452)
    # Processing the call arguments (line 452)
    # Getting the type of 'diff_image' (line 452)
    diff_image_289718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 48), 'diff_image', False)
    # Processing the call keyword arguments (line 452)
    kwargs_289719 = {}
    # Getting the type of 'str' (line 452)
    str_289717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 44), 'str', False)
    # Calling str(args, kwargs) (line 452)
    str_call_result_289720 = invoke(stypy.reporting.localization.Localization(__file__, 452, 44), str_289717, *[diff_image_289718], **kwargs_289719)
    
    keyword_289721 = str_call_result_289720
    # Getting the type of 'tol' (line 452)
    tol_289722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 65), 'tol', False)
    keyword_289723 = tol_289722
    kwargs_289724 = {'expected': keyword_289711, 'rms': keyword_289706, 'actual': keyword_289716, 'tol': keyword_289723, 'diff': keyword_289721}
    # Getting the type of 'dict' (line 451)
    dict_289704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 14), 'dict', False)
    # Calling dict(args, kwargs) (line 451)
    dict_call_result_289725 = invoke(stypy.reporting.localization.Localization(__file__, 451, 14), dict_289704, *[], **kwargs_289724)
    
    # Assigning a type to the variable 'results' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'results', dict_call_result_289725)
    
    
    # Getting the type of 'in_decorator' (line 454)
    in_decorator_289726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 11), 'in_decorator')
    # Applying the 'not' unary operator (line 454)
    result_not__289727 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 7), 'not', in_decorator_289726)
    
    # Testing the type of an if condition (line 454)
    if_condition_289728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 4), result_not__289727)
    # Assigning a type to the variable 'if_condition_289728' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'if_condition_289728', if_condition_289728)
    # SSA begins for if statement (line 454)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 456):
    
    # Assigning a List to a Name (line 456):
    
    # Obtaining an instance of the builtin type 'list' (line 456)
    list_289729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 456)
    # Adding element type (line 456)
    unicode_289730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 20), 'unicode', u'Error: Image files did not match.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 19), list_289729, unicode_289730)
    # Adding element type (line 456)
    unicode_289731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 20), 'unicode', u'RMS Value: {rms}')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 19), list_289729, unicode_289731)
    # Adding element type (line 456)
    unicode_289732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 20), 'unicode', u'Expected:  \n    {expected}')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 19), list_289729, unicode_289732)
    # Adding element type (line 456)
    unicode_289733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 20), 'unicode', u'Actual:    \n    {actual}')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 19), list_289729, unicode_289733)
    # Adding element type (line 456)
    unicode_289734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 20), 'unicode', u'Difference:\n    {diff}')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 19), list_289729, unicode_289734)
    # Adding element type (line 456)
    unicode_289735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 20), 'unicode', u'Tolerance: \n    {tol}')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 19), list_289729, unicode_289735)
    
    # Assigning a type to the variable 'template' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'template', list_289729)
    
    # Assigning a Call to a Name (line 462):
    
    # Assigning a Call to a Name (line 462):
    
    # Call to join(...): (line 462)
    # Processing the call arguments (line 462)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'template' (line 462)
    template_289743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 66), 'template', False)
    comprehension_289744 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 31), template_289743)
    # Assigning a type to the variable 'line' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 31), 'line', comprehension_289744)
    
    # Call to format(...): (line 462)
    # Processing the call keyword arguments (line 462)
    # Getting the type of 'results' (line 462)
    results_289740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 45), 'results', False)
    kwargs_289741 = {'results_289740': results_289740}
    # Getting the type of 'line' (line 462)
    line_289738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 31), 'line', False)
    # Obtaining the member 'format' of a type (line 462)
    format_289739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 31), line_289738, 'format')
    # Calling format(args, kwargs) (line 462)
    format_call_result_289742 = invoke(stypy.reporting.localization.Localization(__file__, 462, 31), format_289739, *[], **kwargs_289741)
    
    list_289745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 31), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 31), list_289745, format_call_result_289742)
    # Processing the call keyword arguments (line 462)
    kwargs_289746 = {}
    unicode_289736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 18), 'unicode', u'\n  ')
    # Obtaining the member 'join' of a type (line 462)
    join_289737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 18), unicode_289736, 'join')
    # Calling join(args, kwargs) (line 462)
    join_call_result_289747 = invoke(stypy.reporting.localization.Localization(__file__, 462, 18), join_289737, *[list_289745], **kwargs_289746)
    
    # Assigning a type to the variable 'results' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'results', join_call_result_289747)
    # SSA join for if statement (line 454)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'results' (line 463)
    results_289748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'results')
    # Assigning a type to the variable 'stypy_return_type' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type', results_289748)
    
    # ################# End of 'compare_images(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compare_images' in the type store
    # Getting the type of 'stypy_return_type' (line 379)
    stypy_return_type_289749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289749)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compare_images'
    return stypy_return_type_289749

# Assigning a type to the variable 'compare_images' (line 379)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 0), 'compare_images', compare_images)

@norecursion
def save_diff_image(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'save_diff_image'
    module_type_store = module_type_store.open_function_context('save_diff_image', 466, 0, False)
    
    # Passed parameters checking function
    save_diff_image.stypy_localization = localization
    save_diff_image.stypy_type_of_self = None
    save_diff_image.stypy_type_store = module_type_store
    save_diff_image.stypy_function_name = 'save_diff_image'
    save_diff_image.stypy_param_names_list = ['expected', 'actual', 'output']
    save_diff_image.stypy_varargs_param_name = None
    save_diff_image.stypy_kwargs_param_name = None
    save_diff_image.stypy_call_defaults = defaults
    save_diff_image.stypy_call_varargs = varargs
    save_diff_image.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'save_diff_image', ['expected', 'actual', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'save_diff_image', localization, ['expected', 'actual', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'save_diff_image(...)' code ##################

    
    # Assigning a Call to a Name (line 467):
    
    # Assigning a Call to a Name (line 467):
    
    # Call to read_png(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'expected' (line 467)
    expected_289752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 34), 'expected', False)
    # Processing the call keyword arguments (line 467)
    kwargs_289753 = {}
    # Getting the type of '_png' (line 467)
    _png_289750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 20), '_png', False)
    # Obtaining the member 'read_png' of a type (line 467)
    read_png_289751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 20), _png_289750, 'read_png')
    # Calling read_png(args, kwargs) (line 467)
    read_png_call_result_289754 = invoke(stypy.reporting.localization.Localization(__file__, 467, 20), read_png_289751, *[expected_289752], **kwargs_289753)
    
    # Assigning a type to the variable 'expectedImage' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'expectedImage', read_png_call_result_289754)
    
    # Assigning a Call to a Name (line 468):
    
    # Assigning a Call to a Name (line 468):
    
    # Call to read_png(...): (line 468)
    # Processing the call arguments (line 468)
    # Getting the type of 'actual' (line 468)
    actual_289757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 32), 'actual', False)
    # Processing the call keyword arguments (line 468)
    kwargs_289758 = {}
    # Getting the type of '_png' (line 468)
    _png_289755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 18), '_png', False)
    # Obtaining the member 'read_png' of a type (line 468)
    read_png_289756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 18), _png_289755, 'read_png')
    # Calling read_png(args, kwargs) (line 468)
    read_png_call_result_289759 = invoke(stypy.reporting.localization.Localization(__file__, 468, 18), read_png_289756, *[actual_289757], **kwargs_289758)
    
    # Assigning a type to the variable 'actualImage' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'actualImage', read_png_call_result_289759)
    
    # Assigning a Call to a Tuple (line 469):
    
    # Assigning a Call to a Name:
    
    # Call to crop_to_same(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of 'actual' (line 470)
    actual_289761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'actual', False)
    # Getting the type of 'actualImage' (line 470)
    actualImage_289762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 16), 'actualImage', False)
    # Getting the type of 'expected' (line 470)
    expected_289763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 29), 'expected', False)
    # Getting the type of 'expectedImage' (line 470)
    expectedImage_289764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 39), 'expectedImage', False)
    # Processing the call keyword arguments (line 469)
    kwargs_289765 = {}
    # Getting the type of 'crop_to_same' (line 469)
    crop_to_same_289760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 33), 'crop_to_same', False)
    # Calling crop_to_same(args, kwargs) (line 469)
    crop_to_same_call_result_289766 = invoke(stypy.reporting.localization.Localization(__file__, 469, 33), crop_to_same_289760, *[actual_289761, actualImage_289762, expected_289763, expectedImage_289764], **kwargs_289765)
    
    # Assigning a type to the variable 'call_assignment_288355' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'call_assignment_288355', crop_to_same_call_result_289766)
    
    # Assigning a Call to a Name (line 469):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_289769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 4), 'int')
    # Processing the call keyword arguments
    kwargs_289770 = {}
    # Getting the type of 'call_assignment_288355' (line 469)
    call_assignment_288355_289767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'call_assignment_288355', False)
    # Obtaining the member '__getitem__' of a type (line 469)
    getitem___289768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 4), call_assignment_288355_289767, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_289771 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289768, *[int_289769], **kwargs_289770)
    
    # Assigning a type to the variable 'call_assignment_288356' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'call_assignment_288356', getitem___call_result_289771)
    
    # Assigning a Name to a Name (line 469):
    # Getting the type of 'call_assignment_288356' (line 469)
    call_assignment_288356_289772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'call_assignment_288356')
    # Assigning a type to the variable 'actualImage' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'actualImage', call_assignment_288356_289772)
    
    # Assigning a Call to a Name (line 469):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_289775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 4), 'int')
    # Processing the call keyword arguments
    kwargs_289776 = {}
    # Getting the type of 'call_assignment_288355' (line 469)
    call_assignment_288355_289773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'call_assignment_288355', False)
    # Obtaining the member '__getitem__' of a type (line 469)
    getitem___289774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 4), call_assignment_288355_289773, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_289777 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___289774, *[int_289775], **kwargs_289776)
    
    # Assigning a type to the variable 'call_assignment_288357' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'call_assignment_288357', getitem___call_result_289777)
    
    # Assigning a Name to a Name (line 469):
    # Getting the type of 'call_assignment_288357' (line 469)
    call_assignment_288357_289778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'call_assignment_288357')
    # Assigning a type to the variable 'expectedImage' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 17), 'expectedImage', call_assignment_288357_289778)
    
    # Assigning a Call to a Name (line 471):
    
    # Assigning a Call to a Name (line 471):
    
    # Call to astype(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'float' (line 471)
    float_289785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 51), 'float', False)
    # Processing the call keyword arguments (line 471)
    kwargs_289786 = {}
    
    # Call to array(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'expectedImage' (line 471)
    expectedImage_289781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 29), 'expectedImage', False)
    # Processing the call keyword arguments (line 471)
    kwargs_289782 = {}
    # Getting the type of 'np' (line 471)
    np_289779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 20), 'np', False)
    # Obtaining the member 'array' of a type (line 471)
    array_289780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 20), np_289779, 'array')
    # Calling array(args, kwargs) (line 471)
    array_call_result_289783 = invoke(stypy.reporting.localization.Localization(__file__, 471, 20), array_289780, *[expectedImage_289781], **kwargs_289782)
    
    # Obtaining the member 'astype' of a type (line 471)
    astype_289784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 20), array_call_result_289783, 'astype')
    # Calling astype(args, kwargs) (line 471)
    astype_call_result_289787 = invoke(stypy.reporting.localization.Localization(__file__, 471, 20), astype_289784, *[float_289785], **kwargs_289786)
    
    # Assigning a type to the variable 'expectedImage' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'expectedImage', astype_call_result_289787)
    
    # Assigning a Call to a Name (line 472):
    
    # Assigning a Call to a Name (line 472):
    
    # Call to astype(...): (line 472)
    # Processing the call arguments (line 472)
    # Getting the type of 'float' (line 472)
    float_289794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 47), 'float', False)
    # Processing the call keyword arguments (line 472)
    kwargs_289795 = {}
    
    # Call to array(...): (line 472)
    # Processing the call arguments (line 472)
    # Getting the type of 'actualImage' (line 472)
    actualImage_289790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 27), 'actualImage', False)
    # Processing the call keyword arguments (line 472)
    kwargs_289791 = {}
    # Getting the type of 'np' (line 472)
    np_289788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 18), 'np', False)
    # Obtaining the member 'array' of a type (line 472)
    array_289789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 18), np_289788, 'array')
    # Calling array(args, kwargs) (line 472)
    array_call_result_289792 = invoke(stypy.reporting.localization.Localization(__file__, 472, 18), array_289789, *[actualImage_289790], **kwargs_289791)
    
    # Obtaining the member 'astype' of a type (line 472)
    astype_289793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 18), array_call_result_289792, 'astype')
    # Calling astype(args, kwargs) (line 472)
    astype_call_result_289796 = invoke(stypy.reporting.localization.Localization(__file__, 472, 18), astype_289793, *[float_289794], **kwargs_289795)
    
    # Assigning a type to the variable 'actualImage' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'actualImage', astype_call_result_289796)
    
    
    # Getting the type of 'expectedImage' (line 473)
    expectedImage_289797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 7), 'expectedImage')
    # Obtaining the member 'shape' of a type (line 473)
    shape_289798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 7), expectedImage_289797, 'shape')
    # Getting the type of 'actualImage' (line 473)
    actualImage_289799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 30), 'actualImage')
    # Obtaining the member 'shape' of a type (line 473)
    shape_289800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 30), actualImage_289799, 'shape')
    # Applying the binary operator '!=' (line 473)
    result_ne_289801 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 7), '!=', shape_289798, shape_289800)
    
    # Testing the type of an if condition (line 473)
    if_condition_289802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 473, 4), result_ne_289801)
    # Assigning a type to the variable 'if_condition_289802' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'if_condition_289802', if_condition_289802)
    # SSA begins for if statement (line 473)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ImageComparisonFailure(...): (line 474)
    # Processing the call arguments (line 474)
    
    # Call to format(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'expectedImage' (line 476)
    expectedImage_289806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 37), 'expectedImage', False)
    # Obtaining the member 'shape' of a type (line 476)
    shape_289807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 37), expectedImage_289806, 'shape')
    # Getting the type of 'actualImage' (line 476)
    actualImage_289808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 58), 'actualImage', False)
    # Obtaining the member 'shape' of a type (line 476)
    shape_289809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 58), actualImage_289808, 'shape')
    # Processing the call keyword arguments (line 475)
    kwargs_289810 = {}
    unicode_289804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'unicode', u'Image sizes do not match expected size: {0} actual size {1}')
    # Obtaining the member 'format' of a type (line 475)
    format_289805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), unicode_289804, 'format')
    # Calling format(args, kwargs) (line 475)
    format_call_result_289811 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), format_289805, *[shape_289807, shape_289809], **kwargs_289810)
    
    # Processing the call keyword arguments (line 474)
    kwargs_289812 = {}
    # Getting the type of 'ImageComparisonFailure' (line 474)
    ImageComparisonFailure_289803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 14), 'ImageComparisonFailure', False)
    # Calling ImageComparisonFailure(args, kwargs) (line 474)
    ImageComparisonFailure_call_result_289813 = invoke(stypy.reporting.localization.Localization(__file__, 474, 14), ImageComparisonFailure_289803, *[format_call_result_289811], **kwargs_289812)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 474, 8), ImageComparisonFailure_call_result_289813, 'raise parameter', BaseException)
    # SSA join for if statement (line 473)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 477):
    
    # Assigning a Call to a Name (line 477):
    
    # Call to abs(...): (line 477)
    # Processing the call arguments (line 477)
    # Getting the type of 'expectedImage' (line 477)
    expectedImage_289816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 26), 'expectedImage', False)
    # Getting the type of 'actualImage' (line 477)
    actualImage_289817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 42), 'actualImage', False)
    # Applying the binary operator '-' (line 477)
    result_sub_289818 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 26), '-', expectedImage_289816, actualImage_289817)
    
    # Processing the call keyword arguments (line 477)
    kwargs_289819 = {}
    # Getting the type of 'np' (line 477)
    np_289814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 19), 'np', False)
    # Obtaining the member 'abs' of a type (line 477)
    abs_289815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 19), np_289814, 'abs')
    # Calling abs(args, kwargs) (line 477)
    abs_call_result_289820 = invoke(stypy.reporting.localization.Localization(__file__, 477, 19), abs_289815, *[result_sub_289818], **kwargs_289819)
    
    # Assigning a type to the variable 'absDiffImage' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'absDiffImage', abs_call_result_289820)
    
    # Getting the type of 'absDiffImage' (line 480)
    absDiffImage_289821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'absDiffImage')
    int_289822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 20), 'int')
    int_289823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 26), 'int')
    # Applying the binary operator '*' (line 480)
    result_mul_289824 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 20), '*', int_289822, int_289823)
    
    # Applying the binary operator '*=' (line 480)
    result_imul_289825 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 4), '*=', absDiffImage_289821, result_mul_289824)
    # Assigning a type to the variable 'absDiffImage' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'absDiffImage', result_imul_289825)
    
    
    # Assigning a Call to a Name (line 481):
    
    # Assigning a Call to a Name (line 481):
    
    # Call to astype(...): (line 481)
    # Processing the call arguments (line 481)
    # Getting the type of 'np' (line 481)
    np_289834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 57), 'np', False)
    # Obtaining the member 'uint8' of a type (line 481)
    uint8_289835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 57), np_289834, 'uint8')
    # Processing the call keyword arguments (line 481)
    kwargs_289836 = {}
    
    # Call to clip(...): (line 481)
    # Processing the call arguments (line 481)
    # Getting the type of 'absDiffImage' (line 481)
    absDiffImage_289828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 28), 'absDiffImage', False)
    int_289829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 42), 'int')
    int_289830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 45), 'int')
    # Processing the call keyword arguments (line 481)
    kwargs_289831 = {}
    # Getting the type of 'np' (line 481)
    np_289826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 20), 'np', False)
    # Obtaining the member 'clip' of a type (line 481)
    clip_289827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 20), np_289826, 'clip')
    # Calling clip(args, kwargs) (line 481)
    clip_call_result_289832 = invoke(stypy.reporting.localization.Localization(__file__, 481, 20), clip_289827, *[absDiffImage_289828, int_289829, int_289830], **kwargs_289831)
    
    # Obtaining the member 'astype' of a type (line 481)
    astype_289833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 20), clip_call_result_289832, 'astype')
    # Calling astype(args, kwargs) (line 481)
    astype_call_result_289837 = invoke(stypy.reporting.localization.Localization(__file__, 481, 20), astype_289833, *[uint8_289835], **kwargs_289836)
    
    # Assigning a type to the variable 'save_image_np' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'save_image_np', astype_call_result_289837)
    
    # Assigning a Attribute to a Tuple (line 482):
    
    # Assigning a Subscript to a Name (line 482):
    
    # Obtaining the type of the subscript
    int_289838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 4), 'int')
    # Getting the type of 'save_image_np' (line 482)
    save_image_np_289839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 27), 'save_image_np')
    # Obtaining the member 'shape' of a type (line 482)
    shape_289840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 27), save_image_np_289839, 'shape')
    # Obtaining the member '__getitem__' of a type (line 482)
    getitem___289841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 4), shape_289840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 482)
    subscript_call_result_289842 = invoke(stypy.reporting.localization.Localization(__file__, 482, 4), getitem___289841, int_289838)
    
    # Assigning a type to the variable 'tuple_var_assignment_288358' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'tuple_var_assignment_288358', subscript_call_result_289842)
    
    # Assigning a Subscript to a Name (line 482):
    
    # Obtaining the type of the subscript
    int_289843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 4), 'int')
    # Getting the type of 'save_image_np' (line 482)
    save_image_np_289844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 27), 'save_image_np')
    # Obtaining the member 'shape' of a type (line 482)
    shape_289845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 27), save_image_np_289844, 'shape')
    # Obtaining the member '__getitem__' of a type (line 482)
    getitem___289846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 4), shape_289845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 482)
    subscript_call_result_289847 = invoke(stypy.reporting.localization.Localization(__file__, 482, 4), getitem___289846, int_289843)
    
    # Assigning a type to the variable 'tuple_var_assignment_288359' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'tuple_var_assignment_288359', subscript_call_result_289847)
    
    # Assigning a Subscript to a Name (line 482):
    
    # Obtaining the type of the subscript
    int_289848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 4), 'int')
    # Getting the type of 'save_image_np' (line 482)
    save_image_np_289849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 27), 'save_image_np')
    # Obtaining the member 'shape' of a type (line 482)
    shape_289850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 27), save_image_np_289849, 'shape')
    # Obtaining the member '__getitem__' of a type (line 482)
    getitem___289851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 4), shape_289850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 482)
    subscript_call_result_289852 = invoke(stypy.reporting.localization.Localization(__file__, 482, 4), getitem___289851, int_289848)
    
    # Assigning a type to the variable 'tuple_var_assignment_288360' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'tuple_var_assignment_288360', subscript_call_result_289852)
    
    # Assigning a Name to a Name (line 482):
    # Getting the type of 'tuple_var_assignment_288358' (line 482)
    tuple_var_assignment_288358_289853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'tuple_var_assignment_288358')
    # Assigning a type to the variable 'height' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'height', tuple_var_assignment_288358_289853)
    
    # Assigning a Name to a Name (line 482):
    # Getting the type of 'tuple_var_assignment_288359' (line 482)
    tuple_var_assignment_288359_289854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'tuple_var_assignment_288359')
    # Assigning a type to the variable 'width' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'width', tuple_var_assignment_288359_289854)
    
    # Assigning a Name to a Name (line 482):
    # Getting the type of 'tuple_var_assignment_288360' (line 482)
    tuple_var_assignment_288360_289855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'tuple_var_assignment_288360')
    # Assigning a type to the variable 'depth' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 19), 'depth', tuple_var_assignment_288360_289855)
    
    
    # Getting the type of 'depth' (line 486)
    depth_289856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 7), 'depth')
    int_289857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 16), 'int')
    # Applying the binary operator '==' (line 486)
    result_eq_289858 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 7), '==', depth_289856, int_289857)
    
    # Testing the type of an if condition (line 486)
    if_condition_289859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 486, 4), result_eq_289858)
    # Assigning a type to the variable 'if_condition_289859' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'if_condition_289859', if_condition_289859)
    # SSA begins for if statement (line 486)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 487):
    
    # Assigning a Call to a Name (line 487):
    
    # Call to empty(...): (line 487)
    # Processing the call arguments (line 487)
    
    # Obtaining an instance of the builtin type 'tuple' (line 487)
    tuple_289862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 487)
    # Adding element type (line 487)
    # Getting the type of 'height' (line 487)
    height_289863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 31), 'height', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 31), tuple_289862, height_289863)
    # Adding element type (line 487)
    # Getting the type of 'width' (line 487)
    width_289864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 39), 'width', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 31), tuple_289862, width_289864)
    # Adding element type (line 487)
    int_289865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 31), tuple_289862, int_289865)
    
    # Processing the call keyword arguments (line 487)
    # Getting the type of 'np' (line 487)
    np_289866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 56), 'np', False)
    # Obtaining the member 'uint8' of a type (line 487)
    uint8_289867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 56), np_289866, 'uint8')
    keyword_289868 = uint8_289867
    kwargs_289869 = {'dtype': keyword_289868}
    # Getting the type of 'np' (line 487)
    np_289860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 21), 'np', False)
    # Obtaining the member 'empty' of a type (line 487)
    empty_289861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 21), np_289860, 'empty')
    # Calling empty(args, kwargs) (line 487)
    empty_call_result_289870 = invoke(stypy.reporting.localization.Localization(__file__, 487, 21), empty_289861, *[tuple_289862], **kwargs_289869)
    
    # Assigning a type to the variable 'with_alpha' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'with_alpha', empty_call_result_289870)
    
    # Assigning a Name to a Subscript (line 488):
    
    # Assigning a Name to a Subscript (line 488):
    # Getting the type of 'save_image_np' (line 488)
    save_image_np_289871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 32), 'save_image_np')
    # Getting the type of 'with_alpha' (line 488)
    with_alpha_289872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'with_alpha')
    slice_289873 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 488, 8), None, None, None)
    slice_289874 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 488, 8), None, None, None)
    int_289875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 25), 'int')
    int_289876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 27), 'int')
    slice_289877 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 488, 8), int_289875, int_289876, None)
    # Storing an element on a container (line 488)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 488, 8), with_alpha_289872, ((slice_289873, slice_289874, slice_289877), save_image_np_289871))
    
    # Assigning a Name to a Name (line 489):
    
    # Assigning a Name to a Name (line 489):
    # Getting the type of 'with_alpha' (line 489)
    with_alpha_289878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 24), 'with_alpha')
    # Assigning a type to the variable 'save_image_np' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'save_image_np', with_alpha_289878)
    # SSA join for if statement (line 486)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 492):
    
    # Assigning a Num to a Subscript (line 492):
    int_289879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 29), 'int')
    # Getting the type of 'save_image_np' (line 492)
    save_image_np_289880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'save_image_np')
    slice_289881 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 492, 4), None, None, None)
    slice_289882 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 492, 4), None, None, None)
    int_289883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 24), 'int')
    # Storing an element on a container (line 492)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 4), save_image_np_289880, ((slice_289881, slice_289882, int_289883), int_289879))
    
    # Call to write_png(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of 'save_image_np' (line 494)
    save_image_np_289886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 19), 'save_image_np', False)
    # Getting the type of 'output' (line 494)
    output_289887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 34), 'output', False)
    # Processing the call keyword arguments (line 494)
    kwargs_289888 = {}
    # Getting the type of '_png' (line 494)
    _png_289884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), '_png', False)
    # Obtaining the member 'write_png' of a type (line 494)
    write_png_289885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 4), _png_289884, 'write_png')
    # Calling write_png(args, kwargs) (line 494)
    write_png_call_result_289889 = invoke(stypy.reporting.localization.Localization(__file__, 494, 4), write_png_289885, *[save_image_np_289886, output_289887], **kwargs_289888)
    
    
    # ################# End of 'save_diff_image(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'save_diff_image' in the type store
    # Getting the type of 'stypy_return_type' (line 466)
    stypy_return_type_289890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289890)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'save_diff_image'
    return stypy_return_type_289890

# Assigning a type to the variable 'save_diff_image' (line 466)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'save_diff_image', save_diff_image)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
