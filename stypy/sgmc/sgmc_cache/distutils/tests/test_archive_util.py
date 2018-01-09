
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 -*-
2: '''Tests for distutils.archive_util.'''
3: __revision__ = "$Id$"
4: 
5: import unittest
6: import os
7: import sys
8: import tarfile
9: from os.path import splitdrive
10: import warnings
11: 
12: from distutils.archive_util import (check_archive_formats, make_tarball,
13:                                     make_zipfile, make_archive,
14:                                     ARCHIVE_FORMATS)
15: from distutils.spawn import find_executable, spawn
16: from distutils.tests import support
17: from test.test_support import check_warnings, run_unittest
18: 
19: try:
20:     import grp
21:     import pwd
22:     UID_GID_SUPPORT = True
23: except ImportError:
24:     UID_GID_SUPPORT = False
25: 
26: try:
27:     import zipfile
28:     ZIP_SUPPORT = True
29: except ImportError:
30:     ZIP_SUPPORT = find_executable('zip')
31: 
32: # some tests will fail if zlib is not available
33: try:
34:     import zlib
35: except ImportError:
36:     zlib = None
37: 
38: def can_fs_encode(filename):
39:     '''
40:     Return True if the filename can be saved in the file system.
41:     '''
42:     if os.path.supports_unicode_filenames:
43:         return True
44:     try:
45:         filename.encode(sys.getfilesystemencoding())
46:     except UnicodeEncodeError:
47:         return False
48:     return True
49: 
50: 
51: class ArchiveUtilTestCase(support.TempdirManager,
52:                           support.LoggingSilencer,
53:                           unittest.TestCase):
54: 
55:     @unittest.skipUnless(zlib, "requires zlib")
56:     def test_make_tarball(self):
57:         self._make_tarball('archive')
58: 
59:     def _make_tarball(self, target_name):
60:         # creating something to tar
61:         tmpdir = self.mkdtemp()
62:         self.write_file([tmpdir, 'file1'], 'xxx')
63:         self.write_file([tmpdir, 'file2'], 'xxx')
64:         os.mkdir(os.path.join(tmpdir, 'sub'))
65:         self.write_file([tmpdir, 'sub', 'file3'], 'xxx')
66: 
67:         tmpdir2 = self.mkdtemp()
68:         unittest.skipUnless(splitdrive(tmpdir)[0] == splitdrive(tmpdir2)[0],
69:                             "source and target should be on same drive")
70: 
71:         base_name = os.path.join(tmpdir2, target_name)
72: 
73:         # working with relative paths to avoid tar warnings
74:         old_dir = os.getcwd()
75:         os.chdir(tmpdir)
76:         try:
77:             make_tarball(splitdrive(base_name)[1], '.')
78:         finally:
79:             os.chdir(old_dir)
80: 
81:         # check if the compressed tarball was created
82:         tarball = base_name + '.tar.gz'
83:         self.assertTrue(os.path.exists(tarball))
84: 
85:         # trying an uncompressed one
86:         base_name = os.path.join(tmpdir2, target_name)
87:         old_dir = os.getcwd()
88:         os.chdir(tmpdir)
89:         try:
90:             make_tarball(splitdrive(base_name)[1], '.', compress=None)
91:         finally:
92:             os.chdir(old_dir)
93:         tarball = base_name + '.tar'
94:         self.assertTrue(os.path.exists(tarball))
95: 
96:     def _tarinfo(self, path):
97:         tar = tarfile.open(path)
98:         try:
99:             names = tar.getnames()
100:             names.sort()
101:             return tuple(names)
102:         finally:
103:             tar.close()
104: 
105:     def _create_files(self):
106:         # creating something to tar
107:         tmpdir = self.mkdtemp()
108:         dist = os.path.join(tmpdir, 'dist')
109:         os.mkdir(dist)
110:         self.write_file([dist, 'file1'], 'xxx')
111:         self.write_file([dist, 'file2'], 'xxx')
112:         os.mkdir(os.path.join(dist, 'sub'))
113:         self.write_file([dist, 'sub', 'file3'], 'xxx')
114:         os.mkdir(os.path.join(dist, 'sub2'))
115:         tmpdir2 = self.mkdtemp()
116:         base_name = os.path.join(tmpdir2, 'archive')
117:         return tmpdir, tmpdir2, base_name
118: 
119:     @unittest.skipUnless(zlib, "Requires zlib")
120:     @unittest.skipUnless(find_executable('tar') and find_executable('gzip'),
121:                          'Need the tar command to run')
122:     def test_tarfile_vs_tar(self):
123:         tmpdir, tmpdir2, base_name =  self._create_files()
124:         old_dir = os.getcwd()
125:         os.chdir(tmpdir)
126:         try:
127:             make_tarball(base_name, 'dist')
128:         finally:
129:             os.chdir(old_dir)
130: 
131:         # check if the compressed tarball was created
132:         tarball = base_name + '.tar.gz'
133:         self.assertTrue(os.path.exists(tarball))
134: 
135:         # now create another tarball using `tar`
136:         tarball2 = os.path.join(tmpdir, 'archive2.tar.gz')
137:         tar_cmd = ['tar', '-cf', 'archive2.tar', 'dist']
138:         gzip_cmd = ['gzip', '-f9', 'archive2.tar']
139:         old_dir = os.getcwd()
140:         os.chdir(tmpdir)
141:         try:
142:             spawn(tar_cmd)
143:             spawn(gzip_cmd)
144:         finally:
145:             os.chdir(old_dir)
146: 
147:         self.assertTrue(os.path.exists(tarball2))
148:         # let's compare both tarballs
149:         self.assertEqual(self._tarinfo(tarball), self._tarinfo(tarball2))
150: 
151:         # trying an uncompressed one
152:         base_name = os.path.join(tmpdir2, 'archive')
153:         old_dir = os.getcwd()
154:         os.chdir(tmpdir)
155:         try:
156:             make_tarball(base_name, 'dist', compress=None)
157:         finally:
158:             os.chdir(old_dir)
159:         tarball = base_name + '.tar'
160:         self.assertTrue(os.path.exists(tarball))
161: 
162:         # now for a dry_run
163:         base_name = os.path.join(tmpdir2, 'archive')
164:         old_dir = os.getcwd()
165:         os.chdir(tmpdir)
166:         try:
167:             make_tarball(base_name, 'dist', compress=None, dry_run=True)
168:         finally:
169:             os.chdir(old_dir)
170:         tarball = base_name + '.tar'
171:         self.assertTrue(os.path.exists(tarball))
172: 
173:     @unittest.skipUnless(find_executable('compress'),
174:                          'The compress program is required')
175:     def test_compress_deprecated(self):
176:         tmpdir, tmpdir2, base_name =  self._create_files()
177: 
178:         # using compress and testing the PendingDeprecationWarning
179:         old_dir = os.getcwd()
180:         os.chdir(tmpdir)
181:         try:
182:             with check_warnings() as w:
183:                 warnings.simplefilter("always")
184:                 make_tarball(base_name, 'dist', compress='compress')
185:         finally:
186:             os.chdir(old_dir)
187:         tarball = base_name + '.tar.Z'
188:         self.assertTrue(os.path.exists(tarball))
189:         self.assertEqual(len(w.warnings), 1)
190: 
191:         # same test with dry_run
192:         os.remove(tarball)
193:         old_dir = os.getcwd()
194:         os.chdir(tmpdir)
195:         try:
196:             with check_warnings() as w:
197:                 warnings.simplefilter("always")
198:                 make_tarball(base_name, 'dist', compress='compress',
199:                              dry_run=True)
200:         finally:
201:             os.chdir(old_dir)
202:         self.assertFalse(os.path.exists(tarball))
203:         self.assertEqual(len(w.warnings), 1)
204: 
205:     @unittest.skipUnless(zlib, "Requires zlib")
206:     @unittest.skipUnless(ZIP_SUPPORT, 'Need zip support to run')
207:     def test_make_zipfile(self):
208:         # creating something to tar
209:         tmpdir = self.mkdtemp()
210:         self.write_file([tmpdir, 'file1'], 'xxx')
211:         self.write_file([tmpdir, 'file2'], 'xxx')
212: 
213:         tmpdir2 = self.mkdtemp()
214:         base_name = os.path.join(tmpdir2, 'archive')
215:         make_zipfile(base_name, tmpdir)
216: 
217:         # check if the compressed tarball was created
218:         tarball = base_name + '.zip'
219: 
220:     def test_check_archive_formats(self):
221:         self.assertEqual(check_archive_formats(['gztar', 'xxx', 'zip']),
222:                          'xxx')
223:         self.assertEqual(check_archive_formats(['gztar', 'zip']), None)
224: 
225:     def test_make_archive(self):
226:         tmpdir = self.mkdtemp()
227:         base_name = os.path.join(tmpdir, 'archive')
228:         self.assertRaises(ValueError, make_archive, base_name, 'xxx')
229: 
230:     @unittest.skipUnless(zlib, "Requires zlib")
231:     def test_make_archive_owner_group(self):
232:         # testing make_archive with owner and group, with various combinations
233:         # this works even if there's not gid/uid support
234:         if UID_GID_SUPPORT:
235:             group = grp.getgrgid(0)[0]
236:             owner = pwd.getpwuid(0)[0]
237:         else:
238:             group = owner = 'root'
239: 
240:         base_dir, root_dir, base_name =  self._create_files()
241:         base_name = os.path.join(self.mkdtemp() , 'archive')
242:         res = make_archive(base_name, 'zip', root_dir, base_dir, owner=owner,
243:                            group=group)
244:         self.assertTrue(os.path.exists(res))
245: 
246:         res = make_archive(base_name, 'zip', root_dir, base_dir)
247:         self.assertTrue(os.path.exists(res))
248: 
249:         res = make_archive(base_name, 'tar', root_dir, base_dir,
250:                            owner=owner, group=group)
251:         self.assertTrue(os.path.exists(res))
252: 
253:         res = make_archive(base_name, 'tar', root_dir, base_dir,
254:                            owner='kjhkjhkjg', group='oihohoh')
255:         self.assertTrue(os.path.exists(res))
256: 
257:     @unittest.skipUnless(zlib, "Requires zlib")
258:     @unittest.skipUnless(UID_GID_SUPPORT, "Requires grp and pwd support")
259:     def test_tarfile_root_owner(self):
260:         tmpdir, tmpdir2, base_name =  self._create_files()
261:         old_dir = os.getcwd()
262:         os.chdir(tmpdir)
263:         group = grp.getgrgid(0)[0]
264:         owner = pwd.getpwuid(0)[0]
265:         try:
266:             archive_name = make_tarball(base_name, 'dist', compress=None,
267:                                         owner=owner, group=group)
268:         finally:
269:             os.chdir(old_dir)
270: 
271:         # check if the compressed tarball was created
272:         self.assertTrue(os.path.exists(archive_name))
273: 
274:         # now checks the rights
275:         archive = tarfile.open(archive_name)
276:         try:
277:             for member in archive.getmembers():
278:                 self.assertEqual(member.uid, 0)
279:                 self.assertEqual(member.gid, 0)
280:         finally:
281:             archive.close()
282: 
283:     def test_make_archive_cwd(self):
284:         current_dir = os.getcwd()
285:         def _breaks(*args, **kw):
286:             raise RuntimeError()
287:         ARCHIVE_FORMATS['xxx'] = (_breaks, [], 'xxx file')
288:         try:
289:             try:
290:                 make_archive('xxx', 'xxx', root_dir=self.mkdtemp())
291:             except:
292:                 pass
293:             self.assertEqual(os.getcwd(), current_dir)
294:         finally:
295:             del ARCHIVE_FORMATS['xxx']
296: 
297:     @unittest.skipUnless(zlib, "requires zlib")
298:     def test_make_tarball_unicode(self):
299:         '''
300:         Mirror test_make_tarball, except filename is unicode.
301:         '''
302:         self._make_tarball(u'archive')
303: 
304:     @unittest.skipUnless(zlib, "requires zlib")
305:     @unittest.skipUnless(can_fs_encode(u'årchiv'),
306:         'File system cannot handle this filename')
307:     def test_make_tarball_unicode_latin1(self):
308:         '''
309:         Mirror test_make_tarball, except filename is unicode and contains
310:         latin characters.
311:         '''
312:         self._make_tarball(u'årchiv') # note this isn't a real word
313: 
314:     @unittest.skipUnless(zlib, "requires zlib")
315:     @unittest.skipUnless(can_fs_encode(u'のアーカイブ'),
316:         'File system cannot handle this filename')
317:     def test_make_tarball_unicode_extended(self):
318:         '''
319:         Mirror test_make_tarball, except filename is unicode and contains
320:         characters outside the latin charset.
321:         '''
322:         self._make_tarball(u'のアーカイブ') # japanese for archive
323: 
324: def test_suite():
325:     return unittest.makeSuite(ArchiveUtilTestCase)
326: 
327: if __name__ == "__main__":
328:     run_unittest(test_suite())
329: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_28891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 0), 'str', 'Tests for distutils.archive_util.')

# Assigning a Str to a Name (line 3):

# Assigning a Str to a Name (line 3):
str_28892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__revision__', str_28892)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import unittest' statement (line 5)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import os' statement (line 6)
import os

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sys' statement (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import tarfile' statement (line 8)
import tarfile

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'tarfile', tarfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from os.path import splitdrive' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28893 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os.path')

if (type(import_28893) is not StypyTypeError):

    if (import_28893 != 'pyd_module'):
        __import__(import_28893)
        sys_modules_28894 = sys.modules[import_28893]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os.path', sys_modules_28894.module_type_store, module_type_store, ['splitdrive'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_28894, sys_modules_28894.module_type_store, module_type_store)
    else:
        from os.path import splitdrive

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os.path', None, module_type_store, ['splitdrive'], [splitdrive])

else:
    # Assigning a type to the variable 'os.path' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'os.path', import_28893)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import warnings' statement (line 10)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.archive_util import check_archive_formats, make_tarball, make_zipfile, make_archive, ARCHIVE_FORMATS' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28895 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.archive_util')

if (type(import_28895) is not StypyTypeError):

    if (import_28895 != 'pyd_module'):
        __import__(import_28895)
        sys_modules_28896 = sys.modules[import_28895]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.archive_util', sys_modules_28896.module_type_store, module_type_store, ['check_archive_formats', 'make_tarball', 'make_zipfile', 'make_archive', 'ARCHIVE_FORMATS'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_28896, sys_modules_28896.module_type_store, module_type_store)
    else:
        from distutils.archive_util import check_archive_formats, make_tarball, make_zipfile, make_archive, ARCHIVE_FORMATS

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.archive_util', None, module_type_store, ['check_archive_formats', 'make_tarball', 'make_zipfile', 'make_archive', 'ARCHIVE_FORMATS'], [check_archive_formats, make_tarball, make_zipfile, make_archive, ARCHIVE_FORMATS])

else:
    # Assigning a type to the variable 'distutils.archive_util' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.archive_util', import_28895)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.spawn import find_executable, spawn' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28897 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.spawn')

if (type(import_28897) is not StypyTypeError):

    if (import_28897 != 'pyd_module'):
        __import__(import_28897)
        sys_modules_28898 = sys.modules[import_28897]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.spawn', sys_modules_28898.module_type_store, module_type_store, ['find_executable', 'spawn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_28898, sys_modules_28898.module_type_store, module_type_store)
    else:
        from distutils.spawn import find_executable, spawn

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.spawn', None, module_type_store, ['find_executable', 'spawn'], [find_executable, spawn])

else:
    # Assigning a type to the variable 'distutils.spawn' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.spawn', import_28897)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.tests import support' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28899 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.tests')

if (type(import_28899) is not StypyTypeError):

    if (import_28899 != 'pyd_module'):
        __import__(import_28899)
        sys_modules_28900 = sys.modules[import_28899]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.tests', sys_modules_28900.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_28900, sys_modules_28900.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.tests', import_28899)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from test.test_support import check_warnings, run_unittest' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28901 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'test.test_support')

if (type(import_28901) is not StypyTypeError):

    if (import_28901 != 'pyd_module'):
        __import__(import_28901)
        sys_modules_28902 = sys.modules[import_28901]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'test.test_support', sys_modules_28902.module_type_store, module_type_store, ['check_warnings', 'run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_28902, sys_modules_28902.module_type_store, module_type_store)
    else:
        from test.test_support import check_warnings, run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'test.test_support', None, module_type_store, ['check_warnings', 'run_unittest'], [check_warnings, run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'test.test_support', import_28901)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')



# SSA begins for try-except statement (line 19)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 4))

# 'import grp' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28903 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'grp')

if (type(import_28903) is not StypyTypeError):

    if (import_28903 != 'pyd_module'):
        __import__(import_28903)
        sys_modules_28904 = sys.modules[import_28903]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'grp', sys_modules_28904.module_type_store, module_type_store)
    else:
        import grp

        import_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'grp', grp, module_type_store)

else:
    # Assigning a type to the variable 'grp' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'grp', import_28903)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 4))

# 'import pwd' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28905 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'pwd')

if (type(import_28905) is not StypyTypeError):

    if (import_28905 != 'pyd_module'):
        __import__(import_28905)
        sys_modules_28906 = sys.modules[import_28905]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'pwd', sys_modules_28906.module_type_store, module_type_store)
    else:
        import pwd

        import_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'pwd', pwd, module_type_store)

else:
    # Assigning a type to the variable 'pwd' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'pwd', import_28905)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Name to a Name (line 22):

# Assigning a Name to a Name (line 22):
# Getting the type of 'True' (line 22)
True_28907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'True')
# Assigning a type to the variable 'UID_GID_SUPPORT' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'UID_GID_SUPPORT', True_28907)
# SSA branch for the except part of a try statement (line 19)
# SSA branch for the except 'ImportError' branch of a try statement (line 19)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 24):

# Assigning a Name to a Name (line 24):
# Getting the type of 'False' (line 24)
False_28908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'False')
# Assigning a type to the variable 'UID_GID_SUPPORT' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'UID_GID_SUPPORT', False_28908)
# SSA join for try-except statement (line 19)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 26)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 4))

# 'import zipfile' statement (line 27)
import zipfile

import_module(stypy.reporting.localization.Localization(__file__, 27, 4), 'zipfile', zipfile, module_type_store)


# Assigning a Name to a Name (line 28):

# Assigning a Name to a Name (line 28):
# Getting the type of 'True' (line 28)
True_28909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'True')
# Assigning a type to the variable 'ZIP_SUPPORT' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'ZIP_SUPPORT', True_28909)
# SSA branch for the except part of a try statement (line 26)
# SSA branch for the except 'ImportError' branch of a try statement (line 26)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 30):

# Assigning a Call to a Name (line 30):

# Call to find_executable(...): (line 30)
# Processing the call arguments (line 30)
str_28911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'str', 'zip')
# Processing the call keyword arguments (line 30)
kwargs_28912 = {}
# Getting the type of 'find_executable' (line 30)
find_executable_28910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 18), 'find_executable', False)
# Calling find_executable(args, kwargs) (line 30)
find_executable_call_result_28913 = invoke(stypy.reporting.localization.Localization(__file__, 30, 18), find_executable_28910, *[str_28911], **kwargs_28912)

# Assigning a type to the variable 'ZIP_SUPPORT' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'ZIP_SUPPORT', find_executable_call_result_28913)
# SSA join for try-except statement (line 26)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 33)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 4))

# 'import zlib' statement (line 34)
import zlib

import_module(stypy.reporting.localization.Localization(__file__, 34, 4), 'zlib', zlib, module_type_store)

# SSA branch for the except part of a try statement (line 33)
# SSA branch for the except 'ImportError' branch of a try statement (line 33)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 36):

# Assigning a Name to a Name (line 36):
# Getting the type of 'None' (line 36)
None_28914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'None')
# Assigning a type to the variable 'zlib' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'zlib', None_28914)
# SSA join for try-except statement (line 33)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def can_fs_encode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'can_fs_encode'
    module_type_store = module_type_store.open_function_context('can_fs_encode', 38, 0, False)
    
    # Passed parameters checking function
    can_fs_encode.stypy_localization = localization
    can_fs_encode.stypy_type_of_self = None
    can_fs_encode.stypy_type_store = module_type_store
    can_fs_encode.stypy_function_name = 'can_fs_encode'
    can_fs_encode.stypy_param_names_list = ['filename']
    can_fs_encode.stypy_varargs_param_name = None
    can_fs_encode.stypy_kwargs_param_name = None
    can_fs_encode.stypy_call_defaults = defaults
    can_fs_encode.stypy_call_varargs = varargs
    can_fs_encode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'can_fs_encode', ['filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'can_fs_encode', localization, ['filename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'can_fs_encode(...)' code ##################

    str_28915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'str', '\n    Return True if the filename can be saved in the file system.\n    ')
    
    # Getting the type of 'os' (line 42)
    os_28916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'os')
    # Obtaining the member 'path' of a type (line 42)
    path_28917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 7), os_28916, 'path')
    # Obtaining the member 'supports_unicode_filenames' of a type (line 42)
    supports_unicode_filenames_28918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 7), path_28917, 'supports_unicode_filenames')
    # Testing the type of an if condition (line 42)
    if_condition_28919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 4), supports_unicode_filenames_28918)
    # Assigning a type to the variable 'if_condition_28919' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_condition_28919', if_condition_28919)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 43)
    True_28920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type', True_28920)
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to encode(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to getfilesystemencoding(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_28925 = {}
    # Getting the type of 'sys' (line 45)
    sys_28923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'sys', False)
    # Obtaining the member 'getfilesystemencoding' of a type (line 45)
    getfilesystemencoding_28924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 24), sys_28923, 'getfilesystemencoding')
    # Calling getfilesystemencoding(args, kwargs) (line 45)
    getfilesystemencoding_call_result_28926 = invoke(stypy.reporting.localization.Localization(__file__, 45, 24), getfilesystemencoding_28924, *[], **kwargs_28925)
    
    # Processing the call keyword arguments (line 45)
    kwargs_28927 = {}
    # Getting the type of 'filename' (line 45)
    filename_28921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'filename', False)
    # Obtaining the member 'encode' of a type (line 45)
    encode_28922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), filename_28921, 'encode')
    # Calling encode(args, kwargs) (line 45)
    encode_call_result_28928 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), encode_28922, *[getfilesystemencoding_call_result_28926], **kwargs_28927)
    
    # SSA branch for the except part of a try statement (line 44)
    # SSA branch for the except 'UnicodeEncodeError' branch of a try statement (line 44)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'False' (line 47)
    False_28929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', False_28929)
    # SSA join for try-except statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'True' (line 48)
    True_28930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type', True_28930)
    
    # ################# End of 'can_fs_encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'can_fs_encode' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_28931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28931)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'can_fs_encode'
    return stypy_return_type_28931

# Assigning a type to the variable 'can_fs_encode' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'can_fs_encode', can_fs_encode)
# Declaration of the 'ArchiveUtilTestCase' class
# Getting the type of 'support' (line 51)
support_28932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'support')
# Obtaining the member 'TempdirManager' of a type (line 51)
TempdirManager_28933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 26), support_28932, 'TempdirManager')
# Getting the type of 'support' (line 52)
support_28934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 52)
LoggingSilencer_28935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 26), support_28934, 'LoggingSilencer')
# Getting the type of 'unittest' (line 53)
unittest_28936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'unittest')
# Obtaining the member 'TestCase' of a type (line 53)
TestCase_28937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 26), unittest_28936, 'TestCase')

class ArchiveUtilTestCase(TempdirManager_28933, LoggingSilencer_28935, TestCase_28937, ):

    @norecursion
    def test_make_tarball(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_make_tarball'
        module_type_store = module_type_store.open_function_context('test_make_tarball', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_make_tarball')
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_make_tarball.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_make_tarball', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_make_tarball', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_make_tarball(...)' code ##################

        
        # Call to _make_tarball(...): (line 57)
        # Processing the call arguments (line 57)
        str_28940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 27), 'str', 'archive')
        # Processing the call keyword arguments (line 57)
        kwargs_28941 = {}
        # Getting the type of 'self' (line 57)
        self_28938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member '_make_tarball' of a type (line 57)
        _make_tarball_28939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_28938, '_make_tarball')
        # Calling _make_tarball(args, kwargs) (line 57)
        _make_tarball_call_result_28942 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), _make_tarball_28939, *[str_28940], **kwargs_28941)
        
        
        # ################# End of 'test_make_tarball(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_make_tarball' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_28943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28943)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_make_tarball'
        return stypy_return_type_28943


    @norecursion
    def _make_tarball(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_make_tarball'
        module_type_store = module_type_store.open_function_context('_make_tarball', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase._make_tarball')
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_param_names_list', ['target_name'])
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase._make_tarball.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase._make_tarball', ['target_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_make_tarball', localization, ['target_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_make_tarball(...)' code ##################

        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to mkdtemp(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_28946 = {}
        # Getting the type of 'self' (line 61)
        self_28944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 61)
        mkdtemp_28945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 17), self_28944, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 61)
        mkdtemp_call_result_28947 = invoke(stypy.reporting.localization.Localization(__file__, 61, 17), mkdtemp_28945, *[], **kwargs_28946)
        
        # Assigning a type to the variable 'tmpdir' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tmpdir', mkdtemp_call_result_28947)
        
        # Call to write_file(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_28950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        # Getting the type of 'tmpdir' (line 62)
        tmpdir_28951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'tmpdir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 24), list_28950, tmpdir_28951)
        # Adding element type (line 62)
        str_28952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 33), 'str', 'file1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 24), list_28950, str_28952)
        
        str_28953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'str', 'xxx')
        # Processing the call keyword arguments (line 62)
        kwargs_28954 = {}
        # Getting the type of 'self' (line 62)
        self_28948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 62)
        write_file_28949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_28948, 'write_file')
        # Calling write_file(args, kwargs) (line 62)
        write_file_call_result_28955 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), write_file_28949, *[list_28950, str_28953], **kwargs_28954)
        
        
        # Call to write_file(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_28958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        # Getting the type of 'tmpdir' (line 63)
        tmpdir_28959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'tmpdir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_28958, tmpdir_28959)
        # Adding element type (line 63)
        str_28960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 33), 'str', 'file2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_28958, str_28960)
        
        str_28961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 43), 'str', 'xxx')
        # Processing the call keyword arguments (line 63)
        kwargs_28962 = {}
        # Getting the type of 'self' (line 63)
        self_28956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 63)
        write_file_28957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_28956, 'write_file')
        # Calling write_file(args, kwargs) (line 63)
        write_file_call_result_28963 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), write_file_28957, *[list_28958, str_28961], **kwargs_28962)
        
        
        # Call to mkdir(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to join(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'tmpdir' (line 64)
        tmpdir_28969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'tmpdir', False)
        str_28970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 38), 'str', 'sub')
        # Processing the call keyword arguments (line 64)
        kwargs_28971 = {}
        # Getting the type of 'os' (line 64)
        os_28966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 64)
        path_28967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 17), os_28966, 'path')
        # Obtaining the member 'join' of a type (line 64)
        join_28968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 17), path_28967, 'join')
        # Calling join(args, kwargs) (line 64)
        join_call_result_28972 = invoke(stypy.reporting.localization.Localization(__file__, 64, 17), join_28968, *[tmpdir_28969, str_28970], **kwargs_28971)
        
        # Processing the call keyword arguments (line 64)
        kwargs_28973 = {}
        # Getting the type of 'os' (line 64)
        os_28964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 64)
        mkdir_28965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), os_28964, 'mkdir')
        # Calling mkdir(args, kwargs) (line 64)
        mkdir_call_result_28974 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), mkdir_28965, *[join_call_result_28972], **kwargs_28973)
        
        
        # Call to write_file(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_28977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        # Getting the type of 'tmpdir' (line 65)
        tmpdir_28978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'tmpdir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), list_28977, tmpdir_28978)
        # Adding element type (line 65)
        str_28979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'str', 'sub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), list_28977, str_28979)
        # Adding element type (line 65)
        str_28980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 40), 'str', 'file3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), list_28977, str_28980)
        
        str_28981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 50), 'str', 'xxx')
        # Processing the call keyword arguments (line 65)
        kwargs_28982 = {}
        # Getting the type of 'self' (line 65)
        self_28975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 65)
        write_file_28976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_28975, 'write_file')
        # Calling write_file(args, kwargs) (line 65)
        write_file_call_result_28983 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), write_file_28976, *[list_28977, str_28981], **kwargs_28982)
        
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to mkdtemp(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_28986 = {}
        # Getting the type of 'self' (line 67)
        self_28984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 67)
        mkdtemp_28985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 18), self_28984, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 67)
        mkdtemp_call_result_28987 = invoke(stypy.reporting.localization.Localization(__file__, 67, 18), mkdtemp_28985, *[], **kwargs_28986)
        
        # Assigning a type to the variable 'tmpdir2' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'tmpdir2', mkdtemp_call_result_28987)
        
        # Call to skipUnless(...): (line 68)
        # Processing the call arguments (line 68)
        
        
        # Obtaining the type of the subscript
        int_28990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 47), 'int')
        
        # Call to splitdrive(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'tmpdir' (line 68)
        tmpdir_28992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 39), 'tmpdir', False)
        # Processing the call keyword arguments (line 68)
        kwargs_28993 = {}
        # Getting the type of 'splitdrive' (line 68)
        splitdrive_28991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'splitdrive', False)
        # Calling splitdrive(args, kwargs) (line 68)
        splitdrive_call_result_28994 = invoke(stypy.reporting.localization.Localization(__file__, 68, 28), splitdrive_28991, *[tmpdir_28992], **kwargs_28993)
        
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___28995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 28), splitdrive_call_result_28994, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_28996 = invoke(stypy.reporting.localization.Localization(__file__, 68, 28), getitem___28995, int_28990)
        
        
        # Obtaining the type of the subscript
        int_28997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 73), 'int')
        
        # Call to splitdrive(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'tmpdir2' (line 68)
        tmpdir2_28999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 64), 'tmpdir2', False)
        # Processing the call keyword arguments (line 68)
        kwargs_29000 = {}
        # Getting the type of 'splitdrive' (line 68)
        splitdrive_28998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 53), 'splitdrive', False)
        # Calling splitdrive(args, kwargs) (line 68)
        splitdrive_call_result_29001 = invoke(stypy.reporting.localization.Localization(__file__, 68, 53), splitdrive_28998, *[tmpdir2_28999], **kwargs_29000)
        
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___29002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 53), splitdrive_call_result_29001, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_29003 = invoke(stypy.reporting.localization.Localization(__file__, 68, 53), getitem___29002, int_28997)
        
        # Applying the binary operator '==' (line 68)
        result_eq_29004 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 28), '==', subscript_call_result_28996, subscript_call_result_29003)
        
        str_29005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'str', 'source and target should be on same drive')
        # Processing the call keyword arguments (line 68)
        kwargs_29006 = {}
        # Getting the type of 'unittest' (line 68)
        unittest_28988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'unittest', False)
        # Obtaining the member 'skipUnless' of a type (line 68)
        skipUnless_28989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), unittest_28988, 'skipUnless')
        # Calling skipUnless(args, kwargs) (line 68)
        skipUnless_call_result_29007 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), skipUnless_28989, *[result_eq_29004, str_29005], **kwargs_29006)
        
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to join(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'tmpdir2' (line 71)
        tmpdir2_29011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 33), 'tmpdir2', False)
        # Getting the type of 'target_name' (line 71)
        target_name_29012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 42), 'target_name', False)
        # Processing the call keyword arguments (line 71)
        kwargs_29013 = {}
        # Getting the type of 'os' (line 71)
        os_29008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 71)
        path_29009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), os_29008, 'path')
        # Obtaining the member 'join' of a type (line 71)
        join_29010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), path_29009, 'join')
        # Calling join(args, kwargs) (line 71)
        join_call_result_29014 = invoke(stypy.reporting.localization.Localization(__file__, 71, 20), join_29010, *[tmpdir2_29011, target_name_29012], **kwargs_29013)
        
        # Assigning a type to the variable 'base_name' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'base_name', join_call_result_29014)
        
        # Assigning a Call to a Name (line 74):
        
        # Assigning a Call to a Name (line 74):
        
        # Call to getcwd(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_29017 = {}
        # Getting the type of 'os' (line 74)
        os_29015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 74)
        getcwd_29016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 18), os_29015, 'getcwd')
        # Calling getcwd(args, kwargs) (line 74)
        getcwd_call_result_29018 = invoke(stypy.reporting.localization.Localization(__file__, 74, 18), getcwd_29016, *[], **kwargs_29017)
        
        # Assigning a type to the variable 'old_dir' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'old_dir', getcwd_call_result_29018)
        
        # Call to chdir(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'tmpdir' (line 75)
        tmpdir_29021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'tmpdir', False)
        # Processing the call keyword arguments (line 75)
        kwargs_29022 = {}
        # Getting the type of 'os' (line 75)
        os_29019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 75)
        chdir_29020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), os_29019, 'chdir')
        # Calling chdir(args, kwargs) (line 75)
        chdir_call_result_29023 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), chdir_29020, *[tmpdir_29021], **kwargs_29022)
        
        
        # Try-finally block (line 76)
        
        # Call to make_tarball(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Obtaining the type of the subscript
        int_29025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 47), 'int')
        
        # Call to splitdrive(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'base_name' (line 77)
        base_name_29027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 36), 'base_name', False)
        # Processing the call keyword arguments (line 77)
        kwargs_29028 = {}
        # Getting the type of 'splitdrive' (line 77)
        splitdrive_29026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'splitdrive', False)
        # Calling splitdrive(args, kwargs) (line 77)
        splitdrive_call_result_29029 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), splitdrive_29026, *[base_name_29027], **kwargs_29028)
        
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___29030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 25), splitdrive_call_result_29029, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_29031 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), getitem___29030, int_29025)
        
        str_29032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 51), 'str', '.')
        # Processing the call keyword arguments (line 77)
        kwargs_29033 = {}
        # Getting the type of 'make_tarball' (line 77)
        make_tarball_29024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'make_tarball', False)
        # Calling make_tarball(args, kwargs) (line 77)
        make_tarball_call_result_29034 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), make_tarball_29024, *[subscript_call_result_29031, str_29032], **kwargs_29033)
        
        
        # finally branch of the try-finally block (line 76)
        
        # Call to chdir(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'old_dir' (line 79)
        old_dir_29037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'old_dir', False)
        # Processing the call keyword arguments (line 79)
        kwargs_29038 = {}
        # Getting the type of 'os' (line 79)
        os_29035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 79)
        chdir_29036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), os_29035, 'chdir')
        # Calling chdir(args, kwargs) (line 79)
        chdir_call_result_29039 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), chdir_29036, *[old_dir_29037], **kwargs_29038)
        
        
        
        # Assigning a BinOp to a Name (line 82):
        
        # Assigning a BinOp to a Name (line 82):
        # Getting the type of 'base_name' (line 82)
        base_name_29040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'base_name')
        str_29041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 30), 'str', '.tar.gz')
        # Applying the binary operator '+' (line 82)
        result_add_29042 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 18), '+', base_name_29040, str_29041)
        
        # Assigning a type to the variable 'tarball' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tarball', result_add_29042)
        
        # Call to assertTrue(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to exists(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'tarball' (line 83)
        tarball_29048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 39), 'tarball', False)
        # Processing the call keyword arguments (line 83)
        kwargs_29049 = {}
        # Getting the type of 'os' (line 83)
        os_29045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 83)
        path_29046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 24), os_29045, 'path')
        # Obtaining the member 'exists' of a type (line 83)
        exists_29047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 24), path_29046, 'exists')
        # Calling exists(args, kwargs) (line 83)
        exists_call_result_29050 = invoke(stypy.reporting.localization.Localization(__file__, 83, 24), exists_29047, *[tarball_29048], **kwargs_29049)
        
        # Processing the call keyword arguments (line 83)
        kwargs_29051 = {}
        # Getting the type of 'self' (line 83)
        self_29043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 83)
        assertTrue_29044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_29043, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 83)
        assertTrue_call_result_29052 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assertTrue_29044, *[exists_call_result_29050], **kwargs_29051)
        
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to join(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'tmpdir2' (line 86)
        tmpdir2_29056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 33), 'tmpdir2', False)
        # Getting the type of 'target_name' (line 86)
        target_name_29057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 42), 'target_name', False)
        # Processing the call keyword arguments (line 86)
        kwargs_29058 = {}
        # Getting the type of 'os' (line 86)
        os_29053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 86)
        path_29054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 20), os_29053, 'path')
        # Obtaining the member 'join' of a type (line 86)
        join_29055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 20), path_29054, 'join')
        # Calling join(args, kwargs) (line 86)
        join_call_result_29059 = invoke(stypy.reporting.localization.Localization(__file__, 86, 20), join_29055, *[tmpdir2_29056, target_name_29057], **kwargs_29058)
        
        # Assigning a type to the variable 'base_name' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'base_name', join_call_result_29059)
        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to getcwd(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_29062 = {}
        # Getting the type of 'os' (line 87)
        os_29060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 87)
        getcwd_29061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 18), os_29060, 'getcwd')
        # Calling getcwd(args, kwargs) (line 87)
        getcwd_call_result_29063 = invoke(stypy.reporting.localization.Localization(__file__, 87, 18), getcwd_29061, *[], **kwargs_29062)
        
        # Assigning a type to the variable 'old_dir' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'old_dir', getcwd_call_result_29063)
        
        # Call to chdir(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'tmpdir' (line 88)
        tmpdir_29066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'tmpdir', False)
        # Processing the call keyword arguments (line 88)
        kwargs_29067 = {}
        # Getting the type of 'os' (line 88)
        os_29064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 88)
        chdir_29065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), os_29064, 'chdir')
        # Calling chdir(args, kwargs) (line 88)
        chdir_call_result_29068 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), chdir_29065, *[tmpdir_29066], **kwargs_29067)
        
        
        # Try-finally block (line 89)
        
        # Call to make_tarball(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Obtaining the type of the subscript
        int_29070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 47), 'int')
        
        # Call to splitdrive(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'base_name' (line 90)
        base_name_29072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 36), 'base_name', False)
        # Processing the call keyword arguments (line 90)
        kwargs_29073 = {}
        # Getting the type of 'splitdrive' (line 90)
        splitdrive_29071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'splitdrive', False)
        # Calling splitdrive(args, kwargs) (line 90)
        splitdrive_call_result_29074 = invoke(stypy.reporting.localization.Localization(__file__, 90, 25), splitdrive_29071, *[base_name_29072], **kwargs_29073)
        
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___29075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 25), splitdrive_call_result_29074, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_29076 = invoke(stypy.reporting.localization.Localization(__file__, 90, 25), getitem___29075, int_29070)
        
        str_29077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 51), 'str', '.')
        # Processing the call keyword arguments (line 90)
        # Getting the type of 'None' (line 90)
        None_29078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 65), 'None', False)
        keyword_29079 = None_29078
        kwargs_29080 = {'compress': keyword_29079}
        # Getting the type of 'make_tarball' (line 90)
        make_tarball_29069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'make_tarball', False)
        # Calling make_tarball(args, kwargs) (line 90)
        make_tarball_call_result_29081 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), make_tarball_29069, *[subscript_call_result_29076, str_29077], **kwargs_29080)
        
        
        # finally branch of the try-finally block (line 89)
        
        # Call to chdir(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'old_dir' (line 92)
        old_dir_29084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 21), 'old_dir', False)
        # Processing the call keyword arguments (line 92)
        kwargs_29085 = {}
        # Getting the type of 'os' (line 92)
        os_29082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 92)
        chdir_29083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), os_29082, 'chdir')
        # Calling chdir(args, kwargs) (line 92)
        chdir_call_result_29086 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), chdir_29083, *[old_dir_29084], **kwargs_29085)
        
        
        
        # Assigning a BinOp to a Name (line 93):
        
        # Assigning a BinOp to a Name (line 93):
        # Getting the type of 'base_name' (line 93)
        base_name_29087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'base_name')
        str_29088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 30), 'str', '.tar')
        # Applying the binary operator '+' (line 93)
        result_add_29089 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 18), '+', base_name_29087, str_29088)
        
        # Assigning a type to the variable 'tarball' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tarball', result_add_29089)
        
        # Call to assertTrue(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to exists(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'tarball' (line 94)
        tarball_29095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 39), 'tarball', False)
        # Processing the call keyword arguments (line 94)
        kwargs_29096 = {}
        # Getting the type of 'os' (line 94)
        os_29092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 94)
        path_29093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 24), os_29092, 'path')
        # Obtaining the member 'exists' of a type (line 94)
        exists_29094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 24), path_29093, 'exists')
        # Calling exists(args, kwargs) (line 94)
        exists_call_result_29097 = invoke(stypy.reporting.localization.Localization(__file__, 94, 24), exists_29094, *[tarball_29095], **kwargs_29096)
        
        # Processing the call keyword arguments (line 94)
        kwargs_29098 = {}
        # Getting the type of 'self' (line 94)
        self_29090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 94)
        assertTrue_29091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_29090, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 94)
        assertTrue_call_result_29099 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), assertTrue_29091, *[exists_call_result_29097], **kwargs_29098)
        
        
        # ################# End of '_make_tarball(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_make_tarball' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_29100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29100)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_make_tarball'
        return stypy_return_type_29100


    @norecursion
    def _tarinfo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_tarinfo'
        module_type_store = module_type_store.open_function_context('_tarinfo', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase._tarinfo')
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_param_names_list', ['path'])
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase._tarinfo.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase._tarinfo', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_tarinfo', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_tarinfo(...)' code ##################

        
        # Assigning a Call to a Name (line 97):
        
        # Assigning a Call to a Name (line 97):
        
        # Call to open(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'path' (line 97)
        path_29103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'path', False)
        # Processing the call keyword arguments (line 97)
        kwargs_29104 = {}
        # Getting the type of 'tarfile' (line 97)
        tarfile_29101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'tarfile', False)
        # Obtaining the member 'open' of a type (line 97)
        open_29102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 14), tarfile_29101, 'open')
        # Calling open(args, kwargs) (line 97)
        open_call_result_29105 = invoke(stypy.reporting.localization.Localization(__file__, 97, 14), open_29102, *[path_29103], **kwargs_29104)
        
        # Assigning a type to the variable 'tar' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tar', open_call_result_29105)
        
        # Try-finally block (line 98)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to getnames(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_29108 = {}
        # Getting the type of 'tar' (line 99)
        tar_29106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'tar', False)
        # Obtaining the member 'getnames' of a type (line 99)
        getnames_29107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 20), tar_29106, 'getnames')
        # Calling getnames(args, kwargs) (line 99)
        getnames_call_result_29109 = invoke(stypy.reporting.localization.Localization(__file__, 99, 20), getnames_29107, *[], **kwargs_29108)
        
        # Assigning a type to the variable 'names' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'names', getnames_call_result_29109)
        
        # Call to sort(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_29112 = {}
        # Getting the type of 'names' (line 100)
        names_29110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'names', False)
        # Obtaining the member 'sort' of a type (line 100)
        sort_29111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), names_29110, 'sort')
        # Calling sort(args, kwargs) (line 100)
        sort_call_result_29113 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), sort_29111, *[], **kwargs_29112)
        
        
        # Call to tuple(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'names' (line 101)
        names_29115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 25), 'names', False)
        # Processing the call keyword arguments (line 101)
        kwargs_29116 = {}
        # Getting the type of 'tuple' (line 101)
        tuple_29114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'tuple', False)
        # Calling tuple(args, kwargs) (line 101)
        tuple_call_result_29117 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), tuple_29114, *[names_29115], **kwargs_29116)
        
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type', tuple_call_result_29117)
        
        # finally branch of the try-finally block (line 98)
        
        # Call to close(...): (line 103)
        # Processing the call keyword arguments (line 103)
        kwargs_29120 = {}
        # Getting the type of 'tar' (line 103)
        tar_29118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'tar', False)
        # Obtaining the member 'close' of a type (line 103)
        close_29119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), tar_29118, 'close')
        # Calling close(args, kwargs) (line 103)
        close_call_result_29121 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), close_29119, *[], **kwargs_29120)
        
        
        
        # ################# End of '_tarinfo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_tarinfo' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_29122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29122)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_tarinfo'
        return stypy_return_type_29122


    @norecursion
    def _create_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_create_files'
        module_type_store = module_type_store.open_function_context('_create_files', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase._create_files')
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase._create_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase._create_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_create_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_create_files(...)' code ##################

        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to mkdtemp(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_29125 = {}
        # Getting the type of 'self' (line 107)
        self_29123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 107)
        mkdtemp_29124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 17), self_29123, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 107)
        mkdtemp_call_result_29126 = invoke(stypy.reporting.localization.Localization(__file__, 107, 17), mkdtemp_29124, *[], **kwargs_29125)
        
        # Assigning a type to the variable 'tmpdir' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'tmpdir', mkdtemp_call_result_29126)
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to join(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'tmpdir' (line 108)
        tmpdir_29130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'tmpdir', False)
        str_29131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 36), 'str', 'dist')
        # Processing the call keyword arguments (line 108)
        kwargs_29132 = {}
        # Getting the type of 'os' (line 108)
        os_29127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 108)
        path_29128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), os_29127, 'path')
        # Obtaining the member 'join' of a type (line 108)
        join_29129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), path_29128, 'join')
        # Calling join(args, kwargs) (line 108)
        join_call_result_29133 = invoke(stypy.reporting.localization.Localization(__file__, 108, 15), join_29129, *[tmpdir_29130, str_29131], **kwargs_29132)
        
        # Assigning a type to the variable 'dist' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'dist', join_call_result_29133)
        
        # Call to mkdir(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'dist' (line 109)
        dist_29136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'dist', False)
        # Processing the call keyword arguments (line 109)
        kwargs_29137 = {}
        # Getting the type of 'os' (line 109)
        os_29134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 109)
        mkdir_29135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), os_29134, 'mkdir')
        # Calling mkdir(args, kwargs) (line 109)
        mkdir_call_result_29138 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), mkdir_29135, *[dist_29136], **kwargs_29137)
        
        
        # Call to write_file(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_29141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        # Getting the type of 'dist' (line 110)
        dist_29142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'dist', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 24), list_29141, dist_29142)
        # Adding element type (line 110)
        str_29143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'str', 'file1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 24), list_29141, str_29143)
        
        str_29144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 41), 'str', 'xxx')
        # Processing the call keyword arguments (line 110)
        kwargs_29145 = {}
        # Getting the type of 'self' (line 110)
        self_29139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 110)
        write_file_29140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_29139, 'write_file')
        # Calling write_file(args, kwargs) (line 110)
        write_file_call_result_29146 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), write_file_29140, *[list_29141, str_29144], **kwargs_29145)
        
        
        # Call to write_file(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_29149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        # Getting the type of 'dist' (line 111)
        dist_29150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'dist', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 24), list_29149, dist_29150)
        # Adding element type (line 111)
        str_29151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 31), 'str', 'file2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 24), list_29149, str_29151)
        
        str_29152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 41), 'str', 'xxx')
        # Processing the call keyword arguments (line 111)
        kwargs_29153 = {}
        # Getting the type of 'self' (line 111)
        self_29147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 111)
        write_file_29148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_29147, 'write_file')
        # Calling write_file(args, kwargs) (line 111)
        write_file_call_result_29154 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), write_file_29148, *[list_29149, str_29152], **kwargs_29153)
        
        
        # Call to mkdir(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to join(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'dist' (line 112)
        dist_29160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'dist', False)
        str_29161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 36), 'str', 'sub')
        # Processing the call keyword arguments (line 112)
        kwargs_29162 = {}
        # Getting the type of 'os' (line 112)
        os_29157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 112)
        path_29158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 17), os_29157, 'path')
        # Obtaining the member 'join' of a type (line 112)
        join_29159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 17), path_29158, 'join')
        # Calling join(args, kwargs) (line 112)
        join_call_result_29163 = invoke(stypy.reporting.localization.Localization(__file__, 112, 17), join_29159, *[dist_29160, str_29161], **kwargs_29162)
        
        # Processing the call keyword arguments (line 112)
        kwargs_29164 = {}
        # Getting the type of 'os' (line 112)
        os_29155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 112)
        mkdir_29156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), os_29155, 'mkdir')
        # Calling mkdir(args, kwargs) (line 112)
        mkdir_call_result_29165 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), mkdir_29156, *[join_call_result_29163], **kwargs_29164)
        
        
        # Call to write_file(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_29168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        # Getting the type of 'dist' (line 113)
        dist_29169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'dist', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_29168, dist_29169)
        # Adding element type (line 113)
        str_29170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 31), 'str', 'sub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_29168, str_29170)
        # Adding element type (line 113)
        str_29171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 38), 'str', 'file3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 24), list_29168, str_29171)
        
        str_29172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 48), 'str', 'xxx')
        # Processing the call keyword arguments (line 113)
        kwargs_29173 = {}
        # Getting the type of 'self' (line 113)
        self_29166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 113)
        write_file_29167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_29166, 'write_file')
        # Calling write_file(args, kwargs) (line 113)
        write_file_call_result_29174 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), write_file_29167, *[list_29168, str_29172], **kwargs_29173)
        
        
        # Call to mkdir(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to join(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'dist' (line 114)
        dist_29180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 30), 'dist', False)
        str_29181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 36), 'str', 'sub2')
        # Processing the call keyword arguments (line 114)
        kwargs_29182 = {}
        # Getting the type of 'os' (line 114)
        os_29177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 114)
        path_29178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 17), os_29177, 'path')
        # Obtaining the member 'join' of a type (line 114)
        join_29179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 17), path_29178, 'join')
        # Calling join(args, kwargs) (line 114)
        join_call_result_29183 = invoke(stypy.reporting.localization.Localization(__file__, 114, 17), join_29179, *[dist_29180, str_29181], **kwargs_29182)
        
        # Processing the call keyword arguments (line 114)
        kwargs_29184 = {}
        # Getting the type of 'os' (line 114)
        os_29175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 114)
        mkdir_29176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), os_29175, 'mkdir')
        # Calling mkdir(args, kwargs) (line 114)
        mkdir_call_result_29185 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), mkdir_29176, *[join_call_result_29183], **kwargs_29184)
        
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to mkdtemp(...): (line 115)
        # Processing the call keyword arguments (line 115)
        kwargs_29188 = {}
        # Getting the type of 'self' (line 115)
        self_29186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 115)
        mkdtemp_29187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 18), self_29186, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 115)
        mkdtemp_call_result_29189 = invoke(stypy.reporting.localization.Localization(__file__, 115, 18), mkdtemp_29187, *[], **kwargs_29188)
        
        # Assigning a type to the variable 'tmpdir2' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'tmpdir2', mkdtemp_call_result_29189)
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to join(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'tmpdir2' (line 116)
        tmpdir2_29193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 33), 'tmpdir2', False)
        str_29194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 42), 'str', 'archive')
        # Processing the call keyword arguments (line 116)
        kwargs_29195 = {}
        # Getting the type of 'os' (line 116)
        os_29190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 116)
        path_29191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 20), os_29190, 'path')
        # Obtaining the member 'join' of a type (line 116)
        join_29192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 20), path_29191, 'join')
        # Calling join(args, kwargs) (line 116)
        join_call_result_29196 = invoke(stypy.reporting.localization.Localization(__file__, 116, 20), join_29192, *[tmpdir2_29193, str_29194], **kwargs_29195)
        
        # Assigning a type to the variable 'base_name' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'base_name', join_call_result_29196)
        
        # Obtaining an instance of the builtin type 'tuple' (line 117)
        tuple_29197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 117)
        # Adding element type (line 117)
        # Getting the type of 'tmpdir' (line 117)
        tmpdir_29198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'tmpdir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 15), tuple_29197, tmpdir_29198)
        # Adding element type (line 117)
        # Getting the type of 'tmpdir2' (line 117)
        tmpdir2_29199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'tmpdir2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 15), tuple_29197, tmpdir2_29199)
        # Adding element type (line 117)
        # Getting the type of 'base_name' (line 117)
        base_name_29200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'base_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 15), tuple_29197, base_name_29200)
        
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', tuple_29197)
        
        # ################# End of '_create_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_create_files' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_29201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29201)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_create_files'
        return stypy_return_type_29201


    @norecursion
    def test_tarfile_vs_tar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tarfile_vs_tar'
        module_type_store = module_type_store.open_function_context('test_tarfile_vs_tar', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_tarfile_vs_tar')
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_tarfile_vs_tar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_tarfile_vs_tar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tarfile_vs_tar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tarfile_vs_tar(...)' code ##################

        
        # Assigning a Call to a Tuple (line 123):
        
        # Assigning a Subscript to a Name (line 123):
        
        # Obtaining the type of the subscript
        int_29202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 8), 'int')
        
        # Call to _create_files(...): (line 123)
        # Processing the call keyword arguments (line 123)
        kwargs_29205 = {}
        # Getting the type of 'self' (line 123)
        self_29203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'self', False)
        # Obtaining the member '_create_files' of a type (line 123)
        _create_files_29204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 38), self_29203, '_create_files')
        # Calling _create_files(args, kwargs) (line 123)
        _create_files_call_result_29206 = invoke(stypy.reporting.localization.Localization(__file__, 123, 38), _create_files_29204, *[], **kwargs_29205)
        
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___29207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), _create_files_call_result_29206, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_29208 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), getitem___29207, int_29202)
        
        # Assigning a type to the variable 'tuple_var_assignment_28879' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'tuple_var_assignment_28879', subscript_call_result_29208)
        
        # Assigning a Subscript to a Name (line 123):
        
        # Obtaining the type of the subscript
        int_29209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 8), 'int')
        
        # Call to _create_files(...): (line 123)
        # Processing the call keyword arguments (line 123)
        kwargs_29212 = {}
        # Getting the type of 'self' (line 123)
        self_29210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'self', False)
        # Obtaining the member '_create_files' of a type (line 123)
        _create_files_29211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 38), self_29210, '_create_files')
        # Calling _create_files(args, kwargs) (line 123)
        _create_files_call_result_29213 = invoke(stypy.reporting.localization.Localization(__file__, 123, 38), _create_files_29211, *[], **kwargs_29212)
        
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___29214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), _create_files_call_result_29213, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_29215 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), getitem___29214, int_29209)
        
        # Assigning a type to the variable 'tuple_var_assignment_28880' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'tuple_var_assignment_28880', subscript_call_result_29215)
        
        # Assigning a Subscript to a Name (line 123):
        
        # Obtaining the type of the subscript
        int_29216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 8), 'int')
        
        # Call to _create_files(...): (line 123)
        # Processing the call keyword arguments (line 123)
        kwargs_29219 = {}
        # Getting the type of 'self' (line 123)
        self_29217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'self', False)
        # Obtaining the member '_create_files' of a type (line 123)
        _create_files_29218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 38), self_29217, '_create_files')
        # Calling _create_files(args, kwargs) (line 123)
        _create_files_call_result_29220 = invoke(stypy.reporting.localization.Localization(__file__, 123, 38), _create_files_29218, *[], **kwargs_29219)
        
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___29221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), _create_files_call_result_29220, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_29222 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), getitem___29221, int_29216)
        
        # Assigning a type to the variable 'tuple_var_assignment_28881' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'tuple_var_assignment_28881', subscript_call_result_29222)
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'tuple_var_assignment_28879' (line 123)
        tuple_var_assignment_28879_29223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'tuple_var_assignment_28879')
        # Assigning a type to the variable 'tmpdir' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'tmpdir', tuple_var_assignment_28879_29223)
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'tuple_var_assignment_28880' (line 123)
        tuple_var_assignment_28880_29224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'tuple_var_assignment_28880')
        # Assigning a type to the variable 'tmpdir2' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'tmpdir2', tuple_var_assignment_28880_29224)
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'tuple_var_assignment_28881' (line 123)
        tuple_var_assignment_28881_29225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'tuple_var_assignment_28881')
        # Assigning a type to the variable 'base_name' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'base_name', tuple_var_assignment_28881_29225)
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to getcwd(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_29228 = {}
        # Getting the type of 'os' (line 124)
        os_29226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 124)
        getcwd_29227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 18), os_29226, 'getcwd')
        # Calling getcwd(args, kwargs) (line 124)
        getcwd_call_result_29229 = invoke(stypy.reporting.localization.Localization(__file__, 124, 18), getcwd_29227, *[], **kwargs_29228)
        
        # Assigning a type to the variable 'old_dir' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'old_dir', getcwd_call_result_29229)
        
        # Call to chdir(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'tmpdir' (line 125)
        tmpdir_29232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'tmpdir', False)
        # Processing the call keyword arguments (line 125)
        kwargs_29233 = {}
        # Getting the type of 'os' (line 125)
        os_29230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 125)
        chdir_29231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), os_29230, 'chdir')
        # Calling chdir(args, kwargs) (line 125)
        chdir_call_result_29234 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), chdir_29231, *[tmpdir_29232], **kwargs_29233)
        
        
        # Try-finally block (line 126)
        
        # Call to make_tarball(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'base_name' (line 127)
        base_name_29236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 25), 'base_name', False)
        str_29237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 36), 'str', 'dist')
        # Processing the call keyword arguments (line 127)
        kwargs_29238 = {}
        # Getting the type of 'make_tarball' (line 127)
        make_tarball_29235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'make_tarball', False)
        # Calling make_tarball(args, kwargs) (line 127)
        make_tarball_call_result_29239 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), make_tarball_29235, *[base_name_29236, str_29237], **kwargs_29238)
        
        
        # finally branch of the try-finally block (line 126)
        
        # Call to chdir(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'old_dir' (line 129)
        old_dir_29242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'old_dir', False)
        # Processing the call keyword arguments (line 129)
        kwargs_29243 = {}
        # Getting the type of 'os' (line 129)
        os_29240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 129)
        chdir_29241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), os_29240, 'chdir')
        # Calling chdir(args, kwargs) (line 129)
        chdir_call_result_29244 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), chdir_29241, *[old_dir_29242], **kwargs_29243)
        
        
        
        # Assigning a BinOp to a Name (line 132):
        
        # Assigning a BinOp to a Name (line 132):
        # Getting the type of 'base_name' (line 132)
        base_name_29245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'base_name')
        str_29246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 30), 'str', '.tar.gz')
        # Applying the binary operator '+' (line 132)
        result_add_29247 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 18), '+', base_name_29245, str_29246)
        
        # Assigning a type to the variable 'tarball' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'tarball', result_add_29247)
        
        # Call to assertTrue(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Call to exists(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'tarball' (line 133)
        tarball_29253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 39), 'tarball', False)
        # Processing the call keyword arguments (line 133)
        kwargs_29254 = {}
        # Getting the type of 'os' (line 133)
        os_29250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 133)
        path_29251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 24), os_29250, 'path')
        # Obtaining the member 'exists' of a type (line 133)
        exists_29252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 24), path_29251, 'exists')
        # Calling exists(args, kwargs) (line 133)
        exists_call_result_29255 = invoke(stypy.reporting.localization.Localization(__file__, 133, 24), exists_29252, *[tarball_29253], **kwargs_29254)
        
        # Processing the call keyword arguments (line 133)
        kwargs_29256 = {}
        # Getting the type of 'self' (line 133)
        self_29248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 133)
        assertTrue_29249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), self_29248, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 133)
        assertTrue_call_result_29257 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assertTrue_29249, *[exists_call_result_29255], **kwargs_29256)
        
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to join(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'tmpdir' (line 136)
        tmpdir_29261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 32), 'tmpdir', False)
        str_29262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 40), 'str', 'archive2.tar.gz')
        # Processing the call keyword arguments (line 136)
        kwargs_29263 = {}
        # Getting the type of 'os' (line 136)
        os_29258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 136)
        path_29259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 19), os_29258, 'path')
        # Obtaining the member 'join' of a type (line 136)
        join_29260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 19), path_29259, 'join')
        # Calling join(args, kwargs) (line 136)
        join_call_result_29264 = invoke(stypy.reporting.localization.Localization(__file__, 136, 19), join_29260, *[tmpdir_29261, str_29262], **kwargs_29263)
        
        # Assigning a type to the variable 'tarball2' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tarball2', join_call_result_29264)
        
        # Assigning a List to a Name (line 137):
        
        # Assigning a List to a Name (line 137):
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_29265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        str_29266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 19), 'str', 'tar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 18), list_29265, str_29266)
        # Adding element type (line 137)
        str_29267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 26), 'str', '-cf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 18), list_29265, str_29267)
        # Adding element type (line 137)
        str_29268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 33), 'str', 'archive2.tar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 18), list_29265, str_29268)
        # Adding element type (line 137)
        str_29269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 49), 'str', 'dist')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 18), list_29265, str_29269)
        
        # Assigning a type to the variable 'tar_cmd' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tar_cmd', list_29265)
        
        # Assigning a List to a Name (line 138):
        
        # Assigning a List to a Name (line 138):
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_29270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        str_29271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'str', 'gzip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), list_29270, str_29271)
        # Adding element type (line 138)
        str_29272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 28), 'str', '-f9')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), list_29270, str_29272)
        # Adding element type (line 138)
        str_29273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 35), 'str', 'archive2.tar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), list_29270, str_29273)
        
        # Assigning a type to the variable 'gzip_cmd' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'gzip_cmd', list_29270)
        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to getcwd(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_29276 = {}
        # Getting the type of 'os' (line 139)
        os_29274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 18), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 139)
        getcwd_29275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 18), os_29274, 'getcwd')
        # Calling getcwd(args, kwargs) (line 139)
        getcwd_call_result_29277 = invoke(stypy.reporting.localization.Localization(__file__, 139, 18), getcwd_29275, *[], **kwargs_29276)
        
        # Assigning a type to the variable 'old_dir' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'old_dir', getcwd_call_result_29277)
        
        # Call to chdir(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'tmpdir' (line 140)
        tmpdir_29280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 17), 'tmpdir', False)
        # Processing the call keyword arguments (line 140)
        kwargs_29281 = {}
        # Getting the type of 'os' (line 140)
        os_29278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 140)
        chdir_29279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), os_29278, 'chdir')
        # Calling chdir(args, kwargs) (line 140)
        chdir_call_result_29282 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), chdir_29279, *[tmpdir_29280], **kwargs_29281)
        
        
        # Try-finally block (line 141)
        
        # Call to spawn(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'tar_cmd' (line 142)
        tar_cmd_29284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'tar_cmd', False)
        # Processing the call keyword arguments (line 142)
        kwargs_29285 = {}
        # Getting the type of 'spawn' (line 142)
        spawn_29283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'spawn', False)
        # Calling spawn(args, kwargs) (line 142)
        spawn_call_result_29286 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), spawn_29283, *[tar_cmd_29284], **kwargs_29285)
        
        
        # Call to spawn(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'gzip_cmd' (line 143)
        gzip_cmd_29288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'gzip_cmd', False)
        # Processing the call keyword arguments (line 143)
        kwargs_29289 = {}
        # Getting the type of 'spawn' (line 143)
        spawn_29287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'spawn', False)
        # Calling spawn(args, kwargs) (line 143)
        spawn_call_result_29290 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), spawn_29287, *[gzip_cmd_29288], **kwargs_29289)
        
        
        # finally branch of the try-finally block (line 141)
        
        # Call to chdir(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'old_dir' (line 145)
        old_dir_29293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 21), 'old_dir', False)
        # Processing the call keyword arguments (line 145)
        kwargs_29294 = {}
        # Getting the type of 'os' (line 145)
        os_29291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 145)
        chdir_29292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), os_29291, 'chdir')
        # Calling chdir(args, kwargs) (line 145)
        chdir_call_result_29295 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), chdir_29292, *[old_dir_29293], **kwargs_29294)
        
        
        
        # Call to assertTrue(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Call to exists(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'tarball2' (line 147)
        tarball2_29301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 39), 'tarball2', False)
        # Processing the call keyword arguments (line 147)
        kwargs_29302 = {}
        # Getting the type of 'os' (line 147)
        os_29298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 147)
        path_29299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 24), os_29298, 'path')
        # Obtaining the member 'exists' of a type (line 147)
        exists_29300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 24), path_29299, 'exists')
        # Calling exists(args, kwargs) (line 147)
        exists_call_result_29303 = invoke(stypy.reporting.localization.Localization(__file__, 147, 24), exists_29300, *[tarball2_29301], **kwargs_29302)
        
        # Processing the call keyword arguments (line 147)
        kwargs_29304 = {}
        # Getting the type of 'self' (line 147)
        self_29296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 147)
        assertTrue_29297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_29296, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 147)
        assertTrue_call_result_29305 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), assertTrue_29297, *[exists_call_result_29303], **kwargs_29304)
        
        
        # Call to assertEqual(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to _tarinfo(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'tarball' (line 149)
        tarball_29310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 39), 'tarball', False)
        # Processing the call keyword arguments (line 149)
        kwargs_29311 = {}
        # Getting the type of 'self' (line 149)
        self_29308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'self', False)
        # Obtaining the member '_tarinfo' of a type (line 149)
        _tarinfo_29309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 25), self_29308, '_tarinfo')
        # Calling _tarinfo(args, kwargs) (line 149)
        _tarinfo_call_result_29312 = invoke(stypy.reporting.localization.Localization(__file__, 149, 25), _tarinfo_29309, *[tarball_29310], **kwargs_29311)
        
        
        # Call to _tarinfo(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'tarball2' (line 149)
        tarball2_29315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 63), 'tarball2', False)
        # Processing the call keyword arguments (line 149)
        kwargs_29316 = {}
        # Getting the type of 'self' (line 149)
        self_29313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 49), 'self', False)
        # Obtaining the member '_tarinfo' of a type (line 149)
        _tarinfo_29314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 49), self_29313, '_tarinfo')
        # Calling _tarinfo(args, kwargs) (line 149)
        _tarinfo_call_result_29317 = invoke(stypy.reporting.localization.Localization(__file__, 149, 49), _tarinfo_29314, *[tarball2_29315], **kwargs_29316)
        
        # Processing the call keyword arguments (line 149)
        kwargs_29318 = {}
        # Getting the type of 'self' (line 149)
        self_29306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 149)
        assertEqual_29307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_29306, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 149)
        assertEqual_call_result_29319 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), assertEqual_29307, *[_tarinfo_call_result_29312, _tarinfo_call_result_29317], **kwargs_29318)
        
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to join(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'tmpdir2' (line 152)
        tmpdir2_29323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 33), 'tmpdir2', False)
        str_29324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 42), 'str', 'archive')
        # Processing the call keyword arguments (line 152)
        kwargs_29325 = {}
        # Getting the type of 'os' (line 152)
        os_29320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 152)
        path_29321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), os_29320, 'path')
        # Obtaining the member 'join' of a type (line 152)
        join_29322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), path_29321, 'join')
        # Calling join(args, kwargs) (line 152)
        join_call_result_29326 = invoke(stypy.reporting.localization.Localization(__file__, 152, 20), join_29322, *[tmpdir2_29323, str_29324], **kwargs_29325)
        
        # Assigning a type to the variable 'base_name' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'base_name', join_call_result_29326)
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to getcwd(...): (line 153)
        # Processing the call keyword arguments (line 153)
        kwargs_29329 = {}
        # Getting the type of 'os' (line 153)
        os_29327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 153)
        getcwd_29328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 18), os_29327, 'getcwd')
        # Calling getcwd(args, kwargs) (line 153)
        getcwd_call_result_29330 = invoke(stypy.reporting.localization.Localization(__file__, 153, 18), getcwd_29328, *[], **kwargs_29329)
        
        # Assigning a type to the variable 'old_dir' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'old_dir', getcwd_call_result_29330)
        
        # Call to chdir(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'tmpdir' (line 154)
        tmpdir_29333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'tmpdir', False)
        # Processing the call keyword arguments (line 154)
        kwargs_29334 = {}
        # Getting the type of 'os' (line 154)
        os_29331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 154)
        chdir_29332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), os_29331, 'chdir')
        # Calling chdir(args, kwargs) (line 154)
        chdir_call_result_29335 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), chdir_29332, *[tmpdir_29333], **kwargs_29334)
        
        
        # Try-finally block (line 155)
        
        # Call to make_tarball(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'base_name' (line 156)
        base_name_29337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'base_name', False)
        str_29338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 36), 'str', 'dist')
        # Processing the call keyword arguments (line 156)
        # Getting the type of 'None' (line 156)
        None_29339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'None', False)
        keyword_29340 = None_29339
        kwargs_29341 = {'compress': keyword_29340}
        # Getting the type of 'make_tarball' (line 156)
        make_tarball_29336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'make_tarball', False)
        # Calling make_tarball(args, kwargs) (line 156)
        make_tarball_call_result_29342 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), make_tarball_29336, *[base_name_29337, str_29338], **kwargs_29341)
        
        
        # finally branch of the try-finally block (line 155)
        
        # Call to chdir(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'old_dir' (line 158)
        old_dir_29345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'old_dir', False)
        # Processing the call keyword arguments (line 158)
        kwargs_29346 = {}
        # Getting the type of 'os' (line 158)
        os_29343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 158)
        chdir_29344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), os_29343, 'chdir')
        # Calling chdir(args, kwargs) (line 158)
        chdir_call_result_29347 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), chdir_29344, *[old_dir_29345], **kwargs_29346)
        
        
        
        # Assigning a BinOp to a Name (line 159):
        
        # Assigning a BinOp to a Name (line 159):
        # Getting the type of 'base_name' (line 159)
        base_name_29348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'base_name')
        str_29349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 30), 'str', '.tar')
        # Applying the binary operator '+' (line 159)
        result_add_29350 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 18), '+', base_name_29348, str_29349)
        
        # Assigning a type to the variable 'tarball' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'tarball', result_add_29350)
        
        # Call to assertTrue(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Call to exists(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'tarball' (line 160)
        tarball_29356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 39), 'tarball', False)
        # Processing the call keyword arguments (line 160)
        kwargs_29357 = {}
        # Getting the type of 'os' (line 160)
        os_29353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 160)
        path_29354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 24), os_29353, 'path')
        # Obtaining the member 'exists' of a type (line 160)
        exists_29355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 24), path_29354, 'exists')
        # Calling exists(args, kwargs) (line 160)
        exists_call_result_29358 = invoke(stypy.reporting.localization.Localization(__file__, 160, 24), exists_29355, *[tarball_29356], **kwargs_29357)
        
        # Processing the call keyword arguments (line 160)
        kwargs_29359 = {}
        # Getting the type of 'self' (line 160)
        self_29351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 160)
        assertTrue_29352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_29351, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 160)
        assertTrue_call_result_29360 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), assertTrue_29352, *[exists_call_result_29358], **kwargs_29359)
        
        
        # Assigning a Call to a Name (line 163):
        
        # Assigning a Call to a Name (line 163):
        
        # Call to join(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'tmpdir2' (line 163)
        tmpdir2_29364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 33), 'tmpdir2', False)
        str_29365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 42), 'str', 'archive')
        # Processing the call keyword arguments (line 163)
        kwargs_29366 = {}
        # Getting the type of 'os' (line 163)
        os_29361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 163)
        path_29362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 20), os_29361, 'path')
        # Obtaining the member 'join' of a type (line 163)
        join_29363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 20), path_29362, 'join')
        # Calling join(args, kwargs) (line 163)
        join_call_result_29367 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), join_29363, *[tmpdir2_29364, str_29365], **kwargs_29366)
        
        # Assigning a type to the variable 'base_name' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'base_name', join_call_result_29367)
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to getcwd(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_29370 = {}
        # Getting the type of 'os' (line 164)
        os_29368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 164)
        getcwd_29369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 18), os_29368, 'getcwd')
        # Calling getcwd(args, kwargs) (line 164)
        getcwd_call_result_29371 = invoke(stypy.reporting.localization.Localization(__file__, 164, 18), getcwd_29369, *[], **kwargs_29370)
        
        # Assigning a type to the variable 'old_dir' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'old_dir', getcwd_call_result_29371)
        
        # Call to chdir(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'tmpdir' (line 165)
        tmpdir_29374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'tmpdir', False)
        # Processing the call keyword arguments (line 165)
        kwargs_29375 = {}
        # Getting the type of 'os' (line 165)
        os_29372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 165)
        chdir_29373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), os_29372, 'chdir')
        # Calling chdir(args, kwargs) (line 165)
        chdir_call_result_29376 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), chdir_29373, *[tmpdir_29374], **kwargs_29375)
        
        
        # Try-finally block (line 166)
        
        # Call to make_tarball(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'base_name' (line 167)
        base_name_29378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'base_name', False)
        str_29379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 36), 'str', 'dist')
        # Processing the call keyword arguments (line 167)
        # Getting the type of 'None' (line 167)
        None_29380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 53), 'None', False)
        keyword_29381 = None_29380
        # Getting the type of 'True' (line 167)
        True_29382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 67), 'True', False)
        keyword_29383 = True_29382
        kwargs_29384 = {'compress': keyword_29381, 'dry_run': keyword_29383}
        # Getting the type of 'make_tarball' (line 167)
        make_tarball_29377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'make_tarball', False)
        # Calling make_tarball(args, kwargs) (line 167)
        make_tarball_call_result_29385 = invoke(stypy.reporting.localization.Localization(__file__, 167, 12), make_tarball_29377, *[base_name_29378, str_29379], **kwargs_29384)
        
        
        # finally branch of the try-finally block (line 166)
        
        # Call to chdir(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'old_dir' (line 169)
        old_dir_29388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'old_dir', False)
        # Processing the call keyword arguments (line 169)
        kwargs_29389 = {}
        # Getting the type of 'os' (line 169)
        os_29386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 169)
        chdir_29387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), os_29386, 'chdir')
        # Calling chdir(args, kwargs) (line 169)
        chdir_call_result_29390 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), chdir_29387, *[old_dir_29388], **kwargs_29389)
        
        
        
        # Assigning a BinOp to a Name (line 170):
        
        # Assigning a BinOp to a Name (line 170):
        # Getting the type of 'base_name' (line 170)
        base_name_29391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'base_name')
        str_29392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 30), 'str', '.tar')
        # Applying the binary operator '+' (line 170)
        result_add_29393 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 18), '+', base_name_29391, str_29392)
        
        # Assigning a type to the variable 'tarball' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tarball', result_add_29393)
        
        # Call to assertTrue(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Call to exists(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'tarball' (line 171)
        tarball_29399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 39), 'tarball', False)
        # Processing the call keyword arguments (line 171)
        kwargs_29400 = {}
        # Getting the type of 'os' (line 171)
        os_29396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 171)
        path_29397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 24), os_29396, 'path')
        # Obtaining the member 'exists' of a type (line 171)
        exists_29398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 24), path_29397, 'exists')
        # Calling exists(args, kwargs) (line 171)
        exists_call_result_29401 = invoke(stypy.reporting.localization.Localization(__file__, 171, 24), exists_29398, *[tarball_29399], **kwargs_29400)
        
        # Processing the call keyword arguments (line 171)
        kwargs_29402 = {}
        # Getting the type of 'self' (line 171)
        self_29394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 171)
        assertTrue_29395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_29394, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 171)
        assertTrue_call_result_29403 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assertTrue_29395, *[exists_call_result_29401], **kwargs_29402)
        
        
        # ################# End of 'test_tarfile_vs_tar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tarfile_vs_tar' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_29404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29404)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tarfile_vs_tar'
        return stypy_return_type_29404


    @norecursion
    def test_compress_deprecated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_compress_deprecated'
        module_type_store = module_type_store.open_function_context('test_compress_deprecated', 173, 4, False)
        # Assigning a type to the variable 'self' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_compress_deprecated')
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_compress_deprecated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_compress_deprecated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_compress_deprecated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_compress_deprecated(...)' code ##################

        
        # Assigning a Call to a Tuple (line 176):
        
        # Assigning a Subscript to a Name (line 176):
        
        # Obtaining the type of the subscript
        int_29405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'int')
        
        # Call to _create_files(...): (line 176)
        # Processing the call keyword arguments (line 176)
        kwargs_29408 = {}
        # Getting the type of 'self' (line 176)
        self_29406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 38), 'self', False)
        # Obtaining the member '_create_files' of a type (line 176)
        _create_files_29407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 38), self_29406, '_create_files')
        # Calling _create_files(args, kwargs) (line 176)
        _create_files_call_result_29409 = invoke(stypy.reporting.localization.Localization(__file__, 176, 38), _create_files_29407, *[], **kwargs_29408)
        
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___29410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), _create_files_call_result_29409, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_29411 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), getitem___29410, int_29405)
        
        # Assigning a type to the variable 'tuple_var_assignment_28882' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_28882', subscript_call_result_29411)
        
        # Assigning a Subscript to a Name (line 176):
        
        # Obtaining the type of the subscript
        int_29412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'int')
        
        # Call to _create_files(...): (line 176)
        # Processing the call keyword arguments (line 176)
        kwargs_29415 = {}
        # Getting the type of 'self' (line 176)
        self_29413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 38), 'self', False)
        # Obtaining the member '_create_files' of a type (line 176)
        _create_files_29414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 38), self_29413, '_create_files')
        # Calling _create_files(args, kwargs) (line 176)
        _create_files_call_result_29416 = invoke(stypy.reporting.localization.Localization(__file__, 176, 38), _create_files_29414, *[], **kwargs_29415)
        
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___29417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), _create_files_call_result_29416, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_29418 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), getitem___29417, int_29412)
        
        # Assigning a type to the variable 'tuple_var_assignment_28883' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_28883', subscript_call_result_29418)
        
        # Assigning a Subscript to a Name (line 176):
        
        # Obtaining the type of the subscript
        int_29419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'int')
        
        # Call to _create_files(...): (line 176)
        # Processing the call keyword arguments (line 176)
        kwargs_29422 = {}
        # Getting the type of 'self' (line 176)
        self_29420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 38), 'self', False)
        # Obtaining the member '_create_files' of a type (line 176)
        _create_files_29421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 38), self_29420, '_create_files')
        # Calling _create_files(args, kwargs) (line 176)
        _create_files_call_result_29423 = invoke(stypy.reporting.localization.Localization(__file__, 176, 38), _create_files_29421, *[], **kwargs_29422)
        
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___29424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), _create_files_call_result_29423, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_29425 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), getitem___29424, int_29419)
        
        # Assigning a type to the variable 'tuple_var_assignment_28884' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_28884', subscript_call_result_29425)
        
        # Assigning a Name to a Name (line 176):
        # Getting the type of 'tuple_var_assignment_28882' (line 176)
        tuple_var_assignment_28882_29426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_28882')
        # Assigning a type to the variable 'tmpdir' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tmpdir', tuple_var_assignment_28882_29426)
        
        # Assigning a Name to a Name (line 176):
        # Getting the type of 'tuple_var_assignment_28883' (line 176)
        tuple_var_assignment_28883_29427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_28883')
        # Assigning a type to the variable 'tmpdir2' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'tmpdir2', tuple_var_assignment_28883_29427)
        
        # Assigning a Name to a Name (line 176):
        # Getting the type of 'tuple_var_assignment_28884' (line 176)
        tuple_var_assignment_28884_29428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_28884')
        # Assigning a type to the variable 'base_name' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'base_name', tuple_var_assignment_28884_29428)
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to getcwd(...): (line 179)
        # Processing the call keyword arguments (line 179)
        kwargs_29431 = {}
        # Getting the type of 'os' (line 179)
        os_29429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 18), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 179)
        getcwd_29430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 18), os_29429, 'getcwd')
        # Calling getcwd(args, kwargs) (line 179)
        getcwd_call_result_29432 = invoke(stypy.reporting.localization.Localization(__file__, 179, 18), getcwd_29430, *[], **kwargs_29431)
        
        # Assigning a type to the variable 'old_dir' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'old_dir', getcwd_call_result_29432)
        
        # Call to chdir(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'tmpdir' (line 180)
        tmpdir_29435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'tmpdir', False)
        # Processing the call keyword arguments (line 180)
        kwargs_29436 = {}
        # Getting the type of 'os' (line 180)
        os_29433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 180)
        chdir_29434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), os_29433, 'chdir')
        # Calling chdir(args, kwargs) (line 180)
        chdir_call_result_29437 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), chdir_29434, *[tmpdir_29435], **kwargs_29436)
        
        
        # Try-finally block (line 181)
        
        # Call to check_warnings(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_29439 = {}
        # Getting the type of 'check_warnings' (line 182)
        check_warnings_29438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'check_warnings', False)
        # Calling check_warnings(args, kwargs) (line 182)
        check_warnings_call_result_29440 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), check_warnings_29438, *[], **kwargs_29439)
        
        with_29441 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 182, 17), check_warnings_call_result_29440, 'with parameter', '__enter__', '__exit__')

        if with_29441:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 182)
            enter___29442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 17), check_warnings_call_result_29440, '__enter__')
            with_enter_29443 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), enter___29442)
            # Assigning a type to the variable 'w' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'w', with_enter_29443)
            
            # Call to simplefilter(...): (line 183)
            # Processing the call arguments (line 183)
            str_29446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 38), 'str', 'always')
            # Processing the call keyword arguments (line 183)
            kwargs_29447 = {}
            # Getting the type of 'warnings' (line 183)
            warnings_29444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'warnings', False)
            # Obtaining the member 'simplefilter' of a type (line 183)
            simplefilter_29445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 16), warnings_29444, 'simplefilter')
            # Calling simplefilter(args, kwargs) (line 183)
            simplefilter_call_result_29448 = invoke(stypy.reporting.localization.Localization(__file__, 183, 16), simplefilter_29445, *[str_29446], **kwargs_29447)
            
            
            # Call to make_tarball(...): (line 184)
            # Processing the call arguments (line 184)
            # Getting the type of 'base_name' (line 184)
            base_name_29450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 29), 'base_name', False)
            str_29451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 40), 'str', 'dist')
            # Processing the call keyword arguments (line 184)
            str_29452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 57), 'str', 'compress')
            keyword_29453 = str_29452
            kwargs_29454 = {'compress': keyword_29453}
            # Getting the type of 'make_tarball' (line 184)
            make_tarball_29449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'make_tarball', False)
            # Calling make_tarball(args, kwargs) (line 184)
            make_tarball_call_result_29455 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), make_tarball_29449, *[base_name_29450, str_29451], **kwargs_29454)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 182)
            exit___29456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 17), check_warnings_call_result_29440, '__exit__')
            with_exit_29457 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), exit___29456, None, None, None)

        
        # finally branch of the try-finally block (line 181)
        
        # Call to chdir(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'old_dir' (line 186)
        old_dir_29460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 21), 'old_dir', False)
        # Processing the call keyword arguments (line 186)
        kwargs_29461 = {}
        # Getting the type of 'os' (line 186)
        os_29458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 186)
        chdir_29459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), os_29458, 'chdir')
        # Calling chdir(args, kwargs) (line 186)
        chdir_call_result_29462 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), chdir_29459, *[old_dir_29460], **kwargs_29461)
        
        
        
        # Assigning a BinOp to a Name (line 187):
        
        # Assigning a BinOp to a Name (line 187):
        # Getting the type of 'base_name' (line 187)
        base_name_29463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 18), 'base_name')
        str_29464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 30), 'str', '.tar.Z')
        # Applying the binary operator '+' (line 187)
        result_add_29465 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 18), '+', base_name_29463, str_29464)
        
        # Assigning a type to the variable 'tarball' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'tarball', result_add_29465)
        
        # Call to assertTrue(...): (line 188)
        # Processing the call arguments (line 188)
        
        # Call to exists(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'tarball' (line 188)
        tarball_29471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 39), 'tarball', False)
        # Processing the call keyword arguments (line 188)
        kwargs_29472 = {}
        # Getting the type of 'os' (line 188)
        os_29468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 188)
        path_29469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), os_29468, 'path')
        # Obtaining the member 'exists' of a type (line 188)
        exists_29470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), path_29469, 'exists')
        # Calling exists(args, kwargs) (line 188)
        exists_call_result_29473 = invoke(stypy.reporting.localization.Localization(__file__, 188, 24), exists_29470, *[tarball_29471], **kwargs_29472)
        
        # Processing the call keyword arguments (line 188)
        kwargs_29474 = {}
        # Getting the type of 'self' (line 188)
        self_29466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 188)
        assertTrue_29467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_29466, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 188)
        assertTrue_call_result_29475 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), assertTrue_29467, *[exists_call_result_29473], **kwargs_29474)
        
        
        # Call to assertEqual(...): (line 189)
        # Processing the call arguments (line 189)
        
        # Call to len(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'w' (line 189)
        w_29479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'w', False)
        # Obtaining the member 'warnings' of a type (line 189)
        warnings_29480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 29), w_29479, 'warnings')
        # Processing the call keyword arguments (line 189)
        kwargs_29481 = {}
        # Getting the type of 'len' (line 189)
        len_29478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'len', False)
        # Calling len(args, kwargs) (line 189)
        len_call_result_29482 = invoke(stypy.reporting.localization.Localization(__file__, 189, 25), len_29478, *[warnings_29480], **kwargs_29481)
        
        int_29483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 42), 'int')
        # Processing the call keyword arguments (line 189)
        kwargs_29484 = {}
        # Getting the type of 'self' (line 189)
        self_29476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 189)
        assertEqual_29477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), self_29476, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 189)
        assertEqual_call_result_29485 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), assertEqual_29477, *[len_call_result_29482, int_29483], **kwargs_29484)
        
        
        # Call to remove(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'tarball' (line 192)
        tarball_29488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 18), 'tarball', False)
        # Processing the call keyword arguments (line 192)
        kwargs_29489 = {}
        # Getting the type of 'os' (line 192)
        os_29486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'os', False)
        # Obtaining the member 'remove' of a type (line 192)
        remove_29487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), os_29486, 'remove')
        # Calling remove(args, kwargs) (line 192)
        remove_call_result_29490 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), remove_29487, *[tarball_29488], **kwargs_29489)
        
        
        # Assigning a Call to a Name (line 193):
        
        # Assigning a Call to a Name (line 193):
        
        # Call to getcwd(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_29493 = {}
        # Getting the type of 'os' (line 193)
        os_29491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 193)
        getcwd_29492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 18), os_29491, 'getcwd')
        # Calling getcwd(args, kwargs) (line 193)
        getcwd_call_result_29494 = invoke(stypy.reporting.localization.Localization(__file__, 193, 18), getcwd_29492, *[], **kwargs_29493)
        
        # Assigning a type to the variable 'old_dir' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'old_dir', getcwd_call_result_29494)
        
        # Call to chdir(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'tmpdir' (line 194)
        tmpdir_29497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'tmpdir', False)
        # Processing the call keyword arguments (line 194)
        kwargs_29498 = {}
        # Getting the type of 'os' (line 194)
        os_29495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 194)
        chdir_29496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), os_29495, 'chdir')
        # Calling chdir(args, kwargs) (line 194)
        chdir_call_result_29499 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), chdir_29496, *[tmpdir_29497], **kwargs_29498)
        
        
        # Try-finally block (line 195)
        
        # Call to check_warnings(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_29501 = {}
        # Getting the type of 'check_warnings' (line 196)
        check_warnings_29500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'check_warnings', False)
        # Calling check_warnings(args, kwargs) (line 196)
        check_warnings_call_result_29502 = invoke(stypy.reporting.localization.Localization(__file__, 196, 17), check_warnings_29500, *[], **kwargs_29501)
        
        with_29503 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 196, 17), check_warnings_call_result_29502, 'with parameter', '__enter__', '__exit__')

        if with_29503:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 196)
            enter___29504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 17), check_warnings_call_result_29502, '__enter__')
            with_enter_29505 = invoke(stypy.reporting.localization.Localization(__file__, 196, 17), enter___29504)
            # Assigning a type to the variable 'w' (line 196)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'w', with_enter_29505)
            
            # Call to simplefilter(...): (line 197)
            # Processing the call arguments (line 197)
            str_29508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 38), 'str', 'always')
            # Processing the call keyword arguments (line 197)
            kwargs_29509 = {}
            # Getting the type of 'warnings' (line 197)
            warnings_29506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'warnings', False)
            # Obtaining the member 'simplefilter' of a type (line 197)
            simplefilter_29507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 16), warnings_29506, 'simplefilter')
            # Calling simplefilter(args, kwargs) (line 197)
            simplefilter_call_result_29510 = invoke(stypy.reporting.localization.Localization(__file__, 197, 16), simplefilter_29507, *[str_29508], **kwargs_29509)
            
            
            # Call to make_tarball(...): (line 198)
            # Processing the call arguments (line 198)
            # Getting the type of 'base_name' (line 198)
            base_name_29512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 29), 'base_name', False)
            str_29513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 40), 'str', 'dist')
            # Processing the call keyword arguments (line 198)
            str_29514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 57), 'str', 'compress')
            keyword_29515 = str_29514
            # Getting the type of 'True' (line 199)
            True_29516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 37), 'True', False)
            keyword_29517 = True_29516
            kwargs_29518 = {'compress': keyword_29515, 'dry_run': keyword_29517}
            # Getting the type of 'make_tarball' (line 198)
            make_tarball_29511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'make_tarball', False)
            # Calling make_tarball(args, kwargs) (line 198)
            make_tarball_call_result_29519 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), make_tarball_29511, *[base_name_29512, str_29513], **kwargs_29518)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 196)
            exit___29520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 17), check_warnings_call_result_29502, '__exit__')
            with_exit_29521 = invoke(stypy.reporting.localization.Localization(__file__, 196, 17), exit___29520, None, None, None)

        
        # finally branch of the try-finally block (line 195)
        
        # Call to chdir(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'old_dir' (line 201)
        old_dir_29524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 21), 'old_dir', False)
        # Processing the call keyword arguments (line 201)
        kwargs_29525 = {}
        # Getting the type of 'os' (line 201)
        os_29522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 201)
        chdir_29523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 12), os_29522, 'chdir')
        # Calling chdir(args, kwargs) (line 201)
        chdir_call_result_29526 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), chdir_29523, *[old_dir_29524], **kwargs_29525)
        
        
        
        # Call to assertFalse(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Call to exists(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'tarball' (line 202)
        tarball_29532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 40), 'tarball', False)
        # Processing the call keyword arguments (line 202)
        kwargs_29533 = {}
        # Getting the type of 'os' (line 202)
        os_29529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 202)
        path_29530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 25), os_29529, 'path')
        # Obtaining the member 'exists' of a type (line 202)
        exists_29531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 25), path_29530, 'exists')
        # Calling exists(args, kwargs) (line 202)
        exists_call_result_29534 = invoke(stypy.reporting.localization.Localization(__file__, 202, 25), exists_29531, *[tarball_29532], **kwargs_29533)
        
        # Processing the call keyword arguments (line 202)
        kwargs_29535 = {}
        # Getting the type of 'self' (line 202)
        self_29527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 202)
        assertFalse_29528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_29527, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 202)
        assertFalse_call_result_29536 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), assertFalse_29528, *[exists_call_result_29534], **kwargs_29535)
        
        
        # Call to assertEqual(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Call to len(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'w' (line 203)
        w_29540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 29), 'w', False)
        # Obtaining the member 'warnings' of a type (line 203)
        warnings_29541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 29), w_29540, 'warnings')
        # Processing the call keyword arguments (line 203)
        kwargs_29542 = {}
        # Getting the type of 'len' (line 203)
        len_29539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'len', False)
        # Calling len(args, kwargs) (line 203)
        len_call_result_29543 = invoke(stypy.reporting.localization.Localization(__file__, 203, 25), len_29539, *[warnings_29541], **kwargs_29542)
        
        int_29544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 42), 'int')
        # Processing the call keyword arguments (line 203)
        kwargs_29545 = {}
        # Getting the type of 'self' (line 203)
        self_29537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 203)
        assertEqual_29538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_29537, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 203)
        assertEqual_call_result_29546 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), assertEqual_29538, *[len_call_result_29543, int_29544], **kwargs_29545)
        
        
        # ################# End of 'test_compress_deprecated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_compress_deprecated' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_29547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29547)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_compress_deprecated'
        return stypy_return_type_29547


    @norecursion
    def test_make_zipfile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_make_zipfile'
        module_type_store = module_type_store.open_function_context('test_make_zipfile', 205, 4, False)
        # Assigning a type to the variable 'self' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_make_zipfile')
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_make_zipfile.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_make_zipfile', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_make_zipfile', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_make_zipfile(...)' code ##################

        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to mkdtemp(...): (line 209)
        # Processing the call keyword arguments (line 209)
        kwargs_29550 = {}
        # Getting the type of 'self' (line 209)
        self_29548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 209)
        mkdtemp_29549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 17), self_29548, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 209)
        mkdtemp_call_result_29551 = invoke(stypy.reporting.localization.Localization(__file__, 209, 17), mkdtemp_29549, *[], **kwargs_29550)
        
        # Assigning a type to the variable 'tmpdir' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'tmpdir', mkdtemp_call_result_29551)
        
        # Call to write_file(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Obtaining an instance of the builtin type 'list' (line 210)
        list_29554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 210)
        # Adding element type (line 210)
        # Getting the type of 'tmpdir' (line 210)
        tmpdir_29555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 25), 'tmpdir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 24), list_29554, tmpdir_29555)
        # Adding element type (line 210)
        str_29556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 33), 'str', 'file1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 24), list_29554, str_29556)
        
        str_29557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 43), 'str', 'xxx')
        # Processing the call keyword arguments (line 210)
        kwargs_29558 = {}
        # Getting the type of 'self' (line 210)
        self_29552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 210)
        write_file_29553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), self_29552, 'write_file')
        # Calling write_file(args, kwargs) (line 210)
        write_file_call_result_29559 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), write_file_29553, *[list_29554, str_29557], **kwargs_29558)
        
        
        # Call to write_file(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_29562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        # Adding element type (line 211)
        # Getting the type of 'tmpdir' (line 211)
        tmpdir_29563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 25), 'tmpdir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 24), list_29562, tmpdir_29563)
        # Adding element type (line 211)
        str_29564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 33), 'str', 'file2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 24), list_29562, str_29564)
        
        str_29565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 43), 'str', 'xxx')
        # Processing the call keyword arguments (line 211)
        kwargs_29566 = {}
        # Getting the type of 'self' (line 211)
        self_29560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 211)
        write_file_29561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_29560, 'write_file')
        # Calling write_file(args, kwargs) (line 211)
        write_file_call_result_29567 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), write_file_29561, *[list_29562, str_29565], **kwargs_29566)
        
        
        # Assigning a Call to a Name (line 213):
        
        # Assigning a Call to a Name (line 213):
        
        # Call to mkdtemp(...): (line 213)
        # Processing the call keyword arguments (line 213)
        kwargs_29570 = {}
        # Getting the type of 'self' (line 213)
        self_29568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 213)
        mkdtemp_29569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 18), self_29568, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 213)
        mkdtemp_call_result_29571 = invoke(stypy.reporting.localization.Localization(__file__, 213, 18), mkdtemp_29569, *[], **kwargs_29570)
        
        # Assigning a type to the variable 'tmpdir2' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'tmpdir2', mkdtemp_call_result_29571)
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to join(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'tmpdir2' (line 214)
        tmpdir2_29575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 33), 'tmpdir2', False)
        str_29576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 42), 'str', 'archive')
        # Processing the call keyword arguments (line 214)
        kwargs_29577 = {}
        # Getting the type of 'os' (line 214)
        os_29572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 214)
        path_29573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), os_29572, 'path')
        # Obtaining the member 'join' of a type (line 214)
        join_29574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), path_29573, 'join')
        # Calling join(args, kwargs) (line 214)
        join_call_result_29578 = invoke(stypy.reporting.localization.Localization(__file__, 214, 20), join_29574, *[tmpdir2_29575, str_29576], **kwargs_29577)
        
        # Assigning a type to the variable 'base_name' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'base_name', join_call_result_29578)
        
        # Call to make_zipfile(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'base_name' (line 215)
        base_name_29580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'base_name', False)
        # Getting the type of 'tmpdir' (line 215)
        tmpdir_29581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 32), 'tmpdir', False)
        # Processing the call keyword arguments (line 215)
        kwargs_29582 = {}
        # Getting the type of 'make_zipfile' (line 215)
        make_zipfile_29579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'make_zipfile', False)
        # Calling make_zipfile(args, kwargs) (line 215)
        make_zipfile_call_result_29583 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), make_zipfile_29579, *[base_name_29580, tmpdir_29581], **kwargs_29582)
        
        
        # Assigning a BinOp to a Name (line 218):
        
        # Assigning a BinOp to a Name (line 218):
        # Getting the type of 'base_name' (line 218)
        base_name_29584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'base_name')
        str_29585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 30), 'str', '.zip')
        # Applying the binary operator '+' (line 218)
        result_add_29586 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 18), '+', base_name_29584, str_29585)
        
        # Assigning a type to the variable 'tarball' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'tarball', result_add_29586)
        
        # ################# End of 'test_make_zipfile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_make_zipfile' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_29587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29587)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_make_zipfile'
        return stypy_return_type_29587


    @norecursion
    def test_check_archive_formats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_archive_formats'
        module_type_store = module_type_store.open_function_context('test_check_archive_formats', 220, 4, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_check_archive_formats')
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_check_archive_formats.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_check_archive_formats', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_archive_formats', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_archive_formats(...)' code ##################

        
        # Call to assertEqual(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Call to check_archive_formats(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining an instance of the builtin type 'list' (line 221)
        list_29591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 221)
        # Adding element type (line 221)
        str_29592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 48), 'str', 'gztar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 47), list_29591, str_29592)
        # Adding element type (line 221)
        str_29593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 57), 'str', 'xxx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 47), list_29591, str_29593)
        # Adding element type (line 221)
        str_29594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 64), 'str', 'zip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 47), list_29591, str_29594)
        
        # Processing the call keyword arguments (line 221)
        kwargs_29595 = {}
        # Getting the type of 'check_archive_formats' (line 221)
        check_archive_formats_29590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 25), 'check_archive_formats', False)
        # Calling check_archive_formats(args, kwargs) (line 221)
        check_archive_formats_call_result_29596 = invoke(stypy.reporting.localization.Localization(__file__, 221, 25), check_archive_formats_29590, *[list_29591], **kwargs_29595)
        
        str_29597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 25), 'str', 'xxx')
        # Processing the call keyword arguments (line 221)
        kwargs_29598 = {}
        # Getting the type of 'self' (line 221)
        self_29588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 221)
        assertEqual_29589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_29588, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 221)
        assertEqual_call_result_29599 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), assertEqual_29589, *[check_archive_formats_call_result_29596, str_29597], **kwargs_29598)
        
        
        # Call to assertEqual(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Call to check_archive_formats(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_29603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        str_29604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 48), 'str', 'gztar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 47), list_29603, str_29604)
        # Adding element type (line 223)
        str_29605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 57), 'str', 'zip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 47), list_29603, str_29605)
        
        # Processing the call keyword arguments (line 223)
        kwargs_29606 = {}
        # Getting the type of 'check_archive_formats' (line 223)
        check_archive_formats_29602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'check_archive_formats', False)
        # Calling check_archive_formats(args, kwargs) (line 223)
        check_archive_formats_call_result_29607 = invoke(stypy.reporting.localization.Localization(__file__, 223, 25), check_archive_formats_29602, *[list_29603], **kwargs_29606)
        
        # Getting the type of 'None' (line 223)
        None_29608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 66), 'None', False)
        # Processing the call keyword arguments (line 223)
        kwargs_29609 = {}
        # Getting the type of 'self' (line 223)
        self_29600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 223)
        assertEqual_29601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), self_29600, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 223)
        assertEqual_call_result_29610 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), assertEqual_29601, *[check_archive_formats_call_result_29607, None_29608], **kwargs_29609)
        
        
        # ################# End of 'test_check_archive_formats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_archive_formats' in the type store
        # Getting the type of 'stypy_return_type' (line 220)
        stypy_return_type_29611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_archive_formats'
        return stypy_return_type_29611


    @norecursion
    def test_make_archive(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_make_archive'
        module_type_store = module_type_store.open_function_context('test_make_archive', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_make_archive')
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_make_archive.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_make_archive', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_make_archive', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_make_archive(...)' code ##################

        
        # Assigning a Call to a Name (line 226):
        
        # Assigning a Call to a Name (line 226):
        
        # Call to mkdtemp(...): (line 226)
        # Processing the call keyword arguments (line 226)
        kwargs_29614 = {}
        # Getting the type of 'self' (line 226)
        self_29612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 226)
        mkdtemp_29613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 17), self_29612, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 226)
        mkdtemp_call_result_29615 = invoke(stypy.reporting.localization.Localization(__file__, 226, 17), mkdtemp_29613, *[], **kwargs_29614)
        
        # Assigning a type to the variable 'tmpdir' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tmpdir', mkdtemp_call_result_29615)
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to join(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'tmpdir' (line 227)
        tmpdir_29619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'tmpdir', False)
        str_29620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 41), 'str', 'archive')
        # Processing the call keyword arguments (line 227)
        kwargs_29621 = {}
        # Getting the type of 'os' (line 227)
        os_29616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 227)
        path_29617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 20), os_29616, 'path')
        # Obtaining the member 'join' of a type (line 227)
        join_29618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 20), path_29617, 'join')
        # Calling join(args, kwargs) (line 227)
        join_call_result_29622 = invoke(stypy.reporting.localization.Localization(__file__, 227, 20), join_29618, *[tmpdir_29619, str_29620], **kwargs_29621)
        
        # Assigning a type to the variable 'base_name' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'base_name', join_call_result_29622)
        
        # Call to assertRaises(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'ValueError' (line 228)
        ValueError_29625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 26), 'ValueError', False)
        # Getting the type of 'make_archive' (line 228)
        make_archive_29626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 38), 'make_archive', False)
        # Getting the type of 'base_name' (line 228)
        base_name_29627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 52), 'base_name', False)
        str_29628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 63), 'str', 'xxx')
        # Processing the call keyword arguments (line 228)
        kwargs_29629 = {}
        # Getting the type of 'self' (line 228)
        self_29623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 228)
        assertRaises_29624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_29623, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 228)
        assertRaises_call_result_29630 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), assertRaises_29624, *[ValueError_29625, make_archive_29626, base_name_29627, str_29628], **kwargs_29629)
        
        
        # ################# End of 'test_make_archive(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_make_archive' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_29631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29631)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_make_archive'
        return stypy_return_type_29631


    @norecursion
    def test_make_archive_owner_group(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_make_archive_owner_group'
        module_type_store = module_type_store.open_function_context('test_make_archive_owner_group', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_make_archive_owner_group')
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_make_archive_owner_group.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_make_archive_owner_group', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_make_archive_owner_group', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_make_archive_owner_group(...)' code ##################

        
        # Getting the type of 'UID_GID_SUPPORT' (line 234)
        UID_GID_SUPPORT_29632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'UID_GID_SUPPORT')
        # Testing the type of an if condition (line 234)
        if_condition_29633 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 8), UID_GID_SUPPORT_29632)
        # Assigning a type to the variable 'if_condition_29633' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'if_condition_29633', if_condition_29633)
        # SSA begins for if statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 235):
        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        int_29634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 36), 'int')
        
        # Call to getgrgid(...): (line 235)
        # Processing the call arguments (line 235)
        int_29637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 33), 'int')
        # Processing the call keyword arguments (line 235)
        kwargs_29638 = {}
        # Getting the type of 'grp' (line 235)
        grp_29635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'grp', False)
        # Obtaining the member 'getgrgid' of a type (line 235)
        getgrgid_29636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 20), grp_29635, 'getgrgid')
        # Calling getgrgid(args, kwargs) (line 235)
        getgrgid_call_result_29639 = invoke(stypy.reporting.localization.Localization(__file__, 235, 20), getgrgid_29636, *[int_29637], **kwargs_29638)
        
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___29640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 20), getgrgid_call_result_29639, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_29641 = invoke(stypy.reporting.localization.Localization(__file__, 235, 20), getitem___29640, int_29634)
        
        # Assigning a type to the variable 'group' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'group', subscript_call_result_29641)
        
        # Assigning a Subscript to a Name (line 236):
        
        # Assigning a Subscript to a Name (line 236):
        
        # Obtaining the type of the subscript
        int_29642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 36), 'int')
        
        # Call to getpwuid(...): (line 236)
        # Processing the call arguments (line 236)
        int_29645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 33), 'int')
        # Processing the call keyword arguments (line 236)
        kwargs_29646 = {}
        # Getting the type of 'pwd' (line 236)
        pwd_29643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'pwd', False)
        # Obtaining the member 'getpwuid' of a type (line 236)
        getpwuid_29644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 20), pwd_29643, 'getpwuid')
        # Calling getpwuid(args, kwargs) (line 236)
        getpwuid_call_result_29647 = invoke(stypy.reporting.localization.Localization(__file__, 236, 20), getpwuid_29644, *[int_29645], **kwargs_29646)
        
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___29648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 20), getpwuid_call_result_29647, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_29649 = invoke(stypy.reporting.localization.Localization(__file__, 236, 20), getitem___29648, int_29642)
        
        # Assigning a type to the variable 'owner' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'owner', subscript_call_result_29649)
        # SSA branch for the else part of an if statement (line 234)
        module_type_store.open_ssa_branch('else')
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Str to a Name (line 238):
        str_29650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 28), 'str', 'root')
        # Assigning a type to the variable 'owner' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'owner', str_29650)
        
        # Assigning a Name to a Name (line 238):
        # Getting the type of 'owner' (line 238)
        owner_29651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'owner')
        # Assigning a type to the variable 'group' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'group', owner_29651)
        # SSA join for if statement (line 234)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 240):
        
        # Assigning a Subscript to a Name (line 240):
        
        # Obtaining the type of the subscript
        int_29652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'int')
        
        # Call to _create_files(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_29655 = {}
        # Getting the type of 'self' (line 240)
        self_29653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 41), 'self', False)
        # Obtaining the member '_create_files' of a type (line 240)
        _create_files_29654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 41), self_29653, '_create_files')
        # Calling _create_files(args, kwargs) (line 240)
        _create_files_call_result_29656 = invoke(stypy.reporting.localization.Localization(__file__, 240, 41), _create_files_29654, *[], **kwargs_29655)
        
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___29657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), _create_files_call_result_29656, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_29658 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), getitem___29657, int_29652)
        
        # Assigning a type to the variable 'tuple_var_assignment_28885' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_var_assignment_28885', subscript_call_result_29658)
        
        # Assigning a Subscript to a Name (line 240):
        
        # Obtaining the type of the subscript
        int_29659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'int')
        
        # Call to _create_files(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_29662 = {}
        # Getting the type of 'self' (line 240)
        self_29660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 41), 'self', False)
        # Obtaining the member '_create_files' of a type (line 240)
        _create_files_29661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 41), self_29660, '_create_files')
        # Calling _create_files(args, kwargs) (line 240)
        _create_files_call_result_29663 = invoke(stypy.reporting.localization.Localization(__file__, 240, 41), _create_files_29661, *[], **kwargs_29662)
        
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___29664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), _create_files_call_result_29663, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_29665 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), getitem___29664, int_29659)
        
        # Assigning a type to the variable 'tuple_var_assignment_28886' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_var_assignment_28886', subscript_call_result_29665)
        
        # Assigning a Subscript to a Name (line 240):
        
        # Obtaining the type of the subscript
        int_29666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'int')
        
        # Call to _create_files(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_29669 = {}
        # Getting the type of 'self' (line 240)
        self_29667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 41), 'self', False)
        # Obtaining the member '_create_files' of a type (line 240)
        _create_files_29668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 41), self_29667, '_create_files')
        # Calling _create_files(args, kwargs) (line 240)
        _create_files_call_result_29670 = invoke(stypy.reporting.localization.Localization(__file__, 240, 41), _create_files_29668, *[], **kwargs_29669)
        
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___29671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), _create_files_call_result_29670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_29672 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), getitem___29671, int_29666)
        
        # Assigning a type to the variable 'tuple_var_assignment_28887' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_var_assignment_28887', subscript_call_result_29672)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'tuple_var_assignment_28885' (line 240)
        tuple_var_assignment_28885_29673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_var_assignment_28885')
        # Assigning a type to the variable 'base_dir' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'base_dir', tuple_var_assignment_28885_29673)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'tuple_var_assignment_28886' (line 240)
        tuple_var_assignment_28886_29674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_var_assignment_28886')
        # Assigning a type to the variable 'root_dir' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'root_dir', tuple_var_assignment_28886_29674)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'tuple_var_assignment_28887' (line 240)
        tuple_var_assignment_28887_29675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_var_assignment_28887')
        # Assigning a type to the variable 'base_name' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 28), 'base_name', tuple_var_assignment_28887_29675)
        
        # Assigning a Call to a Name (line 241):
        
        # Assigning a Call to a Name (line 241):
        
        # Call to join(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Call to mkdtemp(...): (line 241)
        # Processing the call keyword arguments (line 241)
        kwargs_29681 = {}
        # Getting the type of 'self' (line 241)
        self_29679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 33), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 241)
        mkdtemp_29680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 33), self_29679, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 241)
        mkdtemp_call_result_29682 = invoke(stypy.reporting.localization.Localization(__file__, 241, 33), mkdtemp_29680, *[], **kwargs_29681)
        
        str_29683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 50), 'str', 'archive')
        # Processing the call keyword arguments (line 241)
        kwargs_29684 = {}
        # Getting the type of 'os' (line 241)
        os_29676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 241)
        path_29677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 20), os_29676, 'path')
        # Obtaining the member 'join' of a type (line 241)
        join_29678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 20), path_29677, 'join')
        # Calling join(args, kwargs) (line 241)
        join_call_result_29685 = invoke(stypy.reporting.localization.Localization(__file__, 241, 20), join_29678, *[mkdtemp_call_result_29682, str_29683], **kwargs_29684)
        
        # Assigning a type to the variable 'base_name' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'base_name', join_call_result_29685)
        
        # Assigning a Call to a Name (line 242):
        
        # Assigning a Call to a Name (line 242):
        
        # Call to make_archive(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'base_name' (line 242)
        base_name_29687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 27), 'base_name', False)
        str_29688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 38), 'str', 'zip')
        # Getting the type of 'root_dir' (line 242)
        root_dir_29689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 45), 'root_dir', False)
        # Getting the type of 'base_dir' (line 242)
        base_dir_29690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 55), 'base_dir', False)
        # Processing the call keyword arguments (line 242)
        # Getting the type of 'owner' (line 242)
        owner_29691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 71), 'owner', False)
        keyword_29692 = owner_29691
        # Getting the type of 'group' (line 243)
        group_29693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 33), 'group', False)
        keyword_29694 = group_29693
        kwargs_29695 = {'owner': keyword_29692, 'group': keyword_29694}
        # Getting the type of 'make_archive' (line 242)
        make_archive_29686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'make_archive', False)
        # Calling make_archive(args, kwargs) (line 242)
        make_archive_call_result_29696 = invoke(stypy.reporting.localization.Localization(__file__, 242, 14), make_archive_29686, *[base_name_29687, str_29688, root_dir_29689, base_dir_29690], **kwargs_29695)
        
        # Assigning a type to the variable 'res' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'res', make_archive_call_result_29696)
        
        # Call to assertTrue(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Call to exists(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'res' (line 244)
        res_29702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 39), 'res', False)
        # Processing the call keyword arguments (line 244)
        kwargs_29703 = {}
        # Getting the type of 'os' (line 244)
        os_29699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 244)
        path_29700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), os_29699, 'path')
        # Obtaining the member 'exists' of a type (line 244)
        exists_29701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), path_29700, 'exists')
        # Calling exists(args, kwargs) (line 244)
        exists_call_result_29704 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), exists_29701, *[res_29702], **kwargs_29703)
        
        # Processing the call keyword arguments (line 244)
        kwargs_29705 = {}
        # Getting the type of 'self' (line 244)
        self_29697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 244)
        assertTrue_29698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_29697, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 244)
        assertTrue_call_result_29706 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), assertTrue_29698, *[exists_call_result_29704], **kwargs_29705)
        
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to make_archive(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'base_name' (line 246)
        base_name_29708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 27), 'base_name', False)
        str_29709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 38), 'str', 'zip')
        # Getting the type of 'root_dir' (line 246)
        root_dir_29710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 45), 'root_dir', False)
        # Getting the type of 'base_dir' (line 246)
        base_dir_29711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 55), 'base_dir', False)
        # Processing the call keyword arguments (line 246)
        kwargs_29712 = {}
        # Getting the type of 'make_archive' (line 246)
        make_archive_29707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 14), 'make_archive', False)
        # Calling make_archive(args, kwargs) (line 246)
        make_archive_call_result_29713 = invoke(stypy.reporting.localization.Localization(__file__, 246, 14), make_archive_29707, *[base_name_29708, str_29709, root_dir_29710, base_dir_29711], **kwargs_29712)
        
        # Assigning a type to the variable 'res' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'res', make_archive_call_result_29713)
        
        # Call to assertTrue(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Call to exists(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'res' (line 247)
        res_29719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 39), 'res', False)
        # Processing the call keyword arguments (line 247)
        kwargs_29720 = {}
        # Getting the type of 'os' (line 247)
        os_29716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 247)
        path_29717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 24), os_29716, 'path')
        # Obtaining the member 'exists' of a type (line 247)
        exists_29718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 24), path_29717, 'exists')
        # Calling exists(args, kwargs) (line 247)
        exists_call_result_29721 = invoke(stypy.reporting.localization.Localization(__file__, 247, 24), exists_29718, *[res_29719], **kwargs_29720)
        
        # Processing the call keyword arguments (line 247)
        kwargs_29722 = {}
        # Getting the type of 'self' (line 247)
        self_29714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 247)
        assertTrue_29715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), self_29714, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 247)
        assertTrue_call_result_29723 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), assertTrue_29715, *[exists_call_result_29721], **kwargs_29722)
        
        
        # Assigning a Call to a Name (line 249):
        
        # Assigning a Call to a Name (line 249):
        
        # Call to make_archive(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'base_name' (line 249)
        base_name_29725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 27), 'base_name', False)
        str_29726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 38), 'str', 'tar')
        # Getting the type of 'root_dir' (line 249)
        root_dir_29727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 45), 'root_dir', False)
        # Getting the type of 'base_dir' (line 249)
        base_dir_29728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 55), 'base_dir', False)
        # Processing the call keyword arguments (line 249)
        # Getting the type of 'owner' (line 250)
        owner_29729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 33), 'owner', False)
        keyword_29730 = owner_29729
        # Getting the type of 'group' (line 250)
        group_29731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 46), 'group', False)
        keyword_29732 = group_29731
        kwargs_29733 = {'owner': keyword_29730, 'group': keyword_29732}
        # Getting the type of 'make_archive' (line 249)
        make_archive_29724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 14), 'make_archive', False)
        # Calling make_archive(args, kwargs) (line 249)
        make_archive_call_result_29734 = invoke(stypy.reporting.localization.Localization(__file__, 249, 14), make_archive_29724, *[base_name_29725, str_29726, root_dir_29727, base_dir_29728], **kwargs_29733)
        
        # Assigning a type to the variable 'res' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'res', make_archive_call_result_29734)
        
        # Call to assertTrue(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Call to exists(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'res' (line 251)
        res_29740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 39), 'res', False)
        # Processing the call keyword arguments (line 251)
        kwargs_29741 = {}
        # Getting the type of 'os' (line 251)
        os_29737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 251)
        path_29738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 24), os_29737, 'path')
        # Obtaining the member 'exists' of a type (line 251)
        exists_29739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 24), path_29738, 'exists')
        # Calling exists(args, kwargs) (line 251)
        exists_call_result_29742 = invoke(stypy.reporting.localization.Localization(__file__, 251, 24), exists_29739, *[res_29740], **kwargs_29741)
        
        # Processing the call keyword arguments (line 251)
        kwargs_29743 = {}
        # Getting the type of 'self' (line 251)
        self_29735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 251)
        assertTrue_29736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_29735, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 251)
        assertTrue_call_result_29744 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), assertTrue_29736, *[exists_call_result_29742], **kwargs_29743)
        
        
        # Assigning a Call to a Name (line 253):
        
        # Assigning a Call to a Name (line 253):
        
        # Call to make_archive(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'base_name' (line 253)
        base_name_29746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 27), 'base_name', False)
        str_29747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 38), 'str', 'tar')
        # Getting the type of 'root_dir' (line 253)
        root_dir_29748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 45), 'root_dir', False)
        # Getting the type of 'base_dir' (line 253)
        base_dir_29749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 55), 'base_dir', False)
        # Processing the call keyword arguments (line 253)
        str_29750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 33), 'str', 'kjhkjhkjg')
        keyword_29751 = str_29750
        str_29752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 52), 'str', 'oihohoh')
        keyword_29753 = str_29752
        kwargs_29754 = {'owner': keyword_29751, 'group': keyword_29753}
        # Getting the type of 'make_archive' (line 253)
        make_archive_29745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 14), 'make_archive', False)
        # Calling make_archive(args, kwargs) (line 253)
        make_archive_call_result_29755 = invoke(stypy.reporting.localization.Localization(__file__, 253, 14), make_archive_29745, *[base_name_29746, str_29747, root_dir_29748, base_dir_29749], **kwargs_29754)
        
        # Assigning a type to the variable 'res' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'res', make_archive_call_result_29755)
        
        # Call to assertTrue(...): (line 255)
        # Processing the call arguments (line 255)
        
        # Call to exists(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'res' (line 255)
        res_29761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 39), 'res', False)
        # Processing the call keyword arguments (line 255)
        kwargs_29762 = {}
        # Getting the type of 'os' (line 255)
        os_29758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 255)
        path_29759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 24), os_29758, 'path')
        # Obtaining the member 'exists' of a type (line 255)
        exists_29760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 24), path_29759, 'exists')
        # Calling exists(args, kwargs) (line 255)
        exists_call_result_29763 = invoke(stypy.reporting.localization.Localization(__file__, 255, 24), exists_29760, *[res_29761], **kwargs_29762)
        
        # Processing the call keyword arguments (line 255)
        kwargs_29764 = {}
        # Getting the type of 'self' (line 255)
        self_29756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 255)
        assertTrue_29757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), self_29756, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 255)
        assertTrue_call_result_29765 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assertTrue_29757, *[exists_call_result_29763], **kwargs_29764)
        
        
        # ################# End of 'test_make_archive_owner_group(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_make_archive_owner_group' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_29766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29766)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_make_archive_owner_group'
        return stypy_return_type_29766


    @norecursion
    def test_tarfile_root_owner(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tarfile_root_owner'
        module_type_store = module_type_store.open_function_context('test_tarfile_root_owner', 257, 4, False)
        # Assigning a type to the variable 'self' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_tarfile_root_owner')
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_tarfile_root_owner.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_tarfile_root_owner', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tarfile_root_owner', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tarfile_root_owner(...)' code ##################

        
        # Assigning a Call to a Tuple (line 260):
        
        # Assigning a Subscript to a Name (line 260):
        
        # Obtaining the type of the subscript
        int_29767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 8), 'int')
        
        # Call to _create_files(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_29770 = {}
        # Getting the type of 'self' (line 260)
        self_29768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 38), 'self', False)
        # Obtaining the member '_create_files' of a type (line 260)
        _create_files_29769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 38), self_29768, '_create_files')
        # Calling _create_files(args, kwargs) (line 260)
        _create_files_call_result_29771 = invoke(stypy.reporting.localization.Localization(__file__, 260, 38), _create_files_29769, *[], **kwargs_29770)
        
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___29772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), _create_files_call_result_29771, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_29773 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), getitem___29772, int_29767)
        
        # Assigning a type to the variable 'tuple_var_assignment_28888' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_28888', subscript_call_result_29773)
        
        # Assigning a Subscript to a Name (line 260):
        
        # Obtaining the type of the subscript
        int_29774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 8), 'int')
        
        # Call to _create_files(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_29777 = {}
        # Getting the type of 'self' (line 260)
        self_29775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 38), 'self', False)
        # Obtaining the member '_create_files' of a type (line 260)
        _create_files_29776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 38), self_29775, '_create_files')
        # Calling _create_files(args, kwargs) (line 260)
        _create_files_call_result_29778 = invoke(stypy.reporting.localization.Localization(__file__, 260, 38), _create_files_29776, *[], **kwargs_29777)
        
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___29779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), _create_files_call_result_29778, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_29780 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), getitem___29779, int_29774)
        
        # Assigning a type to the variable 'tuple_var_assignment_28889' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_28889', subscript_call_result_29780)
        
        # Assigning a Subscript to a Name (line 260):
        
        # Obtaining the type of the subscript
        int_29781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 8), 'int')
        
        # Call to _create_files(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_29784 = {}
        # Getting the type of 'self' (line 260)
        self_29782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 38), 'self', False)
        # Obtaining the member '_create_files' of a type (line 260)
        _create_files_29783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 38), self_29782, '_create_files')
        # Calling _create_files(args, kwargs) (line 260)
        _create_files_call_result_29785 = invoke(stypy.reporting.localization.Localization(__file__, 260, 38), _create_files_29783, *[], **kwargs_29784)
        
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___29786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), _create_files_call_result_29785, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_29787 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), getitem___29786, int_29781)
        
        # Assigning a type to the variable 'tuple_var_assignment_28890' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_28890', subscript_call_result_29787)
        
        # Assigning a Name to a Name (line 260):
        # Getting the type of 'tuple_var_assignment_28888' (line 260)
        tuple_var_assignment_28888_29788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_28888')
        # Assigning a type to the variable 'tmpdir' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tmpdir', tuple_var_assignment_28888_29788)
        
        # Assigning a Name to a Name (line 260):
        # Getting the type of 'tuple_var_assignment_28889' (line 260)
        tuple_var_assignment_28889_29789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_28889')
        # Assigning a type to the variable 'tmpdir2' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'tmpdir2', tuple_var_assignment_28889_29789)
        
        # Assigning a Name to a Name (line 260):
        # Getting the type of 'tuple_var_assignment_28890' (line 260)
        tuple_var_assignment_28890_29790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_28890')
        # Assigning a type to the variable 'base_name' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 25), 'base_name', tuple_var_assignment_28890_29790)
        
        # Assigning a Call to a Name (line 261):
        
        # Assigning a Call to a Name (line 261):
        
        # Call to getcwd(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_29793 = {}
        # Getting the type of 'os' (line 261)
        os_29791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 18), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 261)
        getcwd_29792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 18), os_29791, 'getcwd')
        # Calling getcwd(args, kwargs) (line 261)
        getcwd_call_result_29794 = invoke(stypy.reporting.localization.Localization(__file__, 261, 18), getcwd_29792, *[], **kwargs_29793)
        
        # Assigning a type to the variable 'old_dir' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'old_dir', getcwd_call_result_29794)
        
        # Call to chdir(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'tmpdir' (line 262)
        tmpdir_29797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 17), 'tmpdir', False)
        # Processing the call keyword arguments (line 262)
        kwargs_29798 = {}
        # Getting the type of 'os' (line 262)
        os_29795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 262)
        chdir_29796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), os_29795, 'chdir')
        # Calling chdir(args, kwargs) (line 262)
        chdir_call_result_29799 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), chdir_29796, *[tmpdir_29797], **kwargs_29798)
        
        
        # Assigning a Subscript to a Name (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        int_29800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 32), 'int')
        
        # Call to getgrgid(...): (line 263)
        # Processing the call arguments (line 263)
        int_29803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 29), 'int')
        # Processing the call keyword arguments (line 263)
        kwargs_29804 = {}
        # Getting the type of 'grp' (line 263)
        grp_29801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'grp', False)
        # Obtaining the member 'getgrgid' of a type (line 263)
        getgrgid_29802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 16), grp_29801, 'getgrgid')
        # Calling getgrgid(args, kwargs) (line 263)
        getgrgid_call_result_29805 = invoke(stypy.reporting.localization.Localization(__file__, 263, 16), getgrgid_29802, *[int_29803], **kwargs_29804)
        
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___29806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 16), getgrgid_call_result_29805, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_29807 = invoke(stypy.reporting.localization.Localization(__file__, 263, 16), getitem___29806, int_29800)
        
        # Assigning a type to the variable 'group' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'group', subscript_call_result_29807)
        
        # Assigning a Subscript to a Name (line 264):
        
        # Assigning a Subscript to a Name (line 264):
        
        # Obtaining the type of the subscript
        int_29808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 32), 'int')
        
        # Call to getpwuid(...): (line 264)
        # Processing the call arguments (line 264)
        int_29811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 29), 'int')
        # Processing the call keyword arguments (line 264)
        kwargs_29812 = {}
        # Getting the type of 'pwd' (line 264)
        pwd_29809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'pwd', False)
        # Obtaining the member 'getpwuid' of a type (line 264)
        getpwuid_29810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), pwd_29809, 'getpwuid')
        # Calling getpwuid(args, kwargs) (line 264)
        getpwuid_call_result_29813 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), getpwuid_29810, *[int_29811], **kwargs_29812)
        
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___29814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), getpwuid_call_result_29813, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_29815 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), getitem___29814, int_29808)
        
        # Assigning a type to the variable 'owner' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'owner', subscript_call_result_29815)
        
        # Try-finally block (line 265)
        
        # Assigning a Call to a Name (line 266):
        
        # Assigning a Call to a Name (line 266):
        
        # Call to make_tarball(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'base_name' (line 266)
        base_name_29817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 40), 'base_name', False)
        str_29818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 51), 'str', 'dist')
        # Processing the call keyword arguments (line 266)
        # Getting the type of 'None' (line 266)
        None_29819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 68), 'None', False)
        keyword_29820 = None_29819
        # Getting the type of 'owner' (line 267)
        owner_29821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 46), 'owner', False)
        keyword_29822 = owner_29821
        # Getting the type of 'group' (line 267)
        group_29823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 59), 'group', False)
        keyword_29824 = group_29823
        kwargs_29825 = {'owner': keyword_29822, 'group': keyword_29824, 'compress': keyword_29820}
        # Getting the type of 'make_tarball' (line 266)
        make_tarball_29816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 27), 'make_tarball', False)
        # Calling make_tarball(args, kwargs) (line 266)
        make_tarball_call_result_29826 = invoke(stypy.reporting.localization.Localization(__file__, 266, 27), make_tarball_29816, *[base_name_29817, str_29818], **kwargs_29825)
        
        # Assigning a type to the variable 'archive_name' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'archive_name', make_tarball_call_result_29826)
        
        # finally branch of the try-finally block (line 265)
        
        # Call to chdir(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'old_dir' (line 269)
        old_dir_29829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 'old_dir', False)
        # Processing the call keyword arguments (line 269)
        kwargs_29830 = {}
        # Getting the type of 'os' (line 269)
        os_29827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 269)
        chdir_29828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), os_29827, 'chdir')
        # Calling chdir(args, kwargs) (line 269)
        chdir_call_result_29831 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), chdir_29828, *[old_dir_29829], **kwargs_29830)
        
        
        
        # Call to assertTrue(...): (line 272)
        # Processing the call arguments (line 272)
        
        # Call to exists(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'archive_name' (line 272)
        archive_name_29837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 39), 'archive_name', False)
        # Processing the call keyword arguments (line 272)
        kwargs_29838 = {}
        # Getting the type of 'os' (line 272)
        os_29834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 272)
        path_29835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 24), os_29834, 'path')
        # Obtaining the member 'exists' of a type (line 272)
        exists_29836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 24), path_29835, 'exists')
        # Calling exists(args, kwargs) (line 272)
        exists_call_result_29839 = invoke(stypy.reporting.localization.Localization(__file__, 272, 24), exists_29836, *[archive_name_29837], **kwargs_29838)
        
        # Processing the call keyword arguments (line 272)
        kwargs_29840 = {}
        # Getting the type of 'self' (line 272)
        self_29832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 272)
        assertTrue_29833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_29832, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 272)
        assertTrue_call_result_29841 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), assertTrue_29833, *[exists_call_result_29839], **kwargs_29840)
        
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to open(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'archive_name' (line 275)
        archive_name_29844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 31), 'archive_name', False)
        # Processing the call keyword arguments (line 275)
        kwargs_29845 = {}
        # Getting the type of 'tarfile' (line 275)
        tarfile_29842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 18), 'tarfile', False)
        # Obtaining the member 'open' of a type (line 275)
        open_29843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 18), tarfile_29842, 'open')
        # Calling open(args, kwargs) (line 275)
        open_call_result_29846 = invoke(stypy.reporting.localization.Localization(__file__, 275, 18), open_29843, *[archive_name_29844], **kwargs_29845)
        
        # Assigning a type to the variable 'archive' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'archive', open_call_result_29846)
        
        # Try-finally block (line 276)
        
        
        # Call to getmembers(...): (line 277)
        # Processing the call keyword arguments (line 277)
        kwargs_29849 = {}
        # Getting the type of 'archive' (line 277)
        archive_29847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 26), 'archive', False)
        # Obtaining the member 'getmembers' of a type (line 277)
        getmembers_29848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 26), archive_29847, 'getmembers')
        # Calling getmembers(args, kwargs) (line 277)
        getmembers_call_result_29850 = invoke(stypy.reporting.localization.Localization(__file__, 277, 26), getmembers_29848, *[], **kwargs_29849)
        
        # Testing the type of a for loop iterable (line 277)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 12), getmembers_call_result_29850)
        # Getting the type of the for loop variable (line 277)
        for_loop_var_29851 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 12), getmembers_call_result_29850)
        # Assigning a type to the variable 'member' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'member', for_loop_var_29851)
        # SSA begins for a for statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertEqual(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'member' (line 278)
        member_29854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 33), 'member', False)
        # Obtaining the member 'uid' of a type (line 278)
        uid_29855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 33), member_29854, 'uid')
        int_29856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 45), 'int')
        # Processing the call keyword arguments (line 278)
        kwargs_29857 = {}
        # Getting the type of 'self' (line 278)
        self_29852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 278)
        assertEqual_29853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 16), self_29852, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 278)
        assertEqual_call_result_29858 = invoke(stypy.reporting.localization.Localization(__file__, 278, 16), assertEqual_29853, *[uid_29855, int_29856], **kwargs_29857)
        
        
        # Call to assertEqual(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'member' (line 279)
        member_29861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 33), 'member', False)
        # Obtaining the member 'gid' of a type (line 279)
        gid_29862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 33), member_29861, 'gid')
        int_29863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 45), 'int')
        # Processing the call keyword arguments (line 279)
        kwargs_29864 = {}
        # Getting the type of 'self' (line 279)
        self_29859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 279)
        assertEqual_29860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 16), self_29859, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 279)
        assertEqual_call_result_29865 = invoke(stypy.reporting.localization.Localization(__file__, 279, 16), assertEqual_29860, *[gid_29862, int_29863], **kwargs_29864)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # finally branch of the try-finally block (line 276)
        
        # Call to close(...): (line 281)
        # Processing the call keyword arguments (line 281)
        kwargs_29868 = {}
        # Getting the type of 'archive' (line 281)
        archive_29866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'archive', False)
        # Obtaining the member 'close' of a type (line 281)
        close_29867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), archive_29866, 'close')
        # Calling close(args, kwargs) (line 281)
        close_call_result_29869 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), close_29867, *[], **kwargs_29868)
        
        
        
        # ################# End of 'test_tarfile_root_owner(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tarfile_root_owner' in the type store
        # Getting the type of 'stypy_return_type' (line 257)
        stypy_return_type_29870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29870)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tarfile_root_owner'
        return stypy_return_type_29870


    @norecursion
    def test_make_archive_cwd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_make_archive_cwd'
        module_type_store = module_type_store.open_function_context('test_make_archive_cwd', 283, 4, False)
        # Assigning a type to the variable 'self' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_make_archive_cwd')
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_make_archive_cwd.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_make_archive_cwd', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_make_archive_cwd', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_make_archive_cwd(...)' code ##################

        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Call to getcwd(...): (line 284)
        # Processing the call keyword arguments (line 284)
        kwargs_29873 = {}
        # Getting the type of 'os' (line 284)
        os_29871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 22), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 284)
        getcwd_29872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 22), os_29871, 'getcwd')
        # Calling getcwd(args, kwargs) (line 284)
        getcwd_call_result_29874 = invoke(stypy.reporting.localization.Localization(__file__, 284, 22), getcwd_29872, *[], **kwargs_29873)
        
        # Assigning a type to the variable 'current_dir' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'current_dir', getcwd_call_result_29874)

        @norecursion
        def _breaks(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_breaks'
            module_type_store = module_type_store.open_function_context('_breaks', 285, 8, False)
            
            # Passed parameters checking function
            _breaks.stypy_localization = localization
            _breaks.stypy_type_of_self = None
            _breaks.stypy_type_store = module_type_store
            _breaks.stypy_function_name = '_breaks'
            _breaks.stypy_param_names_list = []
            _breaks.stypy_varargs_param_name = 'args'
            _breaks.stypy_kwargs_param_name = 'kw'
            _breaks.stypy_call_defaults = defaults
            _breaks.stypy_call_varargs = varargs
            _breaks.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_breaks', [], 'args', 'kw', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_breaks', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_breaks(...)' code ##################

            
            # Call to RuntimeError(...): (line 286)
            # Processing the call keyword arguments (line 286)
            kwargs_29876 = {}
            # Getting the type of 'RuntimeError' (line 286)
            RuntimeError_29875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 286)
            RuntimeError_call_result_29877 = invoke(stypy.reporting.localization.Localization(__file__, 286, 18), RuntimeError_29875, *[], **kwargs_29876)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 286, 12), RuntimeError_call_result_29877, 'raise parameter', BaseException)
            
            # ################# End of '_breaks(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_breaks' in the type store
            # Getting the type of 'stypy_return_type' (line 285)
            stypy_return_type_29878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_29878)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_breaks'
            return stypy_return_type_29878

        # Assigning a type to the variable '_breaks' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), '_breaks', _breaks)
        
        # Assigning a Tuple to a Subscript (line 287):
        
        # Assigning a Tuple to a Subscript (line 287):
        
        # Obtaining an instance of the builtin type 'tuple' (line 287)
        tuple_29879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 287)
        # Adding element type (line 287)
        # Getting the type of '_breaks' (line 287)
        _breaks_29880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 34), '_breaks')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 34), tuple_29879, _breaks_29880)
        # Adding element type (line 287)
        
        # Obtaining an instance of the builtin type 'list' (line 287)
        list_29881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 287)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 34), tuple_29879, list_29881)
        # Adding element type (line 287)
        str_29882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 47), 'str', 'xxx file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 34), tuple_29879, str_29882)
        
        # Getting the type of 'ARCHIVE_FORMATS' (line 287)
        ARCHIVE_FORMATS_29883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'ARCHIVE_FORMATS')
        str_29884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 24), 'str', 'xxx')
        # Storing an element on a container (line 287)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 8), ARCHIVE_FORMATS_29883, (str_29884, tuple_29879))
        
        # Try-finally block (line 288)
        
        
        # SSA begins for try-except statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to make_archive(...): (line 290)
        # Processing the call arguments (line 290)
        str_29886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 29), 'str', 'xxx')
        str_29887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 36), 'str', 'xxx')
        # Processing the call keyword arguments (line 290)
        
        # Call to mkdtemp(...): (line 290)
        # Processing the call keyword arguments (line 290)
        kwargs_29890 = {}
        # Getting the type of 'self' (line 290)
        self_29888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 52), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 290)
        mkdtemp_29889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 52), self_29888, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 290)
        mkdtemp_call_result_29891 = invoke(stypy.reporting.localization.Localization(__file__, 290, 52), mkdtemp_29889, *[], **kwargs_29890)
        
        keyword_29892 = mkdtemp_call_result_29891
        kwargs_29893 = {'root_dir': keyword_29892}
        # Getting the type of 'make_archive' (line 290)
        make_archive_29885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'make_archive', False)
        # Calling make_archive(args, kwargs) (line 290)
        make_archive_call_result_29894 = invoke(stypy.reporting.localization.Localization(__file__, 290, 16), make_archive_29885, *[str_29886, str_29887], **kwargs_29893)
        
        # SSA branch for the except part of a try statement (line 289)
        # SSA branch for the except '<any exception>' branch of a try statement (line 289)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertEqual(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Call to getcwd(...): (line 293)
        # Processing the call keyword arguments (line 293)
        kwargs_29899 = {}
        # Getting the type of 'os' (line 293)
        os_29897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 29), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 293)
        getcwd_29898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 29), os_29897, 'getcwd')
        # Calling getcwd(args, kwargs) (line 293)
        getcwd_call_result_29900 = invoke(stypy.reporting.localization.Localization(__file__, 293, 29), getcwd_29898, *[], **kwargs_29899)
        
        # Getting the type of 'current_dir' (line 293)
        current_dir_29901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 42), 'current_dir', False)
        # Processing the call keyword arguments (line 293)
        kwargs_29902 = {}
        # Getting the type of 'self' (line 293)
        self_29895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 293)
        assertEqual_29896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), self_29895, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 293)
        assertEqual_call_result_29903 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), assertEqual_29896, *[getcwd_call_result_29900, current_dir_29901], **kwargs_29902)
        
        
        # finally branch of the try-finally block (line 288)
        # Deleting a member
        # Getting the type of 'ARCHIVE_FORMATS' (line 295)
        ARCHIVE_FORMATS_29904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'ARCHIVE_FORMATS')
        
        # Obtaining the type of the subscript
        str_29905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 32), 'str', 'xxx')
        # Getting the type of 'ARCHIVE_FORMATS' (line 295)
        ARCHIVE_FORMATS_29906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'ARCHIVE_FORMATS')
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___29907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 16), ARCHIVE_FORMATS_29906, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 295)
        subscript_call_result_29908 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), getitem___29907, str_29905)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 12), ARCHIVE_FORMATS_29904, subscript_call_result_29908)
        
        
        # ################# End of 'test_make_archive_cwd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_make_archive_cwd' in the type store
        # Getting the type of 'stypy_return_type' (line 283)
        stypy_return_type_29909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29909)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_make_archive_cwd'
        return stypy_return_type_29909


    @norecursion
    def test_make_tarball_unicode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_make_tarball_unicode'
        module_type_store = module_type_store.open_function_context('test_make_tarball_unicode', 297, 4, False)
        # Assigning a type to the variable 'self' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_make_tarball_unicode')
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_make_tarball_unicode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_make_tarball_unicode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_make_tarball_unicode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_make_tarball_unicode(...)' code ##################

        str_29910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, (-1)), 'str', '\n        Mirror test_make_tarball, except filename is unicode.\n        ')
        
        # Call to _make_tarball(...): (line 302)
        # Processing the call arguments (line 302)
        unicode_29913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 27), 'unicode', u'archive')
        # Processing the call keyword arguments (line 302)
        kwargs_29914 = {}
        # Getting the type of 'self' (line 302)
        self_29911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'self', False)
        # Obtaining the member '_make_tarball' of a type (line 302)
        _make_tarball_29912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), self_29911, '_make_tarball')
        # Calling _make_tarball(args, kwargs) (line 302)
        _make_tarball_call_result_29915 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), _make_tarball_29912, *[unicode_29913], **kwargs_29914)
        
        
        # ################# End of 'test_make_tarball_unicode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_make_tarball_unicode' in the type store
        # Getting the type of 'stypy_return_type' (line 297)
        stypy_return_type_29916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29916)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_make_tarball_unicode'
        return stypy_return_type_29916


    @norecursion
    def test_make_tarball_unicode_latin1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_make_tarball_unicode_latin1'
        module_type_store = module_type_store.open_function_context('test_make_tarball_unicode_latin1', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_make_tarball_unicode_latin1')
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_make_tarball_unicode_latin1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_make_tarball_unicode_latin1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_make_tarball_unicode_latin1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_make_tarball_unicode_latin1(...)' code ##################

        str_29917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, (-1)), 'str', '\n        Mirror test_make_tarball, except filename is unicode and contains\n        latin characters.\n        ')
        
        # Call to _make_tarball(...): (line 312)
        # Processing the call arguments (line 312)
        unicode_29920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 27), 'unicode', u'\xe5rchiv')
        # Processing the call keyword arguments (line 312)
        kwargs_29921 = {}
        # Getting the type of 'self' (line 312)
        self_29918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self', False)
        # Obtaining the member '_make_tarball' of a type (line 312)
        _make_tarball_29919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), self_29918, '_make_tarball')
        # Calling _make_tarball(args, kwargs) (line 312)
        _make_tarball_call_result_29922 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), _make_tarball_29919, *[unicode_29920], **kwargs_29921)
        
        
        # ################# End of 'test_make_tarball_unicode_latin1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_make_tarball_unicode_latin1' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_29923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29923)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_make_tarball_unicode_latin1'
        return stypy_return_type_29923


    @norecursion
    def test_make_tarball_unicode_extended(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_make_tarball_unicode_extended'
        module_type_store = module_type_store.open_function_context('test_make_tarball_unicode_extended', 314, 4, False)
        # Assigning a type to the variable 'self' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_localization', localization)
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_function_name', 'ArchiveUtilTestCase.test_make_tarball_unicode_extended')
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_param_names_list', [])
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArchiveUtilTestCase.test_make_tarball_unicode_extended.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.test_make_tarball_unicode_extended', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_make_tarball_unicode_extended', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_make_tarball_unicode_extended(...)' code ##################

        str_29924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, (-1)), 'str', '\n        Mirror test_make_tarball, except filename is unicode and contains\n        characters outside the latin charset.\n        ')
        
        # Call to _make_tarball(...): (line 322)
        # Processing the call arguments (line 322)
        unicode_29927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 27), 'unicode', u'\u306e\u30a2\u30fc\u30ab\u30a4\u30d6')
        # Processing the call keyword arguments (line 322)
        kwargs_29928 = {}
        # Getting the type of 'self' (line 322)
        self_29925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'self', False)
        # Obtaining the member '_make_tarball' of a type (line 322)
        _make_tarball_29926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), self_29925, '_make_tarball')
        # Calling _make_tarball(args, kwargs) (line 322)
        _make_tarball_call_result_29929 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), _make_tarball_29926, *[unicode_29927], **kwargs_29928)
        
        
        # ################# End of 'test_make_tarball_unicode_extended(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_make_tarball_unicode_extended' in the type store
        # Getting the type of 'stypy_return_type' (line 314)
        stypy_return_type_29930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29930)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_make_tarball_unicode_extended'
        return stypy_return_type_29930


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 51, 0, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArchiveUtilTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ArchiveUtilTestCase' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'ArchiveUtilTestCase', ArchiveUtilTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 324, 0, False)
    
    # Passed parameters checking function
    test_suite.stypy_localization = localization
    test_suite.stypy_type_of_self = None
    test_suite.stypy_type_store = module_type_store
    test_suite.stypy_function_name = 'test_suite'
    test_suite.stypy_param_names_list = []
    test_suite.stypy_varargs_param_name = None
    test_suite.stypy_kwargs_param_name = None
    test_suite.stypy_call_defaults = defaults
    test_suite.stypy_call_varargs = varargs
    test_suite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_suite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_suite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_suite(...)' code ##################

    
    # Call to makeSuite(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'ArchiveUtilTestCase' (line 325)
    ArchiveUtilTestCase_29933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 30), 'ArchiveUtilTestCase', False)
    # Processing the call keyword arguments (line 325)
    kwargs_29934 = {}
    # Getting the type of 'unittest' (line 325)
    unittest_29931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 325)
    makeSuite_29932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 11), unittest_29931, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 325)
    makeSuite_call_result_29935 = invoke(stypy.reporting.localization.Localization(__file__, 325, 11), makeSuite_29932, *[ArchiveUtilTestCase_29933], **kwargs_29934)
    
    # Assigning a type to the variable 'stypy_return_type' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type', makeSuite_call_result_29935)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 324)
    stypy_return_type_29936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29936)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_29936

# Assigning a type to the variable 'test_suite' (line 324)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 328)
    # Processing the call arguments (line 328)
    
    # Call to test_suite(...): (line 328)
    # Processing the call keyword arguments (line 328)
    kwargs_29939 = {}
    # Getting the type of 'test_suite' (line 328)
    test_suite_29938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 328)
    test_suite_call_result_29940 = invoke(stypy.reporting.localization.Localization(__file__, 328, 17), test_suite_29938, *[], **kwargs_29939)
    
    # Processing the call keyword arguments (line 328)
    kwargs_29941 = {}
    # Getting the type of 'run_unittest' (line 328)
    run_unittest_29937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 328)
    run_unittest_call_result_29942 = invoke(stypy.reporting.localization.Localization(__file__, 328, 4), run_unittest_29937, *[test_suite_call_result_29940], **kwargs_29941)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
