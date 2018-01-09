
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: iso-8859-1 -*-
2: # Copyright (C) 2005, 2006 Martin von Löwis
3: # Licensed to PSF under a Contributor Agreement.
4: # The bdist_wininst command proper
5: # based on bdist_wininst
6: '''
7: Implements the bdist_msi command.
8: '''
9: import sys, os
10: from sysconfig import get_python_version
11: 
12: from distutils.core import Command
13: from distutils.dir_util import remove_tree
14: from distutils.version import StrictVersion
15: from distutils.errors import DistutilsOptionError
16: from distutils import log
17: from distutils.util import get_platform
18: 
19: import msilib
20: from msilib import schema, sequence, text
21: from msilib import Directory, Feature, Dialog, add_data
22: 
23: class PyDialog(Dialog):
24:     '''Dialog class with a fixed layout: controls at the top, then a ruler,
25:     then a list of buttons: back, next, cancel. Optionally a bitmap at the
26:     left.'''
27:     def __init__(self, *args, **kw):
28:         '''Dialog(database, name, x, y, w, h, attributes, title, first,
29:         default, cancel, bitmap=true)'''
30:         Dialog.__init__(self, *args)
31:         ruler = self.h - 36
32:         #if kw.get("bitmap", True):
33:         #    self.bitmap("Bitmap", 0, 0, bmwidth, ruler, "PythonWin")
34:         self.line("BottomLine", 0, ruler, self.w, 0)
35: 
36:     def title(self, title):
37:         "Set the title text of the dialog at the top."
38:         # name, x, y, w, h, flags=Visible|Enabled|Transparent|NoPrefix,
39:         # text, in VerdanaBold10
40:         self.text("Title", 15, 10, 320, 60, 0x30003,
41:                   r"{\VerdanaBold10}%s" % title)
42: 
43:     def back(self, title, next, name = "Back", active = 1):
44:         '''Add a back button with a given title, the tab-next button,
45:         its name in the Control table, possibly initially disabled.
46: 
47:         Return the button, so that events can be associated'''
48:         if active:
49:             flags = 3 # Visible|Enabled
50:         else:
51:             flags = 1 # Visible
52:         return self.pushbutton(name, 180, self.h-27 , 56, 17, flags, title, next)
53: 
54:     def cancel(self, title, next, name = "Cancel", active = 1):
55:         '''Add a cancel button with a given title, the tab-next button,
56:         its name in the Control table, possibly initially disabled.
57: 
58:         Return the button, so that events can be associated'''
59:         if active:
60:             flags = 3 # Visible|Enabled
61:         else:
62:             flags = 1 # Visible
63:         return self.pushbutton(name, 304, self.h-27, 56, 17, flags, title, next)
64: 
65:     def next(self, title, next, name = "Next", active = 1):
66:         '''Add a Next button with a given title, the tab-next button,
67:         its name in the Control table, possibly initially disabled.
68: 
69:         Return the button, so that events can be associated'''
70:         if active:
71:             flags = 3 # Visible|Enabled
72:         else:
73:             flags = 1 # Visible
74:         return self.pushbutton(name, 236, self.h-27, 56, 17, flags, title, next)
75: 
76:     def xbutton(self, name, title, next, xpos):
77:         '''Add a button with a given title, the tab-next button,
78:         its name in the Control table, giving its x position; the
79:         y-position is aligned with the other buttons.
80: 
81:         Return the button, so that events can be associated'''
82:         return self.pushbutton(name, int(self.w*xpos - 28), self.h-27, 56, 17, 3, title, next)
83: 
84: class bdist_msi (Command):
85: 
86:     description = "create a Microsoft Installer (.msi) binary distribution"
87: 
88:     user_options = [('bdist-dir=', None,
89:                      "temporary directory for creating the distribution"),
90:                     ('plat-name=', 'p',
91:                      "platform name to embed in generated filenames "
92:                      "(default: %s)" % get_platform()),
93:                     ('keep-temp', 'k',
94:                      "keep the pseudo-installation tree around after " +
95:                      "creating the distribution archive"),
96:                     ('target-version=', None,
97:                      "require a specific python version" +
98:                      " on the target system"),
99:                     ('no-target-compile', 'c',
100:                      "do not compile .py to .pyc on the target system"),
101:                     ('no-target-optimize', 'o',
102:                      "do not compile .py to .pyo (optimized)"
103:                      "on the target system"),
104:                     ('dist-dir=', 'd',
105:                      "directory to put final built distributions in"),
106:                     ('skip-build', None,
107:                      "skip rebuilding everything (for testing/debugging)"),
108:                     ('install-script=', None,
109:                      "basename of installation script to be run after"
110:                      "installation or before deinstallation"),
111:                     ('pre-install-script=', None,
112:                      "Fully qualified filename of a script to be run before "
113:                      "any files are installed.  This script need not be in the "
114:                      "distribution"),
115:                    ]
116: 
117:     boolean_options = ['keep-temp', 'no-target-compile', 'no-target-optimize',
118:                        'skip-build']
119: 
120:     all_versions = ['2.0', '2.1', '2.2', '2.3', '2.4',
121:                     '2.5', '2.6', '2.7', '2.8', '2.9',
122:                     '3.0', '3.1', '3.2', '3.3', '3.4',
123:                     '3.5', '3.6', '3.7', '3.8', '3.9']
124:     other_version = 'X'
125: 
126:     def initialize_options (self):
127:         self.bdist_dir = None
128:         self.plat_name = None
129:         self.keep_temp = 0
130:         self.no_target_compile = 0
131:         self.no_target_optimize = 0
132:         self.target_version = None
133:         self.dist_dir = None
134:         self.skip_build = None
135:         self.install_script = None
136:         self.pre_install_script = None
137:         self.versions = None
138: 
139:     def finalize_options (self):
140:         self.set_undefined_options('bdist', ('skip_build', 'skip_build'))
141: 
142:         if self.bdist_dir is None:
143:             bdist_base = self.get_finalized_command('bdist').bdist_base
144:             self.bdist_dir = os.path.join(bdist_base, 'msi')
145: 
146:         short_version = get_python_version()
147:         if (not self.target_version) and self.distribution.has_ext_modules():
148:             self.target_version = short_version
149: 
150:         if self.target_version:
151:             self.versions = [self.target_version]
152:             if not self.skip_build and self.distribution.has_ext_modules()\
153:                and self.target_version != short_version:
154:                 raise DistutilsOptionError, \
155:                       "target version can only be %s, or the '--skip-build'" \
156:                       " option must be specified" % (short_version,)
157:         else:
158:             self.versions = list(self.all_versions)
159: 
160:         self.set_undefined_options('bdist',
161:                                    ('dist_dir', 'dist_dir'),
162:                                    ('plat_name', 'plat_name'),
163:                                    )
164: 
165:         if self.pre_install_script:
166:             raise DistutilsOptionError, "the pre-install-script feature is not yet implemented"
167: 
168:         if self.install_script:
169:             for script in self.distribution.scripts:
170:                 if self.install_script == os.path.basename(script):
171:                     break
172:             else:
173:                 raise DistutilsOptionError, \
174:                       "install_script '%s' not found in scripts" % \
175:                       self.install_script
176:         self.install_script_key = None
177:     # finalize_options()
178: 
179: 
180:     def run (self):
181:         if not self.skip_build:
182:             self.run_command('build')
183: 
184:         install = self.reinitialize_command('install', reinit_subcommands=1)
185:         install.prefix = self.bdist_dir
186:         install.skip_build = self.skip_build
187:         install.warn_dir = 0
188: 
189:         install_lib = self.reinitialize_command('install_lib')
190:         # we do not want to include pyc or pyo files
191:         install_lib.compile = 0
192:         install_lib.optimize = 0
193: 
194:         if self.distribution.has_ext_modules():
195:             # If we are building an installer for a Python version other
196:             # than the one we are currently running, then we need to ensure
197:             # our build_lib reflects the other Python version rather than ours.
198:             # Note that for target_version!=sys.version, we must have skipped the
199:             # build step, so there is no issue with enforcing the build of this
200:             # version.
201:             target_version = self.target_version
202:             if not target_version:
203:                 assert self.skip_build, "Should have already checked this"
204:                 target_version = sys.version[0:3]
205:             plat_specifier = ".%s-%s" % (self.plat_name, target_version)
206:             build = self.get_finalized_command('build')
207:             build.build_lib = os.path.join(build.build_base,
208:                                            'lib' + plat_specifier)
209: 
210:         log.info("installing to %s", self.bdist_dir)
211:         install.ensure_finalized()
212: 
213:         # avoid warning of 'install_lib' about installing
214:         # into a directory not in sys.path
215:         sys.path.insert(0, os.path.join(self.bdist_dir, 'PURELIB'))
216: 
217:         install.run()
218: 
219:         del sys.path[0]
220: 
221:         self.mkpath(self.dist_dir)
222:         fullname = self.distribution.get_fullname()
223:         installer_name = self.get_installer_filename(fullname)
224:         installer_name = os.path.abspath(installer_name)
225:         if os.path.exists(installer_name): os.unlink(installer_name)
226: 
227:         metadata = self.distribution.metadata
228:         author = metadata.author
229:         if not author:
230:             author = metadata.maintainer
231:         if not author:
232:             author = "UNKNOWN"
233:         version = metadata.get_version()
234:         # ProductVersion must be strictly numeric
235:         # XXX need to deal with prerelease versions
236:         sversion = "%d.%d.%d" % StrictVersion(version).version
237:         # Prefix ProductName with Python x.y, so that
238:         # it sorts together with the other Python packages
239:         # in Add-Remove-Programs (APR)
240:         fullname = self.distribution.get_fullname()
241:         if self.target_version:
242:             product_name = "Python %s %s" % (self.target_version, fullname)
243:         else:
244:             product_name = "Python %s" % (fullname)
245:         self.db = msilib.init_database(installer_name, schema,
246:                 product_name, msilib.gen_uuid(),
247:                 sversion, author)
248:         msilib.add_tables(self.db, sequence)
249:         props = [('DistVersion', version)]
250:         email = metadata.author_email or metadata.maintainer_email
251:         if email:
252:             props.append(("ARPCONTACT", email))
253:         if metadata.url:
254:             props.append(("ARPURLINFOABOUT", metadata.url))
255:         if props:
256:             add_data(self.db, 'Property', props)
257: 
258:         self.add_find_python()
259:         self.add_files()
260:         self.add_scripts()
261:         self.add_ui()
262:         self.db.Commit()
263: 
264:         if hasattr(self.distribution, 'dist_files'):
265:             tup = 'bdist_msi', self.target_version or 'any', fullname
266:             self.distribution.dist_files.append(tup)
267: 
268:         if not self.keep_temp:
269:             remove_tree(self.bdist_dir, dry_run=self.dry_run)
270: 
271:     def add_files(self):
272:         db = self.db
273:         cab = msilib.CAB("distfiles")
274:         rootdir = os.path.abspath(self.bdist_dir)
275: 
276:         root = Directory(db, cab, None, rootdir, "TARGETDIR", "SourceDir")
277:         f = Feature(db, "Python", "Python", "Everything",
278:                     0, 1, directory="TARGETDIR")
279: 
280:         items = [(f, root, '')]
281:         for version in self.versions + [self.other_version]:
282:             target = "TARGETDIR" + version
283:             name = default = "Python" + version
284:             desc = "Everything"
285:             if version is self.other_version:
286:                 title = "Python from another location"
287:                 level = 2
288:             else:
289:                 title = "Python %s from registry" % version
290:                 level = 1
291:             f = Feature(db, name, title, desc, 1, level, directory=target)
292:             dir = Directory(db, cab, root, rootdir, target, default)
293:             items.append((f, dir, version))
294:         db.Commit()
295: 
296:         seen = {}
297:         for feature, dir, version in items:
298:             todo = [dir]
299:             while todo:
300:                 dir = todo.pop()
301:                 for file in os.listdir(dir.absolute):
302:                     afile = os.path.join(dir.absolute, file)
303:                     if os.path.isdir(afile):
304:                         short = "%s|%s" % (dir.make_short(file), file)
305:                         default = file + version
306:                         newdir = Directory(db, cab, dir, file, default, short)
307:                         todo.append(newdir)
308:                     else:
309:                         if not dir.component:
310:                             dir.start_component(dir.logical, feature, 0)
311:                         if afile not in seen:
312:                             key = seen[afile] = dir.add_file(file)
313:                             if file==self.install_script:
314:                                 if self.install_script_key:
315:                                     raise DistutilsOptionError(
316:                                           "Multiple files with name %s" % file)
317:                                 self.install_script_key = '[#%s]' % key
318:                         else:
319:                             key = seen[afile]
320:                             add_data(self.db, "DuplicateFile",
321:                                 [(key + version, dir.component, key, None, dir.logical)])
322:             db.Commit()
323:         cab.commit(db)
324: 
325:     def add_find_python(self):
326:         '''Adds code to the installer to compute the location of Python.
327: 
328:         Properties PYTHON.MACHINE.X.Y and PYTHON.USER.X.Y will be set from the
329:         registry for each version of Python.
330: 
331:         Properties TARGETDIRX.Y will be set from PYTHON.USER.X.Y if defined,
332:         else from PYTHON.MACHINE.X.Y.
333: 
334:         Properties PYTHONX.Y will be set to TARGETDIRX.Y\\python.exe'''
335: 
336:         start = 402
337:         for ver in self.versions:
338:             install_path = r"SOFTWARE\Python\PythonCore\%s\InstallPath" % ver
339:             machine_reg = "python.machine." + ver
340:             user_reg = "python.user." + ver
341:             machine_prop = "PYTHON.MACHINE." + ver
342:             user_prop = "PYTHON.USER." + ver
343:             machine_action = "PythonFromMachine" + ver
344:             user_action = "PythonFromUser" + ver
345:             exe_action = "PythonExe" + ver
346:             target_dir_prop = "TARGETDIR" + ver
347:             exe_prop = "PYTHON" + ver
348:             if msilib.Win64:
349:                 # type: msidbLocatorTypeRawValue + msidbLocatorType64bit
350:                 Type = 2+16
351:             else:
352:                 Type = 2
353:             add_data(self.db, "RegLocator",
354:                     [(machine_reg, 2, install_path, None, Type),
355:                      (user_reg, 1, install_path, None, Type)])
356:             add_data(self.db, "AppSearch",
357:                     [(machine_prop, machine_reg),
358:                      (user_prop, user_reg)])
359:             add_data(self.db, "CustomAction",
360:                     [(machine_action, 51+256, target_dir_prop, "[" + machine_prop + "]"),
361:                      (user_action, 51+256, target_dir_prop, "[" + user_prop + "]"),
362:                      (exe_action, 51+256, exe_prop, "[" + target_dir_prop + "]\\python.exe"),
363:                     ])
364:             add_data(self.db, "InstallExecuteSequence",
365:                     [(machine_action, machine_prop, start),
366:                      (user_action, user_prop, start + 1),
367:                      (exe_action, None, start + 2),
368:                     ])
369:             add_data(self.db, "InstallUISequence",
370:                     [(machine_action, machine_prop, start),
371:                      (user_action, user_prop, start + 1),
372:                      (exe_action, None, start + 2),
373:                     ])
374:             add_data(self.db, "Condition",
375:                     [("Python" + ver, 0, "NOT TARGETDIR" + ver)])
376:             start += 4
377:             assert start < 500
378: 
379:     def add_scripts(self):
380:         if self.install_script:
381:             start = 6800
382:             for ver in self.versions + [self.other_version]:
383:                 install_action = "install_script." + ver
384:                 exe_prop = "PYTHON" + ver
385:                 add_data(self.db, "CustomAction",
386:                         [(install_action, 50, exe_prop, self.install_script_key)])
387:                 add_data(self.db, "InstallExecuteSequence",
388:                         [(install_action, "&Python%s=3" % ver, start)])
389:                 start += 1
390:         # XXX pre-install scripts are currently refused in finalize_options()
391:         #     but if this feature is completed, it will also need to add
392:         #     entries for each version as the above code does
393:         if self.pre_install_script:
394:             scriptfn = os.path.join(self.bdist_dir, "preinstall.bat")
395:             f = open(scriptfn, "w")
396:             # The batch file will be executed with [PYTHON], so that %1
397:             # is the path to the Python interpreter; %0 will be the path
398:             # of the batch file.
399:             # rem ='''
400:             # %1 %0
401:             # exit
402:             # '''
403:             # <actual script>
404:             f.write('rem ='''\n%1 %0\nexit\n'''\n')
405:             f.write(open(self.pre_install_script).read())
406:             f.close()
407:             add_data(self.db, "Binary",
408:                 [("PreInstall", msilib.Binary(scriptfn))
409:                 ])
410:             add_data(self.db, "CustomAction",
411:                 [("PreInstall", 2, "PreInstall", None)
412:                 ])
413:             add_data(self.db, "InstallExecuteSequence",
414:                     [("PreInstall", "NOT Installed", 450)])
415: 
416: 
417:     def add_ui(self):
418:         db = self.db
419:         x = y = 50
420:         w = 370
421:         h = 300
422:         title = "[ProductName] Setup"
423: 
424:         # see "Dialog Style Bits"
425:         modal = 3      # visible | modal
426:         modeless = 1   # visible
427: 
428:         # UI customization properties
429:         add_data(db, "Property",
430:                  # See "DefaultUIFont Property"
431:                  [("DefaultUIFont", "DlgFont8"),
432:                   # See "ErrorDialog Style Bit"
433:                   ("ErrorDialog", "ErrorDlg"),
434:                   ("Progress1", "Install"),   # modified in maintenance type dlg
435:                   ("Progress2", "installs"),
436:                   ("MaintenanceForm_Action", "Repair"),
437:                   # possible values: ALL, JUSTME
438:                   ("WhichUsers", "ALL")
439:                  ])
440: 
441:         # Fonts, see "TextStyle Table"
442:         add_data(db, "TextStyle",
443:                  [("DlgFont8", "Tahoma", 9, None, 0),
444:                   ("DlgFontBold8", "Tahoma", 8, None, 1), #bold
445:                   ("VerdanaBold10", "Verdana", 10, None, 1),
446:                   ("VerdanaRed9", "Verdana", 9, 255, 0),
447:                  ])
448: 
449:         # UI Sequences, see "InstallUISequence Table", "Using a Sequence Table"
450:         # Numbers indicate sequence; see sequence.py for how these action integrate
451:         add_data(db, "InstallUISequence",
452:                  [("PrepareDlg", "Not Privileged or Windows9x or Installed", 140),
453:                   ("WhichUsersDlg", "Privileged and not Windows9x and not Installed", 141),
454:                   # In the user interface, assume all-users installation if privileged.
455:                   ("SelectFeaturesDlg", "Not Installed", 1230),
456:                   # XXX no support for resume installations yet
457:                   #("ResumeDlg", "Installed AND (RESUME OR Preselected)", 1240),
458:                   ("MaintenanceTypeDlg", "Installed AND NOT RESUME AND NOT Preselected", 1250),
459:                   ("ProgressDlg", None, 1280)])
460: 
461:         add_data(db, 'ActionText', text.ActionText)
462:         add_data(db, 'UIText', text.UIText)
463:         #####################################################################
464:         # Standard dialogs: FatalError, UserExit, ExitDialog
465:         fatal=PyDialog(db, "FatalError", x, y, w, h, modal, title,
466:                      "Finish", "Finish", "Finish")
467:         fatal.title("[ProductName] Installer ended prematurely")
468:         fatal.back("< Back", "Finish", active = 0)
469:         fatal.cancel("Cancel", "Back", active = 0)
470:         fatal.text("Description1", 15, 70, 320, 80, 0x30003,
471:                    "[ProductName] setup ended prematurely because of an error.  Your system has not been modified.  To install this program at a later time, please run the installation again.")
472:         fatal.text("Description2", 15, 155, 320, 20, 0x30003,
473:                    "Click the Finish button to exit the Installer.")
474:         c=fatal.next("Finish", "Cancel", name="Finish")
475:         c.event("EndDialog", "Exit")
476: 
477:         user_exit=PyDialog(db, "UserExit", x, y, w, h, modal, title,
478:                      "Finish", "Finish", "Finish")
479:         user_exit.title("[ProductName] Installer was interrupted")
480:         user_exit.back("< Back", "Finish", active = 0)
481:         user_exit.cancel("Cancel", "Back", active = 0)
482:         user_exit.text("Description1", 15, 70, 320, 80, 0x30003,
483:                    "[ProductName] setup was interrupted.  Your system has not been modified.  "
484:                    "To install this program at a later time, please run the installation again.")
485:         user_exit.text("Description2", 15, 155, 320, 20, 0x30003,
486:                    "Click the Finish button to exit the Installer.")
487:         c = user_exit.next("Finish", "Cancel", name="Finish")
488:         c.event("EndDialog", "Exit")
489: 
490:         exit_dialog = PyDialog(db, "ExitDialog", x, y, w, h, modal, title,
491:                              "Finish", "Finish", "Finish")
492:         exit_dialog.title("Completing the [ProductName] Installer")
493:         exit_dialog.back("< Back", "Finish", active = 0)
494:         exit_dialog.cancel("Cancel", "Back", active = 0)
495:         exit_dialog.text("Description", 15, 235, 320, 20, 0x30003,
496:                    "Click the Finish button to exit the Installer.")
497:         c = exit_dialog.next("Finish", "Cancel", name="Finish")
498:         c.event("EndDialog", "Return")
499: 
500:         #####################################################################
501:         # Required dialog: FilesInUse, ErrorDlg
502:         inuse = PyDialog(db, "FilesInUse",
503:                          x, y, w, h,
504:                          19,                # KeepModeless|Modal|Visible
505:                          title,
506:                          "Retry", "Retry", "Retry", bitmap=False)
507:         inuse.text("Title", 15, 6, 200, 15, 0x30003,
508:                    r"{\DlgFontBold8}Files in Use")
509:         inuse.text("Description", 20, 23, 280, 20, 0x30003,
510:                "Some files that need to be updated are currently in use.")
511:         inuse.text("Text", 20, 55, 330, 50, 3,
512:                    "The following applications are using files that need to be updated by this setup. Close these applications and then click Retry to continue the installation or Cancel to exit it.")
513:         inuse.control("List", "ListBox", 20, 107, 330, 130, 7, "FileInUseProcess",
514:                       None, None, None)
515:         c=inuse.back("Exit", "Ignore", name="Exit")
516:         c.event("EndDialog", "Exit")
517:         c=inuse.next("Ignore", "Retry", name="Ignore")
518:         c.event("EndDialog", "Ignore")
519:         c=inuse.cancel("Retry", "Exit", name="Retry")
520:         c.event("EndDialog","Retry")
521: 
522:         # See "Error Dialog". See "ICE20" for the required names of the controls.
523:         error = Dialog(db, "ErrorDlg",
524:                        50, 10, 330, 101,
525:                        65543,       # Error|Minimize|Modal|Visible
526:                        title,
527:                        "ErrorText", None, None)
528:         error.text("ErrorText", 50,9,280,48,3, "")
529:         #error.control("ErrorIcon", "Icon", 15, 9, 24, 24, 5242881, None, "py.ico", None, None)
530:         error.pushbutton("N",120,72,81,21,3,"No",None).event("EndDialog","ErrorNo")
531:         error.pushbutton("Y",240,72,81,21,3,"Yes",None).event("EndDialog","ErrorYes")
532:         error.pushbutton("A",0,72,81,21,3,"Abort",None).event("EndDialog","ErrorAbort")
533:         error.pushbutton("C",42,72,81,21,3,"Cancel",None).event("EndDialog","ErrorCancel")
534:         error.pushbutton("I",81,72,81,21,3,"Ignore",None).event("EndDialog","ErrorIgnore")
535:         error.pushbutton("O",159,72,81,21,3,"Ok",None).event("EndDialog","ErrorOk")
536:         error.pushbutton("R",198,72,81,21,3,"Retry",None).event("EndDialog","ErrorRetry")
537: 
538:         #####################################################################
539:         # Global "Query Cancel" dialog
540:         cancel = Dialog(db, "CancelDlg", 50, 10, 260, 85, 3, title,
541:                         "No", "No", "No")
542:         cancel.text("Text", 48, 15, 194, 30, 3,
543:                     "Are you sure you want to cancel [ProductName] installation?")
544:         #cancel.control("Icon", "Icon", 15, 15, 24, 24, 5242881, None,
545:         #               "py.ico", None, None)
546:         c=cancel.pushbutton("Yes", 72, 57, 56, 17, 3, "Yes", "No")
547:         c.event("EndDialog", "Exit")
548: 
549:         c=cancel.pushbutton("No", 132, 57, 56, 17, 3, "No", "Yes")
550:         c.event("EndDialog", "Return")
551: 
552:         #####################################################################
553:         # Global "Wait for costing" dialog
554:         costing = Dialog(db, "WaitForCostingDlg", 50, 10, 260, 85, modal, title,
555:                          "Return", "Return", "Return")
556:         costing.text("Text", 48, 15, 194, 30, 3,
557:                      "Please wait while the installer finishes determining your disk space requirements.")
558:         c = costing.pushbutton("Return", 102, 57, 56, 17, 3, "Return", None)
559:         c.event("EndDialog", "Exit")
560: 
561:         #####################################################################
562:         # Preparation dialog: no user input except cancellation
563:         prep = PyDialog(db, "PrepareDlg", x, y, w, h, modeless, title,
564:                         "Cancel", "Cancel", "Cancel")
565:         prep.text("Description", 15, 70, 320, 40, 0x30003,
566:                   "Please wait while the Installer prepares to guide you through the installation.")
567:         prep.title("Welcome to the [ProductName] Installer")
568:         c=prep.text("ActionText", 15, 110, 320, 20, 0x30003, "Pondering...")
569:         c.mapping("ActionText", "Text")
570:         c=prep.text("ActionData", 15, 135, 320, 30, 0x30003, None)
571:         c.mapping("ActionData", "Text")
572:         prep.back("Back", None, active=0)
573:         prep.next("Next", None, active=0)
574:         c=prep.cancel("Cancel", None)
575:         c.event("SpawnDialog", "CancelDlg")
576: 
577:         #####################################################################
578:         # Feature (Python directory) selection
579:         seldlg = PyDialog(db, "SelectFeaturesDlg", x, y, w, h, modal, title,
580:                         "Next", "Next", "Cancel")
581:         seldlg.title("Select Python Installations")
582: 
583:         seldlg.text("Hint", 15, 30, 300, 20, 3,
584:                     "Select the Python locations where %s should be installed."
585:                     % self.distribution.get_fullname())
586: 
587:         seldlg.back("< Back", None, active=0)
588:         c = seldlg.next("Next >", "Cancel")
589:         order = 1
590:         c.event("[TARGETDIR]", "[SourceDir]", ordering=order)
591:         for version in self.versions + [self.other_version]:
592:             order += 1
593:             c.event("[TARGETDIR]", "[TARGETDIR%s]" % version,
594:                     "FEATURE_SELECTED AND &Python%s=3" % version,
595:                     ordering=order)
596:         c.event("SpawnWaitDialog", "WaitForCostingDlg", ordering=order + 1)
597:         c.event("EndDialog", "Return", ordering=order + 2)
598:         c = seldlg.cancel("Cancel", "Features")
599:         c.event("SpawnDialog", "CancelDlg")
600: 
601:         c = seldlg.control("Features", "SelectionTree", 15, 60, 300, 120, 3,
602:                            "FEATURE", None, "PathEdit", None)
603:         c.event("[FEATURE_SELECTED]", "1")
604:         ver = self.other_version
605:         install_other_cond = "FEATURE_SELECTED AND &Python%s=3" % ver
606:         dont_install_other_cond = "FEATURE_SELECTED AND &Python%s<>3" % ver
607: 
608:         c = seldlg.text("Other", 15, 200, 300, 15, 3,
609:                         "Provide an alternate Python location")
610:         c.condition("Enable", install_other_cond)
611:         c.condition("Show", install_other_cond)
612:         c.condition("Disable", dont_install_other_cond)
613:         c.condition("Hide", dont_install_other_cond)
614: 
615:         c = seldlg.control("PathEdit", "PathEdit", 15, 215, 300, 16, 1,
616:                            "TARGETDIR" + ver, None, "Next", None)
617:         c.condition("Enable", install_other_cond)
618:         c.condition("Show", install_other_cond)
619:         c.condition("Disable", dont_install_other_cond)
620:         c.condition("Hide", dont_install_other_cond)
621: 
622:         #####################################################################
623:         # Disk cost
624:         cost = PyDialog(db, "DiskCostDlg", x, y, w, h, modal, title,
625:                         "OK", "OK", "OK", bitmap=False)
626:         cost.text("Title", 15, 6, 200, 15, 0x30003,
627:                   "{\DlgFontBold8}Disk Space Requirements")
628:         cost.text("Description", 20, 20, 280, 20, 0x30003,
629:                   "The disk space required for the installation of the selected features.")
630:         cost.text("Text", 20, 53, 330, 60, 3,
631:                   "The highlighted volumes (if any) do not have enough disk space "
632:               "available for the currently selected features.  You can either "
633:               "remove some files from the highlighted volumes, or choose to "
634:               "install less features onto local drive(s), or select different "
635:               "destination drive(s).")
636:         cost.control("VolumeList", "VolumeCostList", 20, 100, 330, 150, 393223,
637:                      None, "{120}{70}{70}{70}{70}", None, None)
638:         cost.xbutton("OK", "Ok", None, 0.5).event("EndDialog", "Return")
639: 
640:         #####################################################################
641:         # WhichUsers Dialog. Only available on NT, and for privileged users.
642:         # This must be run before FindRelatedProducts, because that will
643:         # take into account whether the previous installation was per-user
644:         # or per-machine. We currently don't support going back to this
645:         # dialog after "Next" was selected; to support this, we would need to
646:         # find how to reset the ALLUSERS property, and how to re-run
647:         # FindRelatedProducts.
648:         # On Windows9x, the ALLUSERS property is ignored on the command line
649:         # and in the Property table, but installer fails according to the documentation
650:         # if a dialog attempts to set ALLUSERS.
651:         whichusers = PyDialog(db, "WhichUsersDlg", x, y, w, h, modal, title,
652:                             "AdminInstall", "Next", "Cancel")
653:         whichusers.title("Select whether to install [ProductName] for all users of this computer.")
654:         # A radio group with two options: allusers, justme
655:         g = whichusers.radiogroup("AdminInstall", 15, 60, 260, 50, 3,
656:                                   "WhichUsers", "", "Next")
657:         g.add("ALL", 0, 5, 150, 20, "Install for all users")
658:         g.add("JUSTME", 0, 25, 150, 20, "Install just for me")
659: 
660:         whichusers.back("Back", None, active=0)
661: 
662:         c = whichusers.next("Next >", "Cancel")
663:         c.event("[ALLUSERS]", "1", 'WhichUsers="ALL"', 1)
664:         c.event("EndDialog", "Return", ordering = 2)
665: 
666:         c = whichusers.cancel("Cancel", "AdminInstall")
667:         c.event("SpawnDialog", "CancelDlg")
668: 
669:         #####################################################################
670:         # Installation Progress dialog (modeless)
671:         progress = PyDialog(db, "ProgressDlg", x, y, w, h, modeless, title,
672:                             "Cancel", "Cancel", "Cancel", bitmap=False)
673:         progress.text("Title", 20, 15, 200, 15, 0x30003,
674:                       "{\DlgFontBold8}[Progress1] [ProductName]")
675:         progress.text("Text", 35, 65, 300, 30, 3,
676:                       "Please wait while the Installer [Progress2] [ProductName]. "
677:                       "This may take several minutes.")
678:         progress.text("StatusLabel", 35, 100, 35, 20, 3, "Status:")
679: 
680:         c=progress.text("ActionText", 70, 100, w-70, 20, 3, "Pondering...")
681:         c.mapping("ActionText", "Text")
682: 
683:         #c=progress.text("ActionData", 35, 140, 300, 20, 3, None)
684:         #c.mapping("ActionData", "Text")
685: 
686:         c=progress.control("ProgressBar", "ProgressBar", 35, 120, 300, 10, 65537,
687:                            None, "Progress done", None, None)
688:         c.mapping("SetProgress", "Progress")
689: 
690:         progress.back("< Back", "Next", active=False)
691:         progress.next("Next >", "Cancel", active=False)
692:         progress.cancel("Cancel", "Back").event("SpawnDialog", "CancelDlg")
693: 
694:         ###################################################################
695:         # Maintenance type: repair/uninstall
696:         maint = PyDialog(db, "MaintenanceTypeDlg", x, y, w, h, modal, title,
697:                          "Next", "Next", "Cancel")
698:         maint.title("Welcome to the [ProductName] Setup Wizard")
699:         maint.text("BodyText", 15, 63, 330, 42, 3,
700:                    "Select whether you want to repair or remove [ProductName].")
701:         g=maint.radiogroup("RepairRadioGroup", 15, 108, 330, 60, 3,
702:                             "MaintenanceForm_Action", "", "Next")
703:         #g.add("Change", 0, 0, 200, 17, "&Change [ProductName]")
704:         g.add("Repair", 0, 18, 200, 17, "&Repair [ProductName]")
705:         g.add("Remove", 0, 36, 200, 17, "Re&move [ProductName]")
706: 
707:         maint.back("< Back", None, active=False)
708:         c=maint.next("Finish", "Cancel")
709:         # Change installation: Change progress dialog to "Change", then ask
710:         # for feature selection
711:         #c.event("[Progress1]", "Change", 'MaintenanceForm_Action="Change"', 1)
712:         #c.event("[Progress2]", "changes", 'MaintenanceForm_Action="Change"', 2)
713: 
714:         # Reinstall: Change progress dialog to "Repair", then invoke reinstall
715:         # Also set list of reinstalled features to "ALL"
716:         c.event("[REINSTALL]", "ALL", 'MaintenanceForm_Action="Repair"', 5)
717:         c.event("[Progress1]", "Repairing", 'MaintenanceForm_Action="Repair"', 6)
718:         c.event("[Progress2]", "repairs", 'MaintenanceForm_Action="Repair"', 7)
719:         c.event("Reinstall", "ALL", 'MaintenanceForm_Action="Repair"', 8)
720: 
721:         # Uninstall: Change progress to "Remove", then invoke uninstall
722:         # Also set list of removed features to "ALL"
723:         c.event("[REMOVE]", "ALL", 'MaintenanceForm_Action="Remove"', 11)
724:         c.event("[Progress1]", "Removing", 'MaintenanceForm_Action="Remove"', 12)
725:         c.event("[Progress2]", "removes", 'MaintenanceForm_Action="Remove"', 13)
726:         c.event("Remove", "ALL", 'MaintenanceForm_Action="Remove"', 14)
727: 
728:         # Close dialog when maintenance action scheduled
729:         c.event("EndDialog", "Return", 'MaintenanceForm_Action<>"Change"', 20)
730:         #c.event("NewDialog", "SelectFeaturesDlg", 'MaintenanceForm_Action="Change"', 21)
731: 
732:         maint.cancel("Cancel", "RepairRadioGroup").event("SpawnDialog", "CancelDlg")
733: 
734:     def get_installer_filename(self, fullname):
735:         # Factored out to allow overriding in subclasses
736:         if self.target_version:
737:             base_name = "%s.%s-py%s.msi" % (fullname, self.plat_name,
738:                                             self.target_version)
739:         else:
740:             base_name = "%s.%s.msi" % (fullname, self.plat_name)
741:         installer_name = os.path.join(self.dist_dir, base_name)
742:         return installer_name
743: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_12259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\nImplements the bdist_msi command.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# Multiple import statement. import sys (1/2) (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/2) (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from sysconfig import get_python_version' statement (line 10)
try:
    from sysconfig import get_python_version

except:
    get_python_version = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sysconfig', None, module_type_store, ['get_python_version'], [get_python_version])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.core import Command' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_12260 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.core')

if (type(import_12260) is not StypyTypeError):

    if (import_12260 != 'pyd_module'):
        __import__(import_12260)
        sys_modules_12261 = sys.modules[import_12260]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.core', sys_modules_12261.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_12261, sys_modules_12261.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.core', import_12260)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.dir_util import remove_tree' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_12262 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dir_util')

if (type(import_12262) is not StypyTypeError):

    if (import_12262 != 'pyd_module'):
        __import__(import_12262)
        sys_modules_12263 = sys.modules[import_12262]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dir_util', sys_modules_12263.module_type_store, module_type_store, ['remove_tree'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_12263, sys_modules_12263.module_type_store, module_type_store)
    else:
        from distutils.dir_util import remove_tree

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dir_util', None, module_type_store, ['remove_tree'], [remove_tree])

else:
    # Assigning a type to the variable 'distutils.dir_util' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dir_util', import_12262)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.version import StrictVersion' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_12264 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.version')

if (type(import_12264) is not StypyTypeError):

    if (import_12264 != 'pyd_module'):
        __import__(import_12264)
        sys_modules_12265 = sys.modules[import_12264]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.version', sys_modules_12265.module_type_store, module_type_store, ['StrictVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_12265, sys_modules_12265.module_type_store, module_type_store)
    else:
        from distutils.version import StrictVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.version', None, module_type_store, ['StrictVersion'], [StrictVersion])

else:
    # Assigning a type to the variable 'distutils.version' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.version', import_12264)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.errors import DistutilsOptionError' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_12266 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors')

if (type(import_12266) is not StypyTypeError):

    if (import_12266 != 'pyd_module'):
        __import__(import_12266)
        sys_modules_12267 = sys.modules[import_12266]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors', sys_modules_12267.module_type_store, module_type_store, ['DistutilsOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_12267, sys_modules_12267.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError'], [DistutilsOptionError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors', import_12266)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils import log' statement (line 16)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.util import get_platform' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_12268 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.util')

if (type(import_12268) is not StypyTypeError):

    if (import_12268 != 'pyd_module'):
        __import__(import_12268)
        sys_modules_12269 = sys.modules[import_12268]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.util', sys_modules_12269.module_type_store, module_type_store, ['get_platform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_12269, sys_modules_12269.module_type_store, module_type_store)
    else:
        from distutils.util import get_platform

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.util', None, module_type_store, ['get_platform'], [get_platform])

else:
    # Assigning a type to the variable 'distutils.util' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.util', import_12268)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import msilib' statement (line 19)
import msilib

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'msilib', msilib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from msilib import schema, sequence, text' statement (line 20)
try:
    from msilib import schema, sequence, text

except:
    schema = UndefinedType
    sequence = UndefinedType
    text = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'msilib', None, module_type_store, ['schema', 'sequence', 'text'], [schema, sequence, text])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from msilib import Directory, Feature, Dialog, add_data' statement (line 21)
try:
    from msilib import Directory, Feature, Dialog, add_data

except:
    Directory = UndefinedType
    Feature = UndefinedType
    Dialog = UndefinedType
    add_data = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'msilib', None, module_type_store, ['Directory', 'Feature', 'Dialog', 'add_data'], [Directory, Feature, Dialog, add_data])

# Declaration of the 'PyDialog' class
# Getting the type of 'Dialog' (line 23)
Dialog_12270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'Dialog')

class PyDialog(Dialog_12270, ):
    str_12271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', 'Dialog class with a fixed layout: controls at the top, then a ruler,\n    then a list of buttons: back, next, cancel. Optionally a bitmap at the\n    left.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyDialog.__init__', [], 'args', 'kw', defaults, varargs, kwargs)

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

        str_12272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', 'Dialog(database, name, x, y, w, h, attributes, title, first,\n        default, cancel, bitmap=true)')
        
        # Call to __init__(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'self' (line 30)
        self_12275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'self', False)
        # Getting the type of 'args' (line 30)
        args_12276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'args', False)
        # Processing the call keyword arguments (line 30)
        kwargs_12277 = {}
        # Getting the type of 'Dialog' (line 30)
        Dialog_12273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'Dialog', False)
        # Obtaining the member '__init__' of a type (line 30)
        init___12274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), Dialog_12273, '__init__')
        # Calling __init__(args, kwargs) (line 30)
        init___call_result_12278 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), init___12274, *[self_12275, args_12276], **kwargs_12277)
        
        
        # Assigning a BinOp to a Name (line 31):
        # Getting the type of 'self' (line 31)
        self_12279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'self')
        # Obtaining the member 'h' of a type (line 31)
        h_12280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), self_12279, 'h')
        int_12281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'int')
        # Applying the binary operator '-' (line 31)
        result_sub_12282 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 16), '-', h_12280, int_12281)
        
        # Assigning a type to the variable 'ruler' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'ruler', result_sub_12282)
        
        # Call to line(...): (line 34)
        # Processing the call arguments (line 34)
        str_12285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 18), 'str', 'BottomLine')
        int_12286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'int')
        # Getting the type of 'ruler' (line 34)
        ruler_12287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 35), 'ruler', False)
        # Getting the type of 'self' (line 34)
        self_12288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 42), 'self', False)
        # Obtaining the member 'w' of a type (line 34)
        w_12289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 42), self_12288, 'w')
        int_12290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 50), 'int')
        # Processing the call keyword arguments (line 34)
        kwargs_12291 = {}
        # Getting the type of 'self' (line 34)
        self_12283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member 'line' of a type (line 34)
        line_12284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_12283, 'line')
        # Calling line(args, kwargs) (line 34)
        line_call_result_12292 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), line_12284, *[str_12285, int_12286, ruler_12287, w_12289, int_12290], **kwargs_12291)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'title'
        module_type_store = module_type_store.open_function_context('title', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyDialog.title.__dict__.__setitem__('stypy_localization', localization)
        PyDialog.title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyDialog.title.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyDialog.title.__dict__.__setitem__('stypy_function_name', 'PyDialog.title')
        PyDialog.title.__dict__.__setitem__('stypy_param_names_list', ['title'])
        PyDialog.title.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyDialog.title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyDialog.title.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyDialog.title.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyDialog.title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyDialog.title.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyDialog.title', ['title'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'title', localization, ['title'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'title(...)' code ##################

        str_12293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'str', 'Set the title text of the dialog at the top.')
        
        # Call to text(...): (line 40)
        # Processing the call arguments (line 40)
        str_12296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'str', 'Title')
        int_12297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 27), 'int')
        int_12298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 31), 'int')
        int_12299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'int')
        int_12300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 40), 'int')
        int_12301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 44), 'int')
        str_12302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 18), 'str', '{\\VerdanaBold10}%s')
        # Getting the type of 'title' (line 41)
        title_12303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 42), 'title', False)
        # Applying the binary operator '%' (line 41)
        result_mod_12304 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 18), '%', str_12302, title_12303)
        
        # Processing the call keyword arguments (line 40)
        kwargs_12305 = {}
        # Getting the type of 'self' (line 40)
        self_12294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self', False)
        # Obtaining the member 'text' of a type (line 40)
        text_12295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_12294, 'text')
        # Calling text(args, kwargs) (line 40)
        text_call_result_12306 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), text_12295, *[str_12296, int_12297, int_12298, int_12299, int_12300, int_12301, result_mod_12304], **kwargs_12305)
        
        
        # ################# End of 'title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'title' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_12307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12307)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'title'
        return stypy_return_type_12307


    @norecursion
    def back(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_12308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'str', 'Back')
        int_12309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 56), 'int')
        defaults = [str_12308, int_12309]
        # Create a new context for function 'back'
        module_type_store = module_type_store.open_function_context('back', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyDialog.back.__dict__.__setitem__('stypy_localization', localization)
        PyDialog.back.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyDialog.back.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyDialog.back.__dict__.__setitem__('stypy_function_name', 'PyDialog.back')
        PyDialog.back.__dict__.__setitem__('stypy_param_names_list', ['title', 'next', 'name', 'active'])
        PyDialog.back.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyDialog.back.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyDialog.back.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyDialog.back.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyDialog.back.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyDialog.back.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyDialog.back', ['title', 'next', 'name', 'active'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'back', localization, ['title', 'next', 'name', 'active'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'back(...)' code ##################

        str_12310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, (-1)), 'str', 'Add a back button with a given title, the tab-next button,\n        its name in the Control table, possibly initially disabled.\n\n        Return the button, so that events can be associated')
        
        # Getting the type of 'active' (line 48)
        active_12311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'active')
        # Testing the type of an if condition (line 48)
        if_condition_12312 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), active_12311)
        # Assigning a type to the variable 'if_condition_12312' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'if_condition_12312', if_condition_12312)
        # SSA begins for if statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 49):
        int_12313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 20), 'int')
        # Assigning a type to the variable 'flags' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'flags', int_12313)
        # SSA branch for the else part of an if statement (line 48)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 51):
        int_12314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 20), 'int')
        # Assigning a type to the variable 'flags' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'flags', int_12314)
        # SSA join for if statement (line 48)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to pushbutton(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'name' (line 52)
        name_12317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'name', False)
        int_12318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 37), 'int')
        # Getting the type of 'self' (line 52)
        self_12319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 42), 'self', False)
        # Obtaining the member 'h' of a type (line 52)
        h_12320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 42), self_12319, 'h')
        int_12321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 49), 'int')
        # Applying the binary operator '-' (line 52)
        result_sub_12322 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 42), '-', h_12320, int_12321)
        
        int_12323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 54), 'int')
        int_12324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 58), 'int')
        # Getting the type of 'flags' (line 52)
        flags_12325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 62), 'flags', False)
        # Getting the type of 'title' (line 52)
        title_12326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 69), 'title', False)
        # Getting the type of 'next' (line 52)
        next_12327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 76), 'next', False)
        # Processing the call keyword arguments (line 52)
        kwargs_12328 = {}
        # Getting the type of 'self' (line 52)
        self_12315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'self', False)
        # Obtaining the member 'pushbutton' of a type (line 52)
        pushbutton_12316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 15), self_12315, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 52)
        pushbutton_call_result_12329 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), pushbutton_12316, *[name_12317, int_12318, result_sub_12322, int_12323, int_12324, flags_12325, title_12326, next_12327], **kwargs_12328)
        
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', pushbutton_call_result_12329)
        
        # ################# End of 'back(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'back' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_12330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12330)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'back'
        return stypy_return_type_12330


    @norecursion
    def cancel(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_12331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 41), 'str', 'Cancel')
        int_12332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 60), 'int')
        defaults = [str_12331, int_12332]
        # Create a new context for function 'cancel'
        module_type_store = module_type_store.open_function_context('cancel', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyDialog.cancel.__dict__.__setitem__('stypy_localization', localization)
        PyDialog.cancel.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyDialog.cancel.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyDialog.cancel.__dict__.__setitem__('stypy_function_name', 'PyDialog.cancel')
        PyDialog.cancel.__dict__.__setitem__('stypy_param_names_list', ['title', 'next', 'name', 'active'])
        PyDialog.cancel.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyDialog.cancel.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyDialog.cancel.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyDialog.cancel.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyDialog.cancel.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyDialog.cancel.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyDialog.cancel', ['title', 'next', 'name', 'active'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cancel', localization, ['title', 'next', 'name', 'active'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cancel(...)' code ##################

        str_12333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', 'Add a cancel button with a given title, the tab-next button,\n        its name in the Control table, possibly initially disabled.\n\n        Return the button, so that events can be associated')
        
        # Getting the type of 'active' (line 59)
        active_12334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'active')
        # Testing the type of an if condition (line 59)
        if_condition_12335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), active_12334)
        # Assigning a type to the variable 'if_condition_12335' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_12335', if_condition_12335)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 60):
        int_12336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'int')
        # Assigning a type to the variable 'flags' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'flags', int_12336)
        # SSA branch for the else part of an if statement (line 59)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 62):
        int_12337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'int')
        # Assigning a type to the variable 'flags' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'flags', int_12337)
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to pushbutton(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'name' (line 63)
        name_12340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'name', False)
        int_12341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 37), 'int')
        # Getting the type of 'self' (line 63)
        self_12342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 42), 'self', False)
        # Obtaining the member 'h' of a type (line 63)
        h_12343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 42), self_12342, 'h')
        int_12344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 49), 'int')
        # Applying the binary operator '-' (line 63)
        result_sub_12345 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 42), '-', h_12343, int_12344)
        
        int_12346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 53), 'int')
        int_12347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 57), 'int')
        # Getting the type of 'flags' (line 63)
        flags_12348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 61), 'flags', False)
        # Getting the type of 'title' (line 63)
        title_12349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 68), 'title', False)
        # Getting the type of 'next' (line 63)
        next_12350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 75), 'next', False)
        # Processing the call keyword arguments (line 63)
        kwargs_12351 = {}
        # Getting the type of 'self' (line 63)
        self_12338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'self', False)
        # Obtaining the member 'pushbutton' of a type (line 63)
        pushbutton_12339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 15), self_12338, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 63)
        pushbutton_call_result_12352 = invoke(stypy.reporting.localization.Localization(__file__, 63, 15), pushbutton_12339, *[name_12340, int_12341, result_sub_12345, int_12346, int_12347, flags_12348, title_12349, next_12350], **kwargs_12351)
        
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', pushbutton_call_result_12352)
        
        # ################# End of 'cancel(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cancel' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_12353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cancel'
        return stypy_return_type_12353


    @norecursion
    def next(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_12354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 39), 'str', 'Next')
        int_12355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 56), 'int')
        defaults = [str_12354, int_12355]
        # Create a new context for function 'next'
        module_type_store = module_type_store.open_function_context('next', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyDialog.next.__dict__.__setitem__('stypy_localization', localization)
        PyDialog.next.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyDialog.next.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyDialog.next.__dict__.__setitem__('stypy_function_name', 'PyDialog.next')
        PyDialog.next.__dict__.__setitem__('stypy_param_names_list', ['title', 'next', 'name', 'active'])
        PyDialog.next.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyDialog.next.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyDialog.next.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyDialog.next.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyDialog.next.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyDialog.next.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyDialog.next', ['title', 'next', 'name', 'active'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'next', localization, ['title', 'next', 'name', 'active'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'next(...)' code ##################

        str_12356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'str', 'Add a Next button with a given title, the tab-next button,\n        its name in the Control table, possibly initially disabled.\n\n        Return the button, so that events can be associated')
        
        # Getting the type of 'active' (line 70)
        active_12357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'active')
        # Testing the type of an if condition (line 70)
        if_condition_12358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 8), active_12357)
        # Assigning a type to the variable 'if_condition_12358' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'if_condition_12358', if_condition_12358)
        # SSA begins for if statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 71):
        int_12359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'int')
        # Assigning a type to the variable 'flags' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'flags', int_12359)
        # SSA branch for the else part of an if statement (line 70)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 73):
        int_12360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'int')
        # Assigning a type to the variable 'flags' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'flags', int_12360)
        # SSA join for if statement (line 70)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to pushbutton(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'name' (line 74)
        name_12363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 31), 'name', False)
        int_12364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 37), 'int')
        # Getting the type of 'self' (line 74)
        self_12365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 42), 'self', False)
        # Obtaining the member 'h' of a type (line 74)
        h_12366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 42), self_12365, 'h')
        int_12367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 49), 'int')
        # Applying the binary operator '-' (line 74)
        result_sub_12368 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 42), '-', h_12366, int_12367)
        
        int_12369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 53), 'int')
        int_12370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 57), 'int')
        # Getting the type of 'flags' (line 74)
        flags_12371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 61), 'flags', False)
        # Getting the type of 'title' (line 74)
        title_12372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 68), 'title', False)
        # Getting the type of 'next' (line 74)
        next_12373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 75), 'next', False)
        # Processing the call keyword arguments (line 74)
        kwargs_12374 = {}
        # Getting the type of 'self' (line 74)
        self_12361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'self', False)
        # Obtaining the member 'pushbutton' of a type (line 74)
        pushbutton_12362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 15), self_12361, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 74)
        pushbutton_call_result_12375 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), pushbutton_12362, *[name_12363, int_12364, result_sub_12368, int_12369, int_12370, flags_12371, title_12372, next_12373], **kwargs_12374)
        
        # Assigning a type to the variable 'stypy_return_type' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'stypy_return_type', pushbutton_call_result_12375)
        
        # ################# End of 'next(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'next' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_12376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'next'
        return stypy_return_type_12376


    @norecursion
    def xbutton(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'xbutton'
        module_type_store = module_type_store.open_function_context('xbutton', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyDialog.xbutton.__dict__.__setitem__('stypy_localization', localization)
        PyDialog.xbutton.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyDialog.xbutton.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyDialog.xbutton.__dict__.__setitem__('stypy_function_name', 'PyDialog.xbutton')
        PyDialog.xbutton.__dict__.__setitem__('stypy_param_names_list', ['name', 'title', 'next', 'xpos'])
        PyDialog.xbutton.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyDialog.xbutton.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyDialog.xbutton.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyDialog.xbutton.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyDialog.xbutton.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyDialog.xbutton.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyDialog.xbutton', ['name', 'title', 'next', 'xpos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'xbutton', localization, ['name', 'title', 'next', 'xpos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'xbutton(...)' code ##################

        str_12377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', 'Add a button with a given title, the tab-next button,\n        its name in the Control table, giving its x position; the\n        y-position is aligned with the other buttons.\n\n        Return the button, so that events can be associated')
        
        # Call to pushbutton(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'name' (line 82)
        name_12380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 31), 'name', False)
        
        # Call to int(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'self' (line 82)
        self_12382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'self', False)
        # Obtaining the member 'w' of a type (line 82)
        w_12383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 41), self_12382, 'w')
        # Getting the type of 'xpos' (line 82)
        xpos_12384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 48), 'xpos', False)
        # Applying the binary operator '*' (line 82)
        result_mul_12385 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 41), '*', w_12383, xpos_12384)
        
        int_12386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 55), 'int')
        # Applying the binary operator '-' (line 82)
        result_sub_12387 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 41), '-', result_mul_12385, int_12386)
        
        # Processing the call keyword arguments (line 82)
        kwargs_12388 = {}
        # Getting the type of 'int' (line 82)
        int_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'int', False)
        # Calling int(args, kwargs) (line 82)
        int_call_result_12389 = invoke(stypy.reporting.localization.Localization(__file__, 82, 37), int_12381, *[result_sub_12387], **kwargs_12388)
        
        # Getting the type of 'self' (line 82)
        self_12390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 60), 'self', False)
        # Obtaining the member 'h' of a type (line 82)
        h_12391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 60), self_12390, 'h')
        int_12392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 67), 'int')
        # Applying the binary operator '-' (line 82)
        result_sub_12393 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 60), '-', h_12391, int_12392)
        
        int_12394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 71), 'int')
        int_12395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 75), 'int')
        int_12396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 79), 'int')
        # Getting the type of 'title' (line 82)
        title_12397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 82), 'title', False)
        # Getting the type of 'next' (line 82)
        next_12398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 89), 'next', False)
        # Processing the call keyword arguments (line 82)
        kwargs_12399 = {}
        # Getting the type of 'self' (line 82)
        self_12378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'self', False)
        # Obtaining the member 'pushbutton' of a type (line 82)
        pushbutton_12379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), self_12378, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 82)
        pushbutton_call_result_12400 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), pushbutton_12379, *[name_12380, int_call_result_12389, result_sub_12393, int_12394, int_12395, int_12396, title_12397, next_12398], **kwargs_12399)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', pushbutton_call_result_12400)
        
        # ################# End of 'xbutton(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'xbutton' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_12401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12401)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'xbutton'
        return stypy_return_type_12401


# Assigning a type to the variable 'PyDialog' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'PyDialog', PyDialog)
# Declaration of the 'bdist_msi' class
# Getting the type of 'Command' (line 84)
Command_12402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'Command')

class bdist_msi(Command_12402, ):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_function_name', 'bdist_msi.initialize_options')
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_msi.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_msi.initialize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize_options(...)' code ##################

        
        # Assigning a Name to a Attribute (line 127):
        # Getting the type of 'None' (line 127)
        None_12403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 25), 'None')
        # Getting the type of 'self' (line 127)
        self_12404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self')
        # Setting the type of the member 'bdist_dir' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_12404, 'bdist_dir', None_12403)
        
        # Assigning a Name to a Attribute (line 128):
        # Getting the type of 'None' (line 128)
        None_12405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'None')
        # Getting the type of 'self' (line 128)
        self_12406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member 'plat_name' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_12406, 'plat_name', None_12405)
        
        # Assigning a Num to a Attribute (line 129):
        int_12407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 25), 'int')
        # Getting the type of 'self' (line 129)
        self_12408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member 'keep_temp' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_12408, 'keep_temp', int_12407)
        
        # Assigning a Num to a Attribute (line 130):
        int_12409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 33), 'int')
        # Getting the type of 'self' (line 130)
        self_12410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self')
        # Setting the type of the member 'no_target_compile' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_12410, 'no_target_compile', int_12409)
        
        # Assigning a Num to a Attribute (line 131):
        int_12411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 34), 'int')
        # Getting the type of 'self' (line 131)
        self_12412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self')
        # Setting the type of the member 'no_target_optimize' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_12412, 'no_target_optimize', int_12411)
        
        # Assigning a Name to a Attribute (line 132):
        # Getting the type of 'None' (line 132)
        None_12413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 30), 'None')
        # Getting the type of 'self' (line 132)
        self_12414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self')
        # Setting the type of the member 'target_version' of a type (line 132)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_12414, 'target_version', None_12413)
        
        # Assigning a Name to a Attribute (line 133):
        # Getting the type of 'None' (line 133)
        None_12415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'None')
        # Getting the type of 'self' (line 133)
        self_12416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'self')
        # Setting the type of the member 'dist_dir' of a type (line 133)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), self_12416, 'dist_dir', None_12415)
        
        # Assigning a Name to a Attribute (line 134):
        # Getting the type of 'None' (line 134)
        None_12417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'None')
        # Getting the type of 'self' (line 134)
        self_12418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self')
        # Setting the type of the member 'skip_build' of a type (line 134)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_12418, 'skip_build', None_12417)
        
        # Assigning a Name to a Attribute (line 135):
        # Getting the type of 'None' (line 135)
        None_12419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 30), 'None')
        # Getting the type of 'self' (line 135)
        self_12420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'self')
        # Setting the type of the member 'install_script' of a type (line 135)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), self_12420, 'install_script', None_12419)
        
        # Assigning a Name to a Attribute (line 136):
        # Getting the type of 'None' (line 136)
        None_12421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 34), 'None')
        # Getting the type of 'self' (line 136)
        self_12422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'self')
        # Setting the type of the member 'pre_install_script' of a type (line 136)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), self_12422, 'pre_install_script', None_12421)
        
        # Assigning a Name to a Attribute (line 137):
        # Getting the type of 'None' (line 137)
        None_12423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'None')
        # Getting the type of 'self' (line 137)
        self_12424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self')
        # Setting the type of the member 'versions' of a type (line 137)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_12424, 'versions', None_12423)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_12425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12425)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_12425


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_function_name', 'bdist_msi.finalize_options')
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_msi.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_msi.finalize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'finalize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'finalize_options(...)' code ##################

        
        # Call to set_undefined_options(...): (line 140)
        # Processing the call arguments (line 140)
        str_12428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 35), 'str', 'bdist')
        
        # Obtaining an instance of the builtin type 'tuple' (line 140)
        tuple_12429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 140)
        # Adding element type (line 140)
        str_12430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 45), 'str', 'skip_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 45), tuple_12429, str_12430)
        # Adding element type (line 140)
        str_12431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 59), 'str', 'skip_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 45), tuple_12429, str_12431)
        
        # Processing the call keyword arguments (line 140)
        kwargs_12432 = {}
        # Getting the type of 'self' (line 140)
        self_12426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 140)
        set_undefined_options_12427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_12426, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 140)
        set_undefined_options_call_result_12433 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), set_undefined_options_12427, *[str_12428, tuple_12429], **kwargs_12432)
        
        
        # Type idiom detected: calculating its left and rigth part (line 142)
        # Getting the type of 'self' (line 142)
        self_12434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'self')
        # Obtaining the member 'bdist_dir' of a type (line 142)
        bdist_dir_12435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), self_12434, 'bdist_dir')
        # Getting the type of 'None' (line 142)
        None_12436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'None')
        
        (may_be_12437, more_types_in_union_12438) = may_be_none(bdist_dir_12435, None_12436)

        if may_be_12437:

            if more_types_in_union_12438:
                # Runtime conditional SSA (line 142)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 143):
            
            # Call to get_finalized_command(...): (line 143)
            # Processing the call arguments (line 143)
            str_12441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 52), 'str', 'bdist')
            # Processing the call keyword arguments (line 143)
            kwargs_12442 = {}
            # Getting the type of 'self' (line 143)
            self_12439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 25), 'self', False)
            # Obtaining the member 'get_finalized_command' of a type (line 143)
            get_finalized_command_12440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 25), self_12439, 'get_finalized_command')
            # Calling get_finalized_command(args, kwargs) (line 143)
            get_finalized_command_call_result_12443 = invoke(stypy.reporting.localization.Localization(__file__, 143, 25), get_finalized_command_12440, *[str_12441], **kwargs_12442)
            
            # Obtaining the member 'bdist_base' of a type (line 143)
            bdist_base_12444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 25), get_finalized_command_call_result_12443, 'bdist_base')
            # Assigning a type to the variable 'bdist_base' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'bdist_base', bdist_base_12444)
            
            # Assigning a Call to a Attribute (line 144):
            
            # Call to join(...): (line 144)
            # Processing the call arguments (line 144)
            # Getting the type of 'bdist_base' (line 144)
            bdist_base_12448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 42), 'bdist_base', False)
            str_12449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 54), 'str', 'msi')
            # Processing the call keyword arguments (line 144)
            kwargs_12450 = {}
            # Getting the type of 'os' (line 144)
            os_12445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 29), 'os', False)
            # Obtaining the member 'path' of a type (line 144)
            path_12446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 29), os_12445, 'path')
            # Obtaining the member 'join' of a type (line 144)
            join_12447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 29), path_12446, 'join')
            # Calling join(args, kwargs) (line 144)
            join_call_result_12451 = invoke(stypy.reporting.localization.Localization(__file__, 144, 29), join_12447, *[bdist_base_12448, str_12449], **kwargs_12450)
            
            # Getting the type of 'self' (line 144)
            self_12452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'self')
            # Setting the type of the member 'bdist_dir' of a type (line 144)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), self_12452, 'bdist_dir', join_call_result_12451)

            if more_types_in_union_12438:
                # SSA join for if statement (line 142)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 146):
        
        # Call to get_python_version(...): (line 146)
        # Processing the call keyword arguments (line 146)
        kwargs_12454 = {}
        # Getting the type of 'get_python_version' (line 146)
        get_python_version_12453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'get_python_version', False)
        # Calling get_python_version(args, kwargs) (line 146)
        get_python_version_call_result_12455 = invoke(stypy.reporting.localization.Localization(__file__, 146, 24), get_python_version_12453, *[], **kwargs_12454)
        
        # Assigning a type to the variable 'short_version' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'short_version', get_python_version_call_result_12455)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 147)
        self_12456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'self')
        # Obtaining the member 'target_version' of a type (line 147)
        target_version_12457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 16), self_12456, 'target_version')
        # Applying the 'not' unary operator (line 147)
        result_not__12458 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 12), 'not', target_version_12457)
        
        
        # Call to has_ext_modules(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_12462 = {}
        # Getting the type of 'self' (line 147)
        self_12459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 41), 'self', False)
        # Obtaining the member 'distribution' of a type (line 147)
        distribution_12460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 41), self_12459, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 147)
        has_ext_modules_12461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 41), distribution_12460, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 147)
        has_ext_modules_call_result_12463 = invoke(stypy.reporting.localization.Localization(__file__, 147, 41), has_ext_modules_12461, *[], **kwargs_12462)
        
        # Applying the binary operator 'and' (line 147)
        result_and_keyword_12464 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 11), 'and', result_not__12458, has_ext_modules_call_result_12463)
        
        # Testing the type of an if condition (line 147)
        if_condition_12465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 8), result_and_keyword_12464)
        # Assigning a type to the variable 'if_condition_12465' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'if_condition_12465', if_condition_12465)
        # SSA begins for if statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 148):
        # Getting the type of 'short_version' (line 148)
        short_version_12466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 34), 'short_version')
        # Getting the type of 'self' (line 148)
        self_12467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'self')
        # Setting the type of the member 'target_version' of a type (line 148)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), self_12467, 'target_version', short_version_12466)
        # SSA join for if statement (line 147)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 150)
        self_12468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'self')
        # Obtaining the member 'target_version' of a type (line 150)
        target_version_12469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), self_12468, 'target_version')
        # Testing the type of an if condition (line 150)
        if_condition_12470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), target_version_12469)
        # Assigning a type to the variable 'if_condition_12470' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_12470', if_condition_12470)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_12471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        # Getting the type of 'self' (line 151)
        self_12472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'self')
        # Obtaining the member 'target_version' of a type (line 151)
        target_version_12473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 29), self_12472, 'target_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 28), list_12471, target_version_12473)
        
        # Getting the type of 'self' (line 151)
        self_12474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'self')
        # Setting the type of the member 'versions' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), self_12474, 'versions', list_12471)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 152)
        self_12475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'self')
        # Obtaining the member 'skip_build' of a type (line 152)
        skip_build_12476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 19), self_12475, 'skip_build')
        # Applying the 'not' unary operator (line 152)
        result_not__12477 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 15), 'not', skip_build_12476)
        
        
        # Call to has_ext_modules(...): (line 152)
        # Processing the call keyword arguments (line 152)
        kwargs_12481 = {}
        # Getting the type of 'self' (line 152)
        self_12478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 39), 'self', False)
        # Obtaining the member 'distribution' of a type (line 152)
        distribution_12479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 39), self_12478, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 152)
        has_ext_modules_12480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 39), distribution_12479, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 152)
        has_ext_modules_call_result_12482 = invoke(stypy.reporting.localization.Localization(__file__, 152, 39), has_ext_modules_12480, *[], **kwargs_12481)
        
        # Applying the binary operator 'and' (line 152)
        result_and_keyword_12483 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 15), 'and', result_not__12477, has_ext_modules_call_result_12482)
        
        # Getting the type of 'self' (line 153)
        self_12484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'self')
        # Obtaining the member 'target_version' of a type (line 153)
        target_version_12485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 19), self_12484, 'target_version')
        # Getting the type of 'short_version' (line 153)
        short_version_12486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 42), 'short_version')
        # Applying the binary operator '!=' (line 153)
        result_ne_12487 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 19), '!=', target_version_12485, short_version_12486)
        
        # Applying the binary operator 'and' (line 152)
        result_and_keyword_12488 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 15), 'and', result_and_keyword_12483, result_ne_12487)
        
        # Testing the type of an if condition (line 152)
        if_condition_12489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 12), result_and_keyword_12488)
        # Assigning a type to the variable 'if_condition_12489' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'if_condition_12489', if_condition_12489)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsOptionError' (line 154)
        DistutilsOptionError_12490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 154, 16), DistutilsOptionError_12490, 'raise parameter', BaseException)
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 150)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 158):
        
        # Call to list(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'self' (line 158)
        self_12492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'self', False)
        # Obtaining the member 'all_versions' of a type (line 158)
        all_versions_12493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 33), self_12492, 'all_versions')
        # Processing the call keyword arguments (line 158)
        kwargs_12494 = {}
        # Getting the type of 'list' (line 158)
        list_12491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'list', False)
        # Calling list(args, kwargs) (line 158)
        list_call_result_12495 = invoke(stypy.reporting.localization.Localization(__file__, 158, 28), list_12491, *[all_versions_12493], **kwargs_12494)
        
        # Getting the type of 'self' (line 158)
        self_12496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'self')
        # Setting the type of the member 'versions' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), self_12496, 'versions', list_call_result_12495)
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_undefined_options(...): (line 160)
        # Processing the call arguments (line 160)
        str_12499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 35), 'str', 'bdist')
        
        # Obtaining an instance of the builtin type 'tuple' (line 161)
        tuple_12500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 161)
        # Adding element type (line 161)
        str_12501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 36), 'str', 'dist_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 36), tuple_12500, str_12501)
        # Adding element type (line 161)
        str_12502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 48), 'str', 'dist_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 36), tuple_12500, str_12502)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 162)
        tuple_12503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 162)
        # Adding element type (line 162)
        str_12504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 36), 'str', 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 36), tuple_12503, str_12504)
        # Adding element type (line 162)
        str_12505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 49), 'str', 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 36), tuple_12503, str_12505)
        
        # Processing the call keyword arguments (line 160)
        kwargs_12506 = {}
        # Getting the type of 'self' (line 160)
        self_12497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 160)
        set_undefined_options_12498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_12497, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 160)
        set_undefined_options_call_result_12507 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), set_undefined_options_12498, *[str_12499, tuple_12500, tuple_12503], **kwargs_12506)
        
        
        # Getting the type of 'self' (line 165)
        self_12508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'self')
        # Obtaining the member 'pre_install_script' of a type (line 165)
        pre_install_script_12509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 11), self_12508, 'pre_install_script')
        # Testing the type of an if condition (line 165)
        if_condition_12510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 8), pre_install_script_12509)
        # Assigning a type to the variable 'if_condition_12510' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'if_condition_12510', if_condition_12510)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsOptionError' (line 166)
        DistutilsOptionError_12511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 18), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 166, 12), DistutilsOptionError_12511, 'raise parameter', BaseException)
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 168)
        self_12512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'self')
        # Obtaining the member 'install_script' of a type (line 168)
        install_script_12513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 11), self_12512, 'install_script')
        # Testing the type of an if condition (line 168)
        if_condition_12514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 8), install_script_12513)
        # Assigning a type to the variable 'if_condition_12514' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'if_condition_12514', if_condition_12514)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 169)
        self_12515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'self')
        # Obtaining the member 'distribution' of a type (line 169)
        distribution_12516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 26), self_12515, 'distribution')
        # Obtaining the member 'scripts' of a type (line 169)
        scripts_12517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 26), distribution_12516, 'scripts')
        # Testing the type of a for loop iterable (line 169)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 169, 12), scripts_12517)
        # Getting the type of the for loop variable (line 169)
        for_loop_var_12518 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 169, 12), scripts_12517)
        # Assigning a type to the variable 'script' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'script', for_loop_var_12518)
        # SSA begins for a for statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'self' (line 170)
        self_12519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'self')
        # Obtaining the member 'install_script' of a type (line 170)
        install_script_12520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 19), self_12519, 'install_script')
        
        # Call to basename(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'script' (line 170)
        script_12524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 59), 'script', False)
        # Processing the call keyword arguments (line 170)
        kwargs_12525 = {}
        # Getting the type of 'os' (line 170)
        os_12521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 42), 'os', False)
        # Obtaining the member 'path' of a type (line 170)
        path_12522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 42), os_12521, 'path')
        # Obtaining the member 'basename' of a type (line 170)
        basename_12523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 42), path_12522, 'basename')
        # Calling basename(args, kwargs) (line 170)
        basename_call_result_12526 = invoke(stypy.reporting.localization.Localization(__file__, 170, 42), basename_12523, *[script_12524], **kwargs_12525)
        
        # Applying the binary operator '==' (line 170)
        result_eq_12527 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 19), '==', install_script_12520, basename_call_result_12526)
        
        # Testing the type of an if condition (line 170)
        if_condition_12528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 16), result_eq_12527)
        # Assigning a type to the variable 'if_condition_12528' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'if_condition_12528', if_condition_12528)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 169)
        module_type_store.open_ssa_branch('for loop else')
        # Getting the type of 'DistutilsOptionError' (line 173)
        DistutilsOptionError_12529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 173, 16), DistutilsOptionError_12529, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 176):
        # Getting the type of 'None' (line 176)
        None_12530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 34), 'None')
        # Getting the type of 'self' (line 176)
        self_12531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self')
        # Setting the type of the member 'install_script_key' of a type (line 176)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_12531, 'install_script_key', None_12530)
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_12532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12532)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_12532


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_msi.run.__dict__.__setitem__('stypy_localization', localization)
        bdist_msi.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_msi.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_msi.run.__dict__.__setitem__('stypy_function_name', 'bdist_msi.run')
        bdist_msi.run.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_msi.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_msi.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_msi.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_msi.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_msi.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_msi.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_msi.run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        
        # Getting the type of 'self' (line 181)
        self_12533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'self')
        # Obtaining the member 'skip_build' of a type (line 181)
        skip_build_12534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 15), self_12533, 'skip_build')
        # Applying the 'not' unary operator (line 181)
        result_not__12535 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), 'not', skip_build_12534)
        
        # Testing the type of an if condition (line 181)
        if_condition_12536 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 8), result_not__12535)
        # Assigning a type to the variable 'if_condition_12536' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'if_condition_12536', if_condition_12536)
        # SSA begins for if statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to run_command(...): (line 182)
        # Processing the call arguments (line 182)
        str_12539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 29), 'str', 'build')
        # Processing the call keyword arguments (line 182)
        kwargs_12540 = {}
        # Getting the type of 'self' (line 182)
        self_12537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 182)
        run_command_12538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), self_12537, 'run_command')
        # Calling run_command(args, kwargs) (line 182)
        run_command_call_result_12541 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), run_command_12538, *[str_12539], **kwargs_12540)
        
        # SSA join for if statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 184):
        
        # Call to reinitialize_command(...): (line 184)
        # Processing the call arguments (line 184)
        str_12544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 44), 'str', 'install')
        # Processing the call keyword arguments (line 184)
        int_12545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 74), 'int')
        keyword_12546 = int_12545
        kwargs_12547 = {'reinit_subcommands': keyword_12546}
        # Getting the type of 'self' (line 184)
        self_12542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'self', False)
        # Obtaining the member 'reinitialize_command' of a type (line 184)
        reinitialize_command_12543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 18), self_12542, 'reinitialize_command')
        # Calling reinitialize_command(args, kwargs) (line 184)
        reinitialize_command_call_result_12548 = invoke(stypy.reporting.localization.Localization(__file__, 184, 18), reinitialize_command_12543, *[str_12544], **kwargs_12547)
        
        # Assigning a type to the variable 'install' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'install', reinitialize_command_call_result_12548)
        
        # Assigning a Attribute to a Attribute (line 185):
        # Getting the type of 'self' (line 185)
        self_12549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 'self')
        # Obtaining the member 'bdist_dir' of a type (line 185)
        bdist_dir_12550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 25), self_12549, 'bdist_dir')
        # Getting the type of 'install' (line 185)
        install_12551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'install')
        # Setting the type of the member 'prefix' of a type (line 185)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), install_12551, 'prefix', bdist_dir_12550)
        
        # Assigning a Attribute to a Attribute (line 186):
        # Getting the type of 'self' (line 186)
        self_12552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 29), 'self')
        # Obtaining the member 'skip_build' of a type (line 186)
        skip_build_12553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 29), self_12552, 'skip_build')
        # Getting the type of 'install' (line 186)
        install_12554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'install')
        # Setting the type of the member 'skip_build' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), install_12554, 'skip_build', skip_build_12553)
        
        # Assigning a Num to a Attribute (line 187):
        int_12555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 27), 'int')
        # Getting the type of 'install' (line 187)
        install_12556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'install')
        # Setting the type of the member 'warn_dir' of a type (line 187)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), install_12556, 'warn_dir', int_12555)
        
        # Assigning a Call to a Name (line 189):
        
        # Call to reinitialize_command(...): (line 189)
        # Processing the call arguments (line 189)
        str_12559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 48), 'str', 'install_lib')
        # Processing the call keyword arguments (line 189)
        kwargs_12560 = {}
        # Getting the type of 'self' (line 189)
        self_12557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 22), 'self', False)
        # Obtaining the member 'reinitialize_command' of a type (line 189)
        reinitialize_command_12558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 22), self_12557, 'reinitialize_command')
        # Calling reinitialize_command(args, kwargs) (line 189)
        reinitialize_command_call_result_12561 = invoke(stypy.reporting.localization.Localization(__file__, 189, 22), reinitialize_command_12558, *[str_12559], **kwargs_12560)
        
        # Assigning a type to the variable 'install_lib' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'install_lib', reinitialize_command_call_result_12561)
        
        # Assigning a Num to a Attribute (line 191):
        int_12562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 30), 'int')
        # Getting the type of 'install_lib' (line 191)
        install_lib_12563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'install_lib')
        # Setting the type of the member 'compile' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), install_lib_12563, 'compile', int_12562)
        
        # Assigning a Num to a Attribute (line 192):
        int_12564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 31), 'int')
        # Getting the type of 'install_lib' (line 192)
        install_lib_12565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'install_lib')
        # Setting the type of the member 'optimize' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), install_lib_12565, 'optimize', int_12564)
        
        
        # Call to has_ext_modules(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_12569 = {}
        # Getting the type of 'self' (line 194)
        self_12566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 194)
        distribution_12567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 11), self_12566, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 194)
        has_ext_modules_12568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 11), distribution_12567, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 194)
        has_ext_modules_call_result_12570 = invoke(stypy.reporting.localization.Localization(__file__, 194, 11), has_ext_modules_12568, *[], **kwargs_12569)
        
        # Testing the type of an if condition (line 194)
        if_condition_12571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), has_ext_modules_call_result_12570)
        # Assigning a type to the variable 'if_condition_12571' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_12571', if_condition_12571)
        # SSA begins for if statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 201):
        # Getting the type of 'self' (line 201)
        self_12572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 29), 'self')
        # Obtaining the member 'target_version' of a type (line 201)
        target_version_12573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 29), self_12572, 'target_version')
        # Assigning a type to the variable 'target_version' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'target_version', target_version_12573)
        
        
        # Getting the type of 'target_version' (line 202)
        target_version_12574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'target_version')
        # Applying the 'not' unary operator (line 202)
        result_not__12575 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 15), 'not', target_version_12574)
        
        # Testing the type of an if condition (line 202)
        if_condition_12576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 12), result_not__12575)
        # Assigning a type to the variable 'if_condition_12576' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'if_condition_12576', if_condition_12576)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Evaluating assert statement condition
        # Getting the type of 'self' (line 203)
        self_12577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'self')
        # Obtaining the member 'skip_build' of a type (line 203)
        skip_build_12578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 23), self_12577, 'skip_build')
        
        # Assigning a Subscript to a Name (line 204):
        
        # Obtaining the type of the subscript
        int_12579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 45), 'int')
        int_12580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 47), 'int')
        slice_12581 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 204, 33), int_12579, int_12580, None)
        # Getting the type of 'sys' (line 204)
        sys_12582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'sys')
        # Obtaining the member 'version' of a type (line 204)
        version_12583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 33), sys_12582, 'version')
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___12584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 33), version_12583, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_12585 = invoke(stypy.reporting.localization.Localization(__file__, 204, 33), getitem___12584, slice_12581)
        
        # Assigning a type to the variable 'target_version' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'target_version', subscript_call_result_12585)
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 205):
        str_12586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 29), 'str', '.%s-%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 205)
        tuple_12587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 205)
        # Adding element type (line 205)
        # Getting the type of 'self' (line 205)
        self_12588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 41), 'self')
        # Obtaining the member 'plat_name' of a type (line 205)
        plat_name_12589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 41), self_12588, 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 41), tuple_12587, plat_name_12589)
        # Adding element type (line 205)
        # Getting the type of 'target_version' (line 205)
        target_version_12590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 57), 'target_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 41), tuple_12587, target_version_12590)
        
        # Applying the binary operator '%' (line 205)
        result_mod_12591 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 29), '%', str_12586, tuple_12587)
        
        # Assigning a type to the variable 'plat_specifier' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'plat_specifier', result_mod_12591)
        
        # Assigning a Call to a Name (line 206):
        
        # Call to get_finalized_command(...): (line 206)
        # Processing the call arguments (line 206)
        str_12594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 47), 'str', 'build')
        # Processing the call keyword arguments (line 206)
        kwargs_12595 = {}
        # Getting the type of 'self' (line 206)
        self_12592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 206)
        get_finalized_command_12593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 20), self_12592, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 206)
        get_finalized_command_call_result_12596 = invoke(stypy.reporting.localization.Localization(__file__, 206, 20), get_finalized_command_12593, *[str_12594], **kwargs_12595)
        
        # Assigning a type to the variable 'build' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'build', get_finalized_command_call_result_12596)
        
        # Assigning a Call to a Attribute (line 207):
        
        # Call to join(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'build' (line 207)
        build_12600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 43), 'build', False)
        # Obtaining the member 'build_base' of a type (line 207)
        build_base_12601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 43), build_12600, 'build_base')
        str_12602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 43), 'str', 'lib')
        # Getting the type of 'plat_specifier' (line 208)
        plat_specifier_12603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 51), 'plat_specifier', False)
        # Applying the binary operator '+' (line 208)
        result_add_12604 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 43), '+', str_12602, plat_specifier_12603)
        
        # Processing the call keyword arguments (line 207)
        kwargs_12605 = {}
        # Getting the type of 'os' (line 207)
        os_12597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 207)
        path_12598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 30), os_12597, 'path')
        # Obtaining the member 'join' of a type (line 207)
        join_12599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 30), path_12598, 'join')
        # Calling join(args, kwargs) (line 207)
        join_call_result_12606 = invoke(stypy.reporting.localization.Localization(__file__, 207, 30), join_12599, *[build_base_12601, result_add_12604], **kwargs_12605)
        
        # Getting the type of 'build' (line 207)
        build_12607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'build')
        # Setting the type of the member 'build_lib' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), build_12607, 'build_lib', join_call_result_12606)
        # SSA join for if statement (line 194)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 210)
        # Processing the call arguments (line 210)
        str_12610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 17), 'str', 'installing to %s')
        # Getting the type of 'self' (line 210)
        self_12611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 37), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 210)
        bdist_dir_12612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 37), self_12611, 'bdist_dir')
        # Processing the call keyword arguments (line 210)
        kwargs_12613 = {}
        # Getting the type of 'log' (line 210)
        log_12608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 210)
        info_12609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), log_12608, 'info')
        # Calling info(args, kwargs) (line 210)
        info_call_result_12614 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), info_12609, *[str_12610, bdist_dir_12612], **kwargs_12613)
        
        
        # Call to ensure_finalized(...): (line 211)
        # Processing the call keyword arguments (line 211)
        kwargs_12617 = {}
        # Getting the type of 'install' (line 211)
        install_12615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'install', False)
        # Obtaining the member 'ensure_finalized' of a type (line 211)
        ensure_finalized_12616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), install_12615, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 211)
        ensure_finalized_call_result_12618 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), ensure_finalized_12616, *[], **kwargs_12617)
        
        
        # Call to insert(...): (line 215)
        # Processing the call arguments (line 215)
        int_12622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 24), 'int')
        
        # Call to join(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'self' (line 215)
        self_12626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 40), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 215)
        bdist_dir_12627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 40), self_12626, 'bdist_dir')
        str_12628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 56), 'str', 'PURELIB')
        # Processing the call keyword arguments (line 215)
        kwargs_12629 = {}
        # Getting the type of 'os' (line 215)
        os_12623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 215)
        path_12624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 27), os_12623, 'path')
        # Obtaining the member 'join' of a type (line 215)
        join_12625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 27), path_12624, 'join')
        # Calling join(args, kwargs) (line 215)
        join_call_result_12630 = invoke(stypy.reporting.localization.Localization(__file__, 215, 27), join_12625, *[bdist_dir_12627, str_12628], **kwargs_12629)
        
        # Processing the call keyword arguments (line 215)
        kwargs_12631 = {}
        # Getting the type of 'sys' (line 215)
        sys_12619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'sys', False)
        # Obtaining the member 'path' of a type (line 215)
        path_12620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), sys_12619, 'path')
        # Obtaining the member 'insert' of a type (line 215)
        insert_12621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), path_12620, 'insert')
        # Calling insert(args, kwargs) (line 215)
        insert_call_result_12632 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), insert_12621, *[int_12622, join_call_result_12630], **kwargs_12631)
        
        
        # Call to run(...): (line 217)
        # Processing the call keyword arguments (line 217)
        kwargs_12635 = {}
        # Getting the type of 'install' (line 217)
        install_12633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'install', False)
        # Obtaining the member 'run' of a type (line 217)
        run_12634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), install_12633, 'run')
        # Calling run(args, kwargs) (line 217)
        run_call_result_12636 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), run_12634, *[], **kwargs_12635)
        
        # Deleting a member
        # Getting the type of 'sys' (line 219)
        sys_12637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'sys')
        # Obtaining the member 'path' of a type (line 219)
        path_12638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 12), sys_12637, 'path')
        
        # Obtaining the type of the subscript
        int_12639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 21), 'int')
        # Getting the type of 'sys' (line 219)
        sys_12640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'sys')
        # Obtaining the member 'path' of a type (line 219)
        path_12641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 12), sys_12640, 'path')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___12642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 12), path_12641, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_12643 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), getitem___12642, int_12639)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 8), path_12638, subscript_call_result_12643)
        
        # Call to mkpath(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_12646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 221)
        dist_dir_12647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), self_12646, 'dist_dir')
        # Processing the call keyword arguments (line 221)
        kwargs_12648 = {}
        # Getting the type of 'self' (line 221)
        self_12644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 221)
        mkpath_12645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_12644, 'mkpath')
        # Calling mkpath(args, kwargs) (line 221)
        mkpath_call_result_12649 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), mkpath_12645, *[dist_dir_12647], **kwargs_12648)
        
        
        # Assigning a Call to a Name (line 222):
        
        # Call to get_fullname(...): (line 222)
        # Processing the call keyword arguments (line 222)
        kwargs_12653 = {}
        # Getting the type of 'self' (line 222)
        self_12650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'self', False)
        # Obtaining the member 'distribution' of a type (line 222)
        distribution_12651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 19), self_12650, 'distribution')
        # Obtaining the member 'get_fullname' of a type (line 222)
        get_fullname_12652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 19), distribution_12651, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 222)
        get_fullname_call_result_12654 = invoke(stypy.reporting.localization.Localization(__file__, 222, 19), get_fullname_12652, *[], **kwargs_12653)
        
        # Assigning a type to the variable 'fullname' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'fullname', get_fullname_call_result_12654)
        
        # Assigning a Call to a Name (line 223):
        
        # Call to get_installer_filename(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'fullname' (line 223)
        fullname_12657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 53), 'fullname', False)
        # Processing the call keyword arguments (line 223)
        kwargs_12658 = {}
        # Getting the type of 'self' (line 223)
        self_12655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'self', False)
        # Obtaining the member 'get_installer_filename' of a type (line 223)
        get_installer_filename_12656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 25), self_12655, 'get_installer_filename')
        # Calling get_installer_filename(args, kwargs) (line 223)
        get_installer_filename_call_result_12659 = invoke(stypy.reporting.localization.Localization(__file__, 223, 25), get_installer_filename_12656, *[fullname_12657], **kwargs_12658)
        
        # Assigning a type to the variable 'installer_name' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'installer_name', get_installer_filename_call_result_12659)
        
        # Assigning a Call to a Name (line 224):
        
        # Call to abspath(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'installer_name' (line 224)
        installer_name_12663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 41), 'installer_name', False)
        # Processing the call keyword arguments (line 224)
        kwargs_12664 = {}
        # Getting the type of 'os' (line 224)
        os_12660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 224)
        path_12661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 25), os_12660, 'path')
        # Obtaining the member 'abspath' of a type (line 224)
        abspath_12662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 25), path_12661, 'abspath')
        # Calling abspath(args, kwargs) (line 224)
        abspath_call_result_12665 = invoke(stypy.reporting.localization.Localization(__file__, 224, 25), abspath_12662, *[installer_name_12663], **kwargs_12664)
        
        # Assigning a type to the variable 'installer_name' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'installer_name', abspath_call_result_12665)
        
        
        # Call to exists(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'installer_name' (line 225)
        installer_name_12669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 26), 'installer_name', False)
        # Processing the call keyword arguments (line 225)
        kwargs_12670 = {}
        # Getting the type of 'os' (line 225)
        os_12666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 225)
        path_12667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 11), os_12666, 'path')
        # Obtaining the member 'exists' of a type (line 225)
        exists_12668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 11), path_12667, 'exists')
        # Calling exists(args, kwargs) (line 225)
        exists_call_result_12671 = invoke(stypy.reporting.localization.Localization(__file__, 225, 11), exists_12668, *[installer_name_12669], **kwargs_12670)
        
        # Testing the type of an if condition (line 225)
        if_condition_12672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 8), exists_call_result_12671)
        # Assigning a type to the variable 'if_condition_12672' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'if_condition_12672', if_condition_12672)
        # SSA begins for if statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to unlink(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'installer_name' (line 225)
        installer_name_12675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 53), 'installer_name', False)
        # Processing the call keyword arguments (line 225)
        kwargs_12676 = {}
        # Getting the type of 'os' (line 225)
        os_12673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 43), 'os', False)
        # Obtaining the member 'unlink' of a type (line 225)
        unlink_12674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 43), os_12673, 'unlink')
        # Calling unlink(args, kwargs) (line 225)
        unlink_call_result_12677 = invoke(stypy.reporting.localization.Localization(__file__, 225, 43), unlink_12674, *[installer_name_12675], **kwargs_12676)
        
        # SSA join for if statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 227):
        # Getting the type of 'self' (line 227)
        self_12678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 19), 'self')
        # Obtaining the member 'distribution' of a type (line 227)
        distribution_12679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 19), self_12678, 'distribution')
        # Obtaining the member 'metadata' of a type (line 227)
        metadata_12680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 19), distribution_12679, 'metadata')
        # Assigning a type to the variable 'metadata' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'metadata', metadata_12680)
        
        # Assigning a Attribute to a Name (line 228):
        # Getting the type of 'metadata' (line 228)
        metadata_12681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'metadata')
        # Obtaining the member 'author' of a type (line 228)
        author_12682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 17), metadata_12681, 'author')
        # Assigning a type to the variable 'author' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'author', author_12682)
        
        
        # Getting the type of 'author' (line 229)
        author_12683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'author')
        # Applying the 'not' unary operator (line 229)
        result_not__12684 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), 'not', author_12683)
        
        # Testing the type of an if condition (line 229)
        if_condition_12685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 8), result_not__12684)
        # Assigning a type to the variable 'if_condition_12685' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'if_condition_12685', if_condition_12685)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 230):
        # Getting the type of 'metadata' (line 230)
        metadata_12686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'metadata')
        # Obtaining the member 'maintainer' of a type (line 230)
        maintainer_12687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 21), metadata_12686, 'maintainer')
        # Assigning a type to the variable 'author' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'author', maintainer_12687)
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'author' (line 231)
        author_12688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'author')
        # Applying the 'not' unary operator (line 231)
        result_not__12689 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 11), 'not', author_12688)
        
        # Testing the type of an if condition (line 231)
        if_condition_12690 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 8), result_not__12689)
        # Assigning a type to the variable 'if_condition_12690' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'if_condition_12690', if_condition_12690)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 232):
        str_12691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 21), 'str', 'UNKNOWN')
        # Assigning a type to the variable 'author' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'author', str_12691)
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 233):
        
        # Call to get_version(...): (line 233)
        # Processing the call keyword arguments (line 233)
        kwargs_12694 = {}
        # Getting the type of 'metadata' (line 233)
        metadata_12692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'metadata', False)
        # Obtaining the member 'get_version' of a type (line 233)
        get_version_12693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 18), metadata_12692, 'get_version')
        # Calling get_version(args, kwargs) (line 233)
        get_version_call_result_12695 = invoke(stypy.reporting.localization.Localization(__file__, 233, 18), get_version_12693, *[], **kwargs_12694)
        
        # Assigning a type to the variable 'version' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'version', get_version_call_result_12695)
        
        # Assigning a BinOp to a Name (line 236):
        str_12696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 19), 'str', '%d.%d.%d')
        
        # Call to StrictVersion(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'version' (line 236)
        version_12698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 46), 'version', False)
        # Processing the call keyword arguments (line 236)
        kwargs_12699 = {}
        # Getting the type of 'StrictVersion' (line 236)
        StrictVersion_12697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 32), 'StrictVersion', False)
        # Calling StrictVersion(args, kwargs) (line 236)
        StrictVersion_call_result_12700 = invoke(stypy.reporting.localization.Localization(__file__, 236, 32), StrictVersion_12697, *[version_12698], **kwargs_12699)
        
        # Obtaining the member 'version' of a type (line 236)
        version_12701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 32), StrictVersion_call_result_12700, 'version')
        # Applying the binary operator '%' (line 236)
        result_mod_12702 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 19), '%', str_12696, version_12701)
        
        # Assigning a type to the variable 'sversion' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'sversion', result_mod_12702)
        
        # Assigning a Call to a Name (line 240):
        
        # Call to get_fullname(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_12706 = {}
        # Getting the type of 'self' (line 240)
        self_12703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'self', False)
        # Obtaining the member 'distribution' of a type (line 240)
        distribution_12704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 19), self_12703, 'distribution')
        # Obtaining the member 'get_fullname' of a type (line 240)
        get_fullname_12705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 19), distribution_12704, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 240)
        get_fullname_call_result_12707 = invoke(stypy.reporting.localization.Localization(__file__, 240, 19), get_fullname_12705, *[], **kwargs_12706)
        
        # Assigning a type to the variable 'fullname' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'fullname', get_fullname_call_result_12707)
        
        # Getting the type of 'self' (line 241)
        self_12708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'self')
        # Obtaining the member 'target_version' of a type (line 241)
        target_version_12709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 11), self_12708, 'target_version')
        # Testing the type of an if condition (line 241)
        if_condition_12710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 8), target_version_12709)
        # Assigning a type to the variable 'if_condition_12710' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'if_condition_12710', if_condition_12710)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 242):
        str_12711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 27), 'str', 'Python %s %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 242)
        tuple_12712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 242)
        # Adding element type (line 242)
        # Getting the type of 'self' (line 242)
        self_12713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 45), 'self')
        # Obtaining the member 'target_version' of a type (line 242)
        target_version_12714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 45), self_12713, 'target_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 45), tuple_12712, target_version_12714)
        # Adding element type (line 242)
        # Getting the type of 'fullname' (line 242)
        fullname_12715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 66), 'fullname')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 45), tuple_12712, fullname_12715)
        
        # Applying the binary operator '%' (line 242)
        result_mod_12716 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 27), '%', str_12711, tuple_12712)
        
        # Assigning a type to the variable 'product_name' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'product_name', result_mod_12716)
        # SSA branch for the else part of an if statement (line 241)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 244):
        str_12717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 27), 'str', 'Python %s')
        # Getting the type of 'fullname' (line 244)
        fullname_12718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 42), 'fullname')
        # Applying the binary operator '%' (line 244)
        result_mod_12719 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 27), '%', str_12717, fullname_12718)
        
        # Assigning a type to the variable 'product_name' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'product_name', result_mod_12719)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 245):
        
        # Call to init_database(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'installer_name' (line 245)
        installer_name_12722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 39), 'installer_name', False)
        # Getting the type of 'schema' (line 245)
        schema_12723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 55), 'schema', False)
        # Getting the type of 'product_name' (line 246)
        product_name_12724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'product_name', False)
        
        # Call to gen_uuid(...): (line 246)
        # Processing the call keyword arguments (line 246)
        kwargs_12727 = {}
        # Getting the type of 'msilib' (line 246)
        msilib_12725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 30), 'msilib', False)
        # Obtaining the member 'gen_uuid' of a type (line 246)
        gen_uuid_12726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 30), msilib_12725, 'gen_uuid')
        # Calling gen_uuid(args, kwargs) (line 246)
        gen_uuid_call_result_12728 = invoke(stypy.reporting.localization.Localization(__file__, 246, 30), gen_uuid_12726, *[], **kwargs_12727)
        
        # Getting the type of 'sversion' (line 247)
        sversion_12729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'sversion', False)
        # Getting the type of 'author' (line 247)
        author_12730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 26), 'author', False)
        # Processing the call keyword arguments (line 245)
        kwargs_12731 = {}
        # Getting the type of 'msilib' (line 245)
        msilib_12720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 18), 'msilib', False)
        # Obtaining the member 'init_database' of a type (line 245)
        init_database_12721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 18), msilib_12720, 'init_database')
        # Calling init_database(args, kwargs) (line 245)
        init_database_call_result_12732 = invoke(stypy.reporting.localization.Localization(__file__, 245, 18), init_database_12721, *[installer_name_12722, schema_12723, product_name_12724, gen_uuid_call_result_12728, sversion_12729, author_12730], **kwargs_12731)
        
        # Getting the type of 'self' (line 245)
        self_12733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self')
        # Setting the type of the member 'db' of a type (line 245)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_12733, 'db', init_database_call_result_12732)
        
        # Call to add_tables(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'self' (line 248)
        self_12736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 26), 'self', False)
        # Obtaining the member 'db' of a type (line 248)
        db_12737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 26), self_12736, 'db')
        # Getting the type of 'sequence' (line 248)
        sequence_12738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 35), 'sequence', False)
        # Processing the call keyword arguments (line 248)
        kwargs_12739 = {}
        # Getting the type of 'msilib' (line 248)
        msilib_12734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'msilib', False)
        # Obtaining the member 'add_tables' of a type (line 248)
        add_tables_12735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), msilib_12734, 'add_tables')
        # Calling add_tables(args, kwargs) (line 248)
        add_tables_call_result_12740 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), add_tables_12735, *[db_12737, sequence_12738], **kwargs_12739)
        
        
        # Assigning a List to a Name (line 249):
        
        # Obtaining an instance of the builtin type 'list' (line 249)
        list_12741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 249)
        # Adding element type (line 249)
        
        # Obtaining an instance of the builtin type 'tuple' (line 249)
        tuple_12742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 249)
        # Adding element type (line 249)
        str_12743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 18), 'str', 'DistVersion')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), tuple_12742, str_12743)
        # Adding element type (line 249)
        # Getting the type of 'version' (line 249)
        version_12744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 33), 'version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), tuple_12742, version_12744)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 16), list_12741, tuple_12742)
        
        # Assigning a type to the variable 'props' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'props', list_12741)
        
        # Assigning a BoolOp to a Name (line 250):
        
        # Evaluating a boolean operation
        # Getting the type of 'metadata' (line 250)
        metadata_12745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'metadata')
        # Obtaining the member 'author_email' of a type (line 250)
        author_email_12746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), metadata_12745, 'author_email')
        # Getting the type of 'metadata' (line 250)
        metadata_12747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 41), 'metadata')
        # Obtaining the member 'maintainer_email' of a type (line 250)
        maintainer_email_12748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 41), metadata_12747, 'maintainer_email')
        # Applying the binary operator 'or' (line 250)
        result_or_keyword_12749 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 16), 'or', author_email_12746, maintainer_email_12748)
        
        # Assigning a type to the variable 'email' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'email', result_or_keyword_12749)
        
        # Getting the type of 'email' (line 251)
        email_12750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'email')
        # Testing the type of an if condition (line 251)
        if_condition_12751 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), email_12750)
        # Assigning a type to the variable 'if_condition_12751' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'if_condition_12751', if_condition_12751)
        # SSA begins for if statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 252)
        # Processing the call arguments (line 252)
        
        # Obtaining an instance of the builtin type 'tuple' (line 252)
        tuple_12754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 252)
        # Adding element type (line 252)
        str_12755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 26), 'str', 'ARPCONTACT')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 26), tuple_12754, str_12755)
        # Adding element type (line 252)
        # Getting the type of 'email' (line 252)
        email_12756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 40), 'email', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 26), tuple_12754, email_12756)
        
        # Processing the call keyword arguments (line 252)
        kwargs_12757 = {}
        # Getting the type of 'props' (line 252)
        props_12752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'props', False)
        # Obtaining the member 'append' of a type (line 252)
        append_12753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), props_12752, 'append')
        # Calling append(args, kwargs) (line 252)
        append_call_result_12758 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), append_12753, *[tuple_12754], **kwargs_12757)
        
        # SSA join for if statement (line 251)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'metadata' (line 253)
        metadata_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'metadata')
        # Obtaining the member 'url' of a type (line 253)
        url_12760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 11), metadata_12759, 'url')
        # Testing the type of an if condition (line 253)
        if_condition_12761 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 8), url_12760)
        # Assigning a type to the variable 'if_condition_12761' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'if_condition_12761', if_condition_12761)
        # SSA begins for if statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Obtaining an instance of the builtin type 'tuple' (line 254)
        tuple_12764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 254)
        # Adding element type (line 254)
        str_12765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 26), 'str', 'ARPURLINFOABOUT')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 26), tuple_12764, str_12765)
        # Adding element type (line 254)
        # Getting the type of 'metadata' (line 254)
        metadata_12766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 45), 'metadata', False)
        # Obtaining the member 'url' of a type (line 254)
        url_12767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 45), metadata_12766, 'url')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 26), tuple_12764, url_12767)
        
        # Processing the call keyword arguments (line 254)
        kwargs_12768 = {}
        # Getting the type of 'props' (line 254)
        props_12762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'props', False)
        # Obtaining the member 'append' of a type (line 254)
        append_12763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), props_12762, 'append')
        # Calling append(args, kwargs) (line 254)
        append_call_result_12769 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), append_12763, *[tuple_12764], **kwargs_12768)
        
        # SSA join for if statement (line 253)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'props' (line 255)
        props_12770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'props')
        # Testing the type of an if condition (line 255)
        if_condition_12771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), props_12770)
        # Assigning a type to the variable 'if_condition_12771' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_12771', if_condition_12771)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add_data(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'self' (line 256)
        self_12773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 21), 'self', False)
        # Obtaining the member 'db' of a type (line 256)
        db_12774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 21), self_12773, 'db')
        str_12775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 30), 'str', 'Property')
        # Getting the type of 'props' (line 256)
        props_12776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 42), 'props', False)
        # Processing the call keyword arguments (line 256)
        kwargs_12777 = {}
        # Getting the type of 'add_data' (line 256)
        add_data_12772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'add_data', False)
        # Calling add_data(args, kwargs) (line 256)
        add_data_call_result_12778 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), add_data_12772, *[db_12774, str_12775, props_12776], **kwargs_12777)
        
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to add_find_python(...): (line 258)
        # Processing the call keyword arguments (line 258)
        kwargs_12781 = {}
        # Getting the type of 'self' (line 258)
        self_12779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'self', False)
        # Obtaining the member 'add_find_python' of a type (line 258)
        add_find_python_12780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), self_12779, 'add_find_python')
        # Calling add_find_python(args, kwargs) (line 258)
        add_find_python_call_result_12782 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), add_find_python_12780, *[], **kwargs_12781)
        
        
        # Call to add_files(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_12785 = {}
        # Getting the type of 'self' (line 259)
        self_12783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self', False)
        # Obtaining the member 'add_files' of a type (line 259)
        add_files_12784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_12783, 'add_files')
        # Calling add_files(args, kwargs) (line 259)
        add_files_call_result_12786 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), add_files_12784, *[], **kwargs_12785)
        
        
        # Call to add_scripts(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_12789 = {}
        # Getting the type of 'self' (line 260)
        self_12787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'self', False)
        # Obtaining the member 'add_scripts' of a type (line 260)
        add_scripts_12788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), self_12787, 'add_scripts')
        # Calling add_scripts(args, kwargs) (line 260)
        add_scripts_call_result_12790 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), add_scripts_12788, *[], **kwargs_12789)
        
        
        # Call to add_ui(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_12793 = {}
        # Getting the type of 'self' (line 261)
        self_12791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self', False)
        # Obtaining the member 'add_ui' of a type (line 261)
        add_ui_12792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_12791, 'add_ui')
        # Calling add_ui(args, kwargs) (line 261)
        add_ui_call_result_12794 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), add_ui_12792, *[], **kwargs_12793)
        
        
        # Call to Commit(...): (line 262)
        # Processing the call keyword arguments (line 262)
        kwargs_12798 = {}
        # Getting the type of 'self' (line 262)
        self_12795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
        # Obtaining the member 'db' of a type (line 262)
        db_12796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_12795, 'db')
        # Obtaining the member 'Commit' of a type (line 262)
        Commit_12797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), db_12796, 'Commit')
        # Calling Commit(args, kwargs) (line 262)
        Commit_call_result_12799 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), Commit_12797, *[], **kwargs_12798)
        
        
        # Type idiom detected: calculating its left and rigth part (line 264)
        str_12800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 38), 'str', 'dist_files')
        # Getting the type of 'self' (line 264)
        self_12801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 19), 'self')
        # Obtaining the member 'distribution' of a type (line 264)
        distribution_12802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 19), self_12801, 'distribution')
        
        (may_be_12803, more_types_in_union_12804) = may_provide_member(str_12800, distribution_12802)

        if may_be_12803:

            if more_types_in_union_12804:
                # Runtime conditional SSA (line 264)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 264)
            self_12805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self')
            # Obtaining the member 'distribution' of a type (line 264)
            distribution_12806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_12805, 'distribution')
            # Setting the type of the member 'distribution' of a type (line 264)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_12805, 'distribution', remove_not_member_provider_from_union(distribution_12802, 'dist_files'))
            
            # Assigning a Tuple to a Name (line 265):
            
            # Obtaining an instance of the builtin type 'tuple' (line 265)
            tuple_12807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 18), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 265)
            # Adding element type (line 265)
            str_12808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 18), 'str', 'bdist_msi')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 18), tuple_12807, str_12808)
            # Adding element type (line 265)
            
            # Evaluating a boolean operation
            # Getting the type of 'self' (line 265)
            self_12809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 31), 'self')
            # Obtaining the member 'target_version' of a type (line 265)
            target_version_12810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 31), self_12809, 'target_version')
            str_12811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 54), 'str', 'any')
            # Applying the binary operator 'or' (line 265)
            result_or_keyword_12812 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 31), 'or', target_version_12810, str_12811)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 18), tuple_12807, result_or_keyword_12812)
            # Adding element type (line 265)
            # Getting the type of 'fullname' (line 265)
            fullname_12813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 61), 'fullname')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 18), tuple_12807, fullname_12813)
            
            # Assigning a type to the variable 'tup' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'tup', tuple_12807)
            
            # Call to append(...): (line 266)
            # Processing the call arguments (line 266)
            # Getting the type of 'tup' (line 266)
            tup_12818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 48), 'tup', False)
            # Processing the call keyword arguments (line 266)
            kwargs_12819 = {}
            # Getting the type of 'self' (line 266)
            self_12814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'self', False)
            # Obtaining the member 'distribution' of a type (line 266)
            distribution_12815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), self_12814, 'distribution')
            # Obtaining the member 'dist_files' of a type (line 266)
            dist_files_12816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), distribution_12815, 'dist_files')
            # Obtaining the member 'append' of a type (line 266)
            append_12817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), dist_files_12816, 'append')
            # Calling append(args, kwargs) (line 266)
            append_call_result_12820 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), append_12817, *[tup_12818], **kwargs_12819)
            

            if more_types_in_union_12804:
                # SSA join for if statement (line 264)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'self' (line 268)
        self_12821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'self')
        # Obtaining the member 'keep_temp' of a type (line 268)
        keep_temp_12822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), self_12821, 'keep_temp')
        # Applying the 'not' unary operator (line 268)
        result_not__12823 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 11), 'not', keep_temp_12822)
        
        # Testing the type of an if condition (line 268)
        if_condition_12824 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 8), result_not__12823)
        # Assigning a type to the variable 'if_condition_12824' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'if_condition_12824', if_condition_12824)
        # SSA begins for if statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove_tree(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'self' (line 269)
        self_12826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 24), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 269)
        bdist_dir_12827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 24), self_12826, 'bdist_dir')
        # Processing the call keyword arguments (line 269)
        # Getting the type of 'self' (line 269)
        self_12828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 48), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 269)
        dry_run_12829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 48), self_12828, 'dry_run')
        keyword_12830 = dry_run_12829
        kwargs_12831 = {'dry_run': keyword_12830}
        # Getting the type of 'remove_tree' (line 269)
        remove_tree_12825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 269)
        remove_tree_call_result_12832 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), remove_tree_12825, *[bdist_dir_12827], **kwargs_12831)
        
        # SSA join for if statement (line 268)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_12833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12833)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_12833


    @norecursion
    def add_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_files'
        module_type_store = module_type_store.open_function_context('add_files', 271, 4, False)
        # Assigning a type to the variable 'self' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_msi.add_files.__dict__.__setitem__('stypy_localization', localization)
        bdist_msi.add_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_msi.add_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_msi.add_files.__dict__.__setitem__('stypy_function_name', 'bdist_msi.add_files')
        bdist_msi.add_files.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_msi.add_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_msi.add_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_msi.add_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_msi.add_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_msi.add_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_msi.add_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_msi.add_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_files(...)' code ##################

        
        # Assigning a Attribute to a Name (line 272):
        # Getting the type of 'self' (line 272)
        self_12834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 13), 'self')
        # Obtaining the member 'db' of a type (line 272)
        db_12835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 13), self_12834, 'db')
        # Assigning a type to the variable 'db' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'db', db_12835)
        
        # Assigning a Call to a Name (line 273):
        
        # Call to CAB(...): (line 273)
        # Processing the call arguments (line 273)
        str_12838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 25), 'str', 'distfiles')
        # Processing the call keyword arguments (line 273)
        kwargs_12839 = {}
        # Getting the type of 'msilib' (line 273)
        msilib_12836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 14), 'msilib', False)
        # Obtaining the member 'CAB' of a type (line 273)
        CAB_12837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 14), msilib_12836, 'CAB')
        # Calling CAB(args, kwargs) (line 273)
        CAB_call_result_12840 = invoke(stypy.reporting.localization.Localization(__file__, 273, 14), CAB_12837, *[str_12838], **kwargs_12839)
        
        # Assigning a type to the variable 'cab' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'cab', CAB_call_result_12840)
        
        # Assigning a Call to a Name (line 274):
        
        # Call to abspath(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'self' (line 274)
        self_12844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 34), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 274)
        bdist_dir_12845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 34), self_12844, 'bdist_dir')
        # Processing the call keyword arguments (line 274)
        kwargs_12846 = {}
        # Getting the type of 'os' (line 274)
        os_12841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 274)
        path_12842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 18), os_12841, 'path')
        # Obtaining the member 'abspath' of a type (line 274)
        abspath_12843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 18), path_12842, 'abspath')
        # Calling abspath(args, kwargs) (line 274)
        abspath_call_result_12847 = invoke(stypy.reporting.localization.Localization(__file__, 274, 18), abspath_12843, *[bdist_dir_12845], **kwargs_12846)
        
        # Assigning a type to the variable 'rootdir' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'rootdir', abspath_call_result_12847)
        
        # Assigning a Call to a Name (line 276):
        
        # Call to Directory(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'db' (line 276)
        db_12849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 25), 'db', False)
        # Getting the type of 'cab' (line 276)
        cab_12850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 29), 'cab', False)
        # Getting the type of 'None' (line 276)
        None_12851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 'None', False)
        # Getting the type of 'rootdir' (line 276)
        rootdir_12852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 40), 'rootdir', False)
        str_12853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 49), 'str', 'TARGETDIR')
        str_12854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 62), 'str', 'SourceDir')
        # Processing the call keyword arguments (line 276)
        kwargs_12855 = {}
        # Getting the type of 'Directory' (line 276)
        Directory_12848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'Directory', False)
        # Calling Directory(args, kwargs) (line 276)
        Directory_call_result_12856 = invoke(stypy.reporting.localization.Localization(__file__, 276, 15), Directory_12848, *[db_12849, cab_12850, None_12851, rootdir_12852, str_12853, str_12854], **kwargs_12855)
        
        # Assigning a type to the variable 'root' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'root', Directory_call_result_12856)
        
        # Assigning a Call to a Name (line 277):
        
        # Call to Feature(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'db' (line 277)
        db_12858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'db', False)
        str_12859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 24), 'str', 'Python')
        str_12860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 34), 'str', 'Python')
        str_12861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 44), 'str', 'Everything')
        int_12862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 20), 'int')
        int_12863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 23), 'int')
        # Processing the call keyword arguments (line 277)
        str_12864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 36), 'str', 'TARGETDIR')
        keyword_12865 = str_12864
        kwargs_12866 = {'directory': keyword_12865}
        # Getting the type of 'Feature' (line 277)
        Feature_12857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'Feature', False)
        # Calling Feature(args, kwargs) (line 277)
        Feature_call_result_12867 = invoke(stypy.reporting.localization.Localization(__file__, 277, 12), Feature_12857, *[db_12858, str_12859, str_12860, str_12861, int_12862, int_12863], **kwargs_12866)
        
        # Assigning a type to the variable 'f' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'f', Feature_call_result_12867)
        
        # Assigning a List to a Name (line 280):
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_12868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        
        # Obtaining an instance of the builtin type 'tuple' (line 280)
        tuple_12869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 280)
        # Adding element type (line 280)
        # Getting the type of 'f' (line 280)
        f_12870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 18), 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 18), tuple_12869, f_12870)
        # Adding element type (line 280)
        # Getting the type of 'root' (line 280)
        root_12871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 21), 'root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 18), tuple_12869, root_12871)
        # Adding element type (line 280)
        str_12872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 27), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 18), tuple_12869, str_12872)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 16), list_12868, tuple_12869)
        
        # Assigning a type to the variable 'items' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'items', list_12868)
        
        # Getting the type of 'self' (line 281)
        self_12873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'self')
        # Obtaining the member 'versions' of a type (line 281)
        versions_12874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 23), self_12873, 'versions')
        
        # Obtaining an instance of the builtin type 'list' (line 281)
        list_12875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 281)
        # Adding element type (line 281)
        # Getting the type of 'self' (line 281)
        self_12876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 40), 'self')
        # Obtaining the member 'other_version' of a type (line 281)
        other_version_12877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 40), self_12876, 'other_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 39), list_12875, other_version_12877)
        
        # Applying the binary operator '+' (line 281)
        result_add_12878 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 23), '+', versions_12874, list_12875)
        
        # Testing the type of a for loop iterable (line 281)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 281, 8), result_add_12878)
        # Getting the type of the for loop variable (line 281)
        for_loop_var_12879 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 281, 8), result_add_12878)
        # Assigning a type to the variable 'version' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'version', for_loop_var_12879)
        # SSA begins for a for statement (line 281)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 282):
        str_12880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 21), 'str', 'TARGETDIR')
        # Getting the type of 'version' (line 282)
        version_12881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 35), 'version')
        # Applying the binary operator '+' (line 282)
        result_add_12882 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 21), '+', str_12880, version_12881)
        
        # Assigning a type to the variable 'target' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'target', result_add_12882)
        
        # Multiple assignment of 2 elements.
        str_12883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 29), 'str', 'Python')
        # Getting the type of 'version' (line 283)
        version_12884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 40), 'version')
        # Applying the binary operator '+' (line 283)
        result_add_12885 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 29), '+', str_12883, version_12884)
        
        # Assigning a type to the variable 'default' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'default', result_add_12885)
        # Getting the type of 'default' (line 283)
        default_12886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'default')
        # Assigning a type to the variable 'name' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'name', default_12886)
        
        # Assigning a Str to a Name (line 284):
        str_12887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 19), 'str', 'Everything')
        # Assigning a type to the variable 'desc' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'desc', str_12887)
        
        
        # Getting the type of 'version' (line 285)
        version_12888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'version')
        # Getting the type of 'self' (line 285)
        self_12889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 26), 'self')
        # Obtaining the member 'other_version' of a type (line 285)
        other_version_12890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 26), self_12889, 'other_version')
        # Applying the binary operator 'is' (line 285)
        result_is__12891 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 15), 'is', version_12888, other_version_12890)
        
        # Testing the type of an if condition (line 285)
        if_condition_12892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 12), result_is__12891)
        # Assigning a type to the variable 'if_condition_12892' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'if_condition_12892', if_condition_12892)
        # SSA begins for if statement (line 285)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 286):
        str_12893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 24), 'str', 'Python from another location')
        # Assigning a type to the variable 'title' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'title', str_12893)
        
        # Assigning a Num to a Name (line 287):
        int_12894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 24), 'int')
        # Assigning a type to the variable 'level' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'level', int_12894)
        # SSA branch for the else part of an if statement (line 285)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 289):
        str_12895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 24), 'str', 'Python %s from registry')
        # Getting the type of 'version' (line 289)
        version_12896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 52), 'version')
        # Applying the binary operator '%' (line 289)
        result_mod_12897 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 24), '%', str_12895, version_12896)
        
        # Assigning a type to the variable 'title' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'title', result_mod_12897)
        
        # Assigning a Num to a Name (line 290):
        int_12898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 24), 'int')
        # Assigning a type to the variable 'level' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'level', int_12898)
        # SSA join for if statement (line 285)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 291):
        
        # Call to Feature(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'db' (line 291)
        db_12900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'db', False)
        # Getting the type of 'name' (line 291)
        name_12901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 28), 'name', False)
        # Getting the type of 'title' (line 291)
        title_12902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 34), 'title', False)
        # Getting the type of 'desc' (line 291)
        desc_12903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 41), 'desc', False)
        int_12904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 47), 'int')
        # Getting the type of 'level' (line 291)
        level_12905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 50), 'level', False)
        # Processing the call keyword arguments (line 291)
        # Getting the type of 'target' (line 291)
        target_12906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 67), 'target', False)
        keyword_12907 = target_12906
        kwargs_12908 = {'directory': keyword_12907}
        # Getting the type of 'Feature' (line 291)
        Feature_12899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'Feature', False)
        # Calling Feature(args, kwargs) (line 291)
        Feature_call_result_12909 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), Feature_12899, *[db_12900, name_12901, title_12902, desc_12903, int_12904, level_12905], **kwargs_12908)
        
        # Assigning a type to the variable 'f' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'f', Feature_call_result_12909)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to Directory(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'db' (line 292)
        db_12911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 28), 'db', False)
        # Getting the type of 'cab' (line 292)
        cab_12912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 32), 'cab', False)
        # Getting the type of 'root' (line 292)
        root_12913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 37), 'root', False)
        # Getting the type of 'rootdir' (line 292)
        rootdir_12914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 43), 'rootdir', False)
        # Getting the type of 'target' (line 292)
        target_12915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 52), 'target', False)
        # Getting the type of 'default' (line 292)
        default_12916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 60), 'default', False)
        # Processing the call keyword arguments (line 292)
        kwargs_12917 = {}
        # Getting the type of 'Directory' (line 292)
        Directory_12910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 18), 'Directory', False)
        # Calling Directory(args, kwargs) (line 292)
        Directory_call_result_12918 = invoke(stypy.reporting.localization.Localization(__file__, 292, 18), Directory_12910, *[db_12911, cab_12912, root_12913, rootdir_12914, target_12915, default_12916], **kwargs_12917)
        
        # Assigning a type to the variable 'dir' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'dir', Directory_call_result_12918)
        
        # Call to append(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Obtaining an instance of the builtin type 'tuple' (line 293)
        tuple_12921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 293)
        # Adding element type (line 293)
        # Getting the type of 'f' (line 293)
        f_12922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 26), tuple_12921, f_12922)
        # Adding element type (line 293)
        # Getting the type of 'dir' (line 293)
        dir_12923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 29), 'dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 26), tuple_12921, dir_12923)
        # Adding element type (line 293)
        # Getting the type of 'version' (line 293)
        version_12924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 34), 'version', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 26), tuple_12921, version_12924)
        
        # Processing the call keyword arguments (line 293)
        kwargs_12925 = {}
        # Getting the type of 'items' (line 293)
        items_12919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'items', False)
        # Obtaining the member 'append' of a type (line 293)
        append_12920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), items_12919, 'append')
        # Calling append(args, kwargs) (line 293)
        append_call_result_12926 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), append_12920, *[tuple_12921], **kwargs_12925)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to Commit(...): (line 294)
        # Processing the call keyword arguments (line 294)
        kwargs_12929 = {}
        # Getting the type of 'db' (line 294)
        db_12927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'db', False)
        # Obtaining the member 'Commit' of a type (line 294)
        Commit_12928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), db_12927, 'Commit')
        # Calling Commit(args, kwargs) (line 294)
        Commit_call_result_12930 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), Commit_12928, *[], **kwargs_12929)
        
        
        # Assigning a Dict to a Name (line 296):
        
        # Obtaining an instance of the builtin type 'dict' (line 296)
        dict_12931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 296)
        
        # Assigning a type to the variable 'seen' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'seen', dict_12931)
        
        # Getting the type of 'items' (line 297)
        items_12932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 37), 'items')
        # Testing the type of a for loop iterable (line 297)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 297, 8), items_12932)
        # Getting the type of the for loop variable (line 297)
        for_loop_var_12933 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 297, 8), items_12932)
        # Assigning a type to the variable 'feature' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'feature', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 8), for_loop_var_12933))
        # Assigning a type to the variable 'dir' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'dir', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 8), for_loop_var_12933))
        # Assigning a type to the variable 'version' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'version', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 8), for_loop_var_12933))
        # SSA begins for a for statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 298):
        
        # Obtaining an instance of the builtin type 'list' (line 298)
        list_12934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 298)
        # Adding element type (line 298)
        # Getting the type of 'dir' (line 298)
        dir_12935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 19), list_12934, dir_12935)
        
        # Assigning a type to the variable 'todo' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'todo', list_12934)
        
        # Getting the type of 'todo' (line 299)
        todo_12936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 18), 'todo')
        # Testing the type of an if condition (line 299)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 12), todo_12936)
        # SSA begins for while statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 300):
        
        # Call to pop(...): (line 300)
        # Processing the call keyword arguments (line 300)
        kwargs_12939 = {}
        # Getting the type of 'todo' (line 300)
        todo_12937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 22), 'todo', False)
        # Obtaining the member 'pop' of a type (line 300)
        pop_12938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 22), todo_12937, 'pop')
        # Calling pop(args, kwargs) (line 300)
        pop_call_result_12940 = invoke(stypy.reporting.localization.Localization(__file__, 300, 22), pop_12938, *[], **kwargs_12939)
        
        # Assigning a type to the variable 'dir' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'dir', pop_call_result_12940)
        
        
        # Call to listdir(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'dir' (line 301)
        dir_12943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 39), 'dir', False)
        # Obtaining the member 'absolute' of a type (line 301)
        absolute_12944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 39), dir_12943, 'absolute')
        # Processing the call keyword arguments (line 301)
        kwargs_12945 = {}
        # Getting the type of 'os' (line 301)
        os_12941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 28), 'os', False)
        # Obtaining the member 'listdir' of a type (line 301)
        listdir_12942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 28), os_12941, 'listdir')
        # Calling listdir(args, kwargs) (line 301)
        listdir_call_result_12946 = invoke(stypy.reporting.localization.Localization(__file__, 301, 28), listdir_12942, *[absolute_12944], **kwargs_12945)
        
        # Testing the type of a for loop iterable (line 301)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 301, 16), listdir_call_result_12946)
        # Getting the type of the for loop variable (line 301)
        for_loop_var_12947 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 301, 16), listdir_call_result_12946)
        # Assigning a type to the variable 'file' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'file', for_loop_var_12947)
        # SSA begins for a for statement (line 301)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 302):
        
        # Call to join(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'dir' (line 302)
        dir_12951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 41), 'dir', False)
        # Obtaining the member 'absolute' of a type (line 302)
        absolute_12952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 41), dir_12951, 'absolute')
        # Getting the type of 'file' (line 302)
        file_12953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 55), 'file', False)
        # Processing the call keyword arguments (line 302)
        kwargs_12954 = {}
        # Getting the type of 'os' (line 302)
        os_12948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 302)
        path_12949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 28), os_12948, 'path')
        # Obtaining the member 'join' of a type (line 302)
        join_12950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 28), path_12949, 'join')
        # Calling join(args, kwargs) (line 302)
        join_call_result_12955 = invoke(stypy.reporting.localization.Localization(__file__, 302, 28), join_12950, *[absolute_12952, file_12953], **kwargs_12954)
        
        # Assigning a type to the variable 'afile' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 20), 'afile', join_call_result_12955)
        
        
        # Call to isdir(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'afile' (line 303)
        afile_12959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 37), 'afile', False)
        # Processing the call keyword arguments (line 303)
        kwargs_12960 = {}
        # Getting the type of 'os' (line 303)
        os_12956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 303)
        path_12957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 23), os_12956, 'path')
        # Obtaining the member 'isdir' of a type (line 303)
        isdir_12958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 23), path_12957, 'isdir')
        # Calling isdir(args, kwargs) (line 303)
        isdir_call_result_12961 = invoke(stypy.reporting.localization.Localization(__file__, 303, 23), isdir_12958, *[afile_12959], **kwargs_12960)
        
        # Testing the type of an if condition (line 303)
        if_condition_12962 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 20), isdir_call_result_12961)
        # Assigning a type to the variable 'if_condition_12962' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'if_condition_12962', if_condition_12962)
        # SSA begins for if statement (line 303)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 304):
        str_12963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 32), 'str', '%s|%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 304)
        tuple_12964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 304)
        # Adding element type (line 304)
        
        # Call to make_short(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'file' (line 304)
        file_12967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 58), 'file', False)
        # Processing the call keyword arguments (line 304)
        kwargs_12968 = {}
        # Getting the type of 'dir' (line 304)
        dir_12965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'dir', False)
        # Obtaining the member 'make_short' of a type (line 304)
        make_short_12966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 43), dir_12965, 'make_short')
        # Calling make_short(args, kwargs) (line 304)
        make_short_call_result_12969 = invoke(stypy.reporting.localization.Localization(__file__, 304, 43), make_short_12966, *[file_12967], **kwargs_12968)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 43), tuple_12964, make_short_call_result_12969)
        # Adding element type (line 304)
        # Getting the type of 'file' (line 304)
        file_12970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 65), 'file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 43), tuple_12964, file_12970)
        
        # Applying the binary operator '%' (line 304)
        result_mod_12971 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 32), '%', str_12963, tuple_12964)
        
        # Assigning a type to the variable 'short' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'short', result_mod_12971)
        
        # Assigning a BinOp to a Name (line 305):
        # Getting the type of 'file' (line 305)
        file_12972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 34), 'file')
        # Getting the type of 'version' (line 305)
        version_12973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 41), 'version')
        # Applying the binary operator '+' (line 305)
        result_add_12974 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 34), '+', file_12972, version_12973)
        
        # Assigning a type to the variable 'default' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 24), 'default', result_add_12974)
        
        # Assigning a Call to a Name (line 306):
        
        # Call to Directory(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'db' (line 306)
        db_12976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 43), 'db', False)
        # Getting the type of 'cab' (line 306)
        cab_12977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 47), 'cab', False)
        # Getting the type of 'dir' (line 306)
        dir_12978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 52), 'dir', False)
        # Getting the type of 'file' (line 306)
        file_12979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 57), 'file', False)
        # Getting the type of 'default' (line 306)
        default_12980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 63), 'default', False)
        # Getting the type of 'short' (line 306)
        short_12981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 72), 'short', False)
        # Processing the call keyword arguments (line 306)
        kwargs_12982 = {}
        # Getting the type of 'Directory' (line 306)
        Directory_12975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 33), 'Directory', False)
        # Calling Directory(args, kwargs) (line 306)
        Directory_call_result_12983 = invoke(stypy.reporting.localization.Localization(__file__, 306, 33), Directory_12975, *[db_12976, cab_12977, dir_12978, file_12979, default_12980, short_12981], **kwargs_12982)
        
        # Assigning a type to the variable 'newdir' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 24), 'newdir', Directory_call_result_12983)
        
        # Call to append(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'newdir' (line 307)
        newdir_12986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 36), 'newdir', False)
        # Processing the call keyword arguments (line 307)
        kwargs_12987 = {}
        # Getting the type of 'todo' (line 307)
        todo_12984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 24), 'todo', False)
        # Obtaining the member 'append' of a type (line 307)
        append_12985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 24), todo_12984, 'append')
        # Calling append(args, kwargs) (line 307)
        append_call_result_12988 = invoke(stypy.reporting.localization.Localization(__file__, 307, 24), append_12985, *[newdir_12986], **kwargs_12987)
        
        # SSA branch for the else part of an if statement (line 303)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'dir' (line 309)
        dir_12989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 31), 'dir')
        # Obtaining the member 'component' of a type (line 309)
        component_12990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 31), dir_12989, 'component')
        # Applying the 'not' unary operator (line 309)
        result_not__12991 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 27), 'not', component_12990)
        
        # Testing the type of an if condition (line 309)
        if_condition_12992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 24), result_not__12991)
        # Assigning a type to the variable 'if_condition_12992' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 24), 'if_condition_12992', if_condition_12992)
        # SSA begins for if statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to start_component(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'dir' (line 310)
        dir_12995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 48), 'dir', False)
        # Obtaining the member 'logical' of a type (line 310)
        logical_12996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 48), dir_12995, 'logical')
        # Getting the type of 'feature' (line 310)
        feature_12997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 61), 'feature', False)
        int_12998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 70), 'int')
        # Processing the call keyword arguments (line 310)
        kwargs_12999 = {}
        # Getting the type of 'dir' (line 310)
        dir_12993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 28), 'dir', False)
        # Obtaining the member 'start_component' of a type (line 310)
        start_component_12994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 28), dir_12993, 'start_component')
        # Calling start_component(args, kwargs) (line 310)
        start_component_call_result_13000 = invoke(stypy.reporting.localization.Localization(__file__, 310, 28), start_component_12994, *[logical_12996, feature_12997, int_12998], **kwargs_12999)
        
        # SSA join for if statement (line 309)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'afile' (line 311)
        afile_13001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 27), 'afile')
        # Getting the type of 'seen' (line 311)
        seen_13002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 40), 'seen')
        # Applying the binary operator 'notin' (line 311)
        result_contains_13003 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 27), 'notin', afile_13001, seen_13002)
        
        # Testing the type of an if condition (line 311)
        if_condition_13004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 24), result_contains_13003)
        # Assigning a type to the variable 'if_condition_13004' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 24), 'if_condition_13004', if_condition_13004)
        # SSA begins for if statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Multiple assignment of 2 elements.
        
        # Call to add_file(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'file' (line 312)
        file_13007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 61), 'file', False)
        # Processing the call keyword arguments (line 312)
        kwargs_13008 = {}
        # Getting the type of 'dir' (line 312)
        dir_13005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 48), 'dir', False)
        # Obtaining the member 'add_file' of a type (line 312)
        add_file_13006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 48), dir_13005, 'add_file')
        # Calling add_file(args, kwargs) (line 312)
        add_file_call_result_13009 = invoke(stypy.reporting.localization.Localization(__file__, 312, 48), add_file_13006, *[file_13007], **kwargs_13008)
        
        # Getting the type of 'seen' (line 312)
        seen_13010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'seen')
        # Getting the type of 'afile' (line 312)
        afile_13011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 39), 'afile')
        # Storing an element on a container (line 312)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 34), seen_13010, (afile_13011, add_file_call_result_13009))
        
        # Obtaining the type of the subscript
        # Getting the type of 'afile' (line 312)
        afile_13012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 39), 'afile')
        # Getting the type of 'seen' (line 312)
        seen_13013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'seen')
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___13014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 34), seen_13013, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_13015 = invoke(stypy.reporting.localization.Localization(__file__, 312, 34), getitem___13014, afile_13012)
        
        # Assigning a type to the variable 'key' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 28), 'key', subscript_call_result_13015)
        
        
        # Getting the type of 'file' (line 313)
        file_13016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 31), 'file')
        # Getting the type of 'self' (line 313)
        self_13017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 37), 'self')
        # Obtaining the member 'install_script' of a type (line 313)
        install_script_13018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 37), self_13017, 'install_script')
        # Applying the binary operator '==' (line 313)
        result_eq_13019 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 31), '==', file_13016, install_script_13018)
        
        # Testing the type of an if condition (line 313)
        if_condition_13020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 28), result_eq_13019)
        # Assigning a type to the variable 'if_condition_13020' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 28), 'if_condition_13020', if_condition_13020)
        # SSA begins for if statement (line 313)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 314)
        self_13021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 35), 'self')
        # Obtaining the member 'install_script_key' of a type (line 314)
        install_script_key_13022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 35), self_13021, 'install_script_key')
        # Testing the type of an if condition (line 314)
        if_condition_13023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 32), install_script_key_13022)
        # Assigning a type to the variable 'if_condition_13023' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 32), 'if_condition_13023', if_condition_13023)
        # SSA begins for if statement (line 314)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsOptionError(...): (line 315)
        # Processing the call arguments (line 315)
        str_13025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 42), 'str', 'Multiple files with name %s')
        # Getting the type of 'file' (line 316)
        file_13026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 74), 'file', False)
        # Applying the binary operator '%' (line 316)
        result_mod_13027 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 42), '%', str_13025, file_13026)
        
        # Processing the call keyword arguments (line 315)
        kwargs_13028 = {}
        # Getting the type of 'DistutilsOptionError' (line 315)
        DistutilsOptionError_13024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 42), 'DistutilsOptionError', False)
        # Calling DistutilsOptionError(args, kwargs) (line 315)
        DistutilsOptionError_call_result_13029 = invoke(stypy.reporting.localization.Localization(__file__, 315, 42), DistutilsOptionError_13024, *[result_mod_13027], **kwargs_13028)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 315, 36), DistutilsOptionError_call_result_13029, 'raise parameter', BaseException)
        # SSA join for if statement (line 314)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Attribute (line 317):
        str_13030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 58), 'str', '[#%s]')
        # Getting the type of 'key' (line 317)
        key_13031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 68), 'key')
        # Applying the binary operator '%' (line 317)
        result_mod_13032 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 58), '%', str_13030, key_13031)
        
        # Getting the type of 'self' (line 317)
        self_13033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'self')
        # Setting the type of the member 'install_script_key' of a type (line 317)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 32), self_13033, 'install_script_key', result_mod_13032)
        # SSA join for if statement (line 313)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 311)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 319):
        
        # Obtaining the type of the subscript
        # Getting the type of 'afile' (line 319)
        afile_13034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 39), 'afile')
        # Getting the type of 'seen' (line 319)
        seen_13035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 34), 'seen')
        # Obtaining the member '__getitem__' of a type (line 319)
        getitem___13036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 34), seen_13035, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 319)
        subscript_call_result_13037 = invoke(stypy.reporting.localization.Localization(__file__, 319, 34), getitem___13036, afile_13034)
        
        # Assigning a type to the variable 'key' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 28), 'key', subscript_call_result_13037)
        
        # Call to add_data(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'self' (line 320)
        self_13039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 37), 'self', False)
        # Obtaining the member 'db' of a type (line 320)
        db_13040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 37), self_13039, 'db')
        str_13041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 46), 'str', 'DuplicateFile')
        
        # Obtaining an instance of the builtin type 'list' (line 321)
        list_13042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 321)
        # Adding element type (line 321)
        
        # Obtaining an instance of the builtin type 'tuple' (line 321)
        tuple_13043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 321)
        # Adding element type (line 321)
        # Getting the type of 'key' (line 321)
        key_13044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 34), 'key', False)
        # Getting the type of 'version' (line 321)
        version_13045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 40), 'version', False)
        # Applying the binary operator '+' (line 321)
        result_add_13046 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 34), '+', key_13044, version_13045)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 34), tuple_13043, result_add_13046)
        # Adding element type (line 321)
        # Getting the type of 'dir' (line 321)
        dir_13047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 49), 'dir', False)
        # Obtaining the member 'component' of a type (line 321)
        component_13048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 49), dir_13047, 'component')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 34), tuple_13043, component_13048)
        # Adding element type (line 321)
        # Getting the type of 'key' (line 321)
        key_13049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 64), 'key', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 34), tuple_13043, key_13049)
        # Adding element type (line 321)
        # Getting the type of 'None' (line 321)
        None_13050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 69), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 34), tuple_13043, None_13050)
        # Adding element type (line 321)
        # Getting the type of 'dir' (line 321)
        dir_13051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 75), 'dir', False)
        # Obtaining the member 'logical' of a type (line 321)
        logical_13052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 75), dir_13051, 'logical')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 34), tuple_13043, logical_13052)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 32), list_13042, tuple_13043)
        
        # Processing the call keyword arguments (line 320)
        kwargs_13053 = {}
        # Getting the type of 'add_data' (line 320)
        add_data_13038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 28), 'add_data', False)
        # Calling add_data(args, kwargs) (line 320)
        add_data_call_result_13054 = invoke(stypy.reporting.localization.Localization(__file__, 320, 28), add_data_13038, *[db_13040, str_13041, list_13042], **kwargs_13053)
        
        # SSA join for if statement (line 311)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 303)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 299)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to Commit(...): (line 322)
        # Processing the call keyword arguments (line 322)
        kwargs_13057 = {}
        # Getting the type of 'db' (line 322)
        db_13055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'db', False)
        # Obtaining the member 'Commit' of a type (line 322)
        Commit_13056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 12), db_13055, 'Commit')
        # Calling Commit(args, kwargs) (line 322)
        Commit_call_result_13058 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), Commit_13056, *[], **kwargs_13057)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to commit(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'db' (line 323)
        db_13061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'db', False)
        # Processing the call keyword arguments (line 323)
        kwargs_13062 = {}
        # Getting the type of 'cab' (line 323)
        cab_13059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'cab', False)
        # Obtaining the member 'commit' of a type (line 323)
        commit_13060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), cab_13059, 'commit')
        # Calling commit(args, kwargs) (line 323)
        commit_call_result_13063 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), commit_13060, *[db_13061], **kwargs_13062)
        
        
        # ################# End of 'add_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_files' in the type store
        # Getting the type of 'stypy_return_type' (line 271)
        stypy_return_type_13064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13064)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_files'
        return stypy_return_type_13064


    @norecursion
    def add_find_python(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_find_python'
        module_type_store = module_type_store.open_function_context('add_find_python', 325, 4, False)
        # Assigning a type to the variable 'self' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_localization', localization)
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_function_name', 'bdist_msi.add_find_python')
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_msi.add_find_python.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_msi.add_find_python', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_find_python', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_find_python(...)' code ##################

        str_13065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, (-1)), 'str', 'Adds code to the installer to compute the location of Python.\n\n        Properties PYTHON.MACHINE.X.Y and PYTHON.USER.X.Y will be set from the\n        registry for each version of Python.\n\n        Properties TARGETDIRX.Y will be set from PYTHON.USER.X.Y if defined,\n        else from PYTHON.MACHINE.X.Y.\n\n        Properties PYTHONX.Y will be set to TARGETDIRX.Y\\python.exe')
        
        # Assigning a Num to a Name (line 336):
        int_13066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 16), 'int')
        # Assigning a type to the variable 'start' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'start', int_13066)
        
        # Getting the type of 'self' (line 337)
        self_13067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 19), 'self')
        # Obtaining the member 'versions' of a type (line 337)
        versions_13068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 19), self_13067, 'versions')
        # Testing the type of a for loop iterable (line 337)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 337, 8), versions_13068)
        # Getting the type of the for loop variable (line 337)
        for_loop_var_13069 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 337, 8), versions_13068)
        # Assigning a type to the variable 'ver' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'ver', for_loop_var_13069)
        # SSA begins for a for statement (line 337)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 338):
        str_13070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 27), 'str', 'SOFTWARE\\Python\\PythonCore\\%s\\InstallPath')
        # Getting the type of 'ver' (line 338)
        ver_13071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 74), 'ver')
        # Applying the binary operator '%' (line 338)
        result_mod_13072 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 27), '%', str_13070, ver_13071)
        
        # Assigning a type to the variable 'install_path' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'install_path', result_mod_13072)
        
        # Assigning a BinOp to a Name (line 339):
        str_13073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 26), 'str', 'python.machine.')
        # Getting the type of 'ver' (line 339)
        ver_13074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 46), 'ver')
        # Applying the binary operator '+' (line 339)
        result_add_13075 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 26), '+', str_13073, ver_13074)
        
        # Assigning a type to the variable 'machine_reg' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'machine_reg', result_add_13075)
        
        # Assigning a BinOp to a Name (line 340):
        str_13076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 23), 'str', 'python.user.')
        # Getting the type of 'ver' (line 340)
        ver_13077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 40), 'ver')
        # Applying the binary operator '+' (line 340)
        result_add_13078 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 23), '+', str_13076, ver_13077)
        
        # Assigning a type to the variable 'user_reg' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'user_reg', result_add_13078)
        
        # Assigning a BinOp to a Name (line 341):
        str_13079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 27), 'str', 'PYTHON.MACHINE.')
        # Getting the type of 'ver' (line 341)
        ver_13080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 47), 'ver')
        # Applying the binary operator '+' (line 341)
        result_add_13081 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 27), '+', str_13079, ver_13080)
        
        # Assigning a type to the variable 'machine_prop' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'machine_prop', result_add_13081)
        
        # Assigning a BinOp to a Name (line 342):
        str_13082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 24), 'str', 'PYTHON.USER.')
        # Getting the type of 'ver' (line 342)
        ver_13083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 41), 'ver')
        # Applying the binary operator '+' (line 342)
        result_add_13084 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 24), '+', str_13082, ver_13083)
        
        # Assigning a type to the variable 'user_prop' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'user_prop', result_add_13084)
        
        # Assigning a BinOp to a Name (line 343):
        str_13085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 29), 'str', 'PythonFromMachine')
        # Getting the type of 'ver' (line 343)
        ver_13086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 51), 'ver')
        # Applying the binary operator '+' (line 343)
        result_add_13087 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 29), '+', str_13085, ver_13086)
        
        # Assigning a type to the variable 'machine_action' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'machine_action', result_add_13087)
        
        # Assigning a BinOp to a Name (line 344):
        str_13088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 26), 'str', 'PythonFromUser')
        # Getting the type of 'ver' (line 344)
        ver_13089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 45), 'ver')
        # Applying the binary operator '+' (line 344)
        result_add_13090 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 26), '+', str_13088, ver_13089)
        
        # Assigning a type to the variable 'user_action' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'user_action', result_add_13090)
        
        # Assigning a BinOp to a Name (line 345):
        str_13091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 25), 'str', 'PythonExe')
        # Getting the type of 'ver' (line 345)
        ver_13092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 39), 'ver')
        # Applying the binary operator '+' (line 345)
        result_add_13093 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 25), '+', str_13091, ver_13092)
        
        # Assigning a type to the variable 'exe_action' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'exe_action', result_add_13093)
        
        # Assigning a BinOp to a Name (line 346):
        str_13094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 30), 'str', 'TARGETDIR')
        # Getting the type of 'ver' (line 346)
        ver_13095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 44), 'ver')
        # Applying the binary operator '+' (line 346)
        result_add_13096 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 30), '+', str_13094, ver_13095)
        
        # Assigning a type to the variable 'target_dir_prop' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'target_dir_prop', result_add_13096)
        
        # Assigning a BinOp to a Name (line 347):
        str_13097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 23), 'str', 'PYTHON')
        # Getting the type of 'ver' (line 347)
        ver_13098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 34), 'ver')
        # Applying the binary operator '+' (line 347)
        result_add_13099 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 23), '+', str_13097, ver_13098)
        
        # Assigning a type to the variable 'exe_prop' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'exe_prop', result_add_13099)
        
        # Getting the type of 'msilib' (line 348)
        msilib_13100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 15), 'msilib')
        # Obtaining the member 'Win64' of a type (line 348)
        Win64_13101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 15), msilib_13100, 'Win64')
        # Testing the type of an if condition (line 348)
        if_condition_13102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 12), Win64_13101)
        # Assigning a type to the variable 'if_condition_13102' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'if_condition_13102', if_condition_13102)
        # SSA begins for if statement (line 348)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 350):
        int_13103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 23), 'int')
        int_13104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 25), 'int')
        # Applying the binary operator '+' (line 350)
        result_add_13105 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 23), '+', int_13103, int_13104)
        
        # Assigning a type to the variable 'Type' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'Type', result_add_13105)
        # SSA branch for the else part of an if statement (line 348)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 352):
        int_13106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 23), 'int')
        # Assigning a type to the variable 'Type' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'Type', int_13106)
        # SSA join for if statement (line 348)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to add_data(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'self' (line 353)
        self_13108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 21), 'self', False)
        # Obtaining the member 'db' of a type (line 353)
        db_13109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 21), self_13108, 'db')
        str_13110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 30), 'str', 'RegLocator')
        
        # Obtaining an instance of the builtin type 'list' (line 354)
        list_13111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 354)
        # Adding element type (line 354)
        
        # Obtaining an instance of the builtin type 'tuple' (line 354)
        tuple_13112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 354)
        # Adding element type (line 354)
        # Getting the type of 'machine_reg' (line 354)
        machine_reg_13113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 22), 'machine_reg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 22), tuple_13112, machine_reg_13113)
        # Adding element type (line 354)
        int_13114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 22), tuple_13112, int_13114)
        # Adding element type (line 354)
        # Getting the type of 'install_path' (line 354)
        install_path_13115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 38), 'install_path', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 22), tuple_13112, install_path_13115)
        # Adding element type (line 354)
        # Getting the type of 'None' (line 354)
        None_13116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 52), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 22), tuple_13112, None_13116)
        # Adding element type (line 354)
        # Getting the type of 'Type' (line 354)
        Type_13117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 58), 'Type', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 22), tuple_13112, Type_13117)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 20), list_13111, tuple_13112)
        # Adding element type (line 354)
        
        # Obtaining an instance of the builtin type 'tuple' (line 355)
        tuple_13118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 355)
        # Adding element type (line 355)
        # Getting the type of 'user_reg' (line 355)
        user_reg_13119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 22), 'user_reg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), tuple_13118, user_reg_13119)
        # Adding element type (line 355)
        int_13120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), tuple_13118, int_13120)
        # Adding element type (line 355)
        # Getting the type of 'install_path' (line 355)
        install_path_13121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 35), 'install_path', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), tuple_13118, install_path_13121)
        # Adding element type (line 355)
        # Getting the type of 'None' (line 355)
        None_13122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 49), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), tuple_13118, None_13122)
        # Adding element type (line 355)
        # Getting the type of 'Type' (line 355)
        Type_13123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 55), 'Type', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), tuple_13118, Type_13123)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 20), list_13111, tuple_13118)
        
        # Processing the call keyword arguments (line 353)
        kwargs_13124 = {}
        # Getting the type of 'add_data' (line 353)
        add_data_13107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'add_data', False)
        # Calling add_data(args, kwargs) (line 353)
        add_data_call_result_13125 = invoke(stypy.reporting.localization.Localization(__file__, 353, 12), add_data_13107, *[db_13109, str_13110, list_13111], **kwargs_13124)
        
        
        # Call to add_data(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_13127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 21), 'self', False)
        # Obtaining the member 'db' of a type (line 356)
        db_13128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 21), self_13127, 'db')
        str_13129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 30), 'str', 'AppSearch')
        
        # Obtaining an instance of the builtin type 'list' (line 357)
        list_13130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 357)
        # Adding element type (line 357)
        
        # Obtaining an instance of the builtin type 'tuple' (line 357)
        tuple_13131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 357)
        # Adding element type (line 357)
        # Getting the type of 'machine_prop' (line 357)
        machine_prop_13132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 22), 'machine_prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 22), tuple_13131, machine_prop_13132)
        # Adding element type (line 357)
        # Getting the type of 'machine_reg' (line 357)
        machine_reg_13133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 36), 'machine_reg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 22), tuple_13131, machine_reg_13133)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 20), list_13130, tuple_13131)
        # Adding element type (line 357)
        
        # Obtaining an instance of the builtin type 'tuple' (line 358)
        tuple_13134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 358)
        # Adding element type (line 358)
        # Getting the type of 'user_prop' (line 358)
        user_prop_13135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 22), 'user_prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 22), tuple_13134, user_prop_13135)
        # Adding element type (line 358)
        # Getting the type of 'user_reg' (line 358)
        user_reg_13136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 33), 'user_reg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 22), tuple_13134, user_reg_13136)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 20), list_13130, tuple_13134)
        
        # Processing the call keyword arguments (line 356)
        kwargs_13137 = {}
        # Getting the type of 'add_data' (line 356)
        add_data_13126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'add_data', False)
        # Calling add_data(args, kwargs) (line 356)
        add_data_call_result_13138 = invoke(stypy.reporting.localization.Localization(__file__, 356, 12), add_data_13126, *[db_13128, str_13129, list_13130], **kwargs_13137)
        
        
        # Call to add_data(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'self' (line 359)
        self_13140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 21), 'self', False)
        # Obtaining the member 'db' of a type (line 359)
        db_13141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 21), self_13140, 'db')
        str_13142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 30), 'str', 'CustomAction')
        
        # Obtaining an instance of the builtin type 'list' (line 360)
        list_13143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 360)
        # Adding element type (line 360)
        
        # Obtaining an instance of the builtin type 'tuple' (line 360)
        tuple_13144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 360)
        # Adding element type (line 360)
        # Getting the type of 'machine_action' (line 360)
        machine_action_13145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 22), 'machine_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 22), tuple_13144, machine_action_13145)
        # Adding element type (line 360)
        int_13146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 38), 'int')
        int_13147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 41), 'int')
        # Applying the binary operator '+' (line 360)
        result_add_13148 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 38), '+', int_13146, int_13147)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 22), tuple_13144, result_add_13148)
        # Adding element type (line 360)
        # Getting the type of 'target_dir_prop' (line 360)
        target_dir_prop_13149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 46), 'target_dir_prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 22), tuple_13144, target_dir_prop_13149)
        # Adding element type (line 360)
        str_13150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 63), 'str', '[')
        # Getting the type of 'machine_prop' (line 360)
        machine_prop_13151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 69), 'machine_prop', False)
        # Applying the binary operator '+' (line 360)
        result_add_13152 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 63), '+', str_13150, machine_prop_13151)
        
        str_13153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 84), 'str', ']')
        # Applying the binary operator '+' (line 360)
        result_add_13154 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 82), '+', result_add_13152, str_13153)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 22), tuple_13144, result_add_13154)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 20), list_13143, tuple_13144)
        # Adding element type (line 360)
        
        # Obtaining an instance of the builtin type 'tuple' (line 361)
        tuple_13155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 361)
        # Adding element type (line 361)
        # Getting the type of 'user_action' (line 361)
        user_action_13156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 22), 'user_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 22), tuple_13155, user_action_13156)
        # Adding element type (line 361)
        int_13157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 35), 'int')
        int_13158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 38), 'int')
        # Applying the binary operator '+' (line 361)
        result_add_13159 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 35), '+', int_13157, int_13158)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 22), tuple_13155, result_add_13159)
        # Adding element type (line 361)
        # Getting the type of 'target_dir_prop' (line 361)
        target_dir_prop_13160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 43), 'target_dir_prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 22), tuple_13155, target_dir_prop_13160)
        # Adding element type (line 361)
        str_13161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 60), 'str', '[')
        # Getting the type of 'user_prop' (line 361)
        user_prop_13162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 66), 'user_prop', False)
        # Applying the binary operator '+' (line 361)
        result_add_13163 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 60), '+', str_13161, user_prop_13162)
        
        str_13164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 78), 'str', ']')
        # Applying the binary operator '+' (line 361)
        result_add_13165 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 76), '+', result_add_13163, str_13164)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 22), tuple_13155, result_add_13165)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 20), list_13143, tuple_13155)
        # Adding element type (line 360)
        
        # Obtaining an instance of the builtin type 'tuple' (line 362)
        tuple_13166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 362)
        # Adding element type (line 362)
        # Getting the type of 'exe_action' (line 362)
        exe_action_13167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 22), 'exe_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 22), tuple_13166, exe_action_13167)
        # Adding element type (line 362)
        int_13168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 34), 'int')
        int_13169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 37), 'int')
        # Applying the binary operator '+' (line 362)
        result_add_13170 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 34), '+', int_13168, int_13169)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 22), tuple_13166, result_add_13170)
        # Adding element type (line 362)
        # Getting the type of 'exe_prop' (line 362)
        exe_prop_13171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 42), 'exe_prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 22), tuple_13166, exe_prop_13171)
        # Adding element type (line 362)
        str_13172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 52), 'str', '[')
        # Getting the type of 'target_dir_prop' (line 362)
        target_dir_prop_13173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 58), 'target_dir_prop', False)
        # Applying the binary operator '+' (line 362)
        result_add_13174 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 52), '+', str_13172, target_dir_prop_13173)
        
        str_13175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 76), 'str', ']\\python.exe')
        # Applying the binary operator '+' (line 362)
        result_add_13176 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 74), '+', result_add_13174, str_13175)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 22), tuple_13166, result_add_13176)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 20), list_13143, tuple_13166)
        
        # Processing the call keyword arguments (line 359)
        kwargs_13177 = {}
        # Getting the type of 'add_data' (line 359)
        add_data_13139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'add_data', False)
        # Calling add_data(args, kwargs) (line 359)
        add_data_call_result_13178 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), add_data_13139, *[db_13141, str_13142, list_13143], **kwargs_13177)
        
        
        # Call to add_data(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'self' (line 364)
        self_13180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 21), 'self', False)
        # Obtaining the member 'db' of a type (line 364)
        db_13181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 21), self_13180, 'db')
        str_13182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 30), 'str', 'InstallExecuteSequence')
        
        # Obtaining an instance of the builtin type 'list' (line 365)
        list_13183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 365)
        # Adding element type (line 365)
        
        # Obtaining an instance of the builtin type 'tuple' (line 365)
        tuple_13184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 365)
        # Adding element type (line 365)
        # Getting the type of 'machine_action' (line 365)
        machine_action_13185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 22), 'machine_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 22), tuple_13184, machine_action_13185)
        # Adding element type (line 365)
        # Getting the type of 'machine_prop' (line 365)
        machine_prop_13186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 38), 'machine_prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 22), tuple_13184, machine_prop_13186)
        # Adding element type (line 365)
        # Getting the type of 'start' (line 365)
        start_13187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 52), 'start', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 22), tuple_13184, start_13187)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 20), list_13183, tuple_13184)
        # Adding element type (line 365)
        
        # Obtaining an instance of the builtin type 'tuple' (line 366)
        tuple_13188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 366)
        # Adding element type (line 366)
        # Getting the type of 'user_action' (line 366)
        user_action_13189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'user_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 22), tuple_13188, user_action_13189)
        # Adding element type (line 366)
        # Getting the type of 'user_prop' (line 366)
        user_prop_13190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 35), 'user_prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 22), tuple_13188, user_prop_13190)
        # Adding element type (line 366)
        # Getting the type of 'start' (line 366)
        start_13191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 46), 'start', False)
        int_13192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 54), 'int')
        # Applying the binary operator '+' (line 366)
        result_add_13193 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 46), '+', start_13191, int_13192)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 22), tuple_13188, result_add_13193)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 20), list_13183, tuple_13188)
        # Adding element type (line 365)
        
        # Obtaining an instance of the builtin type 'tuple' (line 367)
        tuple_13194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 367)
        # Adding element type (line 367)
        # Getting the type of 'exe_action' (line 367)
        exe_action_13195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), 'exe_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), tuple_13194, exe_action_13195)
        # Adding element type (line 367)
        # Getting the type of 'None' (line 367)
        None_13196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 34), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), tuple_13194, None_13196)
        # Adding element type (line 367)
        # Getting the type of 'start' (line 367)
        start_13197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 40), 'start', False)
        int_13198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 48), 'int')
        # Applying the binary operator '+' (line 367)
        result_add_13199 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 40), '+', start_13197, int_13198)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), tuple_13194, result_add_13199)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 20), list_13183, tuple_13194)
        
        # Processing the call keyword arguments (line 364)
        kwargs_13200 = {}
        # Getting the type of 'add_data' (line 364)
        add_data_13179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'add_data', False)
        # Calling add_data(args, kwargs) (line 364)
        add_data_call_result_13201 = invoke(stypy.reporting.localization.Localization(__file__, 364, 12), add_data_13179, *[db_13181, str_13182, list_13183], **kwargs_13200)
        
        
        # Call to add_data(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'self' (line 369)
        self_13203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'self', False)
        # Obtaining the member 'db' of a type (line 369)
        db_13204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), self_13203, 'db')
        str_13205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 30), 'str', 'InstallUISequence')
        
        # Obtaining an instance of the builtin type 'list' (line 370)
        list_13206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 370)
        # Adding element type (line 370)
        
        # Obtaining an instance of the builtin type 'tuple' (line 370)
        tuple_13207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 370)
        # Adding element type (line 370)
        # Getting the type of 'machine_action' (line 370)
        machine_action_13208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 22), 'machine_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 22), tuple_13207, machine_action_13208)
        # Adding element type (line 370)
        # Getting the type of 'machine_prop' (line 370)
        machine_prop_13209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 38), 'machine_prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 22), tuple_13207, machine_prop_13209)
        # Adding element type (line 370)
        # Getting the type of 'start' (line 370)
        start_13210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 52), 'start', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 22), tuple_13207, start_13210)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 20), list_13206, tuple_13207)
        # Adding element type (line 370)
        
        # Obtaining an instance of the builtin type 'tuple' (line 371)
        tuple_13211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 371)
        # Adding element type (line 371)
        # Getting the type of 'user_action' (line 371)
        user_action_13212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 22), 'user_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 22), tuple_13211, user_action_13212)
        # Adding element type (line 371)
        # Getting the type of 'user_prop' (line 371)
        user_prop_13213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 35), 'user_prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 22), tuple_13211, user_prop_13213)
        # Adding element type (line 371)
        # Getting the type of 'start' (line 371)
        start_13214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 46), 'start', False)
        int_13215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 54), 'int')
        # Applying the binary operator '+' (line 371)
        result_add_13216 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 46), '+', start_13214, int_13215)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 22), tuple_13211, result_add_13216)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 20), list_13206, tuple_13211)
        # Adding element type (line 370)
        
        # Obtaining an instance of the builtin type 'tuple' (line 372)
        tuple_13217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 372)
        # Adding element type (line 372)
        # Getting the type of 'exe_action' (line 372)
        exe_action_13218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'exe_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 22), tuple_13217, exe_action_13218)
        # Adding element type (line 372)
        # Getting the type of 'None' (line 372)
        None_13219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 34), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 22), tuple_13217, None_13219)
        # Adding element type (line 372)
        # Getting the type of 'start' (line 372)
        start_13220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 40), 'start', False)
        int_13221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 48), 'int')
        # Applying the binary operator '+' (line 372)
        result_add_13222 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 40), '+', start_13220, int_13221)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 22), tuple_13217, result_add_13222)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 20), list_13206, tuple_13217)
        
        # Processing the call keyword arguments (line 369)
        kwargs_13223 = {}
        # Getting the type of 'add_data' (line 369)
        add_data_13202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'add_data', False)
        # Calling add_data(args, kwargs) (line 369)
        add_data_call_result_13224 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), add_data_13202, *[db_13204, str_13205, list_13206], **kwargs_13223)
        
        
        # Call to add_data(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'self' (line 374)
        self_13226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 21), 'self', False)
        # Obtaining the member 'db' of a type (line 374)
        db_13227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 21), self_13226, 'db')
        str_13228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 30), 'str', 'Condition')
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_13229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        
        # Obtaining an instance of the builtin type 'tuple' (line 375)
        tuple_13230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 375)
        # Adding element type (line 375)
        str_13231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 22), 'str', 'Python')
        # Getting the type of 'ver' (line 375)
        ver_13232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 33), 'ver', False)
        # Applying the binary operator '+' (line 375)
        result_add_13233 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 22), '+', str_13231, ver_13232)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 22), tuple_13230, result_add_13233)
        # Adding element type (line 375)
        int_13234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 22), tuple_13230, int_13234)
        # Adding element type (line 375)
        str_13235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 41), 'str', 'NOT TARGETDIR')
        # Getting the type of 'ver' (line 375)
        ver_13236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 59), 'ver', False)
        # Applying the binary operator '+' (line 375)
        result_add_13237 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 41), '+', str_13235, ver_13236)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 22), tuple_13230, result_add_13237)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 20), list_13229, tuple_13230)
        
        # Processing the call keyword arguments (line 374)
        kwargs_13238 = {}
        # Getting the type of 'add_data' (line 374)
        add_data_13225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'add_data', False)
        # Calling add_data(args, kwargs) (line 374)
        add_data_call_result_13239 = invoke(stypy.reporting.localization.Localization(__file__, 374, 12), add_data_13225, *[db_13227, str_13228, list_13229], **kwargs_13238)
        
        
        # Getting the type of 'start' (line 376)
        start_13240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'start')
        int_13241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 21), 'int')
        # Applying the binary operator '+=' (line 376)
        result_iadd_13242 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 12), '+=', start_13240, int_13241)
        # Assigning a type to the variable 'start' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'start', result_iadd_13242)
        
        # Evaluating assert statement condition
        
        # Getting the type of 'start' (line 377)
        start_13243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'start')
        int_13244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 27), 'int')
        # Applying the binary operator '<' (line 377)
        result_lt_13245 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 19), '<', start_13243, int_13244)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'add_find_python(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_find_python' in the type store
        # Getting the type of 'stypy_return_type' (line 325)
        stypy_return_type_13246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13246)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_find_python'
        return stypy_return_type_13246


    @norecursion
    def add_scripts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_scripts'
        module_type_store = module_type_store.open_function_context('add_scripts', 379, 4, False)
        # Assigning a type to the variable 'self' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_localization', localization)
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_function_name', 'bdist_msi.add_scripts')
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_msi.add_scripts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_msi.add_scripts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_scripts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_scripts(...)' code ##################

        
        # Getting the type of 'self' (line 380)
        self_13247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 11), 'self')
        # Obtaining the member 'install_script' of a type (line 380)
        install_script_13248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 11), self_13247, 'install_script')
        # Testing the type of an if condition (line 380)
        if_condition_13249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 8), install_script_13248)
        # Assigning a type to the variable 'if_condition_13249' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'if_condition_13249', if_condition_13249)
        # SSA begins for if statement (line 380)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 381):
        int_13250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 20), 'int')
        # Assigning a type to the variable 'start' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'start', int_13250)
        
        # Getting the type of 'self' (line 382)
        self_13251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'self')
        # Obtaining the member 'versions' of a type (line 382)
        versions_13252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 23), self_13251, 'versions')
        
        # Obtaining an instance of the builtin type 'list' (line 382)
        list_13253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 382)
        # Adding element type (line 382)
        # Getting the type of 'self' (line 382)
        self_13254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 40), 'self')
        # Obtaining the member 'other_version' of a type (line 382)
        other_version_13255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 40), self_13254, 'other_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 39), list_13253, other_version_13255)
        
        # Applying the binary operator '+' (line 382)
        result_add_13256 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 23), '+', versions_13252, list_13253)
        
        # Testing the type of a for loop iterable (line 382)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 382, 12), result_add_13256)
        # Getting the type of the for loop variable (line 382)
        for_loop_var_13257 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 382, 12), result_add_13256)
        # Assigning a type to the variable 'ver' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'ver', for_loop_var_13257)
        # SSA begins for a for statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 383):
        str_13258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 33), 'str', 'install_script.')
        # Getting the type of 'ver' (line 383)
        ver_13259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 53), 'ver')
        # Applying the binary operator '+' (line 383)
        result_add_13260 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 33), '+', str_13258, ver_13259)
        
        # Assigning a type to the variable 'install_action' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'install_action', result_add_13260)
        
        # Assigning a BinOp to a Name (line 384):
        str_13261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 27), 'str', 'PYTHON')
        # Getting the type of 'ver' (line 384)
        ver_13262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 38), 'ver')
        # Applying the binary operator '+' (line 384)
        result_add_13263 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 27), '+', str_13261, ver_13262)
        
        # Assigning a type to the variable 'exe_prop' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'exe_prop', result_add_13263)
        
        # Call to add_data(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'self' (line 385)
        self_13265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 25), 'self', False)
        # Obtaining the member 'db' of a type (line 385)
        db_13266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 25), self_13265, 'db')
        str_13267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 34), 'str', 'CustomAction')
        
        # Obtaining an instance of the builtin type 'list' (line 386)
        list_13268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 386)
        # Adding element type (line 386)
        
        # Obtaining an instance of the builtin type 'tuple' (line 386)
        tuple_13269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 386)
        # Adding element type (line 386)
        # Getting the type of 'install_action' (line 386)
        install_action_13270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 26), 'install_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 26), tuple_13269, install_action_13270)
        # Adding element type (line 386)
        int_13271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 26), tuple_13269, int_13271)
        # Adding element type (line 386)
        # Getting the type of 'exe_prop' (line 386)
        exe_prop_13272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 46), 'exe_prop', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 26), tuple_13269, exe_prop_13272)
        # Adding element type (line 386)
        # Getting the type of 'self' (line 386)
        self_13273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 56), 'self', False)
        # Obtaining the member 'install_script_key' of a type (line 386)
        install_script_key_13274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 56), self_13273, 'install_script_key')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 26), tuple_13269, install_script_key_13274)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 24), list_13268, tuple_13269)
        
        # Processing the call keyword arguments (line 385)
        kwargs_13275 = {}
        # Getting the type of 'add_data' (line 385)
        add_data_13264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'add_data', False)
        # Calling add_data(args, kwargs) (line 385)
        add_data_call_result_13276 = invoke(stypy.reporting.localization.Localization(__file__, 385, 16), add_data_13264, *[db_13266, str_13267, list_13268], **kwargs_13275)
        
        
        # Call to add_data(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'self' (line 387)
        self_13278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 25), 'self', False)
        # Obtaining the member 'db' of a type (line 387)
        db_13279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 25), self_13278, 'db')
        str_13280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 34), 'str', 'InstallExecuteSequence')
        
        # Obtaining an instance of the builtin type 'list' (line 388)
        list_13281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 388)
        # Adding element type (line 388)
        
        # Obtaining an instance of the builtin type 'tuple' (line 388)
        tuple_13282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 388)
        # Adding element type (line 388)
        # Getting the type of 'install_action' (line 388)
        install_action_13283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 26), 'install_action', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 26), tuple_13282, install_action_13283)
        # Adding element type (line 388)
        str_13284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 42), 'str', '&Python%s=3')
        # Getting the type of 'ver' (line 388)
        ver_13285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 58), 'ver', False)
        # Applying the binary operator '%' (line 388)
        result_mod_13286 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 42), '%', str_13284, ver_13285)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 26), tuple_13282, result_mod_13286)
        # Adding element type (line 388)
        # Getting the type of 'start' (line 388)
        start_13287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 63), 'start', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 26), tuple_13282, start_13287)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 24), list_13281, tuple_13282)
        
        # Processing the call keyword arguments (line 387)
        kwargs_13288 = {}
        # Getting the type of 'add_data' (line 387)
        add_data_13277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'add_data', False)
        # Calling add_data(args, kwargs) (line 387)
        add_data_call_result_13289 = invoke(stypy.reporting.localization.Localization(__file__, 387, 16), add_data_13277, *[db_13279, str_13280, list_13281], **kwargs_13288)
        
        
        # Getting the type of 'start' (line 389)
        start_13290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'start')
        int_13291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 25), 'int')
        # Applying the binary operator '+=' (line 389)
        result_iadd_13292 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 16), '+=', start_13290, int_13291)
        # Assigning a type to the variable 'start' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'start', result_iadd_13292)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 380)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 393)
        self_13293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 11), 'self')
        # Obtaining the member 'pre_install_script' of a type (line 393)
        pre_install_script_13294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 11), self_13293, 'pre_install_script')
        # Testing the type of an if condition (line 393)
        if_condition_13295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 8), pre_install_script_13294)
        # Assigning a type to the variable 'if_condition_13295' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'if_condition_13295', if_condition_13295)
        # SSA begins for if statement (line 393)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 394):
        
        # Call to join(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'self' (line 394)
        self_13299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 36), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 394)
        bdist_dir_13300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 36), self_13299, 'bdist_dir')
        str_13301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 52), 'str', 'preinstall.bat')
        # Processing the call keyword arguments (line 394)
        kwargs_13302 = {}
        # Getting the type of 'os' (line 394)
        os_13296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 394)
        path_13297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 23), os_13296, 'path')
        # Obtaining the member 'join' of a type (line 394)
        join_13298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 23), path_13297, 'join')
        # Calling join(args, kwargs) (line 394)
        join_call_result_13303 = invoke(stypy.reporting.localization.Localization(__file__, 394, 23), join_13298, *[bdist_dir_13300, str_13301], **kwargs_13302)
        
        # Assigning a type to the variable 'scriptfn' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'scriptfn', join_call_result_13303)
        
        # Assigning a Call to a Name (line 395):
        
        # Call to open(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'scriptfn' (line 395)
        scriptfn_13305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 21), 'scriptfn', False)
        str_13306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 31), 'str', 'w')
        # Processing the call keyword arguments (line 395)
        kwargs_13307 = {}
        # Getting the type of 'open' (line 395)
        open_13304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'open', False)
        # Calling open(args, kwargs) (line 395)
        open_call_result_13308 = invoke(stypy.reporting.localization.Localization(__file__, 395, 16), open_13304, *[scriptfn_13305, str_13306], **kwargs_13307)
        
        # Assigning a type to the variable 'f' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'f', open_call_result_13308)
        
        # Call to write(...): (line 404)
        # Processing the call arguments (line 404)
        str_13311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 20), 'str', 'rem ="""\n%1 %0\nexit\n"""\n')
        # Processing the call keyword arguments (line 404)
        kwargs_13312 = {}
        # Getting the type of 'f' (line 404)
        f_13309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 404)
        write_13310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 12), f_13309, 'write')
        # Calling write(args, kwargs) (line 404)
        write_call_result_13313 = invoke(stypy.reporting.localization.Localization(__file__, 404, 12), write_13310, *[str_13311], **kwargs_13312)
        
        
        # Call to write(...): (line 405)
        # Processing the call arguments (line 405)
        
        # Call to read(...): (line 405)
        # Processing the call keyword arguments (line 405)
        kwargs_13322 = {}
        
        # Call to open(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'self' (line 405)
        self_13317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'self', False)
        # Obtaining the member 'pre_install_script' of a type (line 405)
        pre_install_script_13318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 25), self_13317, 'pre_install_script')
        # Processing the call keyword arguments (line 405)
        kwargs_13319 = {}
        # Getting the type of 'open' (line 405)
        open_13316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 20), 'open', False)
        # Calling open(args, kwargs) (line 405)
        open_call_result_13320 = invoke(stypy.reporting.localization.Localization(__file__, 405, 20), open_13316, *[pre_install_script_13318], **kwargs_13319)
        
        # Obtaining the member 'read' of a type (line 405)
        read_13321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 20), open_call_result_13320, 'read')
        # Calling read(args, kwargs) (line 405)
        read_call_result_13323 = invoke(stypy.reporting.localization.Localization(__file__, 405, 20), read_13321, *[], **kwargs_13322)
        
        # Processing the call keyword arguments (line 405)
        kwargs_13324 = {}
        # Getting the type of 'f' (line 405)
        f_13314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 405)
        write_13315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 12), f_13314, 'write')
        # Calling write(args, kwargs) (line 405)
        write_call_result_13325 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), write_13315, *[read_call_result_13323], **kwargs_13324)
        
        
        # Call to close(...): (line 406)
        # Processing the call keyword arguments (line 406)
        kwargs_13328 = {}
        # Getting the type of 'f' (line 406)
        f_13326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 406)
        close_13327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 12), f_13326, 'close')
        # Calling close(args, kwargs) (line 406)
        close_call_result_13329 = invoke(stypy.reporting.localization.Localization(__file__, 406, 12), close_13327, *[], **kwargs_13328)
        
        
        # Call to add_data(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'self' (line 407)
        self_13331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'self', False)
        # Obtaining the member 'db' of a type (line 407)
        db_13332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 21), self_13331, 'db')
        str_13333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 30), 'str', 'Binary')
        
        # Obtaining an instance of the builtin type 'list' (line 408)
        list_13334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 408)
        # Adding element type (line 408)
        
        # Obtaining an instance of the builtin type 'tuple' (line 408)
        tuple_13335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 408)
        # Adding element type (line 408)
        str_13336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 18), 'str', 'PreInstall')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 18), tuple_13335, str_13336)
        # Adding element type (line 408)
        
        # Call to Binary(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'scriptfn' (line 408)
        scriptfn_13339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 46), 'scriptfn', False)
        # Processing the call keyword arguments (line 408)
        kwargs_13340 = {}
        # Getting the type of 'msilib' (line 408)
        msilib_13337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 32), 'msilib', False)
        # Obtaining the member 'Binary' of a type (line 408)
        Binary_13338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 32), msilib_13337, 'Binary')
        # Calling Binary(args, kwargs) (line 408)
        Binary_call_result_13341 = invoke(stypy.reporting.localization.Localization(__file__, 408, 32), Binary_13338, *[scriptfn_13339], **kwargs_13340)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 18), tuple_13335, Binary_call_result_13341)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 16), list_13334, tuple_13335)
        
        # Processing the call keyword arguments (line 407)
        kwargs_13342 = {}
        # Getting the type of 'add_data' (line 407)
        add_data_13330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'add_data', False)
        # Calling add_data(args, kwargs) (line 407)
        add_data_call_result_13343 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), add_data_13330, *[db_13332, str_13333, list_13334], **kwargs_13342)
        
        
        # Call to add_data(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'self' (line 410)
        self_13345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 21), 'self', False)
        # Obtaining the member 'db' of a type (line 410)
        db_13346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 21), self_13345, 'db')
        str_13347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 30), 'str', 'CustomAction')
        
        # Obtaining an instance of the builtin type 'list' (line 411)
        list_13348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 411)
        # Adding element type (line 411)
        
        # Obtaining an instance of the builtin type 'tuple' (line 411)
        tuple_13349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 411)
        # Adding element type (line 411)
        str_13350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 18), 'str', 'PreInstall')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 18), tuple_13349, str_13350)
        # Adding element type (line 411)
        int_13351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 18), tuple_13349, int_13351)
        # Adding element type (line 411)
        str_13352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 35), 'str', 'PreInstall')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 18), tuple_13349, str_13352)
        # Adding element type (line 411)
        # Getting the type of 'None' (line 411)
        None_13353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 49), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 18), tuple_13349, None_13353)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 16), list_13348, tuple_13349)
        
        # Processing the call keyword arguments (line 410)
        kwargs_13354 = {}
        # Getting the type of 'add_data' (line 410)
        add_data_13344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'add_data', False)
        # Calling add_data(args, kwargs) (line 410)
        add_data_call_result_13355 = invoke(stypy.reporting.localization.Localization(__file__, 410, 12), add_data_13344, *[db_13346, str_13347, list_13348], **kwargs_13354)
        
        
        # Call to add_data(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'self' (line 413)
        self_13357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 21), 'self', False)
        # Obtaining the member 'db' of a type (line 413)
        db_13358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 21), self_13357, 'db')
        str_13359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 30), 'str', 'InstallExecuteSequence')
        
        # Obtaining an instance of the builtin type 'list' (line 414)
        list_13360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 414)
        # Adding element type (line 414)
        
        # Obtaining an instance of the builtin type 'tuple' (line 414)
        tuple_13361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 414)
        # Adding element type (line 414)
        str_13362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 22), 'str', 'PreInstall')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 22), tuple_13361, str_13362)
        # Adding element type (line 414)
        str_13363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 36), 'str', 'NOT Installed')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 22), tuple_13361, str_13363)
        # Adding element type (line 414)
        int_13364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 22), tuple_13361, int_13364)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 20), list_13360, tuple_13361)
        
        # Processing the call keyword arguments (line 413)
        kwargs_13365 = {}
        # Getting the type of 'add_data' (line 413)
        add_data_13356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'add_data', False)
        # Calling add_data(args, kwargs) (line 413)
        add_data_call_result_13366 = invoke(stypy.reporting.localization.Localization(__file__, 413, 12), add_data_13356, *[db_13358, str_13359, list_13360], **kwargs_13365)
        
        # SSA join for if statement (line 393)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'add_scripts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_scripts' in the type store
        # Getting the type of 'stypy_return_type' (line 379)
        stypy_return_type_13367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13367)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_scripts'
        return stypy_return_type_13367


    @norecursion
    def add_ui(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_ui'
        module_type_store = module_type_store.open_function_context('add_ui', 417, 4, False)
        # Assigning a type to the variable 'self' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_msi.add_ui.__dict__.__setitem__('stypy_localization', localization)
        bdist_msi.add_ui.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_msi.add_ui.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_msi.add_ui.__dict__.__setitem__('stypy_function_name', 'bdist_msi.add_ui')
        bdist_msi.add_ui.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_msi.add_ui.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_msi.add_ui.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_msi.add_ui.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_msi.add_ui.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_msi.add_ui.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_msi.add_ui.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_msi.add_ui', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_ui', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_ui(...)' code ##################

        
        # Assigning a Attribute to a Name (line 418):
        # Getting the type of 'self' (line 418)
        self_13368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 13), 'self')
        # Obtaining the member 'db' of a type (line 418)
        db_13369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 13), self_13368, 'db')
        # Assigning a type to the variable 'db' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'db', db_13369)
        
        # Multiple assignment of 2 elements.
        int_13370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 16), 'int')
        # Assigning a type to the variable 'y' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'y', int_13370)
        # Getting the type of 'y' (line 419)
        y_13371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'y')
        # Assigning a type to the variable 'x' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'x', y_13371)
        
        # Assigning a Num to a Name (line 420):
        int_13372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 12), 'int')
        # Assigning a type to the variable 'w' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'w', int_13372)
        
        # Assigning a Num to a Name (line 421):
        int_13373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 12), 'int')
        # Assigning a type to the variable 'h' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'h', int_13373)
        
        # Assigning a Str to a Name (line 422):
        str_13374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 16), 'str', '[ProductName] Setup')
        # Assigning a type to the variable 'title' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'title', str_13374)
        
        # Assigning a Num to a Name (line 425):
        int_13375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 16), 'int')
        # Assigning a type to the variable 'modal' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'modal', int_13375)
        
        # Assigning a Num to a Name (line 426):
        int_13376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 19), 'int')
        # Assigning a type to the variable 'modeless' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'modeless', int_13376)
        
        # Call to add_data(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'db' (line 429)
        db_13378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 17), 'db', False)
        str_13379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 21), 'str', 'Property')
        
        # Obtaining an instance of the builtin type 'list' (line 431)
        list_13380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 431)
        # Adding element type (line 431)
        
        # Obtaining an instance of the builtin type 'tuple' (line 431)
        tuple_13381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 431)
        # Adding element type (line 431)
        str_13382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 19), 'str', 'DefaultUIFont')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 19), tuple_13381, str_13382)
        # Adding element type (line 431)
        str_13383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 36), 'str', 'DlgFont8')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 19), tuple_13381, str_13383)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 17), list_13380, tuple_13381)
        # Adding element type (line 431)
        
        # Obtaining an instance of the builtin type 'tuple' (line 433)
        tuple_13384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 433)
        # Adding element type (line 433)
        str_13385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 19), 'str', 'ErrorDialog')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 19), tuple_13384, str_13385)
        # Adding element type (line 433)
        str_13386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 34), 'str', 'ErrorDlg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 19), tuple_13384, str_13386)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 17), list_13380, tuple_13384)
        # Adding element type (line 431)
        
        # Obtaining an instance of the builtin type 'tuple' (line 434)
        tuple_13387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 434)
        # Adding element type (line 434)
        str_13388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 19), 'str', 'Progress1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 19), tuple_13387, str_13388)
        # Adding element type (line 434)
        str_13389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 32), 'str', 'Install')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 19), tuple_13387, str_13389)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 17), list_13380, tuple_13387)
        # Adding element type (line 431)
        
        # Obtaining an instance of the builtin type 'tuple' (line 435)
        tuple_13390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 435)
        # Adding element type (line 435)
        str_13391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 19), 'str', 'Progress2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 19), tuple_13390, str_13391)
        # Adding element type (line 435)
        str_13392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 32), 'str', 'installs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 19), tuple_13390, str_13392)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 17), list_13380, tuple_13390)
        # Adding element type (line 431)
        
        # Obtaining an instance of the builtin type 'tuple' (line 436)
        tuple_13393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 436)
        # Adding element type (line 436)
        str_13394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 19), 'str', 'MaintenanceForm_Action')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 19), tuple_13393, str_13394)
        # Adding element type (line 436)
        str_13395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 45), 'str', 'Repair')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 19), tuple_13393, str_13395)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 17), list_13380, tuple_13393)
        # Adding element type (line 431)
        
        # Obtaining an instance of the builtin type 'tuple' (line 438)
        tuple_13396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 438)
        # Adding element type (line 438)
        str_13397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 19), 'str', 'WhichUsers')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 19), tuple_13396, str_13397)
        # Adding element type (line 438)
        str_13398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 33), 'str', 'ALL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 19), tuple_13396, str_13398)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 17), list_13380, tuple_13396)
        
        # Processing the call keyword arguments (line 429)
        kwargs_13399 = {}
        # Getting the type of 'add_data' (line 429)
        add_data_13377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'add_data', False)
        # Calling add_data(args, kwargs) (line 429)
        add_data_call_result_13400 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), add_data_13377, *[db_13378, str_13379, list_13380], **kwargs_13399)
        
        
        # Call to add_data(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'db' (line 442)
        db_13402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 17), 'db', False)
        str_13403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 21), 'str', 'TextStyle')
        
        # Obtaining an instance of the builtin type 'list' (line 443)
        list_13404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 443)
        # Adding element type (line 443)
        
        # Obtaining an instance of the builtin type 'tuple' (line 443)
        tuple_13405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 443)
        # Adding element type (line 443)
        str_13406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 19), 'str', 'DlgFont8')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 19), tuple_13405, str_13406)
        # Adding element type (line 443)
        str_13407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 31), 'str', 'Tahoma')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 19), tuple_13405, str_13407)
        # Adding element type (line 443)
        int_13408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 19), tuple_13405, int_13408)
        # Adding element type (line 443)
        # Getting the type of 'None' (line 443)
        None_13409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 44), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 19), tuple_13405, None_13409)
        # Adding element type (line 443)
        int_13410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 19), tuple_13405, int_13410)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 17), list_13404, tuple_13405)
        # Adding element type (line 443)
        
        # Obtaining an instance of the builtin type 'tuple' (line 444)
        tuple_13411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 444)
        # Adding element type (line 444)
        str_13412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 19), 'str', 'DlgFontBold8')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 19), tuple_13411, str_13412)
        # Adding element type (line 444)
        str_13413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 35), 'str', 'Tahoma')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 19), tuple_13411, str_13413)
        # Adding element type (line 444)
        int_13414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 19), tuple_13411, int_13414)
        # Adding element type (line 444)
        # Getting the type of 'None' (line 444)
        None_13415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 48), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 19), tuple_13411, None_13415)
        # Adding element type (line 444)
        int_13416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 19), tuple_13411, int_13416)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 17), list_13404, tuple_13411)
        # Adding element type (line 443)
        
        # Obtaining an instance of the builtin type 'tuple' (line 445)
        tuple_13417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 445)
        # Adding element type (line 445)
        str_13418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 19), 'str', 'VerdanaBold10')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 19), tuple_13417, str_13418)
        # Adding element type (line 445)
        str_13419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 36), 'str', 'Verdana')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 19), tuple_13417, str_13419)
        # Adding element type (line 445)
        int_13420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 19), tuple_13417, int_13420)
        # Adding element type (line 445)
        # Getting the type of 'None' (line 445)
        None_13421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 51), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 19), tuple_13417, None_13421)
        # Adding element type (line 445)
        int_13422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 19), tuple_13417, int_13422)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 17), list_13404, tuple_13417)
        # Adding element type (line 443)
        
        # Obtaining an instance of the builtin type 'tuple' (line 446)
        tuple_13423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 446)
        # Adding element type (line 446)
        str_13424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 19), 'str', 'VerdanaRed9')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 19), tuple_13423, str_13424)
        # Adding element type (line 446)
        str_13425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 34), 'str', 'Verdana')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 19), tuple_13423, str_13425)
        # Adding element type (line 446)
        int_13426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 19), tuple_13423, int_13426)
        # Adding element type (line 446)
        int_13427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 19), tuple_13423, int_13427)
        # Adding element type (line 446)
        int_13428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 19), tuple_13423, int_13428)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 17), list_13404, tuple_13423)
        
        # Processing the call keyword arguments (line 442)
        kwargs_13429 = {}
        # Getting the type of 'add_data' (line 442)
        add_data_13401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'add_data', False)
        # Calling add_data(args, kwargs) (line 442)
        add_data_call_result_13430 = invoke(stypy.reporting.localization.Localization(__file__, 442, 8), add_data_13401, *[db_13402, str_13403, list_13404], **kwargs_13429)
        
        
        # Call to add_data(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'db' (line 451)
        db_13432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 17), 'db', False)
        str_13433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 21), 'str', 'InstallUISequence')
        
        # Obtaining an instance of the builtin type 'list' (line 452)
        list_13434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 452)
        # Adding element type (line 452)
        
        # Obtaining an instance of the builtin type 'tuple' (line 452)
        tuple_13435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 452)
        # Adding element type (line 452)
        str_13436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 19), 'str', 'PrepareDlg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 19), tuple_13435, str_13436)
        # Adding element type (line 452)
        str_13437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 33), 'str', 'Not Privileged or Windows9x or Installed')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 19), tuple_13435, str_13437)
        # Adding element type (line 452)
        int_13438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 77), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 19), tuple_13435, int_13438)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 17), list_13434, tuple_13435)
        # Adding element type (line 452)
        
        # Obtaining an instance of the builtin type 'tuple' (line 453)
        tuple_13439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 453)
        # Adding element type (line 453)
        str_13440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 19), 'str', 'WhichUsersDlg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 19), tuple_13439, str_13440)
        # Adding element type (line 453)
        str_13441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 36), 'str', 'Privileged and not Windows9x and not Installed')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 19), tuple_13439, str_13441)
        # Adding element type (line 453)
        int_13442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 86), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 19), tuple_13439, int_13442)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 17), list_13434, tuple_13439)
        # Adding element type (line 452)
        
        # Obtaining an instance of the builtin type 'tuple' (line 455)
        tuple_13443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 455)
        # Adding element type (line 455)
        str_13444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 19), 'str', 'SelectFeaturesDlg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 19), tuple_13443, str_13444)
        # Adding element type (line 455)
        str_13445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 40), 'str', 'Not Installed')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 19), tuple_13443, str_13445)
        # Adding element type (line 455)
        int_13446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 19), tuple_13443, int_13446)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 17), list_13434, tuple_13443)
        # Adding element type (line 452)
        
        # Obtaining an instance of the builtin type 'tuple' (line 458)
        tuple_13447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 458)
        # Adding element type (line 458)
        str_13448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 19), 'str', 'MaintenanceTypeDlg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 19), tuple_13447, str_13448)
        # Adding element type (line 458)
        str_13449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 41), 'str', 'Installed AND NOT RESUME AND NOT Preselected')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 19), tuple_13447, str_13449)
        # Adding element type (line 458)
        int_13450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 89), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 19), tuple_13447, int_13450)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 17), list_13434, tuple_13447)
        # Adding element type (line 452)
        
        # Obtaining an instance of the builtin type 'tuple' (line 459)
        tuple_13451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 459)
        # Adding element type (line 459)
        str_13452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 19), 'str', 'ProgressDlg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 19), tuple_13451, str_13452)
        # Adding element type (line 459)
        # Getting the type of 'None' (line 459)
        None_13453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 34), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 19), tuple_13451, None_13453)
        # Adding element type (line 459)
        int_13454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 19), tuple_13451, int_13454)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 17), list_13434, tuple_13451)
        
        # Processing the call keyword arguments (line 451)
        kwargs_13455 = {}
        # Getting the type of 'add_data' (line 451)
        add_data_13431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'add_data', False)
        # Calling add_data(args, kwargs) (line 451)
        add_data_call_result_13456 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), add_data_13431, *[db_13432, str_13433, list_13434], **kwargs_13455)
        
        
        # Call to add_data(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'db' (line 461)
        db_13458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 17), 'db', False)
        str_13459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 21), 'str', 'ActionText')
        # Getting the type of 'text' (line 461)
        text_13460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 35), 'text', False)
        # Obtaining the member 'ActionText' of a type (line 461)
        ActionText_13461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 35), text_13460, 'ActionText')
        # Processing the call keyword arguments (line 461)
        kwargs_13462 = {}
        # Getting the type of 'add_data' (line 461)
        add_data_13457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'add_data', False)
        # Calling add_data(args, kwargs) (line 461)
        add_data_call_result_13463 = invoke(stypy.reporting.localization.Localization(__file__, 461, 8), add_data_13457, *[db_13458, str_13459, ActionText_13461], **kwargs_13462)
        
        
        # Call to add_data(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'db' (line 462)
        db_13465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 17), 'db', False)
        str_13466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 21), 'str', 'UIText')
        # Getting the type of 'text' (line 462)
        text_13467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 31), 'text', False)
        # Obtaining the member 'UIText' of a type (line 462)
        UIText_13468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 31), text_13467, 'UIText')
        # Processing the call keyword arguments (line 462)
        kwargs_13469 = {}
        # Getting the type of 'add_data' (line 462)
        add_data_13464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'add_data', False)
        # Calling add_data(args, kwargs) (line 462)
        add_data_call_result_13470 = invoke(stypy.reporting.localization.Localization(__file__, 462, 8), add_data_13464, *[db_13465, str_13466, UIText_13468], **kwargs_13469)
        
        
        # Assigning a Call to a Name (line 465):
        
        # Call to PyDialog(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'db' (line 465)
        db_13472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 23), 'db', False)
        str_13473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 27), 'str', 'FatalError')
        # Getting the type of 'x' (line 465)
        x_13474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 41), 'x', False)
        # Getting the type of 'y' (line 465)
        y_13475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 44), 'y', False)
        # Getting the type of 'w' (line 465)
        w_13476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 47), 'w', False)
        # Getting the type of 'h' (line 465)
        h_13477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 50), 'h', False)
        # Getting the type of 'modal' (line 465)
        modal_13478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 53), 'modal', False)
        # Getting the type of 'title' (line 465)
        title_13479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 60), 'title', False)
        str_13480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 21), 'str', 'Finish')
        str_13481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 31), 'str', 'Finish')
        str_13482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 41), 'str', 'Finish')
        # Processing the call keyword arguments (line 465)
        kwargs_13483 = {}
        # Getting the type of 'PyDialog' (line 465)
        PyDialog_13471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 14), 'PyDialog', False)
        # Calling PyDialog(args, kwargs) (line 465)
        PyDialog_call_result_13484 = invoke(stypy.reporting.localization.Localization(__file__, 465, 14), PyDialog_13471, *[db_13472, str_13473, x_13474, y_13475, w_13476, h_13477, modal_13478, title_13479, str_13480, str_13481, str_13482], **kwargs_13483)
        
        # Assigning a type to the variable 'fatal' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'fatal', PyDialog_call_result_13484)
        
        # Call to title(...): (line 467)
        # Processing the call arguments (line 467)
        str_13487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 20), 'str', '[ProductName] Installer ended prematurely')
        # Processing the call keyword arguments (line 467)
        kwargs_13488 = {}
        # Getting the type of 'fatal' (line 467)
        fatal_13485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'fatal', False)
        # Obtaining the member 'title' of a type (line 467)
        title_13486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 8), fatal_13485, 'title')
        # Calling title(args, kwargs) (line 467)
        title_call_result_13489 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), title_13486, *[str_13487], **kwargs_13488)
        
        
        # Call to back(...): (line 468)
        # Processing the call arguments (line 468)
        str_13492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 19), 'str', '< Back')
        str_13493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 29), 'str', 'Finish')
        # Processing the call keyword arguments (line 468)
        int_13494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 48), 'int')
        keyword_13495 = int_13494
        kwargs_13496 = {'active': keyword_13495}
        # Getting the type of 'fatal' (line 468)
        fatal_13490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'fatal', False)
        # Obtaining the member 'back' of a type (line 468)
        back_13491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), fatal_13490, 'back')
        # Calling back(args, kwargs) (line 468)
        back_call_result_13497 = invoke(stypy.reporting.localization.Localization(__file__, 468, 8), back_13491, *[str_13492, str_13493], **kwargs_13496)
        
        
        # Call to cancel(...): (line 469)
        # Processing the call arguments (line 469)
        str_13500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 21), 'str', 'Cancel')
        str_13501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 31), 'str', 'Back')
        # Processing the call keyword arguments (line 469)
        int_13502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 48), 'int')
        keyword_13503 = int_13502
        kwargs_13504 = {'active': keyword_13503}
        # Getting the type of 'fatal' (line 469)
        fatal_13498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'fatal', False)
        # Obtaining the member 'cancel' of a type (line 469)
        cancel_13499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 8), fatal_13498, 'cancel')
        # Calling cancel(args, kwargs) (line 469)
        cancel_call_result_13505 = invoke(stypy.reporting.localization.Localization(__file__, 469, 8), cancel_13499, *[str_13500, str_13501], **kwargs_13504)
        
        
        # Call to text(...): (line 470)
        # Processing the call arguments (line 470)
        str_13508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 19), 'str', 'Description1')
        int_13509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 35), 'int')
        int_13510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 39), 'int')
        int_13511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 43), 'int')
        int_13512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 48), 'int')
        int_13513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 52), 'int')
        str_13514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 19), 'str', '[ProductName] setup ended prematurely because of an error.  Your system has not been modified.  To install this program at a later time, please run the installation again.')
        # Processing the call keyword arguments (line 470)
        kwargs_13515 = {}
        # Getting the type of 'fatal' (line 470)
        fatal_13506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'fatal', False)
        # Obtaining the member 'text' of a type (line 470)
        text_13507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 8), fatal_13506, 'text')
        # Calling text(args, kwargs) (line 470)
        text_call_result_13516 = invoke(stypy.reporting.localization.Localization(__file__, 470, 8), text_13507, *[str_13508, int_13509, int_13510, int_13511, int_13512, int_13513, str_13514], **kwargs_13515)
        
        
        # Call to text(...): (line 472)
        # Processing the call arguments (line 472)
        str_13519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 19), 'str', 'Description2')
        int_13520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 35), 'int')
        int_13521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 39), 'int')
        int_13522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 44), 'int')
        int_13523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 49), 'int')
        int_13524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 53), 'int')
        str_13525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 19), 'str', 'Click the Finish button to exit the Installer.')
        # Processing the call keyword arguments (line 472)
        kwargs_13526 = {}
        # Getting the type of 'fatal' (line 472)
        fatal_13517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'fatal', False)
        # Obtaining the member 'text' of a type (line 472)
        text_13518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), fatal_13517, 'text')
        # Calling text(args, kwargs) (line 472)
        text_call_result_13527 = invoke(stypy.reporting.localization.Localization(__file__, 472, 8), text_13518, *[str_13519, int_13520, int_13521, int_13522, int_13523, int_13524, str_13525], **kwargs_13526)
        
        
        # Assigning a Call to a Name (line 474):
        
        # Call to next(...): (line 474)
        # Processing the call arguments (line 474)
        str_13530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 21), 'str', 'Finish')
        str_13531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 31), 'str', 'Cancel')
        # Processing the call keyword arguments (line 474)
        str_13532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 46), 'str', 'Finish')
        keyword_13533 = str_13532
        kwargs_13534 = {'name': keyword_13533}
        # Getting the type of 'fatal' (line 474)
        fatal_13528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 10), 'fatal', False)
        # Obtaining the member 'next' of a type (line 474)
        next_13529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 10), fatal_13528, 'next')
        # Calling next(args, kwargs) (line 474)
        next_call_result_13535 = invoke(stypy.reporting.localization.Localization(__file__, 474, 10), next_13529, *[str_13530, str_13531], **kwargs_13534)
        
        # Assigning a type to the variable 'c' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'c', next_call_result_13535)
        
        # Call to event(...): (line 475)
        # Processing the call arguments (line 475)
        str_13538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 16), 'str', 'EndDialog')
        str_13539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 29), 'str', 'Exit')
        # Processing the call keyword arguments (line 475)
        kwargs_13540 = {}
        # Getting the type of 'c' (line 475)
        c_13536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 475)
        event_13537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), c_13536, 'event')
        # Calling event(args, kwargs) (line 475)
        event_call_result_13541 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), event_13537, *[str_13538, str_13539], **kwargs_13540)
        
        
        # Assigning a Call to a Name (line 477):
        
        # Call to PyDialog(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'db' (line 477)
        db_13543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 27), 'db', False)
        str_13544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 31), 'str', 'UserExit')
        # Getting the type of 'x' (line 477)
        x_13545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 43), 'x', False)
        # Getting the type of 'y' (line 477)
        y_13546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 46), 'y', False)
        # Getting the type of 'w' (line 477)
        w_13547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 49), 'w', False)
        # Getting the type of 'h' (line 477)
        h_13548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 52), 'h', False)
        # Getting the type of 'modal' (line 477)
        modal_13549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 55), 'modal', False)
        # Getting the type of 'title' (line 477)
        title_13550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 62), 'title', False)
        str_13551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 21), 'str', 'Finish')
        str_13552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 31), 'str', 'Finish')
        str_13553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 41), 'str', 'Finish')
        # Processing the call keyword arguments (line 477)
        kwargs_13554 = {}
        # Getting the type of 'PyDialog' (line 477)
        PyDialog_13542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 18), 'PyDialog', False)
        # Calling PyDialog(args, kwargs) (line 477)
        PyDialog_call_result_13555 = invoke(stypy.reporting.localization.Localization(__file__, 477, 18), PyDialog_13542, *[db_13543, str_13544, x_13545, y_13546, w_13547, h_13548, modal_13549, title_13550, str_13551, str_13552, str_13553], **kwargs_13554)
        
        # Assigning a type to the variable 'user_exit' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'user_exit', PyDialog_call_result_13555)
        
        # Call to title(...): (line 479)
        # Processing the call arguments (line 479)
        str_13558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 24), 'str', '[ProductName] Installer was interrupted')
        # Processing the call keyword arguments (line 479)
        kwargs_13559 = {}
        # Getting the type of 'user_exit' (line 479)
        user_exit_13556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'user_exit', False)
        # Obtaining the member 'title' of a type (line 479)
        title_13557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 8), user_exit_13556, 'title')
        # Calling title(args, kwargs) (line 479)
        title_call_result_13560 = invoke(stypy.reporting.localization.Localization(__file__, 479, 8), title_13557, *[str_13558], **kwargs_13559)
        
        
        # Call to back(...): (line 480)
        # Processing the call arguments (line 480)
        str_13563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 23), 'str', '< Back')
        str_13564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 33), 'str', 'Finish')
        # Processing the call keyword arguments (line 480)
        int_13565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 52), 'int')
        keyword_13566 = int_13565
        kwargs_13567 = {'active': keyword_13566}
        # Getting the type of 'user_exit' (line 480)
        user_exit_13561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'user_exit', False)
        # Obtaining the member 'back' of a type (line 480)
        back_13562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), user_exit_13561, 'back')
        # Calling back(args, kwargs) (line 480)
        back_call_result_13568 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), back_13562, *[str_13563, str_13564], **kwargs_13567)
        
        
        # Call to cancel(...): (line 481)
        # Processing the call arguments (line 481)
        str_13571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 25), 'str', 'Cancel')
        str_13572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 35), 'str', 'Back')
        # Processing the call keyword arguments (line 481)
        int_13573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 52), 'int')
        keyword_13574 = int_13573
        kwargs_13575 = {'active': keyword_13574}
        # Getting the type of 'user_exit' (line 481)
        user_exit_13569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'user_exit', False)
        # Obtaining the member 'cancel' of a type (line 481)
        cancel_13570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), user_exit_13569, 'cancel')
        # Calling cancel(args, kwargs) (line 481)
        cancel_call_result_13576 = invoke(stypy.reporting.localization.Localization(__file__, 481, 8), cancel_13570, *[str_13571, str_13572], **kwargs_13575)
        
        
        # Call to text(...): (line 482)
        # Processing the call arguments (line 482)
        str_13579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 23), 'str', 'Description1')
        int_13580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 39), 'int')
        int_13581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 43), 'int')
        int_13582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 47), 'int')
        int_13583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 52), 'int')
        int_13584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 56), 'int')
        str_13585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 19), 'str', '[ProductName] setup was interrupted.  Your system has not been modified.  To install this program at a later time, please run the installation again.')
        # Processing the call keyword arguments (line 482)
        kwargs_13586 = {}
        # Getting the type of 'user_exit' (line 482)
        user_exit_13577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'user_exit', False)
        # Obtaining the member 'text' of a type (line 482)
        text_13578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 8), user_exit_13577, 'text')
        # Calling text(args, kwargs) (line 482)
        text_call_result_13587 = invoke(stypy.reporting.localization.Localization(__file__, 482, 8), text_13578, *[str_13579, int_13580, int_13581, int_13582, int_13583, int_13584, str_13585], **kwargs_13586)
        
        
        # Call to text(...): (line 485)
        # Processing the call arguments (line 485)
        str_13590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 23), 'str', 'Description2')
        int_13591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 39), 'int')
        int_13592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 43), 'int')
        int_13593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 48), 'int')
        int_13594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 53), 'int')
        int_13595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 57), 'int')
        str_13596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 19), 'str', 'Click the Finish button to exit the Installer.')
        # Processing the call keyword arguments (line 485)
        kwargs_13597 = {}
        # Getting the type of 'user_exit' (line 485)
        user_exit_13588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'user_exit', False)
        # Obtaining the member 'text' of a type (line 485)
        text_13589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), user_exit_13588, 'text')
        # Calling text(args, kwargs) (line 485)
        text_call_result_13598 = invoke(stypy.reporting.localization.Localization(__file__, 485, 8), text_13589, *[str_13590, int_13591, int_13592, int_13593, int_13594, int_13595, str_13596], **kwargs_13597)
        
        
        # Assigning a Call to a Name (line 487):
        
        # Call to next(...): (line 487)
        # Processing the call arguments (line 487)
        str_13601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 27), 'str', 'Finish')
        str_13602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 37), 'str', 'Cancel')
        # Processing the call keyword arguments (line 487)
        str_13603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 52), 'str', 'Finish')
        keyword_13604 = str_13603
        kwargs_13605 = {'name': keyword_13604}
        # Getting the type of 'user_exit' (line 487)
        user_exit_13599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'user_exit', False)
        # Obtaining the member 'next' of a type (line 487)
        next_13600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), user_exit_13599, 'next')
        # Calling next(args, kwargs) (line 487)
        next_call_result_13606 = invoke(stypy.reporting.localization.Localization(__file__, 487, 12), next_13600, *[str_13601, str_13602], **kwargs_13605)
        
        # Assigning a type to the variable 'c' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'c', next_call_result_13606)
        
        # Call to event(...): (line 488)
        # Processing the call arguments (line 488)
        str_13609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 16), 'str', 'EndDialog')
        str_13610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 29), 'str', 'Exit')
        # Processing the call keyword arguments (line 488)
        kwargs_13611 = {}
        # Getting the type of 'c' (line 488)
        c_13607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 488)
        event_13608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), c_13607, 'event')
        # Calling event(args, kwargs) (line 488)
        event_call_result_13612 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), event_13608, *[str_13609, str_13610], **kwargs_13611)
        
        
        # Assigning a Call to a Name (line 490):
        
        # Call to PyDialog(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'db' (line 490)
        db_13614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 31), 'db', False)
        str_13615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 35), 'str', 'ExitDialog')
        # Getting the type of 'x' (line 490)
        x_13616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 49), 'x', False)
        # Getting the type of 'y' (line 490)
        y_13617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 52), 'y', False)
        # Getting the type of 'w' (line 490)
        w_13618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 55), 'w', False)
        # Getting the type of 'h' (line 490)
        h_13619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 58), 'h', False)
        # Getting the type of 'modal' (line 490)
        modal_13620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 61), 'modal', False)
        # Getting the type of 'title' (line 490)
        title_13621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 68), 'title', False)
        str_13622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 29), 'str', 'Finish')
        str_13623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 39), 'str', 'Finish')
        str_13624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 49), 'str', 'Finish')
        # Processing the call keyword arguments (line 490)
        kwargs_13625 = {}
        # Getting the type of 'PyDialog' (line 490)
        PyDialog_13613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 22), 'PyDialog', False)
        # Calling PyDialog(args, kwargs) (line 490)
        PyDialog_call_result_13626 = invoke(stypy.reporting.localization.Localization(__file__, 490, 22), PyDialog_13613, *[db_13614, str_13615, x_13616, y_13617, w_13618, h_13619, modal_13620, title_13621, str_13622, str_13623, str_13624], **kwargs_13625)
        
        # Assigning a type to the variable 'exit_dialog' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'exit_dialog', PyDialog_call_result_13626)
        
        # Call to title(...): (line 492)
        # Processing the call arguments (line 492)
        str_13629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 26), 'str', 'Completing the [ProductName] Installer')
        # Processing the call keyword arguments (line 492)
        kwargs_13630 = {}
        # Getting the type of 'exit_dialog' (line 492)
        exit_dialog_13627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'exit_dialog', False)
        # Obtaining the member 'title' of a type (line 492)
        title_13628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 8), exit_dialog_13627, 'title')
        # Calling title(args, kwargs) (line 492)
        title_call_result_13631 = invoke(stypy.reporting.localization.Localization(__file__, 492, 8), title_13628, *[str_13629], **kwargs_13630)
        
        
        # Call to back(...): (line 493)
        # Processing the call arguments (line 493)
        str_13634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 25), 'str', '< Back')
        str_13635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 35), 'str', 'Finish')
        # Processing the call keyword arguments (line 493)
        int_13636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 54), 'int')
        keyword_13637 = int_13636
        kwargs_13638 = {'active': keyword_13637}
        # Getting the type of 'exit_dialog' (line 493)
        exit_dialog_13632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'exit_dialog', False)
        # Obtaining the member 'back' of a type (line 493)
        back_13633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), exit_dialog_13632, 'back')
        # Calling back(args, kwargs) (line 493)
        back_call_result_13639 = invoke(stypy.reporting.localization.Localization(__file__, 493, 8), back_13633, *[str_13634, str_13635], **kwargs_13638)
        
        
        # Call to cancel(...): (line 494)
        # Processing the call arguments (line 494)
        str_13642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 27), 'str', 'Cancel')
        str_13643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 37), 'str', 'Back')
        # Processing the call keyword arguments (line 494)
        int_13644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 54), 'int')
        keyword_13645 = int_13644
        kwargs_13646 = {'active': keyword_13645}
        # Getting the type of 'exit_dialog' (line 494)
        exit_dialog_13640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'exit_dialog', False)
        # Obtaining the member 'cancel' of a type (line 494)
        cancel_13641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 8), exit_dialog_13640, 'cancel')
        # Calling cancel(args, kwargs) (line 494)
        cancel_call_result_13647 = invoke(stypy.reporting.localization.Localization(__file__, 494, 8), cancel_13641, *[str_13642, str_13643], **kwargs_13646)
        
        
        # Call to text(...): (line 495)
        # Processing the call arguments (line 495)
        str_13650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 25), 'str', 'Description')
        int_13651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 40), 'int')
        int_13652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 44), 'int')
        int_13653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 49), 'int')
        int_13654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 54), 'int')
        int_13655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 58), 'int')
        str_13656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 19), 'str', 'Click the Finish button to exit the Installer.')
        # Processing the call keyword arguments (line 495)
        kwargs_13657 = {}
        # Getting the type of 'exit_dialog' (line 495)
        exit_dialog_13648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'exit_dialog', False)
        # Obtaining the member 'text' of a type (line 495)
        text_13649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 8), exit_dialog_13648, 'text')
        # Calling text(args, kwargs) (line 495)
        text_call_result_13658 = invoke(stypy.reporting.localization.Localization(__file__, 495, 8), text_13649, *[str_13650, int_13651, int_13652, int_13653, int_13654, int_13655, str_13656], **kwargs_13657)
        
        
        # Assigning a Call to a Name (line 497):
        
        # Call to next(...): (line 497)
        # Processing the call arguments (line 497)
        str_13661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 29), 'str', 'Finish')
        str_13662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 39), 'str', 'Cancel')
        # Processing the call keyword arguments (line 497)
        str_13663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 54), 'str', 'Finish')
        keyword_13664 = str_13663
        kwargs_13665 = {'name': keyword_13664}
        # Getting the type of 'exit_dialog' (line 497)
        exit_dialog_13659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'exit_dialog', False)
        # Obtaining the member 'next' of a type (line 497)
        next_13660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 12), exit_dialog_13659, 'next')
        # Calling next(args, kwargs) (line 497)
        next_call_result_13666 = invoke(stypy.reporting.localization.Localization(__file__, 497, 12), next_13660, *[str_13661, str_13662], **kwargs_13665)
        
        # Assigning a type to the variable 'c' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'c', next_call_result_13666)
        
        # Call to event(...): (line 498)
        # Processing the call arguments (line 498)
        str_13669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 16), 'str', 'EndDialog')
        str_13670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 29), 'str', 'Return')
        # Processing the call keyword arguments (line 498)
        kwargs_13671 = {}
        # Getting the type of 'c' (line 498)
        c_13667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 498)
        event_13668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), c_13667, 'event')
        # Calling event(args, kwargs) (line 498)
        event_call_result_13672 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), event_13668, *[str_13669, str_13670], **kwargs_13671)
        
        
        # Assigning a Call to a Name (line 502):
        
        # Call to PyDialog(...): (line 502)
        # Processing the call arguments (line 502)
        # Getting the type of 'db' (line 502)
        db_13674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'db', False)
        str_13675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 29), 'str', 'FilesInUse')
        # Getting the type of 'x' (line 503)
        x_13676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 25), 'x', False)
        # Getting the type of 'y' (line 503)
        y_13677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 28), 'y', False)
        # Getting the type of 'w' (line 503)
        w_13678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 31), 'w', False)
        # Getting the type of 'h' (line 503)
        h_13679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 34), 'h', False)
        int_13680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 25), 'int')
        # Getting the type of 'title' (line 505)
        title_13681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 25), 'title', False)
        str_13682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 25), 'str', 'Retry')
        str_13683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 34), 'str', 'Retry')
        str_13684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 43), 'str', 'Retry')
        # Processing the call keyword arguments (line 502)
        # Getting the type of 'False' (line 506)
        False_13685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 59), 'False', False)
        keyword_13686 = False_13685
        kwargs_13687 = {'bitmap': keyword_13686}
        # Getting the type of 'PyDialog' (line 502)
        PyDialog_13673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'PyDialog', False)
        # Calling PyDialog(args, kwargs) (line 502)
        PyDialog_call_result_13688 = invoke(stypy.reporting.localization.Localization(__file__, 502, 16), PyDialog_13673, *[db_13674, str_13675, x_13676, y_13677, w_13678, h_13679, int_13680, title_13681, str_13682, str_13683, str_13684], **kwargs_13687)
        
        # Assigning a type to the variable 'inuse' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'inuse', PyDialog_call_result_13688)
        
        # Call to text(...): (line 507)
        # Processing the call arguments (line 507)
        str_13691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 19), 'str', 'Title')
        int_13692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 28), 'int')
        int_13693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 32), 'int')
        int_13694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 35), 'int')
        int_13695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 40), 'int')
        int_13696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 44), 'int')
        str_13697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 19), 'str', '{\\DlgFontBold8}Files in Use')
        # Processing the call keyword arguments (line 507)
        kwargs_13698 = {}
        # Getting the type of 'inuse' (line 507)
        inuse_13689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'inuse', False)
        # Obtaining the member 'text' of a type (line 507)
        text_13690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), inuse_13689, 'text')
        # Calling text(args, kwargs) (line 507)
        text_call_result_13699 = invoke(stypy.reporting.localization.Localization(__file__, 507, 8), text_13690, *[str_13691, int_13692, int_13693, int_13694, int_13695, int_13696, str_13697], **kwargs_13698)
        
        
        # Call to text(...): (line 509)
        # Processing the call arguments (line 509)
        str_13702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 19), 'str', 'Description')
        int_13703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 34), 'int')
        int_13704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 38), 'int')
        int_13705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 42), 'int')
        int_13706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 47), 'int')
        int_13707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 51), 'int')
        str_13708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 15), 'str', 'Some files that need to be updated are currently in use.')
        # Processing the call keyword arguments (line 509)
        kwargs_13709 = {}
        # Getting the type of 'inuse' (line 509)
        inuse_13700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'inuse', False)
        # Obtaining the member 'text' of a type (line 509)
        text_13701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), inuse_13700, 'text')
        # Calling text(args, kwargs) (line 509)
        text_call_result_13710 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), text_13701, *[str_13702, int_13703, int_13704, int_13705, int_13706, int_13707, str_13708], **kwargs_13709)
        
        
        # Call to text(...): (line 511)
        # Processing the call arguments (line 511)
        str_13713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 19), 'str', 'Text')
        int_13714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 27), 'int')
        int_13715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 31), 'int')
        int_13716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 35), 'int')
        int_13717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 40), 'int')
        int_13718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 44), 'int')
        str_13719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 19), 'str', 'The following applications are using files that need to be updated by this setup. Close these applications and then click Retry to continue the installation or Cancel to exit it.')
        # Processing the call keyword arguments (line 511)
        kwargs_13720 = {}
        # Getting the type of 'inuse' (line 511)
        inuse_13711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'inuse', False)
        # Obtaining the member 'text' of a type (line 511)
        text_13712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 8), inuse_13711, 'text')
        # Calling text(args, kwargs) (line 511)
        text_call_result_13721 = invoke(stypy.reporting.localization.Localization(__file__, 511, 8), text_13712, *[str_13713, int_13714, int_13715, int_13716, int_13717, int_13718, str_13719], **kwargs_13720)
        
        
        # Call to control(...): (line 513)
        # Processing the call arguments (line 513)
        str_13724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 22), 'str', 'List')
        str_13725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 30), 'str', 'ListBox')
        int_13726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 41), 'int')
        int_13727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 45), 'int')
        int_13728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 50), 'int')
        int_13729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 55), 'int')
        int_13730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 60), 'int')
        str_13731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 63), 'str', 'FileInUseProcess')
        # Getting the type of 'None' (line 514)
        None_13732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 22), 'None', False)
        # Getting the type of 'None' (line 514)
        None_13733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 28), 'None', False)
        # Getting the type of 'None' (line 514)
        None_13734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 34), 'None', False)
        # Processing the call keyword arguments (line 513)
        kwargs_13735 = {}
        # Getting the type of 'inuse' (line 513)
        inuse_13722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'inuse', False)
        # Obtaining the member 'control' of a type (line 513)
        control_13723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 8), inuse_13722, 'control')
        # Calling control(args, kwargs) (line 513)
        control_call_result_13736 = invoke(stypy.reporting.localization.Localization(__file__, 513, 8), control_13723, *[str_13724, str_13725, int_13726, int_13727, int_13728, int_13729, int_13730, str_13731, None_13732, None_13733, None_13734], **kwargs_13735)
        
        
        # Assigning a Call to a Name (line 515):
        
        # Call to back(...): (line 515)
        # Processing the call arguments (line 515)
        str_13739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 21), 'str', 'Exit')
        str_13740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 29), 'str', 'Ignore')
        # Processing the call keyword arguments (line 515)
        str_13741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 44), 'str', 'Exit')
        keyword_13742 = str_13741
        kwargs_13743 = {'name': keyword_13742}
        # Getting the type of 'inuse' (line 515)
        inuse_13737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 10), 'inuse', False)
        # Obtaining the member 'back' of a type (line 515)
        back_13738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 10), inuse_13737, 'back')
        # Calling back(args, kwargs) (line 515)
        back_call_result_13744 = invoke(stypy.reporting.localization.Localization(__file__, 515, 10), back_13738, *[str_13739, str_13740], **kwargs_13743)
        
        # Assigning a type to the variable 'c' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'c', back_call_result_13744)
        
        # Call to event(...): (line 516)
        # Processing the call arguments (line 516)
        str_13747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 16), 'str', 'EndDialog')
        str_13748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 29), 'str', 'Exit')
        # Processing the call keyword arguments (line 516)
        kwargs_13749 = {}
        # Getting the type of 'c' (line 516)
        c_13745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 516)
        event_13746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), c_13745, 'event')
        # Calling event(args, kwargs) (line 516)
        event_call_result_13750 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), event_13746, *[str_13747, str_13748], **kwargs_13749)
        
        
        # Assigning a Call to a Name (line 517):
        
        # Call to next(...): (line 517)
        # Processing the call arguments (line 517)
        str_13753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 21), 'str', 'Ignore')
        str_13754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 31), 'str', 'Retry')
        # Processing the call keyword arguments (line 517)
        str_13755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 45), 'str', 'Ignore')
        keyword_13756 = str_13755
        kwargs_13757 = {'name': keyword_13756}
        # Getting the type of 'inuse' (line 517)
        inuse_13751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 10), 'inuse', False)
        # Obtaining the member 'next' of a type (line 517)
        next_13752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 10), inuse_13751, 'next')
        # Calling next(args, kwargs) (line 517)
        next_call_result_13758 = invoke(stypy.reporting.localization.Localization(__file__, 517, 10), next_13752, *[str_13753, str_13754], **kwargs_13757)
        
        # Assigning a type to the variable 'c' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'c', next_call_result_13758)
        
        # Call to event(...): (line 518)
        # Processing the call arguments (line 518)
        str_13761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 16), 'str', 'EndDialog')
        str_13762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 29), 'str', 'Ignore')
        # Processing the call keyword arguments (line 518)
        kwargs_13763 = {}
        # Getting the type of 'c' (line 518)
        c_13759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 518)
        event_13760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 8), c_13759, 'event')
        # Calling event(args, kwargs) (line 518)
        event_call_result_13764 = invoke(stypy.reporting.localization.Localization(__file__, 518, 8), event_13760, *[str_13761, str_13762], **kwargs_13763)
        
        
        # Assigning a Call to a Name (line 519):
        
        # Call to cancel(...): (line 519)
        # Processing the call arguments (line 519)
        str_13767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 23), 'str', 'Retry')
        str_13768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 32), 'str', 'Exit')
        # Processing the call keyword arguments (line 519)
        str_13769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 45), 'str', 'Retry')
        keyword_13770 = str_13769
        kwargs_13771 = {'name': keyword_13770}
        # Getting the type of 'inuse' (line 519)
        inuse_13765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 10), 'inuse', False)
        # Obtaining the member 'cancel' of a type (line 519)
        cancel_13766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 10), inuse_13765, 'cancel')
        # Calling cancel(args, kwargs) (line 519)
        cancel_call_result_13772 = invoke(stypy.reporting.localization.Localization(__file__, 519, 10), cancel_13766, *[str_13767, str_13768], **kwargs_13771)
        
        # Assigning a type to the variable 'c' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'c', cancel_call_result_13772)
        
        # Call to event(...): (line 520)
        # Processing the call arguments (line 520)
        str_13775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 16), 'str', 'EndDialog')
        str_13776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 28), 'str', 'Retry')
        # Processing the call keyword arguments (line 520)
        kwargs_13777 = {}
        # Getting the type of 'c' (line 520)
        c_13773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 520)
        event_13774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 8), c_13773, 'event')
        # Calling event(args, kwargs) (line 520)
        event_call_result_13778 = invoke(stypy.reporting.localization.Localization(__file__, 520, 8), event_13774, *[str_13775, str_13776], **kwargs_13777)
        
        
        # Assigning a Call to a Name (line 523):
        
        # Call to Dialog(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 'db' (line 523)
        db_13780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 23), 'db', False)
        str_13781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 27), 'str', 'ErrorDlg')
        int_13782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 23), 'int')
        int_13783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 27), 'int')
        int_13784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 31), 'int')
        int_13785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 36), 'int')
        int_13786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 23), 'int')
        # Getting the type of 'title' (line 526)
        title_13787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 23), 'title', False)
        str_13788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 23), 'str', 'ErrorText')
        # Getting the type of 'None' (line 527)
        None_13789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 36), 'None', False)
        # Getting the type of 'None' (line 527)
        None_13790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 42), 'None', False)
        # Processing the call keyword arguments (line 523)
        kwargs_13791 = {}
        # Getting the type of 'Dialog' (line 523)
        Dialog_13779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'Dialog', False)
        # Calling Dialog(args, kwargs) (line 523)
        Dialog_call_result_13792 = invoke(stypy.reporting.localization.Localization(__file__, 523, 16), Dialog_13779, *[db_13780, str_13781, int_13782, int_13783, int_13784, int_13785, int_13786, title_13787, str_13788, None_13789, None_13790], **kwargs_13791)
        
        # Assigning a type to the variable 'error' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'error', Dialog_call_result_13792)
        
        # Call to text(...): (line 528)
        # Processing the call arguments (line 528)
        str_13795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 19), 'str', 'ErrorText')
        int_13796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 32), 'int')
        int_13797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 35), 'int')
        int_13798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 37), 'int')
        int_13799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 41), 'int')
        int_13800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 44), 'int')
        str_13801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 47), 'str', '')
        # Processing the call keyword arguments (line 528)
        kwargs_13802 = {}
        # Getting the type of 'error' (line 528)
        error_13793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'error', False)
        # Obtaining the member 'text' of a type (line 528)
        text_13794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 8), error_13793, 'text')
        # Calling text(args, kwargs) (line 528)
        text_call_result_13803 = invoke(stypy.reporting.localization.Localization(__file__, 528, 8), text_13794, *[str_13795, int_13796, int_13797, int_13798, int_13799, int_13800, str_13801], **kwargs_13802)
        
        
        # Call to event(...): (line 530)
        # Processing the call arguments (line 530)
        str_13817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 61), 'str', 'EndDialog')
        str_13818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 73), 'str', 'ErrorNo')
        # Processing the call keyword arguments (line 530)
        kwargs_13819 = {}
        
        # Call to pushbutton(...): (line 530)
        # Processing the call arguments (line 530)
        str_13806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 25), 'str', 'N')
        int_13807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 29), 'int')
        int_13808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 33), 'int')
        int_13809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 36), 'int')
        int_13810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 39), 'int')
        int_13811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 42), 'int')
        str_13812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 44), 'str', 'No')
        # Getting the type of 'None' (line 530)
        None_13813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 49), 'None', False)
        # Processing the call keyword arguments (line 530)
        kwargs_13814 = {}
        # Getting the type of 'error' (line 530)
        error_13804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'error', False)
        # Obtaining the member 'pushbutton' of a type (line 530)
        pushbutton_13805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 8), error_13804, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 530)
        pushbutton_call_result_13815 = invoke(stypy.reporting.localization.Localization(__file__, 530, 8), pushbutton_13805, *[str_13806, int_13807, int_13808, int_13809, int_13810, int_13811, str_13812, None_13813], **kwargs_13814)
        
        # Obtaining the member 'event' of a type (line 530)
        event_13816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 8), pushbutton_call_result_13815, 'event')
        # Calling event(args, kwargs) (line 530)
        event_call_result_13820 = invoke(stypy.reporting.localization.Localization(__file__, 530, 8), event_13816, *[str_13817, str_13818], **kwargs_13819)
        
        
        # Call to event(...): (line 531)
        # Processing the call arguments (line 531)
        str_13834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 62), 'str', 'EndDialog')
        str_13835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 74), 'str', 'ErrorYes')
        # Processing the call keyword arguments (line 531)
        kwargs_13836 = {}
        
        # Call to pushbutton(...): (line 531)
        # Processing the call arguments (line 531)
        str_13823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 25), 'str', 'Y')
        int_13824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 29), 'int')
        int_13825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 33), 'int')
        int_13826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 36), 'int')
        int_13827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 39), 'int')
        int_13828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 42), 'int')
        str_13829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 44), 'str', 'Yes')
        # Getting the type of 'None' (line 531)
        None_13830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 50), 'None', False)
        # Processing the call keyword arguments (line 531)
        kwargs_13831 = {}
        # Getting the type of 'error' (line 531)
        error_13821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'error', False)
        # Obtaining the member 'pushbutton' of a type (line 531)
        pushbutton_13822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 8), error_13821, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 531)
        pushbutton_call_result_13832 = invoke(stypy.reporting.localization.Localization(__file__, 531, 8), pushbutton_13822, *[str_13823, int_13824, int_13825, int_13826, int_13827, int_13828, str_13829, None_13830], **kwargs_13831)
        
        # Obtaining the member 'event' of a type (line 531)
        event_13833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 8), pushbutton_call_result_13832, 'event')
        # Calling event(args, kwargs) (line 531)
        event_call_result_13837 = invoke(stypy.reporting.localization.Localization(__file__, 531, 8), event_13833, *[str_13834, str_13835], **kwargs_13836)
        
        
        # Call to event(...): (line 532)
        # Processing the call arguments (line 532)
        str_13851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 62), 'str', 'EndDialog')
        str_13852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 74), 'str', 'ErrorAbort')
        # Processing the call keyword arguments (line 532)
        kwargs_13853 = {}
        
        # Call to pushbutton(...): (line 532)
        # Processing the call arguments (line 532)
        str_13840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 25), 'str', 'A')
        int_13841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 29), 'int')
        int_13842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 31), 'int')
        int_13843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 34), 'int')
        int_13844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 37), 'int')
        int_13845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 40), 'int')
        str_13846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 42), 'str', 'Abort')
        # Getting the type of 'None' (line 532)
        None_13847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 50), 'None', False)
        # Processing the call keyword arguments (line 532)
        kwargs_13848 = {}
        # Getting the type of 'error' (line 532)
        error_13838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'error', False)
        # Obtaining the member 'pushbutton' of a type (line 532)
        pushbutton_13839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 8), error_13838, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 532)
        pushbutton_call_result_13849 = invoke(stypy.reporting.localization.Localization(__file__, 532, 8), pushbutton_13839, *[str_13840, int_13841, int_13842, int_13843, int_13844, int_13845, str_13846, None_13847], **kwargs_13848)
        
        # Obtaining the member 'event' of a type (line 532)
        event_13850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 8), pushbutton_call_result_13849, 'event')
        # Calling event(args, kwargs) (line 532)
        event_call_result_13854 = invoke(stypy.reporting.localization.Localization(__file__, 532, 8), event_13850, *[str_13851, str_13852], **kwargs_13853)
        
        
        # Call to event(...): (line 533)
        # Processing the call arguments (line 533)
        str_13868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 64), 'str', 'EndDialog')
        str_13869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 76), 'str', 'ErrorCancel')
        # Processing the call keyword arguments (line 533)
        kwargs_13870 = {}
        
        # Call to pushbutton(...): (line 533)
        # Processing the call arguments (line 533)
        str_13857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 25), 'str', 'C')
        int_13858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 29), 'int')
        int_13859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 32), 'int')
        int_13860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 35), 'int')
        int_13861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 38), 'int')
        int_13862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 41), 'int')
        str_13863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 43), 'str', 'Cancel')
        # Getting the type of 'None' (line 533)
        None_13864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 52), 'None', False)
        # Processing the call keyword arguments (line 533)
        kwargs_13865 = {}
        # Getting the type of 'error' (line 533)
        error_13855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'error', False)
        # Obtaining the member 'pushbutton' of a type (line 533)
        pushbutton_13856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), error_13855, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 533)
        pushbutton_call_result_13866 = invoke(stypy.reporting.localization.Localization(__file__, 533, 8), pushbutton_13856, *[str_13857, int_13858, int_13859, int_13860, int_13861, int_13862, str_13863, None_13864], **kwargs_13865)
        
        # Obtaining the member 'event' of a type (line 533)
        event_13867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), pushbutton_call_result_13866, 'event')
        # Calling event(args, kwargs) (line 533)
        event_call_result_13871 = invoke(stypy.reporting.localization.Localization(__file__, 533, 8), event_13867, *[str_13868, str_13869], **kwargs_13870)
        
        
        # Call to event(...): (line 534)
        # Processing the call arguments (line 534)
        str_13885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 64), 'str', 'EndDialog')
        str_13886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 76), 'str', 'ErrorIgnore')
        # Processing the call keyword arguments (line 534)
        kwargs_13887 = {}
        
        # Call to pushbutton(...): (line 534)
        # Processing the call arguments (line 534)
        str_13874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 25), 'str', 'I')
        int_13875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 29), 'int')
        int_13876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 32), 'int')
        int_13877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 35), 'int')
        int_13878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 38), 'int')
        int_13879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 41), 'int')
        str_13880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 43), 'str', 'Ignore')
        # Getting the type of 'None' (line 534)
        None_13881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 52), 'None', False)
        # Processing the call keyword arguments (line 534)
        kwargs_13882 = {}
        # Getting the type of 'error' (line 534)
        error_13872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'error', False)
        # Obtaining the member 'pushbutton' of a type (line 534)
        pushbutton_13873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 8), error_13872, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 534)
        pushbutton_call_result_13883 = invoke(stypy.reporting.localization.Localization(__file__, 534, 8), pushbutton_13873, *[str_13874, int_13875, int_13876, int_13877, int_13878, int_13879, str_13880, None_13881], **kwargs_13882)
        
        # Obtaining the member 'event' of a type (line 534)
        event_13884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 8), pushbutton_call_result_13883, 'event')
        # Calling event(args, kwargs) (line 534)
        event_call_result_13888 = invoke(stypy.reporting.localization.Localization(__file__, 534, 8), event_13884, *[str_13885, str_13886], **kwargs_13887)
        
        
        # Call to event(...): (line 535)
        # Processing the call arguments (line 535)
        str_13902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 61), 'str', 'EndDialog')
        str_13903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 73), 'str', 'ErrorOk')
        # Processing the call keyword arguments (line 535)
        kwargs_13904 = {}
        
        # Call to pushbutton(...): (line 535)
        # Processing the call arguments (line 535)
        str_13891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 25), 'str', 'O')
        int_13892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 29), 'int')
        int_13893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 33), 'int')
        int_13894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 36), 'int')
        int_13895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 39), 'int')
        int_13896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 42), 'int')
        str_13897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 44), 'str', 'Ok')
        # Getting the type of 'None' (line 535)
        None_13898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 49), 'None', False)
        # Processing the call keyword arguments (line 535)
        kwargs_13899 = {}
        # Getting the type of 'error' (line 535)
        error_13889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'error', False)
        # Obtaining the member 'pushbutton' of a type (line 535)
        pushbutton_13890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), error_13889, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 535)
        pushbutton_call_result_13900 = invoke(stypy.reporting.localization.Localization(__file__, 535, 8), pushbutton_13890, *[str_13891, int_13892, int_13893, int_13894, int_13895, int_13896, str_13897, None_13898], **kwargs_13899)
        
        # Obtaining the member 'event' of a type (line 535)
        event_13901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), pushbutton_call_result_13900, 'event')
        # Calling event(args, kwargs) (line 535)
        event_call_result_13905 = invoke(stypy.reporting.localization.Localization(__file__, 535, 8), event_13901, *[str_13902, str_13903], **kwargs_13904)
        
        
        # Call to event(...): (line 536)
        # Processing the call arguments (line 536)
        str_13919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 64), 'str', 'EndDialog')
        str_13920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 76), 'str', 'ErrorRetry')
        # Processing the call keyword arguments (line 536)
        kwargs_13921 = {}
        
        # Call to pushbutton(...): (line 536)
        # Processing the call arguments (line 536)
        str_13908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 25), 'str', 'R')
        int_13909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 29), 'int')
        int_13910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 33), 'int')
        int_13911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 36), 'int')
        int_13912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 39), 'int')
        int_13913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 42), 'int')
        str_13914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 44), 'str', 'Retry')
        # Getting the type of 'None' (line 536)
        None_13915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 52), 'None', False)
        # Processing the call keyword arguments (line 536)
        kwargs_13916 = {}
        # Getting the type of 'error' (line 536)
        error_13906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'error', False)
        # Obtaining the member 'pushbutton' of a type (line 536)
        pushbutton_13907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), error_13906, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 536)
        pushbutton_call_result_13917 = invoke(stypy.reporting.localization.Localization(__file__, 536, 8), pushbutton_13907, *[str_13908, int_13909, int_13910, int_13911, int_13912, int_13913, str_13914, None_13915], **kwargs_13916)
        
        # Obtaining the member 'event' of a type (line 536)
        event_13918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), pushbutton_call_result_13917, 'event')
        # Calling event(args, kwargs) (line 536)
        event_call_result_13922 = invoke(stypy.reporting.localization.Localization(__file__, 536, 8), event_13918, *[str_13919, str_13920], **kwargs_13921)
        
        
        # Assigning a Call to a Name (line 540):
        
        # Call to Dialog(...): (line 540)
        # Processing the call arguments (line 540)
        # Getting the type of 'db' (line 540)
        db_13924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 24), 'db', False)
        str_13925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 28), 'str', 'CancelDlg')
        int_13926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 41), 'int')
        int_13927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 45), 'int')
        int_13928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 49), 'int')
        int_13929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 54), 'int')
        int_13930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 58), 'int')
        # Getting the type of 'title' (line 540)
        title_13931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 61), 'title', False)
        str_13932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 24), 'str', 'No')
        str_13933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 30), 'str', 'No')
        str_13934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 36), 'str', 'No')
        # Processing the call keyword arguments (line 540)
        kwargs_13935 = {}
        # Getting the type of 'Dialog' (line 540)
        Dialog_13923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 17), 'Dialog', False)
        # Calling Dialog(args, kwargs) (line 540)
        Dialog_call_result_13936 = invoke(stypy.reporting.localization.Localization(__file__, 540, 17), Dialog_13923, *[db_13924, str_13925, int_13926, int_13927, int_13928, int_13929, int_13930, title_13931, str_13932, str_13933, str_13934], **kwargs_13935)
        
        # Assigning a type to the variable 'cancel' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'cancel', Dialog_call_result_13936)
        
        # Call to text(...): (line 542)
        # Processing the call arguments (line 542)
        str_13939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 20), 'str', 'Text')
        int_13940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 28), 'int')
        int_13941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 32), 'int')
        int_13942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 36), 'int')
        int_13943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 41), 'int')
        int_13944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 45), 'int')
        str_13945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 20), 'str', 'Are you sure you want to cancel [ProductName] installation?')
        # Processing the call keyword arguments (line 542)
        kwargs_13946 = {}
        # Getting the type of 'cancel' (line 542)
        cancel_13937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'cancel', False)
        # Obtaining the member 'text' of a type (line 542)
        text_13938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 8), cancel_13937, 'text')
        # Calling text(args, kwargs) (line 542)
        text_call_result_13947 = invoke(stypy.reporting.localization.Localization(__file__, 542, 8), text_13938, *[str_13939, int_13940, int_13941, int_13942, int_13943, int_13944, str_13945], **kwargs_13946)
        
        
        # Assigning a Call to a Name (line 546):
        
        # Call to pushbutton(...): (line 546)
        # Processing the call arguments (line 546)
        str_13950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 28), 'str', 'Yes')
        int_13951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 35), 'int')
        int_13952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 39), 'int')
        int_13953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 43), 'int')
        int_13954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 47), 'int')
        int_13955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 51), 'int')
        str_13956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 54), 'str', 'Yes')
        str_13957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 61), 'str', 'No')
        # Processing the call keyword arguments (line 546)
        kwargs_13958 = {}
        # Getting the type of 'cancel' (line 546)
        cancel_13948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 10), 'cancel', False)
        # Obtaining the member 'pushbutton' of a type (line 546)
        pushbutton_13949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 10), cancel_13948, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 546)
        pushbutton_call_result_13959 = invoke(stypy.reporting.localization.Localization(__file__, 546, 10), pushbutton_13949, *[str_13950, int_13951, int_13952, int_13953, int_13954, int_13955, str_13956, str_13957], **kwargs_13958)
        
        # Assigning a type to the variable 'c' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'c', pushbutton_call_result_13959)
        
        # Call to event(...): (line 547)
        # Processing the call arguments (line 547)
        str_13962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 16), 'str', 'EndDialog')
        str_13963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 29), 'str', 'Exit')
        # Processing the call keyword arguments (line 547)
        kwargs_13964 = {}
        # Getting the type of 'c' (line 547)
        c_13960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 547)
        event_13961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 8), c_13960, 'event')
        # Calling event(args, kwargs) (line 547)
        event_call_result_13965 = invoke(stypy.reporting.localization.Localization(__file__, 547, 8), event_13961, *[str_13962, str_13963], **kwargs_13964)
        
        
        # Assigning a Call to a Name (line 549):
        
        # Call to pushbutton(...): (line 549)
        # Processing the call arguments (line 549)
        str_13968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 28), 'str', 'No')
        int_13969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 34), 'int')
        int_13970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 39), 'int')
        int_13971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 43), 'int')
        int_13972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 47), 'int')
        int_13973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 51), 'int')
        str_13974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 54), 'str', 'No')
        str_13975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 60), 'str', 'Yes')
        # Processing the call keyword arguments (line 549)
        kwargs_13976 = {}
        # Getting the type of 'cancel' (line 549)
        cancel_13966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 10), 'cancel', False)
        # Obtaining the member 'pushbutton' of a type (line 549)
        pushbutton_13967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 10), cancel_13966, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 549)
        pushbutton_call_result_13977 = invoke(stypy.reporting.localization.Localization(__file__, 549, 10), pushbutton_13967, *[str_13968, int_13969, int_13970, int_13971, int_13972, int_13973, str_13974, str_13975], **kwargs_13976)
        
        # Assigning a type to the variable 'c' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'c', pushbutton_call_result_13977)
        
        # Call to event(...): (line 550)
        # Processing the call arguments (line 550)
        str_13980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 16), 'str', 'EndDialog')
        str_13981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 29), 'str', 'Return')
        # Processing the call keyword arguments (line 550)
        kwargs_13982 = {}
        # Getting the type of 'c' (line 550)
        c_13978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 550)
        event_13979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), c_13978, 'event')
        # Calling event(args, kwargs) (line 550)
        event_call_result_13983 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), event_13979, *[str_13980, str_13981], **kwargs_13982)
        
        
        # Assigning a Call to a Name (line 554):
        
        # Call to Dialog(...): (line 554)
        # Processing the call arguments (line 554)
        # Getting the type of 'db' (line 554)
        db_13985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 25), 'db', False)
        str_13986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 29), 'str', 'WaitForCostingDlg')
        int_13987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 50), 'int')
        int_13988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 54), 'int')
        int_13989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 58), 'int')
        int_13990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 63), 'int')
        # Getting the type of 'modal' (line 554)
        modal_13991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 67), 'modal', False)
        # Getting the type of 'title' (line 554)
        title_13992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 74), 'title', False)
        str_13993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 25), 'str', 'Return')
        str_13994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 35), 'str', 'Return')
        str_13995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 45), 'str', 'Return')
        # Processing the call keyword arguments (line 554)
        kwargs_13996 = {}
        # Getting the type of 'Dialog' (line 554)
        Dialog_13984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 18), 'Dialog', False)
        # Calling Dialog(args, kwargs) (line 554)
        Dialog_call_result_13997 = invoke(stypy.reporting.localization.Localization(__file__, 554, 18), Dialog_13984, *[db_13985, str_13986, int_13987, int_13988, int_13989, int_13990, modal_13991, title_13992, str_13993, str_13994, str_13995], **kwargs_13996)
        
        # Assigning a type to the variable 'costing' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'costing', Dialog_call_result_13997)
        
        # Call to text(...): (line 556)
        # Processing the call arguments (line 556)
        str_14000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 21), 'str', 'Text')
        int_14001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 29), 'int')
        int_14002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 33), 'int')
        int_14003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 37), 'int')
        int_14004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 42), 'int')
        int_14005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 46), 'int')
        str_14006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 21), 'str', 'Please wait while the installer finishes determining your disk space requirements.')
        # Processing the call keyword arguments (line 556)
        kwargs_14007 = {}
        # Getting the type of 'costing' (line 556)
        costing_13998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'costing', False)
        # Obtaining the member 'text' of a type (line 556)
        text_13999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 8), costing_13998, 'text')
        # Calling text(args, kwargs) (line 556)
        text_call_result_14008 = invoke(stypy.reporting.localization.Localization(__file__, 556, 8), text_13999, *[str_14000, int_14001, int_14002, int_14003, int_14004, int_14005, str_14006], **kwargs_14007)
        
        
        # Assigning a Call to a Name (line 558):
        
        # Call to pushbutton(...): (line 558)
        # Processing the call arguments (line 558)
        str_14011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 31), 'str', 'Return')
        int_14012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 41), 'int')
        int_14013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 46), 'int')
        int_14014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 50), 'int')
        int_14015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 54), 'int')
        int_14016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 58), 'int')
        str_14017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 61), 'str', 'Return')
        # Getting the type of 'None' (line 558)
        None_14018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 71), 'None', False)
        # Processing the call keyword arguments (line 558)
        kwargs_14019 = {}
        # Getting the type of 'costing' (line 558)
        costing_14009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'costing', False)
        # Obtaining the member 'pushbutton' of a type (line 558)
        pushbutton_14010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 12), costing_14009, 'pushbutton')
        # Calling pushbutton(args, kwargs) (line 558)
        pushbutton_call_result_14020 = invoke(stypy.reporting.localization.Localization(__file__, 558, 12), pushbutton_14010, *[str_14011, int_14012, int_14013, int_14014, int_14015, int_14016, str_14017, None_14018], **kwargs_14019)
        
        # Assigning a type to the variable 'c' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'c', pushbutton_call_result_14020)
        
        # Call to event(...): (line 559)
        # Processing the call arguments (line 559)
        str_14023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 16), 'str', 'EndDialog')
        str_14024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 29), 'str', 'Exit')
        # Processing the call keyword arguments (line 559)
        kwargs_14025 = {}
        # Getting the type of 'c' (line 559)
        c_14021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 559)
        event_14022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 8), c_14021, 'event')
        # Calling event(args, kwargs) (line 559)
        event_call_result_14026 = invoke(stypy.reporting.localization.Localization(__file__, 559, 8), event_14022, *[str_14023, str_14024], **kwargs_14025)
        
        
        # Assigning a Call to a Name (line 563):
        
        # Call to PyDialog(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'db' (line 563)
        db_14028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 24), 'db', False)
        str_14029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 28), 'str', 'PrepareDlg')
        # Getting the type of 'x' (line 563)
        x_14030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 42), 'x', False)
        # Getting the type of 'y' (line 563)
        y_14031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 45), 'y', False)
        # Getting the type of 'w' (line 563)
        w_14032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 48), 'w', False)
        # Getting the type of 'h' (line 563)
        h_14033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 51), 'h', False)
        # Getting the type of 'modeless' (line 563)
        modeless_14034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 54), 'modeless', False)
        # Getting the type of 'title' (line 563)
        title_14035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 64), 'title', False)
        str_14036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 24), 'str', 'Cancel')
        str_14037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 34), 'str', 'Cancel')
        str_14038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 44), 'str', 'Cancel')
        # Processing the call keyword arguments (line 563)
        kwargs_14039 = {}
        # Getting the type of 'PyDialog' (line 563)
        PyDialog_14027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 15), 'PyDialog', False)
        # Calling PyDialog(args, kwargs) (line 563)
        PyDialog_call_result_14040 = invoke(stypy.reporting.localization.Localization(__file__, 563, 15), PyDialog_14027, *[db_14028, str_14029, x_14030, y_14031, w_14032, h_14033, modeless_14034, title_14035, str_14036, str_14037, str_14038], **kwargs_14039)
        
        # Assigning a type to the variable 'prep' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'prep', PyDialog_call_result_14040)
        
        # Call to text(...): (line 565)
        # Processing the call arguments (line 565)
        str_14043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 18), 'str', 'Description')
        int_14044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 33), 'int')
        int_14045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 37), 'int')
        int_14046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 41), 'int')
        int_14047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 46), 'int')
        int_14048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 50), 'int')
        str_14049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 18), 'str', 'Please wait while the Installer prepares to guide you through the installation.')
        # Processing the call keyword arguments (line 565)
        kwargs_14050 = {}
        # Getting the type of 'prep' (line 565)
        prep_14041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'prep', False)
        # Obtaining the member 'text' of a type (line 565)
        text_14042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 8), prep_14041, 'text')
        # Calling text(args, kwargs) (line 565)
        text_call_result_14051 = invoke(stypy.reporting.localization.Localization(__file__, 565, 8), text_14042, *[str_14043, int_14044, int_14045, int_14046, int_14047, int_14048, str_14049], **kwargs_14050)
        
        
        # Call to title(...): (line 567)
        # Processing the call arguments (line 567)
        str_14054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 19), 'str', 'Welcome to the [ProductName] Installer')
        # Processing the call keyword arguments (line 567)
        kwargs_14055 = {}
        # Getting the type of 'prep' (line 567)
        prep_14052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'prep', False)
        # Obtaining the member 'title' of a type (line 567)
        title_14053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 8), prep_14052, 'title')
        # Calling title(args, kwargs) (line 567)
        title_call_result_14056 = invoke(stypy.reporting.localization.Localization(__file__, 567, 8), title_14053, *[str_14054], **kwargs_14055)
        
        
        # Assigning a Call to a Name (line 568):
        
        # Call to text(...): (line 568)
        # Processing the call arguments (line 568)
        str_14059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 20), 'str', 'ActionText')
        int_14060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 34), 'int')
        int_14061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 38), 'int')
        int_14062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 43), 'int')
        int_14063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 48), 'int')
        int_14064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 52), 'int')
        str_14065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 61), 'str', 'Pondering...')
        # Processing the call keyword arguments (line 568)
        kwargs_14066 = {}
        # Getting the type of 'prep' (line 568)
        prep_14057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 10), 'prep', False)
        # Obtaining the member 'text' of a type (line 568)
        text_14058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 10), prep_14057, 'text')
        # Calling text(args, kwargs) (line 568)
        text_call_result_14067 = invoke(stypy.reporting.localization.Localization(__file__, 568, 10), text_14058, *[str_14059, int_14060, int_14061, int_14062, int_14063, int_14064, str_14065], **kwargs_14066)
        
        # Assigning a type to the variable 'c' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'c', text_call_result_14067)
        
        # Call to mapping(...): (line 569)
        # Processing the call arguments (line 569)
        str_14070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 18), 'str', 'ActionText')
        str_14071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 32), 'str', 'Text')
        # Processing the call keyword arguments (line 569)
        kwargs_14072 = {}
        # Getting the type of 'c' (line 569)
        c_14068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'c', False)
        # Obtaining the member 'mapping' of a type (line 569)
        mapping_14069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 8), c_14068, 'mapping')
        # Calling mapping(args, kwargs) (line 569)
        mapping_call_result_14073 = invoke(stypy.reporting.localization.Localization(__file__, 569, 8), mapping_14069, *[str_14070, str_14071], **kwargs_14072)
        
        
        # Assigning a Call to a Name (line 570):
        
        # Call to text(...): (line 570)
        # Processing the call arguments (line 570)
        str_14076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 20), 'str', 'ActionData')
        int_14077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 34), 'int')
        int_14078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 38), 'int')
        int_14079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 43), 'int')
        int_14080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 48), 'int')
        int_14081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 52), 'int')
        # Getting the type of 'None' (line 570)
        None_14082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 61), 'None', False)
        # Processing the call keyword arguments (line 570)
        kwargs_14083 = {}
        # Getting the type of 'prep' (line 570)
        prep_14074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 10), 'prep', False)
        # Obtaining the member 'text' of a type (line 570)
        text_14075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 10), prep_14074, 'text')
        # Calling text(args, kwargs) (line 570)
        text_call_result_14084 = invoke(stypy.reporting.localization.Localization(__file__, 570, 10), text_14075, *[str_14076, int_14077, int_14078, int_14079, int_14080, int_14081, None_14082], **kwargs_14083)
        
        # Assigning a type to the variable 'c' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'c', text_call_result_14084)
        
        # Call to mapping(...): (line 571)
        # Processing the call arguments (line 571)
        str_14087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 18), 'str', 'ActionData')
        str_14088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 32), 'str', 'Text')
        # Processing the call keyword arguments (line 571)
        kwargs_14089 = {}
        # Getting the type of 'c' (line 571)
        c_14085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'c', False)
        # Obtaining the member 'mapping' of a type (line 571)
        mapping_14086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 8), c_14085, 'mapping')
        # Calling mapping(args, kwargs) (line 571)
        mapping_call_result_14090 = invoke(stypy.reporting.localization.Localization(__file__, 571, 8), mapping_14086, *[str_14087, str_14088], **kwargs_14089)
        
        
        # Call to back(...): (line 572)
        # Processing the call arguments (line 572)
        str_14093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 18), 'str', 'Back')
        # Getting the type of 'None' (line 572)
        None_14094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 26), 'None', False)
        # Processing the call keyword arguments (line 572)
        int_14095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 39), 'int')
        keyword_14096 = int_14095
        kwargs_14097 = {'active': keyword_14096}
        # Getting the type of 'prep' (line 572)
        prep_14091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'prep', False)
        # Obtaining the member 'back' of a type (line 572)
        back_14092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 8), prep_14091, 'back')
        # Calling back(args, kwargs) (line 572)
        back_call_result_14098 = invoke(stypy.reporting.localization.Localization(__file__, 572, 8), back_14092, *[str_14093, None_14094], **kwargs_14097)
        
        
        # Call to next(...): (line 573)
        # Processing the call arguments (line 573)
        str_14101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 18), 'str', 'Next')
        # Getting the type of 'None' (line 573)
        None_14102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 26), 'None', False)
        # Processing the call keyword arguments (line 573)
        int_14103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 39), 'int')
        keyword_14104 = int_14103
        kwargs_14105 = {'active': keyword_14104}
        # Getting the type of 'prep' (line 573)
        prep_14099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'prep', False)
        # Obtaining the member 'next' of a type (line 573)
        next_14100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 8), prep_14099, 'next')
        # Calling next(args, kwargs) (line 573)
        next_call_result_14106 = invoke(stypy.reporting.localization.Localization(__file__, 573, 8), next_14100, *[str_14101, None_14102], **kwargs_14105)
        
        
        # Assigning a Call to a Name (line 574):
        
        # Call to cancel(...): (line 574)
        # Processing the call arguments (line 574)
        str_14109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 22), 'str', 'Cancel')
        # Getting the type of 'None' (line 574)
        None_14110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 32), 'None', False)
        # Processing the call keyword arguments (line 574)
        kwargs_14111 = {}
        # Getting the type of 'prep' (line 574)
        prep_14107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 10), 'prep', False)
        # Obtaining the member 'cancel' of a type (line 574)
        cancel_14108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 10), prep_14107, 'cancel')
        # Calling cancel(args, kwargs) (line 574)
        cancel_call_result_14112 = invoke(stypy.reporting.localization.Localization(__file__, 574, 10), cancel_14108, *[str_14109, None_14110], **kwargs_14111)
        
        # Assigning a type to the variable 'c' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'c', cancel_call_result_14112)
        
        # Call to event(...): (line 575)
        # Processing the call arguments (line 575)
        str_14115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 16), 'str', 'SpawnDialog')
        str_14116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 31), 'str', 'CancelDlg')
        # Processing the call keyword arguments (line 575)
        kwargs_14117 = {}
        # Getting the type of 'c' (line 575)
        c_14113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 575)
        event_14114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 8), c_14113, 'event')
        # Calling event(args, kwargs) (line 575)
        event_call_result_14118 = invoke(stypy.reporting.localization.Localization(__file__, 575, 8), event_14114, *[str_14115, str_14116], **kwargs_14117)
        
        
        # Assigning a Call to a Name (line 579):
        
        # Call to PyDialog(...): (line 579)
        # Processing the call arguments (line 579)
        # Getting the type of 'db' (line 579)
        db_14120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 26), 'db', False)
        str_14121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 30), 'str', 'SelectFeaturesDlg')
        # Getting the type of 'x' (line 579)
        x_14122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 51), 'x', False)
        # Getting the type of 'y' (line 579)
        y_14123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 54), 'y', False)
        # Getting the type of 'w' (line 579)
        w_14124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 57), 'w', False)
        # Getting the type of 'h' (line 579)
        h_14125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 60), 'h', False)
        # Getting the type of 'modal' (line 579)
        modal_14126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 63), 'modal', False)
        # Getting the type of 'title' (line 579)
        title_14127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 70), 'title', False)
        str_14128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 24), 'str', 'Next')
        str_14129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 32), 'str', 'Next')
        str_14130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 40), 'str', 'Cancel')
        # Processing the call keyword arguments (line 579)
        kwargs_14131 = {}
        # Getting the type of 'PyDialog' (line 579)
        PyDialog_14119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 17), 'PyDialog', False)
        # Calling PyDialog(args, kwargs) (line 579)
        PyDialog_call_result_14132 = invoke(stypy.reporting.localization.Localization(__file__, 579, 17), PyDialog_14119, *[db_14120, str_14121, x_14122, y_14123, w_14124, h_14125, modal_14126, title_14127, str_14128, str_14129, str_14130], **kwargs_14131)
        
        # Assigning a type to the variable 'seldlg' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'seldlg', PyDialog_call_result_14132)
        
        # Call to title(...): (line 581)
        # Processing the call arguments (line 581)
        str_14135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 21), 'str', 'Select Python Installations')
        # Processing the call keyword arguments (line 581)
        kwargs_14136 = {}
        # Getting the type of 'seldlg' (line 581)
        seldlg_14133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'seldlg', False)
        # Obtaining the member 'title' of a type (line 581)
        title_14134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), seldlg_14133, 'title')
        # Calling title(args, kwargs) (line 581)
        title_call_result_14137 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), title_14134, *[str_14135], **kwargs_14136)
        
        
        # Call to text(...): (line 583)
        # Processing the call arguments (line 583)
        str_14140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 20), 'str', 'Hint')
        int_14141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 28), 'int')
        int_14142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 32), 'int')
        int_14143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 36), 'int')
        int_14144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 41), 'int')
        int_14145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 45), 'int')
        str_14146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 20), 'str', 'Select the Python locations where %s should be installed.')
        
        # Call to get_fullname(...): (line 585)
        # Processing the call keyword arguments (line 585)
        kwargs_14150 = {}
        # Getting the type of 'self' (line 585)
        self_14147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 22), 'self', False)
        # Obtaining the member 'distribution' of a type (line 585)
        distribution_14148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 22), self_14147, 'distribution')
        # Obtaining the member 'get_fullname' of a type (line 585)
        get_fullname_14149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 22), distribution_14148, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 585)
        get_fullname_call_result_14151 = invoke(stypy.reporting.localization.Localization(__file__, 585, 22), get_fullname_14149, *[], **kwargs_14150)
        
        # Applying the binary operator '%' (line 584)
        result_mod_14152 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 20), '%', str_14146, get_fullname_call_result_14151)
        
        # Processing the call keyword arguments (line 583)
        kwargs_14153 = {}
        # Getting the type of 'seldlg' (line 583)
        seldlg_14138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'seldlg', False)
        # Obtaining the member 'text' of a type (line 583)
        text_14139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 8), seldlg_14138, 'text')
        # Calling text(args, kwargs) (line 583)
        text_call_result_14154 = invoke(stypy.reporting.localization.Localization(__file__, 583, 8), text_14139, *[str_14140, int_14141, int_14142, int_14143, int_14144, int_14145, result_mod_14152], **kwargs_14153)
        
        
        # Call to back(...): (line 587)
        # Processing the call arguments (line 587)
        str_14157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 20), 'str', '< Back')
        # Getting the type of 'None' (line 587)
        None_14158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'None', False)
        # Processing the call keyword arguments (line 587)
        int_14159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 43), 'int')
        keyword_14160 = int_14159
        kwargs_14161 = {'active': keyword_14160}
        # Getting the type of 'seldlg' (line 587)
        seldlg_14155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'seldlg', False)
        # Obtaining the member 'back' of a type (line 587)
        back_14156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 8), seldlg_14155, 'back')
        # Calling back(args, kwargs) (line 587)
        back_call_result_14162 = invoke(stypy.reporting.localization.Localization(__file__, 587, 8), back_14156, *[str_14157, None_14158], **kwargs_14161)
        
        
        # Assigning a Call to a Name (line 588):
        
        # Call to next(...): (line 588)
        # Processing the call arguments (line 588)
        str_14165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 24), 'str', 'Next >')
        str_14166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 34), 'str', 'Cancel')
        # Processing the call keyword arguments (line 588)
        kwargs_14167 = {}
        # Getting the type of 'seldlg' (line 588)
        seldlg_14163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'seldlg', False)
        # Obtaining the member 'next' of a type (line 588)
        next_14164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 12), seldlg_14163, 'next')
        # Calling next(args, kwargs) (line 588)
        next_call_result_14168 = invoke(stypy.reporting.localization.Localization(__file__, 588, 12), next_14164, *[str_14165, str_14166], **kwargs_14167)
        
        # Assigning a type to the variable 'c' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'c', next_call_result_14168)
        
        # Assigning a Num to a Name (line 589):
        int_14169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 16), 'int')
        # Assigning a type to the variable 'order' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'order', int_14169)
        
        # Call to event(...): (line 590)
        # Processing the call arguments (line 590)
        str_14172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 16), 'str', '[TARGETDIR]')
        str_14173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 31), 'str', '[SourceDir]')
        # Processing the call keyword arguments (line 590)
        # Getting the type of 'order' (line 590)
        order_14174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 55), 'order', False)
        keyword_14175 = order_14174
        kwargs_14176 = {'ordering': keyword_14175}
        # Getting the type of 'c' (line 590)
        c_14170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 590)
        event_14171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 8), c_14170, 'event')
        # Calling event(args, kwargs) (line 590)
        event_call_result_14177 = invoke(stypy.reporting.localization.Localization(__file__, 590, 8), event_14171, *[str_14172, str_14173], **kwargs_14176)
        
        
        # Getting the type of 'self' (line 591)
        self_14178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 23), 'self')
        # Obtaining the member 'versions' of a type (line 591)
        versions_14179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 23), self_14178, 'versions')
        
        # Obtaining an instance of the builtin type 'list' (line 591)
        list_14180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 591)
        # Adding element type (line 591)
        # Getting the type of 'self' (line 591)
        self_14181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 40), 'self')
        # Obtaining the member 'other_version' of a type (line 591)
        other_version_14182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 40), self_14181, 'other_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 39), list_14180, other_version_14182)
        
        # Applying the binary operator '+' (line 591)
        result_add_14183 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 23), '+', versions_14179, list_14180)
        
        # Testing the type of a for loop iterable (line 591)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 591, 8), result_add_14183)
        # Getting the type of the for loop variable (line 591)
        for_loop_var_14184 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 591, 8), result_add_14183)
        # Assigning a type to the variable 'version' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'version', for_loop_var_14184)
        # SSA begins for a for statement (line 591)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'order' (line 592)
        order_14185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'order')
        int_14186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 21), 'int')
        # Applying the binary operator '+=' (line 592)
        result_iadd_14187 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 12), '+=', order_14185, int_14186)
        # Assigning a type to the variable 'order' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'order', result_iadd_14187)
        
        
        # Call to event(...): (line 593)
        # Processing the call arguments (line 593)
        str_14190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 20), 'str', '[TARGETDIR]')
        str_14191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 35), 'str', '[TARGETDIR%s]')
        # Getting the type of 'version' (line 593)
        version_14192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 53), 'version', False)
        # Applying the binary operator '%' (line 593)
        result_mod_14193 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 35), '%', str_14191, version_14192)
        
        str_14194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 20), 'str', 'FEATURE_SELECTED AND &Python%s=3')
        # Getting the type of 'version' (line 594)
        version_14195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 57), 'version', False)
        # Applying the binary operator '%' (line 594)
        result_mod_14196 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 20), '%', str_14194, version_14195)
        
        # Processing the call keyword arguments (line 593)
        # Getting the type of 'order' (line 595)
        order_14197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 29), 'order', False)
        keyword_14198 = order_14197
        kwargs_14199 = {'ordering': keyword_14198}
        # Getting the type of 'c' (line 593)
        c_14188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'c', False)
        # Obtaining the member 'event' of a type (line 593)
        event_14189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 12), c_14188, 'event')
        # Calling event(args, kwargs) (line 593)
        event_call_result_14200 = invoke(stypy.reporting.localization.Localization(__file__, 593, 12), event_14189, *[str_14190, result_mod_14193, result_mod_14196], **kwargs_14199)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to event(...): (line 596)
        # Processing the call arguments (line 596)
        str_14203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 16), 'str', 'SpawnWaitDialog')
        str_14204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 35), 'str', 'WaitForCostingDlg')
        # Processing the call keyword arguments (line 596)
        # Getting the type of 'order' (line 596)
        order_14205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 65), 'order', False)
        int_14206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 73), 'int')
        # Applying the binary operator '+' (line 596)
        result_add_14207 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 65), '+', order_14205, int_14206)
        
        keyword_14208 = result_add_14207
        kwargs_14209 = {'ordering': keyword_14208}
        # Getting the type of 'c' (line 596)
        c_14201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 596)
        event_14202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 8), c_14201, 'event')
        # Calling event(args, kwargs) (line 596)
        event_call_result_14210 = invoke(stypy.reporting.localization.Localization(__file__, 596, 8), event_14202, *[str_14203, str_14204], **kwargs_14209)
        
        
        # Call to event(...): (line 597)
        # Processing the call arguments (line 597)
        str_14213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 16), 'str', 'EndDialog')
        str_14214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 29), 'str', 'Return')
        # Processing the call keyword arguments (line 597)
        # Getting the type of 'order' (line 597)
        order_14215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 48), 'order', False)
        int_14216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 56), 'int')
        # Applying the binary operator '+' (line 597)
        result_add_14217 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 48), '+', order_14215, int_14216)
        
        keyword_14218 = result_add_14217
        kwargs_14219 = {'ordering': keyword_14218}
        # Getting the type of 'c' (line 597)
        c_14211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 597)
        event_14212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 8), c_14211, 'event')
        # Calling event(args, kwargs) (line 597)
        event_call_result_14220 = invoke(stypy.reporting.localization.Localization(__file__, 597, 8), event_14212, *[str_14213, str_14214], **kwargs_14219)
        
        
        # Assigning a Call to a Name (line 598):
        
        # Call to cancel(...): (line 598)
        # Processing the call arguments (line 598)
        str_14223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 26), 'str', 'Cancel')
        str_14224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 36), 'str', 'Features')
        # Processing the call keyword arguments (line 598)
        kwargs_14225 = {}
        # Getting the type of 'seldlg' (line 598)
        seldlg_14221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'seldlg', False)
        # Obtaining the member 'cancel' of a type (line 598)
        cancel_14222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 12), seldlg_14221, 'cancel')
        # Calling cancel(args, kwargs) (line 598)
        cancel_call_result_14226 = invoke(stypy.reporting.localization.Localization(__file__, 598, 12), cancel_14222, *[str_14223, str_14224], **kwargs_14225)
        
        # Assigning a type to the variable 'c' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'c', cancel_call_result_14226)
        
        # Call to event(...): (line 599)
        # Processing the call arguments (line 599)
        str_14229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 16), 'str', 'SpawnDialog')
        str_14230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 31), 'str', 'CancelDlg')
        # Processing the call keyword arguments (line 599)
        kwargs_14231 = {}
        # Getting the type of 'c' (line 599)
        c_14227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 599)
        event_14228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 8), c_14227, 'event')
        # Calling event(args, kwargs) (line 599)
        event_call_result_14232 = invoke(stypy.reporting.localization.Localization(__file__, 599, 8), event_14228, *[str_14229, str_14230], **kwargs_14231)
        
        
        # Assigning a Call to a Name (line 601):
        
        # Call to control(...): (line 601)
        # Processing the call arguments (line 601)
        str_14235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 27), 'str', 'Features')
        str_14236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 39), 'str', 'SelectionTree')
        int_14237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 56), 'int')
        int_14238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 60), 'int')
        int_14239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 64), 'int')
        int_14240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 69), 'int')
        int_14241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 74), 'int')
        str_14242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 27), 'str', 'FEATURE')
        # Getting the type of 'None' (line 602)
        None_14243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 38), 'None', False)
        str_14244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 44), 'str', 'PathEdit')
        # Getting the type of 'None' (line 602)
        None_14245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 56), 'None', False)
        # Processing the call keyword arguments (line 601)
        kwargs_14246 = {}
        # Getting the type of 'seldlg' (line 601)
        seldlg_14233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'seldlg', False)
        # Obtaining the member 'control' of a type (line 601)
        control_14234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 12), seldlg_14233, 'control')
        # Calling control(args, kwargs) (line 601)
        control_call_result_14247 = invoke(stypy.reporting.localization.Localization(__file__, 601, 12), control_14234, *[str_14235, str_14236, int_14237, int_14238, int_14239, int_14240, int_14241, str_14242, None_14243, str_14244, None_14245], **kwargs_14246)
        
        # Assigning a type to the variable 'c' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'c', control_call_result_14247)
        
        # Call to event(...): (line 603)
        # Processing the call arguments (line 603)
        str_14250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 16), 'str', '[FEATURE_SELECTED]')
        str_14251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 38), 'str', '1')
        # Processing the call keyword arguments (line 603)
        kwargs_14252 = {}
        # Getting the type of 'c' (line 603)
        c_14248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 603)
        event_14249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 8), c_14248, 'event')
        # Calling event(args, kwargs) (line 603)
        event_call_result_14253 = invoke(stypy.reporting.localization.Localization(__file__, 603, 8), event_14249, *[str_14250, str_14251], **kwargs_14252)
        
        
        # Assigning a Attribute to a Name (line 604):
        # Getting the type of 'self' (line 604)
        self_14254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 14), 'self')
        # Obtaining the member 'other_version' of a type (line 604)
        other_version_14255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 14), self_14254, 'other_version')
        # Assigning a type to the variable 'ver' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'ver', other_version_14255)
        
        # Assigning a BinOp to a Name (line 605):
        str_14256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 29), 'str', 'FEATURE_SELECTED AND &Python%s=3')
        # Getting the type of 'ver' (line 605)
        ver_14257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 66), 'ver')
        # Applying the binary operator '%' (line 605)
        result_mod_14258 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 29), '%', str_14256, ver_14257)
        
        # Assigning a type to the variable 'install_other_cond' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'install_other_cond', result_mod_14258)
        
        # Assigning a BinOp to a Name (line 606):
        str_14259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 34), 'str', 'FEATURE_SELECTED AND &Python%s<>3')
        # Getting the type of 'ver' (line 606)
        ver_14260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 72), 'ver')
        # Applying the binary operator '%' (line 606)
        result_mod_14261 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 34), '%', str_14259, ver_14260)
        
        # Assigning a type to the variable 'dont_install_other_cond' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'dont_install_other_cond', result_mod_14261)
        
        # Assigning a Call to a Name (line 608):
        
        # Call to text(...): (line 608)
        # Processing the call arguments (line 608)
        str_14264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 24), 'str', 'Other')
        int_14265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 33), 'int')
        int_14266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 37), 'int')
        int_14267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 42), 'int')
        int_14268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 47), 'int')
        int_14269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 51), 'int')
        str_14270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 24), 'str', 'Provide an alternate Python location')
        # Processing the call keyword arguments (line 608)
        kwargs_14271 = {}
        # Getting the type of 'seldlg' (line 608)
        seldlg_14262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'seldlg', False)
        # Obtaining the member 'text' of a type (line 608)
        text_14263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 12), seldlg_14262, 'text')
        # Calling text(args, kwargs) (line 608)
        text_call_result_14272 = invoke(stypy.reporting.localization.Localization(__file__, 608, 12), text_14263, *[str_14264, int_14265, int_14266, int_14267, int_14268, int_14269, str_14270], **kwargs_14271)
        
        # Assigning a type to the variable 'c' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'c', text_call_result_14272)
        
        # Call to condition(...): (line 610)
        # Processing the call arguments (line 610)
        str_14275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 20), 'str', 'Enable')
        # Getting the type of 'install_other_cond' (line 610)
        install_other_cond_14276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 30), 'install_other_cond', False)
        # Processing the call keyword arguments (line 610)
        kwargs_14277 = {}
        # Getting the type of 'c' (line 610)
        c_14273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'c', False)
        # Obtaining the member 'condition' of a type (line 610)
        condition_14274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 8), c_14273, 'condition')
        # Calling condition(args, kwargs) (line 610)
        condition_call_result_14278 = invoke(stypy.reporting.localization.Localization(__file__, 610, 8), condition_14274, *[str_14275, install_other_cond_14276], **kwargs_14277)
        
        
        # Call to condition(...): (line 611)
        # Processing the call arguments (line 611)
        str_14281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 20), 'str', 'Show')
        # Getting the type of 'install_other_cond' (line 611)
        install_other_cond_14282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 28), 'install_other_cond', False)
        # Processing the call keyword arguments (line 611)
        kwargs_14283 = {}
        # Getting the type of 'c' (line 611)
        c_14279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'c', False)
        # Obtaining the member 'condition' of a type (line 611)
        condition_14280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 8), c_14279, 'condition')
        # Calling condition(args, kwargs) (line 611)
        condition_call_result_14284 = invoke(stypy.reporting.localization.Localization(__file__, 611, 8), condition_14280, *[str_14281, install_other_cond_14282], **kwargs_14283)
        
        
        # Call to condition(...): (line 612)
        # Processing the call arguments (line 612)
        str_14287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 20), 'str', 'Disable')
        # Getting the type of 'dont_install_other_cond' (line 612)
        dont_install_other_cond_14288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 31), 'dont_install_other_cond', False)
        # Processing the call keyword arguments (line 612)
        kwargs_14289 = {}
        # Getting the type of 'c' (line 612)
        c_14285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'c', False)
        # Obtaining the member 'condition' of a type (line 612)
        condition_14286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 8), c_14285, 'condition')
        # Calling condition(args, kwargs) (line 612)
        condition_call_result_14290 = invoke(stypy.reporting.localization.Localization(__file__, 612, 8), condition_14286, *[str_14287, dont_install_other_cond_14288], **kwargs_14289)
        
        
        # Call to condition(...): (line 613)
        # Processing the call arguments (line 613)
        str_14293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 20), 'str', 'Hide')
        # Getting the type of 'dont_install_other_cond' (line 613)
        dont_install_other_cond_14294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 28), 'dont_install_other_cond', False)
        # Processing the call keyword arguments (line 613)
        kwargs_14295 = {}
        # Getting the type of 'c' (line 613)
        c_14291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'c', False)
        # Obtaining the member 'condition' of a type (line 613)
        condition_14292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 8), c_14291, 'condition')
        # Calling condition(args, kwargs) (line 613)
        condition_call_result_14296 = invoke(stypy.reporting.localization.Localization(__file__, 613, 8), condition_14292, *[str_14293, dont_install_other_cond_14294], **kwargs_14295)
        
        
        # Assigning a Call to a Name (line 615):
        
        # Call to control(...): (line 615)
        # Processing the call arguments (line 615)
        str_14299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 27), 'str', 'PathEdit')
        str_14300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 39), 'str', 'PathEdit')
        int_14301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 51), 'int')
        int_14302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 55), 'int')
        int_14303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 60), 'int')
        int_14304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 65), 'int')
        int_14305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 69), 'int')
        str_14306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 27), 'str', 'TARGETDIR')
        # Getting the type of 'ver' (line 616)
        ver_14307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 41), 'ver', False)
        # Applying the binary operator '+' (line 616)
        result_add_14308 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 27), '+', str_14306, ver_14307)
        
        # Getting the type of 'None' (line 616)
        None_14309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 46), 'None', False)
        str_14310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 52), 'str', 'Next')
        # Getting the type of 'None' (line 616)
        None_14311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 60), 'None', False)
        # Processing the call keyword arguments (line 615)
        kwargs_14312 = {}
        # Getting the type of 'seldlg' (line 615)
        seldlg_14297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'seldlg', False)
        # Obtaining the member 'control' of a type (line 615)
        control_14298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 12), seldlg_14297, 'control')
        # Calling control(args, kwargs) (line 615)
        control_call_result_14313 = invoke(stypy.reporting.localization.Localization(__file__, 615, 12), control_14298, *[str_14299, str_14300, int_14301, int_14302, int_14303, int_14304, int_14305, result_add_14308, None_14309, str_14310, None_14311], **kwargs_14312)
        
        # Assigning a type to the variable 'c' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'c', control_call_result_14313)
        
        # Call to condition(...): (line 617)
        # Processing the call arguments (line 617)
        str_14316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 20), 'str', 'Enable')
        # Getting the type of 'install_other_cond' (line 617)
        install_other_cond_14317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 30), 'install_other_cond', False)
        # Processing the call keyword arguments (line 617)
        kwargs_14318 = {}
        # Getting the type of 'c' (line 617)
        c_14314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'c', False)
        # Obtaining the member 'condition' of a type (line 617)
        condition_14315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 8), c_14314, 'condition')
        # Calling condition(args, kwargs) (line 617)
        condition_call_result_14319 = invoke(stypy.reporting.localization.Localization(__file__, 617, 8), condition_14315, *[str_14316, install_other_cond_14317], **kwargs_14318)
        
        
        # Call to condition(...): (line 618)
        # Processing the call arguments (line 618)
        str_14322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 20), 'str', 'Show')
        # Getting the type of 'install_other_cond' (line 618)
        install_other_cond_14323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 28), 'install_other_cond', False)
        # Processing the call keyword arguments (line 618)
        kwargs_14324 = {}
        # Getting the type of 'c' (line 618)
        c_14320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'c', False)
        # Obtaining the member 'condition' of a type (line 618)
        condition_14321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 8), c_14320, 'condition')
        # Calling condition(args, kwargs) (line 618)
        condition_call_result_14325 = invoke(stypy.reporting.localization.Localization(__file__, 618, 8), condition_14321, *[str_14322, install_other_cond_14323], **kwargs_14324)
        
        
        # Call to condition(...): (line 619)
        # Processing the call arguments (line 619)
        str_14328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 20), 'str', 'Disable')
        # Getting the type of 'dont_install_other_cond' (line 619)
        dont_install_other_cond_14329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 31), 'dont_install_other_cond', False)
        # Processing the call keyword arguments (line 619)
        kwargs_14330 = {}
        # Getting the type of 'c' (line 619)
        c_14326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'c', False)
        # Obtaining the member 'condition' of a type (line 619)
        condition_14327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 8), c_14326, 'condition')
        # Calling condition(args, kwargs) (line 619)
        condition_call_result_14331 = invoke(stypy.reporting.localization.Localization(__file__, 619, 8), condition_14327, *[str_14328, dont_install_other_cond_14329], **kwargs_14330)
        
        
        # Call to condition(...): (line 620)
        # Processing the call arguments (line 620)
        str_14334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 20), 'str', 'Hide')
        # Getting the type of 'dont_install_other_cond' (line 620)
        dont_install_other_cond_14335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 28), 'dont_install_other_cond', False)
        # Processing the call keyword arguments (line 620)
        kwargs_14336 = {}
        # Getting the type of 'c' (line 620)
        c_14332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'c', False)
        # Obtaining the member 'condition' of a type (line 620)
        condition_14333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 8), c_14332, 'condition')
        # Calling condition(args, kwargs) (line 620)
        condition_call_result_14337 = invoke(stypy.reporting.localization.Localization(__file__, 620, 8), condition_14333, *[str_14334, dont_install_other_cond_14335], **kwargs_14336)
        
        
        # Assigning a Call to a Name (line 624):
        
        # Call to PyDialog(...): (line 624)
        # Processing the call arguments (line 624)
        # Getting the type of 'db' (line 624)
        db_14339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 24), 'db', False)
        str_14340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 28), 'str', 'DiskCostDlg')
        # Getting the type of 'x' (line 624)
        x_14341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 43), 'x', False)
        # Getting the type of 'y' (line 624)
        y_14342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 46), 'y', False)
        # Getting the type of 'w' (line 624)
        w_14343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 49), 'w', False)
        # Getting the type of 'h' (line 624)
        h_14344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 52), 'h', False)
        # Getting the type of 'modal' (line 624)
        modal_14345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 55), 'modal', False)
        # Getting the type of 'title' (line 624)
        title_14346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 62), 'title', False)
        str_14347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 24), 'str', 'OK')
        str_14348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 30), 'str', 'OK')
        str_14349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 36), 'str', 'OK')
        # Processing the call keyword arguments (line 624)
        # Getting the type of 'False' (line 625)
        False_14350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 49), 'False', False)
        keyword_14351 = False_14350
        kwargs_14352 = {'bitmap': keyword_14351}
        # Getting the type of 'PyDialog' (line 624)
        PyDialog_14338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 15), 'PyDialog', False)
        # Calling PyDialog(args, kwargs) (line 624)
        PyDialog_call_result_14353 = invoke(stypy.reporting.localization.Localization(__file__, 624, 15), PyDialog_14338, *[db_14339, str_14340, x_14341, y_14342, w_14343, h_14344, modal_14345, title_14346, str_14347, str_14348, str_14349], **kwargs_14352)
        
        # Assigning a type to the variable 'cost' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'cost', PyDialog_call_result_14353)
        
        # Call to text(...): (line 626)
        # Processing the call arguments (line 626)
        str_14356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 18), 'str', 'Title')
        int_14357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 27), 'int')
        int_14358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 31), 'int')
        int_14359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 34), 'int')
        int_14360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 39), 'int')
        int_14361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 43), 'int')
        str_14362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 18), 'str', '{\\DlgFontBold8}Disk Space Requirements')
        # Processing the call keyword arguments (line 626)
        kwargs_14363 = {}
        # Getting the type of 'cost' (line 626)
        cost_14354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'cost', False)
        # Obtaining the member 'text' of a type (line 626)
        text_14355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 8), cost_14354, 'text')
        # Calling text(args, kwargs) (line 626)
        text_call_result_14364 = invoke(stypy.reporting.localization.Localization(__file__, 626, 8), text_14355, *[str_14356, int_14357, int_14358, int_14359, int_14360, int_14361, str_14362], **kwargs_14363)
        
        
        # Call to text(...): (line 628)
        # Processing the call arguments (line 628)
        str_14367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 18), 'str', 'Description')
        int_14368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 33), 'int')
        int_14369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 37), 'int')
        int_14370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 41), 'int')
        int_14371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 46), 'int')
        int_14372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 50), 'int')
        str_14373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 18), 'str', 'The disk space required for the installation of the selected features.')
        # Processing the call keyword arguments (line 628)
        kwargs_14374 = {}
        # Getting the type of 'cost' (line 628)
        cost_14365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'cost', False)
        # Obtaining the member 'text' of a type (line 628)
        text_14366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 8), cost_14365, 'text')
        # Calling text(args, kwargs) (line 628)
        text_call_result_14375 = invoke(stypy.reporting.localization.Localization(__file__, 628, 8), text_14366, *[str_14367, int_14368, int_14369, int_14370, int_14371, int_14372, str_14373], **kwargs_14374)
        
        
        # Call to text(...): (line 630)
        # Processing the call arguments (line 630)
        str_14378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 18), 'str', 'Text')
        int_14379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 26), 'int')
        int_14380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 30), 'int')
        int_14381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 34), 'int')
        int_14382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 39), 'int')
        int_14383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 43), 'int')
        str_14384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 18), 'str', 'The highlighted volumes (if any) do not have enough disk space available for the currently selected features.  You can either remove some files from the highlighted volumes, or choose to install less features onto local drive(s), or select different destination drive(s).')
        # Processing the call keyword arguments (line 630)
        kwargs_14385 = {}
        # Getting the type of 'cost' (line 630)
        cost_14376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'cost', False)
        # Obtaining the member 'text' of a type (line 630)
        text_14377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 8), cost_14376, 'text')
        # Calling text(args, kwargs) (line 630)
        text_call_result_14386 = invoke(stypy.reporting.localization.Localization(__file__, 630, 8), text_14377, *[str_14378, int_14379, int_14380, int_14381, int_14382, int_14383, str_14384], **kwargs_14385)
        
        
        # Call to control(...): (line 636)
        # Processing the call arguments (line 636)
        str_14389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 21), 'str', 'VolumeList')
        str_14390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 35), 'str', 'VolumeCostList')
        int_14391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 53), 'int')
        int_14392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 57), 'int')
        int_14393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 62), 'int')
        int_14394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 67), 'int')
        int_14395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 72), 'int')
        # Getting the type of 'None' (line 637)
        None_14396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 21), 'None', False)
        str_14397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 27), 'str', '{120}{70}{70}{70}{70}')
        # Getting the type of 'None' (line 637)
        None_14398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 52), 'None', False)
        # Getting the type of 'None' (line 637)
        None_14399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 58), 'None', False)
        # Processing the call keyword arguments (line 636)
        kwargs_14400 = {}
        # Getting the type of 'cost' (line 636)
        cost_14387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'cost', False)
        # Obtaining the member 'control' of a type (line 636)
        control_14388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 8), cost_14387, 'control')
        # Calling control(args, kwargs) (line 636)
        control_call_result_14401 = invoke(stypy.reporting.localization.Localization(__file__, 636, 8), control_14388, *[str_14389, str_14390, int_14391, int_14392, int_14393, int_14394, int_14395, None_14396, str_14397, None_14398, None_14399], **kwargs_14400)
        
        
        # Call to event(...): (line 638)
        # Processing the call arguments (line 638)
        str_14411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 50), 'str', 'EndDialog')
        str_14412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 63), 'str', 'Return')
        # Processing the call keyword arguments (line 638)
        kwargs_14413 = {}
        
        # Call to xbutton(...): (line 638)
        # Processing the call arguments (line 638)
        str_14404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 21), 'str', 'OK')
        str_14405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 27), 'str', 'Ok')
        # Getting the type of 'None' (line 638)
        None_14406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 33), 'None', False)
        float_14407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 39), 'float')
        # Processing the call keyword arguments (line 638)
        kwargs_14408 = {}
        # Getting the type of 'cost' (line 638)
        cost_14402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'cost', False)
        # Obtaining the member 'xbutton' of a type (line 638)
        xbutton_14403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 8), cost_14402, 'xbutton')
        # Calling xbutton(args, kwargs) (line 638)
        xbutton_call_result_14409 = invoke(stypy.reporting.localization.Localization(__file__, 638, 8), xbutton_14403, *[str_14404, str_14405, None_14406, float_14407], **kwargs_14408)
        
        # Obtaining the member 'event' of a type (line 638)
        event_14410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 8), xbutton_call_result_14409, 'event')
        # Calling event(args, kwargs) (line 638)
        event_call_result_14414 = invoke(stypy.reporting.localization.Localization(__file__, 638, 8), event_14410, *[str_14411, str_14412], **kwargs_14413)
        
        
        # Assigning a Call to a Name (line 651):
        
        # Call to PyDialog(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'db' (line 651)
        db_14416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 30), 'db', False)
        str_14417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 34), 'str', 'WhichUsersDlg')
        # Getting the type of 'x' (line 651)
        x_14418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 51), 'x', False)
        # Getting the type of 'y' (line 651)
        y_14419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 54), 'y', False)
        # Getting the type of 'w' (line 651)
        w_14420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 57), 'w', False)
        # Getting the type of 'h' (line 651)
        h_14421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 60), 'h', False)
        # Getting the type of 'modal' (line 651)
        modal_14422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 63), 'modal', False)
        # Getting the type of 'title' (line 651)
        title_14423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 70), 'title', False)
        str_14424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 28), 'str', 'AdminInstall')
        str_14425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 44), 'str', 'Next')
        str_14426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 52), 'str', 'Cancel')
        # Processing the call keyword arguments (line 651)
        kwargs_14427 = {}
        # Getting the type of 'PyDialog' (line 651)
        PyDialog_14415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 21), 'PyDialog', False)
        # Calling PyDialog(args, kwargs) (line 651)
        PyDialog_call_result_14428 = invoke(stypy.reporting.localization.Localization(__file__, 651, 21), PyDialog_14415, *[db_14416, str_14417, x_14418, y_14419, w_14420, h_14421, modal_14422, title_14423, str_14424, str_14425, str_14426], **kwargs_14427)
        
        # Assigning a type to the variable 'whichusers' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'whichusers', PyDialog_call_result_14428)
        
        # Call to title(...): (line 653)
        # Processing the call arguments (line 653)
        str_14431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 25), 'str', 'Select whether to install [ProductName] for all users of this computer.')
        # Processing the call keyword arguments (line 653)
        kwargs_14432 = {}
        # Getting the type of 'whichusers' (line 653)
        whichusers_14429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'whichusers', False)
        # Obtaining the member 'title' of a type (line 653)
        title_14430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 8), whichusers_14429, 'title')
        # Calling title(args, kwargs) (line 653)
        title_call_result_14433 = invoke(stypy.reporting.localization.Localization(__file__, 653, 8), title_14430, *[str_14431], **kwargs_14432)
        
        
        # Assigning a Call to a Name (line 655):
        
        # Call to radiogroup(...): (line 655)
        # Processing the call arguments (line 655)
        str_14436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 34), 'str', 'AdminInstall')
        int_14437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 50), 'int')
        int_14438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 54), 'int')
        int_14439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 58), 'int')
        int_14440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 63), 'int')
        int_14441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 67), 'int')
        str_14442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 34), 'str', 'WhichUsers')
        str_14443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 48), 'str', '')
        str_14444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 52), 'str', 'Next')
        # Processing the call keyword arguments (line 655)
        kwargs_14445 = {}
        # Getting the type of 'whichusers' (line 655)
        whichusers_14434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 12), 'whichusers', False)
        # Obtaining the member 'radiogroup' of a type (line 655)
        radiogroup_14435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 12), whichusers_14434, 'radiogroup')
        # Calling radiogroup(args, kwargs) (line 655)
        radiogroup_call_result_14446 = invoke(stypy.reporting.localization.Localization(__file__, 655, 12), radiogroup_14435, *[str_14436, int_14437, int_14438, int_14439, int_14440, int_14441, str_14442, str_14443, str_14444], **kwargs_14445)
        
        # Assigning a type to the variable 'g' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'g', radiogroup_call_result_14446)
        
        # Call to add(...): (line 657)
        # Processing the call arguments (line 657)
        str_14449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 14), 'str', 'ALL')
        int_14450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 21), 'int')
        int_14451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 24), 'int')
        int_14452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 27), 'int')
        int_14453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 32), 'int')
        str_14454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 36), 'str', 'Install for all users')
        # Processing the call keyword arguments (line 657)
        kwargs_14455 = {}
        # Getting the type of 'g' (line 657)
        g_14447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'g', False)
        # Obtaining the member 'add' of a type (line 657)
        add_14448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 8), g_14447, 'add')
        # Calling add(args, kwargs) (line 657)
        add_call_result_14456 = invoke(stypy.reporting.localization.Localization(__file__, 657, 8), add_14448, *[str_14449, int_14450, int_14451, int_14452, int_14453, str_14454], **kwargs_14455)
        
        
        # Call to add(...): (line 658)
        # Processing the call arguments (line 658)
        str_14459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 14), 'str', 'JUSTME')
        int_14460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 24), 'int')
        int_14461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 27), 'int')
        int_14462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 31), 'int')
        int_14463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 36), 'int')
        str_14464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 40), 'str', 'Install just for me')
        # Processing the call keyword arguments (line 658)
        kwargs_14465 = {}
        # Getting the type of 'g' (line 658)
        g_14457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'g', False)
        # Obtaining the member 'add' of a type (line 658)
        add_14458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 8), g_14457, 'add')
        # Calling add(args, kwargs) (line 658)
        add_call_result_14466 = invoke(stypy.reporting.localization.Localization(__file__, 658, 8), add_14458, *[str_14459, int_14460, int_14461, int_14462, int_14463, str_14464], **kwargs_14465)
        
        
        # Call to back(...): (line 660)
        # Processing the call arguments (line 660)
        str_14469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 24), 'str', 'Back')
        # Getting the type of 'None' (line 660)
        None_14470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 32), 'None', False)
        # Processing the call keyword arguments (line 660)
        int_14471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 45), 'int')
        keyword_14472 = int_14471
        kwargs_14473 = {'active': keyword_14472}
        # Getting the type of 'whichusers' (line 660)
        whichusers_14467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'whichusers', False)
        # Obtaining the member 'back' of a type (line 660)
        back_14468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 8), whichusers_14467, 'back')
        # Calling back(args, kwargs) (line 660)
        back_call_result_14474 = invoke(stypy.reporting.localization.Localization(__file__, 660, 8), back_14468, *[str_14469, None_14470], **kwargs_14473)
        
        
        # Assigning a Call to a Name (line 662):
        
        # Call to next(...): (line 662)
        # Processing the call arguments (line 662)
        str_14477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 28), 'str', 'Next >')
        str_14478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 38), 'str', 'Cancel')
        # Processing the call keyword arguments (line 662)
        kwargs_14479 = {}
        # Getting the type of 'whichusers' (line 662)
        whichusers_14475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'whichusers', False)
        # Obtaining the member 'next' of a type (line 662)
        next_14476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 12), whichusers_14475, 'next')
        # Calling next(args, kwargs) (line 662)
        next_call_result_14480 = invoke(stypy.reporting.localization.Localization(__file__, 662, 12), next_14476, *[str_14477, str_14478], **kwargs_14479)
        
        # Assigning a type to the variable 'c' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'c', next_call_result_14480)
        
        # Call to event(...): (line 663)
        # Processing the call arguments (line 663)
        str_14483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 16), 'str', '[ALLUSERS]')
        str_14484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 30), 'str', '1')
        str_14485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 35), 'str', 'WhichUsers="ALL"')
        int_14486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 55), 'int')
        # Processing the call keyword arguments (line 663)
        kwargs_14487 = {}
        # Getting the type of 'c' (line 663)
        c_14481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 663)
        event_14482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 8), c_14481, 'event')
        # Calling event(args, kwargs) (line 663)
        event_call_result_14488 = invoke(stypy.reporting.localization.Localization(__file__, 663, 8), event_14482, *[str_14483, str_14484, str_14485, int_14486], **kwargs_14487)
        
        
        # Call to event(...): (line 664)
        # Processing the call arguments (line 664)
        str_14491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 16), 'str', 'EndDialog')
        str_14492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 29), 'str', 'Return')
        # Processing the call keyword arguments (line 664)
        int_14493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 50), 'int')
        keyword_14494 = int_14493
        kwargs_14495 = {'ordering': keyword_14494}
        # Getting the type of 'c' (line 664)
        c_14489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 664)
        event_14490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 8), c_14489, 'event')
        # Calling event(args, kwargs) (line 664)
        event_call_result_14496 = invoke(stypy.reporting.localization.Localization(__file__, 664, 8), event_14490, *[str_14491, str_14492], **kwargs_14495)
        
        
        # Assigning a Call to a Name (line 666):
        
        # Call to cancel(...): (line 666)
        # Processing the call arguments (line 666)
        str_14499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 30), 'str', 'Cancel')
        str_14500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 40), 'str', 'AdminInstall')
        # Processing the call keyword arguments (line 666)
        kwargs_14501 = {}
        # Getting the type of 'whichusers' (line 666)
        whichusers_14497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'whichusers', False)
        # Obtaining the member 'cancel' of a type (line 666)
        cancel_14498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 12), whichusers_14497, 'cancel')
        # Calling cancel(args, kwargs) (line 666)
        cancel_call_result_14502 = invoke(stypy.reporting.localization.Localization(__file__, 666, 12), cancel_14498, *[str_14499, str_14500], **kwargs_14501)
        
        # Assigning a type to the variable 'c' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'c', cancel_call_result_14502)
        
        # Call to event(...): (line 667)
        # Processing the call arguments (line 667)
        str_14505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 16), 'str', 'SpawnDialog')
        str_14506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 31), 'str', 'CancelDlg')
        # Processing the call keyword arguments (line 667)
        kwargs_14507 = {}
        # Getting the type of 'c' (line 667)
        c_14503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 667)
        event_14504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 8), c_14503, 'event')
        # Calling event(args, kwargs) (line 667)
        event_call_result_14508 = invoke(stypy.reporting.localization.Localization(__file__, 667, 8), event_14504, *[str_14505, str_14506], **kwargs_14507)
        
        
        # Assigning a Call to a Name (line 671):
        
        # Call to PyDialog(...): (line 671)
        # Processing the call arguments (line 671)
        # Getting the type of 'db' (line 671)
        db_14510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 28), 'db', False)
        str_14511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 32), 'str', 'ProgressDlg')
        # Getting the type of 'x' (line 671)
        x_14512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 47), 'x', False)
        # Getting the type of 'y' (line 671)
        y_14513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 50), 'y', False)
        # Getting the type of 'w' (line 671)
        w_14514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 53), 'w', False)
        # Getting the type of 'h' (line 671)
        h_14515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 56), 'h', False)
        # Getting the type of 'modeless' (line 671)
        modeless_14516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 59), 'modeless', False)
        # Getting the type of 'title' (line 671)
        title_14517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 69), 'title', False)
        str_14518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 28), 'str', 'Cancel')
        str_14519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 38), 'str', 'Cancel')
        str_14520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 48), 'str', 'Cancel')
        # Processing the call keyword arguments (line 671)
        # Getting the type of 'False' (line 672)
        False_14521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 65), 'False', False)
        keyword_14522 = False_14521
        kwargs_14523 = {'bitmap': keyword_14522}
        # Getting the type of 'PyDialog' (line 671)
        PyDialog_14509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 19), 'PyDialog', False)
        # Calling PyDialog(args, kwargs) (line 671)
        PyDialog_call_result_14524 = invoke(stypy.reporting.localization.Localization(__file__, 671, 19), PyDialog_14509, *[db_14510, str_14511, x_14512, y_14513, w_14514, h_14515, modeless_14516, title_14517, str_14518, str_14519, str_14520], **kwargs_14523)
        
        # Assigning a type to the variable 'progress' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'progress', PyDialog_call_result_14524)
        
        # Call to text(...): (line 673)
        # Processing the call arguments (line 673)
        str_14527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 22), 'str', 'Title')
        int_14528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 31), 'int')
        int_14529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 35), 'int')
        int_14530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 39), 'int')
        int_14531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 44), 'int')
        int_14532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 48), 'int')
        str_14533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 22), 'str', '{\\DlgFontBold8}[Progress1] [ProductName]')
        # Processing the call keyword arguments (line 673)
        kwargs_14534 = {}
        # Getting the type of 'progress' (line 673)
        progress_14525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'progress', False)
        # Obtaining the member 'text' of a type (line 673)
        text_14526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 8), progress_14525, 'text')
        # Calling text(args, kwargs) (line 673)
        text_call_result_14535 = invoke(stypy.reporting.localization.Localization(__file__, 673, 8), text_14526, *[str_14527, int_14528, int_14529, int_14530, int_14531, int_14532, str_14533], **kwargs_14534)
        
        
        # Call to text(...): (line 675)
        # Processing the call arguments (line 675)
        str_14538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 22), 'str', 'Text')
        int_14539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 30), 'int')
        int_14540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 34), 'int')
        int_14541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 38), 'int')
        int_14542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 43), 'int')
        int_14543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 47), 'int')
        str_14544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 22), 'str', 'Please wait while the Installer [Progress2] [ProductName]. This may take several minutes.')
        # Processing the call keyword arguments (line 675)
        kwargs_14545 = {}
        # Getting the type of 'progress' (line 675)
        progress_14536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'progress', False)
        # Obtaining the member 'text' of a type (line 675)
        text_14537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 8), progress_14536, 'text')
        # Calling text(args, kwargs) (line 675)
        text_call_result_14546 = invoke(stypy.reporting.localization.Localization(__file__, 675, 8), text_14537, *[str_14538, int_14539, int_14540, int_14541, int_14542, int_14543, str_14544], **kwargs_14545)
        
        
        # Call to text(...): (line 678)
        # Processing the call arguments (line 678)
        str_14549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 22), 'str', 'StatusLabel')
        int_14550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 37), 'int')
        int_14551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 41), 'int')
        int_14552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 46), 'int')
        int_14553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 50), 'int')
        int_14554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 54), 'int')
        str_14555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 57), 'str', 'Status:')
        # Processing the call keyword arguments (line 678)
        kwargs_14556 = {}
        # Getting the type of 'progress' (line 678)
        progress_14547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 8), 'progress', False)
        # Obtaining the member 'text' of a type (line 678)
        text_14548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 8), progress_14547, 'text')
        # Calling text(args, kwargs) (line 678)
        text_call_result_14557 = invoke(stypy.reporting.localization.Localization(__file__, 678, 8), text_14548, *[str_14549, int_14550, int_14551, int_14552, int_14553, int_14554, str_14555], **kwargs_14556)
        
        
        # Assigning a Call to a Name (line 680):
        
        # Call to text(...): (line 680)
        # Processing the call arguments (line 680)
        str_14560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 24), 'str', 'ActionText')
        int_14561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 38), 'int')
        int_14562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 42), 'int')
        # Getting the type of 'w' (line 680)
        w_14563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 47), 'w', False)
        int_14564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 49), 'int')
        # Applying the binary operator '-' (line 680)
        result_sub_14565 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 47), '-', w_14563, int_14564)
        
        int_14566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 53), 'int')
        int_14567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 57), 'int')
        str_14568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 60), 'str', 'Pondering...')
        # Processing the call keyword arguments (line 680)
        kwargs_14569 = {}
        # Getting the type of 'progress' (line 680)
        progress_14558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 10), 'progress', False)
        # Obtaining the member 'text' of a type (line 680)
        text_14559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 10), progress_14558, 'text')
        # Calling text(args, kwargs) (line 680)
        text_call_result_14570 = invoke(stypy.reporting.localization.Localization(__file__, 680, 10), text_14559, *[str_14560, int_14561, int_14562, result_sub_14565, int_14566, int_14567, str_14568], **kwargs_14569)
        
        # Assigning a type to the variable 'c' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'c', text_call_result_14570)
        
        # Call to mapping(...): (line 681)
        # Processing the call arguments (line 681)
        str_14573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 18), 'str', 'ActionText')
        str_14574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 32), 'str', 'Text')
        # Processing the call keyword arguments (line 681)
        kwargs_14575 = {}
        # Getting the type of 'c' (line 681)
        c_14571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'c', False)
        # Obtaining the member 'mapping' of a type (line 681)
        mapping_14572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 8), c_14571, 'mapping')
        # Calling mapping(args, kwargs) (line 681)
        mapping_call_result_14576 = invoke(stypy.reporting.localization.Localization(__file__, 681, 8), mapping_14572, *[str_14573, str_14574], **kwargs_14575)
        
        
        # Assigning a Call to a Name (line 686):
        
        # Call to control(...): (line 686)
        # Processing the call arguments (line 686)
        str_14579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 27), 'str', 'ProgressBar')
        str_14580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 42), 'str', 'ProgressBar')
        int_14581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 57), 'int')
        int_14582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 61), 'int')
        int_14583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 66), 'int')
        int_14584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 71), 'int')
        int_14585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 75), 'int')
        # Getting the type of 'None' (line 687)
        None_14586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 27), 'None', False)
        str_14587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 33), 'str', 'Progress done')
        # Getting the type of 'None' (line 687)
        None_14588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 50), 'None', False)
        # Getting the type of 'None' (line 687)
        None_14589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 56), 'None', False)
        # Processing the call keyword arguments (line 686)
        kwargs_14590 = {}
        # Getting the type of 'progress' (line 686)
        progress_14577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 10), 'progress', False)
        # Obtaining the member 'control' of a type (line 686)
        control_14578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 10), progress_14577, 'control')
        # Calling control(args, kwargs) (line 686)
        control_call_result_14591 = invoke(stypy.reporting.localization.Localization(__file__, 686, 10), control_14578, *[str_14579, str_14580, int_14581, int_14582, int_14583, int_14584, int_14585, None_14586, str_14587, None_14588, None_14589], **kwargs_14590)
        
        # Assigning a type to the variable 'c' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'c', control_call_result_14591)
        
        # Call to mapping(...): (line 688)
        # Processing the call arguments (line 688)
        str_14594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 18), 'str', 'SetProgress')
        str_14595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 33), 'str', 'Progress')
        # Processing the call keyword arguments (line 688)
        kwargs_14596 = {}
        # Getting the type of 'c' (line 688)
        c_14592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 8), 'c', False)
        # Obtaining the member 'mapping' of a type (line 688)
        mapping_14593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 8), c_14592, 'mapping')
        # Calling mapping(args, kwargs) (line 688)
        mapping_call_result_14597 = invoke(stypy.reporting.localization.Localization(__file__, 688, 8), mapping_14593, *[str_14594, str_14595], **kwargs_14596)
        
        
        # Call to back(...): (line 690)
        # Processing the call arguments (line 690)
        str_14600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 22), 'str', '< Back')
        str_14601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 32), 'str', 'Next')
        # Processing the call keyword arguments (line 690)
        # Getting the type of 'False' (line 690)
        False_14602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 47), 'False', False)
        keyword_14603 = False_14602
        kwargs_14604 = {'active': keyword_14603}
        # Getting the type of 'progress' (line 690)
        progress_14598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'progress', False)
        # Obtaining the member 'back' of a type (line 690)
        back_14599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 8), progress_14598, 'back')
        # Calling back(args, kwargs) (line 690)
        back_call_result_14605 = invoke(stypy.reporting.localization.Localization(__file__, 690, 8), back_14599, *[str_14600, str_14601], **kwargs_14604)
        
        
        # Call to next(...): (line 691)
        # Processing the call arguments (line 691)
        str_14608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 22), 'str', 'Next >')
        str_14609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 32), 'str', 'Cancel')
        # Processing the call keyword arguments (line 691)
        # Getting the type of 'False' (line 691)
        False_14610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 49), 'False', False)
        keyword_14611 = False_14610
        kwargs_14612 = {'active': keyword_14611}
        # Getting the type of 'progress' (line 691)
        progress_14606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'progress', False)
        # Obtaining the member 'next' of a type (line 691)
        next_14607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 8), progress_14606, 'next')
        # Calling next(args, kwargs) (line 691)
        next_call_result_14613 = invoke(stypy.reporting.localization.Localization(__file__, 691, 8), next_14607, *[str_14608, str_14609], **kwargs_14612)
        
        
        # Call to event(...): (line 692)
        # Processing the call arguments (line 692)
        str_14621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 48), 'str', 'SpawnDialog')
        str_14622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 63), 'str', 'CancelDlg')
        # Processing the call keyword arguments (line 692)
        kwargs_14623 = {}
        
        # Call to cancel(...): (line 692)
        # Processing the call arguments (line 692)
        str_14616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 24), 'str', 'Cancel')
        str_14617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 34), 'str', 'Back')
        # Processing the call keyword arguments (line 692)
        kwargs_14618 = {}
        # Getting the type of 'progress' (line 692)
        progress_14614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'progress', False)
        # Obtaining the member 'cancel' of a type (line 692)
        cancel_14615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 8), progress_14614, 'cancel')
        # Calling cancel(args, kwargs) (line 692)
        cancel_call_result_14619 = invoke(stypy.reporting.localization.Localization(__file__, 692, 8), cancel_14615, *[str_14616, str_14617], **kwargs_14618)
        
        # Obtaining the member 'event' of a type (line 692)
        event_14620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 8), cancel_call_result_14619, 'event')
        # Calling event(args, kwargs) (line 692)
        event_call_result_14624 = invoke(stypy.reporting.localization.Localization(__file__, 692, 8), event_14620, *[str_14621, str_14622], **kwargs_14623)
        
        
        # Assigning a Call to a Name (line 696):
        
        # Call to PyDialog(...): (line 696)
        # Processing the call arguments (line 696)
        # Getting the type of 'db' (line 696)
        db_14626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 25), 'db', False)
        str_14627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 29), 'str', 'MaintenanceTypeDlg')
        # Getting the type of 'x' (line 696)
        x_14628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 51), 'x', False)
        # Getting the type of 'y' (line 696)
        y_14629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 54), 'y', False)
        # Getting the type of 'w' (line 696)
        w_14630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 57), 'w', False)
        # Getting the type of 'h' (line 696)
        h_14631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 60), 'h', False)
        # Getting the type of 'modal' (line 696)
        modal_14632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 63), 'modal', False)
        # Getting the type of 'title' (line 696)
        title_14633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 70), 'title', False)
        str_14634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 25), 'str', 'Next')
        str_14635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 33), 'str', 'Next')
        str_14636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 41), 'str', 'Cancel')
        # Processing the call keyword arguments (line 696)
        kwargs_14637 = {}
        # Getting the type of 'PyDialog' (line 696)
        PyDialog_14625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 16), 'PyDialog', False)
        # Calling PyDialog(args, kwargs) (line 696)
        PyDialog_call_result_14638 = invoke(stypy.reporting.localization.Localization(__file__, 696, 16), PyDialog_14625, *[db_14626, str_14627, x_14628, y_14629, w_14630, h_14631, modal_14632, title_14633, str_14634, str_14635, str_14636], **kwargs_14637)
        
        # Assigning a type to the variable 'maint' (line 696)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'maint', PyDialog_call_result_14638)
        
        # Call to title(...): (line 698)
        # Processing the call arguments (line 698)
        str_14641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 20), 'str', 'Welcome to the [ProductName] Setup Wizard')
        # Processing the call keyword arguments (line 698)
        kwargs_14642 = {}
        # Getting the type of 'maint' (line 698)
        maint_14639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'maint', False)
        # Obtaining the member 'title' of a type (line 698)
        title_14640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), maint_14639, 'title')
        # Calling title(args, kwargs) (line 698)
        title_call_result_14643 = invoke(stypy.reporting.localization.Localization(__file__, 698, 8), title_14640, *[str_14641], **kwargs_14642)
        
        
        # Call to text(...): (line 699)
        # Processing the call arguments (line 699)
        str_14646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 19), 'str', 'BodyText')
        int_14647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 31), 'int')
        int_14648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 35), 'int')
        int_14649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 39), 'int')
        int_14650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 44), 'int')
        int_14651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 48), 'int')
        str_14652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 19), 'str', 'Select whether you want to repair or remove [ProductName].')
        # Processing the call keyword arguments (line 699)
        kwargs_14653 = {}
        # Getting the type of 'maint' (line 699)
        maint_14644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'maint', False)
        # Obtaining the member 'text' of a type (line 699)
        text_14645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 8), maint_14644, 'text')
        # Calling text(args, kwargs) (line 699)
        text_call_result_14654 = invoke(stypy.reporting.localization.Localization(__file__, 699, 8), text_14645, *[str_14646, int_14647, int_14648, int_14649, int_14650, int_14651, str_14652], **kwargs_14653)
        
        
        # Assigning a Call to a Name (line 701):
        
        # Call to radiogroup(...): (line 701)
        # Processing the call arguments (line 701)
        str_14657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 27), 'str', 'RepairRadioGroup')
        int_14658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 47), 'int')
        int_14659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 51), 'int')
        int_14660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 56), 'int')
        int_14661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 61), 'int')
        int_14662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 65), 'int')
        str_14663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 28), 'str', 'MaintenanceForm_Action')
        str_14664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 54), 'str', '')
        str_14665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 58), 'str', 'Next')
        # Processing the call keyword arguments (line 701)
        kwargs_14666 = {}
        # Getting the type of 'maint' (line 701)
        maint_14655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 10), 'maint', False)
        # Obtaining the member 'radiogroup' of a type (line 701)
        radiogroup_14656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 10), maint_14655, 'radiogroup')
        # Calling radiogroup(args, kwargs) (line 701)
        radiogroup_call_result_14667 = invoke(stypy.reporting.localization.Localization(__file__, 701, 10), radiogroup_14656, *[str_14657, int_14658, int_14659, int_14660, int_14661, int_14662, str_14663, str_14664, str_14665], **kwargs_14666)
        
        # Assigning a type to the variable 'g' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'g', radiogroup_call_result_14667)
        
        # Call to add(...): (line 704)
        # Processing the call arguments (line 704)
        str_14670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 14), 'str', 'Repair')
        int_14671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 24), 'int')
        int_14672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 27), 'int')
        int_14673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 31), 'int')
        int_14674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 36), 'int')
        str_14675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 40), 'str', '&Repair [ProductName]')
        # Processing the call keyword arguments (line 704)
        kwargs_14676 = {}
        # Getting the type of 'g' (line 704)
        g_14668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'g', False)
        # Obtaining the member 'add' of a type (line 704)
        add_14669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), g_14668, 'add')
        # Calling add(args, kwargs) (line 704)
        add_call_result_14677 = invoke(stypy.reporting.localization.Localization(__file__, 704, 8), add_14669, *[str_14670, int_14671, int_14672, int_14673, int_14674, str_14675], **kwargs_14676)
        
        
        # Call to add(...): (line 705)
        # Processing the call arguments (line 705)
        str_14680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 14), 'str', 'Remove')
        int_14681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 24), 'int')
        int_14682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 27), 'int')
        int_14683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 31), 'int')
        int_14684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 36), 'int')
        str_14685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 40), 'str', 'Re&move [ProductName]')
        # Processing the call keyword arguments (line 705)
        kwargs_14686 = {}
        # Getting the type of 'g' (line 705)
        g_14678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'g', False)
        # Obtaining the member 'add' of a type (line 705)
        add_14679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 8), g_14678, 'add')
        # Calling add(args, kwargs) (line 705)
        add_call_result_14687 = invoke(stypy.reporting.localization.Localization(__file__, 705, 8), add_14679, *[str_14680, int_14681, int_14682, int_14683, int_14684, str_14685], **kwargs_14686)
        
        
        # Call to back(...): (line 707)
        # Processing the call arguments (line 707)
        str_14690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 19), 'str', '< Back')
        # Getting the type of 'None' (line 707)
        None_14691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 29), 'None', False)
        # Processing the call keyword arguments (line 707)
        # Getting the type of 'False' (line 707)
        False_14692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 42), 'False', False)
        keyword_14693 = False_14692
        kwargs_14694 = {'active': keyword_14693}
        # Getting the type of 'maint' (line 707)
        maint_14688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'maint', False)
        # Obtaining the member 'back' of a type (line 707)
        back_14689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 8), maint_14688, 'back')
        # Calling back(args, kwargs) (line 707)
        back_call_result_14695 = invoke(stypy.reporting.localization.Localization(__file__, 707, 8), back_14689, *[str_14690, None_14691], **kwargs_14694)
        
        
        # Assigning a Call to a Name (line 708):
        
        # Call to next(...): (line 708)
        # Processing the call arguments (line 708)
        str_14698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 21), 'str', 'Finish')
        str_14699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 31), 'str', 'Cancel')
        # Processing the call keyword arguments (line 708)
        kwargs_14700 = {}
        # Getting the type of 'maint' (line 708)
        maint_14696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 10), 'maint', False)
        # Obtaining the member 'next' of a type (line 708)
        next_14697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 10), maint_14696, 'next')
        # Calling next(args, kwargs) (line 708)
        next_call_result_14701 = invoke(stypy.reporting.localization.Localization(__file__, 708, 10), next_14697, *[str_14698, str_14699], **kwargs_14700)
        
        # Assigning a type to the variable 'c' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'c', next_call_result_14701)
        
        # Call to event(...): (line 716)
        # Processing the call arguments (line 716)
        str_14704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 16), 'str', '[REINSTALL]')
        str_14705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 31), 'str', 'ALL')
        str_14706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 38), 'str', 'MaintenanceForm_Action="Repair"')
        int_14707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 73), 'int')
        # Processing the call keyword arguments (line 716)
        kwargs_14708 = {}
        # Getting the type of 'c' (line 716)
        c_14702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 716)
        event_14703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 8), c_14702, 'event')
        # Calling event(args, kwargs) (line 716)
        event_call_result_14709 = invoke(stypy.reporting.localization.Localization(__file__, 716, 8), event_14703, *[str_14704, str_14705, str_14706, int_14707], **kwargs_14708)
        
        
        # Call to event(...): (line 717)
        # Processing the call arguments (line 717)
        str_14712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 16), 'str', '[Progress1]')
        str_14713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 31), 'str', 'Repairing')
        str_14714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 44), 'str', 'MaintenanceForm_Action="Repair"')
        int_14715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 79), 'int')
        # Processing the call keyword arguments (line 717)
        kwargs_14716 = {}
        # Getting the type of 'c' (line 717)
        c_14710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 717)
        event_14711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 8), c_14710, 'event')
        # Calling event(args, kwargs) (line 717)
        event_call_result_14717 = invoke(stypy.reporting.localization.Localization(__file__, 717, 8), event_14711, *[str_14712, str_14713, str_14714, int_14715], **kwargs_14716)
        
        
        # Call to event(...): (line 718)
        # Processing the call arguments (line 718)
        str_14720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 16), 'str', '[Progress2]')
        str_14721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 31), 'str', 'repairs')
        str_14722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 42), 'str', 'MaintenanceForm_Action="Repair"')
        int_14723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 77), 'int')
        # Processing the call keyword arguments (line 718)
        kwargs_14724 = {}
        # Getting the type of 'c' (line 718)
        c_14718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 718)
        event_14719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 8), c_14718, 'event')
        # Calling event(args, kwargs) (line 718)
        event_call_result_14725 = invoke(stypy.reporting.localization.Localization(__file__, 718, 8), event_14719, *[str_14720, str_14721, str_14722, int_14723], **kwargs_14724)
        
        
        # Call to event(...): (line 719)
        # Processing the call arguments (line 719)
        str_14728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 16), 'str', 'Reinstall')
        str_14729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 29), 'str', 'ALL')
        str_14730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 36), 'str', 'MaintenanceForm_Action="Repair"')
        int_14731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 71), 'int')
        # Processing the call keyword arguments (line 719)
        kwargs_14732 = {}
        # Getting the type of 'c' (line 719)
        c_14726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 719)
        event_14727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 8), c_14726, 'event')
        # Calling event(args, kwargs) (line 719)
        event_call_result_14733 = invoke(stypy.reporting.localization.Localization(__file__, 719, 8), event_14727, *[str_14728, str_14729, str_14730, int_14731], **kwargs_14732)
        
        
        # Call to event(...): (line 723)
        # Processing the call arguments (line 723)
        str_14736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 16), 'str', '[REMOVE]')
        str_14737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 28), 'str', 'ALL')
        str_14738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 35), 'str', 'MaintenanceForm_Action="Remove"')
        int_14739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 70), 'int')
        # Processing the call keyword arguments (line 723)
        kwargs_14740 = {}
        # Getting the type of 'c' (line 723)
        c_14734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 723)
        event_14735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 8), c_14734, 'event')
        # Calling event(args, kwargs) (line 723)
        event_call_result_14741 = invoke(stypy.reporting.localization.Localization(__file__, 723, 8), event_14735, *[str_14736, str_14737, str_14738, int_14739], **kwargs_14740)
        
        
        # Call to event(...): (line 724)
        # Processing the call arguments (line 724)
        str_14744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 16), 'str', '[Progress1]')
        str_14745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 31), 'str', 'Removing')
        str_14746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 43), 'str', 'MaintenanceForm_Action="Remove"')
        int_14747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 78), 'int')
        # Processing the call keyword arguments (line 724)
        kwargs_14748 = {}
        # Getting the type of 'c' (line 724)
        c_14742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 724)
        event_14743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 8), c_14742, 'event')
        # Calling event(args, kwargs) (line 724)
        event_call_result_14749 = invoke(stypy.reporting.localization.Localization(__file__, 724, 8), event_14743, *[str_14744, str_14745, str_14746, int_14747], **kwargs_14748)
        
        
        # Call to event(...): (line 725)
        # Processing the call arguments (line 725)
        str_14752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 16), 'str', '[Progress2]')
        str_14753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 31), 'str', 'removes')
        str_14754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 42), 'str', 'MaintenanceForm_Action="Remove"')
        int_14755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 77), 'int')
        # Processing the call keyword arguments (line 725)
        kwargs_14756 = {}
        # Getting the type of 'c' (line 725)
        c_14750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 725)
        event_14751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 8), c_14750, 'event')
        # Calling event(args, kwargs) (line 725)
        event_call_result_14757 = invoke(stypy.reporting.localization.Localization(__file__, 725, 8), event_14751, *[str_14752, str_14753, str_14754, int_14755], **kwargs_14756)
        
        
        # Call to event(...): (line 726)
        # Processing the call arguments (line 726)
        str_14760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 16), 'str', 'Remove')
        str_14761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 26), 'str', 'ALL')
        str_14762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 33), 'str', 'MaintenanceForm_Action="Remove"')
        int_14763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 68), 'int')
        # Processing the call keyword arguments (line 726)
        kwargs_14764 = {}
        # Getting the type of 'c' (line 726)
        c_14758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 726)
        event_14759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 8), c_14758, 'event')
        # Calling event(args, kwargs) (line 726)
        event_call_result_14765 = invoke(stypy.reporting.localization.Localization(__file__, 726, 8), event_14759, *[str_14760, str_14761, str_14762, int_14763], **kwargs_14764)
        
        
        # Call to event(...): (line 729)
        # Processing the call arguments (line 729)
        str_14768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 16), 'str', 'EndDialog')
        str_14769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 29), 'str', 'Return')
        str_14770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 39), 'str', 'MaintenanceForm_Action<>"Change"')
        int_14771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 75), 'int')
        # Processing the call keyword arguments (line 729)
        kwargs_14772 = {}
        # Getting the type of 'c' (line 729)
        c_14766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 8), 'c', False)
        # Obtaining the member 'event' of a type (line 729)
        event_14767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 8), c_14766, 'event')
        # Calling event(args, kwargs) (line 729)
        event_call_result_14773 = invoke(stypy.reporting.localization.Localization(__file__, 729, 8), event_14767, *[str_14768, str_14769, str_14770, int_14771], **kwargs_14772)
        
        
        # Call to event(...): (line 732)
        # Processing the call arguments (line 732)
        str_14781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 57), 'str', 'SpawnDialog')
        str_14782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 72), 'str', 'CancelDlg')
        # Processing the call keyword arguments (line 732)
        kwargs_14783 = {}
        
        # Call to cancel(...): (line 732)
        # Processing the call arguments (line 732)
        str_14776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 21), 'str', 'Cancel')
        str_14777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 31), 'str', 'RepairRadioGroup')
        # Processing the call keyword arguments (line 732)
        kwargs_14778 = {}
        # Getting the type of 'maint' (line 732)
        maint_14774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 8), 'maint', False)
        # Obtaining the member 'cancel' of a type (line 732)
        cancel_14775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 8), maint_14774, 'cancel')
        # Calling cancel(args, kwargs) (line 732)
        cancel_call_result_14779 = invoke(stypy.reporting.localization.Localization(__file__, 732, 8), cancel_14775, *[str_14776, str_14777], **kwargs_14778)
        
        # Obtaining the member 'event' of a type (line 732)
        event_14780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 8), cancel_call_result_14779, 'event')
        # Calling event(args, kwargs) (line 732)
        event_call_result_14784 = invoke(stypy.reporting.localization.Localization(__file__, 732, 8), event_14780, *[str_14781, str_14782], **kwargs_14783)
        
        
        # ################# End of 'add_ui(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_ui' in the type store
        # Getting the type of 'stypy_return_type' (line 417)
        stypy_return_type_14785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14785)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_ui'
        return stypy_return_type_14785


    @norecursion
    def get_installer_filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_installer_filename'
        module_type_store = module_type_store.open_function_context('get_installer_filename', 734, 4, False)
        # Assigning a type to the variable 'self' (line 735)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_localization', localization)
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_function_name', 'bdist_msi.get_installer_filename')
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_param_names_list', ['fullname'])
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_msi.get_installer_filename.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_msi.get_installer_filename', ['fullname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_installer_filename', localization, ['fullname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_installer_filename(...)' code ##################

        
        # Getting the type of 'self' (line 736)
        self_14786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 11), 'self')
        # Obtaining the member 'target_version' of a type (line 736)
        target_version_14787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), self_14786, 'target_version')
        # Testing the type of an if condition (line 736)
        if_condition_14788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 736, 8), target_version_14787)
        # Assigning a type to the variable 'if_condition_14788' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'if_condition_14788', if_condition_14788)
        # SSA begins for if statement (line 736)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 737):
        str_14789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 24), 'str', '%s.%s-py%s.msi')
        
        # Obtaining an instance of the builtin type 'tuple' (line 737)
        tuple_14790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 737)
        # Adding element type (line 737)
        # Getting the type of 'fullname' (line 737)
        fullname_14791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 44), 'fullname')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 737, 44), tuple_14790, fullname_14791)
        # Adding element type (line 737)
        # Getting the type of 'self' (line 737)
        self_14792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 54), 'self')
        # Obtaining the member 'plat_name' of a type (line 737)
        plat_name_14793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 54), self_14792, 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 737, 44), tuple_14790, plat_name_14793)
        # Adding element type (line 737)
        # Getting the type of 'self' (line 738)
        self_14794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 44), 'self')
        # Obtaining the member 'target_version' of a type (line 738)
        target_version_14795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 44), self_14794, 'target_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 737, 44), tuple_14790, target_version_14795)
        
        # Applying the binary operator '%' (line 737)
        result_mod_14796 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 24), '%', str_14789, tuple_14790)
        
        # Assigning a type to the variable 'base_name' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 12), 'base_name', result_mod_14796)
        # SSA branch for the else part of an if statement (line 736)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 740):
        str_14797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 24), 'str', '%s.%s.msi')
        
        # Obtaining an instance of the builtin type 'tuple' (line 740)
        tuple_14798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 740)
        # Adding element type (line 740)
        # Getting the type of 'fullname' (line 740)
        fullname_14799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 39), 'fullname')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 39), tuple_14798, fullname_14799)
        # Adding element type (line 740)
        # Getting the type of 'self' (line 740)
        self_14800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 49), 'self')
        # Obtaining the member 'plat_name' of a type (line 740)
        plat_name_14801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 49), self_14800, 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 39), tuple_14798, plat_name_14801)
        
        # Applying the binary operator '%' (line 740)
        result_mod_14802 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 24), '%', str_14797, tuple_14798)
        
        # Assigning a type to the variable 'base_name' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'base_name', result_mod_14802)
        # SSA join for if statement (line 736)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 741):
        
        # Call to join(...): (line 741)
        # Processing the call arguments (line 741)
        # Getting the type of 'self' (line 741)
        self_14806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 38), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 741)
        dist_dir_14807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 38), self_14806, 'dist_dir')
        # Getting the type of 'base_name' (line 741)
        base_name_14808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 53), 'base_name', False)
        # Processing the call keyword arguments (line 741)
        kwargs_14809 = {}
        # Getting the type of 'os' (line 741)
        os_14803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 741)
        path_14804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 25), os_14803, 'path')
        # Obtaining the member 'join' of a type (line 741)
        join_14805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 25), path_14804, 'join')
        # Calling join(args, kwargs) (line 741)
        join_call_result_14810 = invoke(stypy.reporting.localization.Localization(__file__, 741, 25), join_14805, *[dist_dir_14807, base_name_14808], **kwargs_14809)
        
        # Assigning a type to the variable 'installer_name' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'installer_name', join_call_result_14810)
        # Getting the type of 'installer_name' (line 742)
        installer_name_14811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 15), 'installer_name')
        # Assigning a type to the variable 'stypy_return_type' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'stypy_return_type', installer_name_14811)
        
        # ################# End of 'get_installer_filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_installer_filename' in the type store
        # Getting the type of 'stypy_return_type' (line 734)
        stypy_return_type_14812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14812)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_installer_filename'
        return stypy_return_type_14812


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 84, 0, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_msi.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'bdist_msi' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'bdist_msi', bdist_msi)

# Assigning a Str to a Name (line 86):
str_14813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'str', 'create a Microsoft Installer (.msi) binary distribution')
# Getting the type of 'bdist_msi'
bdist_msi_14814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_msi')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_msi_14814, 'description', str_14813)

# Assigning a List to a Name (line 88):

# Obtaining an instance of the builtin type 'list' (line 88)
list_14815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 88)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 88)
tuple_14816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 88)
# Adding element type (line 88)
str_14817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'str', 'bdist-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), tuple_14816, str_14817)
# Adding element type (line 88)
# Getting the type of 'None' (line 88)
None_14818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 35), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), tuple_14816, None_14818)
# Adding element type (line 88)
str_14819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'str', 'temporary directory for creating the distribution')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), tuple_14816, str_14819)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), list_14815, tuple_14816)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 90)
tuple_14820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 90)
# Adding element type (line 90)
str_14821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', 'plat-name=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 21), tuple_14820, str_14821)
# Adding element type (line 90)
str_14822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 35), 'str', 'p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 21), tuple_14820, str_14822)
# Adding element type (line 90)
str_14823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'str', 'platform name to embed in generated filenames (default: %s)')

# Call to get_platform(...): (line 92)
# Processing the call keyword arguments (line 92)
kwargs_14825 = {}
# Getting the type of 'get_platform' (line 92)
get_platform_14824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 39), 'get_platform', False)
# Calling get_platform(args, kwargs) (line 92)
get_platform_call_result_14826 = invoke(stypy.reporting.localization.Localization(__file__, 92, 39), get_platform_14824, *[], **kwargs_14825)

# Applying the binary operator '%' (line 91)
result_mod_14827 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 21), '%', str_14823, get_platform_call_result_14826)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 21), tuple_14820, result_mod_14827)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), list_14815, tuple_14820)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 93)
tuple_14828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 93)
# Adding element type (line 93)
str_14829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'str', 'keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), tuple_14828, str_14829)
# Adding element type (line 93)
str_14830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 34), 'str', 'k')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), tuple_14828, str_14830)
# Adding element type (line 93)
str_14831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'str', 'keep the pseudo-installation tree around after ')
str_14832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 21), 'str', 'creating the distribution archive')
# Applying the binary operator '+' (line 94)
result_add_14833 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 21), '+', str_14831, str_14832)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), tuple_14828, result_add_14833)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), list_14815, tuple_14828)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 96)
tuple_14834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 96)
# Adding element type (line 96)
str_14835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 21), 'str', 'target-version=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 21), tuple_14834, str_14835)
# Adding element type (line 96)
# Getting the type of 'None' (line 96)
None_14836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 21), tuple_14834, None_14836)
# Adding element type (line 96)
str_14837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 21), 'str', 'require a specific python version')
str_14838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 21), 'str', ' on the target system')
# Applying the binary operator '+' (line 97)
result_add_14839 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 21), '+', str_14837, str_14838)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 21), tuple_14834, result_add_14839)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), list_14815, tuple_14834)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 99)
tuple_14840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 99)
# Adding element type (line 99)
str_14841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'str', 'no-target-compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 21), tuple_14840, str_14841)
# Adding element type (line 99)
str_14842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 42), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 21), tuple_14840, str_14842)
# Adding element type (line 99)
str_14843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 21), 'str', 'do not compile .py to .pyc on the target system')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 21), tuple_14840, str_14843)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), list_14815, tuple_14840)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 101)
tuple_14844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 101)
# Adding element type (line 101)
str_14845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 21), 'str', 'no-target-optimize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 21), tuple_14844, str_14845)
# Adding element type (line 101)
str_14846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 43), 'str', 'o')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 21), tuple_14844, str_14846)
# Adding element type (line 101)
str_14847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'str', 'do not compile .py to .pyo (optimized)on the target system')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 21), tuple_14844, str_14847)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), list_14815, tuple_14844)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 104)
tuple_14848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 104)
# Adding element type (line 104)
str_14849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 21), 'str', 'dist-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 21), tuple_14848, str_14849)
# Adding element type (line 104)
str_14850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 34), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 21), tuple_14848, str_14850)
# Adding element type (line 104)
str_14851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 21), 'str', 'directory to put final built distributions in')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 21), tuple_14848, str_14851)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), list_14815, tuple_14848)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 106)
tuple_14852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 106)
# Adding element type (line 106)
str_14853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 21), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 21), tuple_14852, str_14853)
# Adding element type (line 106)
# Getting the type of 'None' (line 106)
None_14854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 35), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 21), tuple_14852, None_14854)
# Adding element type (line 106)
str_14855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'str', 'skip rebuilding everything (for testing/debugging)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 21), tuple_14852, str_14855)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), list_14815, tuple_14852)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 108)
tuple_14856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 108)
# Adding element type (line 108)
str_14857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'str', 'install-script=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 21), tuple_14856, str_14857)
# Adding element type (line 108)
# Getting the type of 'None' (line 108)
None_14858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 21), tuple_14856, None_14858)
# Adding element type (line 108)
str_14859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'str', 'basename of installation script to be run afterinstallation or before deinstallation')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 21), tuple_14856, str_14859)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), list_14815, tuple_14856)
# Adding element type (line 88)

# Obtaining an instance of the builtin type 'tuple' (line 111)
tuple_14860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 111)
# Adding element type (line 111)
str_14861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'str', 'pre-install-script=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), tuple_14860, str_14861)
# Adding element type (line 111)
# Getting the type of 'None' (line 111)
None_14862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 44), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), tuple_14860, None_14862)
# Adding element type (line 111)
str_14863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 21), 'str', 'Fully qualified filename of a script to be run before any files are installed.  This script need not be in the distribution')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), tuple_14860, str_14863)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 19), list_14815, tuple_14860)

# Getting the type of 'bdist_msi'
bdist_msi_14864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_msi')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_msi_14864, 'user_options', list_14815)

# Assigning a List to a Name (line 117):

# Obtaining an instance of the builtin type 'list' (line 117)
list_14865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 117)
# Adding element type (line 117)
str_14866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'str', 'keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 22), list_14865, str_14866)
# Adding element type (line 117)
str_14867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 36), 'str', 'no-target-compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 22), list_14865, str_14867)
# Adding element type (line 117)
str_14868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 57), 'str', 'no-target-optimize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 22), list_14865, str_14868)
# Adding element type (line 117)
str_14869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 23), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 22), list_14865, str_14869)

# Getting the type of 'bdist_msi'
bdist_msi_14870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_msi')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_msi_14870, 'boolean_options', list_14865)

# Assigning a List to a Name (line 120):

# Obtaining an instance of the builtin type 'list' (line 120)
list_14871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 120)
# Adding element type (line 120)
str_14872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 20), 'str', '2.0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14872)
# Adding element type (line 120)
str_14873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 27), 'str', '2.1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14873)
# Adding element type (line 120)
str_14874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 34), 'str', '2.2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14874)
# Adding element type (line 120)
str_14875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 41), 'str', '2.3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14875)
# Adding element type (line 120)
str_14876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 48), 'str', '2.4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14876)
# Adding element type (line 120)
str_14877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 20), 'str', '2.5')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14877)
# Adding element type (line 120)
str_14878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 27), 'str', '2.6')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14878)
# Adding element type (line 120)
str_14879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 34), 'str', '2.7')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14879)
# Adding element type (line 120)
str_14880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 41), 'str', '2.8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14880)
# Adding element type (line 120)
str_14881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 48), 'str', '2.9')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14881)
# Adding element type (line 120)
str_14882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 20), 'str', '3.0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14882)
# Adding element type (line 120)
str_14883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 27), 'str', '3.1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14883)
# Adding element type (line 120)
str_14884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 34), 'str', '3.2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14884)
# Adding element type (line 120)
str_14885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 41), 'str', '3.3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14885)
# Adding element type (line 120)
str_14886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 48), 'str', '3.4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14886)
# Adding element type (line 120)
str_14887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 20), 'str', '3.5')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14887)
# Adding element type (line 120)
str_14888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 27), 'str', '3.6')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14888)
# Adding element type (line 120)
str_14889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 34), 'str', '3.7')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14889)
# Adding element type (line 120)
str_14890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'str', '3.8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14890)
# Adding element type (line 120)
str_14891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 48), 'str', '3.9')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), list_14871, str_14891)

# Getting the type of 'bdist_msi'
bdist_msi_14892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_msi')
# Setting the type of the member 'all_versions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_msi_14892, 'all_versions', list_14871)

# Assigning a Str to a Name (line 124):
str_14893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 20), 'str', 'X')
# Getting the type of 'bdist_msi'
bdist_msi_14894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_msi')
# Setting the type of the member 'other_version' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_msi_14894, 'other_version', str_14893)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
