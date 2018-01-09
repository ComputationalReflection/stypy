
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Tests for netcdf '''
2: from __future__ import division, print_function, absolute_import
3: 
4: import os
5: from os.path import join as pjoin, dirname
6: import shutil
7: import tempfile
8: import warnings
9: from io import BytesIO
10: from glob import glob
11: from contextlib import contextmanager
12: 
13: import numpy as np
14: from numpy.testing import assert_, assert_allclose, assert_equal
15: from pytest import raises as assert_raises
16: 
17: from scipy.io.netcdf import netcdf_file
18: 
19: from scipy._lib._numpy_compat import suppress_warnings
20: from scipy._lib._tmpdirs import in_tempdir
21: 
22: TEST_DATA_PATH = pjoin(dirname(__file__), 'data')
23: 
24: N_EG_ELS = 11  # number of elements for example variable
25: VARTYPE_EG = 'b'  # var type for example variable
26: 
27: 
28: @contextmanager
29: def make_simple(*args, **kwargs):
30:     f = netcdf_file(*args, **kwargs)
31:     f.history = 'Created for a test'
32:     f.createDimension('time', N_EG_ELS)
33:     time = f.createVariable('time', VARTYPE_EG, ('time',))
34:     time[:] = np.arange(N_EG_ELS)
35:     time.units = 'days since 2008-01-01'
36:     f.flush()
37:     yield f
38:     f.close()
39: 
40: 
41: def check_simple(ncfileobj):
42:     '''Example fileobj tests '''
43:     assert_equal(ncfileobj.history, b'Created for a test')
44:     time = ncfileobj.variables['time']
45:     assert_equal(time.units, b'days since 2008-01-01')
46:     assert_equal(time.shape, (N_EG_ELS,))
47:     assert_equal(time[-1], N_EG_ELS-1)
48: 
49: def assert_mask_matches(arr, expected_mask):
50:     '''
51:     Asserts that the mask of arr is effectively the same as expected_mask.
52: 
53:     In contrast to numpy.ma.testutils.assert_mask_equal, this function allows
54:     testing the 'mask' of a standard numpy array (the mask in this case is treated
55:     as all False).
56: 
57:     Parameters
58:     ----------
59:     arr: ndarray or MaskedArray
60:         Array to test.
61:     expected_mask: array_like of booleans
62:         A list giving the expected mask.
63:     '''
64: 
65:     mask = np.ma.getmaskarray(arr)
66:     assert_equal(mask, expected_mask)
67: 
68: 
69: def test_read_write_files():
70:     # test round trip for example file
71:     cwd = os.getcwd()
72:     try:
73:         tmpdir = tempfile.mkdtemp()
74:         os.chdir(tmpdir)
75:         with make_simple('simple.nc', 'w') as f:
76:             pass
77:         # read the file we just created in 'a' mode
78:         with netcdf_file('simple.nc', 'a') as f:
79:             check_simple(f)
80:             # add something
81:             f._attributes['appendRan'] = 1
82: 
83:         # To read the NetCDF file we just created::
84:         with netcdf_file('simple.nc') as f:
85:             # Using mmap is the default
86:             assert_(f.use_mmap)
87:             check_simple(f)
88:             assert_equal(f._attributes['appendRan'], 1)
89: 
90:         # Read it in append (and check mmap is off)
91:         with netcdf_file('simple.nc', 'a') as f:
92:             assert_(not f.use_mmap)
93:             check_simple(f)
94:             assert_equal(f._attributes['appendRan'], 1)
95: 
96:         # Now without mmap
97:         with netcdf_file('simple.nc', mmap=False) as f:
98:             # Using mmap is the default
99:             assert_(not f.use_mmap)
100:             check_simple(f)
101: 
102:         # To read the NetCDF file we just created, as file object, no
103:         # mmap.  When n * n_bytes(var_type) is not divisible by 4, this
104:         # raised an error in pupynere 1.0.12 and scipy rev 5893, because
105:         # calculated vsize was rounding up in units of 4 - see
106:         # https://www.unidata.ucar.edu/software/netcdf/docs/user_guide.html
107:         with open('simple.nc', 'rb') as fobj:
108:             with netcdf_file(fobj) as f:
109:                 # by default, don't use mmap for file-like
110:                 assert_(not f.use_mmap)
111:                 check_simple(f)
112: 
113:         # Read file from fileobj, with mmap
114:         with open('simple.nc', 'rb') as fobj:
115:             with netcdf_file(fobj, mmap=True) as f:
116:                 assert_(f.use_mmap)
117:                 check_simple(f)
118: 
119:         # Again read it in append mode (adding another att)
120:         with open('simple.nc', 'r+b') as fobj:
121:             with netcdf_file(fobj, 'a') as f:
122:                 assert_(not f.use_mmap)
123:                 check_simple(f)
124:                 f.createDimension('app_dim', 1)
125:                 var = f.createVariable('app_var', 'i', ('app_dim',))
126:                 var[:] = 42
127: 
128:         # And... check that app_var made it in...
129:         with netcdf_file('simple.nc') as f:
130:             check_simple(f)
131:             assert_equal(f.variables['app_var'][:], 42)
132: 
133:     except:
134:         os.chdir(cwd)
135:         shutil.rmtree(tmpdir)
136:         raise
137:     os.chdir(cwd)
138:     shutil.rmtree(tmpdir)
139: 
140: 
141: def test_read_write_sio():
142:     eg_sio1 = BytesIO()
143:     with make_simple(eg_sio1, 'w') as f1:
144:         str_val = eg_sio1.getvalue()
145: 
146:     eg_sio2 = BytesIO(str_val)
147:     with netcdf_file(eg_sio2) as f2:
148:         check_simple(f2)
149: 
150:     # Test that error is raised if attempting mmap for sio
151:     eg_sio3 = BytesIO(str_val)
152:     assert_raises(ValueError, netcdf_file, eg_sio3, 'r', True)
153:     # Test 64-bit offset write / read
154:     eg_sio_64 = BytesIO()
155:     with make_simple(eg_sio_64, 'w', version=2) as f_64:
156:         str_val = eg_sio_64.getvalue()
157: 
158:     eg_sio_64 = BytesIO(str_val)
159:     with netcdf_file(eg_sio_64) as f_64:
160:         check_simple(f_64)
161:         assert_equal(f_64.version_byte, 2)
162:     # also when version 2 explicitly specified
163:     eg_sio_64 = BytesIO(str_val)
164:     with netcdf_file(eg_sio_64, version=2) as f_64:
165:         check_simple(f_64)
166:         assert_equal(f_64.version_byte, 2)
167: 
168: 
169: def test_read_example_data():
170:     # read any example data files
171:     for fname in glob(pjoin(TEST_DATA_PATH, '*.nc')):
172:         with netcdf_file(fname, 'r') as f:
173:             pass
174:         with netcdf_file(fname, 'r', mmap=False) as f:
175:             pass
176: 
177: 
178: def test_itemset_no_segfault_on_readonly():
179:     # Regression test for ticket #1202.
180:     # Open the test file in read-only mode.
181: 
182:     filename = pjoin(TEST_DATA_PATH, 'example_1.nc')
183:     with suppress_warnings() as sup:
184:         sup.filter(RuntimeWarning,
185:                    "Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or arrays referring to its data still exist")
186:         with netcdf_file(filename, 'r') as f:
187:             time_var = f.variables['time']
188: 
189:     # time_var.assignValue(42) should raise a RuntimeError--not seg. fault!
190:     assert_raises(RuntimeError, time_var.assignValue, 42)
191: 
192: 
193: def test_write_invalid_dtype():
194:     dtypes = ['int64', 'uint64']
195:     if np.dtype('int').itemsize == 8:   # 64-bit machines
196:         dtypes.append('int')
197:     if np.dtype('uint').itemsize == 8:   # 64-bit machines
198:         dtypes.append('uint')
199: 
200:     with netcdf_file(BytesIO(), 'w') as f:
201:         f.createDimension('time', N_EG_ELS)
202:         for dt in dtypes:
203:             assert_raises(ValueError, f.createVariable, 'time', dt, ('time',))
204: 
205: 
206: def test_flush_rewind():
207:     stream = BytesIO()
208:     with make_simple(stream, mode='w') as f:
209:         x = f.createDimension('x',4)
210:         v = f.createVariable('v', 'i2', ['x'])
211:         v[:] = 1
212:         f.flush()
213:         len_single = len(stream.getvalue())
214:         f.flush()
215:         len_double = len(stream.getvalue())
216: 
217:     assert_(len_single == len_double)
218: 
219: 
220: def test_dtype_specifiers():
221:     # Numpy 1.7.0-dev had a bug where 'i2' wouldn't work.
222:     # Specifying np.int16 or similar only works from the same commit as this
223:     # comment was made.
224:     with make_simple(BytesIO(), mode='w') as f:
225:         f.createDimension('x',4)
226:         f.createVariable('v1', 'i2', ['x'])
227:         f.createVariable('v2', np.int16, ['x'])
228:         f.createVariable('v3', np.dtype(np.int16), ['x'])
229: 
230: 
231: def test_ticket_1720():
232:     io = BytesIO()
233: 
234:     items = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
235: 
236:     with netcdf_file(io, 'w') as f:
237:         f.history = 'Created for a test'
238:         f.createDimension('float_var', 10)
239:         float_var = f.createVariable('float_var', 'f', ('float_var',))
240:         float_var[:] = items
241:         float_var.units = 'metres'
242:         f.flush()
243:         contents = io.getvalue()
244: 
245:     io = BytesIO(contents)
246:     with netcdf_file(io, 'r') as f:
247:         assert_equal(f.history, b'Created for a test')
248:         float_var = f.variables['float_var']
249:         assert_equal(float_var.units, b'metres')
250:         assert_equal(float_var.shape, (10,))
251:         assert_allclose(float_var[:], items)
252: 
253: 
254: def test_mmaps_segfault():
255:     filename = pjoin(TEST_DATA_PATH, 'example_1.nc')
256: 
257:     with warnings.catch_warnings():
258:         warnings.simplefilter("error")
259:         with netcdf_file(filename, mmap=True) as f:
260:             x = f.variables['lat'][:]
261:             # should not raise warnings
262:             del x
263: 
264:     def doit():
265:         with netcdf_file(filename, mmap=True) as f:
266:             return f.variables['lat'][:]
267: 
268:     # should not crash
269:     with suppress_warnings() as sup:
270:         sup.filter(RuntimeWarning,
271:                    "Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or arrays referring to its data still exist")
272:         x = doit()
273:     x.sum()
274: 
275: 
276: def test_zero_dimensional_var():
277:     io = BytesIO()
278:     with make_simple(io, 'w') as f:
279:         v = f.createVariable('zerodim', 'i2', [])
280:         # This is checking that .isrec returns a boolean - don't simplify it
281:         # to 'assert not ...'
282:         assert v.isrec is False, v.isrec
283:         f.flush()
284: 
285: 
286: def test_byte_gatts():
287:     # Check that global "string" atts work like they did before py3k
288:     # unicode and general bytes confusion
289:     with in_tempdir():
290:         filename = 'g_byte_atts.nc'
291:         f = netcdf_file(filename, 'w')
292:         f._attributes['holy'] = b'grail'
293:         f._attributes['witch'] = 'floats'
294:         f.close()
295:         f = netcdf_file(filename, 'r')
296:         assert_equal(f._attributes['holy'], b'grail')
297:         assert_equal(f._attributes['witch'], b'floats')
298:         f.close()
299: 
300: 
301: def test_open_append():
302:     # open 'w' put one attr
303:     with in_tempdir():
304:         filename = 'append_dat.nc'
305:         f = netcdf_file(filename, 'w')
306:         f._attributes['Kilroy'] = 'was here'
307:         f.close()
308: 
309:         # open again in 'a', read the att and and a new one
310:         f = netcdf_file(filename, 'a')
311:         assert_equal(f._attributes['Kilroy'], b'was here')
312:         f._attributes['naughty'] = b'Zoot'
313:         f.close()
314: 
315:         # open yet again in 'r' and check both atts
316:         f = netcdf_file(filename, 'r')
317:         assert_equal(f._attributes['Kilroy'], b'was here')
318:         assert_equal(f._attributes['naughty'], b'Zoot')
319:         f.close()
320: 
321: 
322: def test_append_recordDimension(): 
323:     dataSize = 100   
324:     
325:     with in_tempdir():
326:         # Create file with record time dimension
327:         with netcdf_file('withRecordDimension.nc', 'w') as f:
328:             f.createDimension('time', None)
329:             f.createVariable('time', 'd', ('time',))
330:             f.createDimension('x', dataSize)
331:             x = f.createVariable('x', 'd', ('x',))
332:             x[:] = np.array(range(dataSize))
333:             f.createDimension('y', dataSize)
334:             y = f.createVariable('y', 'd', ('y',))
335:             y[:] = np.array(range(dataSize))
336:             f.createVariable('testData', 'i', ('time', 'x', 'y'))  
337:             f.flush()
338:             f.close()        
339:         
340:         for i in range(2): 
341:             # Open the file in append mode and add data 
342:             with netcdf_file('withRecordDimension.nc', 'a') as f:
343:                 f.variables['time'].data = np.append(f.variables["time"].data, i)
344:                 f.variables['testData'][i, :, :] = np.ones((dataSize, dataSize))*i
345:                 f.flush()
346:                 
347:             # Read the file and check that append worked
348:             with netcdf_file('withRecordDimension.nc') as f:            
349:                 assert_equal(f.variables['time'][-1], i)
350:                 assert_equal(f.variables['testData'][-1, :, :].copy(), np.ones((dataSize, dataSize))*i)
351:                 assert_equal(f.variables['time'].data.shape[0], i+1)
352:                 assert_equal(f.variables['testData'].data.shape[0], i+1)
353:                 
354:         # Read the file and check that 'data' was not saved as user defined
355:         # attribute of testData variable during append operation
356:         with netcdf_file('withRecordDimension.nc') as f:
357:             with assert_raises(KeyError) as ar:            
358:                 f.variables['testData']._attributes['data']
359:             ex = ar.value
360:             assert_equal(ex.args[0], 'data')
361: 
362: def test_maskandscale():
363:     t = np.linspace(20, 30, 15)
364:     t[3] = 100
365:     tm = np.ma.masked_greater(t, 99)
366:     fname = pjoin(TEST_DATA_PATH, 'example_2.nc')
367:     with netcdf_file(fname, maskandscale=True) as f:
368:         Temp = f.variables['Temperature']
369:         assert_equal(Temp.missing_value, 9999)
370:         assert_equal(Temp.add_offset, 20)
371:         assert_equal(Temp.scale_factor, np.float32(0.01))
372:         found = Temp[:].compressed()
373:         del Temp  # Remove ref to mmap, so file can be closed.
374:         expected = np.round(tm.compressed(), 2)
375:         assert_allclose(found, expected)
376: 
377:     with in_tempdir():
378:         newfname = 'ms.nc'
379:         f = netcdf_file(newfname, 'w', maskandscale=True)
380:         f.createDimension('Temperature', len(tm))
381:         temp = f.createVariable('Temperature', 'i', ('Temperature',))
382:         temp.missing_value = 9999
383:         temp.scale_factor = 0.01
384:         temp.add_offset = 20
385:         temp[:] = tm
386:         f.close()
387: 
388:         with netcdf_file(newfname, maskandscale=True) as f:
389:             Temp = f.variables['Temperature']
390:             assert_equal(Temp.missing_value, 9999)
391:             assert_equal(Temp.add_offset, 20)
392:             assert_equal(Temp.scale_factor, np.float32(0.01))
393:             expected = np.round(tm.compressed(), 2)
394:             found = Temp[:].compressed()
395:             del Temp
396:             assert_allclose(found, expected)
397: 
398: 
399: # ------------------------------------------------------------------------
400: # Test reading with masked values (_FillValue / missing_value)
401: # ------------------------------------------------------------------------
402: 
403: def test_read_withValuesNearFillValue():
404:     # Regression test for ticket #5626
405:     fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
406:     with netcdf_file(fname, maskandscale=True) as f:
407:         vardata = f.variables['var1_fillval0'][:]
408:         assert_mask_matches(vardata, [False, True, False])
409: 
410: def test_read_withNoFillValue():
411:     # For a variable with no fill value, reading data with maskandscale=True
412:     # should return unmasked data
413:     fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
414:     with netcdf_file(fname, maskandscale=True) as f:
415:         vardata = f.variables['var2_noFillval'][:]
416:         assert_mask_matches(vardata, [False, False, False])
417:         assert_equal(vardata, [1,2,3])
418: 
419: def test_read_withFillValueAndMissingValue():
420:     # For a variable with both _FillValue and missing_value, the _FillValue
421:     # should be used
422:     IRRELEVANT_VALUE = 9999
423:     fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
424:     with netcdf_file(fname, maskandscale=True) as f:
425:         vardata = f.variables['var3_fillvalAndMissingValue'][:]
426:         assert_mask_matches(vardata, [True, False, False])
427:         assert_equal(vardata, [IRRELEVANT_VALUE, 2, 3])
428: 
429: def test_read_withMissingValue():
430:     # For a variable with missing_value but not _FillValue, the missing_value
431:     # should be used
432:     fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
433:     with netcdf_file(fname, maskandscale=True) as f:
434:         vardata = f.variables['var4_missingValue'][:]
435:         assert_mask_matches(vardata, [False, True, False])
436: 
437: def test_read_withFillValNaN():
438:     fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
439:     with netcdf_file(fname, maskandscale=True) as f:
440:         vardata = f.variables['var5_fillvalNaN'][:]
441:         assert_mask_matches(vardata, [False, True, False])
442: 
443: def test_read_withChar():
444:     fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
445:     with netcdf_file(fname, maskandscale=True) as f:
446:         vardata = f.variables['var6_char'][:]
447:         assert_mask_matches(vardata, [False, True, False])
448: 
449: def test_read_with2dVar():
450:     fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
451:     with netcdf_file(fname, maskandscale=True) as f:
452:         vardata = f.variables['var7_2d'][:]
453:         assert_mask_matches(vardata, [[True, False], [False, False], [False, True]])
454: 
455: def test_read_withMaskAndScaleFalse():
456:     # If a variable has a _FillValue (or missing_value) attribute, but is read
457:     # with maskandscale set to False, the result should be unmasked
458:     fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
459:     # Open file with mmap=False to avoid problems with closing a mmap'ed file
460:     # when arrays referring to its data still exist:
461:     with netcdf_file(fname, maskandscale=False, mmap=False) as f:
462:         vardata = f.variables['var3_fillvalAndMissingValue'][:]
463:         assert_mask_matches(vardata, [False, False, False])
464:         assert_equal(vardata, [1, 2, 3])
465: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_7182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', ' Tests for netcdf ')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from os.path import pjoin, dirname' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_7183 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path')

if (type(import_7183) is not StypyTypeError):

    if (import_7183 != 'pyd_module'):
        __import__(import_7183)
        sys_modules_7184 = sys.modules[import_7183]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', sys_modules_7184.module_type_store, module_type_store, ['join', 'dirname'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_7184, sys_modules_7184.module_type_store, module_type_store)
    else:
        from os.path import join as pjoin, dirname

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', None, module_type_store, ['join', 'dirname'], [pjoin, dirname])

else:
    # Assigning a type to the variable 'os.path' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', import_7183)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import shutil' statement (line 6)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import tempfile' statement (line 7)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import warnings' statement (line 8)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from io import BytesIO' statement (line 9)
try:
    from io import BytesIO

except:
    BytesIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'io', None, module_type_store, ['BytesIO'], [BytesIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from glob import glob' statement (line 10)
try:
    from glob import glob

except:
    glob = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'glob', None, module_type_store, ['glob'], [glob])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from contextlib import contextmanager' statement (line 11)
try:
    from contextlib import contextmanager

except:
    contextmanager = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'contextlib', None, module_type_store, ['contextmanager'], [contextmanager])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_7185 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_7185) is not StypyTypeError):

    if (import_7185 != 'pyd_module'):
        __import__(import_7185)
        sys_modules_7186 = sys.modules[import_7185]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_7186.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_7185)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.testing import assert_, assert_allclose, assert_equal' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_7187 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing')

if (type(import_7187) is not StypyTypeError):

    if (import_7187 != 'pyd_module'):
        __import__(import_7187)
        sys_modules_7188 = sys.modules[import_7187]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', sys_modules_7188.module_type_store, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_7188, sys_modules_7188.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose', 'assert_equal'], [assert_, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', import_7187)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from pytest import assert_raises' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_7189 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'pytest')

if (type(import_7189) is not StypyTypeError):

    if (import_7189 != 'pyd_module'):
        __import__(import_7189)
        sys_modules_7190 = sys.modules[import_7189]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'pytest', sys_modules_7190.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_7190, sys_modules_7190.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'pytest', import_7189)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.io.netcdf import netcdf_file' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_7191 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.io.netcdf')

if (type(import_7191) is not StypyTypeError):

    if (import_7191 != 'pyd_module'):
        __import__(import_7191)
        sys_modules_7192 = sys.modules[import_7191]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.io.netcdf', sys_modules_7192.module_type_store, module_type_store, ['netcdf_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_7192, sys_modules_7192.module_type_store, module_type_store)
    else:
        from scipy.io.netcdf import netcdf_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.io.netcdf', None, module_type_store, ['netcdf_file'], [netcdf_file])

else:
    # Assigning a type to the variable 'scipy.io.netcdf' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.io.netcdf', import_7191)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_7193 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._numpy_compat')

if (type(import_7193) is not StypyTypeError):

    if (import_7193 != 'pyd_module'):
        __import__(import_7193)
        sys_modules_7194 = sys.modules[import_7193]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._numpy_compat', sys_modules_7194.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_7194, sys_modules_7194.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._numpy_compat', import_7193)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy._lib._tmpdirs import in_tempdir' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_7195 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy._lib._tmpdirs')

if (type(import_7195) is not StypyTypeError):

    if (import_7195 != 'pyd_module'):
        __import__(import_7195)
        sys_modules_7196 = sys.modules[import_7195]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy._lib._tmpdirs', sys_modules_7196.module_type_store, module_type_store, ['in_tempdir'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_7196, sys_modules_7196.module_type_store, module_type_store)
    else:
        from scipy._lib._tmpdirs import in_tempdir

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy._lib._tmpdirs', None, module_type_store, ['in_tempdir'], [in_tempdir])

else:
    # Assigning a type to the variable 'scipy._lib._tmpdirs' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy._lib._tmpdirs', import_7195)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')


# Assigning a Call to a Name (line 22):

# Call to pjoin(...): (line 22)
# Processing the call arguments (line 22)

# Call to dirname(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of '__file__' (line 22)
file___7199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), '__file__', False)
# Processing the call keyword arguments (line 22)
kwargs_7200 = {}
# Getting the type of 'dirname' (line 22)
dirname_7198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'dirname', False)
# Calling dirname(args, kwargs) (line 22)
dirname_call_result_7201 = invoke(stypy.reporting.localization.Localization(__file__, 22, 23), dirname_7198, *[file___7199], **kwargs_7200)

str_7202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 42), 'str', 'data')
# Processing the call keyword arguments (line 22)
kwargs_7203 = {}
# Getting the type of 'pjoin' (line 22)
pjoin_7197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 22)
pjoin_call_result_7204 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), pjoin_7197, *[dirname_call_result_7201, str_7202], **kwargs_7203)

# Assigning a type to the variable 'TEST_DATA_PATH' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'TEST_DATA_PATH', pjoin_call_result_7204)

# Assigning a Num to a Name (line 24):
int_7205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'int')
# Assigning a type to the variable 'N_EG_ELS' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'N_EG_ELS', int_7205)

# Assigning a Str to a Name (line 25):
str_7206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'str', 'b')
# Assigning a type to the variable 'VARTYPE_EG' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'VARTYPE_EG', str_7206)

@norecursion
def make_simple(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'make_simple'
    module_type_store = module_type_store.open_function_context('make_simple', 28, 0, False)
    
    # Passed parameters checking function
    make_simple.stypy_localization = localization
    make_simple.stypy_type_of_self = None
    make_simple.stypy_type_store = module_type_store
    make_simple.stypy_function_name = 'make_simple'
    make_simple.stypy_param_names_list = []
    make_simple.stypy_varargs_param_name = 'args'
    make_simple.stypy_kwargs_param_name = 'kwargs'
    make_simple.stypy_call_defaults = defaults
    make_simple.stypy_call_varargs = varargs
    make_simple.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_simple', [], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_simple', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_simple(...)' code ##################

    
    # Assigning a Call to a Name (line 30):
    
    # Call to netcdf_file(...): (line 30)
    # Getting the type of 'args' (line 30)
    args_7208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'args', False)
    # Processing the call keyword arguments (line 30)
    # Getting the type of 'kwargs' (line 30)
    kwargs_7209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'kwargs', False)
    kwargs_7210 = {'kwargs_7209': kwargs_7209}
    # Getting the type of 'netcdf_file' (line 30)
    netcdf_file_7207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 30)
    netcdf_file_call_result_7211 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), netcdf_file_7207, *[args_7208], **kwargs_7210)
    
    # Assigning a type to the variable 'f' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'f', netcdf_file_call_result_7211)
    
    # Assigning a Str to a Attribute (line 31):
    str_7212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'str', 'Created for a test')
    # Getting the type of 'f' (line 31)
    f_7213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'f')
    # Setting the type of the member 'history' of a type (line 31)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 4), f_7213, 'history', str_7212)
    
    # Call to createDimension(...): (line 32)
    # Processing the call arguments (line 32)
    str_7216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 22), 'str', 'time')
    # Getting the type of 'N_EG_ELS' (line 32)
    N_EG_ELS_7217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 30), 'N_EG_ELS', False)
    # Processing the call keyword arguments (line 32)
    kwargs_7218 = {}
    # Getting the type of 'f' (line 32)
    f_7214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'f', False)
    # Obtaining the member 'createDimension' of a type (line 32)
    createDimension_7215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 4), f_7214, 'createDimension')
    # Calling createDimension(args, kwargs) (line 32)
    createDimension_call_result_7219 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), createDimension_7215, *[str_7216, N_EG_ELS_7217], **kwargs_7218)
    
    
    # Assigning a Call to a Name (line 33):
    
    # Call to createVariable(...): (line 33)
    # Processing the call arguments (line 33)
    str_7222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'str', 'time')
    # Getting the type of 'VARTYPE_EG' (line 33)
    VARTYPE_EG_7223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 36), 'VARTYPE_EG', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 33)
    tuple_7224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 33)
    # Adding element type (line 33)
    str_7225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 49), 'str', 'time')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 49), tuple_7224, str_7225)
    
    # Processing the call keyword arguments (line 33)
    kwargs_7226 = {}
    # Getting the type of 'f' (line 33)
    f_7220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'f', False)
    # Obtaining the member 'createVariable' of a type (line 33)
    createVariable_7221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 11), f_7220, 'createVariable')
    # Calling createVariable(args, kwargs) (line 33)
    createVariable_call_result_7227 = invoke(stypy.reporting.localization.Localization(__file__, 33, 11), createVariable_7221, *[str_7222, VARTYPE_EG_7223, tuple_7224], **kwargs_7226)
    
    # Assigning a type to the variable 'time' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'time', createVariable_call_result_7227)
    
    # Assigning a Call to a Subscript (line 34):
    
    # Call to arange(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'N_EG_ELS' (line 34)
    N_EG_ELS_7230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'N_EG_ELS', False)
    # Processing the call keyword arguments (line 34)
    kwargs_7231 = {}
    # Getting the type of 'np' (line 34)
    np_7228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'np', False)
    # Obtaining the member 'arange' of a type (line 34)
    arange_7229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 14), np_7228, 'arange')
    # Calling arange(args, kwargs) (line 34)
    arange_call_result_7232 = invoke(stypy.reporting.localization.Localization(__file__, 34, 14), arange_7229, *[N_EG_ELS_7230], **kwargs_7231)
    
    # Getting the type of 'time' (line 34)
    time_7233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'time')
    slice_7234 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 34, 4), None, None, None)
    # Storing an element on a container (line 34)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 4), time_7233, (slice_7234, arange_call_result_7232))
    
    # Assigning a Str to a Attribute (line 35):
    str_7235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'str', 'days since 2008-01-01')
    # Getting the type of 'time' (line 35)
    time_7236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'time')
    # Setting the type of the member 'units' of a type (line 35)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), time_7236, 'units', str_7235)
    
    # Call to flush(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_7239 = {}
    # Getting the type of 'f' (line 36)
    f_7237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'f', False)
    # Obtaining the member 'flush' of a type (line 36)
    flush_7238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), f_7237, 'flush')
    # Calling flush(args, kwargs) (line 36)
    flush_call_result_7240 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), flush_7238, *[], **kwargs_7239)
    
    # Creating a generator
    # Getting the type of 'f' (line 37)
    f_7241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'f')
    GeneratorType_7242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 4), GeneratorType_7242, f_7241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', GeneratorType_7242)
    
    # Call to close(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_7245 = {}
    # Getting the type of 'f' (line 38)
    f_7243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 38)
    close_7244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), f_7243, 'close')
    # Calling close(args, kwargs) (line 38)
    close_call_result_7246 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), close_7244, *[], **kwargs_7245)
    
    
    # ################# End of 'make_simple(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_simple' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_7247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7247)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_simple'
    return stypy_return_type_7247

# Assigning a type to the variable 'make_simple' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'make_simple', make_simple)

@norecursion
def check_simple(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_simple'
    module_type_store = module_type_store.open_function_context('check_simple', 41, 0, False)
    
    # Passed parameters checking function
    check_simple.stypy_localization = localization
    check_simple.stypy_type_of_self = None
    check_simple.stypy_type_store = module_type_store
    check_simple.stypy_function_name = 'check_simple'
    check_simple.stypy_param_names_list = ['ncfileobj']
    check_simple.stypy_varargs_param_name = None
    check_simple.stypy_kwargs_param_name = None
    check_simple.stypy_call_defaults = defaults
    check_simple.stypy_call_varargs = varargs
    check_simple.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_simple', ['ncfileobj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_simple', localization, ['ncfileobj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_simple(...)' code ##################

    str_7248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'str', 'Example fileobj tests ')
    
    # Call to assert_equal(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'ncfileobj' (line 43)
    ncfileobj_7250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'ncfileobj', False)
    # Obtaining the member 'history' of a type (line 43)
    history_7251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 17), ncfileobj_7250, 'history')
    str_7252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 36), 'str', 'Created for a test')
    # Processing the call keyword arguments (line 43)
    kwargs_7253 = {}
    # Getting the type of 'assert_equal' (line 43)
    assert_equal_7249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 43)
    assert_equal_call_result_7254 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), assert_equal_7249, *[history_7251, str_7252], **kwargs_7253)
    
    
    # Assigning a Subscript to a Name (line 44):
    
    # Obtaining the type of the subscript
    str_7255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'str', 'time')
    # Getting the type of 'ncfileobj' (line 44)
    ncfileobj_7256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'ncfileobj')
    # Obtaining the member 'variables' of a type (line 44)
    variables_7257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), ncfileobj_7256, 'variables')
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___7258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), variables_7257, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_7259 = invoke(stypy.reporting.localization.Localization(__file__, 44, 11), getitem___7258, str_7255)
    
    # Assigning a type to the variable 'time' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'time', subscript_call_result_7259)
    
    # Call to assert_equal(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'time' (line 45)
    time_7261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'time', False)
    # Obtaining the member 'units' of a type (line 45)
    units_7262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 17), time_7261, 'units')
    str_7263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 29), 'str', 'days since 2008-01-01')
    # Processing the call keyword arguments (line 45)
    kwargs_7264 = {}
    # Getting the type of 'assert_equal' (line 45)
    assert_equal_7260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 45)
    assert_equal_call_result_7265 = invoke(stypy.reporting.localization.Localization(__file__, 45, 4), assert_equal_7260, *[units_7262, str_7263], **kwargs_7264)
    
    
    # Call to assert_equal(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'time' (line 46)
    time_7267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'time', False)
    # Obtaining the member 'shape' of a type (line 46)
    shape_7268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 17), time_7267, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_7269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'N_EG_ELS' (line 46)
    N_EG_ELS_7270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'N_EG_ELS', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 30), tuple_7269, N_EG_ELS_7270)
    
    # Processing the call keyword arguments (line 46)
    kwargs_7271 = {}
    # Getting the type of 'assert_equal' (line 46)
    assert_equal_7266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 46)
    assert_equal_call_result_7272 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), assert_equal_7266, *[shape_7268, tuple_7269], **kwargs_7271)
    
    
    # Call to assert_equal(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Obtaining the type of the subscript
    int_7274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'int')
    # Getting the type of 'time' (line 47)
    time_7275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'time', False)
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___7276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 17), time_7275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_7277 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), getitem___7276, int_7274)
    
    # Getting the type of 'N_EG_ELS' (line 47)
    N_EG_ELS_7278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'N_EG_ELS', False)
    int_7279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 36), 'int')
    # Applying the binary operator '-' (line 47)
    result_sub_7280 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 27), '-', N_EG_ELS_7278, int_7279)
    
    # Processing the call keyword arguments (line 47)
    kwargs_7281 = {}
    # Getting the type of 'assert_equal' (line 47)
    assert_equal_7273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 47)
    assert_equal_call_result_7282 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), assert_equal_7273, *[subscript_call_result_7277, result_sub_7280], **kwargs_7281)
    
    
    # ################# End of 'check_simple(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_simple' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_7283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7283)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_simple'
    return stypy_return_type_7283

# Assigning a type to the variable 'check_simple' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'check_simple', check_simple)

@norecursion
def assert_mask_matches(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assert_mask_matches'
    module_type_store = module_type_store.open_function_context('assert_mask_matches', 49, 0, False)
    
    # Passed parameters checking function
    assert_mask_matches.stypy_localization = localization
    assert_mask_matches.stypy_type_of_self = None
    assert_mask_matches.stypy_type_store = module_type_store
    assert_mask_matches.stypy_function_name = 'assert_mask_matches'
    assert_mask_matches.stypy_param_names_list = ['arr', 'expected_mask']
    assert_mask_matches.stypy_varargs_param_name = None
    assert_mask_matches.stypy_kwargs_param_name = None
    assert_mask_matches.stypy_call_defaults = defaults
    assert_mask_matches.stypy_call_varargs = varargs
    assert_mask_matches.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_mask_matches', ['arr', 'expected_mask'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_mask_matches', localization, ['arr', 'expected_mask'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_mask_matches(...)' code ##################

    str_7284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'str', "\n    Asserts that the mask of arr is effectively the same as expected_mask.\n\n    In contrast to numpy.ma.testutils.assert_mask_equal, this function allows\n    testing the 'mask' of a standard numpy array (the mask in this case is treated\n    as all False).\n\n    Parameters\n    ----------\n    arr: ndarray or MaskedArray\n        Array to test.\n    expected_mask: array_like of booleans\n        A list giving the expected mask.\n    ")
    
    # Assigning a Call to a Name (line 65):
    
    # Call to getmaskarray(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'arr' (line 65)
    arr_7288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'arr', False)
    # Processing the call keyword arguments (line 65)
    kwargs_7289 = {}
    # Getting the type of 'np' (line 65)
    np_7285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'np', False)
    # Obtaining the member 'ma' of a type (line 65)
    ma_7286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), np_7285, 'ma')
    # Obtaining the member 'getmaskarray' of a type (line 65)
    getmaskarray_7287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), ma_7286, 'getmaskarray')
    # Calling getmaskarray(args, kwargs) (line 65)
    getmaskarray_call_result_7290 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), getmaskarray_7287, *[arr_7288], **kwargs_7289)
    
    # Assigning a type to the variable 'mask' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'mask', getmaskarray_call_result_7290)
    
    # Call to assert_equal(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'mask' (line 66)
    mask_7292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'mask', False)
    # Getting the type of 'expected_mask' (line 66)
    expected_mask_7293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'expected_mask', False)
    # Processing the call keyword arguments (line 66)
    kwargs_7294 = {}
    # Getting the type of 'assert_equal' (line 66)
    assert_equal_7291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 66)
    assert_equal_call_result_7295 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), assert_equal_7291, *[mask_7292, expected_mask_7293], **kwargs_7294)
    
    
    # ################# End of 'assert_mask_matches(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_mask_matches' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_7296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7296)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_mask_matches'
    return stypy_return_type_7296

# Assigning a type to the variable 'assert_mask_matches' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'assert_mask_matches', assert_mask_matches)

@norecursion
def test_read_write_files(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_write_files'
    module_type_store = module_type_store.open_function_context('test_read_write_files', 69, 0, False)
    
    # Passed parameters checking function
    test_read_write_files.stypy_localization = localization
    test_read_write_files.stypy_type_of_self = None
    test_read_write_files.stypy_type_store = module_type_store
    test_read_write_files.stypy_function_name = 'test_read_write_files'
    test_read_write_files.stypy_param_names_list = []
    test_read_write_files.stypy_varargs_param_name = None
    test_read_write_files.stypy_kwargs_param_name = None
    test_read_write_files.stypy_call_defaults = defaults
    test_read_write_files.stypy_call_varargs = varargs
    test_read_write_files.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_write_files', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_write_files', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_write_files(...)' code ##################

    
    # Assigning a Call to a Name (line 71):
    
    # Call to getcwd(...): (line 71)
    # Processing the call keyword arguments (line 71)
    kwargs_7299 = {}
    # Getting the type of 'os' (line 71)
    os_7297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'os', False)
    # Obtaining the member 'getcwd' of a type (line 71)
    getcwd_7298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 10), os_7297, 'getcwd')
    # Calling getcwd(args, kwargs) (line 71)
    getcwd_call_result_7300 = invoke(stypy.reporting.localization.Localization(__file__, 71, 10), getcwd_7298, *[], **kwargs_7299)
    
    # Assigning a type to the variable 'cwd' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'cwd', getcwd_call_result_7300)
    
    
    # SSA begins for try-except statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 73):
    
    # Call to mkdtemp(...): (line 73)
    # Processing the call keyword arguments (line 73)
    kwargs_7303 = {}
    # Getting the type of 'tempfile' (line 73)
    tempfile_7301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'tempfile', False)
    # Obtaining the member 'mkdtemp' of a type (line 73)
    mkdtemp_7302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 17), tempfile_7301, 'mkdtemp')
    # Calling mkdtemp(args, kwargs) (line 73)
    mkdtemp_call_result_7304 = invoke(stypy.reporting.localization.Localization(__file__, 73, 17), mkdtemp_7302, *[], **kwargs_7303)
    
    # Assigning a type to the variable 'tmpdir' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tmpdir', mkdtemp_call_result_7304)
    
    # Call to chdir(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'tmpdir' (line 74)
    tmpdir_7307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'tmpdir', False)
    # Processing the call keyword arguments (line 74)
    kwargs_7308 = {}
    # Getting the type of 'os' (line 74)
    os_7305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'os', False)
    # Obtaining the member 'chdir' of a type (line 74)
    chdir_7306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), os_7305, 'chdir')
    # Calling chdir(args, kwargs) (line 74)
    chdir_call_result_7309 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), chdir_7306, *[tmpdir_7307], **kwargs_7308)
    
    
    # Call to make_simple(...): (line 75)
    # Processing the call arguments (line 75)
    str_7311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'str', 'simple.nc')
    str_7312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 38), 'str', 'w')
    # Processing the call keyword arguments (line 75)
    kwargs_7313 = {}
    # Getting the type of 'make_simple' (line 75)
    make_simple_7310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'make_simple', False)
    # Calling make_simple(args, kwargs) (line 75)
    make_simple_call_result_7314 = invoke(stypy.reporting.localization.Localization(__file__, 75, 13), make_simple_7310, *[str_7311, str_7312], **kwargs_7313)
    
    with_7315 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 75, 13), make_simple_call_result_7314, 'with parameter', '__enter__', '__exit__')

    if with_7315:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 75)
        enter___7316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 13), make_simple_call_result_7314, '__enter__')
        with_enter_7317 = invoke(stypy.reporting.localization.Localization(__file__, 75, 13), enter___7316)
        # Assigning a type to the variable 'f' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'f', with_enter_7317)
        pass
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 75)
        exit___7318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 13), make_simple_call_result_7314, '__exit__')
        with_exit_7319 = invoke(stypy.reporting.localization.Localization(__file__, 75, 13), exit___7318, None, None, None)

    
    # Call to netcdf_file(...): (line 78)
    # Processing the call arguments (line 78)
    str_7321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 25), 'str', 'simple.nc')
    str_7322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 38), 'str', 'a')
    # Processing the call keyword arguments (line 78)
    kwargs_7323 = {}
    # Getting the type of 'netcdf_file' (line 78)
    netcdf_file_7320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 78)
    netcdf_file_call_result_7324 = invoke(stypy.reporting.localization.Localization(__file__, 78, 13), netcdf_file_7320, *[str_7321, str_7322], **kwargs_7323)
    
    with_7325 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 78, 13), netcdf_file_call_result_7324, 'with parameter', '__enter__', '__exit__')

    if with_7325:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 78)
        enter___7326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 13), netcdf_file_call_result_7324, '__enter__')
        with_enter_7327 = invoke(stypy.reporting.localization.Localization(__file__, 78, 13), enter___7326)
        # Assigning a type to the variable 'f' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'f', with_enter_7327)
        
        # Call to check_simple(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'f' (line 79)
        f_7329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'f', False)
        # Processing the call keyword arguments (line 79)
        kwargs_7330 = {}
        # Getting the type of 'check_simple' (line 79)
        check_simple_7328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'check_simple', False)
        # Calling check_simple(args, kwargs) (line 79)
        check_simple_call_result_7331 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), check_simple_7328, *[f_7329], **kwargs_7330)
        
        
        # Assigning a Num to a Subscript (line 81):
        int_7332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 41), 'int')
        # Getting the type of 'f' (line 81)
        f_7333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'f')
        # Obtaining the member '_attributes' of a type (line 81)
        _attributes_7334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), f_7333, '_attributes')
        str_7335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 26), 'str', 'appendRan')
        # Storing an element on a container (line 81)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 12), _attributes_7334, (str_7335, int_7332))
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 78)
        exit___7336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 13), netcdf_file_call_result_7324, '__exit__')
        with_exit_7337 = invoke(stypy.reporting.localization.Localization(__file__, 78, 13), exit___7336, None, None, None)

    
    # Call to netcdf_file(...): (line 84)
    # Processing the call arguments (line 84)
    str_7339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 25), 'str', 'simple.nc')
    # Processing the call keyword arguments (line 84)
    kwargs_7340 = {}
    # Getting the type of 'netcdf_file' (line 84)
    netcdf_file_7338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 84)
    netcdf_file_call_result_7341 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), netcdf_file_7338, *[str_7339], **kwargs_7340)
    
    with_7342 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 84, 13), netcdf_file_call_result_7341, 'with parameter', '__enter__', '__exit__')

    if with_7342:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 84)
        enter___7343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), netcdf_file_call_result_7341, '__enter__')
        with_enter_7344 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), enter___7343)
        # Assigning a type to the variable 'f' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'f', with_enter_7344)
        
        # Call to assert_(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'f' (line 86)
        f_7346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'f', False)
        # Obtaining the member 'use_mmap' of a type (line 86)
        use_mmap_7347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 20), f_7346, 'use_mmap')
        # Processing the call keyword arguments (line 86)
        kwargs_7348 = {}
        # Getting the type of 'assert_' (line 86)
        assert__7345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 86)
        assert__call_result_7349 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), assert__7345, *[use_mmap_7347], **kwargs_7348)
        
        
        # Call to check_simple(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'f' (line 87)
        f_7351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'f', False)
        # Processing the call keyword arguments (line 87)
        kwargs_7352 = {}
        # Getting the type of 'check_simple' (line 87)
        check_simple_7350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'check_simple', False)
        # Calling check_simple(args, kwargs) (line 87)
        check_simple_call_result_7353 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), check_simple_7350, *[f_7351], **kwargs_7352)
        
        
        # Call to assert_equal(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining the type of the subscript
        str_7355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 39), 'str', 'appendRan')
        # Getting the type of 'f' (line 88)
        f_7356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'f', False)
        # Obtaining the member '_attributes' of a type (line 88)
        _attributes_7357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 25), f_7356, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___7358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 25), _attributes_7357, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_7359 = invoke(stypy.reporting.localization.Localization(__file__, 88, 25), getitem___7358, str_7355)
        
        int_7360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 53), 'int')
        # Processing the call keyword arguments (line 88)
        kwargs_7361 = {}
        # Getting the type of 'assert_equal' (line 88)
        assert_equal_7354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 88)
        assert_equal_call_result_7362 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), assert_equal_7354, *[subscript_call_result_7359, int_7360], **kwargs_7361)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 84)
        exit___7363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), netcdf_file_call_result_7341, '__exit__')
        with_exit_7364 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), exit___7363, None, None, None)

    
    # Call to netcdf_file(...): (line 91)
    # Processing the call arguments (line 91)
    str_7366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 25), 'str', 'simple.nc')
    str_7367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 38), 'str', 'a')
    # Processing the call keyword arguments (line 91)
    kwargs_7368 = {}
    # Getting the type of 'netcdf_file' (line 91)
    netcdf_file_7365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 91)
    netcdf_file_call_result_7369 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), netcdf_file_7365, *[str_7366, str_7367], **kwargs_7368)
    
    with_7370 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 91, 13), netcdf_file_call_result_7369, 'with parameter', '__enter__', '__exit__')

    if with_7370:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 91)
        enter___7371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), netcdf_file_call_result_7369, '__enter__')
        with_enter_7372 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), enter___7371)
        # Assigning a type to the variable 'f' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'f', with_enter_7372)
        
        # Call to assert_(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Getting the type of 'f' (line 92)
        f_7374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'f', False)
        # Obtaining the member 'use_mmap' of a type (line 92)
        use_mmap_7375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), f_7374, 'use_mmap')
        # Applying the 'not' unary operator (line 92)
        result_not__7376 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 20), 'not', use_mmap_7375)
        
        # Processing the call keyword arguments (line 92)
        kwargs_7377 = {}
        # Getting the type of 'assert_' (line 92)
        assert__7373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 92)
        assert__call_result_7378 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), assert__7373, *[result_not__7376], **kwargs_7377)
        
        
        # Call to check_simple(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'f' (line 93)
        f_7380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'f', False)
        # Processing the call keyword arguments (line 93)
        kwargs_7381 = {}
        # Getting the type of 'check_simple' (line 93)
        check_simple_7379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'check_simple', False)
        # Calling check_simple(args, kwargs) (line 93)
        check_simple_call_result_7382 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), check_simple_7379, *[f_7380], **kwargs_7381)
        
        
        # Call to assert_equal(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Obtaining the type of the subscript
        str_7384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 39), 'str', 'appendRan')
        # Getting the type of 'f' (line 94)
        f_7385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'f', False)
        # Obtaining the member '_attributes' of a type (line 94)
        _attributes_7386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 25), f_7385, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___7387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 25), _attributes_7386, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_7388 = invoke(stypy.reporting.localization.Localization(__file__, 94, 25), getitem___7387, str_7384)
        
        int_7389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 53), 'int')
        # Processing the call keyword arguments (line 94)
        kwargs_7390 = {}
        # Getting the type of 'assert_equal' (line 94)
        assert_equal_7383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 94)
        assert_equal_call_result_7391 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), assert_equal_7383, *[subscript_call_result_7388, int_7389], **kwargs_7390)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 91)
        exit___7392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), netcdf_file_call_result_7369, '__exit__')
        with_exit_7393 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), exit___7392, None, None, None)

    
    # Call to netcdf_file(...): (line 97)
    # Processing the call arguments (line 97)
    str_7395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 25), 'str', 'simple.nc')
    # Processing the call keyword arguments (line 97)
    # Getting the type of 'False' (line 97)
    False_7396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 43), 'False', False)
    keyword_7397 = False_7396
    kwargs_7398 = {'mmap': keyword_7397}
    # Getting the type of 'netcdf_file' (line 97)
    netcdf_file_7394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 97)
    netcdf_file_call_result_7399 = invoke(stypy.reporting.localization.Localization(__file__, 97, 13), netcdf_file_7394, *[str_7395], **kwargs_7398)
    
    with_7400 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 97, 13), netcdf_file_call_result_7399, 'with parameter', '__enter__', '__exit__')

    if with_7400:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 97)
        enter___7401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 13), netcdf_file_call_result_7399, '__enter__')
        with_enter_7402 = invoke(stypy.reporting.localization.Localization(__file__, 97, 13), enter___7401)
        # Assigning a type to the variable 'f' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'f', with_enter_7402)
        
        # Call to assert_(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Getting the type of 'f' (line 99)
        f_7404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'f', False)
        # Obtaining the member 'use_mmap' of a type (line 99)
        use_mmap_7405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 24), f_7404, 'use_mmap')
        # Applying the 'not' unary operator (line 99)
        result_not__7406 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 20), 'not', use_mmap_7405)
        
        # Processing the call keyword arguments (line 99)
        kwargs_7407 = {}
        # Getting the type of 'assert_' (line 99)
        assert__7403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 99)
        assert__call_result_7408 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), assert__7403, *[result_not__7406], **kwargs_7407)
        
        
        # Call to check_simple(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'f' (line 100)
        f_7410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'f', False)
        # Processing the call keyword arguments (line 100)
        kwargs_7411 = {}
        # Getting the type of 'check_simple' (line 100)
        check_simple_7409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'check_simple', False)
        # Calling check_simple(args, kwargs) (line 100)
        check_simple_call_result_7412 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), check_simple_7409, *[f_7410], **kwargs_7411)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 97)
        exit___7413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 13), netcdf_file_call_result_7399, '__exit__')
        with_exit_7414 = invoke(stypy.reporting.localization.Localization(__file__, 97, 13), exit___7413, None, None, None)

    
    # Call to open(...): (line 107)
    # Processing the call arguments (line 107)
    str_7416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 18), 'str', 'simple.nc')
    str_7417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 31), 'str', 'rb')
    # Processing the call keyword arguments (line 107)
    kwargs_7418 = {}
    # Getting the type of 'open' (line 107)
    open_7415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'open', False)
    # Calling open(args, kwargs) (line 107)
    open_call_result_7419 = invoke(stypy.reporting.localization.Localization(__file__, 107, 13), open_7415, *[str_7416, str_7417], **kwargs_7418)
    
    with_7420 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 107, 13), open_call_result_7419, 'with parameter', '__enter__', '__exit__')

    if with_7420:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 107)
        enter___7421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 13), open_call_result_7419, '__enter__')
        with_enter_7422 = invoke(stypy.reporting.localization.Localization(__file__, 107, 13), enter___7421)
        # Assigning a type to the variable 'fobj' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'fobj', with_enter_7422)
        
        # Call to netcdf_file(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'fobj' (line 108)
        fobj_7424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'fobj', False)
        # Processing the call keyword arguments (line 108)
        kwargs_7425 = {}
        # Getting the type of 'netcdf_file' (line 108)
        netcdf_file_7423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 108)
        netcdf_file_call_result_7426 = invoke(stypy.reporting.localization.Localization(__file__, 108, 17), netcdf_file_7423, *[fobj_7424], **kwargs_7425)
        
        with_7427 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 108, 17), netcdf_file_call_result_7426, 'with parameter', '__enter__', '__exit__')

        if with_7427:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 108)
            enter___7428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 17), netcdf_file_call_result_7426, '__enter__')
            with_enter_7429 = invoke(stypy.reporting.localization.Localization(__file__, 108, 17), enter___7428)
            # Assigning a type to the variable 'f' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'f', with_enter_7429)
            
            # Call to assert_(...): (line 110)
            # Processing the call arguments (line 110)
            
            # Getting the type of 'f' (line 110)
            f_7431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'f', False)
            # Obtaining the member 'use_mmap' of a type (line 110)
            use_mmap_7432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 28), f_7431, 'use_mmap')
            # Applying the 'not' unary operator (line 110)
            result_not__7433 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 24), 'not', use_mmap_7432)
            
            # Processing the call keyword arguments (line 110)
            kwargs_7434 = {}
            # Getting the type of 'assert_' (line 110)
            assert__7430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'assert_', False)
            # Calling assert_(args, kwargs) (line 110)
            assert__call_result_7435 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), assert__7430, *[result_not__7433], **kwargs_7434)
            
            
            # Call to check_simple(...): (line 111)
            # Processing the call arguments (line 111)
            # Getting the type of 'f' (line 111)
            f_7437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 29), 'f', False)
            # Processing the call keyword arguments (line 111)
            kwargs_7438 = {}
            # Getting the type of 'check_simple' (line 111)
            check_simple_7436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'check_simple', False)
            # Calling check_simple(args, kwargs) (line 111)
            check_simple_call_result_7439 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), check_simple_7436, *[f_7437], **kwargs_7438)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 108)
            exit___7440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 17), netcdf_file_call_result_7426, '__exit__')
            with_exit_7441 = invoke(stypy.reporting.localization.Localization(__file__, 108, 17), exit___7440, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 107)
        exit___7442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 13), open_call_result_7419, '__exit__')
        with_exit_7443 = invoke(stypy.reporting.localization.Localization(__file__, 107, 13), exit___7442, None, None, None)

    
    # Call to open(...): (line 114)
    # Processing the call arguments (line 114)
    str_7445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 18), 'str', 'simple.nc')
    str_7446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 31), 'str', 'rb')
    # Processing the call keyword arguments (line 114)
    kwargs_7447 = {}
    # Getting the type of 'open' (line 114)
    open_7444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'open', False)
    # Calling open(args, kwargs) (line 114)
    open_call_result_7448 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), open_7444, *[str_7445, str_7446], **kwargs_7447)
    
    with_7449 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 114, 13), open_call_result_7448, 'with parameter', '__enter__', '__exit__')

    if with_7449:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 114)
        enter___7450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 13), open_call_result_7448, '__enter__')
        with_enter_7451 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), enter___7450)
        # Assigning a type to the variable 'fobj' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'fobj', with_enter_7451)
        
        # Call to netcdf_file(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'fobj' (line 115)
        fobj_7453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'fobj', False)
        # Processing the call keyword arguments (line 115)
        # Getting the type of 'True' (line 115)
        True_7454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 40), 'True', False)
        keyword_7455 = True_7454
        kwargs_7456 = {'mmap': keyword_7455}
        # Getting the type of 'netcdf_file' (line 115)
        netcdf_file_7452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 115)
        netcdf_file_call_result_7457 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), netcdf_file_7452, *[fobj_7453], **kwargs_7456)
        
        with_7458 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 115, 17), netcdf_file_call_result_7457, 'with parameter', '__enter__', '__exit__')

        if with_7458:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 115)
            enter___7459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), netcdf_file_call_result_7457, '__enter__')
            with_enter_7460 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), enter___7459)
            # Assigning a type to the variable 'f' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'f', with_enter_7460)
            
            # Call to assert_(...): (line 116)
            # Processing the call arguments (line 116)
            # Getting the type of 'f' (line 116)
            f_7462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'f', False)
            # Obtaining the member 'use_mmap' of a type (line 116)
            use_mmap_7463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), f_7462, 'use_mmap')
            # Processing the call keyword arguments (line 116)
            kwargs_7464 = {}
            # Getting the type of 'assert_' (line 116)
            assert__7461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'assert_', False)
            # Calling assert_(args, kwargs) (line 116)
            assert__call_result_7465 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), assert__7461, *[use_mmap_7463], **kwargs_7464)
            
            
            # Call to check_simple(...): (line 117)
            # Processing the call arguments (line 117)
            # Getting the type of 'f' (line 117)
            f_7467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'f', False)
            # Processing the call keyword arguments (line 117)
            kwargs_7468 = {}
            # Getting the type of 'check_simple' (line 117)
            check_simple_7466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'check_simple', False)
            # Calling check_simple(args, kwargs) (line 117)
            check_simple_call_result_7469 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), check_simple_7466, *[f_7467], **kwargs_7468)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 115)
            exit___7470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), netcdf_file_call_result_7457, '__exit__')
            with_exit_7471 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), exit___7470, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 114)
        exit___7472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 13), open_call_result_7448, '__exit__')
        with_exit_7473 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), exit___7472, None, None, None)

    
    # Call to open(...): (line 120)
    # Processing the call arguments (line 120)
    str_7475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'str', 'simple.nc')
    str_7476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'str', 'r+b')
    # Processing the call keyword arguments (line 120)
    kwargs_7477 = {}
    # Getting the type of 'open' (line 120)
    open_7474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 13), 'open', False)
    # Calling open(args, kwargs) (line 120)
    open_call_result_7478 = invoke(stypy.reporting.localization.Localization(__file__, 120, 13), open_7474, *[str_7475, str_7476], **kwargs_7477)
    
    with_7479 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 120, 13), open_call_result_7478, 'with parameter', '__enter__', '__exit__')

    if with_7479:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 120)
        enter___7480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 13), open_call_result_7478, '__enter__')
        with_enter_7481 = invoke(stypy.reporting.localization.Localization(__file__, 120, 13), enter___7480)
        # Assigning a type to the variable 'fobj' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 13), 'fobj', with_enter_7481)
        
        # Call to netcdf_file(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'fobj' (line 121)
        fobj_7483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'fobj', False)
        str_7484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 35), 'str', 'a')
        # Processing the call keyword arguments (line 121)
        kwargs_7485 = {}
        # Getting the type of 'netcdf_file' (line 121)
        netcdf_file_7482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 121)
        netcdf_file_call_result_7486 = invoke(stypy.reporting.localization.Localization(__file__, 121, 17), netcdf_file_7482, *[fobj_7483, str_7484], **kwargs_7485)
        
        with_7487 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 121, 17), netcdf_file_call_result_7486, 'with parameter', '__enter__', '__exit__')

        if with_7487:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 121)
            enter___7488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 17), netcdf_file_call_result_7486, '__enter__')
            with_enter_7489 = invoke(stypy.reporting.localization.Localization(__file__, 121, 17), enter___7488)
            # Assigning a type to the variable 'f' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'f', with_enter_7489)
            
            # Call to assert_(...): (line 122)
            # Processing the call arguments (line 122)
            
            # Getting the type of 'f' (line 122)
            f_7491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'f', False)
            # Obtaining the member 'use_mmap' of a type (line 122)
            use_mmap_7492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 28), f_7491, 'use_mmap')
            # Applying the 'not' unary operator (line 122)
            result_not__7493 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 24), 'not', use_mmap_7492)
            
            # Processing the call keyword arguments (line 122)
            kwargs_7494 = {}
            # Getting the type of 'assert_' (line 122)
            assert__7490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'assert_', False)
            # Calling assert_(args, kwargs) (line 122)
            assert__call_result_7495 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), assert__7490, *[result_not__7493], **kwargs_7494)
            
            
            # Call to check_simple(...): (line 123)
            # Processing the call arguments (line 123)
            # Getting the type of 'f' (line 123)
            f_7497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'f', False)
            # Processing the call keyword arguments (line 123)
            kwargs_7498 = {}
            # Getting the type of 'check_simple' (line 123)
            check_simple_7496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'check_simple', False)
            # Calling check_simple(args, kwargs) (line 123)
            check_simple_call_result_7499 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), check_simple_7496, *[f_7497], **kwargs_7498)
            
            
            # Call to createDimension(...): (line 124)
            # Processing the call arguments (line 124)
            str_7502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 34), 'str', 'app_dim')
            int_7503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 45), 'int')
            # Processing the call keyword arguments (line 124)
            kwargs_7504 = {}
            # Getting the type of 'f' (line 124)
            f_7500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'f', False)
            # Obtaining the member 'createDimension' of a type (line 124)
            createDimension_7501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), f_7500, 'createDimension')
            # Calling createDimension(args, kwargs) (line 124)
            createDimension_call_result_7505 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), createDimension_7501, *[str_7502, int_7503], **kwargs_7504)
            
            
            # Assigning a Call to a Name (line 125):
            
            # Call to createVariable(...): (line 125)
            # Processing the call arguments (line 125)
            str_7508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 39), 'str', 'app_var')
            str_7509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 50), 'str', 'i')
            
            # Obtaining an instance of the builtin type 'tuple' (line 125)
            tuple_7510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 56), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 125)
            # Adding element type (line 125)
            str_7511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 56), 'str', 'app_dim')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 56), tuple_7510, str_7511)
            
            # Processing the call keyword arguments (line 125)
            kwargs_7512 = {}
            # Getting the type of 'f' (line 125)
            f_7506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'f', False)
            # Obtaining the member 'createVariable' of a type (line 125)
            createVariable_7507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 22), f_7506, 'createVariable')
            # Calling createVariable(args, kwargs) (line 125)
            createVariable_call_result_7513 = invoke(stypy.reporting.localization.Localization(__file__, 125, 22), createVariable_7507, *[str_7508, str_7509, tuple_7510], **kwargs_7512)
            
            # Assigning a type to the variable 'var' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'var', createVariable_call_result_7513)
            
            # Assigning a Num to a Subscript (line 126):
            int_7514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 25), 'int')
            # Getting the type of 'var' (line 126)
            var_7515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'var')
            slice_7516 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 126, 16), None, None, None)
            # Storing an element on a container (line 126)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 16), var_7515, (slice_7516, int_7514))
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 121)
            exit___7517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 17), netcdf_file_call_result_7486, '__exit__')
            with_exit_7518 = invoke(stypy.reporting.localization.Localization(__file__, 121, 17), exit___7517, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 120)
        exit___7519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 13), open_call_result_7478, '__exit__')
        with_exit_7520 = invoke(stypy.reporting.localization.Localization(__file__, 120, 13), exit___7519, None, None, None)

    
    # Call to netcdf_file(...): (line 129)
    # Processing the call arguments (line 129)
    str_7522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 25), 'str', 'simple.nc')
    # Processing the call keyword arguments (line 129)
    kwargs_7523 = {}
    # Getting the type of 'netcdf_file' (line 129)
    netcdf_file_7521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 13), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 129)
    netcdf_file_call_result_7524 = invoke(stypy.reporting.localization.Localization(__file__, 129, 13), netcdf_file_7521, *[str_7522], **kwargs_7523)
    
    with_7525 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 129, 13), netcdf_file_call_result_7524, 'with parameter', '__enter__', '__exit__')

    if with_7525:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 129)
        enter___7526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 13), netcdf_file_call_result_7524, '__enter__')
        with_enter_7527 = invoke(stypy.reporting.localization.Localization(__file__, 129, 13), enter___7526)
        # Assigning a type to the variable 'f' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 13), 'f', with_enter_7527)
        
        # Call to check_simple(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'f' (line 130)
        f_7529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'f', False)
        # Processing the call keyword arguments (line 130)
        kwargs_7530 = {}
        # Getting the type of 'check_simple' (line 130)
        check_simple_7528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'check_simple', False)
        # Calling check_simple(args, kwargs) (line 130)
        check_simple_call_result_7531 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), check_simple_7528, *[f_7529], **kwargs_7530)
        
        
        # Call to assert_equal(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Obtaining the type of the subscript
        slice_7533 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 131, 25), None, None, None)
        
        # Obtaining the type of the subscript
        str_7534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 37), 'str', 'app_var')
        # Getting the type of 'f' (line 131)
        f_7535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'f', False)
        # Obtaining the member 'variables' of a type (line 131)
        variables_7536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 25), f_7535, 'variables')
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___7537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 25), variables_7536, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_7538 = invoke(stypy.reporting.localization.Localization(__file__, 131, 25), getitem___7537, str_7534)
        
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___7539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 25), subscript_call_result_7538, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_7540 = invoke(stypy.reporting.localization.Localization(__file__, 131, 25), getitem___7539, slice_7533)
        
        int_7541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 52), 'int')
        # Processing the call keyword arguments (line 131)
        kwargs_7542 = {}
        # Getting the type of 'assert_equal' (line 131)
        assert_equal_7532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 131)
        assert_equal_call_result_7543 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), assert_equal_7532, *[subscript_call_result_7540, int_7541], **kwargs_7542)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 129)
        exit___7544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 13), netcdf_file_call_result_7524, '__exit__')
        with_exit_7545 = invoke(stypy.reporting.localization.Localization(__file__, 129, 13), exit___7544, None, None, None)

    # SSA branch for the except part of a try statement (line 72)
    # SSA branch for the except '<any exception>' branch of a try statement (line 72)
    module_type_store.open_ssa_branch('except')
    
    # Call to chdir(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'cwd' (line 134)
    cwd_7548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'cwd', False)
    # Processing the call keyword arguments (line 134)
    kwargs_7549 = {}
    # Getting the type of 'os' (line 134)
    os_7546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'os', False)
    # Obtaining the member 'chdir' of a type (line 134)
    chdir_7547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), os_7546, 'chdir')
    # Calling chdir(args, kwargs) (line 134)
    chdir_call_result_7550 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), chdir_7547, *[cwd_7548], **kwargs_7549)
    
    
    # Call to rmtree(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'tmpdir' (line 135)
    tmpdir_7553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'tmpdir', False)
    # Processing the call keyword arguments (line 135)
    kwargs_7554 = {}
    # Getting the type of 'shutil' (line 135)
    shutil_7551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'shutil', False)
    # Obtaining the member 'rmtree' of a type (line 135)
    rmtree_7552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), shutil_7551, 'rmtree')
    # Calling rmtree(args, kwargs) (line 135)
    rmtree_call_result_7555 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), rmtree_7552, *[tmpdir_7553], **kwargs_7554)
    
    # SSA join for try-except statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to chdir(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'cwd' (line 137)
    cwd_7558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'cwd', False)
    # Processing the call keyword arguments (line 137)
    kwargs_7559 = {}
    # Getting the type of 'os' (line 137)
    os_7556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'os', False)
    # Obtaining the member 'chdir' of a type (line 137)
    chdir_7557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 4), os_7556, 'chdir')
    # Calling chdir(args, kwargs) (line 137)
    chdir_call_result_7560 = invoke(stypy.reporting.localization.Localization(__file__, 137, 4), chdir_7557, *[cwd_7558], **kwargs_7559)
    
    
    # Call to rmtree(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'tmpdir' (line 138)
    tmpdir_7563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'tmpdir', False)
    # Processing the call keyword arguments (line 138)
    kwargs_7564 = {}
    # Getting the type of 'shutil' (line 138)
    shutil_7561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'shutil', False)
    # Obtaining the member 'rmtree' of a type (line 138)
    rmtree_7562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 4), shutil_7561, 'rmtree')
    # Calling rmtree(args, kwargs) (line 138)
    rmtree_call_result_7565 = invoke(stypy.reporting.localization.Localization(__file__, 138, 4), rmtree_7562, *[tmpdir_7563], **kwargs_7564)
    
    
    # ################# End of 'test_read_write_files(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_write_files' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_7566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7566)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_write_files'
    return stypy_return_type_7566

# Assigning a type to the variable 'test_read_write_files' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'test_read_write_files', test_read_write_files)

@norecursion
def test_read_write_sio(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_write_sio'
    module_type_store = module_type_store.open_function_context('test_read_write_sio', 141, 0, False)
    
    # Passed parameters checking function
    test_read_write_sio.stypy_localization = localization
    test_read_write_sio.stypy_type_of_self = None
    test_read_write_sio.stypy_type_store = module_type_store
    test_read_write_sio.stypy_function_name = 'test_read_write_sio'
    test_read_write_sio.stypy_param_names_list = []
    test_read_write_sio.stypy_varargs_param_name = None
    test_read_write_sio.stypy_kwargs_param_name = None
    test_read_write_sio.stypy_call_defaults = defaults
    test_read_write_sio.stypy_call_varargs = varargs
    test_read_write_sio.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_write_sio', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_write_sio', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_write_sio(...)' code ##################

    
    # Assigning a Call to a Name (line 142):
    
    # Call to BytesIO(...): (line 142)
    # Processing the call keyword arguments (line 142)
    kwargs_7568 = {}
    # Getting the type of 'BytesIO' (line 142)
    BytesIO_7567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 142)
    BytesIO_call_result_7569 = invoke(stypy.reporting.localization.Localization(__file__, 142, 14), BytesIO_7567, *[], **kwargs_7568)
    
    # Assigning a type to the variable 'eg_sio1' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'eg_sio1', BytesIO_call_result_7569)
    
    # Call to make_simple(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'eg_sio1' (line 143)
    eg_sio1_7571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'eg_sio1', False)
    str_7572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 30), 'str', 'w')
    # Processing the call keyword arguments (line 143)
    kwargs_7573 = {}
    # Getting the type of 'make_simple' (line 143)
    make_simple_7570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'make_simple', False)
    # Calling make_simple(args, kwargs) (line 143)
    make_simple_call_result_7574 = invoke(stypy.reporting.localization.Localization(__file__, 143, 9), make_simple_7570, *[eg_sio1_7571, str_7572], **kwargs_7573)
    
    with_7575 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 143, 9), make_simple_call_result_7574, 'with parameter', '__enter__', '__exit__')

    if with_7575:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 143)
        enter___7576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 9), make_simple_call_result_7574, '__enter__')
        with_enter_7577 = invoke(stypy.reporting.localization.Localization(__file__, 143, 9), enter___7576)
        # Assigning a type to the variable 'f1' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 'f1', with_enter_7577)
        
        # Assigning a Call to a Name (line 144):
        
        # Call to getvalue(...): (line 144)
        # Processing the call keyword arguments (line 144)
        kwargs_7580 = {}
        # Getting the type of 'eg_sio1' (line 144)
        eg_sio1_7578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'eg_sio1', False)
        # Obtaining the member 'getvalue' of a type (line 144)
        getvalue_7579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 18), eg_sio1_7578, 'getvalue')
        # Calling getvalue(args, kwargs) (line 144)
        getvalue_call_result_7581 = invoke(stypy.reporting.localization.Localization(__file__, 144, 18), getvalue_7579, *[], **kwargs_7580)
        
        # Assigning a type to the variable 'str_val' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'str_val', getvalue_call_result_7581)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 143)
        exit___7582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 9), make_simple_call_result_7574, '__exit__')
        with_exit_7583 = invoke(stypy.reporting.localization.Localization(__file__, 143, 9), exit___7582, None, None, None)

    
    # Assigning a Call to a Name (line 146):
    
    # Call to BytesIO(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'str_val' (line 146)
    str_val_7585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 22), 'str_val', False)
    # Processing the call keyword arguments (line 146)
    kwargs_7586 = {}
    # Getting the type of 'BytesIO' (line 146)
    BytesIO_7584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 14), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 146)
    BytesIO_call_result_7587 = invoke(stypy.reporting.localization.Localization(__file__, 146, 14), BytesIO_7584, *[str_val_7585], **kwargs_7586)
    
    # Assigning a type to the variable 'eg_sio2' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'eg_sio2', BytesIO_call_result_7587)
    
    # Call to netcdf_file(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'eg_sio2' (line 147)
    eg_sio2_7589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'eg_sio2', False)
    # Processing the call keyword arguments (line 147)
    kwargs_7590 = {}
    # Getting the type of 'netcdf_file' (line 147)
    netcdf_file_7588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 147)
    netcdf_file_call_result_7591 = invoke(stypy.reporting.localization.Localization(__file__, 147, 9), netcdf_file_7588, *[eg_sio2_7589], **kwargs_7590)
    
    with_7592 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 147, 9), netcdf_file_call_result_7591, 'with parameter', '__enter__', '__exit__')

    if with_7592:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 147)
        enter___7593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 9), netcdf_file_call_result_7591, '__enter__')
        with_enter_7594 = invoke(stypy.reporting.localization.Localization(__file__, 147, 9), enter___7593)
        # Assigning a type to the variable 'f2' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 9), 'f2', with_enter_7594)
        
        # Call to check_simple(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'f2' (line 148)
        f2_7596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'f2', False)
        # Processing the call keyword arguments (line 148)
        kwargs_7597 = {}
        # Getting the type of 'check_simple' (line 148)
        check_simple_7595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'check_simple', False)
        # Calling check_simple(args, kwargs) (line 148)
        check_simple_call_result_7598 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), check_simple_7595, *[f2_7596], **kwargs_7597)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 147)
        exit___7599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 9), netcdf_file_call_result_7591, '__exit__')
        with_exit_7600 = invoke(stypy.reporting.localization.Localization(__file__, 147, 9), exit___7599, None, None, None)

    
    # Assigning a Call to a Name (line 151):
    
    # Call to BytesIO(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'str_val' (line 151)
    str_val_7602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'str_val', False)
    # Processing the call keyword arguments (line 151)
    kwargs_7603 = {}
    # Getting the type of 'BytesIO' (line 151)
    BytesIO_7601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 151)
    BytesIO_call_result_7604 = invoke(stypy.reporting.localization.Localization(__file__, 151, 14), BytesIO_7601, *[str_val_7602], **kwargs_7603)
    
    # Assigning a type to the variable 'eg_sio3' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'eg_sio3', BytesIO_call_result_7604)
    
    # Call to assert_raises(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'ValueError' (line 152)
    ValueError_7606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'ValueError', False)
    # Getting the type of 'netcdf_file' (line 152)
    netcdf_file_7607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'netcdf_file', False)
    # Getting the type of 'eg_sio3' (line 152)
    eg_sio3_7608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 43), 'eg_sio3', False)
    str_7609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 52), 'str', 'r')
    # Getting the type of 'True' (line 152)
    True_7610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 57), 'True', False)
    # Processing the call keyword arguments (line 152)
    kwargs_7611 = {}
    # Getting the type of 'assert_raises' (line 152)
    assert_raises_7605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 152)
    assert_raises_call_result_7612 = invoke(stypy.reporting.localization.Localization(__file__, 152, 4), assert_raises_7605, *[ValueError_7606, netcdf_file_7607, eg_sio3_7608, str_7609, True_7610], **kwargs_7611)
    
    
    # Assigning a Call to a Name (line 154):
    
    # Call to BytesIO(...): (line 154)
    # Processing the call keyword arguments (line 154)
    kwargs_7614 = {}
    # Getting the type of 'BytesIO' (line 154)
    BytesIO_7613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 154)
    BytesIO_call_result_7615 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), BytesIO_7613, *[], **kwargs_7614)
    
    # Assigning a type to the variable 'eg_sio_64' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'eg_sio_64', BytesIO_call_result_7615)
    
    # Call to make_simple(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'eg_sio_64' (line 155)
    eg_sio_64_7617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'eg_sio_64', False)
    str_7618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 32), 'str', 'w')
    # Processing the call keyword arguments (line 155)
    int_7619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 45), 'int')
    keyword_7620 = int_7619
    kwargs_7621 = {'version': keyword_7620}
    # Getting the type of 'make_simple' (line 155)
    make_simple_7616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 9), 'make_simple', False)
    # Calling make_simple(args, kwargs) (line 155)
    make_simple_call_result_7622 = invoke(stypy.reporting.localization.Localization(__file__, 155, 9), make_simple_7616, *[eg_sio_64_7617, str_7618], **kwargs_7621)
    
    with_7623 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 155, 9), make_simple_call_result_7622, 'with parameter', '__enter__', '__exit__')

    if with_7623:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 155)
        enter___7624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 9), make_simple_call_result_7622, '__enter__')
        with_enter_7625 = invoke(stypy.reporting.localization.Localization(__file__, 155, 9), enter___7624)
        # Assigning a type to the variable 'f_64' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 9), 'f_64', with_enter_7625)
        
        # Assigning a Call to a Name (line 156):
        
        # Call to getvalue(...): (line 156)
        # Processing the call keyword arguments (line 156)
        kwargs_7628 = {}
        # Getting the type of 'eg_sio_64' (line 156)
        eg_sio_64_7626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 18), 'eg_sio_64', False)
        # Obtaining the member 'getvalue' of a type (line 156)
        getvalue_7627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 18), eg_sio_64_7626, 'getvalue')
        # Calling getvalue(args, kwargs) (line 156)
        getvalue_call_result_7629 = invoke(stypy.reporting.localization.Localization(__file__, 156, 18), getvalue_7627, *[], **kwargs_7628)
        
        # Assigning a type to the variable 'str_val' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'str_val', getvalue_call_result_7629)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 155)
        exit___7630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 9), make_simple_call_result_7622, '__exit__')
        with_exit_7631 = invoke(stypy.reporting.localization.Localization(__file__, 155, 9), exit___7630, None, None, None)

    
    # Assigning a Call to a Name (line 158):
    
    # Call to BytesIO(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'str_val' (line 158)
    str_val_7633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'str_val', False)
    # Processing the call keyword arguments (line 158)
    kwargs_7634 = {}
    # Getting the type of 'BytesIO' (line 158)
    BytesIO_7632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 158)
    BytesIO_call_result_7635 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), BytesIO_7632, *[str_val_7633], **kwargs_7634)
    
    # Assigning a type to the variable 'eg_sio_64' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'eg_sio_64', BytesIO_call_result_7635)
    
    # Call to netcdf_file(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'eg_sio_64' (line 159)
    eg_sio_64_7637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), 'eg_sio_64', False)
    # Processing the call keyword arguments (line 159)
    kwargs_7638 = {}
    # Getting the type of 'netcdf_file' (line 159)
    netcdf_file_7636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 159)
    netcdf_file_call_result_7639 = invoke(stypy.reporting.localization.Localization(__file__, 159, 9), netcdf_file_7636, *[eg_sio_64_7637], **kwargs_7638)
    
    with_7640 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 159, 9), netcdf_file_call_result_7639, 'with parameter', '__enter__', '__exit__')

    if with_7640:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 159)
        enter___7641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 9), netcdf_file_call_result_7639, '__enter__')
        with_enter_7642 = invoke(stypy.reporting.localization.Localization(__file__, 159, 9), enter___7641)
        # Assigning a type to the variable 'f_64' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 9), 'f_64', with_enter_7642)
        
        # Call to check_simple(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'f_64' (line 160)
        f_64_7644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'f_64', False)
        # Processing the call keyword arguments (line 160)
        kwargs_7645 = {}
        # Getting the type of 'check_simple' (line 160)
        check_simple_7643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'check_simple', False)
        # Calling check_simple(args, kwargs) (line 160)
        check_simple_call_result_7646 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), check_simple_7643, *[f_64_7644], **kwargs_7645)
        
        
        # Call to assert_equal(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'f_64' (line 161)
        f_64_7648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 21), 'f_64', False)
        # Obtaining the member 'version_byte' of a type (line 161)
        version_byte_7649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 21), f_64_7648, 'version_byte')
        int_7650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 40), 'int')
        # Processing the call keyword arguments (line 161)
        kwargs_7651 = {}
        # Getting the type of 'assert_equal' (line 161)
        assert_equal_7647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 161)
        assert_equal_call_result_7652 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), assert_equal_7647, *[version_byte_7649, int_7650], **kwargs_7651)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 159)
        exit___7653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 9), netcdf_file_call_result_7639, '__exit__')
        with_exit_7654 = invoke(stypy.reporting.localization.Localization(__file__, 159, 9), exit___7653, None, None, None)

    
    # Assigning a Call to a Name (line 163):
    
    # Call to BytesIO(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'str_val' (line 163)
    str_val_7656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'str_val', False)
    # Processing the call keyword arguments (line 163)
    kwargs_7657 = {}
    # Getting the type of 'BytesIO' (line 163)
    BytesIO_7655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 163)
    BytesIO_call_result_7658 = invoke(stypy.reporting.localization.Localization(__file__, 163, 16), BytesIO_7655, *[str_val_7656], **kwargs_7657)
    
    # Assigning a type to the variable 'eg_sio_64' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'eg_sio_64', BytesIO_call_result_7658)
    
    # Call to netcdf_file(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'eg_sio_64' (line 164)
    eg_sio_64_7660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'eg_sio_64', False)
    # Processing the call keyword arguments (line 164)
    int_7661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 40), 'int')
    keyword_7662 = int_7661
    kwargs_7663 = {'version': keyword_7662}
    # Getting the type of 'netcdf_file' (line 164)
    netcdf_file_7659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 164)
    netcdf_file_call_result_7664 = invoke(stypy.reporting.localization.Localization(__file__, 164, 9), netcdf_file_7659, *[eg_sio_64_7660], **kwargs_7663)
    
    with_7665 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 164, 9), netcdf_file_call_result_7664, 'with parameter', '__enter__', '__exit__')

    if with_7665:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 164)
        enter___7666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 9), netcdf_file_call_result_7664, '__enter__')
        with_enter_7667 = invoke(stypy.reporting.localization.Localization(__file__, 164, 9), enter___7666)
        # Assigning a type to the variable 'f_64' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 9), 'f_64', with_enter_7667)
        
        # Call to check_simple(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'f_64' (line 165)
        f_64_7669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'f_64', False)
        # Processing the call keyword arguments (line 165)
        kwargs_7670 = {}
        # Getting the type of 'check_simple' (line 165)
        check_simple_7668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'check_simple', False)
        # Calling check_simple(args, kwargs) (line 165)
        check_simple_call_result_7671 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), check_simple_7668, *[f_64_7669], **kwargs_7670)
        
        
        # Call to assert_equal(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'f_64' (line 166)
        f_64_7673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'f_64', False)
        # Obtaining the member 'version_byte' of a type (line 166)
        version_byte_7674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 21), f_64_7673, 'version_byte')
        int_7675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 40), 'int')
        # Processing the call keyword arguments (line 166)
        kwargs_7676 = {}
        # Getting the type of 'assert_equal' (line 166)
        assert_equal_7672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 166)
        assert_equal_call_result_7677 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), assert_equal_7672, *[version_byte_7674, int_7675], **kwargs_7676)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 164)
        exit___7678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 9), netcdf_file_call_result_7664, '__exit__')
        with_exit_7679 = invoke(stypy.reporting.localization.Localization(__file__, 164, 9), exit___7678, None, None, None)

    
    # ################# End of 'test_read_write_sio(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_write_sio' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_7680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7680)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_write_sio'
    return stypy_return_type_7680

# Assigning a type to the variable 'test_read_write_sio' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'test_read_write_sio', test_read_write_sio)

@norecursion
def test_read_example_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_example_data'
    module_type_store = module_type_store.open_function_context('test_read_example_data', 169, 0, False)
    
    # Passed parameters checking function
    test_read_example_data.stypy_localization = localization
    test_read_example_data.stypy_type_of_self = None
    test_read_example_data.stypy_type_store = module_type_store
    test_read_example_data.stypy_function_name = 'test_read_example_data'
    test_read_example_data.stypy_param_names_list = []
    test_read_example_data.stypy_varargs_param_name = None
    test_read_example_data.stypy_kwargs_param_name = None
    test_read_example_data.stypy_call_defaults = defaults
    test_read_example_data.stypy_call_varargs = varargs
    test_read_example_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_example_data', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_example_data', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_example_data(...)' code ##################

    
    
    # Call to glob(...): (line 171)
    # Processing the call arguments (line 171)
    
    # Call to pjoin(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'TEST_DATA_PATH' (line 171)
    TEST_DATA_PATH_7683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 28), 'TEST_DATA_PATH', False)
    str_7684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 44), 'str', '*.nc')
    # Processing the call keyword arguments (line 171)
    kwargs_7685 = {}
    # Getting the type of 'pjoin' (line 171)
    pjoin_7682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 22), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 171)
    pjoin_call_result_7686 = invoke(stypy.reporting.localization.Localization(__file__, 171, 22), pjoin_7682, *[TEST_DATA_PATH_7683, str_7684], **kwargs_7685)
    
    # Processing the call keyword arguments (line 171)
    kwargs_7687 = {}
    # Getting the type of 'glob' (line 171)
    glob_7681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'glob', False)
    # Calling glob(args, kwargs) (line 171)
    glob_call_result_7688 = invoke(stypy.reporting.localization.Localization(__file__, 171, 17), glob_7681, *[pjoin_call_result_7686], **kwargs_7687)
    
    # Testing the type of a for loop iterable (line 171)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 171, 4), glob_call_result_7688)
    # Getting the type of the for loop variable (line 171)
    for_loop_var_7689 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 171, 4), glob_call_result_7688)
    # Assigning a type to the variable 'fname' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'fname', for_loop_var_7689)
    # SSA begins for a for statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to netcdf_file(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'fname' (line 172)
    fname_7691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'fname', False)
    str_7692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 32), 'str', 'r')
    # Processing the call keyword arguments (line 172)
    kwargs_7693 = {}
    # Getting the type of 'netcdf_file' (line 172)
    netcdf_file_7690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 13), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 172)
    netcdf_file_call_result_7694 = invoke(stypy.reporting.localization.Localization(__file__, 172, 13), netcdf_file_7690, *[fname_7691, str_7692], **kwargs_7693)
    
    with_7695 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 172, 13), netcdf_file_call_result_7694, 'with parameter', '__enter__', '__exit__')

    if with_7695:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 172)
        enter___7696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 13), netcdf_file_call_result_7694, '__enter__')
        with_enter_7697 = invoke(stypy.reporting.localization.Localization(__file__, 172, 13), enter___7696)
        # Assigning a type to the variable 'f' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 13), 'f', with_enter_7697)
        pass
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 172)
        exit___7698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 13), netcdf_file_call_result_7694, '__exit__')
        with_exit_7699 = invoke(stypy.reporting.localization.Localization(__file__, 172, 13), exit___7698, None, None, None)

    
    # Call to netcdf_file(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'fname' (line 174)
    fname_7701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'fname', False)
    str_7702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 32), 'str', 'r')
    # Processing the call keyword arguments (line 174)
    # Getting the type of 'False' (line 174)
    False_7703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 42), 'False', False)
    keyword_7704 = False_7703
    kwargs_7705 = {'mmap': keyword_7704}
    # Getting the type of 'netcdf_file' (line 174)
    netcdf_file_7700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 174)
    netcdf_file_call_result_7706 = invoke(stypy.reporting.localization.Localization(__file__, 174, 13), netcdf_file_7700, *[fname_7701, str_7702], **kwargs_7705)
    
    with_7707 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 174, 13), netcdf_file_call_result_7706, 'with parameter', '__enter__', '__exit__')

    if with_7707:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 174)
        enter___7708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 13), netcdf_file_call_result_7706, '__enter__')
        with_enter_7709 = invoke(stypy.reporting.localization.Localization(__file__, 174, 13), enter___7708)
        # Assigning a type to the variable 'f' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 'f', with_enter_7709)
        pass
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 174)
        exit___7710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 13), netcdf_file_call_result_7706, '__exit__')
        with_exit_7711 = invoke(stypy.reporting.localization.Localization(__file__, 174, 13), exit___7710, None, None, None)

    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_read_example_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_example_data' in the type store
    # Getting the type of 'stypy_return_type' (line 169)
    stypy_return_type_7712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7712)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_example_data'
    return stypy_return_type_7712

# Assigning a type to the variable 'test_read_example_data' (line 169)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'test_read_example_data', test_read_example_data)

@norecursion
def test_itemset_no_segfault_on_readonly(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_itemset_no_segfault_on_readonly'
    module_type_store = module_type_store.open_function_context('test_itemset_no_segfault_on_readonly', 178, 0, False)
    
    # Passed parameters checking function
    test_itemset_no_segfault_on_readonly.stypy_localization = localization
    test_itemset_no_segfault_on_readonly.stypy_type_of_self = None
    test_itemset_no_segfault_on_readonly.stypy_type_store = module_type_store
    test_itemset_no_segfault_on_readonly.stypy_function_name = 'test_itemset_no_segfault_on_readonly'
    test_itemset_no_segfault_on_readonly.stypy_param_names_list = []
    test_itemset_no_segfault_on_readonly.stypy_varargs_param_name = None
    test_itemset_no_segfault_on_readonly.stypy_kwargs_param_name = None
    test_itemset_no_segfault_on_readonly.stypy_call_defaults = defaults
    test_itemset_no_segfault_on_readonly.stypy_call_varargs = varargs
    test_itemset_no_segfault_on_readonly.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_itemset_no_segfault_on_readonly', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_itemset_no_segfault_on_readonly', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_itemset_no_segfault_on_readonly(...)' code ##################

    
    # Assigning a Call to a Name (line 182):
    
    # Call to pjoin(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'TEST_DATA_PATH' (line 182)
    TEST_DATA_PATH_7714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 21), 'TEST_DATA_PATH', False)
    str_7715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 37), 'str', 'example_1.nc')
    # Processing the call keyword arguments (line 182)
    kwargs_7716 = {}
    # Getting the type of 'pjoin' (line 182)
    pjoin_7713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 182)
    pjoin_call_result_7717 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), pjoin_7713, *[TEST_DATA_PATH_7714, str_7715], **kwargs_7716)
    
    # Assigning a type to the variable 'filename' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'filename', pjoin_call_result_7717)
    
    # Call to suppress_warnings(...): (line 183)
    # Processing the call keyword arguments (line 183)
    kwargs_7719 = {}
    # Getting the type of 'suppress_warnings' (line 183)
    suppress_warnings_7718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 183)
    suppress_warnings_call_result_7720 = invoke(stypy.reporting.localization.Localization(__file__, 183, 9), suppress_warnings_7718, *[], **kwargs_7719)
    
    with_7721 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 183, 9), suppress_warnings_call_result_7720, 'with parameter', '__enter__', '__exit__')

    if with_7721:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 183)
        enter___7722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 9), suppress_warnings_call_result_7720, '__enter__')
        with_enter_7723 = invoke(stypy.reporting.localization.Localization(__file__, 183, 9), enter___7722)
        # Assigning a type to the variable 'sup' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 9), 'sup', with_enter_7723)
        
        # Call to filter(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'RuntimeWarning' (line 184)
        RuntimeWarning_7726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'RuntimeWarning', False)
        str_7727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 19), 'str', 'Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or arrays referring to its data still exist')
        # Processing the call keyword arguments (line 184)
        kwargs_7728 = {}
        # Getting the type of 'sup' (line 184)
        sup_7724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 184)
        filter_7725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), sup_7724, 'filter')
        # Calling filter(args, kwargs) (line 184)
        filter_call_result_7729 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), filter_7725, *[RuntimeWarning_7726, str_7727], **kwargs_7728)
        
        
        # Call to netcdf_file(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'filename' (line 186)
        filename_7731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'filename', False)
        str_7732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 35), 'str', 'r')
        # Processing the call keyword arguments (line 186)
        kwargs_7733 = {}
        # Getting the type of 'netcdf_file' (line 186)
        netcdf_file_7730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 186)
        netcdf_file_call_result_7734 = invoke(stypy.reporting.localization.Localization(__file__, 186, 13), netcdf_file_7730, *[filename_7731, str_7732], **kwargs_7733)
        
        with_7735 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 186, 13), netcdf_file_call_result_7734, 'with parameter', '__enter__', '__exit__')

        if with_7735:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 186)
            enter___7736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 13), netcdf_file_call_result_7734, '__enter__')
            with_enter_7737 = invoke(stypy.reporting.localization.Localization(__file__, 186, 13), enter___7736)
            # Assigning a type to the variable 'f' (line 186)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'f', with_enter_7737)
            
            # Assigning a Subscript to a Name (line 187):
            
            # Obtaining the type of the subscript
            str_7738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 35), 'str', 'time')
            # Getting the type of 'f' (line 187)
            f_7739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'f')
            # Obtaining the member 'variables' of a type (line 187)
            variables_7740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 23), f_7739, 'variables')
            # Obtaining the member '__getitem__' of a type (line 187)
            getitem___7741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 23), variables_7740, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 187)
            subscript_call_result_7742 = invoke(stypy.reporting.localization.Localization(__file__, 187, 23), getitem___7741, str_7738)
            
            # Assigning a type to the variable 'time_var' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'time_var', subscript_call_result_7742)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 186)
            exit___7743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 13), netcdf_file_call_result_7734, '__exit__')
            with_exit_7744 = invoke(stypy.reporting.localization.Localization(__file__, 186, 13), exit___7743, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 183)
        exit___7745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 9), suppress_warnings_call_result_7720, '__exit__')
        with_exit_7746 = invoke(stypy.reporting.localization.Localization(__file__, 183, 9), exit___7745, None, None, None)

    
    # Call to assert_raises(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'RuntimeError' (line 190)
    RuntimeError_7748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 18), 'RuntimeError', False)
    # Getting the type of 'time_var' (line 190)
    time_var_7749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 32), 'time_var', False)
    # Obtaining the member 'assignValue' of a type (line 190)
    assignValue_7750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 32), time_var_7749, 'assignValue')
    int_7751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 54), 'int')
    # Processing the call keyword arguments (line 190)
    kwargs_7752 = {}
    # Getting the type of 'assert_raises' (line 190)
    assert_raises_7747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 190)
    assert_raises_call_result_7753 = invoke(stypy.reporting.localization.Localization(__file__, 190, 4), assert_raises_7747, *[RuntimeError_7748, assignValue_7750, int_7751], **kwargs_7752)
    
    
    # ################# End of 'test_itemset_no_segfault_on_readonly(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_itemset_no_segfault_on_readonly' in the type store
    # Getting the type of 'stypy_return_type' (line 178)
    stypy_return_type_7754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7754)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_itemset_no_segfault_on_readonly'
    return stypy_return_type_7754

# Assigning a type to the variable 'test_itemset_no_segfault_on_readonly' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'test_itemset_no_segfault_on_readonly', test_itemset_no_segfault_on_readonly)

@norecursion
def test_write_invalid_dtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_write_invalid_dtype'
    module_type_store = module_type_store.open_function_context('test_write_invalid_dtype', 193, 0, False)
    
    # Passed parameters checking function
    test_write_invalid_dtype.stypy_localization = localization
    test_write_invalid_dtype.stypy_type_of_self = None
    test_write_invalid_dtype.stypy_type_store = module_type_store
    test_write_invalid_dtype.stypy_function_name = 'test_write_invalid_dtype'
    test_write_invalid_dtype.stypy_param_names_list = []
    test_write_invalid_dtype.stypy_varargs_param_name = None
    test_write_invalid_dtype.stypy_kwargs_param_name = None
    test_write_invalid_dtype.stypy_call_defaults = defaults
    test_write_invalid_dtype.stypy_call_varargs = varargs
    test_write_invalid_dtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_write_invalid_dtype', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_write_invalid_dtype', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_write_invalid_dtype(...)' code ##################

    
    # Assigning a List to a Name (line 194):
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_7755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    # Adding element type (line 194)
    str_7756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 14), 'str', 'int64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 13), list_7755, str_7756)
    # Adding element type (line 194)
    str_7757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 23), 'str', 'uint64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 13), list_7755, str_7757)
    
    # Assigning a type to the variable 'dtypes' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'dtypes', list_7755)
    
    
    
    # Call to dtype(...): (line 195)
    # Processing the call arguments (line 195)
    str_7760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 16), 'str', 'int')
    # Processing the call keyword arguments (line 195)
    kwargs_7761 = {}
    # Getting the type of 'np' (line 195)
    np_7758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 7), 'np', False)
    # Obtaining the member 'dtype' of a type (line 195)
    dtype_7759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 7), np_7758, 'dtype')
    # Calling dtype(args, kwargs) (line 195)
    dtype_call_result_7762 = invoke(stypy.reporting.localization.Localization(__file__, 195, 7), dtype_7759, *[str_7760], **kwargs_7761)
    
    # Obtaining the member 'itemsize' of a type (line 195)
    itemsize_7763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 7), dtype_call_result_7762, 'itemsize')
    int_7764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 35), 'int')
    # Applying the binary operator '==' (line 195)
    result_eq_7765 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 7), '==', itemsize_7763, int_7764)
    
    # Testing the type of an if condition (line 195)
    if_condition_7766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 4), result_eq_7765)
    # Assigning a type to the variable 'if_condition_7766' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'if_condition_7766', if_condition_7766)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 196)
    # Processing the call arguments (line 196)
    str_7769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 22), 'str', 'int')
    # Processing the call keyword arguments (line 196)
    kwargs_7770 = {}
    # Getting the type of 'dtypes' (line 196)
    dtypes_7767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'dtypes', False)
    # Obtaining the member 'append' of a type (line 196)
    append_7768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), dtypes_7767, 'append')
    # Calling append(args, kwargs) (line 196)
    append_call_result_7771 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), append_7768, *[str_7769], **kwargs_7770)
    
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to dtype(...): (line 197)
    # Processing the call arguments (line 197)
    str_7774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 16), 'str', 'uint')
    # Processing the call keyword arguments (line 197)
    kwargs_7775 = {}
    # Getting the type of 'np' (line 197)
    np_7772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 7), 'np', False)
    # Obtaining the member 'dtype' of a type (line 197)
    dtype_7773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 7), np_7772, 'dtype')
    # Calling dtype(args, kwargs) (line 197)
    dtype_call_result_7776 = invoke(stypy.reporting.localization.Localization(__file__, 197, 7), dtype_7773, *[str_7774], **kwargs_7775)
    
    # Obtaining the member 'itemsize' of a type (line 197)
    itemsize_7777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 7), dtype_call_result_7776, 'itemsize')
    int_7778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 36), 'int')
    # Applying the binary operator '==' (line 197)
    result_eq_7779 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 7), '==', itemsize_7777, int_7778)
    
    # Testing the type of an if condition (line 197)
    if_condition_7780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 4), result_eq_7779)
    # Assigning a type to the variable 'if_condition_7780' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'if_condition_7780', if_condition_7780)
    # SSA begins for if statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 198)
    # Processing the call arguments (line 198)
    str_7783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 22), 'str', 'uint')
    # Processing the call keyword arguments (line 198)
    kwargs_7784 = {}
    # Getting the type of 'dtypes' (line 198)
    dtypes_7781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'dtypes', False)
    # Obtaining the member 'append' of a type (line 198)
    append_7782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), dtypes_7781, 'append')
    # Calling append(args, kwargs) (line 198)
    append_call_result_7785 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), append_7782, *[str_7783], **kwargs_7784)
    
    # SSA join for if statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to netcdf_file(...): (line 200)
    # Processing the call arguments (line 200)
    
    # Call to BytesIO(...): (line 200)
    # Processing the call keyword arguments (line 200)
    kwargs_7788 = {}
    # Getting the type of 'BytesIO' (line 200)
    BytesIO_7787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 200)
    BytesIO_call_result_7789 = invoke(stypy.reporting.localization.Localization(__file__, 200, 21), BytesIO_7787, *[], **kwargs_7788)
    
    str_7790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 32), 'str', 'w')
    # Processing the call keyword arguments (line 200)
    kwargs_7791 = {}
    # Getting the type of 'netcdf_file' (line 200)
    netcdf_file_7786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 200)
    netcdf_file_call_result_7792 = invoke(stypy.reporting.localization.Localization(__file__, 200, 9), netcdf_file_7786, *[BytesIO_call_result_7789, str_7790], **kwargs_7791)
    
    with_7793 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 200, 9), netcdf_file_call_result_7792, 'with parameter', '__enter__', '__exit__')

    if with_7793:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 200)
        enter___7794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 9), netcdf_file_call_result_7792, '__enter__')
        with_enter_7795 = invoke(stypy.reporting.localization.Localization(__file__, 200, 9), enter___7794)
        # Assigning a type to the variable 'f' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 9), 'f', with_enter_7795)
        
        # Call to createDimension(...): (line 201)
        # Processing the call arguments (line 201)
        str_7798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 26), 'str', 'time')
        # Getting the type of 'N_EG_ELS' (line 201)
        N_EG_ELS_7799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 34), 'N_EG_ELS', False)
        # Processing the call keyword arguments (line 201)
        kwargs_7800 = {}
        # Getting the type of 'f' (line 201)
        f_7796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'f', False)
        # Obtaining the member 'createDimension' of a type (line 201)
        createDimension_7797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), f_7796, 'createDimension')
        # Calling createDimension(args, kwargs) (line 201)
        createDimension_call_result_7801 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), createDimension_7797, *[str_7798, N_EG_ELS_7799], **kwargs_7800)
        
        
        # Getting the type of 'dtypes' (line 202)
        dtypes_7802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'dtypes')
        # Testing the type of a for loop iterable (line 202)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 202, 8), dtypes_7802)
        # Getting the type of the for loop variable (line 202)
        for_loop_var_7803 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 202, 8), dtypes_7802)
        # Assigning a type to the variable 'dt' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'dt', for_loop_var_7803)
        # SSA begins for a for statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_raises(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'ValueError' (line 203)
        ValueError_7805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 26), 'ValueError', False)
        # Getting the type of 'f' (line 203)
        f_7806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 38), 'f', False)
        # Obtaining the member 'createVariable' of a type (line 203)
        createVariable_7807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 38), f_7806, 'createVariable')
        str_7808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 56), 'str', 'time')
        # Getting the type of 'dt' (line 203)
        dt_7809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 64), 'dt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 203)
        tuple_7810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 69), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 203)
        # Adding element type (line 203)
        str_7811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 69), 'str', 'time')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 69), tuple_7810, str_7811)
        
        # Processing the call keyword arguments (line 203)
        kwargs_7812 = {}
        # Getting the type of 'assert_raises' (line 203)
        assert_raises_7804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 203)
        assert_raises_call_result_7813 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), assert_raises_7804, *[ValueError_7805, createVariable_7807, str_7808, dt_7809, tuple_7810], **kwargs_7812)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 200)
        exit___7814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 9), netcdf_file_call_result_7792, '__exit__')
        with_exit_7815 = invoke(stypy.reporting.localization.Localization(__file__, 200, 9), exit___7814, None, None, None)

    
    # ################# End of 'test_write_invalid_dtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_write_invalid_dtype' in the type store
    # Getting the type of 'stypy_return_type' (line 193)
    stypy_return_type_7816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7816)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_write_invalid_dtype'
    return stypy_return_type_7816

# Assigning a type to the variable 'test_write_invalid_dtype' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'test_write_invalid_dtype', test_write_invalid_dtype)

@norecursion
def test_flush_rewind(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_flush_rewind'
    module_type_store = module_type_store.open_function_context('test_flush_rewind', 206, 0, False)
    
    # Passed parameters checking function
    test_flush_rewind.stypy_localization = localization
    test_flush_rewind.stypy_type_of_self = None
    test_flush_rewind.stypy_type_store = module_type_store
    test_flush_rewind.stypy_function_name = 'test_flush_rewind'
    test_flush_rewind.stypy_param_names_list = []
    test_flush_rewind.stypy_varargs_param_name = None
    test_flush_rewind.stypy_kwargs_param_name = None
    test_flush_rewind.stypy_call_defaults = defaults
    test_flush_rewind.stypy_call_varargs = varargs
    test_flush_rewind.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_flush_rewind', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_flush_rewind', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_flush_rewind(...)' code ##################

    
    # Assigning a Call to a Name (line 207):
    
    # Call to BytesIO(...): (line 207)
    # Processing the call keyword arguments (line 207)
    kwargs_7818 = {}
    # Getting the type of 'BytesIO' (line 207)
    BytesIO_7817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 207)
    BytesIO_call_result_7819 = invoke(stypy.reporting.localization.Localization(__file__, 207, 13), BytesIO_7817, *[], **kwargs_7818)
    
    # Assigning a type to the variable 'stream' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stream', BytesIO_call_result_7819)
    
    # Call to make_simple(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'stream' (line 208)
    stream_7821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 21), 'stream', False)
    # Processing the call keyword arguments (line 208)
    str_7822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 34), 'str', 'w')
    keyword_7823 = str_7822
    kwargs_7824 = {'mode': keyword_7823}
    # Getting the type of 'make_simple' (line 208)
    make_simple_7820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 9), 'make_simple', False)
    # Calling make_simple(args, kwargs) (line 208)
    make_simple_call_result_7825 = invoke(stypy.reporting.localization.Localization(__file__, 208, 9), make_simple_7820, *[stream_7821], **kwargs_7824)
    
    with_7826 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 208, 9), make_simple_call_result_7825, 'with parameter', '__enter__', '__exit__')

    if with_7826:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 208)
        enter___7827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 9), make_simple_call_result_7825, '__enter__')
        with_enter_7828 = invoke(stypy.reporting.localization.Localization(__file__, 208, 9), enter___7827)
        # Assigning a type to the variable 'f' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 9), 'f', with_enter_7828)
        
        # Assigning a Call to a Name (line 209):
        
        # Call to createDimension(...): (line 209)
        # Processing the call arguments (line 209)
        str_7831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 30), 'str', 'x')
        int_7832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 34), 'int')
        # Processing the call keyword arguments (line 209)
        kwargs_7833 = {}
        # Getting the type of 'f' (line 209)
        f_7829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'f', False)
        # Obtaining the member 'createDimension' of a type (line 209)
        createDimension_7830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), f_7829, 'createDimension')
        # Calling createDimension(args, kwargs) (line 209)
        createDimension_call_result_7834 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), createDimension_7830, *[str_7831, int_7832], **kwargs_7833)
        
        # Assigning a type to the variable 'x' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'x', createDimension_call_result_7834)
        
        # Assigning a Call to a Name (line 210):
        
        # Call to createVariable(...): (line 210)
        # Processing the call arguments (line 210)
        str_7837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 29), 'str', 'v')
        str_7838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 34), 'str', 'i2')
        
        # Obtaining an instance of the builtin type 'list' (line 210)
        list_7839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 210)
        # Adding element type (line 210)
        str_7840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 41), 'str', 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 40), list_7839, str_7840)
        
        # Processing the call keyword arguments (line 210)
        kwargs_7841 = {}
        # Getting the type of 'f' (line 210)
        f_7835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'f', False)
        # Obtaining the member 'createVariable' of a type (line 210)
        createVariable_7836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), f_7835, 'createVariable')
        # Calling createVariable(args, kwargs) (line 210)
        createVariable_call_result_7842 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), createVariable_7836, *[str_7837, str_7838, list_7839], **kwargs_7841)
        
        # Assigning a type to the variable 'v' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'v', createVariable_call_result_7842)
        
        # Assigning a Num to a Subscript (line 211):
        int_7843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 15), 'int')
        # Getting the type of 'v' (line 211)
        v_7844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'v')
        slice_7845 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 211, 8), None, None, None)
        # Storing an element on a container (line 211)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 8), v_7844, (slice_7845, int_7843))
        
        # Call to flush(...): (line 212)
        # Processing the call keyword arguments (line 212)
        kwargs_7848 = {}
        # Getting the type of 'f' (line 212)
        f_7846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'f', False)
        # Obtaining the member 'flush' of a type (line 212)
        flush_7847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), f_7846, 'flush')
        # Calling flush(args, kwargs) (line 212)
        flush_call_result_7849 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), flush_7847, *[], **kwargs_7848)
        
        
        # Assigning a Call to a Name (line 213):
        
        # Call to len(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to getvalue(...): (line 213)
        # Processing the call keyword arguments (line 213)
        kwargs_7853 = {}
        # Getting the type of 'stream' (line 213)
        stream_7851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'stream', False)
        # Obtaining the member 'getvalue' of a type (line 213)
        getvalue_7852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 25), stream_7851, 'getvalue')
        # Calling getvalue(args, kwargs) (line 213)
        getvalue_call_result_7854 = invoke(stypy.reporting.localization.Localization(__file__, 213, 25), getvalue_7852, *[], **kwargs_7853)
        
        # Processing the call keyword arguments (line 213)
        kwargs_7855 = {}
        # Getting the type of 'len' (line 213)
        len_7850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 21), 'len', False)
        # Calling len(args, kwargs) (line 213)
        len_call_result_7856 = invoke(stypy.reporting.localization.Localization(__file__, 213, 21), len_7850, *[getvalue_call_result_7854], **kwargs_7855)
        
        # Assigning a type to the variable 'len_single' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'len_single', len_call_result_7856)
        
        # Call to flush(...): (line 214)
        # Processing the call keyword arguments (line 214)
        kwargs_7859 = {}
        # Getting the type of 'f' (line 214)
        f_7857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'f', False)
        # Obtaining the member 'flush' of a type (line 214)
        flush_7858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), f_7857, 'flush')
        # Calling flush(args, kwargs) (line 214)
        flush_call_result_7860 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), flush_7858, *[], **kwargs_7859)
        
        
        # Assigning a Call to a Name (line 215):
        
        # Call to len(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Call to getvalue(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_7864 = {}
        # Getting the type of 'stream' (line 215)
        stream_7862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 25), 'stream', False)
        # Obtaining the member 'getvalue' of a type (line 215)
        getvalue_7863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 25), stream_7862, 'getvalue')
        # Calling getvalue(args, kwargs) (line 215)
        getvalue_call_result_7865 = invoke(stypy.reporting.localization.Localization(__file__, 215, 25), getvalue_7863, *[], **kwargs_7864)
        
        # Processing the call keyword arguments (line 215)
        kwargs_7866 = {}
        # Getting the type of 'len' (line 215)
        len_7861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'len', False)
        # Calling len(args, kwargs) (line 215)
        len_call_result_7867 = invoke(stypy.reporting.localization.Localization(__file__, 215, 21), len_7861, *[getvalue_call_result_7865], **kwargs_7866)
        
        # Assigning a type to the variable 'len_double' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'len_double', len_call_result_7867)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 208)
        exit___7868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 9), make_simple_call_result_7825, '__exit__')
        with_exit_7869 = invoke(stypy.reporting.localization.Localization(__file__, 208, 9), exit___7868, None, None, None)

    
    # Call to assert_(...): (line 217)
    # Processing the call arguments (line 217)
    
    # Getting the type of 'len_single' (line 217)
    len_single_7871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'len_single', False)
    # Getting the type of 'len_double' (line 217)
    len_double_7872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'len_double', False)
    # Applying the binary operator '==' (line 217)
    result_eq_7873 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 12), '==', len_single_7871, len_double_7872)
    
    # Processing the call keyword arguments (line 217)
    kwargs_7874 = {}
    # Getting the type of 'assert_' (line 217)
    assert__7870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 217)
    assert__call_result_7875 = invoke(stypy.reporting.localization.Localization(__file__, 217, 4), assert__7870, *[result_eq_7873], **kwargs_7874)
    
    
    # ################# End of 'test_flush_rewind(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_flush_rewind' in the type store
    # Getting the type of 'stypy_return_type' (line 206)
    stypy_return_type_7876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7876)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_flush_rewind'
    return stypy_return_type_7876

# Assigning a type to the variable 'test_flush_rewind' (line 206)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'test_flush_rewind', test_flush_rewind)

@norecursion
def test_dtype_specifiers(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dtype_specifiers'
    module_type_store = module_type_store.open_function_context('test_dtype_specifiers', 220, 0, False)
    
    # Passed parameters checking function
    test_dtype_specifiers.stypy_localization = localization
    test_dtype_specifiers.stypy_type_of_self = None
    test_dtype_specifiers.stypy_type_store = module_type_store
    test_dtype_specifiers.stypy_function_name = 'test_dtype_specifiers'
    test_dtype_specifiers.stypy_param_names_list = []
    test_dtype_specifiers.stypy_varargs_param_name = None
    test_dtype_specifiers.stypy_kwargs_param_name = None
    test_dtype_specifiers.stypy_call_defaults = defaults
    test_dtype_specifiers.stypy_call_varargs = varargs
    test_dtype_specifiers.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dtype_specifiers', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dtype_specifiers', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dtype_specifiers(...)' code ##################

    
    # Call to make_simple(...): (line 224)
    # Processing the call arguments (line 224)
    
    # Call to BytesIO(...): (line 224)
    # Processing the call keyword arguments (line 224)
    kwargs_7879 = {}
    # Getting the type of 'BytesIO' (line 224)
    BytesIO_7878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 224)
    BytesIO_call_result_7880 = invoke(stypy.reporting.localization.Localization(__file__, 224, 21), BytesIO_7878, *[], **kwargs_7879)
    
    # Processing the call keyword arguments (line 224)
    str_7881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 37), 'str', 'w')
    keyword_7882 = str_7881
    kwargs_7883 = {'mode': keyword_7882}
    # Getting the type of 'make_simple' (line 224)
    make_simple_7877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 9), 'make_simple', False)
    # Calling make_simple(args, kwargs) (line 224)
    make_simple_call_result_7884 = invoke(stypy.reporting.localization.Localization(__file__, 224, 9), make_simple_7877, *[BytesIO_call_result_7880], **kwargs_7883)
    
    with_7885 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 224, 9), make_simple_call_result_7884, 'with parameter', '__enter__', '__exit__')

    if with_7885:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 224)
        enter___7886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 9), make_simple_call_result_7884, '__enter__')
        with_enter_7887 = invoke(stypy.reporting.localization.Localization(__file__, 224, 9), enter___7886)
        # Assigning a type to the variable 'f' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 9), 'f', with_enter_7887)
        
        # Call to createDimension(...): (line 225)
        # Processing the call arguments (line 225)
        str_7890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 26), 'str', 'x')
        int_7891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 30), 'int')
        # Processing the call keyword arguments (line 225)
        kwargs_7892 = {}
        # Getting the type of 'f' (line 225)
        f_7888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'f', False)
        # Obtaining the member 'createDimension' of a type (line 225)
        createDimension_7889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), f_7888, 'createDimension')
        # Calling createDimension(args, kwargs) (line 225)
        createDimension_call_result_7893 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), createDimension_7889, *[str_7890, int_7891], **kwargs_7892)
        
        
        # Call to createVariable(...): (line 226)
        # Processing the call arguments (line 226)
        str_7896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 25), 'str', 'v1')
        str_7897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 31), 'str', 'i2')
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_7898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        str_7899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 38), 'str', 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 37), list_7898, str_7899)
        
        # Processing the call keyword arguments (line 226)
        kwargs_7900 = {}
        # Getting the type of 'f' (line 226)
        f_7894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'f', False)
        # Obtaining the member 'createVariable' of a type (line 226)
        createVariable_7895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), f_7894, 'createVariable')
        # Calling createVariable(args, kwargs) (line 226)
        createVariable_call_result_7901 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), createVariable_7895, *[str_7896, str_7897, list_7898], **kwargs_7900)
        
        
        # Call to createVariable(...): (line 227)
        # Processing the call arguments (line 227)
        str_7904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 25), 'str', 'v2')
        # Getting the type of 'np' (line 227)
        np_7905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 31), 'np', False)
        # Obtaining the member 'int16' of a type (line 227)
        int16_7906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 31), np_7905, 'int16')
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_7907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        str_7908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 42), 'str', 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 41), list_7907, str_7908)
        
        # Processing the call keyword arguments (line 227)
        kwargs_7909 = {}
        # Getting the type of 'f' (line 227)
        f_7902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'f', False)
        # Obtaining the member 'createVariable' of a type (line 227)
        createVariable_7903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), f_7902, 'createVariable')
        # Calling createVariable(args, kwargs) (line 227)
        createVariable_call_result_7910 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), createVariable_7903, *[str_7904, int16_7906, list_7907], **kwargs_7909)
        
        
        # Call to createVariable(...): (line 228)
        # Processing the call arguments (line 228)
        str_7913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 25), 'str', 'v3')
        
        # Call to dtype(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'np' (line 228)
        np_7916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 40), 'np', False)
        # Obtaining the member 'int16' of a type (line 228)
        int16_7917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 40), np_7916, 'int16')
        # Processing the call keyword arguments (line 228)
        kwargs_7918 = {}
        # Getting the type of 'np' (line 228)
        np_7914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 31), 'np', False)
        # Obtaining the member 'dtype' of a type (line 228)
        dtype_7915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 31), np_7914, 'dtype')
        # Calling dtype(args, kwargs) (line 228)
        dtype_call_result_7919 = invoke(stypy.reporting.localization.Localization(__file__, 228, 31), dtype_7915, *[int16_7917], **kwargs_7918)
        
        
        # Obtaining an instance of the builtin type 'list' (line 228)
        list_7920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 228)
        # Adding element type (line 228)
        str_7921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 52), 'str', 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 51), list_7920, str_7921)
        
        # Processing the call keyword arguments (line 228)
        kwargs_7922 = {}
        # Getting the type of 'f' (line 228)
        f_7911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'f', False)
        # Obtaining the member 'createVariable' of a type (line 228)
        createVariable_7912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), f_7911, 'createVariable')
        # Calling createVariable(args, kwargs) (line 228)
        createVariable_call_result_7923 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), createVariable_7912, *[str_7913, dtype_call_result_7919, list_7920], **kwargs_7922)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 224)
        exit___7924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 9), make_simple_call_result_7884, '__exit__')
        with_exit_7925 = invoke(stypy.reporting.localization.Localization(__file__, 224, 9), exit___7924, None, None, None)

    
    # ################# End of 'test_dtype_specifiers(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dtype_specifiers' in the type store
    # Getting the type of 'stypy_return_type' (line 220)
    stypy_return_type_7926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7926)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dtype_specifiers'
    return stypy_return_type_7926

# Assigning a type to the variable 'test_dtype_specifiers' (line 220)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'test_dtype_specifiers', test_dtype_specifiers)

@norecursion
def test_ticket_1720(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ticket_1720'
    module_type_store = module_type_store.open_function_context('test_ticket_1720', 231, 0, False)
    
    # Passed parameters checking function
    test_ticket_1720.stypy_localization = localization
    test_ticket_1720.stypy_type_of_self = None
    test_ticket_1720.stypy_type_store = module_type_store
    test_ticket_1720.stypy_function_name = 'test_ticket_1720'
    test_ticket_1720.stypy_param_names_list = []
    test_ticket_1720.stypy_varargs_param_name = None
    test_ticket_1720.stypy_kwargs_param_name = None
    test_ticket_1720.stypy_call_defaults = defaults
    test_ticket_1720.stypy_call_varargs = varargs
    test_ticket_1720.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ticket_1720', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ticket_1720', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ticket_1720(...)' code ##################

    
    # Assigning a Call to a Name (line 232):
    
    # Call to BytesIO(...): (line 232)
    # Processing the call keyword arguments (line 232)
    kwargs_7928 = {}
    # Getting the type of 'BytesIO' (line 232)
    BytesIO_7927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 9), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 232)
    BytesIO_call_result_7929 = invoke(stypy.reporting.localization.Localization(__file__, 232, 9), BytesIO_7927, *[], **kwargs_7928)
    
    # Assigning a type to the variable 'io' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'io', BytesIO_call_result_7929)
    
    # Assigning a List to a Name (line 234):
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_7930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    # Adding element type (line 234)
    int_7931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 12), list_7930, int_7931)
    # Adding element type (line 234)
    float_7932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 12), list_7930, float_7932)
    # Adding element type (line 234)
    float_7933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 12), list_7930, float_7933)
    # Adding element type (line 234)
    float_7934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 12), list_7930, float_7934)
    # Adding element type (line 234)
    float_7935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 27), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 12), list_7930, float_7935)
    # Adding element type (line 234)
    float_7936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 12), list_7930, float_7936)
    # Adding element type (line 234)
    float_7937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 12), list_7930, float_7937)
    # Adding element type (line 234)
    float_7938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 12), list_7930, float_7938)
    # Adding element type (line 234)
    float_7939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 12), list_7930, float_7939)
    # Adding element type (line 234)
    float_7940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 47), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 12), list_7930, float_7940)
    
    # Assigning a type to the variable 'items' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'items', list_7930)
    
    # Call to netcdf_file(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'io' (line 236)
    io_7942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 21), 'io', False)
    str_7943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 25), 'str', 'w')
    # Processing the call keyword arguments (line 236)
    kwargs_7944 = {}
    # Getting the type of 'netcdf_file' (line 236)
    netcdf_file_7941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 236)
    netcdf_file_call_result_7945 = invoke(stypy.reporting.localization.Localization(__file__, 236, 9), netcdf_file_7941, *[io_7942, str_7943], **kwargs_7944)
    
    with_7946 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 236, 9), netcdf_file_call_result_7945, 'with parameter', '__enter__', '__exit__')

    if with_7946:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 236)
        enter___7947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 9), netcdf_file_call_result_7945, '__enter__')
        with_enter_7948 = invoke(stypy.reporting.localization.Localization(__file__, 236, 9), enter___7947)
        # Assigning a type to the variable 'f' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 9), 'f', with_enter_7948)
        
        # Assigning a Str to a Attribute (line 237):
        str_7949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 20), 'str', 'Created for a test')
        # Getting the type of 'f' (line 237)
        f_7950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'f')
        # Setting the type of the member 'history' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), f_7950, 'history', str_7949)
        
        # Call to createDimension(...): (line 238)
        # Processing the call arguments (line 238)
        str_7953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 26), 'str', 'float_var')
        int_7954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 39), 'int')
        # Processing the call keyword arguments (line 238)
        kwargs_7955 = {}
        # Getting the type of 'f' (line 238)
        f_7951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'f', False)
        # Obtaining the member 'createDimension' of a type (line 238)
        createDimension_7952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), f_7951, 'createDimension')
        # Calling createDimension(args, kwargs) (line 238)
        createDimension_call_result_7956 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), createDimension_7952, *[str_7953, int_7954], **kwargs_7955)
        
        
        # Assigning a Call to a Name (line 239):
        
        # Call to createVariable(...): (line 239)
        # Processing the call arguments (line 239)
        str_7959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 37), 'str', 'float_var')
        str_7960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 50), 'str', 'f')
        
        # Obtaining an instance of the builtin type 'tuple' (line 239)
        tuple_7961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 239)
        # Adding element type (line 239)
        str_7962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 56), 'str', 'float_var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 56), tuple_7961, str_7962)
        
        # Processing the call keyword arguments (line 239)
        kwargs_7963 = {}
        # Getting the type of 'f' (line 239)
        f_7957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 20), 'f', False)
        # Obtaining the member 'createVariable' of a type (line 239)
        createVariable_7958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 20), f_7957, 'createVariable')
        # Calling createVariable(args, kwargs) (line 239)
        createVariable_call_result_7964 = invoke(stypy.reporting.localization.Localization(__file__, 239, 20), createVariable_7958, *[str_7959, str_7960, tuple_7961], **kwargs_7963)
        
        # Assigning a type to the variable 'float_var' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'float_var', createVariable_call_result_7964)
        
        # Assigning a Name to a Subscript (line 240):
        # Getting the type of 'items' (line 240)
        items_7965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'items')
        # Getting the type of 'float_var' (line 240)
        float_var_7966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'float_var')
        slice_7967 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 240, 8), None, None, None)
        # Storing an element on a container (line 240)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 8), float_var_7966, (slice_7967, items_7965))
        
        # Assigning a Str to a Attribute (line 241):
        str_7968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 26), 'str', 'metres')
        # Getting the type of 'float_var' (line 241)
        float_var_7969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'float_var')
        # Setting the type of the member 'units' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), float_var_7969, 'units', str_7968)
        
        # Call to flush(...): (line 242)
        # Processing the call keyword arguments (line 242)
        kwargs_7972 = {}
        # Getting the type of 'f' (line 242)
        f_7970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'f', False)
        # Obtaining the member 'flush' of a type (line 242)
        flush_7971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), f_7970, 'flush')
        # Calling flush(args, kwargs) (line 242)
        flush_call_result_7973 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), flush_7971, *[], **kwargs_7972)
        
        
        # Assigning a Call to a Name (line 243):
        
        # Call to getvalue(...): (line 243)
        # Processing the call keyword arguments (line 243)
        kwargs_7976 = {}
        # Getting the type of 'io' (line 243)
        io_7974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'io', False)
        # Obtaining the member 'getvalue' of a type (line 243)
        getvalue_7975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 19), io_7974, 'getvalue')
        # Calling getvalue(args, kwargs) (line 243)
        getvalue_call_result_7977 = invoke(stypy.reporting.localization.Localization(__file__, 243, 19), getvalue_7975, *[], **kwargs_7976)
        
        # Assigning a type to the variable 'contents' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'contents', getvalue_call_result_7977)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 236)
        exit___7978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 9), netcdf_file_call_result_7945, '__exit__')
        with_exit_7979 = invoke(stypy.reporting.localization.Localization(__file__, 236, 9), exit___7978, None, None, None)

    
    # Assigning a Call to a Name (line 245):
    
    # Call to BytesIO(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'contents' (line 245)
    contents_7981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'contents', False)
    # Processing the call keyword arguments (line 245)
    kwargs_7982 = {}
    # Getting the type of 'BytesIO' (line 245)
    BytesIO_7980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 9), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 245)
    BytesIO_call_result_7983 = invoke(stypy.reporting.localization.Localization(__file__, 245, 9), BytesIO_7980, *[contents_7981], **kwargs_7982)
    
    # Assigning a type to the variable 'io' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'io', BytesIO_call_result_7983)
    
    # Call to netcdf_file(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'io' (line 246)
    io_7985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'io', False)
    str_7986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 25), 'str', 'r')
    # Processing the call keyword arguments (line 246)
    kwargs_7987 = {}
    # Getting the type of 'netcdf_file' (line 246)
    netcdf_file_7984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 246)
    netcdf_file_call_result_7988 = invoke(stypy.reporting.localization.Localization(__file__, 246, 9), netcdf_file_7984, *[io_7985, str_7986], **kwargs_7987)
    
    with_7989 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 246, 9), netcdf_file_call_result_7988, 'with parameter', '__enter__', '__exit__')

    if with_7989:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 246)
        enter___7990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 9), netcdf_file_call_result_7988, '__enter__')
        with_enter_7991 = invoke(stypy.reporting.localization.Localization(__file__, 246, 9), enter___7990)
        # Assigning a type to the variable 'f' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'f', with_enter_7991)
        
        # Call to assert_equal(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'f' (line 247)
        f_7993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 21), 'f', False)
        # Obtaining the member 'history' of a type (line 247)
        history_7994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 21), f_7993, 'history')
        str_7995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 32), 'str', 'Created for a test')
        # Processing the call keyword arguments (line 247)
        kwargs_7996 = {}
        # Getting the type of 'assert_equal' (line 247)
        assert_equal_7992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 247)
        assert_equal_call_result_7997 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), assert_equal_7992, *[history_7994, str_7995], **kwargs_7996)
        
        
        # Assigning a Subscript to a Name (line 248):
        
        # Obtaining the type of the subscript
        str_7998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 32), 'str', 'float_var')
        # Getting the type of 'f' (line 248)
        f_7999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'f')
        # Obtaining the member 'variables' of a type (line 248)
        variables_8000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 20), f_7999, 'variables')
        # Obtaining the member '__getitem__' of a type (line 248)
        getitem___8001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 20), variables_8000, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 248)
        subscript_call_result_8002 = invoke(stypy.reporting.localization.Localization(__file__, 248, 20), getitem___8001, str_7998)
        
        # Assigning a type to the variable 'float_var' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'float_var', subscript_call_result_8002)
        
        # Call to assert_equal(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'float_var' (line 249)
        float_var_8004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'float_var', False)
        # Obtaining the member 'units' of a type (line 249)
        units_8005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 21), float_var_8004, 'units')
        str_8006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 38), 'str', 'metres')
        # Processing the call keyword arguments (line 249)
        kwargs_8007 = {}
        # Getting the type of 'assert_equal' (line 249)
        assert_equal_8003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 249)
        assert_equal_call_result_8008 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), assert_equal_8003, *[units_8005, str_8006], **kwargs_8007)
        
        
        # Call to assert_equal(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'float_var' (line 250)
        float_var_8010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 21), 'float_var', False)
        # Obtaining the member 'shape' of a type (line 250)
        shape_8011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 21), float_var_8010, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 250)
        tuple_8012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 250)
        # Adding element type (line 250)
        int_8013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 39), tuple_8012, int_8013)
        
        # Processing the call keyword arguments (line 250)
        kwargs_8014 = {}
        # Getting the type of 'assert_equal' (line 250)
        assert_equal_8009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 250)
        assert_equal_call_result_8015 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), assert_equal_8009, *[shape_8011, tuple_8012], **kwargs_8014)
        
        
        # Call to assert_allclose(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Obtaining the type of the subscript
        slice_8017 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 251, 24), None, None, None)
        # Getting the type of 'float_var' (line 251)
        float_var_8018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 'float_var', False)
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___8019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 24), float_var_8018, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_8020 = invoke(stypy.reporting.localization.Localization(__file__, 251, 24), getitem___8019, slice_8017)
        
        # Getting the type of 'items' (line 251)
        items_8021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 38), 'items', False)
        # Processing the call keyword arguments (line 251)
        kwargs_8022 = {}
        # Getting the type of 'assert_allclose' (line 251)
        assert_allclose_8016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 251)
        assert_allclose_call_result_8023 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), assert_allclose_8016, *[subscript_call_result_8020, items_8021], **kwargs_8022)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 246)
        exit___8024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 9), netcdf_file_call_result_7988, '__exit__')
        with_exit_8025 = invoke(stypy.reporting.localization.Localization(__file__, 246, 9), exit___8024, None, None, None)

    
    # ################# End of 'test_ticket_1720(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ticket_1720' in the type store
    # Getting the type of 'stypy_return_type' (line 231)
    stypy_return_type_8026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8026)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ticket_1720'
    return stypy_return_type_8026

# Assigning a type to the variable 'test_ticket_1720' (line 231)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 0), 'test_ticket_1720', test_ticket_1720)

@norecursion
def test_mmaps_segfault(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_mmaps_segfault'
    module_type_store = module_type_store.open_function_context('test_mmaps_segfault', 254, 0, False)
    
    # Passed parameters checking function
    test_mmaps_segfault.stypy_localization = localization
    test_mmaps_segfault.stypy_type_of_self = None
    test_mmaps_segfault.stypy_type_store = module_type_store
    test_mmaps_segfault.stypy_function_name = 'test_mmaps_segfault'
    test_mmaps_segfault.stypy_param_names_list = []
    test_mmaps_segfault.stypy_varargs_param_name = None
    test_mmaps_segfault.stypy_kwargs_param_name = None
    test_mmaps_segfault.stypy_call_defaults = defaults
    test_mmaps_segfault.stypy_call_varargs = varargs
    test_mmaps_segfault.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_mmaps_segfault', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_mmaps_segfault', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_mmaps_segfault(...)' code ##################

    
    # Assigning a Call to a Name (line 255):
    
    # Call to pjoin(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'TEST_DATA_PATH' (line 255)
    TEST_DATA_PATH_8028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 21), 'TEST_DATA_PATH', False)
    str_8029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 37), 'str', 'example_1.nc')
    # Processing the call keyword arguments (line 255)
    kwargs_8030 = {}
    # Getting the type of 'pjoin' (line 255)
    pjoin_8027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 255)
    pjoin_call_result_8031 = invoke(stypy.reporting.localization.Localization(__file__, 255, 15), pjoin_8027, *[TEST_DATA_PATH_8028, str_8029], **kwargs_8030)
    
    # Assigning a type to the variable 'filename' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'filename', pjoin_call_result_8031)
    
    # Call to catch_warnings(...): (line 257)
    # Processing the call keyword arguments (line 257)
    kwargs_8034 = {}
    # Getting the type of 'warnings' (line 257)
    warnings_8032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 9), 'warnings', False)
    # Obtaining the member 'catch_warnings' of a type (line 257)
    catch_warnings_8033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 9), warnings_8032, 'catch_warnings')
    # Calling catch_warnings(args, kwargs) (line 257)
    catch_warnings_call_result_8035 = invoke(stypy.reporting.localization.Localization(__file__, 257, 9), catch_warnings_8033, *[], **kwargs_8034)
    
    with_8036 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 257, 9), catch_warnings_call_result_8035, 'with parameter', '__enter__', '__exit__')

    if with_8036:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 257)
        enter___8037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 9), catch_warnings_call_result_8035, '__enter__')
        with_enter_8038 = invoke(stypy.reporting.localization.Localization(__file__, 257, 9), enter___8037)
        
        # Call to simplefilter(...): (line 258)
        # Processing the call arguments (line 258)
        str_8041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 30), 'str', 'error')
        # Processing the call keyword arguments (line 258)
        kwargs_8042 = {}
        # Getting the type of 'warnings' (line 258)
        warnings_8039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'warnings', False)
        # Obtaining the member 'simplefilter' of a type (line 258)
        simplefilter_8040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), warnings_8039, 'simplefilter')
        # Calling simplefilter(args, kwargs) (line 258)
        simplefilter_call_result_8043 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), simplefilter_8040, *[str_8041], **kwargs_8042)
        
        
        # Call to netcdf_file(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'filename' (line 259)
        filename_8045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 25), 'filename', False)
        # Processing the call keyword arguments (line 259)
        # Getting the type of 'True' (line 259)
        True_8046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 40), 'True', False)
        keyword_8047 = True_8046
        kwargs_8048 = {'mmap': keyword_8047}
        # Getting the type of 'netcdf_file' (line 259)
        netcdf_file_8044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 13), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 259)
        netcdf_file_call_result_8049 = invoke(stypy.reporting.localization.Localization(__file__, 259, 13), netcdf_file_8044, *[filename_8045], **kwargs_8048)
        
        with_8050 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 259, 13), netcdf_file_call_result_8049, 'with parameter', '__enter__', '__exit__')

        if with_8050:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 259)
            enter___8051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 13), netcdf_file_call_result_8049, '__enter__')
            with_enter_8052 = invoke(stypy.reporting.localization.Localization(__file__, 259, 13), enter___8051)
            # Assigning a type to the variable 'f' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 13), 'f', with_enter_8052)
            
            # Assigning a Subscript to a Name (line 260):
            
            # Obtaining the type of the subscript
            slice_8053 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 260, 16), None, None, None)
            
            # Obtaining the type of the subscript
            str_8054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 28), 'str', 'lat')
            # Getting the type of 'f' (line 260)
            f_8055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'f')
            # Obtaining the member 'variables' of a type (line 260)
            variables_8056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 16), f_8055, 'variables')
            # Obtaining the member '__getitem__' of a type (line 260)
            getitem___8057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 16), variables_8056, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 260)
            subscript_call_result_8058 = invoke(stypy.reporting.localization.Localization(__file__, 260, 16), getitem___8057, str_8054)
            
            # Obtaining the member '__getitem__' of a type (line 260)
            getitem___8059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 16), subscript_call_result_8058, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 260)
            subscript_call_result_8060 = invoke(stypy.reporting.localization.Localization(__file__, 260, 16), getitem___8059, slice_8053)
            
            # Assigning a type to the variable 'x' (line 260)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'x', subscript_call_result_8060)
            # Deleting a member
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 262, 12), module_type_store, 'x')
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 259)
            exit___8061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 13), netcdf_file_call_result_8049, '__exit__')
            with_exit_8062 = invoke(stypy.reporting.localization.Localization(__file__, 259, 13), exit___8061, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 257)
        exit___8063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 9), catch_warnings_call_result_8035, '__exit__')
        with_exit_8064 = invoke(stypy.reporting.localization.Localization(__file__, 257, 9), exit___8063, None, None, None)


    @norecursion
    def doit(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'doit'
        module_type_store = module_type_store.open_function_context('doit', 264, 4, False)
        
        # Passed parameters checking function
        doit.stypy_localization = localization
        doit.stypy_type_of_self = None
        doit.stypy_type_store = module_type_store
        doit.stypy_function_name = 'doit'
        doit.stypy_param_names_list = []
        doit.stypy_varargs_param_name = None
        doit.stypy_kwargs_param_name = None
        doit.stypy_call_defaults = defaults
        doit.stypy_call_varargs = varargs
        doit.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'doit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'doit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'doit(...)' code ##################

        
        # Call to netcdf_file(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'filename' (line 265)
        filename_8066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 'filename', False)
        # Processing the call keyword arguments (line 265)
        # Getting the type of 'True' (line 265)
        True_8067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 40), 'True', False)
        keyword_8068 = True_8067
        kwargs_8069 = {'mmap': keyword_8068}
        # Getting the type of 'netcdf_file' (line 265)
        netcdf_file_8065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 265)
        netcdf_file_call_result_8070 = invoke(stypy.reporting.localization.Localization(__file__, 265, 13), netcdf_file_8065, *[filename_8066], **kwargs_8069)
        
        with_8071 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 265, 13), netcdf_file_call_result_8070, 'with parameter', '__enter__', '__exit__')

        if with_8071:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 265)
            enter___8072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 13), netcdf_file_call_result_8070, '__enter__')
            with_enter_8073 = invoke(stypy.reporting.localization.Localization(__file__, 265, 13), enter___8072)
            # Assigning a type to the variable 'f' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'f', with_enter_8073)
            
            # Obtaining the type of the subscript
            slice_8074 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 266, 19), None, None, None)
            
            # Obtaining the type of the subscript
            str_8075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 31), 'str', 'lat')
            # Getting the type of 'f' (line 266)
            f_8076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 19), 'f')
            # Obtaining the member 'variables' of a type (line 266)
            variables_8077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 19), f_8076, 'variables')
            # Obtaining the member '__getitem__' of a type (line 266)
            getitem___8078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 19), variables_8077, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 266)
            subscript_call_result_8079 = invoke(stypy.reporting.localization.Localization(__file__, 266, 19), getitem___8078, str_8075)
            
            # Obtaining the member '__getitem__' of a type (line 266)
            getitem___8080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 19), subscript_call_result_8079, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 266)
            subscript_call_result_8081 = invoke(stypy.reporting.localization.Localization(__file__, 266, 19), getitem___8080, slice_8074)
            
            # Assigning a type to the variable 'stypy_return_type' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'stypy_return_type', subscript_call_result_8081)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 265)
            exit___8082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 13), netcdf_file_call_result_8070, '__exit__')
            with_exit_8083 = invoke(stypy.reporting.localization.Localization(__file__, 265, 13), exit___8082, None, None, None)

        
        # ################# End of 'doit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'doit' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_8084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'doit'
        return stypy_return_type_8084

    # Assigning a type to the variable 'doit' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'doit', doit)
    
    # Call to suppress_warnings(...): (line 269)
    # Processing the call keyword arguments (line 269)
    kwargs_8086 = {}
    # Getting the type of 'suppress_warnings' (line 269)
    suppress_warnings_8085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 269)
    suppress_warnings_call_result_8087 = invoke(stypy.reporting.localization.Localization(__file__, 269, 9), suppress_warnings_8085, *[], **kwargs_8086)
    
    with_8088 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 269, 9), suppress_warnings_call_result_8087, 'with parameter', '__enter__', '__exit__')

    if with_8088:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 269)
        enter___8089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 9), suppress_warnings_call_result_8087, '__enter__')
        with_enter_8090 = invoke(stypy.reporting.localization.Localization(__file__, 269, 9), enter___8089)
        # Assigning a type to the variable 'sup' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 9), 'sup', with_enter_8090)
        
        # Call to filter(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'RuntimeWarning' (line 270)
        RuntimeWarning_8093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'RuntimeWarning', False)
        str_8094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 19), 'str', 'Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or arrays referring to its data still exist')
        # Processing the call keyword arguments (line 270)
        kwargs_8095 = {}
        # Getting the type of 'sup' (line 270)
        sup_8091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 270)
        filter_8092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), sup_8091, 'filter')
        # Calling filter(args, kwargs) (line 270)
        filter_call_result_8096 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), filter_8092, *[RuntimeWarning_8093, str_8094], **kwargs_8095)
        
        
        # Assigning a Call to a Name (line 272):
        
        # Call to doit(...): (line 272)
        # Processing the call keyword arguments (line 272)
        kwargs_8098 = {}
        # Getting the type of 'doit' (line 272)
        doit_8097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'doit', False)
        # Calling doit(args, kwargs) (line 272)
        doit_call_result_8099 = invoke(stypy.reporting.localization.Localization(__file__, 272, 12), doit_8097, *[], **kwargs_8098)
        
        # Assigning a type to the variable 'x' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'x', doit_call_result_8099)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 269)
        exit___8100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 9), suppress_warnings_call_result_8087, '__exit__')
        with_exit_8101 = invoke(stypy.reporting.localization.Localization(__file__, 269, 9), exit___8100, None, None, None)

    
    # Call to sum(...): (line 273)
    # Processing the call keyword arguments (line 273)
    kwargs_8104 = {}
    # Getting the type of 'x' (line 273)
    x_8102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'x', False)
    # Obtaining the member 'sum' of a type (line 273)
    sum_8103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 4), x_8102, 'sum')
    # Calling sum(args, kwargs) (line 273)
    sum_call_result_8105 = invoke(stypy.reporting.localization.Localization(__file__, 273, 4), sum_8103, *[], **kwargs_8104)
    
    
    # ################# End of 'test_mmaps_segfault(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_mmaps_segfault' in the type store
    # Getting the type of 'stypy_return_type' (line 254)
    stypy_return_type_8106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8106)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_mmaps_segfault'
    return stypy_return_type_8106

# Assigning a type to the variable 'test_mmaps_segfault' (line 254)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'test_mmaps_segfault', test_mmaps_segfault)

@norecursion
def test_zero_dimensional_var(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_zero_dimensional_var'
    module_type_store = module_type_store.open_function_context('test_zero_dimensional_var', 276, 0, False)
    
    # Passed parameters checking function
    test_zero_dimensional_var.stypy_localization = localization
    test_zero_dimensional_var.stypy_type_of_self = None
    test_zero_dimensional_var.stypy_type_store = module_type_store
    test_zero_dimensional_var.stypy_function_name = 'test_zero_dimensional_var'
    test_zero_dimensional_var.stypy_param_names_list = []
    test_zero_dimensional_var.stypy_varargs_param_name = None
    test_zero_dimensional_var.stypy_kwargs_param_name = None
    test_zero_dimensional_var.stypy_call_defaults = defaults
    test_zero_dimensional_var.stypy_call_varargs = varargs
    test_zero_dimensional_var.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_zero_dimensional_var', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_zero_dimensional_var', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_zero_dimensional_var(...)' code ##################

    
    # Assigning a Call to a Name (line 277):
    
    # Call to BytesIO(...): (line 277)
    # Processing the call keyword arguments (line 277)
    kwargs_8108 = {}
    # Getting the type of 'BytesIO' (line 277)
    BytesIO_8107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 9), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 277)
    BytesIO_call_result_8109 = invoke(stypy.reporting.localization.Localization(__file__, 277, 9), BytesIO_8107, *[], **kwargs_8108)
    
    # Assigning a type to the variable 'io' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'io', BytesIO_call_result_8109)
    
    # Call to make_simple(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'io' (line 278)
    io_8111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'io', False)
    str_8112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 25), 'str', 'w')
    # Processing the call keyword arguments (line 278)
    kwargs_8113 = {}
    # Getting the type of 'make_simple' (line 278)
    make_simple_8110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 9), 'make_simple', False)
    # Calling make_simple(args, kwargs) (line 278)
    make_simple_call_result_8114 = invoke(stypy.reporting.localization.Localization(__file__, 278, 9), make_simple_8110, *[io_8111, str_8112], **kwargs_8113)
    
    with_8115 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 278, 9), make_simple_call_result_8114, 'with parameter', '__enter__', '__exit__')

    if with_8115:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 278)
        enter___8116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 9), make_simple_call_result_8114, '__enter__')
        with_enter_8117 = invoke(stypy.reporting.localization.Localization(__file__, 278, 9), enter___8116)
        # Assigning a type to the variable 'f' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 9), 'f', with_enter_8117)
        
        # Assigning a Call to a Name (line 279):
        
        # Call to createVariable(...): (line 279)
        # Processing the call arguments (line 279)
        str_8120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 29), 'str', 'zerodim')
        str_8121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 40), 'str', 'i2')
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_8122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        
        # Processing the call keyword arguments (line 279)
        kwargs_8123 = {}
        # Getting the type of 'f' (line 279)
        f_8118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'f', False)
        # Obtaining the member 'createVariable' of a type (line 279)
        createVariable_8119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), f_8118, 'createVariable')
        # Calling createVariable(args, kwargs) (line 279)
        createVariable_call_result_8124 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), createVariable_8119, *[str_8120, str_8121, list_8122], **kwargs_8123)
        
        # Assigning a type to the variable 'v' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'v', createVariable_call_result_8124)
        # Evaluating assert statement condition
        
        # Getting the type of 'v' (line 282)
        v_8125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'v')
        # Obtaining the member 'isrec' of a type (line 282)
        isrec_8126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 15), v_8125, 'isrec')
        # Getting the type of 'False' (line 282)
        False_8127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 26), 'False')
        # Applying the binary operator 'is' (line 282)
        result_is__8128 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 15), 'is', isrec_8126, False_8127)
        
        
        # Call to flush(...): (line 283)
        # Processing the call keyword arguments (line 283)
        kwargs_8131 = {}
        # Getting the type of 'f' (line 283)
        f_8129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'f', False)
        # Obtaining the member 'flush' of a type (line 283)
        flush_8130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), f_8129, 'flush')
        # Calling flush(args, kwargs) (line 283)
        flush_call_result_8132 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), flush_8130, *[], **kwargs_8131)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 278)
        exit___8133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 9), make_simple_call_result_8114, '__exit__')
        with_exit_8134 = invoke(stypy.reporting.localization.Localization(__file__, 278, 9), exit___8133, None, None, None)

    
    # ################# End of 'test_zero_dimensional_var(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_zero_dimensional_var' in the type store
    # Getting the type of 'stypy_return_type' (line 276)
    stypy_return_type_8135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8135)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_zero_dimensional_var'
    return stypy_return_type_8135

# Assigning a type to the variable 'test_zero_dimensional_var' (line 276)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 0), 'test_zero_dimensional_var', test_zero_dimensional_var)

@norecursion
def test_byte_gatts(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_byte_gatts'
    module_type_store = module_type_store.open_function_context('test_byte_gatts', 286, 0, False)
    
    # Passed parameters checking function
    test_byte_gatts.stypy_localization = localization
    test_byte_gatts.stypy_type_of_self = None
    test_byte_gatts.stypy_type_store = module_type_store
    test_byte_gatts.stypy_function_name = 'test_byte_gatts'
    test_byte_gatts.stypy_param_names_list = []
    test_byte_gatts.stypy_varargs_param_name = None
    test_byte_gatts.stypy_kwargs_param_name = None
    test_byte_gatts.stypy_call_defaults = defaults
    test_byte_gatts.stypy_call_varargs = varargs
    test_byte_gatts.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_byte_gatts', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_byte_gatts', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_byte_gatts(...)' code ##################

    
    # Call to in_tempdir(...): (line 289)
    # Processing the call keyword arguments (line 289)
    kwargs_8137 = {}
    # Getting the type of 'in_tempdir' (line 289)
    in_tempdir_8136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 9), 'in_tempdir', False)
    # Calling in_tempdir(args, kwargs) (line 289)
    in_tempdir_call_result_8138 = invoke(stypy.reporting.localization.Localization(__file__, 289, 9), in_tempdir_8136, *[], **kwargs_8137)
    
    with_8139 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 289, 9), in_tempdir_call_result_8138, 'with parameter', '__enter__', '__exit__')

    if with_8139:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 289)
        enter___8140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 9), in_tempdir_call_result_8138, '__enter__')
        with_enter_8141 = invoke(stypy.reporting.localization.Localization(__file__, 289, 9), enter___8140)
        
        # Assigning a Str to a Name (line 290):
        str_8142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 19), 'str', 'g_byte_atts.nc')
        # Assigning a type to the variable 'filename' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'filename', str_8142)
        
        # Assigning a Call to a Name (line 291):
        
        # Call to netcdf_file(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'filename' (line 291)
        filename_8144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'filename', False)
        str_8145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 34), 'str', 'w')
        # Processing the call keyword arguments (line 291)
        kwargs_8146 = {}
        # Getting the type of 'netcdf_file' (line 291)
        netcdf_file_8143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 291)
        netcdf_file_call_result_8147 = invoke(stypy.reporting.localization.Localization(__file__, 291, 12), netcdf_file_8143, *[filename_8144, str_8145], **kwargs_8146)
        
        # Assigning a type to the variable 'f' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'f', netcdf_file_call_result_8147)
        
        # Assigning a Str to a Subscript (line 292):
        str_8148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 32), 'str', 'grail')
        # Getting the type of 'f' (line 292)
        f_8149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'f')
        # Obtaining the member '_attributes' of a type (line 292)
        _attributes_8150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), f_8149, '_attributes')
        str_8151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 22), 'str', 'holy')
        # Storing an element on a container (line 292)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 8), _attributes_8150, (str_8151, str_8148))
        
        # Assigning a Str to a Subscript (line 293):
        str_8152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 33), 'str', 'floats')
        # Getting the type of 'f' (line 293)
        f_8153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'f')
        # Obtaining the member '_attributes' of a type (line 293)
        _attributes_8154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), f_8153, '_attributes')
        str_8155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 22), 'str', 'witch')
        # Storing an element on a container (line 293)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 8), _attributes_8154, (str_8155, str_8152))
        
        # Call to close(...): (line 294)
        # Processing the call keyword arguments (line 294)
        kwargs_8158 = {}
        # Getting the type of 'f' (line 294)
        f_8156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'f', False)
        # Obtaining the member 'close' of a type (line 294)
        close_8157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), f_8156, 'close')
        # Calling close(args, kwargs) (line 294)
        close_call_result_8159 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), close_8157, *[], **kwargs_8158)
        
        
        # Assigning a Call to a Name (line 295):
        
        # Call to netcdf_file(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'filename' (line 295)
        filename_8161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'filename', False)
        str_8162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 34), 'str', 'r')
        # Processing the call keyword arguments (line 295)
        kwargs_8163 = {}
        # Getting the type of 'netcdf_file' (line 295)
        netcdf_file_8160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 295)
        netcdf_file_call_result_8164 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), netcdf_file_8160, *[filename_8161, str_8162], **kwargs_8163)
        
        # Assigning a type to the variable 'f' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'f', netcdf_file_call_result_8164)
        
        # Call to assert_equal(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Obtaining the type of the subscript
        str_8166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 35), 'str', 'holy')
        # Getting the type of 'f' (line 296)
        f_8167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 21), 'f', False)
        # Obtaining the member '_attributes' of a type (line 296)
        _attributes_8168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 21), f_8167, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 296)
        getitem___8169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 21), _attributes_8168, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 296)
        subscript_call_result_8170 = invoke(stypy.reporting.localization.Localization(__file__, 296, 21), getitem___8169, str_8166)
        
        str_8171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 44), 'str', 'grail')
        # Processing the call keyword arguments (line 296)
        kwargs_8172 = {}
        # Getting the type of 'assert_equal' (line 296)
        assert_equal_8165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 296)
        assert_equal_call_result_8173 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), assert_equal_8165, *[subscript_call_result_8170, str_8171], **kwargs_8172)
        
        
        # Call to assert_equal(...): (line 297)
        # Processing the call arguments (line 297)
        
        # Obtaining the type of the subscript
        str_8175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 35), 'str', 'witch')
        # Getting the type of 'f' (line 297)
        f_8176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 21), 'f', False)
        # Obtaining the member '_attributes' of a type (line 297)
        _attributes_8177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 21), f_8176, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___8178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 21), _attributes_8177, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_8179 = invoke(stypy.reporting.localization.Localization(__file__, 297, 21), getitem___8178, str_8175)
        
        str_8180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 45), 'str', 'floats')
        # Processing the call keyword arguments (line 297)
        kwargs_8181 = {}
        # Getting the type of 'assert_equal' (line 297)
        assert_equal_8174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 297)
        assert_equal_call_result_8182 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), assert_equal_8174, *[subscript_call_result_8179, str_8180], **kwargs_8181)
        
        
        # Call to close(...): (line 298)
        # Processing the call keyword arguments (line 298)
        kwargs_8185 = {}
        # Getting the type of 'f' (line 298)
        f_8183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'f', False)
        # Obtaining the member 'close' of a type (line 298)
        close_8184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), f_8183, 'close')
        # Calling close(args, kwargs) (line 298)
        close_call_result_8186 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), close_8184, *[], **kwargs_8185)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 289)
        exit___8187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 9), in_tempdir_call_result_8138, '__exit__')
        with_exit_8188 = invoke(stypy.reporting.localization.Localization(__file__, 289, 9), exit___8187, None, None, None)

    
    # ################# End of 'test_byte_gatts(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_byte_gatts' in the type store
    # Getting the type of 'stypy_return_type' (line 286)
    stypy_return_type_8189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8189)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_byte_gatts'
    return stypy_return_type_8189

# Assigning a type to the variable 'test_byte_gatts' (line 286)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'test_byte_gatts', test_byte_gatts)

@norecursion
def test_open_append(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_open_append'
    module_type_store = module_type_store.open_function_context('test_open_append', 301, 0, False)
    
    # Passed parameters checking function
    test_open_append.stypy_localization = localization
    test_open_append.stypy_type_of_self = None
    test_open_append.stypy_type_store = module_type_store
    test_open_append.stypy_function_name = 'test_open_append'
    test_open_append.stypy_param_names_list = []
    test_open_append.stypy_varargs_param_name = None
    test_open_append.stypy_kwargs_param_name = None
    test_open_append.stypy_call_defaults = defaults
    test_open_append.stypy_call_varargs = varargs
    test_open_append.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_open_append', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_open_append', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_open_append(...)' code ##################

    
    # Call to in_tempdir(...): (line 303)
    # Processing the call keyword arguments (line 303)
    kwargs_8191 = {}
    # Getting the type of 'in_tempdir' (line 303)
    in_tempdir_8190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 9), 'in_tempdir', False)
    # Calling in_tempdir(args, kwargs) (line 303)
    in_tempdir_call_result_8192 = invoke(stypy.reporting.localization.Localization(__file__, 303, 9), in_tempdir_8190, *[], **kwargs_8191)
    
    with_8193 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 303, 9), in_tempdir_call_result_8192, 'with parameter', '__enter__', '__exit__')

    if with_8193:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 303)
        enter___8194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 9), in_tempdir_call_result_8192, '__enter__')
        with_enter_8195 = invoke(stypy.reporting.localization.Localization(__file__, 303, 9), enter___8194)
        
        # Assigning a Str to a Name (line 304):
        str_8196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 19), 'str', 'append_dat.nc')
        # Assigning a type to the variable 'filename' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'filename', str_8196)
        
        # Assigning a Call to a Name (line 305):
        
        # Call to netcdf_file(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'filename' (line 305)
        filename_8198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 24), 'filename', False)
        str_8199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 34), 'str', 'w')
        # Processing the call keyword arguments (line 305)
        kwargs_8200 = {}
        # Getting the type of 'netcdf_file' (line 305)
        netcdf_file_8197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 305)
        netcdf_file_call_result_8201 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), netcdf_file_8197, *[filename_8198, str_8199], **kwargs_8200)
        
        # Assigning a type to the variable 'f' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'f', netcdf_file_call_result_8201)
        
        # Assigning a Str to a Subscript (line 306):
        str_8202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 34), 'str', 'was here')
        # Getting the type of 'f' (line 306)
        f_8203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'f')
        # Obtaining the member '_attributes' of a type (line 306)
        _attributes_8204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), f_8203, '_attributes')
        str_8205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 22), 'str', 'Kilroy')
        # Storing an element on a container (line 306)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 8), _attributes_8204, (str_8205, str_8202))
        
        # Call to close(...): (line 307)
        # Processing the call keyword arguments (line 307)
        kwargs_8208 = {}
        # Getting the type of 'f' (line 307)
        f_8206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'f', False)
        # Obtaining the member 'close' of a type (line 307)
        close_8207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), f_8206, 'close')
        # Calling close(args, kwargs) (line 307)
        close_call_result_8209 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), close_8207, *[], **kwargs_8208)
        
        
        # Assigning a Call to a Name (line 310):
        
        # Call to netcdf_file(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'filename' (line 310)
        filename_8211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 24), 'filename', False)
        str_8212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 34), 'str', 'a')
        # Processing the call keyword arguments (line 310)
        kwargs_8213 = {}
        # Getting the type of 'netcdf_file' (line 310)
        netcdf_file_8210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 310)
        netcdf_file_call_result_8214 = invoke(stypy.reporting.localization.Localization(__file__, 310, 12), netcdf_file_8210, *[filename_8211, str_8212], **kwargs_8213)
        
        # Assigning a type to the variable 'f' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'f', netcdf_file_call_result_8214)
        
        # Call to assert_equal(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining the type of the subscript
        str_8216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 35), 'str', 'Kilroy')
        # Getting the type of 'f' (line 311)
        f_8217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 21), 'f', False)
        # Obtaining the member '_attributes' of a type (line 311)
        _attributes_8218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 21), f_8217, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___8219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 21), _attributes_8218, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_8220 = invoke(stypy.reporting.localization.Localization(__file__, 311, 21), getitem___8219, str_8216)
        
        str_8221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 46), 'str', 'was here')
        # Processing the call keyword arguments (line 311)
        kwargs_8222 = {}
        # Getting the type of 'assert_equal' (line 311)
        assert_equal_8215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 311)
        assert_equal_call_result_8223 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), assert_equal_8215, *[subscript_call_result_8220, str_8221], **kwargs_8222)
        
        
        # Assigning a Str to a Subscript (line 312):
        str_8224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 35), 'str', 'Zoot')
        # Getting the type of 'f' (line 312)
        f_8225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'f')
        # Obtaining the member '_attributes' of a type (line 312)
        _attributes_8226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), f_8225, '_attributes')
        str_8227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 22), 'str', 'naughty')
        # Storing an element on a container (line 312)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 8), _attributes_8226, (str_8227, str_8224))
        
        # Call to close(...): (line 313)
        # Processing the call keyword arguments (line 313)
        kwargs_8230 = {}
        # Getting the type of 'f' (line 313)
        f_8228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'f', False)
        # Obtaining the member 'close' of a type (line 313)
        close_8229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), f_8228, 'close')
        # Calling close(args, kwargs) (line 313)
        close_call_result_8231 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), close_8229, *[], **kwargs_8230)
        
        
        # Assigning a Call to a Name (line 316):
        
        # Call to netcdf_file(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'filename' (line 316)
        filename_8233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 24), 'filename', False)
        str_8234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 34), 'str', 'r')
        # Processing the call keyword arguments (line 316)
        kwargs_8235 = {}
        # Getting the type of 'netcdf_file' (line 316)
        netcdf_file_8232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 316)
        netcdf_file_call_result_8236 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), netcdf_file_8232, *[filename_8233, str_8234], **kwargs_8235)
        
        # Assigning a type to the variable 'f' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'f', netcdf_file_call_result_8236)
        
        # Call to assert_equal(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Obtaining the type of the subscript
        str_8238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 35), 'str', 'Kilroy')
        # Getting the type of 'f' (line 317)
        f_8239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 21), 'f', False)
        # Obtaining the member '_attributes' of a type (line 317)
        _attributes_8240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 21), f_8239, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 317)
        getitem___8241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 21), _attributes_8240, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 317)
        subscript_call_result_8242 = invoke(stypy.reporting.localization.Localization(__file__, 317, 21), getitem___8241, str_8238)
        
        str_8243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 46), 'str', 'was here')
        # Processing the call keyword arguments (line 317)
        kwargs_8244 = {}
        # Getting the type of 'assert_equal' (line 317)
        assert_equal_8237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 317)
        assert_equal_call_result_8245 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), assert_equal_8237, *[subscript_call_result_8242, str_8243], **kwargs_8244)
        
        
        # Call to assert_equal(...): (line 318)
        # Processing the call arguments (line 318)
        
        # Obtaining the type of the subscript
        str_8247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 35), 'str', 'naughty')
        # Getting the type of 'f' (line 318)
        f_8248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'f', False)
        # Obtaining the member '_attributes' of a type (line 318)
        _attributes_8249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 21), f_8248, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___8250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 21), _attributes_8249, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_8251 = invoke(stypy.reporting.localization.Localization(__file__, 318, 21), getitem___8250, str_8247)
        
        str_8252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 47), 'str', 'Zoot')
        # Processing the call keyword arguments (line 318)
        kwargs_8253 = {}
        # Getting the type of 'assert_equal' (line 318)
        assert_equal_8246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 318)
        assert_equal_call_result_8254 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), assert_equal_8246, *[subscript_call_result_8251, str_8252], **kwargs_8253)
        
        
        # Call to close(...): (line 319)
        # Processing the call keyword arguments (line 319)
        kwargs_8257 = {}
        # Getting the type of 'f' (line 319)
        f_8255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'f', False)
        # Obtaining the member 'close' of a type (line 319)
        close_8256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), f_8255, 'close')
        # Calling close(args, kwargs) (line 319)
        close_call_result_8258 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), close_8256, *[], **kwargs_8257)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 303)
        exit___8259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 9), in_tempdir_call_result_8192, '__exit__')
        with_exit_8260 = invoke(stypy.reporting.localization.Localization(__file__, 303, 9), exit___8259, None, None, None)

    
    # ################# End of 'test_open_append(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_open_append' in the type store
    # Getting the type of 'stypy_return_type' (line 301)
    stypy_return_type_8261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8261)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_open_append'
    return stypy_return_type_8261

# Assigning a type to the variable 'test_open_append' (line 301)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'test_open_append', test_open_append)

@norecursion
def test_append_recordDimension(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_append_recordDimension'
    module_type_store = module_type_store.open_function_context('test_append_recordDimension', 322, 0, False)
    
    # Passed parameters checking function
    test_append_recordDimension.stypy_localization = localization
    test_append_recordDimension.stypy_type_of_self = None
    test_append_recordDimension.stypy_type_store = module_type_store
    test_append_recordDimension.stypy_function_name = 'test_append_recordDimension'
    test_append_recordDimension.stypy_param_names_list = []
    test_append_recordDimension.stypy_varargs_param_name = None
    test_append_recordDimension.stypy_kwargs_param_name = None
    test_append_recordDimension.stypy_call_defaults = defaults
    test_append_recordDimension.stypy_call_varargs = varargs
    test_append_recordDimension.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_append_recordDimension', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_append_recordDimension', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_append_recordDimension(...)' code ##################

    
    # Assigning a Num to a Name (line 323):
    int_8262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 15), 'int')
    # Assigning a type to the variable 'dataSize' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'dataSize', int_8262)
    
    # Call to in_tempdir(...): (line 325)
    # Processing the call keyword arguments (line 325)
    kwargs_8264 = {}
    # Getting the type of 'in_tempdir' (line 325)
    in_tempdir_8263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 9), 'in_tempdir', False)
    # Calling in_tempdir(args, kwargs) (line 325)
    in_tempdir_call_result_8265 = invoke(stypy.reporting.localization.Localization(__file__, 325, 9), in_tempdir_8263, *[], **kwargs_8264)
    
    with_8266 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 325, 9), in_tempdir_call_result_8265, 'with parameter', '__enter__', '__exit__')

    if with_8266:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 325)
        enter___8267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 9), in_tempdir_call_result_8265, '__enter__')
        with_enter_8268 = invoke(stypy.reporting.localization.Localization(__file__, 325, 9), enter___8267)
        
        # Call to netcdf_file(...): (line 327)
        # Processing the call arguments (line 327)
        str_8270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 25), 'str', 'withRecordDimension.nc')
        str_8271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 51), 'str', 'w')
        # Processing the call keyword arguments (line 327)
        kwargs_8272 = {}
        # Getting the type of 'netcdf_file' (line 327)
        netcdf_file_8269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 13), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 327)
        netcdf_file_call_result_8273 = invoke(stypy.reporting.localization.Localization(__file__, 327, 13), netcdf_file_8269, *[str_8270, str_8271], **kwargs_8272)
        
        with_8274 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 327, 13), netcdf_file_call_result_8273, 'with parameter', '__enter__', '__exit__')

        if with_8274:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 327)
            enter___8275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 13), netcdf_file_call_result_8273, '__enter__')
            with_enter_8276 = invoke(stypy.reporting.localization.Localization(__file__, 327, 13), enter___8275)
            # Assigning a type to the variable 'f' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 13), 'f', with_enter_8276)
            
            # Call to createDimension(...): (line 328)
            # Processing the call arguments (line 328)
            str_8279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 30), 'str', 'time')
            # Getting the type of 'None' (line 328)
            None_8280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 38), 'None', False)
            # Processing the call keyword arguments (line 328)
            kwargs_8281 = {}
            # Getting the type of 'f' (line 328)
            f_8277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'f', False)
            # Obtaining the member 'createDimension' of a type (line 328)
            createDimension_8278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 12), f_8277, 'createDimension')
            # Calling createDimension(args, kwargs) (line 328)
            createDimension_call_result_8282 = invoke(stypy.reporting.localization.Localization(__file__, 328, 12), createDimension_8278, *[str_8279, None_8280], **kwargs_8281)
            
            
            # Call to createVariable(...): (line 329)
            # Processing the call arguments (line 329)
            str_8285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 29), 'str', 'time')
            str_8286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 37), 'str', 'd')
            
            # Obtaining an instance of the builtin type 'tuple' (line 329)
            tuple_8287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 43), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 329)
            # Adding element type (line 329)
            str_8288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 43), 'str', 'time')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 43), tuple_8287, str_8288)
            
            # Processing the call keyword arguments (line 329)
            kwargs_8289 = {}
            # Getting the type of 'f' (line 329)
            f_8283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'f', False)
            # Obtaining the member 'createVariable' of a type (line 329)
            createVariable_8284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 12), f_8283, 'createVariable')
            # Calling createVariable(args, kwargs) (line 329)
            createVariable_call_result_8290 = invoke(stypy.reporting.localization.Localization(__file__, 329, 12), createVariable_8284, *[str_8285, str_8286, tuple_8287], **kwargs_8289)
            
            
            # Call to createDimension(...): (line 330)
            # Processing the call arguments (line 330)
            str_8293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 30), 'str', 'x')
            # Getting the type of 'dataSize' (line 330)
            dataSize_8294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 35), 'dataSize', False)
            # Processing the call keyword arguments (line 330)
            kwargs_8295 = {}
            # Getting the type of 'f' (line 330)
            f_8291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'f', False)
            # Obtaining the member 'createDimension' of a type (line 330)
            createDimension_8292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), f_8291, 'createDimension')
            # Calling createDimension(args, kwargs) (line 330)
            createDimension_call_result_8296 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), createDimension_8292, *[str_8293, dataSize_8294], **kwargs_8295)
            
            
            # Assigning a Call to a Name (line 331):
            
            # Call to createVariable(...): (line 331)
            # Processing the call arguments (line 331)
            str_8299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 33), 'str', 'x')
            str_8300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 38), 'str', 'd')
            
            # Obtaining an instance of the builtin type 'tuple' (line 331)
            tuple_8301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 44), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 331)
            # Adding element type (line 331)
            str_8302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 44), 'str', 'x')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 44), tuple_8301, str_8302)
            
            # Processing the call keyword arguments (line 331)
            kwargs_8303 = {}
            # Getting the type of 'f' (line 331)
            f_8297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'f', False)
            # Obtaining the member 'createVariable' of a type (line 331)
            createVariable_8298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 16), f_8297, 'createVariable')
            # Calling createVariable(args, kwargs) (line 331)
            createVariable_call_result_8304 = invoke(stypy.reporting.localization.Localization(__file__, 331, 16), createVariable_8298, *[str_8299, str_8300, tuple_8301], **kwargs_8303)
            
            # Assigning a type to the variable 'x' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'x', createVariable_call_result_8304)
            
            # Assigning a Call to a Subscript (line 332):
            
            # Call to array(...): (line 332)
            # Processing the call arguments (line 332)
            
            # Call to range(...): (line 332)
            # Processing the call arguments (line 332)
            # Getting the type of 'dataSize' (line 332)
            dataSize_8308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'dataSize', False)
            # Processing the call keyword arguments (line 332)
            kwargs_8309 = {}
            # Getting the type of 'range' (line 332)
            range_8307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 28), 'range', False)
            # Calling range(args, kwargs) (line 332)
            range_call_result_8310 = invoke(stypy.reporting.localization.Localization(__file__, 332, 28), range_8307, *[dataSize_8308], **kwargs_8309)
            
            # Processing the call keyword arguments (line 332)
            kwargs_8311 = {}
            # Getting the type of 'np' (line 332)
            np_8305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'np', False)
            # Obtaining the member 'array' of a type (line 332)
            array_8306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), np_8305, 'array')
            # Calling array(args, kwargs) (line 332)
            array_call_result_8312 = invoke(stypy.reporting.localization.Localization(__file__, 332, 19), array_8306, *[range_call_result_8310], **kwargs_8311)
            
            # Getting the type of 'x' (line 332)
            x_8313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'x')
            slice_8314 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 332, 12), None, None, None)
            # Storing an element on a container (line 332)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 12), x_8313, (slice_8314, array_call_result_8312))
            
            # Call to createDimension(...): (line 333)
            # Processing the call arguments (line 333)
            str_8317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 30), 'str', 'y')
            # Getting the type of 'dataSize' (line 333)
            dataSize_8318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 35), 'dataSize', False)
            # Processing the call keyword arguments (line 333)
            kwargs_8319 = {}
            # Getting the type of 'f' (line 333)
            f_8315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'f', False)
            # Obtaining the member 'createDimension' of a type (line 333)
            createDimension_8316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), f_8315, 'createDimension')
            # Calling createDimension(args, kwargs) (line 333)
            createDimension_call_result_8320 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), createDimension_8316, *[str_8317, dataSize_8318], **kwargs_8319)
            
            
            # Assigning a Call to a Name (line 334):
            
            # Call to createVariable(...): (line 334)
            # Processing the call arguments (line 334)
            str_8323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 33), 'str', 'y')
            str_8324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 38), 'str', 'd')
            
            # Obtaining an instance of the builtin type 'tuple' (line 334)
            tuple_8325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 44), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 334)
            # Adding element type (line 334)
            str_8326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 44), 'str', 'y')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 44), tuple_8325, str_8326)
            
            # Processing the call keyword arguments (line 334)
            kwargs_8327 = {}
            # Getting the type of 'f' (line 334)
            f_8321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'f', False)
            # Obtaining the member 'createVariable' of a type (line 334)
            createVariable_8322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 16), f_8321, 'createVariable')
            # Calling createVariable(args, kwargs) (line 334)
            createVariable_call_result_8328 = invoke(stypy.reporting.localization.Localization(__file__, 334, 16), createVariable_8322, *[str_8323, str_8324, tuple_8325], **kwargs_8327)
            
            # Assigning a type to the variable 'y' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'y', createVariable_call_result_8328)
            
            # Assigning a Call to a Subscript (line 335):
            
            # Call to array(...): (line 335)
            # Processing the call arguments (line 335)
            
            # Call to range(...): (line 335)
            # Processing the call arguments (line 335)
            # Getting the type of 'dataSize' (line 335)
            dataSize_8332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 34), 'dataSize', False)
            # Processing the call keyword arguments (line 335)
            kwargs_8333 = {}
            # Getting the type of 'range' (line 335)
            range_8331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 28), 'range', False)
            # Calling range(args, kwargs) (line 335)
            range_call_result_8334 = invoke(stypy.reporting.localization.Localization(__file__, 335, 28), range_8331, *[dataSize_8332], **kwargs_8333)
            
            # Processing the call keyword arguments (line 335)
            kwargs_8335 = {}
            # Getting the type of 'np' (line 335)
            np_8329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'np', False)
            # Obtaining the member 'array' of a type (line 335)
            array_8330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 19), np_8329, 'array')
            # Calling array(args, kwargs) (line 335)
            array_call_result_8336 = invoke(stypy.reporting.localization.Localization(__file__, 335, 19), array_8330, *[range_call_result_8334], **kwargs_8335)
            
            # Getting the type of 'y' (line 335)
            y_8337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'y')
            slice_8338 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 335, 12), None, None, None)
            # Storing an element on a container (line 335)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 12), y_8337, (slice_8338, array_call_result_8336))
            
            # Call to createVariable(...): (line 336)
            # Processing the call arguments (line 336)
            str_8341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 29), 'str', 'testData')
            str_8342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 41), 'str', 'i')
            
            # Obtaining an instance of the builtin type 'tuple' (line 336)
            tuple_8343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 47), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 336)
            # Adding element type (line 336)
            str_8344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 47), 'str', 'time')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 47), tuple_8343, str_8344)
            # Adding element type (line 336)
            str_8345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 55), 'str', 'x')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 47), tuple_8343, str_8345)
            # Adding element type (line 336)
            str_8346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 60), 'str', 'y')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 47), tuple_8343, str_8346)
            
            # Processing the call keyword arguments (line 336)
            kwargs_8347 = {}
            # Getting the type of 'f' (line 336)
            f_8339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'f', False)
            # Obtaining the member 'createVariable' of a type (line 336)
            createVariable_8340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), f_8339, 'createVariable')
            # Calling createVariable(args, kwargs) (line 336)
            createVariable_call_result_8348 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), createVariable_8340, *[str_8341, str_8342, tuple_8343], **kwargs_8347)
            
            
            # Call to flush(...): (line 337)
            # Processing the call keyword arguments (line 337)
            kwargs_8351 = {}
            # Getting the type of 'f' (line 337)
            f_8349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'f', False)
            # Obtaining the member 'flush' of a type (line 337)
            flush_8350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), f_8349, 'flush')
            # Calling flush(args, kwargs) (line 337)
            flush_call_result_8352 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), flush_8350, *[], **kwargs_8351)
            
            
            # Call to close(...): (line 338)
            # Processing the call keyword arguments (line 338)
            kwargs_8355 = {}
            # Getting the type of 'f' (line 338)
            f_8353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'f', False)
            # Obtaining the member 'close' of a type (line 338)
            close_8354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), f_8353, 'close')
            # Calling close(args, kwargs) (line 338)
            close_call_result_8356 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), close_8354, *[], **kwargs_8355)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 327)
            exit___8357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 13), netcdf_file_call_result_8273, '__exit__')
            with_exit_8358 = invoke(stypy.reporting.localization.Localization(__file__, 327, 13), exit___8357, None, None, None)

        
        
        # Call to range(...): (line 340)
        # Processing the call arguments (line 340)
        int_8360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 23), 'int')
        # Processing the call keyword arguments (line 340)
        kwargs_8361 = {}
        # Getting the type of 'range' (line 340)
        range_8359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 17), 'range', False)
        # Calling range(args, kwargs) (line 340)
        range_call_result_8362 = invoke(stypy.reporting.localization.Localization(__file__, 340, 17), range_8359, *[int_8360], **kwargs_8361)
        
        # Testing the type of a for loop iterable (line 340)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 340, 8), range_call_result_8362)
        # Getting the type of the for loop variable (line 340)
        for_loop_var_8363 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 340, 8), range_call_result_8362)
        # Assigning a type to the variable 'i' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'i', for_loop_var_8363)
        # SSA begins for a for statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to netcdf_file(...): (line 342)
        # Processing the call arguments (line 342)
        str_8365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 29), 'str', 'withRecordDimension.nc')
        str_8366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 55), 'str', 'a')
        # Processing the call keyword arguments (line 342)
        kwargs_8367 = {}
        # Getting the type of 'netcdf_file' (line 342)
        netcdf_file_8364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 17), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 342)
        netcdf_file_call_result_8368 = invoke(stypy.reporting.localization.Localization(__file__, 342, 17), netcdf_file_8364, *[str_8365, str_8366], **kwargs_8367)
        
        with_8369 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 342, 17), netcdf_file_call_result_8368, 'with parameter', '__enter__', '__exit__')

        if with_8369:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 342)
            enter___8370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 17), netcdf_file_call_result_8368, '__enter__')
            with_enter_8371 = invoke(stypy.reporting.localization.Localization(__file__, 342, 17), enter___8370)
            # Assigning a type to the variable 'f' (line 342)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 17), 'f', with_enter_8371)
            
            # Assigning a Call to a Attribute (line 343):
            
            # Call to append(...): (line 343)
            # Processing the call arguments (line 343)
            
            # Obtaining the type of the subscript
            str_8374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 65), 'str', 'time')
            # Getting the type of 'f' (line 343)
            f_8375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 53), 'f', False)
            # Obtaining the member 'variables' of a type (line 343)
            variables_8376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 53), f_8375, 'variables')
            # Obtaining the member '__getitem__' of a type (line 343)
            getitem___8377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 53), variables_8376, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 343)
            subscript_call_result_8378 = invoke(stypy.reporting.localization.Localization(__file__, 343, 53), getitem___8377, str_8374)
            
            # Obtaining the member 'data' of a type (line 343)
            data_8379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 53), subscript_call_result_8378, 'data')
            # Getting the type of 'i' (line 343)
            i_8380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 79), 'i', False)
            # Processing the call keyword arguments (line 343)
            kwargs_8381 = {}
            # Getting the type of 'np' (line 343)
            np_8372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 43), 'np', False)
            # Obtaining the member 'append' of a type (line 343)
            append_8373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 43), np_8372, 'append')
            # Calling append(args, kwargs) (line 343)
            append_call_result_8382 = invoke(stypy.reporting.localization.Localization(__file__, 343, 43), append_8373, *[data_8379, i_8380], **kwargs_8381)
            
            
            # Obtaining the type of the subscript
            str_8383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 28), 'str', 'time')
            # Getting the type of 'f' (line 343)
            f_8384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 16), 'f')
            # Obtaining the member 'variables' of a type (line 343)
            variables_8385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 16), f_8384, 'variables')
            # Obtaining the member '__getitem__' of a type (line 343)
            getitem___8386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 16), variables_8385, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 343)
            subscript_call_result_8387 = invoke(stypy.reporting.localization.Localization(__file__, 343, 16), getitem___8386, str_8383)
            
            # Setting the type of the member 'data' of a type (line 343)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 16), subscript_call_result_8387, 'data', append_call_result_8382)
            
            # Assigning a BinOp to a Subscript (line 344):
            
            # Call to ones(...): (line 344)
            # Processing the call arguments (line 344)
            
            # Obtaining an instance of the builtin type 'tuple' (line 344)
            tuple_8390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 60), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 344)
            # Adding element type (line 344)
            # Getting the type of 'dataSize' (line 344)
            dataSize_8391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 60), 'dataSize', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 60), tuple_8390, dataSize_8391)
            # Adding element type (line 344)
            # Getting the type of 'dataSize' (line 344)
            dataSize_8392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 70), 'dataSize', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 60), tuple_8390, dataSize_8392)
            
            # Processing the call keyword arguments (line 344)
            kwargs_8393 = {}
            # Getting the type of 'np' (line 344)
            np_8388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 51), 'np', False)
            # Obtaining the member 'ones' of a type (line 344)
            ones_8389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 51), np_8388, 'ones')
            # Calling ones(args, kwargs) (line 344)
            ones_call_result_8394 = invoke(stypy.reporting.localization.Localization(__file__, 344, 51), ones_8389, *[tuple_8390], **kwargs_8393)
            
            # Getting the type of 'i' (line 344)
            i_8395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 81), 'i')
            # Applying the binary operator '*' (line 344)
            result_mul_8396 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 51), '*', ones_call_result_8394, i_8395)
            
            
            # Obtaining the type of the subscript
            str_8397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 28), 'str', 'testData')
            # Getting the type of 'f' (line 344)
            f_8398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'f')
            # Obtaining the member 'variables' of a type (line 344)
            variables_8399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 16), f_8398, 'variables')
            # Obtaining the member '__getitem__' of a type (line 344)
            getitem___8400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 16), variables_8399, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 344)
            subscript_call_result_8401 = invoke(stypy.reporting.localization.Localization(__file__, 344, 16), getitem___8400, str_8397)
            
            # Getting the type of 'i' (line 344)
            i_8402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 40), 'i')
            slice_8403 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 344, 16), None, None, None)
            slice_8404 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 344, 16), None, None, None)
            # Storing an element on a container (line 344)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 16), subscript_call_result_8401, ((i_8402, slice_8403, slice_8404), result_mul_8396))
            
            # Call to flush(...): (line 345)
            # Processing the call keyword arguments (line 345)
            kwargs_8407 = {}
            # Getting the type of 'f' (line 345)
            f_8405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 16), 'f', False)
            # Obtaining the member 'flush' of a type (line 345)
            flush_8406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 16), f_8405, 'flush')
            # Calling flush(args, kwargs) (line 345)
            flush_call_result_8408 = invoke(stypy.reporting.localization.Localization(__file__, 345, 16), flush_8406, *[], **kwargs_8407)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 342)
            exit___8409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 17), netcdf_file_call_result_8368, '__exit__')
            with_exit_8410 = invoke(stypy.reporting.localization.Localization(__file__, 342, 17), exit___8409, None, None, None)

        
        # Call to netcdf_file(...): (line 348)
        # Processing the call arguments (line 348)
        str_8412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 29), 'str', 'withRecordDimension.nc')
        # Processing the call keyword arguments (line 348)
        kwargs_8413 = {}
        # Getting the type of 'netcdf_file' (line 348)
        netcdf_file_8411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 17), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 348)
        netcdf_file_call_result_8414 = invoke(stypy.reporting.localization.Localization(__file__, 348, 17), netcdf_file_8411, *[str_8412], **kwargs_8413)
        
        with_8415 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 348, 17), netcdf_file_call_result_8414, 'with parameter', '__enter__', '__exit__')

        if with_8415:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 348)
            enter___8416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 17), netcdf_file_call_result_8414, '__enter__')
            with_enter_8417 = invoke(stypy.reporting.localization.Localization(__file__, 348, 17), enter___8416)
            # Assigning a type to the variable 'f' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 17), 'f', with_enter_8417)
            
            # Call to assert_equal(...): (line 349)
            # Processing the call arguments (line 349)
            
            # Obtaining the type of the subscript
            int_8419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 49), 'int')
            
            # Obtaining the type of the subscript
            str_8420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 41), 'str', 'time')
            # Getting the type of 'f' (line 349)
            f_8421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 29), 'f', False)
            # Obtaining the member 'variables' of a type (line 349)
            variables_8422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 29), f_8421, 'variables')
            # Obtaining the member '__getitem__' of a type (line 349)
            getitem___8423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 29), variables_8422, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 349)
            subscript_call_result_8424 = invoke(stypy.reporting.localization.Localization(__file__, 349, 29), getitem___8423, str_8420)
            
            # Obtaining the member '__getitem__' of a type (line 349)
            getitem___8425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 29), subscript_call_result_8424, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 349)
            subscript_call_result_8426 = invoke(stypy.reporting.localization.Localization(__file__, 349, 29), getitem___8425, int_8419)
            
            # Getting the type of 'i' (line 349)
            i_8427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 54), 'i', False)
            # Processing the call keyword arguments (line 349)
            kwargs_8428 = {}
            # Getting the type of 'assert_equal' (line 349)
            assert_equal_8418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 349)
            assert_equal_call_result_8429 = invoke(stypy.reporting.localization.Localization(__file__, 349, 16), assert_equal_8418, *[subscript_call_result_8426, i_8427], **kwargs_8428)
            
            
            # Call to assert_equal(...): (line 350)
            # Processing the call arguments (line 350)
            
            # Call to copy(...): (line 350)
            # Processing the call keyword arguments (line 350)
            kwargs_8442 = {}
            
            # Obtaining the type of the subscript
            int_8431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 53), 'int')
            slice_8432 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 350, 29), None, None, None)
            slice_8433 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 350, 29), None, None, None)
            
            # Obtaining the type of the subscript
            str_8434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 41), 'str', 'testData')
            # Getting the type of 'f' (line 350)
            f_8435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 29), 'f', False)
            # Obtaining the member 'variables' of a type (line 350)
            variables_8436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 29), f_8435, 'variables')
            # Obtaining the member '__getitem__' of a type (line 350)
            getitem___8437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 29), variables_8436, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 350)
            subscript_call_result_8438 = invoke(stypy.reporting.localization.Localization(__file__, 350, 29), getitem___8437, str_8434)
            
            # Obtaining the member '__getitem__' of a type (line 350)
            getitem___8439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 29), subscript_call_result_8438, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 350)
            subscript_call_result_8440 = invoke(stypy.reporting.localization.Localization(__file__, 350, 29), getitem___8439, (int_8431, slice_8432, slice_8433))
            
            # Obtaining the member 'copy' of a type (line 350)
            copy_8441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 29), subscript_call_result_8440, 'copy')
            # Calling copy(args, kwargs) (line 350)
            copy_call_result_8443 = invoke(stypy.reporting.localization.Localization(__file__, 350, 29), copy_8441, *[], **kwargs_8442)
            
            
            # Call to ones(...): (line 350)
            # Processing the call arguments (line 350)
            
            # Obtaining an instance of the builtin type 'tuple' (line 350)
            tuple_8446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 80), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 350)
            # Adding element type (line 350)
            # Getting the type of 'dataSize' (line 350)
            dataSize_8447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 80), 'dataSize', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 80), tuple_8446, dataSize_8447)
            # Adding element type (line 350)
            # Getting the type of 'dataSize' (line 350)
            dataSize_8448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 90), 'dataSize', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 80), tuple_8446, dataSize_8448)
            
            # Processing the call keyword arguments (line 350)
            kwargs_8449 = {}
            # Getting the type of 'np' (line 350)
            np_8444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 71), 'np', False)
            # Obtaining the member 'ones' of a type (line 350)
            ones_8445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 71), np_8444, 'ones')
            # Calling ones(args, kwargs) (line 350)
            ones_call_result_8450 = invoke(stypy.reporting.localization.Localization(__file__, 350, 71), ones_8445, *[tuple_8446], **kwargs_8449)
            
            # Getting the type of 'i' (line 350)
            i_8451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 101), 'i', False)
            # Applying the binary operator '*' (line 350)
            result_mul_8452 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 71), '*', ones_call_result_8450, i_8451)
            
            # Processing the call keyword arguments (line 350)
            kwargs_8453 = {}
            # Getting the type of 'assert_equal' (line 350)
            assert_equal_8430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 350)
            assert_equal_call_result_8454 = invoke(stypy.reporting.localization.Localization(__file__, 350, 16), assert_equal_8430, *[copy_call_result_8443, result_mul_8452], **kwargs_8453)
            
            
            # Call to assert_equal(...): (line 351)
            # Processing the call arguments (line 351)
            
            # Obtaining the type of the subscript
            int_8456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 60), 'int')
            
            # Obtaining the type of the subscript
            str_8457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 41), 'str', 'time')
            # Getting the type of 'f' (line 351)
            f_8458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 29), 'f', False)
            # Obtaining the member 'variables' of a type (line 351)
            variables_8459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 29), f_8458, 'variables')
            # Obtaining the member '__getitem__' of a type (line 351)
            getitem___8460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 29), variables_8459, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 351)
            subscript_call_result_8461 = invoke(stypy.reporting.localization.Localization(__file__, 351, 29), getitem___8460, str_8457)
            
            # Obtaining the member 'data' of a type (line 351)
            data_8462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 29), subscript_call_result_8461, 'data')
            # Obtaining the member 'shape' of a type (line 351)
            shape_8463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 29), data_8462, 'shape')
            # Obtaining the member '__getitem__' of a type (line 351)
            getitem___8464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 29), shape_8463, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 351)
            subscript_call_result_8465 = invoke(stypy.reporting.localization.Localization(__file__, 351, 29), getitem___8464, int_8456)
            
            # Getting the type of 'i' (line 351)
            i_8466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 64), 'i', False)
            int_8467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 66), 'int')
            # Applying the binary operator '+' (line 351)
            result_add_8468 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 64), '+', i_8466, int_8467)
            
            # Processing the call keyword arguments (line 351)
            kwargs_8469 = {}
            # Getting the type of 'assert_equal' (line 351)
            assert_equal_8455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 16), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 351)
            assert_equal_call_result_8470 = invoke(stypy.reporting.localization.Localization(__file__, 351, 16), assert_equal_8455, *[subscript_call_result_8465, result_add_8468], **kwargs_8469)
            
            
            # Call to assert_equal(...): (line 352)
            # Processing the call arguments (line 352)
            
            # Obtaining the type of the subscript
            int_8472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 64), 'int')
            
            # Obtaining the type of the subscript
            str_8473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 41), 'str', 'testData')
            # Getting the type of 'f' (line 352)
            f_8474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 29), 'f', False)
            # Obtaining the member 'variables' of a type (line 352)
            variables_8475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 29), f_8474, 'variables')
            # Obtaining the member '__getitem__' of a type (line 352)
            getitem___8476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 29), variables_8475, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 352)
            subscript_call_result_8477 = invoke(stypy.reporting.localization.Localization(__file__, 352, 29), getitem___8476, str_8473)
            
            # Obtaining the member 'data' of a type (line 352)
            data_8478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 29), subscript_call_result_8477, 'data')
            # Obtaining the member 'shape' of a type (line 352)
            shape_8479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 29), data_8478, 'shape')
            # Obtaining the member '__getitem__' of a type (line 352)
            getitem___8480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 29), shape_8479, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 352)
            subscript_call_result_8481 = invoke(stypy.reporting.localization.Localization(__file__, 352, 29), getitem___8480, int_8472)
            
            # Getting the type of 'i' (line 352)
            i_8482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 68), 'i', False)
            int_8483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 70), 'int')
            # Applying the binary operator '+' (line 352)
            result_add_8484 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 68), '+', i_8482, int_8483)
            
            # Processing the call keyword arguments (line 352)
            kwargs_8485 = {}
            # Getting the type of 'assert_equal' (line 352)
            assert_equal_8471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 352)
            assert_equal_call_result_8486 = invoke(stypy.reporting.localization.Localization(__file__, 352, 16), assert_equal_8471, *[subscript_call_result_8481, result_add_8484], **kwargs_8485)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 348)
            exit___8487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 17), netcdf_file_call_result_8414, '__exit__')
            with_exit_8488 = invoke(stypy.reporting.localization.Localization(__file__, 348, 17), exit___8487, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to netcdf_file(...): (line 356)
        # Processing the call arguments (line 356)
        str_8490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 25), 'str', 'withRecordDimension.nc')
        # Processing the call keyword arguments (line 356)
        kwargs_8491 = {}
        # Getting the type of 'netcdf_file' (line 356)
        netcdf_file_8489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 13), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 356)
        netcdf_file_call_result_8492 = invoke(stypy.reporting.localization.Localization(__file__, 356, 13), netcdf_file_8489, *[str_8490], **kwargs_8491)
        
        with_8493 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 356, 13), netcdf_file_call_result_8492, 'with parameter', '__enter__', '__exit__')

        if with_8493:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 356)
            enter___8494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 13), netcdf_file_call_result_8492, '__enter__')
            with_enter_8495 = invoke(stypy.reporting.localization.Localization(__file__, 356, 13), enter___8494)
            # Assigning a type to the variable 'f' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 13), 'f', with_enter_8495)
            
            # Call to assert_raises(...): (line 357)
            # Processing the call arguments (line 357)
            # Getting the type of 'KeyError' (line 357)
            KeyError_8497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 31), 'KeyError', False)
            # Processing the call keyword arguments (line 357)
            kwargs_8498 = {}
            # Getting the type of 'assert_raises' (line 357)
            assert_raises_8496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), 'assert_raises', False)
            # Calling assert_raises(args, kwargs) (line 357)
            assert_raises_call_result_8499 = invoke(stypy.reporting.localization.Localization(__file__, 357, 17), assert_raises_8496, *[KeyError_8497], **kwargs_8498)
            
            with_8500 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 357, 17), assert_raises_call_result_8499, 'with parameter', '__enter__', '__exit__')

            if with_8500:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 357)
                enter___8501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 17), assert_raises_call_result_8499, '__enter__')
                with_enter_8502 = invoke(stypy.reporting.localization.Localization(__file__, 357, 17), enter___8501)
                # Assigning a type to the variable 'ar' (line 357)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), 'ar', with_enter_8502)
                
                # Obtaining the type of the subscript
                str_8503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 52), 'str', 'data')
                
                # Obtaining the type of the subscript
                str_8504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 28), 'str', 'testData')
                # Getting the type of 'f' (line 358)
                f_8505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'f')
                # Obtaining the member 'variables' of a type (line 358)
                variables_8506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 16), f_8505, 'variables')
                # Obtaining the member '__getitem__' of a type (line 358)
                getitem___8507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 16), variables_8506, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 358)
                subscript_call_result_8508 = invoke(stypy.reporting.localization.Localization(__file__, 358, 16), getitem___8507, str_8504)
                
                # Obtaining the member '_attributes' of a type (line 358)
                _attributes_8509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 16), subscript_call_result_8508, '_attributes')
                # Obtaining the member '__getitem__' of a type (line 358)
                getitem___8510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 16), _attributes_8509, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 358)
                subscript_call_result_8511 = invoke(stypy.reporting.localization.Localization(__file__, 358, 16), getitem___8510, str_8503)
                
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 357)
                exit___8512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 17), assert_raises_call_result_8499, '__exit__')
                with_exit_8513 = invoke(stypy.reporting.localization.Localization(__file__, 357, 17), exit___8512, None, None, None)

            
            # Assigning a Attribute to a Name (line 359):
            # Getting the type of 'ar' (line 359)
            ar_8514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 17), 'ar')
            # Obtaining the member 'value' of a type (line 359)
            value_8515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 17), ar_8514, 'value')
            # Assigning a type to the variable 'ex' (line 359)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'ex', value_8515)
            
            # Call to assert_equal(...): (line 360)
            # Processing the call arguments (line 360)
            
            # Obtaining the type of the subscript
            int_8517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 33), 'int')
            # Getting the type of 'ex' (line 360)
            ex_8518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 25), 'ex', False)
            # Obtaining the member 'args' of a type (line 360)
            args_8519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 25), ex_8518, 'args')
            # Obtaining the member '__getitem__' of a type (line 360)
            getitem___8520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 25), args_8519, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 360)
            subscript_call_result_8521 = invoke(stypy.reporting.localization.Localization(__file__, 360, 25), getitem___8520, int_8517)
            
            str_8522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 37), 'str', 'data')
            # Processing the call keyword arguments (line 360)
            kwargs_8523 = {}
            # Getting the type of 'assert_equal' (line 360)
            assert_equal_8516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 360)
            assert_equal_call_result_8524 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), assert_equal_8516, *[subscript_call_result_8521, str_8522], **kwargs_8523)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 356)
            exit___8525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 13), netcdf_file_call_result_8492, '__exit__')
            with_exit_8526 = invoke(stypy.reporting.localization.Localization(__file__, 356, 13), exit___8525, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 325)
        exit___8527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 9), in_tempdir_call_result_8265, '__exit__')
        with_exit_8528 = invoke(stypy.reporting.localization.Localization(__file__, 325, 9), exit___8527, None, None, None)

    
    # ################# End of 'test_append_recordDimension(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_append_recordDimension' in the type store
    # Getting the type of 'stypy_return_type' (line 322)
    stypy_return_type_8529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8529)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_append_recordDimension'
    return stypy_return_type_8529

# Assigning a type to the variable 'test_append_recordDimension' (line 322)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 0), 'test_append_recordDimension', test_append_recordDimension)

@norecursion
def test_maskandscale(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_maskandscale'
    module_type_store = module_type_store.open_function_context('test_maskandscale', 362, 0, False)
    
    # Passed parameters checking function
    test_maskandscale.stypy_localization = localization
    test_maskandscale.stypy_type_of_self = None
    test_maskandscale.stypy_type_store = module_type_store
    test_maskandscale.stypy_function_name = 'test_maskandscale'
    test_maskandscale.stypy_param_names_list = []
    test_maskandscale.stypy_varargs_param_name = None
    test_maskandscale.stypy_kwargs_param_name = None
    test_maskandscale.stypy_call_defaults = defaults
    test_maskandscale.stypy_call_varargs = varargs
    test_maskandscale.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_maskandscale', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_maskandscale', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_maskandscale(...)' code ##################

    
    # Assigning a Call to a Name (line 363):
    
    # Call to linspace(...): (line 363)
    # Processing the call arguments (line 363)
    int_8532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 20), 'int')
    int_8533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 24), 'int')
    int_8534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 28), 'int')
    # Processing the call keyword arguments (line 363)
    kwargs_8535 = {}
    # Getting the type of 'np' (line 363)
    np_8530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 363)
    linspace_8531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), np_8530, 'linspace')
    # Calling linspace(args, kwargs) (line 363)
    linspace_call_result_8536 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), linspace_8531, *[int_8532, int_8533, int_8534], **kwargs_8535)
    
    # Assigning a type to the variable 't' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 't', linspace_call_result_8536)
    
    # Assigning a Num to a Subscript (line 364):
    int_8537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 11), 'int')
    # Getting the type of 't' (line 364)
    t_8538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 't')
    int_8539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 6), 'int')
    # Storing an element on a container (line 364)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 4), t_8538, (int_8539, int_8537))
    
    # Assigning a Call to a Name (line 365):
    
    # Call to masked_greater(...): (line 365)
    # Processing the call arguments (line 365)
    # Getting the type of 't' (line 365)
    t_8543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 30), 't', False)
    int_8544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 33), 'int')
    # Processing the call keyword arguments (line 365)
    kwargs_8545 = {}
    # Getting the type of 'np' (line 365)
    np_8540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 9), 'np', False)
    # Obtaining the member 'ma' of a type (line 365)
    ma_8541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 9), np_8540, 'ma')
    # Obtaining the member 'masked_greater' of a type (line 365)
    masked_greater_8542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 9), ma_8541, 'masked_greater')
    # Calling masked_greater(args, kwargs) (line 365)
    masked_greater_call_result_8546 = invoke(stypy.reporting.localization.Localization(__file__, 365, 9), masked_greater_8542, *[t_8543, int_8544], **kwargs_8545)
    
    # Assigning a type to the variable 'tm' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'tm', masked_greater_call_result_8546)
    
    # Assigning a Call to a Name (line 366):
    
    # Call to pjoin(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'TEST_DATA_PATH' (line 366)
    TEST_DATA_PATH_8548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 18), 'TEST_DATA_PATH', False)
    str_8549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 34), 'str', 'example_2.nc')
    # Processing the call keyword arguments (line 366)
    kwargs_8550 = {}
    # Getting the type of 'pjoin' (line 366)
    pjoin_8547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 366)
    pjoin_call_result_8551 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), pjoin_8547, *[TEST_DATA_PATH_8548, str_8549], **kwargs_8550)
    
    # Assigning a type to the variable 'fname' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'fname', pjoin_call_result_8551)
    
    # Call to netcdf_file(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'fname' (line 367)
    fname_8553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 21), 'fname', False)
    # Processing the call keyword arguments (line 367)
    # Getting the type of 'True' (line 367)
    True_8554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 41), 'True', False)
    keyword_8555 = True_8554
    kwargs_8556 = {'maskandscale': keyword_8555}
    # Getting the type of 'netcdf_file' (line 367)
    netcdf_file_8552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 367)
    netcdf_file_call_result_8557 = invoke(stypy.reporting.localization.Localization(__file__, 367, 9), netcdf_file_8552, *[fname_8553], **kwargs_8556)
    
    with_8558 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 367, 9), netcdf_file_call_result_8557, 'with parameter', '__enter__', '__exit__')

    if with_8558:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 367)
        enter___8559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 9), netcdf_file_call_result_8557, '__enter__')
        with_enter_8560 = invoke(stypy.reporting.localization.Localization(__file__, 367, 9), enter___8559)
        # Assigning a type to the variable 'f' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 9), 'f', with_enter_8560)
        
        # Assigning a Subscript to a Name (line 368):
        
        # Obtaining the type of the subscript
        str_8561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 27), 'str', 'Temperature')
        # Getting the type of 'f' (line 368)
        f_8562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 15), 'f')
        # Obtaining the member 'variables' of a type (line 368)
        variables_8563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 15), f_8562, 'variables')
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___8564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 15), variables_8563, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 368)
        subscript_call_result_8565 = invoke(stypy.reporting.localization.Localization(__file__, 368, 15), getitem___8564, str_8561)
        
        # Assigning a type to the variable 'Temp' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'Temp', subscript_call_result_8565)
        
        # Call to assert_equal(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'Temp' (line 369)
        Temp_8567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'Temp', False)
        # Obtaining the member 'missing_value' of a type (line 369)
        missing_value_8568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), Temp_8567, 'missing_value')
        int_8569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 41), 'int')
        # Processing the call keyword arguments (line 369)
        kwargs_8570 = {}
        # Getting the type of 'assert_equal' (line 369)
        assert_equal_8566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 369)
        assert_equal_call_result_8571 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), assert_equal_8566, *[missing_value_8568, int_8569], **kwargs_8570)
        
        
        # Call to assert_equal(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'Temp' (line 370)
        Temp_8573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 21), 'Temp', False)
        # Obtaining the member 'add_offset' of a type (line 370)
        add_offset_8574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 21), Temp_8573, 'add_offset')
        int_8575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 38), 'int')
        # Processing the call keyword arguments (line 370)
        kwargs_8576 = {}
        # Getting the type of 'assert_equal' (line 370)
        assert_equal_8572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 370)
        assert_equal_call_result_8577 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), assert_equal_8572, *[add_offset_8574, int_8575], **kwargs_8576)
        
        
        # Call to assert_equal(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'Temp' (line 371)
        Temp_8579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), 'Temp', False)
        # Obtaining the member 'scale_factor' of a type (line 371)
        scale_factor_8580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 21), Temp_8579, 'scale_factor')
        
        # Call to float32(...): (line 371)
        # Processing the call arguments (line 371)
        float_8583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 51), 'float')
        # Processing the call keyword arguments (line 371)
        kwargs_8584 = {}
        # Getting the type of 'np' (line 371)
        np_8581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 40), 'np', False)
        # Obtaining the member 'float32' of a type (line 371)
        float32_8582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 40), np_8581, 'float32')
        # Calling float32(args, kwargs) (line 371)
        float32_call_result_8585 = invoke(stypy.reporting.localization.Localization(__file__, 371, 40), float32_8582, *[float_8583], **kwargs_8584)
        
        # Processing the call keyword arguments (line 371)
        kwargs_8586 = {}
        # Getting the type of 'assert_equal' (line 371)
        assert_equal_8578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 371)
        assert_equal_call_result_8587 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), assert_equal_8578, *[scale_factor_8580, float32_call_result_8585], **kwargs_8586)
        
        
        # Assigning a Call to a Name (line 372):
        
        # Call to compressed(...): (line 372)
        # Processing the call keyword arguments (line 372)
        kwargs_8593 = {}
        
        # Obtaining the type of the subscript
        slice_8588 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 372, 16), None, None, None)
        # Getting the type of 'Temp' (line 372)
        Temp_8589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'Temp', False)
        # Obtaining the member '__getitem__' of a type (line 372)
        getitem___8590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 16), Temp_8589, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 372)
        subscript_call_result_8591 = invoke(stypy.reporting.localization.Localization(__file__, 372, 16), getitem___8590, slice_8588)
        
        # Obtaining the member 'compressed' of a type (line 372)
        compressed_8592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 16), subscript_call_result_8591, 'compressed')
        # Calling compressed(args, kwargs) (line 372)
        compressed_call_result_8594 = invoke(stypy.reporting.localization.Localization(__file__, 372, 16), compressed_8592, *[], **kwargs_8593)
        
        # Assigning a type to the variable 'found' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'found', compressed_call_result_8594)
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 373, 8), module_type_store, 'Temp')
        
        # Assigning a Call to a Name (line 374):
        
        # Call to round(...): (line 374)
        # Processing the call arguments (line 374)
        
        # Call to compressed(...): (line 374)
        # Processing the call keyword arguments (line 374)
        kwargs_8599 = {}
        # Getting the type of 'tm' (line 374)
        tm_8597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'tm', False)
        # Obtaining the member 'compressed' of a type (line 374)
        compressed_8598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 28), tm_8597, 'compressed')
        # Calling compressed(args, kwargs) (line 374)
        compressed_call_result_8600 = invoke(stypy.reporting.localization.Localization(__file__, 374, 28), compressed_8598, *[], **kwargs_8599)
        
        int_8601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 45), 'int')
        # Processing the call keyword arguments (line 374)
        kwargs_8602 = {}
        # Getting the type of 'np' (line 374)
        np_8595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 19), 'np', False)
        # Obtaining the member 'round' of a type (line 374)
        round_8596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 19), np_8595, 'round')
        # Calling round(args, kwargs) (line 374)
        round_call_result_8603 = invoke(stypy.reporting.localization.Localization(__file__, 374, 19), round_8596, *[compressed_call_result_8600, int_8601], **kwargs_8602)
        
        # Assigning a type to the variable 'expected' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'expected', round_call_result_8603)
        
        # Call to assert_allclose(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'found' (line 375)
        found_8605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 24), 'found', False)
        # Getting the type of 'expected' (line 375)
        expected_8606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 31), 'expected', False)
        # Processing the call keyword arguments (line 375)
        kwargs_8607 = {}
        # Getting the type of 'assert_allclose' (line 375)
        assert_allclose_8604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 375)
        assert_allclose_call_result_8608 = invoke(stypy.reporting.localization.Localization(__file__, 375, 8), assert_allclose_8604, *[found_8605, expected_8606], **kwargs_8607)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 367)
        exit___8609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 9), netcdf_file_call_result_8557, '__exit__')
        with_exit_8610 = invoke(stypy.reporting.localization.Localization(__file__, 367, 9), exit___8609, None, None, None)

    
    # Call to in_tempdir(...): (line 377)
    # Processing the call keyword arguments (line 377)
    kwargs_8612 = {}
    # Getting the type of 'in_tempdir' (line 377)
    in_tempdir_8611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 9), 'in_tempdir', False)
    # Calling in_tempdir(args, kwargs) (line 377)
    in_tempdir_call_result_8613 = invoke(stypy.reporting.localization.Localization(__file__, 377, 9), in_tempdir_8611, *[], **kwargs_8612)
    
    with_8614 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 377, 9), in_tempdir_call_result_8613, 'with parameter', '__enter__', '__exit__')

    if with_8614:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 377)
        enter___8615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 9), in_tempdir_call_result_8613, '__enter__')
        with_enter_8616 = invoke(stypy.reporting.localization.Localization(__file__, 377, 9), enter___8615)
        
        # Assigning a Str to a Name (line 378):
        str_8617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 19), 'str', 'ms.nc')
        # Assigning a type to the variable 'newfname' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'newfname', str_8617)
        
        # Assigning a Call to a Name (line 379):
        
        # Call to netcdf_file(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'newfname' (line 379)
        newfname_8619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 24), 'newfname', False)
        str_8620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 34), 'str', 'w')
        # Processing the call keyword arguments (line 379)
        # Getting the type of 'True' (line 379)
        True_8621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 52), 'True', False)
        keyword_8622 = True_8621
        kwargs_8623 = {'maskandscale': keyword_8622}
        # Getting the type of 'netcdf_file' (line 379)
        netcdf_file_8618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 379)
        netcdf_file_call_result_8624 = invoke(stypy.reporting.localization.Localization(__file__, 379, 12), netcdf_file_8618, *[newfname_8619, str_8620], **kwargs_8623)
        
        # Assigning a type to the variable 'f' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'f', netcdf_file_call_result_8624)
        
        # Call to createDimension(...): (line 380)
        # Processing the call arguments (line 380)
        str_8627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 26), 'str', 'Temperature')
        
        # Call to len(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'tm' (line 380)
        tm_8629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 45), 'tm', False)
        # Processing the call keyword arguments (line 380)
        kwargs_8630 = {}
        # Getting the type of 'len' (line 380)
        len_8628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 41), 'len', False)
        # Calling len(args, kwargs) (line 380)
        len_call_result_8631 = invoke(stypy.reporting.localization.Localization(__file__, 380, 41), len_8628, *[tm_8629], **kwargs_8630)
        
        # Processing the call keyword arguments (line 380)
        kwargs_8632 = {}
        # Getting the type of 'f' (line 380)
        f_8625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'f', False)
        # Obtaining the member 'createDimension' of a type (line 380)
        createDimension_8626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), f_8625, 'createDimension')
        # Calling createDimension(args, kwargs) (line 380)
        createDimension_call_result_8633 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), createDimension_8626, *[str_8627, len_call_result_8631], **kwargs_8632)
        
        
        # Assigning a Call to a Name (line 381):
        
        # Call to createVariable(...): (line 381)
        # Processing the call arguments (line 381)
        str_8636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 32), 'str', 'Temperature')
        str_8637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 47), 'str', 'i')
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_8638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        str_8639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 53), 'str', 'Temperature')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 53), tuple_8638, str_8639)
        
        # Processing the call keyword arguments (line 381)
        kwargs_8640 = {}
        # Getting the type of 'f' (line 381)
        f_8634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 15), 'f', False)
        # Obtaining the member 'createVariable' of a type (line 381)
        createVariable_8635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 15), f_8634, 'createVariable')
        # Calling createVariable(args, kwargs) (line 381)
        createVariable_call_result_8641 = invoke(stypy.reporting.localization.Localization(__file__, 381, 15), createVariable_8635, *[str_8636, str_8637, tuple_8638], **kwargs_8640)
        
        # Assigning a type to the variable 'temp' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'temp', createVariable_call_result_8641)
        
        # Assigning a Num to a Attribute (line 382):
        int_8642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 29), 'int')
        # Getting the type of 'temp' (line 382)
        temp_8643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'temp')
        # Setting the type of the member 'missing_value' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), temp_8643, 'missing_value', int_8642)
        
        # Assigning a Num to a Attribute (line 383):
        float_8644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 28), 'float')
        # Getting the type of 'temp' (line 383)
        temp_8645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'temp')
        # Setting the type of the member 'scale_factor' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), temp_8645, 'scale_factor', float_8644)
        
        # Assigning a Num to a Attribute (line 384):
        int_8646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 26), 'int')
        # Getting the type of 'temp' (line 384)
        temp_8647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'temp')
        # Setting the type of the member 'add_offset' of a type (line 384)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), temp_8647, 'add_offset', int_8646)
        
        # Assigning a Name to a Subscript (line 385):
        # Getting the type of 'tm' (line 385)
        tm_8648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'tm')
        # Getting the type of 'temp' (line 385)
        temp_8649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'temp')
        slice_8650 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 385, 8), None, None, None)
        # Storing an element on a container (line 385)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 8), temp_8649, (slice_8650, tm_8648))
        
        # Call to close(...): (line 386)
        # Processing the call keyword arguments (line 386)
        kwargs_8653 = {}
        # Getting the type of 'f' (line 386)
        f_8651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'f', False)
        # Obtaining the member 'close' of a type (line 386)
        close_8652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), f_8651, 'close')
        # Calling close(args, kwargs) (line 386)
        close_call_result_8654 = invoke(stypy.reporting.localization.Localization(__file__, 386, 8), close_8652, *[], **kwargs_8653)
        
        
        # Call to netcdf_file(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'newfname' (line 388)
        newfname_8656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 25), 'newfname', False)
        # Processing the call keyword arguments (line 388)
        # Getting the type of 'True' (line 388)
        True_8657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 48), 'True', False)
        keyword_8658 = True_8657
        kwargs_8659 = {'maskandscale': keyword_8658}
        # Getting the type of 'netcdf_file' (line 388)
        netcdf_file_8655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 13), 'netcdf_file', False)
        # Calling netcdf_file(args, kwargs) (line 388)
        netcdf_file_call_result_8660 = invoke(stypy.reporting.localization.Localization(__file__, 388, 13), netcdf_file_8655, *[newfname_8656], **kwargs_8659)
        
        with_8661 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 388, 13), netcdf_file_call_result_8660, 'with parameter', '__enter__', '__exit__')

        if with_8661:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 388)
            enter___8662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 13), netcdf_file_call_result_8660, '__enter__')
            with_enter_8663 = invoke(stypy.reporting.localization.Localization(__file__, 388, 13), enter___8662)
            # Assigning a type to the variable 'f' (line 388)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 13), 'f', with_enter_8663)
            
            # Assigning a Subscript to a Name (line 389):
            
            # Obtaining the type of the subscript
            str_8664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 31), 'str', 'Temperature')
            # Getting the type of 'f' (line 389)
            f_8665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'f')
            # Obtaining the member 'variables' of a type (line 389)
            variables_8666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 19), f_8665, 'variables')
            # Obtaining the member '__getitem__' of a type (line 389)
            getitem___8667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 19), variables_8666, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 389)
            subscript_call_result_8668 = invoke(stypy.reporting.localization.Localization(__file__, 389, 19), getitem___8667, str_8664)
            
            # Assigning a type to the variable 'Temp' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'Temp', subscript_call_result_8668)
            
            # Call to assert_equal(...): (line 390)
            # Processing the call arguments (line 390)
            # Getting the type of 'Temp' (line 390)
            Temp_8670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 25), 'Temp', False)
            # Obtaining the member 'missing_value' of a type (line 390)
            missing_value_8671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 25), Temp_8670, 'missing_value')
            int_8672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 45), 'int')
            # Processing the call keyword arguments (line 390)
            kwargs_8673 = {}
            # Getting the type of 'assert_equal' (line 390)
            assert_equal_8669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 390)
            assert_equal_call_result_8674 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), assert_equal_8669, *[missing_value_8671, int_8672], **kwargs_8673)
            
            
            # Call to assert_equal(...): (line 391)
            # Processing the call arguments (line 391)
            # Getting the type of 'Temp' (line 391)
            Temp_8676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 25), 'Temp', False)
            # Obtaining the member 'add_offset' of a type (line 391)
            add_offset_8677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 25), Temp_8676, 'add_offset')
            int_8678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 42), 'int')
            # Processing the call keyword arguments (line 391)
            kwargs_8679 = {}
            # Getting the type of 'assert_equal' (line 391)
            assert_equal_8675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 391)
            assert_equal_call_result_8680 = invoke(stypy.reporting.localization.Localization(__file__, 391, 12), assert_equal_8675, *[add_offset_8677, int_8678], **kwargs_8679)
            
            
            # Call to assert_equal(...): (line 392)
            # Processing the call arguments (line 392)
            # Getting the type of 'Temp' (line 392)
            Temp_8682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 25), 'Temp', False)
            # Obtaining the member 'scale_factor' of a type (line 392)
            scale_factor_8683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 25), Temp_8682, 'scale_factor')
            
            # Call to float32(...): (line 392)
            # Processing the call arguments (line 392)
            float_8686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 55), 'float')
            # Processing the call keyword arguments (line 392)
            kwargs_8687 = {}
            # Getting the type of 'np' (line 392)
            np_8684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 44), 'np', False)
            # Obtaining the member 'float32' of a type (line 392)
            float32_8685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 44), np_8684, 'float32')
            # Calling float32(args, kwargs) (line 392)
            float32_call_result_8688 = invoke(stypy.reporting.localization.Localization(__file__, 392, 44), float32_8685, *[float_8686], **kwargs_8687)
            
            # Processing the call keyword arguments (line 392)
            kwargs_8689 = {}
            # Getting the type of 'assert_equal' (line 392)
            assert_equal_8681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 392)
            assert_equal_call_result_8690 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), assert_equal_8681, *[scale_factor_8683, float32_call_result_8688], **kwargs_8689)
            
            
            # Assigning a Call to a Name (line 393):
            
            # Call to round(...): (line 393)
            # Processing the call arguments (line 393)
            
            # Call to compressed(...): (line 393)
            # Processing the call keyword arguments (line 393)
            kwargs_8695 = {}
            # Getting the type of 'tm' (line 393)
            tm_8693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 32), 'tm', False)
            # Obtaining the member 'compressed' of a type (line 393)
            compressed_8694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 32), tm_8693, 'compressed')
            # Calling compressed(args, kwargs) (line 393)
            compressed_call_result_8696 = invoke(stypy.reporting.localization.Localization(__file__, 393, 32), compressed_8694, *[], **kwargs_8695)
            
            int_8697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 49), 'int')
            # Processing the call keyword arguments (line 393)
            kwargs_8698 = {}
            # Getting the type of 'np' (line 393)
            np_8691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 23), 'np', False)
            # Obtaining the member 'round' of a type (line 393)
            round_8692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 23), np_8691, 'round')
            # Calling round(args, kwargs) (line 393)
            round_call_result_8699 = invoke(stypy.reporting.localization.Localization(__file__, 393, 23), round_8692, *[compressed_call_result_8696, int_8697], **kwargs_8698)
            
            # Assigning a type to the variable 'expected' (line 393)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'expected', round_call_result_8699)
            
            # Assigning a Call to a Name (line 394):
            
            # Call to compressed(...): (line 394)
            # Processing the call keyword arguments (line 394)
            kwargs_8705 = {}
            
            # Obtaining the type of the subscript
            slice_8700 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 394, 20), None, None, None)
            # Getting the type of 'Temp' (line 394)
            Temp_8701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 20), 'Temp', False)
            # Obtaining the member '__getitem__' of a type (line 394)
            getitem___8702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 20), Temp_8701, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 394)
            subscript_call_result_8703 = invoke(stypy.reporting.localization.Localization(__file__, 394, 20), getitem___8702, slice_8700)
            
            # Obtaining the member 'compressed' of a type (line 394)
            compressed_8704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 20), subscript_call_result_8703, 'compressed')
            # Calling compressed(args, kwargs) (line 394)
            compressed_call_result_8706 = invoke(stypy.reporting.localization.Localization(__file__, 394, 20), compressed_8704, *[], **kwargs_8705)
            
            # Assigning a type to the variable 'found' (line 394)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'found', compressed_call_result_8706)
            # Deleting a member
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 395, 12), module_type_store, 'Temp')
            
            # Call to assert_allclose(...): (line 396)
            # Processing the call arguments (line 396)
            # Getting the type of 'found' (line 396)
            found_8708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 28), 'found', False)
            # Getting the type of 'expected' (line 396)
            expected_8709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 35), 'expected', False)
            # Processing the call keyword arguments (line 396)
            kwargs_8710 = {}
            # Getting the type of 'assert_allclose' (line 396)
            assert_allclose_8707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 396)
            assert_allclose_call_result_8711 = invoke(stypy.reporting.localization.Localization(__file__, 396, 12), assert_allclose_8707, *[found_8708, expected_8709], **kwargs_8710)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 388)
            exit___8712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 13), netcdf_file_call_result_8660, '__exit__')
            with_exit_8713 = invoke(stypy.reporting.localization.Localization(__file__, 388, 13), exit___8712, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 377)
        exit___8714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 9), in_tempdir_call_result_8613, '__exit__')
        with_exit_8715 = invoke(stypy.reporting.localization.Localization(__file__, 377, 9), exit___8714, None, None, None)

    
    # ################# End of 'test_maskandscale(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_maskandscale' in the type store
    # Getting the type of 'stypy_return_type' (line 362)
    stypy_return_type_8716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8716)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_maskandscale'
    return stypy_return_type_8716

# Assigning a type to the variable 'test_maskandscale' (line 362)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 0), 'test_maskandscale', test_maskandscale)

@norecursion
def test_read_withValuesNearFillValue(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_withValuesNearFillValue'
    module_type_store = module_type_store.open_function_context('test_read_withValuesNearFillValue', 403, 0, False)
    
    # Passed parameters checking function
    test_read_withValuesNearFillValue.stypy_localization = localization
    test_read_withValuesNearFillValue.stypy_type_of_self = None
    test_read_withValuesNearFillValue.stypy_type_store = module_type_store
    test_read_withValuesNearFillValue.stypy_function_name = 'test_read_withValuesNearFillValue'
    test_read_withValuesNearFillValue.stypy_param_names_list = []
    test_read_withValuesNearFillValue.stypy_varargs_param_name = None
    test_read_withValuesNearFillValue.stypy_kwargs_param_name = None
    test_read_withValuesNearFillValue.stypy_call_defaults = defaults
    test_read_withValuesNearFillValue.stypy_call_varargs = varargs
    test_read_withValuesNearFillValue.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_withValuesNearFillValue', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_withValuesNearFillValue', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_withValuesNearFillValue(...)' code ##################

    
    # Assigning a Call to a Name (line 405):
    
    # Call to pjoin(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'TEST_DATA_PATH' (line 405)
    TEST_DATA_PATH_8718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 18), 'TEST_DATA_PATH', False)
    str_8719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 34), 'str', 'example_3_maskedvals.nc')
    # Processing the call keyword arguments (line 405)
    kwargs_8720 = {}
    # Getting the type of 'pjoin' (line 405)
    pjoin_8717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 405)
    pjoin_call_result_8721 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), pjoin_8717, *[TEST_DATA_PATH_8718, str_8719], **kwargs_8720)
    
    # Assigning a type to the variable 'fname' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'fname', pjoin_call_result_8721)
    
    # Call to netcdf_file(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'fname' (line 406)
    fname_8723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'fname', False)
    # Processing the call keyword arguments (line 406)
    # Getting the type of 'True' (line 406)
    True_8724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 41), 'True', False)
    keyword_8725 = True_8724
    kwargs_8726 = {'maskandscale': keyword_8725}
    # Getting the type of 'netcdf_file' (line 406)
    netcdf_file_8722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 406)
    netcdf_file_call_result_8727 = invoke(stypy.reporting.localization.Localization(__file__, 406, 9), netcdf_file_8722, *[fname_8723], **kwargs_8726)
    
    with_8728 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 406, 9), netcdf_file_call_result_8727, 'with parameter', '__enter__', '__exit__')

    if with_8728:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 406)
        enter___8729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 9), netcdf_file_call_result_8727, '__enter__')
        with_enter_8730 = invoke(stypy.reporting.localization.Localization(__file__, 406, 9), enter___8729)
        # Assigning a type to the variable 'f' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 9), 'f', with_enter_8730)
        
        # Assigning a Subscript to a Name (line 407):
        
        # Obtaining the type of the subscript
        slice_8731 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 407, 18), None, None, None)
        
        # Obtaining the type of the subscript
        str_8732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 30), 'str', 'var1_fillval0')
        # Getting the type of 'f' (line 407)
        f_8733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'f')
        # Obtaining the member 'variables' of a type (line 407)
        variables_8734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 18), f_8733, 'variables')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___8735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 18), variables_8734, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_8736 = invoke(stypy.reporting.localization.Localization(__file__, 407, 18), getitem___8735, str_8732)
        
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___8737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 18), subscript_call_result_8736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_8738 = invoke(stypy.reporting.localization.Localization(__file__, 407, 18), getitem___8737, slice_8731)
        
        # Assigning a type to the variable 'vardata' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'vardata', subscript_call_result_8738)
        
        # Call to assert_mask_matches(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'vardata' (line 408)
        vardata_8740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 28), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 408)
        list_8741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 408)
        # Adding element type (line 408)
        # Getting the type of 'False' (line 408)
        False_8742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 38), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 37), list_8741, False_8742)
        # Adding element type (line 408)
        # Getting the type of 'True' (line 408)
        True_8743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 45), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 37), list_8741, True_8743)
        # Adding element type (line 408)
        # Getting the type of 'False' (line 408)
        False_8744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 51), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 37), list_8741, False_8744)
        
        # Processing the call keyword arguments (line 408)
        kwargs_8745 = {}
        # Getting the type of 'assert_mask_matches' (line 408)
        assert_mask_matches_8739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'assert_mask_matches', False)
        # Calling assert_mask_matches(args, kwargs) (line 408)
        assert_mask_matches_call_result_8746 = invoke(stypy.reporting.localization.Localization(__file__, 408, 8), assert_mask_matches_8739, *[vardata_8740, list_8741], **kwargs_8745)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 406)
        exit___8747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 9), netcdf_file_call_result_8727, '__exit__')
        with_exit_8748 = invoke(stypy.reporting.localization.Localization(__file__, 406, 9), exit___8747, None, None, None)

    
    # ################# End of 'test_read_withValuesNearFillValue(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_withValuesNearFillValue' in the type store
    # Getting the type of 'stypy_return_type' (line 403)
    stypy_return_type_8749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8749)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_withValuesNearFillValue'
    return stypy_return_type_8749

# Assigning a type to the variable 'test_read_withValuesNearFillValue' (line 403)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 0), 'test_read_withValuesNearFillValue', test_read_withValuesNearFillValue)

@norecursion
def test_read_withNoFillValue(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_withNoFillValue'
    module_type_store = module_type_store.open_function_context('test_read_withNoFillValue', 410, 0, False)
    
    # Passed parameters checking function
    test_read_withNoFillValue.stypy_localization = localization
    test_read_withNoFillValue.stypy_type_of_self = None
    test_read_withNoFillValue.stypy_type_store = module_type_store
    test_read_withNoFillValue.stypy_function_name = 'test_read_withNoFillValue'
    test_read_withNoFillValue.stypy_param_names_list = []
    test_read_withNoFillValue.stypy_varargs_param_name = None
    test_read_withNoFillValue.stypy_kwargs_param_name = None
    test_read_withNoFillValue.stypy_call_defaults = defaults
    test_read_withNoFillValue.stypy_call_varargs = varargs
    test_read_withNoFillValue.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_withNoFillValue', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_withNoFillValue', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_withNoFillValue(...)' code ##################

    
    # Assigning a Call to a Name (line 413):
    
    # Call to pjoin(...): (line 413)
    # Processing the call arguments (line 413)
    # Getting the type of 'TEST_DATA_PATH' (line 413)
    TEST_DATA_PATH_8751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 18), 'TEST_DATA_PATH', False)
    str_8752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 34), 'str', 'example_3_maskedvals.nc')
    # Processing the call keyword arguments (line 413)
    kwargs_8753 = {}
    # Getting the type of 'pjoin' (line 413)
    pjoin_8750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 413)
    pjoin_call_result_8754 = invoke(stypy.reporting.localization.Localization(__file__, 413, 12), pjoin_8750, *[TEST_DATA_PATH_8751, str_8752], **kwargs_8753)
    
    # Assigning a type to the variable 'fname' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'fname', pjoin_call_result_8754)
    
    # Call to netcdf_file(...): (line 414)
    # Processing the call arguments (line 414)
    # Getting the type of 'fname' (line 414)
    fname_8756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 21), 'fname', False)
    # Processing the call keyword arguments (line 414)
    # Getting the type of 'True' (line 414)
    True_8757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 41), 'True', False)
    keyword_8758 = True_8757
    kwargs_8759 = {'maskandscale': keyword_8758}
    # Getting the type of 'netcdf_file' (line 414)
    netcdf_file_8755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 414)
    netcdf_file_call_result_8760 = invoke(stypy.reporting.localization.Localization(__file__, 414, 9), netcdf_file_8755, *[fname_8756], **kwargs_8759)
    
    with_8761 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 414, 9), netcdf_file_call_result_8760, 'with parameter', '__enter__', '__exit__')

    if with_8761:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 414)
        enter___8762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 9), netcdf_file_call_result_8760, '__enter__')
        with_enter_8763 = invoke(stypy.reporting.localization.Localization(__file__, 414, 9), enter___8762)
        # Assigning a type to the variable 'f' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 9), 'f', with_enter_8763)
        
        # Assigning a Subscript to a Name (line 415):
        
        # Obtaining the type of the subscript
        slice_8764 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 415, 18), None, None, None)
        
        # Obtaining the type of the subscript
        str_8765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 30), 'str', 'var2_noFillval')
        # Getting the type of 'f' (line 415)
        f_8766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 18), 'f')
        # Obtaining the member 'variables' of a type (line 415)
        variables_8767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 18), f_8766, 'variables')
        # Obtaining the member '__getitem__' of a type (line 415)
        getitem___8768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 18), variables_8767, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 415)
        subscript_call_result_8769 = invoke(stypy.reporting.localization.Localization(__file__, 415, 18), getitem___8768, str_8765)
        
        # Obtaining the member '__getitem__' of a type (line 415)
        getitem___8770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 18), subscript_call_result_8769, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 415)
        subscript_call_result_8771 = invoke(stypy.reporting.localization.Localization(__file__, 415, 18), getitem___8770, slice_8764)
        
        # Assigning a type to the variable 'vardata' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'vardata', subscript_call_result_8771)
        
        # Call to assert_mask_matches(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'vardata' (line 416)
        vardata_8773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 28), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 416)
        list_8774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 416)
        # Adding element type (line 416)
        # Getting the type of 'False' (line 416)
        False_8775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 38), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 37), list_8774, False_8775)
        # Adding element type (line 416)
        # Getting the type of 'False' (line 416)
        False_8776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 45), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 37), list_8774, False_8776)
        # Adding element type (line 416)
        # Getting the type of 'False' (line 416)
        False_8777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 52), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 37), list_8774, False_8777)
        
        # Processing the call keyword arguments (line 416)
        kwargs_8778 = {}
        # Getting the type of 'assert_mask_matches' (line 416)
        assert_mask_matches_8772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'assert_mask_matches', False)
        # Calling assert_mask_matches(args, kwargs) (line 416)
        assert_mask_matches_call_result_8779 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), assert_mask_matches_8772, *[vardata_8773, list_8774], **kwargs_8778)
        
        
        # Call to assert_equal(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'vardata' (line 417)
        vardata_8781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 21), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 417)
        list_8782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 417)
        # Adding element type (line 417)
        int_8783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 30), list_8782, int_8783)
        # Adding element type (line 417)
        int_8784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 30), list_8782, int_8784)
        # Adding element type (line 417)
        int_8785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 30), list_8782, int_8785)
        
        # Processing the call keyword arguments (line 417)
        kwargs_8786 = {}
        # Getting the type of 'assert_equal' (line 417)
        assert_equal_8780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 417)
        assert_equal_call_result_8787 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), assert_equal_8780, *[vardata_8781, list_8782], **kwargs_8786)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 414)
        exit___8788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 9), netcdf_file_call_result_8760, '__exit__')
        with_exit_8789 = invoke(stypy.reporting.localization.Localization(__file__, 414, 9), exit___8788, None, None, None)

    
    # ################# End of 'test_read_withNoFillValue(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_withNoFillValue' in the type store
    # Getting the type of 'stypy_return_type' (line 410)
    stypy_return_type_8790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8790)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_withNoFillValue'
    return stypy_return_type_8790

# Assigning a type to the variable 'test_read_withNoFillValue' (line 410)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'test_read_withNoFillValue', test_read_withNoFillValue)

@norecursion
def test_read_withFillValueAndMissingValue(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_withFillValueAndMissingValue'
    module_type_store = module_type_store.open_function_context('test_read_withFillValueAndMissingValue', 419, 0, False)
    
    # Passed parameters checking function
    test_read_withFillValueAndMissingValue.stypy_localization = localization
    test_read_withFillValueAndMissingValue.stypy_type_of_self = None
    test_read_withFillValueAndMissingValue.stypy_type_store = module_type_store
    test_read_withFillValueAndMissingValue.stypy_function_name = 'test_read_withFillValueAndMissingValue'
    test_read_withFillValueAndMissingValue.stypy_param_names_list = []
    test_read_withFillValueAndMissingValue.stypy_varargs_param_name = None
    test_read_withFillValueAndMissingValue.stypy_kwargs_param_name = None
    test_read_withFillValueAndMissingValue.stypy_call_defaults = defaults
    test_read_withFillValueAndMissingValue.stypy_call_varargs = varargs
    test_read_withFillValueAndMissingValue.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_withFillValueAndMissingValue', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_withFillValueAndMissingValue', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_withFillValueAndMissingValue(...)' code ##################

    
    # Assigning a Num to a Name (line 422):
    int_8791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 23), 'int')
    # Assigning a type to the variable 'IRRELEVANT_VALUE' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'IRRELEVANT_VALUE', int_8791)
    
    # Assigning a Call to a Name (line 423):
    
    # Call to pjoin(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'TEST_DATA_PATH' (line 423)
    TEST_DATA_PATH_8793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 18), 'TEST_DATA_PATH', False)
    str_8794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 34), 'str', 'example_3_maskedvals.nc')
    # Processing the call keyword arguments (line 423)
    kwargs_8795 = {}
    # Getting the type of 'pjoin' (line 423)
    pjoin_8792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 423)
    pjoin_call_result_8796 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), pjoin_8792, *[TEST_DATA_PATH_8793, str_8794], **kwargs_8795)
    
    # Assigning a type to the variable 'fname' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'fname', pjoin_call_result_8796)
    
    # Call to netcdf_file(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'fname' (line 424)
    fname_8798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 21), 'fname', False)
    # Processing the call keyword arguments (line 424)
    # Getting the type of 'True' (line 424)
    True_8799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 41), 'True', False)
    keyword_8800 = True_8799
    kwargs_8801 = {'maskandscale': keyword_8800}
    # Getting the type of 'netcdf_file' (line 424)
    netcdf_file_8797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 424)
    netcdf_file_call_result_8802 = invoke(stypy.reporting.localization.Localization(__file__, 424, 9), netcdf_file_8797, *[fname_8798], **kwargs_8801)
    
    with_8803 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 424, 9), netcdf_file_call_result_8802, 'with parameter', '__enter__', '__exit__')

    if with_8803:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 424)
        enter___8804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 9), netcdf_file_call_result_8802, '__enter__')
        with_enter_8805 = invoke(stypy.reporting.localization.Localization(__file__, 424, 9), enter___8804)
        # Assigning a type to the variable 'f' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 9), 'f', with_enter_8805)
        
        # Assigning a Subscript to a Name (line 425):
        
        # Obtaining the type of the subscript
        slice_8806 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 425, 18), None, None, None)
        
        # Obtaining the type of the subscript
        str_8807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 30), 'str', 'var3_fillvalAndMissingValue')
        # Getting the type of 'f' (line 425)
        f_8808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 18), 'f')
        # Obtaining the member 'variables' of a type (line 425)
        variables_8809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 18), f_8808, 'variables')
        # Obtaining the member '__getitem__' of a type (line 425)
        getitem___8810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 18), variables_8809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 425)
        subscript_call_result_8811 = invoke(stypy.reporting.localization.Localization(__file__, 425, 18), getitem___8810, str_8807)
        
        # Obtaining the member '__getitem__' of a type (line 425)
        getitem___8812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 18), subscript_call_result_8811, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 425)
        subscript_call_result_8813 = invoke(stypy.reporting.localization.Localization(__file__, 425, 18), getitem___8812, slice_8806)
        
        # Assigning a type to the variable 'vardata' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'vardata', subscript_call_result_8813)
        
        # Call to assert_mask_matches(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'vardata' (line 426)
        vardata_8815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 28), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 426)
        list_8816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 426)
        # Adding element type (line 426)
        # Getting the type of 'True' (line 426)
        True_8817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 38), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 37), list_8816, True_8817)
        # Adding element type (line 426)
        # Getting the type of 'False' (line 426)
        False_8818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 44), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 37), list_8816, False_8818)
        # Adding element type (line 426)
        # Getting the type of 'False' (line 426)
        False_8819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 51), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 37), list_8816, False_8819)
        
        # Processing the call keyword arguments (line 426)
        kwargs_8820 = {}
        # Getting the type of 'assert_mask_matches' (line 426)
        assert_mask_matches_8814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'assert_mask_matches', False)
        # Calling assert_mask_matches(args, kwargs) (line 426)
        assert_mask_matches_call_result_8821 = invoke(stypy.reporting.localization.Localization(__file__, 426, 8), assert_mask_matches_8814, *[vardata_8815, list_8816], **kwargs_8820)
        
        
        # Call to assert_equal(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'vardata' (line 427)
        vardata_8823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 21), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 427)
        list_8824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 427)
        # Adding element type (line 427)
        # Getting the type of 'IRRELEVANT_VALUE' (line 427)
        IRRELEVANT_VALUE_8825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 31), 'IRRELEVANT_VALUE', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 30), list_8824, IRRELEVANT_VALUE_8825)
        # Adding element type (line 427)
        int_8826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 30), list_8824, int_8826)
        # Adding element type (line 427)
        int_8827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 30), list_8824, int_8827)
        
        # Processing the call keyword arguments (line 427)
        kwargs_8828 = {}
        # Getting the type of 'assert_equal' (line 427)
        assert_equal_8822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 427)
        assert_equal_call_result_8829 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), assert_equal_8822, *[vardata_8823, list_8824], **kwargs_8828)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 424)
        exit___8830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 9), netcdf_file_call_result_8802, '__exit__')
        with_exit_8831 = invoke(stypy.reporting.localization.Localization(__file__, 424, 9), exit___8830, None, None, None)

    
    # ################# End of 'test_read_withFillValueAndMissingValue(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_withFillValueAndMissingValue' in the type store
    # Getting the type of 'stypy_return_type' (line 419)
    stypy_return_type_8832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8832)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_withFillValueAndMissingValue'
    return stypy_return_type_8832

# Assigning a type to the variable 'test_read_withFillValueAndMissingValue' (line 419)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 0), 'test_read_withFillValueAndMissingValue', test_read_withFillValueAndMissingValue)

@norecursion
def test_read_withMissingValue(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_withMissingValue'
    module_type_store = module_type_store.open_function_context('test_read_withMissingValue', 429, 0, False)
    
    # Passed parameters checking function
    test_read_withMissingValue.stypy_localization = localization
    test_read_withMissingValue.stypy_type_of_self = None
    test_read_withMissingValue.stypy_type_store = module_type_store
    test_read_withMissingValue.stypy_function_name = 'test_read_withMissingValue'
    test_read_withMissingValue.stypy_param_names_list = []
    test_read_withMissingValue.stypy_varargs_param_name = None
    test_read_withMissingValue.stypy_kwargs_param_name = None
    test_read_withMissingValue.stypy_call_defaults = defaults
    test_read_withMissingValue.stypy_call_varargs = varargs
    test_read_withMissingValue.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_withMissingValue', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_withMissingValue', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_withMissingValue(...)' code ##################

    
    # Assigning a Call to a Name (line 432):
    
    # Call to pjoin(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'TEST_DATA_PATH' (line 432)
    TEST_DATA_PATH_8834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 18), 'TEST_DATA_PATH', False)
    str_8835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 34), 'str', 'example_3_maskedvals.nc')
    # Processing the call keyword arguments (line 432)
    kwargs_8836 = {}
    # Getting the type of 'pjoin' (line 432)
    pjoin_8833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 432)
    pjoin_call_result_8837 = invoke(stypy.reporting.localization.Localization(__file__, 432, 12), pjoin_8833, *[TEST_DATA_PATH_8834, str_8835], **kwargs_8836)
    
    # Assigning a type to the variable 'fname' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'fname', pjoin_call_result_8837)
    
    # Call to netcdf_file(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'fname' (line 433)
    fname_8839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 21), 'fname', False)
    # Processing the call keyword arguments (line 433)
    # Getting the type of 'True' (line 433)
    True_8840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 41), 'True', False)
    keyword_8841 = True_8840
    kwargs_8842 = {'maskandscale': keyword_8841}
    # Getting the type of 'netcdf_file' (line 433)
    netcdf_file_8838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 433)
    netcdf_file_call_result_8843 = invoke(stypy.reporting.localization.Localization(__file__, 433, 9), netcdf_file_8838, *[fname_8839], **kwargs_8842)
    
    with_8844 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 433, 9), netcdf_file_call_result_8843, 'with parameter', '__enter__', '__exit__')

    if with_8844:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 433)
        enter___8845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 9), netcdf_file_call_result_8843, '__enter__')
        with_enter_8846 = invoke(stypy.reporting.localization.Localization(__file__, 433, 9), enter___8845)
        # Assigning a type to the variable 'f' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 9), 'f', with_enter_8846)
        
        # Assigning a Subscript to a Name (line 434):
        
        # Obtaining the type of the subscript
        slice_8847 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 434, 18), None, None, None)
        
        # Obtaining the type of the subscript
        str_8848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 30), 'str', 'var4_missingValue')
        # Getting the type of 'f' (line 434)
        f_8849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 18), 'f')
        # Obtaining the member 'variables' of a type (line 434)
        variables_8850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 18), f_8849, 'variables')
        # Obtaining the member '__getitem__' of a type (line 434)
        getitem___8851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 18), variables_8850, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
        subscript_call_result_8852 = invoke(stypy.reporting.localization.Localization(__file__, 434, 18), getitem___8851, str_8848)
        
        # Obtaining the member '__getitem__' of a type (line 434)
        getitem___8853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 18), subscript_call_result_8852, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
        subscript_call_result_8854 = invoke(stypy.reporting.localization.Localization(__file__, 434, 18), getitem___8853, slice_8847)
        
        # Assigning a type to the variable 'vardata' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'vardata', subscript_call_result_8854)
        
        # Call to assert_mask_matches(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'vardata' (line 435)
        vardata_8856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 28), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 435)
        list_8857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 435)
        # Adding element type (line 435)
        # Getting the type of 'False' (line 435)
        False_8858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 38), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 37), list_8857, False_8858)
        # Adding element type (line 435)
        # Getting the type of 'True' (line 435)
        True_8859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 45), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 37), list_8857, True_8859)
        # Adding element type (line 435)
        # Getting the type of 'False' (line 435)
        False_8860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 51), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 37), list_8857, False_8860)
        
        # Processing the call keyword arguments (line 435)
        kwargs_8861 = {}
        # Getting the type of 'assert_mask_matches' (line 435)
        assert_mask_matches_8855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'assert_mask_matches', False)
        # Calling assert_mask_matches(args, kwargs) (line 435)
        assert_mask_matches_call_result_8862 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), assert_mask_matches_8855, *[vardata_8856, list_8857], **kwargs_8861)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 433)
        exit___8863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 9), netcdf_file_call_result_8843, '__exit__')
        with_exit_8864 = invoke(stypy.reporting.localization.Localization(__file__, 433, 9), exit___8863, None, None, None)

    
    # ################# End of 'test_read_withMissingValue(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_withMissingValue' in the type store
    # Getting the type of 'stypy_return_type' (line 429)
    stypy_return_type_8865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8865)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_withMissingValue'
    return stypy_return_type_8865

# Assigning a type to the variable 'test_read_withMissingValue' (line 429)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 0), 'test_read_withMissingValue', test_read_withMissingValue)

@norecursion
def test_read_withFillValNaN(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_withFillValNaN'
    module_type_store = module_type_store.open_function_context('test_read_withFillValNaN', 437, 0, False)
    
    # Passed parameters checking function
    test_read_withFillValNaN.stypy_localization = localization
    test_read_withFillValNaN.stypy_type_of_self = None
    test_read_withFillValNaN.stypy_type_store = module_type_store
    test_read_withFillValNaN.stypy_function_name = 'test_read_withFillValNaN'
    test_read_withFillValNaN.stypy_param_names_list = []
    test_read_withFillValNaN.stypy_varargs_param_name = None
    test_read_withFillValNaN.stypy_kwargs_param_name = None
    test_read_withFillValNaN.stypy_call_defaults = defaults
    test_read_withFillValNaN.stypy_call_varargs = varargs
    test_read_withFillValNaN.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_withFillValNaN', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_withFillValNaN', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_withFillValNaN(...)' code ##################

    
    # Assigning a Call to a Name (line 438):
    
    # Call to pjoin(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'TEST_DATA_PATH' (line 438)
    TEST_DATA_PATH_8867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'TEST_DATA_PATH', False)
    str_8868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 34), 'str', 'example_3_maskedvals.nc')
    # Processing the call keyword arguments (line 438)
    kwargs_8869 = {}
    # Getting the type of 'pjoin' (line 438)
    pjoin_8866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 438)
    pjoin_call_result_8870 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), pjoin_8866, *[TEST_DATA_PATH_8867, str_8868], **kwargs_8869)
    
    # Assigning a type to the variable 'fname' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'fname', pjoin_call_result_8870)
    
    # Call to netcdf_file(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 'fname' (line 439)
    fname_8872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'fname', False)
    # Processing the call keyword arguments (line 439)
    # Getting the type of 'True' (line 439)
    True_8873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 41), 'True', False)
    keyword_8874 = True_8873
    kwargs_8875 = {'maskandscale': keyword_8874}
    # Getting the type of 'netcdf_file' (line 439)
    netcdf_file_8871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 439)
    netcdf_file_call_result_8876 = invoke(stypy.reporting.localization.Localization(__file__, 439, 9), netcdf_file_8871, *[fname_8872], **kwargs_8875)
    
    with_8877 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 439, 9), netcdf_file_call_result_8876, 'with parameter', '__enter__', '__exit__')

    if with_8877:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 439)
        enter___8878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 9), netcdf_file_call_result_8876, '__enter__')
        with_enter_8879 = invoke(stypy.reporting.localization.Localization(__file__, 439, 9), enter___8878)
        # Assigning a type to the variable 'f' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 9), 'f', with_enter_8879)
        
        # Assigning a Subscript to a Name (line 440):
        
        # Obtaining the type of the subscript
        slice_8880 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 440, 18), None, None, None)
        
        # Obtaining the type of the subscript
        str_8881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 30), 'str', 'var5_fillvalNaN')
        # Getting the type of 'f' (line 440)
        f_8882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 18), 'f')
        # Obtaining the member 'variables' of a type (line 440)
        variables_8883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 18), f_8882, 'variables')
        # Obtaining the member '__getitem__' of a type (line 440)
        getitem___8884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 18), variables_8883, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 440)
        subscript_call_result_8885 = invoke(stypy.reporting.localization.Localization(__file__, 440, 18), getitem___8884, str_8881)
        
        # Obtaining the member '__getitem__' of a type (line 440)
        getitem___8886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 18), subscript_call_result_8885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 440)
        subscript_call_result_8887 = invoke(stypy.reporting.localization.Localization(__file__, 440, 18), getitem___8886, slice_8880)
        
        # Assigning a type to the variable 'vardata' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'vardata', subscript_call_result_8887)
        
        # Call to assert_mask_matches(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'vardata' (line 441)
        vardata_8889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 28), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 441)
        list_8890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 441)
        # Adding element type (line 441)
        # Getting the type of 'False' (line 441)
        False_8891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 38), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 37), list_8890, False_8891)
        # Adding element type (line 441)
        # Getting the type of 'True' (line 441)
        True_8892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 45), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 37), list_8890, True_8892)
        # Adding element type (line 441)
        # Getting the type of 'False' (line 441)
        False_8893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 51), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 37), list_8890, False_8893)
        
        # Processing the call keyword arguments (line 441)
        kwargs_8894 = {}
        # Getting the type of 'assert_mask_matches' (line 441)
        assert_mask_matches_8888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'assert_mask_matches', False)
        # Calling assert_mask_matches(args, kwargs) (line 441)
        assert_mask_matches_call_result_8895 = invoke(stypy.reporting.localization.Localization(__file__, 441, 8), assert_mask_matches_8888, *[vardata_8889, list_8890], **kwargs_8894)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 439)
        exit___8896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 9), netcdf_file_call_result_8876, '__exit__')
        with_exit_8897 = invoke(stypy.reporting.localization.Localization(__file__, 439, 9), exit___8896, None, None, None)

    
    # ################# End of 'test_read_withFillValNaN(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_withFillValNaN' in the type store
    # Getting the type of 'stypy_return_type' (line 437)
    stypy_return_type_8898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8898)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_withFillValNaN'
    return stypy_return_type_8898

# Assigning a type to the variable 'test_read_withFillValNaN' (line 437)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 0), 'test_read_withFillValNaN', test_read_withFillValNaN)

@norecursion
def test_read_withChar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_withChar'
    module_type_store = module_type_store.open_function_context('test_read_withChar', 443, 0, False)
    
    # Passed parameters checking function
    test_read_withChar.stypy_localization = localization
    test_read_withChar.stypy_type_of_self = None
    test_read_withChar.stypy_type_store = module_type_store
    test_read_withChar.stypy_function_name = 'test_read_withChar'
    test_read_withChar.stypy_param_names_list = []
    test_read_withChar.stypy_varargs_param_name = None
    test_read_withChar.stypy_kwargs_param_name = None
    test_read_withChar.stypy_call_defaults = defaults
    test_read_withChar.stypy_call_varargs = varargs
    test_read_withChar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_withChar', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_withChar', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_withChar(...)' code ##################

    
    # Assigning a Call to a Name (line 444):
    
    # Call to pjoin(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'TEST_DATA_PATH' (line 444)
    TEST_DATA_PATH_8900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 18), 'TEST_DATA_PATH', False)
    str_8901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 34), 'str', 'example_3_maskedvals.nc')
    # Processing the call keyword arguments (line 444)
    kwargs_8902 = {}
    # Getting the type of 'pjoin' (line 444)
    pjoin_8899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 444)
    pjoin_call_result_8903 = invoke(stypy.reporting.localization.Localization(__file__, 444, 12), pjoin_8899, *[TEST_DATA_PATH_8900, str_8901], **kwargs_8902)
    
    # Assigning a type to the variable 'fname' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'fname', pjoin_call_result_8903)
    
    # Call to netcdf_file(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'fname' (line 445)
    fname_8905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'fname', False)
    # Processing the call keyword arguments (line 445)
    # Getting the type of 'True' (line 445)
    True_8906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 41), 'True', False)
    keyword_8907 = True_8906
    kwargs_8908 = {'maskandscale': keyword_8907}
    # Getting the type of 'netcdf_file' (line 445)
    netcdf_file_8904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 445)
    netcdf_file_call_result_8909 = invoke(stypy.reporting.localization.Localization(__file__, 445, 9), netcdf_file_8904, *[fname_8905], **kwargs_8908)
    
    with_8910 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 445, 9), netcdf_file_call_result_8909, 'with parameter', '__enter__', '__exit__')

    if with_8910:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 445)
        enter___8911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 9), netcdf_file_call_result_8909, '__enter__')
        with_enter_8912 = invoke(stypy.reporting.localization.Localization(__file__, 445, 9), enter___8911)
        # Assigning a type to the variable 'f' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 9), 'f', with_enter_8912)
        
        # Assigning a Subscript to a Name (line 446):
        
        # Obtaining the type of the subscript
        slice_8913 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 446, 18), None, None, None)
        
        # Obtaining the type of the subscript
        str_8914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 30), 'str', 'var6_char')
        # Getting the type of 'f' (line 446)
        f_8915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 18), 'f')
        # Obtaining the member 'variables' of a type (line 446)
        variables_8916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 18), f_8915, 'variables')
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___8917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 18), variables_8916, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_8918 = invoke(stypy.reporting.localization.Localization(__file__, 446, 18), getitem___8917, str_8914)
        
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___8919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 18), subscript_call_result_8918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_8920 = invoke(stypy.reporting.localization.Localization(__file__, 446, 18), getitem___8919, slice_8913)
        
        # Assigning a type to the variable 'vardata' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'vardata', subscript_call_result_8920)
        
        # Call to assert_mask_matches(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'vardata' (line 447)
        vardata_8922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 28), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 447)
        list_8923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 447)
        # Adding element type (line 447)
        # Getting the type of 'False' (line 447)
        False_8924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 38), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 37), list_8923, False_8924)
        # Adding element type (line 447)
        # Getting the type of 'True' (line 447)
        True_8925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 45), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 37), list_8923, True_8925)
        # Adding element type (line 447)
        # Getting the type of 'False' (line 447)
        False_8926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 51), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 37), list_8923, False_8926)
        
        # Processing the call keyword arguments (line 447)
        kwargs_8927 = {}
        # Getting the type of 'assert_mask_matches' (line 447)
        assert_mask_matches_8921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'assert_mask_matches', False)
        # Calling assert_mask_matches(args, kwargs) (line 447)
        assert_mask_matches_call_result_8928 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), assert_mask_matches_8921, *[vardata_8922, list_8923], **kwargs_8927)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 445)
        exit___8929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 9), netcdf_file_call_result_8909, '__exit__')
        with_exit_8930 = invoke(stypy.reporting.localization.Localization(__file__, 445, 9), exit___8929, None, None, None)

    
    # ################# End of 'test_read_withChar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_withChar' in the type store
    # Getting the type of 'stypy_return_type' (line 443)
    stypy_return_type_8931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8931)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_withChar'
    return stypy_return_type_8931

# Assigning a type to the variable 'test_read_withChar' (line 443)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 0), 'test_read_withChar', test_read_withChar)

@norecursion
def test_read_with2dVar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_with2dVar'
    module_type_store = module_type_store.open_function_context('test_read_with2dVar', 449, 0, False)
    
    # Passed parameters checking function
    test_read_with2dVar.stypy_localization = localization
    test_read_with2dVar.stypy_type_of_self = None
    test_read_with2dVar.stypy_type_store = module_type_store
    test_read_with2dVar.stypy_function_name = 'test_read_with2dVar'
    test_read_with2dVar.stypy_param_names_list = []
    test_read_with2dVar.stypy_varargs_param_name = None
    test_read_with2dVar.stypy_kwargs_param_name = None
    test_read_with2dVar.stypy_call_defaults = defaults
    test_read_with2dVar.stypy_call_varargs = varargs
    test_read_with2dVar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_with2dVar', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_with2dVar', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_with2dVar(...)' code ##################

    
    # Assigning a Call to a Name (line 450):
    
    # Call to pjoin(...): (line 450)
    # Processing the call arguments (line 450)
    # Getting the type of 'TEST_DATA_PATH' (line 450)
    TEST_DATA_PATH_8933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 18), 'TEST_DATA_PATH', False)
    str_8934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 34), 'str', 'example_3_maskedvals.nc')
    # Processing the call keyword arguments (line 450)
    kwargs_8935 = {}
    # Getting the type of 'pjoin' (line 450)
    pjoin_8932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 450)
    pjoin_call_result_8936 = invoke(stypy.reporting.localization.Localization(__file__, 450, 12), pjoin_8932, *[TEST_DATA_PATH_8933, str_8934], **kwargs_8935)
    
    # Assigning a type to the variable 'fname' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'fname', pjoin_call_result_8936)
    
    # Call to netcdf_file(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'fname' (line 451)
    fname_8938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 21), 'fname', False)
    # Processing the call keyword arguments (line 451)
    # Getting the type of 'True' (line 451)
    True_8939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 41), 'True', False)
    keyword_8940 = True_8939
    kwargs_8941 = {'maskandscale': keyword_8940}
    # Getting the type of 'netcdf_file' (line 451)
    netcdf_file_8937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 451)
    netcdf_file_call_result_8942 = invoke(stypy.reporting.localization.Localization(__file__, 451, 9), netcdf_file_8937, *[fname_8938], **kwargs_8941)
    
    with_8943 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 451, 9), netcdf_file_call_result_8942, 'with parameter', '__enter__', '__exit__')

    if with_8943:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 451)
        enter___8944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 9), netcdf_file_call_result_8942, '__enter__')
        with_enter_8945 = invoke(stypy.reporting.localization.Localization(__file__, 451, 9), enter___8944)
        # Assigning a type to the variable 'f' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 9), 'f', with_enter_8945)
        
        # Assigning a Subscript to a Name (line 452):
        
        # Obtaining the type of the subscript
        slice_8946 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 452, 18), None, None, None)
        
        # Obtaining the type of the subscript
        str_8947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 30), 'str', 'var7_2d')
        # Getting the type of 'f' (line 452)
        f_8948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 18), 'f')
        # Obtaining the member 'variables' of a type (line 452)
        variables_8949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 18), f_8948, 'variables')
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___8950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 18), variables_8949, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_8951 = invoke(stypy.reporting.localization.Localization(__file__, 452, 18), getitem___8950, str_8947)
        
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___8952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 18), subscript_call_result_8951, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_8953 = invoke(stypy.reporting.localization.Localization(__file__, 452, 18), getitem___8952, slice_8946)
        
        # Assigning a type to the variable 'vardata' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'vardata', subscript_call_result_8953)
        
        # Call to assert_mask_matches(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'vardata' (line 453)
        vardata_8955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 28), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 453)
        list_8956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 453)
        # Adding element type (line 453)
        
        # Obtaining an instance of the builtin type 'list' (line 453)
        list_8957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 453)
        # Adding element type (line 453)
        # Getting the type of 'True' (line 453)
        True_8958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 39), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 38), list_8957, True_8958)
        # Adding element type (line 453)
        # Getting the type of 'False' (line 453)
        False_8959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 45), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 38), list_8957, False_8959)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 37), list_8956, list_8957)
        # Adding element type (line 453)
        
        # Obtaining an instance of the builtin type 'list' (line 453)
        list_8960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 453)
        # Adding element type (line 453)
        # Getting the type of 'False' (line 453)
        False_8961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 54), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 53), list_8960, False_8961)
        # Adding element type (line 453)
        # Getting the type of 'False' (line 453)
        False_8962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 61), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 53), list_8960, False_8962)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 37), list_8956, list_8960)
        # Adding element type (line 453)
        
        # Obtaining an instance of the builtin type 'list' (line 453)
        list_8963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 453)
        # Adding element type (line 453)
        # Getting the type of 'False' (line 453)
        False_8964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 70), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 69), list_8963, False_8964)
        # Adding element type (line 453)
        # Getting the type of 'True' (line 453)
        True_8965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 77), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 69), list_8963, True_8965)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 37), list_8956, list_8963)
        
        # Processing the call keyword arguments (line 453)
        kwargs_8966 = {}
        # Getting the type of 'assert_mask_matches' (line 453)
        assert_mask_matches_8954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'assert_mask_matches', False)
        # Calling assert_mask_matches(args, kwargs) (line 453)
        assert_mask_matches_call_result_8967 = invoke(stypy.reporting.localization.Localization(__file__, 453, 8), assert_mask_matches_8954, *[vardata_8955, list_8956], **kwargs_8966)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 451)
        exit___8968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 9), netcdf_file_call_result_8942, '__exit__')
        with_exit_8969 = invoke(stypy.reporting.localization.Localization(__file__, 451, 9), exit___8968, None, None, None)

    
    # ################# End of 'test_read_with2dVar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_with2dVar' in the type store
    # Getting the type of 'stypy_return_type' (line 449)
    stypy_return_type_8970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8970)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_with2dVar'
    return stypy_return_type_8970

# Assigning a type to the variable 'test_read_with2dVar' (line 449)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 0), 'test_read_with2dVar', test_read_with2dVar)

@norecursion
def test_read_withMaskAndScaleFalse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_withMaskAndScaleFalse'
    module_type_store = module_type_store.open_function_context('test_read_withMaskAndScaleFalse', 455, 0, False)
    
    # Passed parameters checking function
    test_read_withMaskAndScaleFalse.stypy_localization = localization
    test_read_withMaskAndScaleFalse.stypy_type_of_self = None
    test_read_withMaskAndScaleFalse.stypy_type_store = module_type_store
    test_read_withMaskAndScaleFalse.stypy_function_name = 'test_read_withMaskAndScaleFalse'
    test_read_withMaskAndScaleFalse.stypy_param_names_list = []
    test_read_withMaskAndScaleFalse.stypy_varargs_param_name = None
    test_read_withMaskAndScaleFalse.stypy_kwargs_param_name = None
    test_read_withMaskAndScaleFalse.stypy_call_defaults = defaults
    test_read_withMaskAndScaleFalse.stypy_call_varargs = varargs
    test_read_withMaskAndScaleFalse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_withMaskAndScaleFalse', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_withMaskAndScaleFalse', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_withMaskAndScaleFalse(...)' code ##################

    
    # Assigning a Call to a Name (line 458):
    
    # Call to pjoin(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'TEST_DATA_PATH' (line 458)
    TEST_DATA_PATH_8972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 18), 'TEST_DATA_PATH', False)
    str_8973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 34), 'str', 'example_3_maskedvals.nc')
    # Processing the call keyword arguments (line 458)
    kwargs_8974 = {}
    # Getting the type of 'pjoin' (line 458)
    pjoin_8971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 458)
    pjoin_call_result_8975 = invoke(stypy.reporting.localization.Localization(__file__, 458, 12), pjoin_8971, *[TEST_DATA_PATH_8972, str_8973], **kwargs_8974)
    
    # Assigning a type to the variable 'fname' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'fname', pjoin_call_result_8975)
    
    # Call to netcdf_file(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'fname' (line 461)
    fname_8977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 21), 'fname', False)
    # Processing the call keyword arguments (line 461)
    # Getting the type of 'False' (line 461)
    False_8978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 41), 'False', False)
    keyword_8979 = False_8978
    # Getting the type of 'False' (line 461)
    False_8980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 53), 'False', False)
    keyword_8981 = False_8980
    kwargs_8982 = {'mmap': keyword_8981, 'maskandscale': keyword_8979}
    # Getting the type of 'netcdf_file' (line 461)
    netcdf_file_8976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 9), 'netcdf_file', False)
    # Calling netcdf_file(args, kwargs) (line 461)
    netcdf_file_call_result_8983 = invoke(stypy.reporting.localization.Localization(__file__, 461, 9), netcdf_file_8976, *[fname_8977], **kwargs_8982)
    
    with_8984 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 461, 9), netcdf_file_call_result_8983, 'with parameter', '__enter__', '__exit__')

    if with_8984:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 461)
        enter___8985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 9), netcdf_file_call_result_8983, '__enter__')
        with_enter_8986 = invoke(stypy.reporting.localization.Localization(__file__, 461, 9), enter___8985)
        # Assigning a type to the variable 'f' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 9), 'f', with_enter_8986)
        
        # Assigning a Subscript to a Name (line 462):
        
        # Obtaining the type of the subscript
        slice_8987 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 462, 18), None, None, None)
        
        # Obtaining the type of the subscript
        str_8988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 30), 'str', 'var3_fillvalAndMissingValue')
        # Getting the type of 'f' (line 462)
        f_8989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 18), 'f')
        # Obtaining the member 'variables' of a type (line 462)
        variables_8990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 18), f_8989, 'variables')
        # Obtaining the member '__getitem__' of a type (line 462)
        getitem___8991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 18), variables_8990, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 462)
        subscript_call_result_8992 = invoke(stypy.reporting.localization.Localization(__file__, 462, 18), getitem___8991, str_8988)
        
        # Obtaining the member '__getitem__' of a type (line 462)
        getitem___8993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 18), subscript_call_result_8992, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 462)
        subscript_call_result_8994 = invoke(stypy.reporting.localization.Localization(__file__, 462, 18), getitem___8993, slice_8987)
        
        # Assigning a type to the variable 'vardata' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'vardata', subscript_call_result_8994)
        
        # Call to assert_mask_matches(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'vardata' (line 463)
        vardata_8996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 28), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 463)
        list_8997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 463)
        # Adding element type (line 463)
        # Getting the type of 'False' (line 463)
        False_8998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 38), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 37), list_8997, False_8998)
        # Adding element type (line 463)
        # Getting the type of 'False' (line 463)
        False_8999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 45), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 37), list_8997, False_8999)
        # Adding element type (line 463)
        # Getting the type of 'False' (line 463)
        False_9000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 52), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 37), list_8997, False_9000)
        
        # Processing the call keyword arguments (line 463)
        kwargs_9001 = {}
        # Getting the type of 'assert_mask_matches' (line 463)
        assert_mask_matches_8995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'assert_mask_matches', False)
        # Calling assert_mask_matches(args, kwargs) (line 463)
        assert_mask_matches_call_result_9002 = invoke(stypy.reporting.localization.Localization(__file__, 463, 8), assert_mask_matches_8995, *[vardata_8996, list_8997], **kwargs_9001)
        
        
        # Call to assert_equal(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'vardata' (line 464)
        vardata_9004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 21), 'vardata', False)
        
        # Obtaining an instance of the builtin type 'list' (line 464)
        list_9005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 464)
        # Adding element type (line 464)
        int_9006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 30), list_9005, int_9006)
        # Adding element type (line 464)
        int_9007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 30), list_9005, int_9007)
        # Adding element type (line 464)
        int_9008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 30), list_9005, int_9008)
        
        # Processing the call keyword arguments (line 464)
        kwargs_9009 = {}
        # Getting the type of 'assert_equal' (line 464)
        assert_equal_9003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 464)
        assert_equal_call_result_9010 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), assert_equal_9003, *[vardata_9004, list_9005], **kwargs_9009)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 461)
        exit___9011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 9), netcdf_file_call_result_8983, '__exit__')
        with_exit_9012 = invoke(stypy.reporting.localization.Localization(__file__, 461, 9), exit___9011, None, None, None)

    
    # ################# End of 'test_read_withMaskAndScaleFalse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_withMaskAndScaleFalse' in the type store
    # Getting the type of 'stypy_return_type' (line 455)
    stypy_return_type_9013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9013)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_withMaskAndScaleFalse'
    return stypy_return_type_9013

# Assigning a type to the variable 'test_read_withMaskAndScaleFalse' (line 455)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 0), 'test_read_withMaskAndScaleFalse', test_read_withMaskAndScaleFalse)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
