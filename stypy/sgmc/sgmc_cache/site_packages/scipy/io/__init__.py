
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- encoding:utf-8 -*-
2: '''
3: ==================================
4: Input and output (:mod:`scipy.io`)
5: ==================================
6: 
7: .. currentmodule:: scipy.io
8: 
9: SciPy has many modules, classes, and functions available to read data
10: from and write data to a variety of file formats.
11: 
12: .. seealso:: :ref:`numpy-reference.routines.io` (in Numpy)
13: 
14: MATLAB® files
15: =============
16: 
17: .. autosummary::
18:    :toctree: generated/
19: 
20:    loadmat - Read a MATLAB style mat file (version 4 through 7.1)
21:    savemat - Write a MATLAB style mat file (version 4 through 7.1)
22:    whosmat - List contents of a MATLAB style mat file (version 4 through 7.1)
23: 
24: IDL® files
25: ==========
26: 
27: .. autosummary::
28:    :toctree: generated/
29: 
30:    readsav - Read an IDL 'save' file
31: 
32: Matrix Market files
33: ===================
34: 
35: .. autosummary::
36:    :toctree: generated/
37: 
38:    mminfo - Query matrix info from Matrix Market formatted file
39:    mmread - Read matrix from Matrix Market formatted file
40:    mmwrite - Write matrix to Matrix Market formatted file
41: 
42: Unformatted Fortran files
43: ===============================
44: 
45: .. autosummary::
46:    :toctree: generated/
47: 
48:    FortranFile - A file object for unformatted sequential Fortran files
49: 
50: Netcdf
51: ======
52: 
53: .. autosummary::
54:    :toctree: generated/
55: 
56:    netcdf_file - A file object for NetCDF data
57:    netcdf_variable - A data object for the netcdf module
58: 
59: Harwell-Boeing files
60: ====================
61: 
62: .. autosummary::
63:    :toctree: generated/
64: 
65:    hb_read   -- read H-B file
66:    hb_write  -- write H-B file
67: 
68: Wav sound files (:mod:`scipy.io.wavfile`)
69: =========================================
70: 
71: .. module:: scipy.io.wavfile
72: 
73: .. autosummary::
74:    :toctree: generated/
75: 
76:    read
77:    write
78:    WavFileWarning
79: 
80: Arff files (:mod:`scipy.io.arff`)
81: =================================
82: 
83: .. module:: scipy.io.arff
84: 
85: .. autosummary::
86:    :toctree: generated/
87: 
88:    loadarff
89:    MetaData
90:    ArffError
91:    ParseArffError
92: 
93: '''
94: from __future__ import division, print_function, absolute_import
95: 
96: # matfile read and write
97: from .matlab import loadmat, savemat, whosmat, byteordercodes
98: 
99: # netCDF file support
100: from .netcdf import netcdf_file, netcdf_variable
101: 
102: # Fortran file support
103: from ._fortran import FortranFile
104: 
105: from .mmio import mminfo, mmread, mmwrite
106: from .idl import readsav
107: from .harwell_boeing import hb_read, hb_write
108: 
109: __all__ = [s for s in dir() if not s.startswith('_')]
110: 
111: from scipy._lib._testutils import PytestTester
112: test = PytestTester(__name__)
113: del PytestTester
114: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_128212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'str', "\n==================================\nInput and output (:mod:`scipy.io`)\n==================================\n\n.. currentmodule:: scipy.io\n\nSciPy has many modules, classes, and functions available to read data\nfrom and write data to a variety of file formats.\n\n.. seealso:: :ref:`numpy-reference.routines.io` (in Numpy)\n\nMATLAB\xc2\xae files\n=============\n\n.. autosummary::\n   :toctree: generated/\n\n   loadmat - Read a MATLAB style mat file (version 4 through 7.1)\n   savemat - Write a MATLAB style mat file (version 4 through 7.1)\n   whosmat - List contents of a MATLAB style mat file (version 4 through 7.1)\n\nIDL\xc2\xae files\n==========\n\n.. autosummary::\n   :toctree: generated/\n\n   readsav - Read an IDL 'save' file\n\nMatrix Market files\n===================\n\n.. autosummary::\n   :toctree: generated/\n\n   mminfo - Query matrix info from Matrix Market formatted file\n   mmread - Read matrix from Matrix Market formatted file\n   mmwrite - Write matrix to Matrix Market formatted file\n\nUnformatted Fortran files\n===============================\n\n.. autosummary::\n   :toctree: generated/\n\n   FortranFile - A file object for unformatted sequential Fortran files\n\nNetcdf\n======\n\n.. autosummary::\n   :toctree: generated/\n\n   netcdf_file - A file object for NetCDF data\n   netcdf_variable - A data object for the netcdf module\n\nHarwell-Boeing files\n====================\n\n.. autosummary::\n   :toctree: generated/\n\n   hb_read   -- read H-B file\n   hb_write  -- write H-B file\n\nWav sound files (:mod:`scipy.io.wavfile`)\n=========================================\n\n.. module:: scipy.io.wavfile\n\n.. autosummary::\n   :toctree: generated/\n\n   read\n   write\n   WavFileWarning\n\nArff files (:mod:`scipy.io.arff`)\n=================================\n\n.. module:: scipy.io.arff\n\n.. autosummary::\n   :toctree: generated/\n\n   loadarff\n   MetaData\n   ArffError\n   ParseArffError\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 97, 0))

# 'from scipy.io.matlab import loadmat, savemat, whosmat, byteordercodes' statement (line 97)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_128213 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.io.matlab')

if (type(import_128213) is not StypyTypeError):

    if (import_128213 != 'pyd_module'):
        __import__(import_128213)
        sys_modules_128214 = sys.modules[import_128213]
        import_from_module(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.io.matlab', sys_modules_128214.module_type_store, module_type_store, ['loadmat', 'savemat', 'whosmat', 'byteordercodes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 97, 0), __file__, sys_modules_128214, sys_modules_128214.module_type_store, module_type_store)
    else:
        from scipy.io.matlab import loadmat, savemat, whosmat, byteordercodes

        import_from_module(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.io.matlab', None, module_type_store, ['loadmat', 'savemat', 'whosmat', 'byteordercodes'], [loadmat, savemat, whosmat, byteordercodes])

else:
    # Assigning a type to the variable 'scipy.io.matlab' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.io.matlab', import_128213)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 100, 0))

# 'from scipy.io.netcdf import netcdf_file, netcdf_variable' statement (line 100)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_128215 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.io.netcdf')

if (type(import_128215) is not StypyTypeError):

    if (import_128215 != 'pyd_module'):
        __import__(import_128215)
        sys_modules_128216 = sys.modules[import_128215]
        import_from_module(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.io.netcdf', sys_modules_128216.module_type_store, module_type_store, ['netcdf_file', 'netcdf_variable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 100, 0), __file__, sys_modules_128216, sys_modules_128216.module_type_store, module_type_store)
    else:
        from scipy.io.netcdf import netcdf_file, netcdf_variable

        import_from_module(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.io.netcdf', None, module_type_store, ['netcdf_file', 'netcdf_variable'], [netcdf_file, netcdf_variable])

else:
    # Assigning a type to the variable 'scipy.io.netcdf' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.io.netcdf', import_128215)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 103, 0))

# 'from scipy.io._fortran import FortranFile' statement (line 103)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_128217 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy.io._fortran')

if (type(import_128217) is not StypyTypeError):

    if (import_128217 != 'pyd_module'):
        __import__(import_128217)
        sys_modules_128218 = sys.modules[import_128217]
        import_from_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy.io._fortran', sys_modules_128218.module_type_store, module_type_store, ['FortranFile'])
        nest_module(stypy.reporting.localization.Localization(__file__, 103, 0), __file__, sys_modules_128218, sys_modules_128218.module_type_store, module_type_store)
    else:
        from scipy.io._fortran import FortranFile

        import_from_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy.io._fortran', None, module_type_store, ['FortranFile'], [FortranFile])

else:
    # Assigning a type to the variable 'scipy.io._fortran' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'scipy.io._fortran', import_128217)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 105, 0))

# 'from scipy.io.mmio import mminfo, mmread, mmwrite' statement (line 105)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_128219 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy.io.mmio')

if (type(import_128219) is not StypyTypeError):

    if (import_128219 != 'pyd_module'):
        __import__(import_128219)
        sys_modules_128220 = sys.modules[import_128219]
        import_from_module(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy.io.mmio', sys_modules_128220.module_type_store, module_type_store, ['mminfo', 'mmread', 'mmwrite'])
        nest_module(stypy.reporting.localization.Localization(__file__, 105, 0), __file__, sys_modules_128220, sys_modules_128220.module_type_store, module_type_store)
    else:
        from scipy.io.mmio import mminfo, mmread, mmwrite

        import_from_module(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy.io.mmio', None, module_type_store, ['mminfo', 'mmread', 'mmwrite'], [mminfo, mmread, mmwrite])

else:
    # Assigning a type to the variable 'scipy.io.mmio' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'scipy.io.mmio', import_128219)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 106, 0))

# 'from scipy.io.idl import readsav' statement (line 106)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_128221 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 106, 0), 'scipy.io.idl')

if (type(import_128221) is not StypyTypeError):

    if (import_128221 != 'pyd_module'):
        __import__(import_128221)
        sys_modules_128222 = sys.modules[import_128221]
        import_from_module(stypy.reporting.localization.Localization(__file__, 106, 0), 'scipy.io.idl', sys_modules_128222.module_type_store, module_type_store, ['readsav'])
        nest_module(stypy.reporting.localization.Localization(__file__, 106, 0), __file__, sys_modules_128222, sys_modules_128222.module_type_store, module_type_store)
    else:
        from scipy.io.idl import readsav

        import_from_module(stypy.reporting.localization.Localization(__file__, 106, 0), 'scipy.io.idl', None, module_type_store, ['readsav'], [readsav])

else:
    # Assigning a type to the variable 'scipy.io.idl' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'scipy.io.idl', import_128221)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 107, 0))

# 'from scipy.io.harwell_boeing import hb_read, hb_write' statement (line 107)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_128223 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 107, 0), 'scipy.io.harwell_boeing')

if (type(import_128223) is not StypyTypeError):

    if (import_128223 != 'pyd_module'):
        __import__(import_128223)
        sys_modules_128224 = sys.modules[import_128223]
        import_from_module(stypy.reporting.localization.Localization(__file__, 107, 0), 'scipy.io.harwell_boeing', sys_modules_128224.module_type_store, module_type_store, ['hb_read', 'hb_write'])
        nest_module(stypy.reporting.localization.Localization(__file__, 107, 0), __file__, sys_modules_128224, sys_modules_128224.module_type_store, module_type_store)
    else:
        from scipy.io.harwell_boeing import hb_read, hb_write

        import_from_module(stypy.reporting.localization.Localization(__file__, 107, 0), 'scipy.io.harwell_boeing', None, module_type_store, ['hb_read', 'hb_write'], [hb_read, hb_write])

else:
    # Assigning a type to the variable 'scipy.io.harwell_boeing' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'scipy.io.harwell_boeing', import_128223)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')


# Assigning a ListComp to a Name (line 109):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 109)
# Processing the call keyword arguments (line 109)
kwargs_128233 = {}
# Getting the type of 'dir' (line 109)
dir_128232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 22), 'dir', False)
# Calling dir(args, kwargs) (line 109)
dir_call_result_128234 = invoke(stypy.reporting.localization.Localization(__file__, 109, 22), dir_128232, *[], **kwargs_128233)

comprehension_128235 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 11), dir_call_result_128234)
# Assigning a type to the variable 's' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 's', comprehension_128235)


# Call to startswith(...): (line 109)
# Processing the call arguments (line 109)
str_128228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 48), 'str', '_')
# Processing the call keyword arguments (line 109)
kwargs_128229 = {}
# Getting the type of 's' (line 109)
s_128226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 109)
startswith_128227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 35), s_128226, 'startswith')
# Calling startswith(args, kwargs) (line 109)
startswith_call_result_128230 = invoke(stypy.reporting.localization.Localization(__file__, 109, 35), startswith_128227, *[str_128228], **kwargs_128229)

# Applying the 'not' unary operator (line 109)
result_not__128231 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 31), 'not', startswith_call_result_128230)

# Getting the type of 's' (line 109)
s_128225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 's')
list_128236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 11), list_128236, s_128225)
# Assigning a type to the variable '__all__' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), '__all__', list_128236)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 111, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 111)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_128237 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 111, 0), 'scipy._lib._testutils')

if (type(import_128237) is not StypyTypeError):

    if (import_128237 != 'pyd_module'):
        __import__(import_128237)
        sys_modules_128238 = sys.modules[import_128237]
        import_from_module(stypy.reporting.localization.Localization(__file__, 111, 0), 'scipy._lib._testutils', sys_modules_128238.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 111, 0), __file__, sys_modules_128238, sys_modules_128238.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 111, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'scipy._lib._testutils', import_128237)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')


# Assigning a Call to a Name (line 112):

# Call to PytestTester(...): (line 112)
# Processing the call arguments (line 112)
# Getting the type of '__name__' (line 112)
name___128240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), '__name__', False)
# Processing the call keyword arguments (line 112)
kwargs_128241 = {}
# Getting the type of 'PytestTester' (line 112)
PytestTester_128239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 112)
PytestTester_call_result_128242 = invoke(stypy.reporting.localization.Localization(__file__, 112, 7), PytestTester_128239, *[name___128240], **kwargs_128241)

# Assigning a type to the variable 'test' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'test', PytestTester_call_result_128242)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 113, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
