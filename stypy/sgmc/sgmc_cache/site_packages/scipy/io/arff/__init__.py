
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Module to read ARFF files, which are the standard data format for WEKA.
3: 
4: ARFF is a text file format which support numerical, string and data values.
5: The format can also represent missing data and sparse data.
6: 
7: Notes
8: -----
9: The ARFF support in ``scipy.io`` provides file reading functionality only.
10: For more extensive ARFF functionality, see `liac-arff
11: <https://github.com/renatopp/liac-arff>`_.
12: 
13: See the `WEKA website <http://weka.wikispaces.com/ARFF>`_
14: for more details about the ARFF format and available datasets.
15: 
16: '''
17: from __future__ import division, print_function, absolute_import
18: 
19: from .arffread import *
20: from . import arffread
21: 
22: __all__ = arffread.__all__
23: 
24: from scipy._lib._testutils import PytestTester
25: test = PytestTester(__name__)
26: del PytestTester
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_129629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, (-1)), 'str', '\nModule to read ARFF files, which are the standard data format for WEKA.\n\nARFF is a text file format which support numerical, string and data values.\nThe format can also represent missing data and sparse data.\n\nNotes\n-----\nThe ARFF support in ``scipy.io`` provides file reading functionality only.\nFor more extensive ARFF functionality, see `liac-arff\n<https://github.com/renatopp/liac-arff>`_.\n\nSee the `WEKA website <http://weka.wikispaces.com/ARFF>`_\nfor more details about the ARFF format and available datasets.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.io.arff.arffread import ' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/')
import_129630 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.io.arff.arffread')

if (type(import_129630) is not StypyTypeError):

    if (import_129630 != 'pyd_module'):
        __import__(import_129630)
        sys_modules_129631 = sys.modules[import_129630]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.io.arff.arffread', sys_modules_129631.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_129631, sys_modules_129631.module_type_store, module_type_store)
    else:
        from scipy.io.arff.arffread import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.io.arff.arffread', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.io.arff.arffread' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.io.arff.arffread', import_129630)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.io.arff import arffread' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/')
import_129632 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.io.arff')

if (type(import_129632) is not StypyTypeError):

    if (import_129632 != 'pyd_module'):
        __import__(import_129632)
        sys_modules_129633 = sys.modules[import_129632]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.io.arff', sys_modules_129633.module_type_store, module_type_store, ['arffread'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_129633, sys_modules_129633.module_type_store, module_type_store)
    else:
        from scipy.io.arff import arffread

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.io.arff', None, module_type_store, ['arffread'], [arffread])

else:
    # Assigning a type to the variable 'scipy.io.arff' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.io.arff', import_129632)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/')


# Assigning a Attribute to a Name (line 22):
# Getting the type of 'arffread' (line 22)
arffread_129634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'arffread')
# Obtaining the member '__all__' of a type (line 22)
all___129635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 10), arffread_129634, '__all__')
# Assigning a type to the variable '__all__' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '__all__', all___129635)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/')
import_129636 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy._lib._testutils')

if (type(import_129636) is not StypyTypeError):

    if (import_129636 != 'pyd_module'):
        __import__(import_129636)
        sys_modules_129637 = sys.modules[import_129636]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy._lib._testutils', sys_modules_129637.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_129637, sys_modules_129637.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy._lib._testutils', import_129636)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/')


# Assigning a Call to a Name (line 25):

# Call to PytestTester(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of '__name__' (line 25)
name___129639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), '__name__', False)
# Processing the call keyword arguments (line 25)
kwargs_129640 = {}
# Getting the type of 'PytestTester' (line 25)
PytestTester_129638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 25)
PytestTester_call_result_129641 = invoke(stypy.reporting.localization.Localization(__file__, 25, 7), PytestTester_129638, *[name___129639], **kwargs_129640)

# Assigning a type to the variable 'test' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'test', PytestTester_call_result_129641)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 26, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
