
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =========================================
3: Clustering package (:mod:`scipy.cluster`)
4: =========================================
5: 
6: .. currentmodule:: scipy.cluster
7: 
8: :mod:`scipy.cluster.vq`
9: 
10: Clustering algorithms are useful in information theory, target detection,
11: communications, compression, and other areas.  The `vq` module only
12: supports vector quantization and the k-means algorithms.
13: 
14: :mod:`scipy.cluster.hierarchy`
15: 
16: The `hierarchy` module provides functions for hierarchical and
17: agglomerative clustering.  Its features include generating hierarchical
18: clusters from distance matrices,
19: calculating statistics on clusters, cutting linkages
20: to generate flat clusters, and visualizing clusters with dendrograms.
21: 
22: '''
23: from __future__ import division, print_function, absolute_import
24: 
25: __all__ = ['vq', 'hierarchy']
26: 
27: from . import vq, hierarchy
28: 
29: from scipy._lib._testutils import PytestTester
30: test = PytestTester(__name__)
31: del PytestTester
32: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_6566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n=========================================\nClustering package (:mod:`scipy.cluster`)\n=========================================\n\n.. currentmodule:: scipy.cluster\n\n:mod:`scipy.cluster.vq`\n\nClustering algorithms are useful in information theory, target detection,\ncommunications, compression, and other areas.  The `vq` module only\nsupports vector quantization and the k-means algorithms.\n\n:mod:`scipy.cluster.hierarchy`\n\nThe `hierarchy` module provides functions for hierarchical and\nagglomerative clustering.  Its features include generating hierarchical\nclusters from distance matrices,\ncalculating statistics on clusters, cutting linkages\nto generate flat clusters, and visualizing clusters with dendrograms.\n\n')

# Assigning a List to a Name (line 25):
__all__ = ['vq', 'hierarchy']
module_type_store.set_exportable_members(['vq', 'hierarchy'])

# Obtaining an instance of the builtin type 'list' (line 25)
list_6567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
str_6568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'str', 'vq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_6567, str_6568)
# Adding element type (line 25)
str_6569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'str', 'hierarchy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_6567, str_6569)

# Assigning a type to the variable '__all__' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '__all__', list_6567)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from scipy.cluster import vq, hierarchy' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/')
import_6570 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.cluster')

if (type(import_6570) is not StypyTypeError):

    if (import_6570 != 'pyd_module'):
        __import__(import_6570)
        sys_modules_6571 = sys.modules[import_6570]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.cluster', sys_modules_6571.module_type_store, module_type_store, ['vq', 'hierarchy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_6571, sys_modules_6571.module_type_store, module_type_store)
    else:
        from scipy.cluster import vq, hierarchy

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.cluster', None, module_type_store, ['vq', 'hierarchy'], [vq, hierarchy])

else:
    # Assigning a type to the variable 'scipy.cluster' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.cluster', import_6570)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/')
import_6572 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy._lib._testutils')

if (type(import_6572) is not StypyTypeError):

    if (import_6572 != 'pyd_module'):
        __import__(import_6572)
        sys_modules_6573 = sys.modules[import_6572]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy._lib._testutils', sys_modules_6573.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_6573, sys_modules_6573.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy._lib._testutils', import_6572)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/')


# Assigning a Call to a Name (line 30):

# Call to PytestTester(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of '__name__' (line 30)
name___6575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), '__name__', False)
# Processing the call keyword arguments (line 30)
kwargs_6576 = {}
# Getting the type of 'PytestTester' (line 30)
PytestTester_6574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 30)
PytestTester_call_result_6577 = invoke(stypy.reporting.localization.Localization(__file__, 30, 7), PytestTester_6574, *[name___6575], **kwargs_6576)

# Assigning a type to the variable 'test' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'test', PytestTester_call_result_6577)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 31, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
