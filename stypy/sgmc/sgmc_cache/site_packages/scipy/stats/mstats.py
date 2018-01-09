
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ===================================================================
3: Statistical functions for masked arrays (:mod:`scipy.stats.mstats`)
4: ===================================================================
5: 
6: .. currentmodule:: scipy.stats.mstats
7: 
8: This module contains a large number of statistical functions that can
9: be used with masked arrays.
10: 
11: Most of these functions are similar to those in scipy.stats but might
12: have small differences in the API or in the algorithm used. Since this
13: is a relatively new package, some API changes are still possible.
14: 
15: .. autosummary::
16:    :toctree: generated/
17: 
18:    argstoarray
19:    chisquare
20:    count_tied_groups
21:    describe
22:    f_oneway
23:    find_repeats
24:    friedmanchisquare
25:    kendalltau
26:    kendalltau_seasonal
27:    kruskalwallis
28:    ks_twosamp
29:    kurtosis
30:    kurtosistest
31:    linregress
32:    mannwhitneyu
33:    plotting_positions
34:    mode
35:    moment
36:    mquantiles
37:    msign
38:    normaltest
39:    obrientransform
40:    pearsonr
41:    plotting_positions
42:    pointbiserialr
43:    rankdata
44:    scoreatpercentile
45:    sem
46:    skew
47:    skewtest
48:    spearmanr
49:    theilslopes
50:    tmax
51:    tmean
52:    tmin
53:    trim
54:    trima
55:    trimboth
56:    trimmed_stde
57:    trimr
58:    trimtail
59:    tsem
60:    ttest_onesamp
61:    ttest_ind
62:    ttest_onesamp
63:    ttest_rel
64:    tvar
65:    variation
66:    winsorize
67:    zmap
68:    zscore
69:    compare_medians_ms
70:    gmean
71:    hdmedian
72:    hdquantiles
73:    hdquantiles_sd
74:    hmean
75:    idealfourths
76:    kruskal
77:    ks_2samp
78:    median_cihs
79:    meppf
80:    mjci
81:    mquantiles_cimj
82:    rsh
83:    sen_seasonal_slopes
84:    trimmed_mean
85:    trimmed_mean_ci
86:    trimmed_std
87:    trimmed_var
88:    ttest_1samp
89: 
90: '''
91: from __future__ import division, print_function, absolute_import
92: 
93: from .mstats_basic import *
94: from .mstats_extras import *
95: # Functions that support masked array input in stats but need to be kept in the
96: # mstats namespace for backwards compatibility:
97: from scipy.stats import gmean, hmean, zmap, zscore, chisquare
98: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_571417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'str', '\n===================================================================\nStatistical functions for masked arrays (:mod:`scipy.stats.mstats`)\n===================================================================\n\n.. currentmodule:: scipy.stats.mstats\n\nThis module contains a large number of statistical functions that can\nbe used with masked arrays.\n\nMost of these functions are similar to those in scipy.stats but might\nhave small differences in the API or in the algorithm used. Since this\nis a relatively new package, some API changes are still possible.\n\n.. autosummary::\n   :toctree: generated/\n\n   argstoarray\n   chisquare\n   count_tied_groups\n   describe\n   f_oneway\n   find_repeats\n   friedmanchisquare\n   kendalltau\n   kendalltau_seasonal\n   kruskalwallis\n   ks_twosamp\n   kurtosis\n   kurtosistest\n   linregress\n   mannwhitneyu\n   plotting_positions\n   mode\n   moment\n   mquantiles\n   msign\n   normaltest\n   obrientransform\n   pearsonr\n   plotting_positions\n   pointbiserialr\n   rankdata\n   scoreatpercentile\n   sem\n   skew\n   skewtest\n   spearmanr\n   theilslopes\n   tmax\n   tmean\n   tmin\n   trim\n   trima\n   trimboth\n   trimmed_stde\n   trimr\n   trimtail\n   tsem\n   ttest_onesamp\n   ttest_ind\n   ttest_onesamp\n   ttest_rel\n   tvar\n   variation\n   winsorize\n   zmap\n   zscore\n   compare_medians_ms\n   gmean\n   hdmedian\n   hdquantiles\n   hdquantiles_sd\n   hmean\n   idealfourths\n   kruskal\n   ks_2samp\n   median_cihs\n   meppf\n   mjci\n   mquantiles_cimj\n   rsh\n   sen_seasonal_slopes\n   trimmed_mean\n   trimmed_mean_ci\n   trimmed_std\n   trimmed_var\n   ttest_1samp\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 93, 0))

# 'from scipy.stats.mstats_basic import ' statement (line 93)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_571418 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.stats.mstats_basic')

if (type(import_571418) is not StypyTypeError):

    if (import_571418 != 'pyd_module'):
        __import__(import_571418)
        sys_modules_571419 = sys.modules[import_571418]
        import_from_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.stats.mstats_basic', sys_modules_571419.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 93, 0), __file__, sys_modules_571419, sys_modules_571419.module_type_store, module_type_store)
    else:
        from scipy.stats.mstats_basic import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.stats.mstats_basic', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.stats.mstats_basic' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.stats.mstats_basic', import_571418)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 94, 0))

# 'from scipy.stats.mstats_extras import ' statement (line 94)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_571420 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.stats.mstats_extras')

if (type(import_571420) is not StypyTypeError):

    if (import_571420 != 'pyd_module'):
        __import__(import_571420)
        sys_modules_571421 = sys.modules[import_571420]
        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.stats.mstats_extras', sys_modules_571421.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 94, 0), __file__, sys_modules_571421, sys_modules_571421.module_type_store, module_type_store)
    else:
        from scipy.stats.mstats_extras import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.stats.mstats_extras', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.stats.mstats_extras' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.stats.mstats_extras', import_571420)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 97, 0))

# 'from scipy.stats import gmean, hmean, zmap, zscore, chisquare' statement (line 97)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_571422 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.stats')

if (type(import_571422) is not StypyTypeError):

    if (import_571422 != 'pyd_module'):
        __import__(import_571422)
        sys_modules_571423 = sys.modules[import_571422]
        import_from_module(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.stats', sys_modules_571423.module_type_store, module_type_store, ['gmean', 'hmean', 'zmap', 'zscore', 'chisquare'])
        nest_module(stypy.reporting.localization.Localization(__file__, 97, 0), __file__, sys_modules_571423, sys_modules_571423.module_type_store, module_type_store)
    else:
        from scipy.stats import gmean, hmean, zmap, zscore, chisquare

        import_from_module(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.stats', None, module_type_store, ['gmean', 'hmean', 'zmap', 'zscore', 'chisquare'], [gmean, hmean, zmap, zscore, chisquare])

else:
    # Assigning a type to the variable 'scipy.stats' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'scipy.stats', import_571422)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
