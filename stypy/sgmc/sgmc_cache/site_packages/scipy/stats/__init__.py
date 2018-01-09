
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ==========================================
3: Statistical functions (:mod:`scipy.stats`)
4: ==========================================
5: 
6: .. module:: scipy.stats
7: 
8: This module contains a large number of probability distributions as
9: well as a growing library of statistical functions.
10: 
11: Each univariate distribution is an instance of a subclass of `rv_continuous`
12: (`rv_discrete` for discrete distributions):
13: 
14: .. autosummary::
15:    :toctree: generated/
16: 
17:    rv_continuous
18:    rv_discrete
19:    rv_histogram
20: 
21: Continuous distributions
22: ========================
23: 
24: .. autosummary::
25:    :toctree: generated/
26: 
27:    alpha             -- Alpha
28:    anglit            -- Anglit
29:    arcsine           -- Arcsine
30:    argus             -- Argus
31:    beta              -- Beta
32:    betaprime         -- Beta Prime
33:    bradford          -- Bradford
34:    burr              -- Burr (Type III)
35:    burr12            -- Burr (Type XII)
36:    cauchy            -- Cauchy
37:    chi               -- Chi
38:    chi2              -- Chi-squared
39:    cosine            -- Cosine
40:    crystalball       -- Crystalball
41:    dgamma            -- Double Gamma
42:    dweibull          -- Double Weibull
43:    erlang            -- Erlang
44:    expon             -- Exponential
45:    exponnorm         -- Exponentially Modified Normal
46:    exponweib         -- Exponentiated Weibull
47:    exponpow          -- Exponential Power
48:    f                 -- F (Snecdor F)
49:    fatiguelife       -- Fatigue Life (Birnbaum-Saunders)
50:    fisk              -- Fisk
51:    foldcauchy        -- Folded Cauchy
52:    foldnorm          -- Folded Normal
53:    frechet_r         -- Deprecated. Alias for weibull_min
54:    frechet_l         -- Deprecated. Alias for weibull_max
55:    genlogistic       -- Generalized Logistic
56:    gennorm           -- Generalized normal
57:    genpareto         -- Generalized Pareto
58:    genexpon          -- Generalized Exponential
59:    genextreme        -- Generalized Extreme Value
60:    gausshyper        -- Gauss Hypergeometric
61:    gamma             -- Gamma
62:    gengamma          -- Generalized gamma
63:    genhalflogistic   -- Generalized Half Logistic
64:    gilbrat           -- Gilbrat
65:    gompertz          -- Gompertz (Truncated Gumbel)
66:    gumbel_r          -- Right Sided Gumbel, Log-Weibull, Fisher-Tippett, Extreme Value Type I
67:    gumbel_l          -- Left Sided Gumbel, etc.
68:    halfcauchy        -- Half Cauchy
69:    halflogistic      -- Half Logistic
70:    halfnorm          -- Half Normal
71:    halfgennorm       -- Generalized Half Normal
72:    hypsecant         -- Hyperbolic Secant
73:    invgamma          -- Inverse Gamma
74:    invgauss          -- Inverse Gaussian
75:    invweibull        -- Inverse Weibull
76:    johnsonsb         -- Johnson SB
77:    johnsonsu         -- Johnson SU
78:    kappa4            -- Kappa 4 parameter
79:    kappa3            -- Kappa 3 parameter
80:    ksone             -- Kolmogorov-Smirnov one-sided (no stats)
81:    kstwobign         -- Kolmogorov-Smirnov two-sided test for Large N (no stats)
82:    laplace           -- Laplace
83:    levy              -- Levy
84:    levy_l
85:    levy_stable
86:    logistic          -- Logistic
87:    loggamma          -- Log-Gamma
88:    loglaplace        -- Log-Laplace (Log Double Exponential)
89:    lognorm           -- Log-Normal
90:    lomax             -- Lomax (Pareto of the second kind)
91:    maxwell           -- Maxwell
92:    mielke            -- Mielke's Beta-Kappa
93:    nakagami          -- Nakagami
94:    ncx2              -- Non-central chi-squared
95:    ncf               -- Non-central F
96:    nct               -- Non-central Student's T
97:    norm              -- Normal (Gaussian)
98:    pareto            -- Pareto
99:    pearson3          -- Pearson type III
100:    powerlaw          -- Power-function
101:    powerlognorm      -- Power log normal
102:    powernorm         -- Power normal
103:    rdist             -- R-distribution
104:    reciprocal        -- Reciprocal
105:    rayleigh          -- Rayleigh
106:    rice              -- Rice
107:    recipinvgauss     -- Reciprocal Inverse Gaussian
108:    semicircular      -- Semicircular
109:    skewnorm          -- Skew normal
110:    t                 -- Student's T
111:    trapz              -- Trapezoidal
112:    triang            -- Triangular
113:    truncexpon        -- Truncated Exponential
114:    truncnorm         -- Truncated Normal
115:    tukeylambda       -- Tukey-Lambda
116:    uniform           -- Uniform
117:    vonmises          -- Von-Mises (Circular)
118:    vonmises_line     -- Von-Mises (Line)
119:    wald              -- Wald
120:    weibull_min       -- Minimum Weibull (see Frechet)
121:    weibull_max       -- Maximum Weibull (see Frechet)
122:    wrapcauchy        -- Wrapped Cauchy
123: 
124: Multivariate distributions
125: ==========================
126: 
127: .. autosummary::
128:    :toctree: generated/
129: 
130:    multivariate_normal   -- Multivariate normal distribution
131:    matrix_normal         -- Matrix normal distribution
132:    dirichlet             -- Dirichlet
133:    wishart               -- Wishart
134:    invwishart            -- Inverse Wishart
135:    multinomial           -- Multinomial distribution
136:    special_ortho_group   -- SO(N) group
137:    ortho_group           -- O(N) group
138:    unitary_group         -- U(N) gropu
139:    random_correlation    -- random correlation matrices
140: 
141: Discrete distributions
142: ======================
143: 
144: .. autosummary::
145:    :toctree: generated/
146: 
147:    bernoulli         -- Bernoulli
148:    binom             -- Binomial
149:    boltzmann         -- Boltzmann (Truncated Discrete Exponential)
150:    dlaplace          -- Discrete Laplacian
151:    geom              -- Geometric
152:    hypergeom         -- Hypergeometric
153:    logser            -- Logarithmic (Log-Series, Series)
154:    nbinom            -- Negative Binomial
155:    planck            -- Planck (Discrete Exponential)
156:    poisson           -- Poisson
157:    randint           -- Discrete Uniform
158:    skellam           -- Skellam
159:    zipf              -- Zipf
160: 
161: Statistical functions
162: =====================
163: 
164: Several of these functions have a similar version in scipy.stats.mstats
165: which work for masked arrays.
166: 
167: .. autosummary::
168:    :toctree: generated/
169: 
170:    describe          -- Descriptive statistics
171:    gmean             -- Geometric mean
172:    hmean             -- Harmonic mean
173:    kurtosis          -- Fisher or Pearson kurtosis
174:    kurtosistest      --
175:    mode              -- Modal value
176:    moment            -- Central moment
177:    normaltest        --
178:    skew              -- Skewness
179:    skewtest          --
180:    kstat             --
181:    kstatvar          --
182:    tmean             -- Truncated arithmetic mean
183:    tvar              -- Truncated variance
184:    tmin              --
185:    tmax              --
186:    tstd              --
187:    tsem              --
188:    variation         -- Coefficient of variation
189:    find_repeats
190:    trim_mean
191: 
192: .. autosummary::
193:    :toctree: generated/
194: 
195:    cumfreq
196:    itemfreq
197:    percentileofscore
198:    scoreatpercentile
199:    relfreq
200: 
201: .. autosummary::
202:    :toctree: generated/
203: 
204:    binned_statistic     -- Compute a binned statistic for a set of data.
205:    binned_statistic_2d  -- Compute a 2-D binned statistic for a set of data.
206:    binned_statistic_dd  -- Compute a d-D binned statistic for a set of data.
207: 
208: .. autosummary::
209:    :toctree: generated/
210: 
211:    obrientransform
212:    bayes_mvs
213:    mvsdist
214:    sem
215:    zmap
216:    zscore
217:    iqr
218: 
219: .. autosummary::
220:    :toctree: generated/
221: 
222:    sigmaclip
223:    trimboth
224:    trim1
225: 
226: .. autosummary::
227:    :toctree: generated/
228: 
229:    f_oneway
230:    pearsonr
231:    spearmanr
232:    pointbiserialr
233:    kendalltau
234:    weightedtau
235:    linregress
236:    theilslopes
237: 
238: .. autosummary::
239:    :toctree: generated/
240: 
241:    ttest_1samp
242:    ttest_ind
243:    ttest_ind_from_stats
244:    ttest_rel
245:    kstest
246:    chisquare
247:    power_divergence
248:    ks_2samp
249:    mannwhitneyu
250:    tiecorrect
251:    rankdata
252:    ranksums
253:    wilcoxon
254:    kruskal
255:    friedmanchisquare
256:    combine_pvalues
257:    jarque_bera
258: 
259: .. autosummary::
260:    :toctree: generated/
261: 
262:    ansari
263:    bartlett
264:    levene
265:    shapiro
266:    anderson
267:    anderson_ksamp
268:    binom_test
269:    fligner
270:    median_test
271:    mood
272: 
273: .. autosummary::
274:    :toctree: generated/
275: 
276:    boxcox
277:    boxcox_normmax
278:    boxcox_llf
279: 
280:    entropy
281: 
282: .. autosummary::
283:    :toctree: generated/
284: 
285:    wasserstein_distance
286:    energy_distance
287: 
288: Circular statistical functions
289: ==============================
290: 
291: .. autosummary::
292:    :toctree: generated/
293: 
294:    circmean
295:    circvar
296:    circstd
297: 
298: Contingency table functions
299: ===========================
300: 
301: .. autosummary::
302:    :toctree: generated/
303: 
304:    chi2_contingency
305:    contingency.expected_freq
306:    contingency.margins
307:    fisher_exact
308: 
309: Plot-tests
310: ==========
311: 
312: .. autosummary::
313:    :toctree: generated/
314: 
315:    ppcc_max
316:    ppcc_plot
317:    probplot
318:    boxcox_normplot
319: 
320: 
321: Masked statistics functions
322: ===========================
323: 
324: .. toctree::
325: 
326:    stats.mstats
327: 
328: 
329: Univariate and multivariate kernel density estimation (:mod:`scipy.stats.kde`)
330: ==============================================================================
331: 
332: .. autosummary::
333:    :toctree: generated/
334: 
335:    gaussian_kde
336: 
337: For many more stat related functions install the software R and the
338: interface package rpy.
339: 
340: '''
341: from __future__ import division, print_function, absolute_import
342: 
343: from .stats import *
344: from .distributions import *
345: from .morestats import *
346: from ._binned_statistic import *
347: from .kde import gaussian_kde
348: from . import mstats
349: from .contingency import chi2_contingency
350: from ._multivariate import *
351: 
352: __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
353: 
354: from scipy._lib._testutils import PytestTester
355: test = PytestTester(__name__)
356: del PytestTester
357: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_626784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, (-1)), 'str', "\n==========================================\nStatistical functions (:mod:`scipy.stats`)\n==========================================\n\n.. module:: scipy.stats\n\nThis module contains a large number of probability distributions as\nwell as a growing library of statistical functions.\n\nEach univariate distribution is an instance of a subclass of `rv_continuous`\n(`rv_discrete` for discrete distributions):\n\n.. autosummary::\n   :toctree: generated/\n\n   rv_continuous\n   rv_discrete\n   rv_histogram\n\nContinuous distributions\n========================\n\n.. autosummary::\n   :toctree: generated/\n\n   alpha             -- Alpha\n   anglit            -- Anglit\n   arcsine           -- Arcsine\n   argus             -- Argus\n   beta              -- Beta\n   betaprime         -- Beta Prime\n   bradford          -- Bradford\n   burr              -- Burr (Type III)\n   burr12            -- Burr (Type XII)\n   cauchy            -- Cauchy\n   chi               -- Chi\n   chi2              -- Chi-squared\n   cosine            -- Cosine\n   crystalball       -- Crystalball\n   dgamma            -- Double Gamma\n   dweibull          -- Double Weibull\n   erlang            -- Erlang\n   expon             -- Exponential\n   exponnorm         -- Exponentially Modified Normal\n   exponweib         -- Exponentiated Weibull\n   exponpow          -- Exponential Power\n   f                 -- F (Snecdor F)\n   fatiguelife       -- Fatigue Life (Birnbaum-Saunders)\n   fisk              -- Fisk\n   foldcauchy        -- Folded Cauchy\n   foldnorm          -- Folded Normal\n   frechet_r         -- Deprecated. Alias for weibull_min\n   frechet_l         -- Deprecated. Alias for weibull_max\n   genlogistic       -- Generalized Logistic\n   gennorm           -- Generalized normal\n   genpareto         -- Generalized Pareto\n   genexpon          -- Generalized Exponential\n   genextreme        -- Generalized Extreme Value\n   gausshyper        -- Gauss Hypergeometric\n   gamma             -- Gamma\n   gengamma          -- Generalized gamma\n   genhalflogistic   -- Generalized Half Logistic\n   gilbrat           -- Gilbrat\n   gompertz          -- Gompertz (Truncated Gumbel)\n   gumbel_r          -- Right Sided Gumbel, Log-Weibull, Fisher-Tippett, Extreme Value Type I\n   gumbel_l          -- Left Sided Gumbel, etc.\n   halfcauchy        -- Half Cauchy\n   halflogistic      -- Half Logistic\n   halfnorm          -- Half Normal\n   halfgennorm       -- Generalized Half Normal\n   hypsecant         -- Hyperbolic Secant\n   invgamma          -- Inverse Gamma\n   invgauss          -- Inverse Gaussian\n   invweibull        -- Inverse Weibull\n   johnsonsb         -- Johnson SB\n   johnsonsu         -- Johnson SU\n   kappa4            -- Kappa 4 parameter\n   kappa3            -- Kappa 3 parameter\n   ksone             -- Kolmogorov-Smirnov one-sided (no stats)\n   kstwobign         -- Kolmogorov-Smirnov two-sided test for Large N (no stats)\n   laplace           -- Laplace\n   levy              -- Levy\n   levy_l\n   levy_stable\n   logistic          -- Logistic\n   loggamma          -- Log-Gamma\n   loglaplace        -- Log-Laplace (Log Double Exponential)\n   lognorm           -- Log-Normal\n   lomax             -- Lomax (Pareto of the second kind)\n   maxwell           -- Maxwell\n   mielke            -- Mielke's Beta-Kappa\n   nakagami          -- Nakagami\n   ncx2              -- Non-central chi-squared\n   ncf               -- Non-central F\n   nct               -- Non-central Student's T\n   norm              -- Normal (Gaussian)\n   pareto            -- Pareto\n   pearson3          -- Pearson type III\n   powerlaw          -- Power-function\n   powerlognorm      -- Power log normal\n   powernorm         -- Power normal\n   rdist             -- R-distribution\n   reciprocal        -- Reciprocal\n   rayleigh          -- Rayleigh\n   rice              -- Rice\n   recipinvgauss     -- Reciprocal Inverse Gaussian\n   semicircular      -- Semicircular\n   skewnorm          -- Skew normal\n   t                 -- Student's T\n   trapz              -- Trapezoidal\n   triang            -- Triangular\n   truncexpon        -- Truncated Exponential\n   truncnorm         -- Truncated Normal\n   tukeylambda       -- Tukey-Lambda\n   uniform           -- Uniform\n   vonmises          -- Von-Mises (Circular)\n   vonmises_line     -- Von-Mises (Line)\n   wald              -- Wald\n   weibull_min       -- Minimum Weibull (see Frechet)\n   weibull_max       -- Maximum Weibull (see Frechet)\n   wrapcauchy        -- Wrapped Cauchy\n\nMultivariate distributions\n==========================\n\n.. autosummary::\n   :toctree: generated/\n\n   multivariate_normal   -- Multivariate normal distribution\n   matrix_normal         -- Matrix normal distribution\n   dirichlet             -- Dirichlet\n   wishart               -- Wishart\n   invwishart            -- Inverse Wishart\n   multinomial           -- Multinomial distribution\n   special_ortho_group   -- SO(N) group\n   ortho_group           -- O(N) group\n   unitary_group         -- U(N) gropu\n   random_correlation    -- random correlation matrices\n\nDiscrete distributions\n======================\n\n.. autosummary::\n   :toctree: generated/\n\n   bernoulli         -- Bernoulli\n   binom             -- Binomial\n   boltzmann         -- Boltzmann (Truncated Discrete Exponential)\n   dlaplace          -- Discrete Laplacian\n   geom              -- Geometric\n   hypergeom         -- Hypergeometric\n   logser            -- Logarithmic (Log-Series, Series)\n   nbinom            -- Negative Binomial\n   planck            -- Planck (Discrete Exponential)\n   poisson           -- Poisson\n   randint           -- Discrete Uniform\n   skellam           -- Skellam\n   zipf              -- Zipf\n\nStatistical functions\n=====================\n\nSeveral of these functions have a similar version in scipy.stats.mstats\nwhich work for masked arrays.\n\n.. autosummary::\n   :toctree: generated/\n\n   describe          -- Descriptive statistics\n   gmean             -- Geometric mean\n   hmean             -- Harmonic mean\n   kurtosis          -- Fisher or Pearson kurtosis\n   kurtosistest      --\n   mode              -- Modal value\n   moment            -- Central moment\n   normaltest        --\n   skew              -- Skewness\n   skewtest          --\n   kstat             --\n   kstatvar          --\n   tmean             -- Truncated arithmetic mean\n   tvar              -- Truncated variance\n   tmin              --\n   tmax              --\n   tstd              --\n   tsem              --\n   variation         -- Coefficient of variation\n   find_repeats\n   trim_mean\n\n.. autosummary::\n   :toctree: generated/\n\n   cumfreq\n   itemfreq\n   percentileofscore\n   scoreatpercentile\n   relfreq\n\n.. autosummary::\n   :toctree: generated/\n\n   binned_statistic     -- Compute a binned statistic for a set of data.\n   binned_statistic_2d  -- Compute a 2-D binned statistic for a set of data.\n   binned_statistic_dd  -- Compute a d-D binned statistic for a set of data.\n\n.. autosummary::\n   :toctree: generated/\n\n   obrientransform\n   bayes_mvs\n   mvsdist\n   sem\n   zmap\n   zscore\n   iqr\n\n.. autosummary::\n   :toctree: generated/\n\n   sigmaclip\n   trimboth\n   trim1\n\n.. autosummary::\n   :toctree: generated/\n\n   f_oneway\n   pearsonr\n   spearmanr\n   pointbiserialr\n   kendalltau\n   weightedtau\n   linregress\n   theilslopes\n\n.. autosummary::\n   :toctree: generated/\n\n   ttest_1samp\n   ttest_ind\n   ttest_ind_from_stats\n   ttest_rel\n   kstest\n   chisquare\n   power_divergence\n   ks_2samp\n   mannwhitneyu\n   tiecorrect\n   rankdata\n   ranksums\n   wilcoxon\n   kruskal\n   friedmanchisquare\n   combine_pvalues\n   jarque_bera\n\n.. autosummary::\n   :toctree: generated/\n\n   ansari\n   bartlett\n   levene\n   shapiro\n   anderson\n   anderson_ksamp\n   binom_test\n   fligner\n   median_test\n   mood\n\n.. autosummary::\n   :toctree: generated/\n\n   boxcox\n   boxcox_normmax\n   boxcox_llf\n\n   entropy\n\n.. autosummary::\n   :toctree: generated/\n\n   wasserstein_distance\n   energy_distance\n\nCircular statistical functions\n==============================\n\n.. autosummary::\n   :toctree: generated/\n\n   circmean\n   circvar\n   circstd\n\nContingency table functions\n===========================\n\n.. autosummary::\n   :toctree: generated/\n\n   chi2_contingency\n   contingency.expected_freq\n   contingency.margins\n   fisher_exact\n\nPlot-tests\n==========\n\n.. autosummary::\n   :toctree: generated/\n\n   ppcc_max\n   ppcc_plot\n   probplot\n   boxcox_normplot\n\n\nMasked statistics functions\n===========================\n\n.. toctree::\n\n   stats.mstats\n\n\nUnivariate and multivariate kernel density estimation (:mod:`scipy.stats.kde`)\n==============================================================================\n\n.. autosummary::\n   :toctree: generated/\n\n   gaussian_kde\n\nFor many more stat related functions install the software R and the\ninterface package rpy.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 343, 0))

# 'from scipy.stats.stats import ' statement (line 343)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626785 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 343, 0), 'scipy.stats.stats')

if (type(import_626785) is not StypyTypeError):

    if (import_626785 != 'pyd_module'):
        __import__(import_626785)
        sys_modules_626786 = sys.modules[import_626785]
        import_from_module(stypy.reporting.localization.Localization(__file__, 343, 0), 'scipy.stats.stats', sys_modules_626786.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 343, 0), __file__, sys_modules_626786, sys_modules_626786.module_type_store, module_type_store)
    else:
        from scipy.stats.stats import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 343, 0), 'scipy.stats.stats', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.stats.stats' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 0), 'scipy.stats.stats', import_626785)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 344, 0))

# 'from scipy.stats.distributions import ' statement (line 344)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626787 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 344, 0), 'scipy.stats.distributions')

if (type(import_626787) is not StypyTypeError):

    if (import_626787 != 'pyd_module'):
        __import__(import_626787)
        sys_modules_626788 = sys.modules[import_626787]
        import_from_module(stypy.reporting.localization.Localization(__file__, 344, 0), 'scipy.stats.distributions', sys_modules_626788.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 344, 0), __file__, sys_modules_626788, sys_modules_626788.module_type_store, module_type_store)
    else:
        from scipy.stats.distributions import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 344, 0), 'scipy.stats.distributions', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.stats.distributions' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 0), 'scipy.stats.distributions', import_626787)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 345, 0))

# 'from scipy.stats.morestats import ' statement (line 345)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626789 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 345, 0), 'scipy.stats.morestats')

if (type(import_626789) is not StypyTypeError):

    if (import_626789 != 'pyd_module'):
        __import__(import_626789)
        sys_modules_626790 = sys.modules[import_626789]
        import_from_module(stypy.reporting.localization.Localization(__file__, 345, 0), 'scipy.stats.morestats', sys_modules_626790.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 345, 0), __file__, sys_modules_626790, sys_modules_626790.module_type_store, module_type_store)
    else:
        from scipy.stats.morestats import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 345, 0), 'scipy.stats.morestats', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.stats.morestats' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 0), 'scipy.stats.morestats', import_626789)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 346, 0))

# 'from scipy.stats._binned_statistic import ' statement (line 346)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626791 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 346, 0), 'scipy.stats._binned_statistic')

if (type(import_626791) is not StypyTypeError):

    if (import_626791 != 'pyd_module'):
        __import__(import_626791)
        sys_modules_626792 = sys.modules[import_626791]
        import_from_module(stypy.reporting.localization.Localization(__file__, 346, 0), 'scipy.stats._binned_statistic', sys_modules_626792.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 346, 0), __file__, sys_modules_626792, sys_modules_626792.module_type_store, module_type_store)
    else:
        from scipy.stats._binned_statistic import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 346, 0), 'scipy.stats._binned_statistic', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.stats._binned_statistic' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 0), 'scipy.stats._binned_statistic', import_626791)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 347, 0))

# 'from scipy.stats.kde import gaussian_kde' statement (line 347)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626793 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 347, 0), 'scipy.stats.kde')

if (type(import_626793) is not StypyTypeError):

    if (import_626793 != 'pyd_module'):
        __import__(import_626793)
        sys_modules_626794 = sys.modules[import_626793]
        import_from_module(stypy.reporting.localization.Localization(__file__, 347, 0), 'scipy.stats.kde', sys_modules_626794.module_type_store, module_type_store, ['gaussian_kde'])
        nest_module(stypy.reporting.localization.Localization(__file__, 347, 0), __file__, sys_modules_626794, sys_modules_626794.module_type_store, module_type_store)
    else:
        from scipy.stats.kde import gaussian_kde

        import_from_module(stypy.reporting.localization.Localization(__file__, 347, 0), 'scipy.stats.kde', None, module_type_store, ['gaussian_kde'], [gaussian_kde])

else:
    # Assigning a type to the variable 'scipy.stats.kde' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), 'scipy.stats.kde', import_626793)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 348, 0))

# 'from scipy.stats import mstats' statement (line 348)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626795 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 348, 0), 'scipy.stats')

if (type(import_626795) is not StypyTypeError):

    if (import_626795 != 'pyd_module'):
        __import__(import_626795)
        sys_modules_626796 = sys.modules[import_626795]
        import_from_module(stypy.reporting.localization.Localization(__file__, 348, 0), 'scipy.stats', sys_modules_626796.module_type_store, module_type_store, ['mstats'])
        nest_module(stypy.reporting.localization.Localization(__file__, 348, 0), __file__, sys_modules_626796, sys_modules_626796.module_type_store, module_type_store)
    else:
        from scipy.stats import mstats

        import_from_module(stypy.reporting.localization.Localization(__file__, 348, 0), 'scipy.stats', None, module_type_store, ['mstats'], [mstats])

else:
    # Assigning a type to the variable 'scipy.stats' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 0), 'scipy.stats', import_626795)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 349, 0))

# 'from scipy.stats.contingency import chi2_contingency' statement (line 349)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626797 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 349, 0), 'scipy.stats.contingency')

if (type(import_626797) is not StypyTypeError):

    if (import_626797 != 'pyd_module'):
        __import__(import_626797)
        sys_modules_626798 = sys.modules[import_626797]
        import_from_module(stypy.reporting.localization.Localization(__file__, 349, 0), 'scipy.stats.contingency', sys_modules_626798.module_type_store, module_type_store, ['chi2_contingency'])
        nest_module(stypy.reporting.localization.Localization(__file__, 349, 0), __file__, sys_modules_626798, sys_modules_626798.module_type_store, module_type_store)
    else:
        from scipy.stats.contingency import chi2_contingency

        import_from_module(stypy.reporting.localization.Localization(__file__, 349, 0), 'scipy.stats.contingency', None, module_type_store, ['chi2_contingency'], [chi2_contingency])

else:
    # Assigning a type to the variable 'scipy.stats.contingency' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 0), 'scipy.stats.contingency', import_626797)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 350, 0))

# 'from scipy.stats._multivariate import ' statement (line 350)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626799 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 350, 0), 'scipy.stats._multivariate')

if (type(import_626799) is not StypyTypeError):

    if (import_626799 != 'pyd_module'):
        __import__(import_626799)
        sys_modules_626800 = sys.modules[import_626799]
        import_from_module(stypy.reporting.localization.Localization(__file__, 350, 0), 'scipy.stats._multivariate', sys_modules_626800.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 350, 0), __file__, sys_modules_626800, sys_modules_626800.module_type_store, module_type_store)
    else:
        from scipy.stats._multivariate import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 350, 0), 'scipy.stats._multivariate', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.stats._multivariate' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'scipy.stats._multivariate', import_626799)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


# Assigning a ListComp to a Name (line 352):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 352)
# Processing the call keyword arguments (line 352)
kwargs_626809 = {}
# Getting the type of 'dir' (line 352)
dir_626808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 22), 'dir', False)
# Calling dir(args, kwargs) (line 352)
dir_call_result_626810 = invoke(stypy.reporting.localization.Localization(__file__, 352, 22), dir_626808, *[], **kwargs_626809)

comprehension_626811 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 11), dir_call_result_626810)
# Assigning a type to the variable 's' (line 352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 11), 's', comprehension_626811)


# Call to startswith(...): (line 352)
# Processing the call arguments (line 352)
str_626804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 48), 'str', '_')
# Processing the call keyword arguments (line 352)
kwargs_626805 = {}
# Getting the type of 's' (line 352)
s_626802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 352)
startswith_626803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 35), s_626802, 'startswith')
# Calling startswith(args, kwargs) (line 352)
startswith_call_result_626806 = invoke(stypy.reporting.localization.Localization(__file__, 352, 35), startswith_626803, *[str_626804], **kwargs_626805)

# Applying the 'not' unary operator (line 352)
result_not__626807 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 31), 'not', startswith_call_result_626806)

# Getting the type of 's' (line 352)
s_626801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 11), 's')
list_626812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 11), list_626812, s_626801)
# Assigning a type to the variable '__all__' (line 352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), '__all__', list_626812)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 354, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 354)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626813 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 354, 0), 'scipy._lib._testutils')

if (type(import_626813) is not StypyTypeError):

    if (import_626813 != 'pyd_module'):
        __import__(import_626813)
        sys_modules_626814 = sys.modules[import_626813]
        import_from_module(stypy.reporting.localization.Localization(__file__, 354, 0), 'scipy._lib._testutils', sys_modules_626814.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 354, 0), __file__, sys_modules_626814, sys_modules_626814.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 354, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'scipy._lib._testutils', import_626813)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


# Assigning a Call to a Name (line 355):

# Call to PytestTester(...): (line 355)
# Processing the call arguments (line 355)
# Getting the type of '__name__' (line 355)
name___626816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), '__name__', False)
# Processing the call keyword arguments (line 355)
kwargs_626817 = {}
# Getting the type of 'PytestTester' (line 355)
PytestTester_626815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 355)
PytestTester_call_result_626818 = invoke(stypy.reporting.localization.Localization(__file__, 355, 7), PytestTester_626815, *[name___626816], **kwargs_626817)

# Assigning a type to the variable 'test' (line 355)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 0), 'test', PytestTester_call_result_626818)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 356, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
