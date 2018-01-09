
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ========================
3: Random Number Generation
4: ========================
5: 
6: ==================== =========================================================
7: Utility functions
8: ==============================================================================
9: random_sample        Uniformly distributed floats over ``[0, 1)``.
10: random               Alias for `random_sample`.
11: bytes                Uniformly distributed random bytes.
12: random_integers      Uniformly distributed integers in a given range.
13: permutation          Randomly permute a sequence / generate a random sequence.
14: shuffle              Randomly permute a sequence in place.
15: seed                 Seed the random number generator.
16: choice               Random sample from 1-D array.
17: 
18: ==================== =========================================================
19: 
20: ==================== =========================================================
21: Compatibility functions
22: ==============================================================================
23: rand                 Uniformly distributed values.
24: randn                Normally distributed values.
25: ranf                 Uniformly distributed floating point numbers.
26: randint              Uniformly distributed integers in a given range.
27: ==================== =========================================================
28: 
29: ==================== =========================================================
30: Univariate distributions
31: ==============================================================================
32: beta                 Beta distribution over ``[0, 1]``.
33: binomial             Binomial distribution.
34: chisquare            :math:`\\chi^2` distribution.
35: exponential          Exponential distribution.
36: f                    F (Fisher-Snedecor) distribution.
37: gamma                Gamma distribution.
38: geometric            Geometric distribution.
39: gumbel               Gumbel distribution.
40: hypergeometric       Hypergeometric distribution.
41: laplace              Laplace distribution.
42: logistic             Logistic distribution.
43: lognormal            Log-normal distribution.
44: logseries            Logarithmic series distribution.
45: negative_binomial    Negative binomial distribution.
46: noncentral_chisquare Non-central chi-square distribution.
47: noncentral_f         Non-central F distribution.
48: normal               Normal / Gaussian distribution.
49: pareto               Pareto distribution.
50: poisson              Poisson distribution.
51: power                Power distribution.
52: rayleigh             Rayleigh distribution.
53: triangular           Triangular distribution.
54: uniform              Uniform distribution.
55: vonmises             Von Mises circular distribution.
56: wald                 Wald (inverse Gaussian) distribution.
57: weibull              Weibull distribution.
58: zipf                 Zipf's distribution over ranked data.
59: ==================== =========================================================
60: 
61: ==================== =========================================================
62: Multivariate distributions
63: ==============================================================================
64: dirichlet            Multivariate generalization of Beta distribution.
65: multinomial          Multivariate generalization of the binomial distribution.
66: multivariate_normal  Multivariate generalization of the normal distribution.
67: ==================== =========================================================
68: 
69: ==================== =========================================================
70: Standard distributions
71: ==============================================================================
72: standard_cauchy      Standard Cauchy-Lorentz distribution.
73: standard_exponential Standard exponential distribution.
74: standard_gamma       Standard Gamma distribution.
75: standard_normal      Standard normal distribution.
76: standard_t           Standard Student's t-distribution.
77: ==================== =========================================================
78: 
79: ==================== =========================================================
80: Internal functions
81: ==============================================================================
82: get_state            Get tuple representing internal state of generator.
83: set_state            Set state of generator.
84: ==================== =========================================================
85: 
86: '''
87: from __future__ import division, absolute_import, print_function
88: 
89: depends = ['core']
90: 
91: __all__ = [
92:     'beta',
93:     'binomial',
94:     'bytes',
95:     'chisquare',
96:     'choice',
97:     'dirichlet',
98:     'exponential',
99:     'f',
100:     'gamma',
101:     'geometric',
102:     'get_state',
103:     'gumbel',
104:     'hypergeometric',
105:     'laplace',
106:     'logistic',
107:     'lognormal',
108:     'logseries',
109:     'multinomial',
110:     'multivariate_normal',
111:     'negative_binomial',
112:     'noncentral_chisquare',
113:     'noncentral_f',
114:     'normal',
115:     'pareto',
116:     'permutation',
117:     'poisson',
118:     'power',
119:     'rand',
120:     'randint',
121:     'randn',
122:     'random_integers',
123:     'random_sample',
124:     'rayleigh',
125:     'seed',
126:     'set_state',
127:     'shuffle',
128:     'standard_cauchy',
129:     'standard_exponential',
130:     'standard_gamma',
131:     'standard_normal',
132:     'standard_t',
133:     'triangular',
134:     'uniform',
135:     'vonmises',
136:     'wald',
137:     'weibull',
138:     'zipf'
139: ]
140: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_180574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'str', "\n========================\nRandom Number Generation\n========================\n\n==================== =========================================================\nUtility functions\n==============================================================================\nrandom_sample        Uniformly distributed floats over ``[0, 1)``.\nrandom               Alias for `random_sample`.\nbytes                Uniformly distributed random bytes.\nrandom_integers      Uniformly distributed integers in a given range.\npermutation          Randomly permute a sequence / generate a random sequence.\nshuffle              Randomly permute a sequence in place.\nseed                 Seed the random number generator.\nchoice               Random sample from 1-D array.\n\n==================== =========================================================\n\n==================== =========================================================\nCompatibility functions\n==============================================================================\nrand                 Uniformly distributed values.\nrandn                Normally distributed values.\nranf                 Uniformly distributed floating point numbers.\nrandint              Uniformly distributed integers in a given range.\n==================== =========================================================\n\n==================== =========================================================\nUnivariate distributions\n==============================================================================\nbeta                 Beta distribution over ``[0, 1]``.\nbinomial             Binomial distribution.\nchisquare            :math:`\\chi^2` distribution.\nexponential          Exponential distribution.\nf                    F (Fisher-Snedecor) distribution.\ngamma                Gamma distribution.\ngeometric            Geometric distribution.\ngumbel               Gumbel distribution.\nhypergeometric       Hypergeometric distribution.\nlaplace              Laplace distribution.\nlogistic             Logistic distribution.\nlognormal            Log-normal distribution.\nlogseries            Logarithmic series distribution.\nnegative_binomial    Negative binomial distribution.\nnoncentral_chisquare Non-central chi-square distribution.\nnoncentral_f         Non-central F distribution.\nnormal               Normal / Gaussian distribution.\npareto               Pareto distribution.\npoisson              Poisson distribution.\npower                Power distribution.\nrayleigh             Rayleigh distribution.\ntriangular           Triangular distribution.\nuniform              Uniform distribution.\nvonmises             Von Mises circular distribution.\nwald                 Wald (inverse Gaussian) distribution.\nweibull              Weibull distribution.\nzipf                 Zipf's distribution over ranked data.\n==================== =========================================================\n\n==================== =========================================================\nMultivariate distributions\n==============================================================================\ndirichlet            Multivariate generalization of Beta distribution.\nmultinomial          Multivariate generalization of the binomial distribution.\nmultivariate_normal  Multivariate generalization of the normal distribution.\n==================== =========================================================\n\n==================== =========================================================\nStandard distributions\n==============================================================================\nstandard_cauchy      Standard Cauchy-Lorentz distribution.\nstandard_exponential Standard exponential distribution.\nstandard_gamma       Standard Gamma distribution.\nstandard_normal      Standard normal distribution.\nstandard_t           Standard Student's t-distribution.\n==================== =========================================================\n\n==================== =========================================================\nInternal functions\n==============================================================================\nget_state            Get tuple representing internal state of generator.\nset_state            Set state of generator.\n==================== =========================================================\n\n")

# Assigning a List to a Name (line 89):

# Obtaining an instance of the builtin type 'list' (line 89)
list_180575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 89)
# Adding element type (line 89)
str_180576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 11), 'str', 'core')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 10), list_180575, str_180576)

# Assigning a type to the variable 'depends' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'depends', list_180575)

# Assigning a List to a Name (line 91):
__all__ = ['beta', 'binomial', 'bytes', 'chisquare', 'choice', 'dirichlet', 'exponential', 'f', 'gamma', 'geometric', 'get_state', 'gumbel', 'hypergeometric', 'laplace', 'logistic', 'lognormal', 'logseries', 'multinomial', 'multivariate_normal', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'normal', 'pareto', 'permutation', 'poisson', 'power', 'rand', 'randint', 'randn', 'random_integers', 'random_sample', 'rayleigh', 'seed', 'set_state', 'shuffle', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'uniform', 'vonmises', 'wald', 'weibull', 'zipf']
module_type_store.set_exportable_members(['beta', 'binomial', 'bytes', 'chisquare', 'choice', 'dirichlet', 'exponential', 'f', 'gamma', 'geometric', 'get_state', 'gumbel', 'hypergeometric', 'laplace', 'logistic', 'lognormal', 'logseries', 'multinomial', 'multivariate_normal', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'normal', 'pareto', 'permutation', 'poisson', 'power', 'rand', 'randint', 'randn', 'random_integers', 'random_sample', 'rayleigh', 'seed', 'set_state', 'shuffle', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'uniform', 'vonmises', 'wald', 'weibull', 'zipf'])

# Obtaining an instance of the builtin type 'list' (line 91)
list_180577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 91)
# Adding element type (line 91)
str_180578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'str', 'beta')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180578)
# Adding element type (line 91)
str_180579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'str', 'binomial')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180579)
# Adding element type (line 91)
str_180580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'str', 'bytes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180580)
# Adding element type (line 91)
str_180581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 4), 'str', 'chisquare')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180581)
# Adding element type (line 91)
str_180582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 4), 'str', 'choice')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180582)
# Adding element type (line 91)
str_180583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'str', 'dirichlet')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180583)
# Adding element type (line 91)
str_180584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'str', 'exponential')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180584)
# Adding element type (line 91)
str_180585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180585)
# Adding element type (line 91)
str_180586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 4), 'str', 'gamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180586)
# Adding element type (line 91)
str_180587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'str', 'geometric')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180587)
# Adding element type (line 91)
str_180588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 4), 'str', 'get_state')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180588)
# Adding element type (line 91)
str_180589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 4), 'str', 'gumbel')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180589)
# Adding element type (line 91)
str_180590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'str', 'hypergeometric')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180590)
# Adding element type (line 91)
str_180591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 4), 'str', 'laplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180591)
# Adding element type (line 91)
str_180592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 4), 'str', 'logistic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180592)
# Adding element type (line 91)
str_180593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 4), 'str', 'lognormal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180593)
# Adding element type (line 91)
str_180594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'str', 'logseries')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180594)
# Adding element type (line 91)
str_180595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 4), 'str', 'multinomial')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180595)
# Adding element type (line 91)
str_180596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 4), 'str', 'multivariate_normal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180596)
# Adding element type (line 91)
str_180597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 4), 'str', 'negative_binomial')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180597)
# Adding element type (line 91)
str_180598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'str', 'noncentral_chisquare')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180598)
# Adding element type (line 91)
str_180599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 4), 'str', 'noncentral_f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180599)
# Adding element type (line 91)
str_180600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 4), 'str', 'normal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180600)
# Adding element type (line 91)
str_180601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'str', 'pareto')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180601)
# Adding element type (line 91)
str_180602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 4), 'str', 'permutation')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180602)
# Adding element type (line 91)
str_180603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 4), 'str', 'poisson')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180603)
# Adding element type (line 91)
str_180604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'str', 'power')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180604)
# Adding element type (line 91)
str_180605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 4), 'str', 'rand')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180605)
# Adding element type (line 91)
str_180606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 4), 'str', 'randint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180606)
# Adding element type (line 91)
str_180607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 4), 'str', 'randn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180607)
# Adding element type (line 91)
str_180608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 4), 'str', 'random_integers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180608)
# Adding element type (line 91)
str_180609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 4), 'str', 'random_sample')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180609)
# Adding element type (line 91)
str_180610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'str', 'rayleigh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180610)
# Adding element type (line 91)
str_180611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'str', 'seed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180611)
# Adding element type (line 91)
str_180612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 4), 'str', 'set_state')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180612)
# Adding element type (line 91)
str_180613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 4), 'str', 'shuffle')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180613)
# Adding element type (line 91)
str_180614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 4), 'str', 'standard_cauchy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180614)
# Adding element type (line 91)
str_180615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 4), 'str', 'standard_exponential')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180615)
# Adding element type (line 91)
str_180616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 4), 'str', 'standard_gamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180616)
# Adding element type (line 91)
str_180617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 4), 'str', 'standard_normal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180617)
# Adding element type (line 91)
str_180618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 4), 'str', 'standard_t')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180618)
# Adding element type (line 91)
str_180619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 4), 'str', 'triangular')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180619)
# Adding element type (line 91)
str_180620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 4), 'str', 'uniform')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180620)
# Adding element type (line 91)
str_180621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 4), 'str', 'vonmises')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180621)
# Adding element type (line 91)
str_180622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'str', 'wald')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180622)
# Adding element type (line 91)
str_180623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 4), 'str', 'weibull')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180623)
# Adding element type (line 91)
str_180624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 4), 'str', 'zipf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 10), list_180577, str_180624)

# Assigning a type to the variable '__all__' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), '__all__', list_180577)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
