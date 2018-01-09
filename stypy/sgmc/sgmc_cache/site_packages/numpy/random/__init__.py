
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
9: random               Uniformly distributed values of a given shape.
10: bytes                Uniformly distributed random bytes.
11: random_integers      Uniformly distributed integers in a given range.
12: random_sample        Uniformly distributed floats in a given range.
13: random               Alias for random_sample
14: ranf                 Alias for random_sample
15: sample               Alias for random_sample
16: choice               Generate a weighted random sample from a given array-like
17: permutation          Randomly permute a sequence / generate a random sequence.
18: shuffle              Randomly permute a sequence in place.
19: seed                 Seed the random number generator.
20: ==================== =========================================================
21: 
22: ==================== =========================================================
23: Compatibility functions
24: ==============================================================================
25: rand                 Uniformly distributed values.
26: randn                Normally distributed values.
27: ranf                 Uniformly distributed floating point numbers.
28: randint              Uniformly distributed integers in a given range.
29: ==================== =========================================================
30: 
31: ==================== =========================================================
32: Univariate distributions
33: ==============================================================================
34: beta                 Beta distribution over ``[0, 1]``.
35: binomial             Binomial distribution.
36: chisquare            :math:`\\chi^2` distribution.
37: exponential          Exponential distribution.
38: f                    F (Fisher-Snedecor) distribution.
39: gamma                Gamma distribution.
40: geometric            Geometric distribution.
41: gumbel               Gumbel distribution.
42: hypergeometric       Hypergeometric distribution.
43: laplace              Laplace distribution.
44: logistic             Logistic distribution.
45: lognormal            Log-normal distribution.
46: logseries            Logarithmic series distribution.
47: negative_binomial    Negative binomial distribution.
48: noncentral_chisquare Non-central chi-square distribution.
49: noncentral_f         Non-central F distribution.
50: normal               Normal / Gaussian distribution.
51: pareto               Pareto distribution.
52: poisson              Poisson distribution.
53: power                Power distribution.
54: rayleigh             Rayleigh distribution.
55: triangular           Triangular distribution.
56: uniform              Uniform distribution.
57: vonmises             Von Mises circular distribution.
58: wald                 Wald (inverse Gaussian) distribution.
59: weibull              Weibull distribution.
60: zipf                 Zipf's distribution over ranked data.
61: ==================== =========================================================
62: 
63: ==================== =========================================================
64: Multivariate distributions
65: ==============================================================================
66: dirichlet            Multivariate generalization of Beta distribution.
67: multinomial          Multivariate generalization of the binomial distribution.
68: multivariate_normal  Multivariate generalization of the normal distribution.
69: ==================== =========================================================
70: 
71: ==================== =========================================================
72: Standard distributions
73: ==============================================================================
74: standard_cauchy      Standard Cauchy-Lorentz distribution.
75: standard_exponential Standard exponential distribution.
76: standard_gamma       Standard Gamma distribution.
77: standard_normal      Standard normal distribution.
78: standard_t           Standard Student's t-distribution.
79: ==================== =========================================================
80: 
81: ==================== =========================================================
82: Internal functions
83: ==============================================================================
84: get_state            Get tuple representing internal state of generator.
85: set_state            Set state of generator.
86: ==================== =========================================================
87: 
88: '''
89: from __future__ import division, absolute_import, print_function
90: 
91: import warnings
92: 
93: # To get sub-modules
94: from .info import __doc__, __all__
95: 
96: 
97: with warnings.catch_warnings():
98:     warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
99:     from .mtrand import *
100: 
101: # Some aliases:
102: ranf = random = sample = random_sample
103: __all__.extend(['ranf', 'random', 'sample'])
104: 
105: def __RandomState_ctor():
106:     '''Return a RandomState instance.
107: 
108:     This function exists solely to assist (un)pickling.
109: 
110:     Note that the state of the RandomState returned here is irrelevant, as this function's
111:     entire purpose is to return a newly allocated RandomState whose state pickle can set.
112:     Consequently the RandomState returned by this function is a freshly allocated copy
113:     with a seed=0.
114: 
115:     See https://github.com/numpy/numpy/issues/4763 for a detailed discussion
116: 
117:     '''
118:     return RandomState(seed=0)
119: 
120: from numpy.testing.nosetester import _numpy_tester
121: test = _numpy_tester().test
122: bench = _numpy_tester().bench
123: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_180767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'str', "\n========================\nRandom Number Generation\n========================\n\n==================== =========================================================\nUtility functions\n==============================================================================\nrandom               Uniformly distributed values of a given shape.\nbytes                Uniformly distributed random bytes.\nrandom_integers      Uniformly distributed integers in a given range.\nrandom_sample        Uniformly distributed floats in a given range.\nrandom               Alias for random_sample\nranf                 Alias for random_sample\nsample               Alias for random_sample\nchoice               Generate a weighted random sample from a given array-like\npermutation          Randomly permute a sequence / generate a random sequence.\nshuffle              Randomly permute a sequence in place.\nseed                 Seed the random number generator.\n==================== =========================================================\n\n==================== =========================================================\nCompatibility functions\n==============================================================================\nrand                 Uniformly distributed values.\nrandn                Normally distributed values.\nranf                 Uniformly distributed floating point numbers.\nrandint              Uniformly distributed integers in a given range.\n==================== =========================================================\n\n==================== =========================================================\nUnivariate distributions\n==============================================================================\nbeta                 Beta distribution over ``[0, 1]``.\nbinomial             Binomial distribution.\nchisquare            :math:`\\chi^2` distribution.\nexponential          Exponential distribution.\nf                    F (Fisher-Snedecor) distribution.\ngamma                Gamma distribution.\ngeometric            Geometric distribution.\ngumbel               Gumbel distribution.\nhypergeometric       Hypergeometric distribution.\nlaplace              Laplace distribution.\nlogistic             Logistic distribution.\nlognormal            Log-normal distribution.\nlogseries            Logarithmic series distribution.\nnegative_binomial    Negative binomial distribution.\nnoncentral_chisquare Non-central chi-square distribution.\nnoncentral_f         Non-central F distribution.\nnormal               Normal / Gaussian distribution.\npareto               Pareto distribution.\npoisson              Poisson distribution.\npower                Power distribution.\nrayleigh             Rayleigh distribution.\ntriangular           Triangular distribution.\nuniform              Uniform distribution.\nvonmises             Von Mises circular distribution.\nwald                 Wald (inverse Gaussian) distribution.\nweibull              Weibull distribution.\nzipf                 Zipf's distribution over ranked data.\n==================== =========================================================\n\n==================== =========================================================\nMultivariate distributions\n==============================================================================\ndirichlet            Multivariate generalization of Beta distribution.\nmultinomial          Multivariate generalization of the binomial distribution.\nmultivariate_normal  Multivariate generalization of the normal distribution.\n==================== =========================================================\n\n==================== =========================================================\nStandard distributions\n==============================================================================\nstandard_cauchy      Standard Cauchy-Lorentz distribution.\nstandard_exponential Standard exponential distribution.\nstandard_gamma       Standard Gamma distribution.\nstandard_normal      Standard normal distribution.\nstandard_t           Standard Student's t-distribution.\n==================== =========================================================\n\n==================== =========================================================\nInternal functions\n==============================================================================\nget_state            Get tuple representing internal state of generator.\nset_state            Set state of generator.\n==================== =========================================================\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 91, 0))

# 'import warnings' statement (line 91)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 94, 0))

# 'from numpy.random.info import __doc__, __all__' statement (line 94)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/random/')
import_180768 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'numpy.random.info')

if (type(import_180768) is not StypyTypeError):

    if (import_180768 != 'pyd_module'):
        __import__(import_180768)
        sys_modules_180769 = sys.modules[import_180768]
        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'numpy.random.info', sys_modules_180769.module_type_store, module_type_store, ['__doc__', '__all__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 94, 0), __file__, sys_modules_180769, sys_modules_180769.module_type_store, module_type_store)
    else:
        from numpy.random.info import __doc__, __all__

        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'numpy.random.info', None, module_type_store, ['__doc__', '__all__'], [__doc__, __all__])

else:
    # Assigning a type to the variable 'numpy.random.info' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'numpy.random.info', import_180768)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/random/')


# Call to catch_warnings(...): (line 97)
# Processing the call keyword arguments (line 97)
kwargs_180772 = {}
# Getting the type of 'warnings' (line 97)
warnings_180770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 5), 'warnings', False)
# Obtaining the member 'catch_warnings' of a type (line 97)
catch_warnings_180771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 5), warnings_180770, 'catch_warnings')
# Calling catch_warnings(args, kwargs) (line 97)
catch_warnings_call_result_180773 = invoke(stypy.reporting.localization.Localization(__file__, 97, 5), catch_warnings_180771, *[], **kwargs_180772)

with_180774 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 97, 5), catch_warnings_call_result_180773, 'with parameter', '__enter__', '__exit__')

if with_180774:
    # Calling the __enter__ method to initiate a with section
    # Obtaining the member '__enter__' of a type (line 97)
    enter___180775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 5), catch_warnings_call_result_180773, '__enter__')
    with_enter_180776 = invoke(stypy.reporting.localization.Localization(__file__, 97, 5), enter___180775)
    
    # Call to filterwarnings(...): (line 98)
    # Processing the call arguments (line 98)
    str_180779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 28), 'str', 'ignore')
    # Processing the call keyword arguments (line 98)
    str_180780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 46), 'str', 'numpy.ndarray size changed')
    keyword_180781 = str_180780
    kwargs_180782 = {'message': keyword_180781}
    # Getting the type of 'warnings' (line 98)
    warnings_180777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'warnings', False)
    # Obtaining the member 'filterwarnings' of a type (line 98)
    filterwarnings_180778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), warnings_180777, 'filterwarnings')
    # Calling filterwarnings(args, kwargs) (line 98)
    filterwarnings_call_result_180783 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), filterwarnings_180778, *[str_180779], **kwargs_180782)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 99, 4))
    
    # 'from numpy.random.mtrand import ' statement (line 99)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/random/')
    import_180784 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 99, 4), 'numpy.random.mtrand')

    if (type(import_180784) is not StypyTypeError):

        if (import_180784 != 'pyd_module'):
            __import__(import_180784)
            sys_modules_180785 = sys.modules[import_180784]
            import_from_module(stypy.reporting.localization.Localization(__file__, 99, 4), 'numpy.random.mtrand', sys_modules_180785.module_type_store, module_type_store, ['*'])
            nest_module(stypy.reporting.localization.Localization(__file__, 99, 4), __file__, sys_modules_180785, sys_modules_180785.module_type_store, module_type_store)
        else:
            from numpy.random.mtrand import *

            import_from_module(stypy.reporting.localization.Localization(__file__, 99, 4), 'numpy.random.mtrand', None, module_type_store, ['*'], None)

    else:
        # Assigning a type to the variable 'numpy.random.mtrand' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'numpy.random.mtrand', import_180784)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/random/')
    
    # Calling the __exit__ method to finish a with section
    # Obtaining the member '__exit__' of a type (line 97)
    exit___180786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 5), catch_warnings_call_result_180773, '__exit__')
    with_exit_180787 = invoke(stypy.reporting.localization.Localization(__file__, 97, 5), exit___180786, None, None, None)


# Multiple assignment of 3 elements.
# Getting the type of 'random_sample' (line 102)
random_sample_180788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'random_sample')
# Assigning a type to the variable 'sample' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'sample', random_sample_180788)
# Getting the type of 'sample' (line 102)
sample_180789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'sample')
# Assigning a type to the variable 'random' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 7), 'random', sample_180789)
# Getting the type of 'random' (line 102)
random_180790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 7), 'random')
# Assigning a type to the variable 'ranf' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'ranf', random_180790)

# Call to extend(...): (line 103)
# Processing the call arguments (line 103)

# Obtaining an instance of the builtin type 'list' (line 103)
list_180793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 103)
# Adding element type (line 103)
str_180794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 16), 'str', 'ranf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), list_180793, str_180794)
# Adding element type (line 103)
str_180795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 24), 'str', 'random')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), list_180793, str_180795)
# Adding element type (line 103)
str_180796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 34), 'str', 'sample')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), list_180793, str_180796)

# Processing the call keyword arguments (line 103)
kwargs_180797 = {}
# Getting the type of '__all__' (line 103)
all___180791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), '__all__', False)
# Obtaining the member 'extend' of a type (line 103)
extend_180792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 0), all___180791, 'extend')
# Calling extend(args, kwargs) (line 103)
extend_call_result_180798 = invoke(stypy.reporting.localization.Localization(__file__, 103, 0), extend_180792, *[list_180793], **kwargs_180797)


@norecursion
def __RandomState_ctor(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__RandomState_ctor'
    module_type_store = module_type_store.open_function_context('__RandomState_ctor', 105, 0, False)
    
    # Passed parameters checking function
    __RandomState_ctor.stypy_localization = localization
    __RandomState_ctor.stypy_type_of_self = None
    __RandomState_ctor.stypy_type_store = module_type_store
    __RandomState_ctor.stypy_function_name = '__RandomState_ctor'
    __RandomState_ctor.stypy_param_names_list = []
    __RandomState_ctor.stypy_varargs_param_name = None
    __RandomState_ctor.stypy_kwargs_param_name = None
    __RandomState_ctor.stypy_call_defaults = defaults
    __RandomState_ctor.stypy_call_varargs = varargs
    __RandomState_ctor.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__RandomState_ctor', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__RandomState_ctor', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__RandomState_ctor(...)' code ##################

    str_180799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', "Return a RandomState instance.\n\n    This function exists solely to assist (un)pickling.\n\n    Note that the state of the RandomState returned here is irrelevant, as this function's\n    entire purpose is to return a newly allocated RandomState whose state pickle can set.\n    Consequently the RandomState returned by this function is a freshly allocated copy\n    with a seed=0.\n\n    See https://github.com/numpy/numpy/issues/4763 for a detailed discussion\n\n    ")
    
    # Call to RandomState(...): (line 118)
    # Processing the call keyword arguments (line 118)
    int_180801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 28), 'int')
    keyword_180802 = int_180801
    kwargs_180803 = {'seed': keyword_180802}
    # Getting the type of 'RandomState' (line 118)
    RandomState_180800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'RandomState', False)
    # Calling RandomState(args, kwargs) (line 118)
    RandomState_call_result_180804 = invoke(stypy.reporting.localization.Localization(__file__, 118, 11), RandomState_180800, *[], **kwargs_180803)
    
    # Assigning a type to the variable 'stypy_return_type' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type', RandomState_call_result_180804)
    
    # ################# End of '__RandomState_ctor(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__RandomState_ctor' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_180805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_180805)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__RandomState_ctor'
    return stypy_return_type_180805

# Assigning a type to the variable '__RandomState_ctor' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), '__RandomState_ctor', __RandomState_ctor)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 120, 0))

# 'from numpy.testing.nosetester import _numpy_tester' statement (line 120)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/random/')
import_180806 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 120, 0), 'numpy.testing.nosetester')

if (type(import_180806) is not StypyTypeError):

    if (import_180806 != 'pyd_module'):
        __import__(import_180806)
        sys_modules_180807 = sys.modules[import_180806]
        import_from_module(stypy.reporting.localization.Localization(__file__, 120, 0), 'numpy.testing.nosetester', sys_modules_180807.module_type_store, module_type_store, ['_numpy_tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 120, 0), __file__, sys_modules_180807, sys_modules_180807.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import _numpy_tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 120, 0), 'numpy.testing.nosetester', None, module_type_store, ['_numpy_tester'], [_numpy_tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'numpy.testing.nosetester', import_180806)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/random/')


# Assigning a Attribute to a Name (line 121):

# Call to _numpy_tester(...): (line 121)
# Processing the call keyword arguments (line 121)
kwargs_180809 = {}
# Getting the type of '_numpy_tester' (line 121)
_numpy_tester_180808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 121)
_numpy_tester_call_result_180810 = invoke(stypy.reporting.localization.Localization(__file__, 121, 7), _numpy_tester_180808, *[], **kwargs_180809)

# Obtaining the member 'test' of a type (line 121)
test_180811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 7), _numpy_tester_call_result_180810, 'test')
# Assigning a type to the variable 'test' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'test', test_180811)

# Assigning a Attribute to a Name (line 122):

# Call to _numpy_tester(...): (line 122)
# Processing the call keyword arguments (line 122)
kwargs_180813 = {}
# Getting the type of '_numpy_tester' (line 122)
_numpy_tester_180812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 122)
_numpy_tester_call_result_180814 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), _numpy_tester_180812, *[], **kwargs_180813)

# Obtaining the member 'bench' of a type (line 122)
bench_180815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), _numpy_tester_call_result_180814, 'bench')
# Assigning a type to the variable 'bench' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'bench', bench_180815)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
