
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Compute gammainc and gammaincc for large arguments and parameters
2: and save the values to data files for use in tests. We can't just
3: compare to mpmath's gammainc in test_mpmath.TestSystematic because it
4: would take too long.
5: 
6: Note that mpmath's gammainc is computed using hypercomb, but since it
7: doesn't allow the user to increase the maximum number of terms used in
8: the series it doesn't converge for many arguments. To get around this
9: we copy the mpmath implementation but use more terms.
10: 
11: This takes about 17 minutes to run on a 2.3 GHz Macbook Pro with 4GB
12: ram.
13: 
14: Sources:
15: [1] Fredrik Johansson and others. mpmath: a Python library for
16:     arbitrary-precision floating-point arithmetic (version 0.19),
17:     December 2013. http://mpmath.org/.
18: 
19: '''
20: from __future__ import division, print_function, absolute_import
21: 
22: import os
23: from time import time
24: import numpy as np
25: from numpy import pi
26: 
27: from scipy.special._mptestutils import mpf2float
28: 
29: try:
30:     import mpmath as mp
31: except ImportError:
32:     pass
33: 
34: 
35: def gammainc(a, x, dps=50, maxterms=10**8):
36:     '''Compute gammainc exactly like mpmath does but allow for more
37:     summands in hypercomb. See
38: 
39:     mpmath/functions/expintegrals.py#L134
40:     
41:     in the mpmath github repository.
42: 
43:     '''
44:     with mp.workdps(dps):
45:         z, a, b = mp.mpf(a), mp.mpf(x), mp.mpf(x)
46:         G = [z]
47:         negb = mp.fneg(b, exact=True)
48: 
49:         def h(z):
50:             T1 = [mp.exp(negb), b, z], [1, z, -1], [], G, [1], [1+z], b
51:             return (T1,)
52: 
53:         res = mp.hypercomb(h, [z], maxterms=maxterms)
54:         return mpf2float(res)
55: 
56: 
57: def gammaincc(a, x, dps=50, maxterms=10**8):
58:     '''Compute gammaincc exactly like mpmath does but allow for more
59:     terms in hypercomb. See
60: 
61:     mpmath/functions/expintegrals.py#L187
62: 
63:     in the mpmath github repository.
64: 
65:     '''
66:     with mp.workdps(dps):
67:         z, a = a, x
68:         
69:         if mp.isint(z):
70:             try:
71:                 # mpmath has a fast integer path
72:                 return mpf2float(mp.gammainc(z, a=a, regularized=True))
73:             except mp.libmp.NoConvergence:
74:                 pass
75:         nega = mp.fneg(a, exact=True)
76:         G = [z]
77:         # Use 2F0 series when possible; fall back to lower gamma representation
78:         try:
79:             def h(z):
80:                 r = z-1
81:                 return [([mp.exp(nega), a], [1, r], [], G, [1, -r], [], 1/nega)]
82:             return mpf2float(mp.hypercomb(h, [z], force_series=True))
83:         except mp.libmp.NoConvergence:
84:             def h(z):
85:                 T1 = [], [1, z-1], [z], G, [], [], 0
86:                 T2 = [-mp.exp(nega), a, z], [1, z, -1], [], G, [1], [1+z], a
87:                 return T1, T2
88:             return mpf2float(mp.hypercomb(h, [z], maxterms=maxterms))
89: 
90: 
91: def main():
92:     t0 = time()
93:     # It would be nice to have data for larger values, but either this
94:     # requires prohibitively large precision (dps > 800) or mpmath has
95:     # a bug. For example, gammainc(1e20, 1e20, dps=800) returns a
96:     # value around 0.03, while the true value should be close to 0.5
97:     # (DLMF 8.12.15).
98:     print(__doc__)
99:     pwd = os.path.dirname(__file__)
100:     r = np.logspace(4, 14, 30)
101:     ltheta = np.logspace(np.log10(pi/4), np.log10(np.arctan(0.6)), 30)
102:     utheta = np.logspace(np.log10(pi/4), np.log10(np.arctan(1.4)), 30)
103:     
104:     regimes = [(gammainc, ltheta), (gammaincc, utheta)]
105:     for func, theta in regimes:
106:         rg, thetag = np.meshgrid(r, theta)
107:         a, x = rg*np.cos(thetag), rg*np.sin(thetag)
108:         a, x = a.flatten(), x.flatten()
109:         dataset = []
110:         for i, (a0, x0) in enumerate(zip(a, x)):
111:             if func == gammaincc:
112:                 # Exploit the fast integer path in gammaincc whenever
113:                 # possible so that the computation doesn't take too
114:                 # long
115:                 a0, x0 = np.floor(a0), np.floor(x0)
116:             dataset.append((a0, x0, func(a0, x0)))
117:         dataset = np.array(dataset)
118:         filename = os.path.join(pwd, '..', 'tests', 'data', 'local',
119:                                 '{}.txt'.format(func.__name__))
120:         np.savetxt(filename, dataset)
121: 
122:     print("{} minutes elapsed".format((time() - t0)/60))
123: 
124: 
125: if __name__ == "__main__":
126:     main()
127: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_564081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', "Compute gammainc and gammaincc for large arguments and parameters\nand save the values to data files for use in tests. We can't just\ncompare to mpmath's gammainc in test_mpmath.TestSystematic because it\nwould take too long.\n\nNote that mpmath's gammainc is computed using hypercomb, but since it\ndoesn't allow the user to increase the maximum number of terms used in\nthe series it doesn't converge for many arguments. To get around this\nwe copy the mpmath implementation but use more terms.\n\nThis takes about 17 minutes to run on a 2.3 GHz Macbook Pro with 4GB\nram.\n\nSources:\n[1] Fredrik Johansson and others. mpmath: a Python library for\n    arbitrary-precision floating-point arithmetic (version 0.19),\n    December 2013. http://mpmath.org/.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import os' statement (line 22)
import os

import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from time import time' statement (line 23)
try:
    from time import time

except:
    time = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'time', None, module_type_store, ['time'], [time])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import numpy' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/_precompute/')
import_564082 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy')

if (type(import_564082) is not StypyTypeError):

    if (import_564082 != 'pyd_module'):
        __import__(import_564082)
        sys_modules_564083 = sys.modules[import_564082]
        import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'np', sys_modules_564083.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy', import_564082)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/_precompute/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy import pi' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/_precompute/')
import_564084 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy')

if (type(import_564084) is not StypyTypeError):

    if (import_564084 != 'pyd_module'):
        __import__(import_564084)
        sys_modules_564085 = sys.modules[import_564084]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy', sys_modules_564085.module_type_store, module_type_store, ['pi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_564085, sys_modules_564085.module_type_store, module_type_store)
    else:
        from numpy import pi

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy', None, module_type_store, ['pi'], [pi])

else:
    # Assigning a type to the variable 'numpy' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy', import_564084)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/_precompute/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from scipy.special._mptestutils import mpf2float' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/_precompute/')
import_564086 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.special._mptestutils')

if (type(import_564086) is not StypyTypeError):

    if (import_564086 != 'pyd_module'):
        __import__(import_564086)
        sys_modules_564087 = sys.modules[import_564086]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.special._mptestutils', sys_modules_564087.module_type_store, module_type_store, ['mpf2float'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_564087, sys_modules_564087.module_type_store, module_type_store)
    else:
        from scipy.special._mptestutils import mpf2float

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.special._mptestutils', None, module_type_store, ['mpf2float'], [mpf2float])

else:
    # Assigning a type to the variable 'scipy.special._mptestutils' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.special._mptestutils', import_564086)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/_precompute/')



# SSA begins for try-except statement (line 29)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 4))

# 'import mpmath' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/_precompute/')
import_564088 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 4), 'mpmath')

if (type(import_564088) is not StypyTypeError):

    if (import_564088 != 'pyd_module'):
        __import__(import_564088)
        sys_modules_564089 = sys.modules[import_564088]
        import_module(stypy.reporting.localization.Localization(__file__, 30, 4), 'mp', sys_modules_564089.module_type_store, module_type_store)
    else:
        import mpmath as mp

        import_module(stypy.reporting.localization.Localization(__file__, 30, 4), 'mp', mpmath, module_type_store)

else:
    # Assigning a type to the variable 'mpmath' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'mpmath', import_564088)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/_precompute/')

# SSA branch for the except part of a try statement (line 29)
# SSA branch for the except 'ImportError' branch of a try statement (line 29)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 29)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def gammainc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_564090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'int')
    int_564091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'int')
    int_564092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 40), 'int')
    # Applying the binary operator '**' (line 35)
    result_pow_564093 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 36), '**', int_564091, int_564092)
    
    defaults = [int_564090, result_pow_564093]
    # Create a new context for function 'gammainc'
    module_type_store = module_type_store.open_function_context('gammainc', 35, 0, False)
    
    # Passed parameters checking function
    gammainc.stypy_localization = localization
    gammainc.stypy_type_of_self = None
    gammainc.stypy_type_store = module_type_store
    gammainc.stypy_function_name = 'gammainc'
    gammainc.stypy_param_names_list = ['a', 'x', 'dps', 'maxterms']
    gammainc.stypy_varargs_param_name = None
    gammainc.stypy_kwargs_param_name = None
    gammainc.stypy_call_defaults = defaults
    gammainc.stypy_call_varargs = varargs
    gammainc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gammainc', ['a', 'x', 'dps', 'maxterms'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gammainc', localization, ['a', 'x', 'dps', 'maxterms'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gammainc(...)' code ##################

    str_564094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', 'Compute gammainc exactly like mpmath does but allow for more\n    summands in hypercomb. See\n\n    mpmath/functions/expintegrals.py#L134\n    \n    in the mpmath github repository.\n\n    ')
    
    # Call to workdps(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'dps' (line 44)
    dps_564097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'dps', False)
    # Processing the call keyword arguments (line 44)
    kwargs_564098 = {}
    # Getting the type of 'mp' (line 44)
    mp_564095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 9), 'mp', False)
    # Obtaining the member 'workdps' of a type (line 44)
    workdps_564096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 9), mp_564095, 'workdps')
    # Calling workdps(args, kwargs) (line 44)
    workdps_call_result_564099 = invoke(stypy.reporting.localization.Localization(__file__, 44, 9), workdps_564096, *[dps_564097], **kwargs_564098)
    
    with_564100 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 44, 9), workdps_call_result_564099, 'with parameter', '__enter__', '__exit__')

    if with_564100:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 44)
        enter___564101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 9), workdps_call_result_564099, '__enter__')
        with_enter_564102 = invoke(stypy.reporting.localization.Localization(__file__, 44, 9), enter___564101)
        
        # Assigning a Tuple to a Tuple (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to mpf(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'a' (line 45)
        a_564105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'a', False)
        # Processing the call keyword arguments (line 45)
        kwargs_564106 = {}
        # Getting the type of 'mp' (line 45)
        mp_564103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 45)
        mpf_564104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 18), mp_564103, 'mpf')
        # Calling mpf(args, kwargs) (line 45)
        mpf_call_result_564107 = invoke(stypy.reporting.localization.Localization(__file__, 45, 18), mpf_564104, *[a_564105], **kwargs_564106)
        
        # Assigning a type to the variable 'tuple_assignment_564068' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'tuple_assignment_564068', mpf_call_result_564107)
        
        # Assigning a Call to a Name (line 45):
        
        # Call to mpf(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'x' (line 45)
        x_564110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'x', False)
        # Processing the call keyword arguments (line 45)
        kwargs_564111 = {}
        # Getting the type of 'mp' (line 45)
        mp_564108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 45)
        mpf_564109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 29), mp_564108, 'mpf')
        # Calling mpf(args, kwargs) (line 45)
        mpf_call_result_564112 = invoke(stypy.reporting.localization.Localization(__file__, 45, 29), mpf_564109, *[x_564110], **kwargs_564111)
        
        # Assigning a type to the variable 'tuple_assignment_564069' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'tuple_assignment_564069', mpf_call_result_564112)
        
        # Assigning a Call to a Name (line 45):
        
        # Call to mpf(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'x' (line 45)
        x_564115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 47), 'x', False)
        # Processing the call keyword arguments (line 45)
        kwargs_564116 = {}
        # Getting the type of 'mp' (line 45)
        mp_564113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 40), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 45)
        mpf_564114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 40), mp_564113, 'mpf')
        # Calling mpf(args, kwargs) (line 45)
        mpf_call_result_564117 = invoke(stypy.reporting.localization.Localization(__file__, 45, 40), mpf_564114, *[x_564115], **kwargs_564116)
        
        # Assigning a type to the variable 'tuple_assignment_564070' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'tuple_assignment_564070', mpf_call_result_564117)
        
        # Assigning a Name to a Name (line 45):
        # Getting the type of 'tuple_assignment_564068' (line 45)
        tuple_assignment_564068_564118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'tuple_assignment_564068')
        # Assigning a type to the variable 'z' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'z', tuple_assignment_564068_564118)
        
        # Assigning a Name to a Name (line 45):
        # Getting the type of 'tuple_assignment_564069' (line 45)
        tuple_assignment_564069_564119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'tuple_assignment_564069')
        # Assigning a type to the variable 'a' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'a', tuple_assignment_564069_564119)
        
        # Assigning a Name to a Name (line 45):
        # Getting the type of 'tuple_assignment_564070' (line 45)
        tuple_assignment_564070_564120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'tuple_assignment_564070')
        # Assigning a type to the variable 'b' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), 'b', tuple_assignment_564070_564120)
        
        # Assigning a List to a Name (line 46):
        
        # Assigning a List to a Name (line 46):
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_564121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        # Getting the type of 'z' (line 46)
        z_564122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_564121, z_564122)
        
        # Assigning a type to the variable 'G' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'G', list_564121)
        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to fneg(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'b' (line 47)
        b_564125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'b', False)
        # Processing the call keyword arguments (line 47)
        # Getting the type of 'True' (line 47)
        True_564126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 32), 'True', False)
        keyword_564127 = True_564126
        kwargs_564128 = {'exact': keyword_564127}
        # Getting the type of 'mp' (line 47)
        mp_564123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'mp', False)
        # Obtaining the member 'fneg' of a type (line 47)
        fneg_564124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), mp_564123, 'fneg')
        # Calling fneg(args, kwargs) (line 47)
        fneg_call_result_564129 = invoke(stypy.reporting.localization.Localization(__file__, 47, 15), fneg_564124, *[b_564125], **kwargs_564128)
        
        # Assigning a type to the variable 'negb' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'negb', fneg_call_result_564129)

        @norecursion
        def h(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'h'
            module_type_store = module_type_store.open_function_context('h', 49, 8, False)
            
            # Passed parameters checking function
            h.stypy_localization = localization
            h.stypy_type_of_self = None
            h.stypy_type_store = module_type_store
            h.stypy_function_name = 'h'
            h.stypy_param_names_list = ['z']
            h.stypy_varargs_param_name = None
            h.stypy_kwargs_param_name = None
            h.stypy_call_defaults = defaults
            h.stypy_call_varargs = varargs
            h.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'h', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'h', localization, ['z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'h(...)' code ##################

            
            # Assigning a Tuple to a Name (line 50):
            
            # Assigning a Tuple to a Name (line 50):
            
            # Obtaining an instance of the builtin type 'tuple' (line 50)
            tuple_564130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 50)
            # Adding element type (line 50)
            
            # Obtaining an instance of the builtin type 'list' (line 50)
            list_564131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 50)
            # Adding element type (line 50)
            
            # Call to exp(...): (line 50)
            # Processing the call arguments (line 50)
            # Getting the type of 'negb' (line 50)
            negb_564134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'negb', False)
            # Processing the call keyword arguments (line 50)
            kwargs_564135 = {}
            # Getting the type of 'mp' (line 50)
            mp_564132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'mp', False)
            # Obtaining the member 'exp' of a type (line 50)
            exp_564133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 18), mp_564132, 'exp')
            # Calling exp(args, kwargs) (line 50)
            exp_call_result_564136 = invoke(stypy.reporting.localization.Localization(__file__, 50, 18), exp_564133, *[negb_564134], **kwargs_564135)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_564131, exp_call_result_564136)
            # Adding element type (line 50)
            # Getting the type of 'b' (line 50)
            b_564137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 32), 'b')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_564131, b_564137)
            # Adding element type (line 50)
            # Getting the type of 'z' (line 50)
            z_564138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'z')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_564131, z_564138)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), tuple_564130, list_564131)
            # Adding element type (line 50)
            
            # Obtaining an instance of the builtin type 'list' (line 50)
            list_564139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 39), 'list')
            # Adding type elements to the builtin type 'list' instance (line 50)
            # Adding element type (line 50)
            int_564140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 40), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 39), list_564139, int_564140)
            # Adding element type (line 50)
            # Getting the type of 'z' (line 50)
            z_564141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 43), 'z')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 39), list_564139, z_564141)
            # Adding element type (line 50)
            int_564142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 46), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 39), list_564139, int_564142)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), tuple_564130, list_564139)
            # Adding element type (line 50)
            
            # Obtaining an instance of the builtin type 'list' (line 50)
            list_564143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 51), 'list')
            # Adding type elements to the builtin type 'list' instance (line 50)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), tuple_564130, list_564143)
            # Adding element type (line 50)
            # Getting the type of 'G' (line 50)
            G_564144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 55), 'G')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), tuple_564130, G_564144)
            # Adding element type (line 50)
            
            # Obtaining an instance of the builtin type 'list' (line 50)
            list_564145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 58), 'list')
            # Adding type elements to the builtin type 'list' instance (line 50)
            # Adding element type (line 50)
            int_564146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 59), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 58), list_564145, int_564146)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), tuple_564130, list_564145)
            # Adding element type (line 50)
            
            # Obtaining an instance of the builtin type 'list' (line 50)
            list_564147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 63), 'list')
            # Adding type elements to the builtin type 'list' instance (line 50)
            # Adding element type (line 50)
            int_564148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 64), 'int')
            # Getting the type of 'z' (line 50)
            z_564149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 66), 'z')
            # Applying the binary operator '+' (line 50)
            result_add_564150 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 64), '+', int_564148, z_564149)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 63), list_564147, result_add_564150)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), tuple_564130, list_564147)
            # Adding element type (line 50)
            # Getting the type of 'b' (line 50)
            b_564151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 70), 'b')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), tuple_564130, b_564151)
            
            # Assigning a type to the variable 'T1' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'T1', tuple_564130)
            
            # Obtaining an instance of the builtin type 'tuple' (line 51)
            tuple_564152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 51)
            # Adding element type (line 51)
            # Getting the type of 'T1' (line 51)
            T1_564153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'T1')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_564152, T1_564153)
            
            # Assigning a type to the variable 'stypy_return_type' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'stypy_return_type', tuple_564152)
            
            # ################# End of 'h(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'h' in the type store
            # Getting the type of 'stypy_return_type' (line 49)
            stypy_return_type_564154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_564154)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'h'
            return stypy_return_type_564154

        # Assigning a type to the variable 'h' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'h', h)
        
        # Assigning a Call to a Name (line 53):
        
        # Assigning a Call to a Name (line 53):
        
        # Call to hypercomb(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'h' (line 53)
        h_564157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_564158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        # Getting the type of 'z' (line 53)
        z_564159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'z', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 30), list_564158, z_564159)
        
        # Processing the call keyword arguments (line 53)
        # Getting the type of 'maxterms' (line 53)
        maxterms_564160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 44), 'maxterms', False)
        keyword_564161 = maxterms_564160
        kwargs_564162 = {'maxterms': keyword_564161}
        # Getting the type of 'mp' (line 53)
        mp_564155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'mp', False)
        # Obtaining the member 'hypercomb' of a type (line 53)
        hypercomb_564156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 14), mp_564155, 'hypercomb')
        # Calling hypercomb(args, kwargs) (line 53)
        hypercomb_call_result_564163 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), hypercomb_564156, *[h_564157, list_564158], **kwargs_564162)
        
        # Assigning a type to the variable 'res' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'res', hypercomb_call_result_564163)
        
        # Call to mpf2float(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'res' (line 54)
        res_564165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'res', False)
        # Processing the call keyword arguments (line 54)
        kwargs_564166 = {}
        # Getting the type of 'mpf2float' (line 54)
        mpf2float_564164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'mpf2float', False)
        # Calling mpf2float(args, kwargs) (line 54)
        mpf2float_call_result_564167 = invoke(stypy.reporting.localization.Localization(__file__, 54, 15), mpf2float_564164, *[res_564165], **kwargs_564166)
        
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type', mpf2float_call_result_564167)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 44)
        exit___564168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 9), workdps_call_result_564099, '__exit__')
        with_exit_564169 = invoke(stypy.reporting.localization.Localization(__file__, 44, 9), exit___564168, None, None, None)

    
    # ################# End of 'gammainc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gammainc' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_564170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_564170)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gammainc'
    return stypy_return_type_564170

# Assigning a type to the variable 'gammainc' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'gammainc', gammainc)

@norecursion
def gammaincc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_564171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 24), 'int')
    int_564172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 37), 'int')
    int_564173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 41), 'int')
    # Applying the binary operator '**' (line 57)
    result_pow_564174 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 37), '**', int_564172, int_564173)
    
    defaults = [int_564171, result_pow_564174]
    # Create a new context for function 'gammaincc'
    module_type_store = module_type_store.open_function_context('gammaincc', 57, 0, False)
    
    # Passed parameters checking function
    gammaincc.stypy_localization = localization
    gammaincc.stypy_type_of_self = None
    gammaincc.stypy_type_store = module_type_store
    gammaincc.stypy_function_name = 'gammaincc'
    gammaincc.stypy_param_names_list = ['a', 'x', 'dps', 'maxterms']
    gammaincc.stypy_varargs_param_name = None
    gammaincc.stypy_kwargs_param_name = None
    gammaincc.stypy_call_defaults = defaults
    gammaincc.stypy_call_varargs = varargs
    gammaincc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gammaincc', ['a', 'x', 'dps', 'maxterms'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gammaincc', localization, ['a', 'x', 'dps', 'maxterms'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gammaincc(...)' code ##################

    str_564175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, (-1)), 'str', 'Compute gammaincc exactly like mpmath does but allow for more\n    terms in hypercomb. See\n\n    mpmath/functions/expintegrals.py#L187\n\n    in the mpmath github repository.\n\n    ')
    
    # Call to workdps(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'dps' (line 66)
    dps_564178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'dps', False)
    # Processing the call keyword arguments (line 66)
    kwargs_564179 = {}
    # Getting the type of 'mp' (line 66)
    mp_564176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 9), 'mp', False)
    # Obtaining the member 'workdps' of a type (line 66)
    workdps_564177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 9), mp_564176, 'workdps')
    # Calling workdps(args, kwargs) (line 66)
    workdps_call_result_564180 = invoke(stypy.reporting.localization.Localization(__file__, 66, 9), workdps_564177, *[dps_564178], **kwargs_564179)
    
    with_564181 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 66, 9), workdps_call_result_564180, 'with parameter', '__enter__', '__exit__')

    if with_564181:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 66)
        enter___564182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 9), workdps_call_result_564180, '__enter__')
        with_enter_564183 = invoke(stypy.reporting.localization.Localization(__file__, 66, 9), enter___564182)
        
        # Assigning a Tuple to a Tuple (line 67):
        
        # Assigning a Name to a Name (line 67):
        # Getting the type of 'a' (line 67)
        a_564184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'a')
        # Assigning a type to the variable 'tuple_assignment_564071' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'tuple_assignment_564071', a_564184)
        
        # Assigning a Name to a Name (line 67):
        # Getting the type of 'x' (line 67)
        x_564185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'x')
        # Assigning a type to the variable 'tuple_assignment_564072' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'tuple_assignment_564072', x_564185)
        
        # Assigning a Name to a Name (line 67):
        # Getting the type of 'tuple_assignment_564071' (line 67)
        tuple_assignment_564071_564186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'tuple_assignment_564071')
        # Assigning a type to the variable 'z' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'z', tuple_assignment_564071_564186)
        
        # Assigning a Name to a Name (line 67):
        # Getting the type of 'tuple_assignment_564072' (line 67)
        tuple_assignment_564072_564187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'tuple_assignment_564072')
        # Assigning a type to the variable 'a' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'a', tuple_assignment_564072_564187)
        
        
        # Call to isint(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'z' (line 69)
        z_564190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'z', False)
        # Processing the call keyword arguments (line 69)
        kwargs_564191 = {}
        # Getting the type of 'mp' (line 69)
        mp_564188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'mp', False)
        # Obtaining the member 'isint' of a type (line 69)
        isint_564189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 11), mp_564188, 'isint')
        # Calling isint(args, kwargs) (line 69)
        isint_call_result_564192 = invoke(stypy.reporting.localization.Localization(__file__, 69, 11), isint_564189, *[z_564190], **kwargs_564191)
        
        # Testing the type of an if condition (line 69)
        if_condition_564193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), isint_call_result_564192)
        # Assigning a type to the variable 'if_condition_564193' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_564193', if_condition_564193)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to mpf2float(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to gammainc(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'z' (line 72)
        z_564197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 45), 'z', False)
        # Processing the call keyword arguments (line 72)
        # Getting the type of 'a' (line 72)
        a_564198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 50), 'a', False)
        keyword_564199 = a_564198
        # Getting the type of 'True' (line 72)
        True_564200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 65), 'True', False)
        keyword_564201 = True_564200
        kwargs_564202 = {'a': keyword_564199, 'regularized': keyword_564201}
        # Getting the type of 'mp' (line 72)
        mp_564195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 33), 'mp', False)
        # Obtaining the member 'gammainc' of a type (line 72)
        gammainc_564196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 33), mp_564195, 'gammainc')
        # Calling gammainc(args, kwargs) (line 72)
        gammainc_call_result_564203 = invoke(stypy.reporting.localization.Localization(__file__, 72, 33), gammainc_564196, *[z_564197], **kwargs_564202)
        
        # Processing the call keyword arguments (line 72)
        kwargs_564204 = {}
        # Getting the type of 'mpf2float' (line 72)
        mpf2float_564194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'mpf2float', False)
        # Calling mpf2float(args, kwargs) (line 72)
        mpf2float_call_result_564205 = invoke(stypy.reporting.localization.Localization(__file__, 72, 23), mpf2float_564194, *[gammainc_call_result_564203], **kwargs_564204)
        
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'stypy_return_type', mpf2float_call_result_564205)
        # SSA branch for the except part of a try statement (line 70)
        # SSA branch for the except 'Attribute' branch of a try statement (line 70)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 70)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to fneg(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'a' (line 75)
        a_564208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'a', False)
        # Processing the call keyword arguments (line 75)
        # Getting the type of 'True' (line 75)
        True_564209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'True', False)
        keyword_564210 = True_564209
        kwargs_564211 = {'exact': keyword_564210}
        # Getting the type of 'mp' (line 75)
        mp_564206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'mp', False)
        # Obtaining the member 'fneg' of a type (line 75)
        fneg_564207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), mp_564206, 'fneg')
        # Calling fneg(args, kwargs) (line 75)
        fneg_call_result_564212 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), fneg_564207, *[a_564208], **kwargs_564211)
        
        # Assigning a type to the variable 'nega' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'nega', fneg_call_result_564212)
        
        # Assigning a List to a Name (line 76):
        
        # Assigning a List to a Name (line 76):
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_564213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        # Getting the type of 'z' (line 76)
        z_564214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 12), list_564213, z_564214)
        
        # Assigning a type to the variable 'G' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'G', list_564213)
        
        
        # SSA begins for try-except statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

        @norecursion
        def h(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'h'
            module_type_store = module_type_store.open_function_context('h', 79, 12, False)
            
            # Passed parameters checking function
            h.stypy_localization = localization
            h.stypy_type_of_self = None
            h.stypy_type_store = module_type_store
            h.stypy_function_name = 'h'
            h.stypy_param_names_list = ['z']
            h.stypy_varargs_param_name = None
            h.stypy_kwargs_param_name = None
            h.stypy_call_defaults = defaults
            h.stypy_call_varargs = varargs
            h.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'h', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'h', localization, ['z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'h(...)' code ##################

            
            # Assigning a BinOp to a Name (line 80):
            
            # Assigning a BinOp to a Name (line 80):
            # Getting the type of 'z' (line 80)
            z_564215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'z')
            int_564216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 22), 'int')
            # Applying the binary operator '-' (line 80)
            result_sub_564217 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 20), '-', z_564215, int_564216)
            
            # Assigning a type to the variable 'r' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'r', result_sub_564217)
            
            # Obtaining an instance of the builtin type 'list' (line 81)
            list_564218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 81)
            # Adding element type (line 81)
            
            # Obtaining an instance of the builtin type 'tuple' (line 81)
            tuple_564219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 81)
            # Adding element type (line 81)
            
            # Obtaining an instance of the builtin type 'list' (line 81)
            list_564220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'list')
            # Adding type elements to the builtin type 'list' instance (line 81)
            # Adding element type (line 81)
            
            # Call to exp(...): (line 81)
            # Processing the call arguments (line 81)
            # Getting the type of 'nega' (line 81)
            nega_564223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), 'nega', False)
            # Processing the call keyword arguments (line 81)
            kwargs_564224 = {}
            # Getting the type of 'mp' (line 81)
            mp_564221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'mp', False)
            # Obtaining the member 'exp' of a type (line 81)
            exp_564222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 26), mp_564221, 'exp')
            # Calling exp(args, kwargs) (line 81)
            exp_call_result_564225 = invoke(stypy.reporting.localization.Localization(__file__, 81, 26), exp_564222, *[nega_564223], **kwargs_564224)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), list_564220, exp_call_result_564225)
            # Adding element type (line 81)
            # Getting the type of 'a' (line 81)
            a_564226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 40), 'a')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), list_564220, a_564226)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), tuple_564219, list_564220)
            # Adding element type (line 81)
            
            # Obtaining an instance of the builtin type 'list' (line 81)
            list_564227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 44), 'list')
            # Adding type elements to the builtin type 'list' instance (line 81)
            # Adding element type (line 81)
            int_564228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 45), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 44), list_564227, int_564228)
            # Adding element type (line 81)
            # Getting the type of 'r' (line 81)
            r_564229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 48), 'r')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 44), list_564227, r_564229)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), tuple_564219, list_564227)
            # Adding element type (line 81)
            
            # Obtaining an instance of the builtin type 'list' (line 81)
            list_564230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 52), 'list')
            # Adding type elements to the builtin type 'list' instance (line 81)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), tuple_564219, list_564230)
            # Adding element type (line 81)
            # Getting the type of 'G' (line 81)
            G_564231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 56), 'G')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), tuple_564219, G_564231)
            # Adding element type (line 81)
            
            # Obtaining an instance of the builtin type 'list' (line 81)
            list_564232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 59), 'list')
            # Adding type elements to the builtin type 'list' instance (line 81)
            # Adding element type (line 81)
            int_564233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 60), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 59), list_564232, int_564233)
            # Adding element type (line 81)
            
            # Getting the type of 'r' (line 81)
            r_564234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 64), 'r')
            # Applying the 'usub' unary operator (line 81)
            result___neg___564235 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 63), 'usub', r_564234)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 59), list_564232, result___neg___564235)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), tuple_564219, list_564232)
            # Adding element type (line 81)
            
            # Obtaining an instance of the builtin type 'list' (line 81)
            list_564236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 68), 'list')
            # Adding type elements to the builtin type 'list' instance (line 81)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), tuple_564219, list_564236)
            # Adding element type (line 81)
            int_564237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 72), 'int')
            # Getting the type of 'nega' (line 81)
            nega_564238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 74), 'nega')
            # Applying the binary operator 'div' (line 81)
            result_div_564239 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 72), 'div', int_564237, nega_564238)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), tuple_564219, result_div_564239)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 23), list_564218, tuple_564219)
            
            # Assigning a type to the variable 'stypy_return_type' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'stypy_return_type', list_564218)
            
            # ################# End of 'h(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'h' in the type store
            # Getting the type of 'stypy_return_type' (line 79)
            stypy_return_type_564240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_564240)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'h'
            return stypy_return_type_564240

        # Assigning a type to the variable 'h' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'h', h)
        
        # Call to mpf2float(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Call to hypercomb(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'h' (line 82)
        h_564244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 42), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_564245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        # Getting the type of 'z' (line 82)
        z_564246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 46), 'z', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 45), list_564245, z_564246)
        
        # Processing the call keyword arguments (line 82)
        # Getting the type of 'True' (line 82)
        True_564247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 63), 'True', False)
        keyword_564248 = True_564247
        kwargs_564249 = {'force_series': keyword_564248}
        # Getting the type of 'mp' (line 82)
        mp_564242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'mp', False)
        # Obtaining the member 'hypercomb' of a type (line 82)
        hypercomb_564243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 29), mp_564242, 'hypercomb')
        # Calling hypercomb(args, kwargs) (line 82)
        hypercomb_call_result_564250 = invoke(stypy.reporting.localization.Localization(__file__, 82, 29), hypercomb_564243, *[h_564244, list_564245], **kwargs_564249)
        
        # Processing the call keyword arguments (line 82)
        kwargs_564251 = {}
        # Getting the type of 'mpf2float' (line 82)
        mpf2float_564241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'mpf2float', False)
        # Calling mpf2float(args, kwargs) (line 82)
        mpf2float_call_result_564252 = invoke(stypy.reporting.localization.Localization(__file__, 82, 19), mpf2float_564241, *[hypercomb_call_result_564250], **kwargs_564251)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'stypy_return_type', mpf2float_call_result_564252)
        # SSA branch for the except part of a try statement (line 78)
        # SSA branch for the except 'Attribute' branch of a try statement (line 78)
        module_type_store.open_ssa_branch('except')

        @norecursion
        def h(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'h'
            module_type_store = module_type_store.open_function_context('h', 84, 12, False)
            
            # Passed parameters checking function
            h.stypy_localization = localization
            h.stypy_type_of_self = None
            h.stypy_type_store = module_type_store
            h.stypy_function_name = 'h'
            h.stypy_param_names_list = ['z']
            h.stypy_varargs_param_name = None
            h.stypy_kwargs_param_name = None
            h.stypy_call_defaults = defaults
            h.stypy_call_varargs = varargs
            h.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'h', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'h', localization, ['z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'h(...)' code ##################

            
            # Assigning a Tuple to a Name (line 85):
            
            # Assigning a Tuple to a Name (line 85):
            
            # Obtaining an instance of the builtin type 'tuple' (line 85)
            tuple_564253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 85)
            # Adding element type (line 85)
            
            # Obtaining an instance of the builtin type 'list' (line 85)
            list_564254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'list')
            # Adding type elements to the builtin type 'list' instance (line 85)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_564253, list_564254)
            # Adding element type (line 85)
            
            # Obtaining an instance of the builtin type 'list' (line 85)
            list_564255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'list')
            # Adding type elements to the builtin type 'list' instance (line 85)
            # Adding element type (line 85)
            int_564256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 26), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 25), list_564255, int_564256)
            # Adding element type (line 85)
            # Getting the type of 'z' (line 85)
            z_564257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'z')
            int_564258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 31), 'int')
            # Applying the binary operator '-' (line 85)
            result_sub_564259 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 29), '-', z_564257, int_564258)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 25), list_564255, result_sub_564259)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_564253, list_564255)
            # Adding element type (line 85)
            
            # Obtaining an instance of the builtin type 'list' (line 85)
            list_564260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'list')
            # Adding type elements to the builtin type 'list' instance (line 85)
            # Adding element type (line 85)
            # Getting the type of 'z' (line 85)
            z_564261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'z')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 35), list_564260, z_564261)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_564253, list_564260)
            # Adding element type (line 85)
            # Getting the type of 'G' (line 85)
            G_564262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 40), 'G')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_564253, G_564262)
            # Adding element type (line 85)
            
            # Obtaining an instance of the builtin type 'list' (line 85)
            list_564263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 43), 'list')
            # Adding type elements to the builtin type 'list' instance (line 85)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_564253, list_564263)
            # Adding element type (line 85)
            
            # Obtaining an instance of the builtin type 'list' (line 85)
            list_564264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 47), 'list')
            # Adding type elements to the builtin type 'list' instance (line 85)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_564253, list_564264)
            # Adding element type (line 85)
            int_564265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 51), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_564253, int_564265)
            
            # Assigning a type to the variable 'T1' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'T1', tuple_564253)
            
            # Assigning a Tuple to a Name (line 86):
            
            # Assigning a Tuple to a Name (line 86):
            
            # Obtaining an instance of the builtin type 'tuple' (line 86)
            tuple_564266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 86)
            # Adding element type (line 86)
            
            # Obtaining an instance of the builtin type 'list' (line 86)
            list_564267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'list')
            # Adding type elements to the builtin type 'list' instance (line 86)
            # Adding element type (line 86)
            
            
            # Call to exp(...): (line 86)
            # Processing the call arguments (line 86)
            # Getting the type of 'nega' (line 86)
            nega_564270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'nega', False)
            # Processing the call keyword arguments (line 86)
            kwargs_564271 = {}
            # Getting the type of 'mp' (line 86)
            mp_564268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'mp', False)
            # Obtaining the member 'exp' of a type (line 86)
            exp_564269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 23), mp_564268, 'exp')
            # Calling exp(args, kwargs) (line 86)
            exp_call_result_564272 = invoke(stypy.reporting.localization.Localization(__file__, 86, 23), exp_564269, *[nega_564270], **kwargs_564271)
            
            # Applying the 'usub' unary operator (line 86)
            result___neg___564273 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 22), 'usub', exp_call_result_564272)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), list_564267, result___neg___564273)
            # Adding element type (line 86)
            # Getting the type of 'a' (line 86)
            a_564274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 37), 'a')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), list_564267, a_564274)
            # Adding element type (line 86)
            # Getting the type of 'z' (line 86)
            z_564275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 40), 'z')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), list_564267, z_564275)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), tuple_564266, list_564267)
            # Adding element type (line 86)
            
            # Obtaining an instance of the builtin type 'list' (line 86)
            list_564276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 44), 'list')
            # Adding type elements to the builtin type 'list' instance (line 86)
            # Adding element type (line 86)
            int_564277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 45), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 44), list_564276, int_564277)
            # Adding element type (line 86)
            # Getting the type of 'z' (line 86)
            z_564278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 48), 'z')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 44), list_564276, z_564278)
            # Adding element type (line 86)
            int_564279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 51), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 44), list_564276, int_564279)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), tuple_564266, list_564276)
            # Adding element type (line 86)
            
            # Obtaining an instance of the builtin type 'list' (line 86)
            list_564280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 56), 'list')
            # Adding type elements to the builtin type 'list' instance (line 86)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), tuple_564266, list_564280)
            # Adding element type (line 86)
            # Getting the type of 'G' (line 86)
            G_564281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 60), 'G')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), tuple_564266, G_564281)
            # Adding element type (line 86)
            
            # Obtaining an instance of the builtin type 'list' (line 86)
            list_564282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 63), 'list')
            # Adding type elements to the builtin type 'list' instance (line 86)
            # Adding element type (line 86)
            int_564283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 64), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 63), list_564282, int_564283)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), tuple_564266, list_564282)
            # Adding element type (line 86)
            
            # Obtaining an instance of the builtin type 'list' (line 86)
            list_564284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 68), 'list')
            # Adding type elements to the builtin type 'list' instance (line 86)
            # Adding element type (line 86)
            int_564285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 69), 'int')
            # Getting the type of 'z' (line 86)
            z_564286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 71), 'z')
            # Applying the binary operator '+' (line 86)
            result_add_564287 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 69), '+', int_564285, z_564286)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 68), list_564284, result_add_564287)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), tuple_564266, list_564284)
            # Adding element type (line 86)
            # Getting the type of 'a' (line 86)
            a_564288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 75), 'a')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 21), tuple_564266, a_564288)
            
            # Assigning a type to the variable 'T2' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'T2', tuple_564266)
            
            # Obtaining an instance of the builtin type 'tuple' (line 87)
            tuple_564289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 87)
            # Adding element type (line 87)
            # Getting the type of 'T1' (line 87)
            T1_564290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'T1')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 23), tuple_564289, T1_564290)
            # Adding element type (line 87)
            # Getting the type of 'T2' (line 87)
            T2_564291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 27), 'T2')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 23), tuple_564289, T2_564291)
            
            # Assigning a type to the variable 'stypy_return_type' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'stypy_return_type', tuple_564289)
            
            # ################# End of 'h(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'h' in the type store
            # Getting the type of 'stypy_return_type' (line 84)
            stypy_return_type_564292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_564292)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'h'
            return stypy_return_type_564292

        # Assigning a type to the variable 'h' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'h', h)
        
        # Call to mpf2float(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to hypercomb(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'h' (line 88)
        h_564296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_564297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        # Getting the type of 'z' (line 88)
        z_564298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 46), 'z', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 45), list_564297, z_564298)
        
        # Processing the call keyword arguments (line 88)
        # Getting the type of 'maxterms' (line 88)
        maxterms_564299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 59), 'maxterms', False)
        keyword_564300 = maxterms_564299
        kwargs_564301 = {'maxterms': keyword_564300}
        # Getting the type of 'mp' (line 88)
        mp_564294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'mp', False)
        # Obtaining the member 'hypercomb' of a type (line 88)
        hypercomb_564295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 29), mp_564294, 'hypercomb')
        # Calling hypercomb(args, kwargs) (line 88)
        hypercomb_call_result_564302 = invoke(stypy.reporting.localization.Localization(__file__, 88, 29), hypercomb_564295, *[h_564296, list_564297], **kwargs_564301)
        
        # Processing the call keyword arguments (line 88)
        kwargs_564303 = {}
        # Getting the type of 'mpf2float' (line 88)
        mpf2float_564293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'mpf2float', False)
        # Calling mpf2float(args, kwargs) (line 88)
        mpf2float_call_result_564304 = invoke(stypy.reporting.localization.Localization(__file__, 88, 19), mpf2float_564293, *[hypercomb_call_result_564302], **kwargs_564303)
        
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'stypy_return_type', mpf2float_call_result_564304)
        # SSA join for try-except statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 66)
        exit___564305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 9), workdps_call_result_564180, '__exit__')
        with_exit_564306 = invoke(stypy.reporting.localization.Localization(__file__, 66, 9), exit___564305, None, None, None)

    
    # ################# End of 'gammaincc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gammaincc' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_564307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_564307)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gammaincc'
    return stypy_return_type_564307

# Assigning a type to the variable 'gammaincc' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'gammaincc', gammaincc)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 91, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = []
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to time(...): (line 92)
    # Processing the call keyword arguments (line 92)
    kwargs_564309 = {}
    # Getting the type of 'time' (line 92)
    time_564308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 9), 'time', False)
    # Calling time(args, kwargs) (line 92)
    time_call_result_564310 = invoke(stypy.reporting.localization.Localization(__file__, 92, 9), time_564308, *[], **kwargs_564309)
    
    # Assigning a type to the variable 't0' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 't0', time_call_result_564310)
    
    # Call to print(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of '__doc__' (line 98)
    doc___564312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 10), '__doc__', False)
    # Processing the call keyword arguments (line 98)
    kwargs_564313 = {}
    # Getting the type of 'print' (line 98)
    print_564311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'print', False)
    # Calling print(args, kwargs) (line 98)
    print_call_result_564314 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), print_564311, *[doc___564312], **kwargs_564313)
    
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to dirname(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of '__file__' (line 99)
    file___564318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), '__file__', False)
    # Processing the call keyword arguments (line 99)
    kwargs_564319 = {}
    # Getting the type of 'os' (line 99)
    os_564315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 10), 'os', False)
    # Obtaining the member 'path' of a type (line 99)
    path_564316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 10), os_564315, 'path')
    # Obtaining the member 'dirname' of a type (line 99)
    dirname_564317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 10), path_564316, 'dirname')
    # Calling dirname(args, kwargs) (line 99)
    dirname_call_result_564320 = invoke(stypy.reporting.localization.Localization(__file__, 99, 10), dirname_564317, *[file___564318], **kwargs_564319)
    
    # Assigning a type to the variable 'pwd' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'pwd', dirname_call_result_564320)
    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to logspace(...): (line 100)
    # Processing the call arguments (line 100)
    int_564323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 20), 'int')
    int_564324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'int')
    int_564325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 27), 'int')
    # Processing the call keyword arguments (line 100)
    kwargs_564326 = {}
    # Getting the type of 'np' (line 100)
    np_564321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'np', False)
    # Obtaining the member 'logspace' of a type (line 100)
    logspace_564322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), np_564321, 'logspace')
    # Calling logspace(args, kwargs) (line 100)
    logspace_call_result_564327 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), logspace_564322, *[int_564323, int_564324, int_564325], **kwargs_564326)
    
    # Assigning a type to the variable 'r' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'r', logspace_call_result_564327)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to logspace(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Call to log10(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'pi' (line 101)
    pi_564332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 34), 'pi', False)
    int_564333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 37), 'int')
    # Applying the binary operator 'div' (line 101)
    result_div_564334 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 34), 'div', pi_564332, int_564333)
    
    # Processing the call keyword arguments (line 101)
    kwargs_564335 = {}
    # Getting the type of 'np' (line 101)
    np_564330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 25), 'np', False)
    # Obtaining the member 'log10' of a type (line 101)
    log10_564331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 25), np_564330, 'log10')
    # Calling log10(args, kwargs) (line 101)
    log10_call_result_564336 = invoke(stypy.reporting.localization.Localization(__file__, 101, 25), log10_564331, *[result_div_564334], **kwargs_564335)
    
    
    # Call to log10(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Call to arctan(...): (line 101)
    # Processing the call arguments (line 101)
    float_564341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 60), 'float')
    # Processing the call keyword arguments (line 101)
    kwargs_564342 = {}
    # Getting the type of 'np' (line 101)
    np_564339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'np', False)
    # Obtaining the member 'arctan' of a type (line 101)
    arctan_564340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 50), np_564339, 'arctan')
    # Calling arctan(args, kwargs) (line 101)
    arctan_call_result_564343 = invoke(stypy.reporting.localization.Localization(__file__, 101, 50), arctan_564340, *[float_564341], **kwargs_564342)
    
    # Processing the call keyword arguments (line 101)
    kwargs_564344 = {}
    # Getting the type of 'np' (line 101)
    np_564337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 41), 'np', False)
    # Obtaining the member 'log10' of a type (line 101)
    log10_564338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 41), np_564337, 'log10')
    # Calling log10(args, kwargs) (line 101)
    log10_call_result_564345 = invoke(stypy.reporting.localization.Localization(__file__, 101, 41), log10_564338, *[arctan_call_result_564343], **kwargs_564344)
    
    int_564346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 67), 'int')
    # Processing the call keyword arguments (line 101)
    kwargs_564347 = {}
    # Getting the type of 'np' (line 101)
    np_564328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'np', False)
    # Obtaining the member 'logspace' of a type (line 101)
    logspace_564329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), np_564328, 'logspace')
    # Calling logspace(args, kwargs) (line 101)
    logspace_call_result_564348 = invoke(stypy.reporting.localization.Localization(__file__, 101, 13), logspace_564329, *[log10_call_result_564336, log10_call_result_564345, int_564346], **kwargs_564347)
    
    # Assigning a type to the variable 'ltheta' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'ltheta', logspace_call_result_564348)
    
    # Assigning a Call to a Name (line 102):
    
    # Assigning a Call to a Name (line 102):
    
    # Call to logspace(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Call to log10(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'pi' (line 102)
    pi_564353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'pi', False)
    int_564354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 37), 'int')
    # Applying the binary operator 'div' (line 102)
    result_div_564355 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 34), 'div', pi_564353, int_564354)
    
    # Processing the call keyword arguments (line 102)
    kwargs_564356 = {}
    # Getting the type of 'np' (line 102)
    np_564351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'np', False)
    # Obtaining the member 'log10' of a type (line 102)
    log10_564352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 25), np_564351, 'log10')
    # Calling log10(args, kwargs) (line 102)
    log10_call_result_564357 = invoke(stypy.reporting.localization.Localization(__file__, 102, 25), log10_564352, *[result_div_564355], **kwargs_564356)
    
    
    # Call to log10(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Call to arctan(...): (line 102)
    # Processing the call arguments (line 102)
    float_564362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 60), 'float')
    # Processing the call keyword arguments (line 102)
    kwargs_564363 = {}
    # Getting the type of 'np' (line 102)
    np_564360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 50), 'np', False)
    # Obtaining the member 'arctan' of a type (line 102)
    arctan_564361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 50), np_564360, 'arctan')
    # Calling arctan(args, kwargs) (line 102)
    arctan_call_result_564364 = invoke(stypy.reporting.localization.Localization(__file__, 102, 50), arctan_564361, *[float_564362], **kwargs_564363)
    
    # Processing the call keyword arguments (line 102)
    kwargs_564365 = {}
    # Getting the type of 'np' (line 102)
    np_564358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 41), 'np', False)
    # Obtaining the member 'log10' of a type (line 102)
    log10_564359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 41), np_564358, 'log10')
    # Calling log10(args, kwargs) (line 102)
    log10_call_result_564366 = invoke(stypy.reporting.localization.Localization(__file__, 102, 41), log10_564359, *[arctan_call_result_564364], **kwargs_564365)
    
    int_564367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 67), 'int')
    # Processing the call keyword arguments (line 102)
    kwargs_564368 = {}
    # Getting the type of 'np' (line 102)
    np_564349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'np', False)
    # Obtaining the member 'logspace' of a type (line 102)
    logspace_564350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), np_564349, 'logspace')
    # Calling logspace(args, kwargs) (line 102)
    logspace_call_result_564369 = invoke(stypy.reporting.localization.Localization(__file__, 102, 13), logspace_564350, *[log10_call_result_564357, log10_call_result_564366, int_564367], **kwargs_564368)
    
    # Assigning a type to the variable 'utheta' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'utheta', logspace_call_result_564369)
    
    # Assigning a List to a Name (line 104):
    
    # Assigning a List to a Name (line 104):
    
    # Obtaining an instance of the builtin type 'list' (line 104)
    list_564370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 104)
    # Adding element type (line 104)
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_564371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    # Getting the type of 'gammainc' (line 104)
    gammainc_564372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'gammainc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 16), tuple_564371, gammainc_564372)
    # Adding element type (line 104)
    # Getting the type of 'ltheta' (line 104)
    ltheta_564373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'ltheta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 16), tuple_564371, ltheta_564373)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 14), list_564370, tuple_564371)
    # Adding element type (line 104)
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_564374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    # Getting the type of 'gammaincc' (line 104)
    gammaincc_564375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 36), 'gammaincc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 36), tuple_564374, gammaincc_564375)
    # Adding element type (line 104)
    # Getting the type of 'utheta' (line 104)
    utheta_564376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 47), 'utheta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 36), tuple_564374, utheta_564376)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 14), list_564370, tuple_564374)
    
    # Assigning a type to the variable 'regimes' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'regimes', list_564370)
    
    # Getting the type of 'regimes' (line 105)
    regimes_564377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'regimes')
    # Testing the type of a for loop iterable (line 105)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 4), regimes_564377)
    # Getting the type of the for loop variable (line 105)
    for_loop_var_564378 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 4), regimes_564377)
    # Assigning a type to the variable 'func' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 4), for_loop_var_564378))
    # Assigning a type to the variable 'theta' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'theta', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 4), for_loop_var_564378))
    # SSA begins for a for statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 106):
    
    # Assigning a Subscript to a Name (line 106):
    
    # Obtaining the type of the subscript
    int_564379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
    
    # Call to meshgrid(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'r' (line 106)
    r_564382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'r', False)
    # Getting the type of 'theta' (line 106)
    theta_564383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'theta', False)
    # Processing the call keyword arguments (line 106)
    kwargs_564384 = {}
    # Getting the type of 'np' (line 106)
    np_564380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 106)
    meshgrid_564381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 21), np_564380, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 106)
    meshgrid_call_result_564385 = invoke(stypy.reporting.localization.Localization(__file__, 106, 21), meshgrid_564381, *[r_564382, theta_564383], **kwargs_564384)
    
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___564386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), meshgrid_call_result_564385, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_564387 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___564386, int_564379)
    
    # Assigning a type to the variable 'tuple_var_assignment_564073' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_564073', subscript_call_result_564387)
    
    # Assigning a Subscript to a Name (line 106):
    
    # Obtaining the type of the subscript
    int_564388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
    
    # Call to meshgrid(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'r' (line 106)
    r_564391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'r', False)
    # Getting the type of 'theta' (line 106)
    theta_564392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'theta', False)
    # Processing the call keyword arguments (line 106)
    kwargs_564393 = {}
    # Getting the type of 'np' (line 106)
    np_564389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 106)
    meshgrid_564390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 21), np_564389, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 106)
    meshgrid_call_result_564394 = invoke(stypy.reporting.localization.Localization(__file__, 106, 21), meshgrid_564390, *[r_564391, theta_564392], **kwargs_564393)
    
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___564395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), meshgrid_call_result_564394, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_564396 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___564395, int_564388)
    
    # Assigning a type to the variable 'tuple_var_assignment_564074' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_564074', subscript_call_result_564396)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_var_assignment_564073' (line 106)
    tuple_var_assignment_564073_564397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_564073')
    # Assigning a type to the variable 'rg' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'rg', tuple_var_assignment_564073_564397)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_var_assignment_564074' (line 106)
    tuple_var_assignment_564074_564398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_564074')
    # Assigning a type to the variable 'thetag' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'thetag', tuple_var_assignment_564074_564398)
    
    # Assigning a Tuple to a Tuple (line 107):
    
    # Assigning a BinOp to a Name (line 107):
    # Getting the type of 'rg' (line 107)
    rg_564399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'rg')
    
    # Call to cos(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'thetag' (line 107)
    thetag_564402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'thetag', False)
    # Processing the call keyword arguments (line 107)
    kwargs_564403 = {}
    # Getting the type of 'np' (line 107)
    np_564400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'np', False)
    # Obtaining the member 'cos' of a type (line 107)
    cos_564401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 18), np_564400, 'cos')
    # Calling cos(args, kwargs) (line 107)
    cos_call_result_564404 = invoke(stypy.reporting.localization.Localization(__file__, 107, 18), cos_564401, *[thetag_564402], **kwargs_564403)
    
    # Applying the binary operator '*' (line 107)
    result_mul_564405 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), '*', rg_564399, cos_call_result_564404)
    
    # Assigning a type to the variable 'tuple_assignment_564075' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'tuple_assignment_564075', result_mul_564405)
    
    # Assigning a BinOp to a Name (line 107):
    # Getting the type of 'rg' (line 107)
    rg_564406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'rg')
    
    # Call to sin(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'thetag' (line 107)
    thetag_564409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 44), 'thetag', False)
    # Processing the call keyword arguments (line 107)
    kwargs_564410 = {}
    # Getting the type of 'np' (line 107)
    np_564407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 37), 'np', False)
    # Obtaining the member 'sin' of a type (line 107)
    sin_564408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 37), np_564407, 'sin')
    # Calling sin(args, kwargs) (line 107)
    sin_call_result_564411 = invoke(stypy.reporting.localization.Localization(__file__, 107, 37), sin_564408, *[thetag_564409], **kwargs_564410)
    
    # Applying the binary operator '*' (line 107)
    result_mul_564412 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 34), '*', rg_564406, sin_call_result_564411)
    
    # Assigning a type to the variable 'tuple_assignment_564076' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'tuple_assignment_564076', result_mul_564412)
    
    # Assigning a Name to a Name (line 107):
    # Getting the type of 'tuple_assignment_564075' (line 107)
    tuple_assignment_564075_564413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'tuple_assignment_564075')
    # Assigning a type to the variable 'a' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'a', tuple_assignment_564075_564413)
    
    # Assigning a Name to a Name (line 107):
    # Getting the type of 'tuple_assignment_564076' (line 107)
    tuple_assignment_564076_564414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'tuple_assignment_564076')
    # Assigning a type to the variable 'x' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'x', tuple_assignment_564076_564414)
    
    # Assigning a Tuple to a Tuple (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to flatten(...): (line 108)
    # Processing the call keyword arguments (line 108)
    kwargs_564417 = {}
    # Getting the type of 'a' (line 108)
    a_564415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'a', False)
    # Obtaining the member 'flatten' of a type (line 108)
    flatten_564416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), a_564415, 'flatten')
    # Calling flatten(args, kwargs) (line 108)
    flatten_call_result_564418 = invoke(stypy.reporting.localization.Localization(__file__, 108, 15), flatten_564416, *[], **kwargs_564417)
    
    # Assigning a type to the variable 'tuple_assignment_564077' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_assignment_564077', flatten_call_result_564418)
    
    # Assigning a Call to a Name (line 108):
    
    # Call to flatten(...): (line 108)
    # Processing the call keyword arguments (line 108)
    kwargs_564421 = {}
    # Getting the type of 'x' (line 108)
    x_564419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'x', False)
    # Obtaining the member 'flatten' of a type (line 108)
    flatten_564420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 28), x_564419, 'flatten')
    # Calling flatten(args, kwargs) (line 108)
    flatten_call_result_564422 = invoke(stypy.reporting.localization.Localization(__file__, 108, 28), flatten_564420, *[], **kwargs_564421)
    
    # Assigning a type to the variable 'tuple_assignment_564078' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_assignment_564078', flatten_call_result_564422)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_assignment_564077' (line 108)
    tuple_assignment_564077_564423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_assignment_564077')
    # Assigning a type to the variable 'a' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'a', tuple_assignment_564077_564423)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_assignment_564078' (line 108)
    tuple_assignment_564078_564424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_assignment_564078')
    # Assigning a type to the variable 'x' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'x', tuple_assignment_564078_564424)
    
    # Assigning a List to a Name (line 109):
    
    # Assigning a List to a Name (line 109):
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_564425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    
    # Assigning a type to the variable 'dataset' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'dataset', list_564425)
    
    
    # Call to enumerate(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Call to zip(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'a' (line 110)
    a_564428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 41), 'a', False)
    # Getting the type of 'x' (line 110)
    x_564429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 44), 'x', False)
    # Processing the call keyword arguments (line 110)
    kwargs_564430 = {}
    # Getting the type of 'zip' (line 110)
    zip_564427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 37), 'zip', False)
    # Calling zip(args, kwargs) (line 110)
    zip_call_result_564431 = invoke(stypy.reporting.localization.Localization(__file__, 110, 37), zip_564427, *[a_564428, x_564429], **kwargs_564430)
    
    # Processing the call keyword arguments (line 110)
    kwargs_564432 = {}
    # Getting the type of 'enumerate' (line 110)
    enumerate_564426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 110)
    enumerate_call_result_564433 = invoke(stypy.reporting.localization.Localization(__file__, 110, 27), enumerate_564426, *[zip_call_result_564431], **kwargs_564432)
    
    # Testing the type of a for loop iterable (line 110)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 8), enumerate_call_result_564433)
    # Getting the type of the for loop variable (line 110)
    for_loop_var_564434 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 8), enumerate_call_result_564433)
    # Assigning a type to the variable 'i' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 8), for_loop_var_564434))
    # Assigning a type to the variable 'a0' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'a0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 8), for_loop_var_564434))
    # Assigning a type to the variable 'x0' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'x0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 8), for_loop_var_564434))
    # SSA begins for a for statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'func' (line 111)
    func_564435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'func')
    # Getting the type of 'gammaincc' (line 111)
    gammaincc_564436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'gammaincc')
    # Applying the binary operator '==' (line 111)
    result_eq_564437 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), '==', func_564435, gammaincc_564436)
    
    # Testing the type of an if condition (line 111)
    if_condition_564438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 12), result_eq_564437)
    # Assigning a type to the variable 'if_condition_564438' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'if_condition_564438', if_condition_564438)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 115):
    
    # Assigning a Call to a Name (line 115):
    
    # Call to floor(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'a0' (line 115)
    a0_564441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 34), 'a0', False)
    # Processing the call keyword arguments (line 115)
    kwargs_564442 = {}
    # Getting the type of 'np' (line 115)
    np_564439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'np', False)
    # Obtaining the member 'floor' of a type (line 115)
    floor_564440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 25), np_564439, 'floor')
    # Calling floor(args, kwargs) (line 115)
    floor_call_result_564443 = invoke(stypy.reporting.localization.Localization(__file__, 115, 25), floor_564440, *[a0_564441], **kwargs_564442)
    
    # Assigning a type to the variable 'tuple_assignment_564079' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'tuple_assignment_564079', floor_call_result_564443)
    
    # Assigning a Call to a Name (line 115):
    
    # Call to floor(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'x0' (line 115)
    x0_564446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 48), 'x0', False)
    # Processing the call keyword arguments (line 115)
    kwargs_564447 = {}
    # Getting the type of 'np' (line 115)
    np_564444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 39), 'np', False)
    # Obtaining the member 'floor' of a type (line 115)
    floor_564445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 39), np_564444, 'floor')
    # Calling floor(args, kwargs) (line 115)
    floor_call_result_564448 = invoke(stypy.reporting.localization.Localization(__file__, 115, 39), floor_564445, *[x0_564446], **kwargs_564447)
    
    # Assigning a type to the variable 'tuple_assignment_564080' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'tuple_assignment_564080', floor_call_result_564448)
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'tuple_assignment_564079' (line 115)
    tuple_assignment_564079_564449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'tuple_assignment_564079')
    # Assigning a type to the variable 'a0' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'a0', tuple_assignment_564079_564449)
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'tuple_assignment_564080' (line 115)
    tuple_assignment_564080_564450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'tuple_assignment_564080')
    # Assigning a type to the variable 'x0' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'x0', tuple_assignment_564080_564450)
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 116)
    # Processing the call arguments (line 116)
    
    # Obtaining an instance of the builtin type 'tuple' (line 116)
    tuple_564453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 116)
    # Adding element type (line 116)
    # Getting the type of 'a0' (line 116)
    a0_564454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'a0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 28), tuple_564453, a0_564454)
    # Adding element type (line 116)
    # Getting the type of 'x0' (line 116)
    x0_564455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 32), 'x0', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 28), tuple_564453, x0_564455)
    # Adding element type (line 116)
    
    # Call to func(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'a0' (line 116)
    a0_564457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 41), 'a0', False)
    # Getting the type of 'x0' (line 116)
    x0_564458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 45), 'x0', False)
    # Processing the call keyword arguments (line 116)
    kwargs_564459 = {}
    # Getting the type of 'func' (line 116)
    func_564456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 36), 'func', False)
    # Calling func(args, kwargs) (line 116)
    func_call_result_564460 = invoke(stypy.reporting.localization.Localization(__file__, 116, 36), func_564456, *[a0_564457, x0_564458], **kwargs_564459)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 28), tuple_564453, func_call_result_564460)
    
    # Processing the call keyword arguments (line 116)
    kwargs_564461 = {}
    # Getting the type of 'dataset' (line 116)
    dataset_564451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'dataset', False)
    # Obtaining the member 'append' of a type (line 116)
    append_564452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), dataset_564451, 'append')
    # Calling append(args, kwargs) (line 116)
    append_call_result_564462 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), append_564452, *[tuple_564453], **kwargs_564461)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 117):
    
    # Assigning a Call to a Name (line 117):
    
    # Call to array(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'dataset' (line 117)
    dataset_564465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'dataset', False)
    # Processing the call keyword arguments (line 117)
    kwargs_564466 = {}
    # Getting the type of 'np' (line 117)
    np_564463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'np', False)
    # Obtaining the member 'array' of a type (line 117)
    array_564464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 18), np_564463, 'array')
    # Calling array(args, kwargs) (line 117)
    array_call_result_564467 = invoke(stypy.reporting.localization.Localization(__file__, 117, 18), array_564464, *[dataset_564465], **kwargs_564466)
    
    # Assigning a type to the variable 'dataset' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'dataset', array_call_result_564467)
    
    # Assigning a Call to a Name (line 118):
    
    # Assigning a Call to a Name (line 118):
    
    # Call to join(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'pwd' (line 118)
    pwd_564471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 32), 'pwd', False)
    str_564472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 37), 'str', '..')
    str_564473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 43), 'str', 'tests')
    str_564474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 52), 'str', 'data')
    str_564475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 60), 'str', 'local')
    
    # Call to format(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'func' (line 119)
    func_564478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 48), 'func', False)
    # Obtaining the member '__name__' of a type (line 119)
    name___564479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 48), func_564478, '__name__')
    # Processing the call keyword arguments (line 119)
    kwargs_564480 = {}
    str_564476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'str', '{}.txt')
    # Obtaining the member 'format' of a type (line 119)
    format_564477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 32), str_564476, 'format')
    # Calling format(args, kwargs) (line 119)
    format_call_result_564481 = invoke(stypy.reporting.localization.Localization(__file__, 119, 32), format_564477, *[name___564479], **kwargs_564480)
    
    # Processing the call keyword arguments (line 118)
    kwargs_564482 = {}
    # Getting the type of 'os' (line 118)
    os_564468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 118)
    path_564469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 19), os_564468, 'path')
    # Obtaining the member 'join' of a type (line 118)
    join_564470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 19), path_564469, 'join')
    # Calling join(args, kwargs) (line 118)
    join_call_result_564483 = invoke(stypy.reporting.localization.Localization(__file__, 118, 19), join_564470, *[pwd_564471, str_564472, str_564473, str_564474, str_564475, format_call_result_564481], **kwargs_564482)
    
    # Assigning a type to the variable 'filename' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'filename', join_call_result_564483)
    
    # Call to savetxt(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'filename' (line 120)
    filename_564486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'filename', False)
    # Getting the type of 'dataset' (line 120)
    dataset_564487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'dataset', False)
    # Processing the call keyword arguments (line 120)
    kwargs_564488 = {}
    # Getting the type of 'np' (line 120)
    np_564484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'np', False)
    # Obtaining the member 'savetxt' of a type (line 120)
    savetxt_564485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), np_564484, 'savetxt')
    # Calling savetxt(args, kwargs) (line 120)
    savetxt_call_result_564489 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), savetxt_564485, *[filename_564486, dataset_564487], **kwargs_564488)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Call to format(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Call to time(...): (line 122)
    # Processing the call keyword arguments (line 122)
    kwargs_564494 = {}
    # Getting the type of 'time' (line 122)
    time_564493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'time', False)
    # Calling time(args, kwargs) (line 122)
    time_call_result_564495 = invoke(stypy.reporting.localization.Localization(__file__, 122, 39), time_564493, *[], **kwargs_564494)
    
    # Getting the type of 't0' (line 122)
    t0_564496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 48), 't0', False)
    # Applying the binary operator '-' (line 122)
    result_sub_564497 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 39), '-', time_call_result_564495, t0_564496)
    
    int_564498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 52), 'int')
    # Applying the binary operator 'div' (line 122)
    result_div_564499 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 38), 'div', result_sub_564497, int_564498)
    
    # Processing the call keyword arguments (line 122)
    kwargs_564500 = {}
    str_564491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 10), 'str', '{} minutes elapsed')
    # Obtaining the member 'format' of a type (line 122)
    format_564492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 10), str_564491, 'format')
    # Calling format(args, kwargs) (line 122)
    format_call_result_564501 = invoke(stypy.reporting.localization.Localization(__file__, 122, 10), format_564492, *[result_div_564499], **kwargs_564500)
    
    # Processing the call keyword arguments (line 122)
    kwargs_564502 = {}
    # Getting the type of 'print' (line 122)
    print_564490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'print', False)
    # Calling print(args, kwargs) (line 122)
    print_call_result_564503 = invoke(stypy.reporting.localization.Localization(__file__, 122, 4), print_564490, *[format_call_result_564501], **kwargs_564502)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_564504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_564504)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_564504

# Assigning a type to the variable 'main' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'main', main)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 126)
    # Processing the call keyword arguments (line 126)
    kwargs_564506 = {}
    # Getting the type of 'main' (line 126)
    main_564505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'main', False)
    # Calling main(args, kwargs) (line 126)
    main_call_result_564507 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), main_564505, *[], **kwargs_564506)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
