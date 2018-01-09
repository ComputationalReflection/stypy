
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Newton-CG trust-region optimization.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import math
5: 
6: import numpy as np
7: import scipy.linalg
8: from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
9: 
10: __all__ = []
11: 
12: 
13: def _minimize_trust_ncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
14:                         **trust_region_options):
15:     '''
16:     Minimization of scalar function of one or more variables using
17:     the Newton conjugate gradient trust-region algorithm.
18: 
19:     Options
20:     -------
21:     initial_trust_radius : float
22:         Initial trust-region radius.
23:     max_trust_radius : float
24:         Maximum value of the trust-region radius. No steps that are longer
25:         than this value will be proposed.
26:     eta : float
27:         Trust region related acceptance stringency for proposed steps.
28:     gtol : float
29:         Gradient norm must be less than `gtol` before successful
30:         termination.
31: 
32:     '''
33:     if jac is None:
34:         raise ValueError('Jacobian is required for Newton-CG trust-region '
35:                          'minimization')
36:     if hess is None and hessp is None:
37:         raise ValueError('Either the Hessian or the Hessian-vector product '
38:                          'is required for Newton-CG trust-region minimization')
39:     return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
40:                                   hessp=hessp, subproblem=CGSteihaugSubproblem,
41:                                   **trust_region_options)
42: 
43: 
44: class CGSteihaugSubproblem(BaseQuadraticSubproblem):
45:     '''Quadratic subproblem solved by a conjugate gradient method'''
46:     def solve(self, trust_radius):
47:         '''
48:         Solve the subproblem using a conjugate gradient method.
49: 
50:         Parameters
51:         ----------
52:         trust_radius : float
53:             We are allowed to wander only this far away from the origin.
54: 
55:         Returns
56:         -------
57:         p : ndarray
58:             The proposed step.
59:         hits_boundary : bool
60:             True if the proposed step is on the boundary of the trust region.
61: 
62:         Notes
63:         -----
64:         This is algorithm (7.2) of Nocedal and Wright 2nd edition.
65:         Only the function that computes the Hessian-vector product is required.
66:         The Hessian itself is not required, and the Hessian does
67:         not need to be positive semidefinite.
68:         '''
69: 
70:         # get the norm of jacobian and define the origin
71:         p_origin = np.zeros_like(self.jac)
72: 
73:         # define a default tolerance
74:         tolerance = min(0.5, math.sqrt(self.jac_mag)) * self.jac_mag
75: 
76:         # Stop the method if the search direction
77:         # is a direction of nonpositive curvature.
78:         if self.jac_mag < tolerance:
79:             hits_boundary = False
80:             return p_origin, hits_boundary
81: 
82:         # init the state for the first iteration
83:         z = p_origin
84:         r = self.jac
85:         d = -r
86: 
87:         # Search for the min of the approximation of the objective function.
88:         while True:
89: 
90:             # do an iteration
91:             Bd = self.hessp(d)
92:             dBd = np.dot(d, Bd)
93:             if dBd <= 0:
94:                 # Look at the two boundary points.
95:                 # Find both values of t to get the boundary points such that
96:                 # ||z + t d|| == trust_radius
97:                 # and then choose the one with the predicted min value.
98:                 ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
99:                 pa = z + ta * d
100:                 pb = z + tb * d
101:                 if self(pa) < self(pb):
102:                     p_boundary = pa
103:                 else:
104:                     p_boundary = pb
105:                 hits_boundary = True
106:                 return p_boundary, hits_boundary
107:             r_squared = np.dot(r, r)
108:             alpha = r_squared / dBd
109:             z_next = z + alpha * d
110:             if scipy.linalg.norm(z_next) >= trust_radius:
111:                 # Find t >= 0 to get the boundary point such that
112:                 # ||z + t d|| == trust_radius
113:                 ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
114:                 p_boundary = z + tb * d
115:                 hits_boundary = True
116:                 return p_boundary, hits_boundary
117:             r_next = r + alpha * Bd
118:             r_next_squared = np.dot(r_next, r_next)
119:             if math.sqrt(r_next_squared) < tolerance:
120:                 hits_boundary = False
121:                 return z_next, hits_boundary
122:             beta_next = r_next_squared / r_squared
123:             d_next = -r_next + beta_next * d
124: 
125:             # update the state for the next iteration
126:             z = z_next
127:             r = r_next
128:             d = d_next
129: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_204392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Newton-CG trust-region optimization.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import math' statement (line 4)
import math

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204393 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_204393) is not StypyTypeError):

    if (import_204393 != 'pyd_module'):
        __import__(import_204393)
        sys_modules_204394 = sys.modules[import_204393]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_204394.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_204393)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import scipy.linalg' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204395 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg')

if (type(import_204395) is not StypyTypeError):

    if (import_204395 != 'pyd_module'):
        __import__(import_204395)
        sys_modules_204396 = sys.modules[import_204395]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', sys_modules_204396.module_type_store, module_type_store)
    else:
        import scipy.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', scipy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg', import_204395)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.optimize._trustregion import _minimize_trust_region, BaseQuadraticSubproblem' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_204397 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._trustregion')

if (type(import_204397) is not StypyTypeError):

    if (import_204397 != 'pyd_module'):
        __import__(import_204397)
        sys_modules_204398 = sys.modules[import_204397]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._trustregion', sys_modules_204398.module_type_store, module_type_store, ['_minimize_trust_region', 'BaseQuadraticSubproblem'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_204398, sys_modules_204398.module_type_store, module_type_store)
    else:
        from scipy.optimize._trustregion import _minimize_trust_region, BaseQuadraticSubproblem

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._trustregion', None, module_type_store, ['_minimize_trust_region', 'BaseQuadraticSubproblem'], [_minimize_trust_region, BaseQuadraticSubproblem])

else:
    # Assigning a type to the variable 'scipy.optimize._trustregion' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.optimize._trustregion', import_204397)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a List to a Name (line 10):

# Assigning a List to a Name (line 10):
__all__ = []
module_type_store.set_exportable_members([])

# Obtaining an instance of the builtin type 'list' (line 10)
list_204399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_204399)

@norecursion
def _minimize_trust_ncg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 13)
    tuple_204400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 13)
    
    # Getting the type of 'None' (line 13)
    None_204401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 46), 'None')
    # Getting the type of 'None' (line 13)
    None_204402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 57), 'None')
    # Getting the type of 'None' (line 13)
    None_204403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 69), 'None')
    defaults = [tuple_204400, None_204401, None_204402, None_204403]
    # Create a new context for function '_minimize_trust_ncg'
    module_type_store = module_type_store.open_function_context('_minimize_trust_ncg', 13, 0, False)
    
    # Passed parameters checking function
    _minimize_trust_ncg.stypy_localization = localization
    _minimize_trust_ncg.stypy_type_of_self = None
    _minimize_trust_ncg.stypy_type_store = module_type_store
    _minimize_trust_ncg.stypy_function_name = '_minimize_trust_ncg'
    _minimize_trust_ncg.stypy_param_names_list = ['fun', 'x0', 'args', 'jac', 'hess', 'hessp']
    _minimize_trust_ncg.stypy_varargs_param_name = None
    _minimize_trust_ncg.stypy_kwargs_param_name = 'trust_region_options'
    _minimize_trust_ncg.stypy_call_defaults = defaults
    _minimize_trust_ncg.stypy_call_varargs = varargs
    _minimize_trust_ncg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_minimize_trust_ncg', ['fun', 'x0', 'args', 'jac', 'hess', 'hessp'], None, 'trust_region_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_minimize_trust_ncg', localization, ['fun', 'x0', 'args', 'jac', 'hess', 'hessp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_minimize_trust_ncg(...)' code ##################

    str_204404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n    Minimization of scalar function of one or more variables using\n    the Newton conjugate gradient trust-region algorithm.\n\n    Options\n    -------\n    initial_trust_radius : float\n        Initial trust-region radius.\n    max_trust_radius : float\n        Maximum value of the trust-region radius. No steps that are longer\n        than this value will be proposed.\n    eta : float\n        Trust region related acceptance stringency for proposed steps.\n    gtol : float\n        Gradient norm must be less than `gtol` before successful\n        termination.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 33)
    # Getting the type of 'jac' (line 33)
    jac_204405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'jac')
    # Getting the type of 'None' (line 33)
    None_204406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'None')
    
    (may_be_204407, more_types_in_union_204408) = may_be_none(jac_204405, None_204406)

    if may_be_204407:

        if more_types_in_union_204408:
            # Runtime conditional SSA (line 33)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 34)
        # Processing the call arguments (line 34)
        str_204410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'str', 'Jacobian is required for Newton-CG trust-region minimization')
        # Processing the call keyword arguments (line 34)
        kwargs_204411 = {}
        # Getting the type of 'ValueError' (line 34)
        ValueError_204409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 34)
        ValueError_call_result_204412 = invoke(stypy.reporting.localization.Localization(__file__, 34, 14), ValueError_204409, *[str_204410], **kwargs_204411)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 34, 8), ValueError_call_result_204412, 'raise parameter', BaseException)

        if more_types_in_union_204408:
            # SSA join for if statement (line 33)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'hess' (line 36)
    hess_204413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 7), 'hess')
    # Getting the type of 'None' (line 36)
    None_204414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'None')
    # Applying the binary operator 'is' (line 36)
    result_is__204415 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 7), 'is', hess_204413, None_204414)
    
    
    # Getting the type of 'hessp' (line 36)
    hessp_204416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'hessp')
    # Getting the type of 'None' (line 36)
    None_204417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'None')
    # Applying the binary operator 'is' (line 36)
    result_is__204418 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 24), 'is', hessp_204416, None_204417)
    
    # Applying the binary operator 'and' (line 36)
    result_and_keyword_204419 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 7), 'and', result_is__204415, result_is__204418)
    
    # Testing the type of an if condition (line 36)
    if_condition_204420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 4), result_and_keyword_204419)
    # Assigning a type to the variable 'if_condition_204420' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'if_condition_204420', if_condition_204420)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 37)
    # Processing the call arguments (line 37)
    str_204422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'str', 'Either the Hessian or the Hessian-vector product is required for Newton-CG trust-region minimization')
    # Processing the call keyword arguments (line 37)
    kwargs_204423 = {}
    # Getting the type of 'ValueError' (line 37)
    ValueError_204421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 37)
    ValueError_call_result_204424 = invoke(stypy.reporting.localization.Localization(__file__, 37, 14), ValueError_204421, *[str_204422], **kwargs_204423)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 37, 8), ValueError_call_result_204424, 'raise parameter', BaseException)
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _minimize_trust_region(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'fun' (line 39)
    fun_204426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'fun', False)
    # Getting the type of 'x0' (line 39)
    x0_204427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 39), 'x0', False)
    # Processing the call keyword arguments (line 39)
    # Getting the type of 'args' (line 39)
    args_204428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 48), 'args', False)
    keyword_204429 = args_204428
    # Getting the type of 'jac' (line 39)
    jac_204430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 58), 'jac', False)
    keyword_204431 = jac_204430
    # Getting the type of 'hess' (line 39)
    hess_204432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 68), 'hess', False)
    keyword_204433 = hess_204432
    # Getting the type of 'hessp' (line 40)
    hessp_204434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 40), 'hessp', False)
    keyword_204435 = hessp_204434
    # Getting the type of 'CGSteihaugSubproblem' (line 40)
    CGSteihaugSubproblem_204436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 58), 'CGSteihaugSubproblem', False)
    keyword_204437 = CGSteihaugSubproblem_204436
    # Getting the type of 'trust_region_options' (line 41)
    trust_region_options_204438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 36), 'trust_region_options', False)
    kwargs_204439 = {'trust_region_options_204438': trust_region_options_204438, 'hessp': keyword_204435, 'args': keyword_204429, 'subproblem': keyword_204437, 'hess': keyword_204433, 'jac': keyword_204431}
    # Getting the type of '_minimize_trust_region' (line 39)
    _minimize_trust_region_204425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), '_minimize_trust_region', False)
    # Calling _minimize_trust_region(args, kwargs) (line 39)
    _minimize_trust_region_call_result_204440 = invoke(stypy.reporting.localization.Localization(__file__, 39, 11), _minimize_trust_region_204425, *[fun_204426, x0_204427], **kwargs_204439)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type', _minimize_trust_region_call_result_204440)
    
    # ################# End of '_minimize_trust_ncg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_minimize_trust_ncg' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_204441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204441)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_minimize_trust_ncg'
    return stypy_return_type_204441

# Assigning a type to the variable '_minimize_trust_ncg' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_minimize_trust_ncg', _minimize_trust_ncg)
# Declaration of the 'CGSteihaugSubproblem' class
# Getting the type of 'BaseQuadraticSubproblem' (line 44)
BaseQuadraticSubproblem_204442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'BaseQuadraticSubproblem')

class CGSteihaugSubproblem(BaseQuadraticSubproblem_204442, ):
    str_204443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'str', 'Quadratic subproblem solved by a conjugate gradient method')

    @norecursion
    def solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'solve'
        module_type_store = module_type_store.open_function_context('solve', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_localization', localization)
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_function_name', 'CGSteihaugSubproblem.solve')
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_param_names_list', ['trust_radius'])
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CGSteihaugSubproblem.solve.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CGSteihaugSubproblem.solve', ['trust_radius'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'solve', localization, ['trust_radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'solve(...)' code ##################

        str_204444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', '\n        Solve the subproblem using a conjugate gradient method.\n\n        Parameters\n        ----------\n        trust_radius : float\n            We are allowed to wander only this far away from the origin.\n\n        Returns\n        -------\n        p : ndarray\n            The proposed step.\n        hits_boundary : bool\n            True if the proposed step is on the boundary of the trust region.\n\n        Notes\n        -----\n        This is algorithm (7.2) of Nocedal and Wright 2nd edition.\n        Only the function that computes the Hessian-vector product is required.\n        The Hessian itself is not required, and the Hessian does\n        not need to be positive semidefinite.\n        ')
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to zeros_like(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'self' (line 71)
        self_204447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 33), 'self', False)
        # Obtaining the member 'jac' of a type (line 71)
        jac_204448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 33), self_204447, 'jac')
        # Processing the call keyword arguments (line 71)
        kwargs_204449 = {}
        # Getting the type of 'np' (line 71)
        np_204445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'np', False)
        # Obtaining the member 'zeros_like' of a type (line 71)
        zeros_like_204446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), np_204445, 'zeros_like')
        # Calling zeros_like(args, kwargs) (line 71)
        zeros_like_call_result_204450 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), zeros_like_204446, *[jac_204448], **kwargs_204449)
        
        # Assigning a type to the variable 'p_origin' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'p_origin', zeros_like_call_result_204450)
        
        # Assigning a BinOp to a Name (line 74):
        
        # Assigning a BinOp to a Name (line 74):
        
        # Call to min(...): (line 74)
        # Processing the call arguments (line 74)
        float_204452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'float')
        
        # Call to sqrt(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'self' (line 74)
        self_204455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 39), 'self', False)
        # Obtaining the member 'jac_mag' of a type (line 74)
        jac_mag_204456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 39), self_204455, 'jac_mag')
        # Processing the call keyword arguments (line 74)
        kwargs_204457 = {}
        # Getting the type of 'math' (line 74)
        math_204453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 29), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 74)
        sqrt_204454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 29), math_204453, 'sqrt')
        # Calling sqrt(args, kwargs) (line 74)
        sqrt_call_result_204458 = invoke(stypy.reporting.localization.Localization(__file__, 74, 29), sqrt_204454, *[jac_mag_204456], **kwargs_204457)
        
        # Processing the call keyword arguments (line 74)
        kwargs_204459 = {}
        # Getting the type of 'min' (line 74)
        min_204451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'min', False)
        # Calling min(args, kwargs) (line 74)
        min_call_result_204460 = invoke(stypy.reporting.localization.Localization(__file__, 74, 20), min_204451, *[float_204452, sqrt_call_result_204458], **kwargs_204459)
        
        # Getting the type of 'self' (line 74)
        self_204461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 56), 'self')
        # Obtaining the member 'jac_mag' of a type (line 74)
        jac_mag_204462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 56), self_204461, 'jac_mag')
        # Applying the binary operator '*' (line 74)
        result_mul_204463 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 20), '*', min_call_result_204460, jac_mag_204462)
        
        # Assigning a type to the variable 'tolerance' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'tolerance', result_mul_204463)
        
        
        # Getting the type of 'self' (line 78)
        self_204464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'self')
        # Obtaining the member 'jac_mag' of a type (line 78)
        jac_mag_204465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 11), self_204464, 'jac_mag')
        # Getting the type of 'tolerance' (line 78)
        tolerance_204466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'tolerance')
        # Applying the binary operator '<' (line 78)
        result_lt_204467 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), '<', jac_mag_204465, tolerance_204466)
        
        # Testing the type of an if condition (line 78)
        if_condition_204468 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_lt_204467)
        # Assigning a type to the variable 'if_condition_204468' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_204468', if_condition_204468)
        # SSA begins for if statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 79):
        
        # Assigning a Name to a Name (line 79):
        # Getting the type of 'False' (line 79)
        False_204469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 28), 'False')
        # Assigning a type to the variable 'hits_boundary' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'hits_boundary', False_204469)
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_204470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        # Getting the type of 'p_origin' (line 80)
        p_origin_204471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'p_origin')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 19), tuple_204470, p_origin_204471)
        # Adding element type (line 80)
        # Getting the type of 'hits_boundary' (line 80)
        hits_boundary_204472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'hits_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 19), tuple_204470, hits_boundary_204472)
        
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'stypy_return_type', tuple_204470)
        # SSA join for if statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 83):
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'p_origin' (line 83)
        p_origin_204473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'p_origin')
        # Assigning a type to the variable 'z' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'z', p_origin_204473)
        
        # Assigning a Attribute to a Name (line 84):
        
        # Assigning a Attribute to a Name (line 84):
        # Getting the type of 'self' (line 84)
        self_204474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'self')
        # Obtaining the member 'jac' of a type (line 84)
        jac_204475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), self_204474, 'jac')
        # Assigning a type to the variable 'r' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'r', jac_204475)
        
        # Assigning a UnaryOp to a Name (line 85):
        
        # Assigning a UnaryOp to a Name (line 85):
        
        # Getting the type of 'r' (line 85)
        r_204476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'r')
        # Applying the 'usub' unary operator (line 85)
        result___neg___204477 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 12), 'usub', r_204476)
        
        # Assigning a type to the variable 'd' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'd', result___neg___204477)
        
        # Getting the type of 'True' (line 88)
        True_204478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'True')
        # Testing the type of an if condition (line 88)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 8), True_204478)
        # SSA begins for while statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to hessp(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'd' (line 91)
        d_204481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'd', False)
        # Processing the call keyword arguments (line 91)
        kwargs_204482 = {}
        # Getting the type of 'self' (line 91)
        self_204479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 17), 'self', False)
        # Obtaining the member 'hessp' of a type (line 91)
        hessp_204480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 17), self_204479, 'hessp')
        # Calling hessp(args, kwargs) (line 91)
        hessp_call_result_204483 = invoke(stypy.reporting.localization.Localization(__file__, 91, 17), hessp_204480, *[d_204481], **kwargs_204482)
        
        # Assigning a type to the variable 'Bd' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'Bd', hessp_call_result_204483)
        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to dot(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'd' (line 92)
        d_204486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'd', False)
        # Getting the type of 'Bd' (line 92)
        Bd_204487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'Bd', False)
        # Processing the call keyword arguments (line 92)
        kwargs_204488 = {}
        # Getting the type of 'np' (line 92)
        np_204484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'np', False)
        # Obtaining the member 'dot' of a type (line 92)
        dot_204485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 18), np_204484, 'dot')
        # Calling dot(args, kwargs) (line 92)
        dot_call_result_204489 = invoke(stypy.reporting.localization.Localization(__file__, 92, 18), dot_204485, *[d_204486, Bd_204487], **kwargs_204488)
        
        # Assigning a type to the variable 'dBd' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'dBd', dot_call_result_204489)
        
        
        # Getting the type of 'dBd' (line 93)
        dBd_204490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'dBd')
        int_204491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'int')
        # Applying the binary operator '<=' (line 93)
        result_le_204492 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 15), '<=', dBd_204490, int_204491)
        
        # Testing the type of an if condition (line 93)
        if_condition_204493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 12), result_le_204492)
        # Assigning a type to the variable 'if_condition_204493' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'if_condition_204493', if_condition_204493)
        # SSA begins for if statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 98):
        
        # Assigning a Subscript to a Name (line 98):
        
        # Obtaining the type of the subscript
        int_204494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'int')
        
        # Call to get_boundaries_intersections(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'z' (line 98)
        z_204497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 59), 'z', False)
        # Getting the type of 'd' (line 98)
        d_204498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 62), 'd', False)
        # Getting the type of 'trust_radius' (line 98)
        trust_radius_204499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 65), 'trust_radius', False)
        # Processing the call keyword arguments (line 98)
        kwargs_204500 = {}
        # Getting the type of 'self' (line 98)
        self_204495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'self', False)
        # Obtaining the member 'get_boundaries_intersections' of a type (line 98)
        get_boundaries_intersections_204496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 25), self_204495, 'get_boundaries_intersections')
        # Calling get_boundaries_intersections(args, kwargs) (line 98)
        get_boundaries_intersections_call_result_204501 = invoke(stypy.reporting.localization.Localization(__file__, 98, 25), get_boundaries_intersections_204496, *[z_204497, d_204498, trust_radius_204499], **kwargs_204500)
        
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___204502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), get_boundaries_intersections_call_result_204501, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_204503 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), getitem___204502, int_204494)
        
        # Assigning a type to the variable 'tuple_var_assignment_204388' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'tuple_var_assignment_204388', subscript_call_result_204503)
        
        # Assigning a Subscript to a Name (line 98):
        
        # Obtaining the type of the subscript
        int_204504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'int')
        
        # Call to get_boundaries_intersections(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'z' (line 98)
        z_204507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 59), 'z', False)
        # Getting the type of 'd' (line 98)
        d_204508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 62), 'd', False)
        # Getting the type of 'trust_radius' (line 98)
        trust_radius_204509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 65), 'trust_radius', False)
        # Processing the call keyword arguments (line 98)
        kwargs_204510 = {}
        # Getting the type of 'self' (line 98)
        self_204505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'self', False)
        # Obtaining the member 'get_boundaries_intersections' of a type (line 98)
        get_boundaries_intersections_204506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 25), self_204505, 'get_boundaries_intersections')
        # Calling get_boundaries_intersections(args, kwargs) (line 98)
        get_boundaries_intersections_call_result_204511 = invoke(stypy.reporting.localization.Localization(__file__, 98, 25), get_boundaries_intersections_204506, *[z_204507, d_204508, trust_radius_204509], **kwargs_204510)
        
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___204512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), get_boundaries_intersections_call_result_204511, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_204513 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), getitem___204512, int_204504)
        
        # Assigning a type to the variable 'tuple_var_assignment_204389' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'tuple_var_assignment_204389', subscript_call_result_204513)
        
        # Assigning a Name to a Name (line 98):
        # Getting the type of 'tuple_var_assignment_204388' (line 98)
        tuple_var_assignment_204388_204514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'tuple_var_assignment_204388')
        # Assigning a type to the variable 'ta' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'ta', tuple_var_assignment_204388_204514)
        
        # Assigning a Name to a Name (line 98):
        # Getting the type of 'tuple_var_assignment_204389' (line 98)
        tuple_var_assignment_204389_204515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'tuple_var_assignment_204389')
        # Assigning a type to the variable 'tb' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'tb', tuple_var_assignment_204389_204515)
        
        # Assigning a BinOp to a Name (line 99):
        
        # Assigning a BinOp to a Name (line 99):
        # Getting the type of 'z' (line 99)
        z_204516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'z')
        # Getting the type of 'ta' (line 99)
        ta_204517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'ta')
        # Getting the type of 'd' (line 99)
        d_204518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 30), 'd')
        # Applying the binary operator '*' (line 99)
        result_mul_204519 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 25), '*', ta_204517, d_204518)
        
        # Applying the binary operator '+' (line 99)
        result_add_204520 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 21), '+', z_204516, result_mul_204519)
        
        # Assigning a type to the variable 'pa' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'pa', result_add_204520)
        
        # Assigning a BinOp to a Name (line 100):
        
        # Assigning a BinOp to a Name (line 100):
        # Getting the type of 'z' (line 100)
        z_204521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 21), 'z')
        # Getting the type of 'tb' (line 100)
        tb_204522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'tb')
        # Getting the type of 'd' (line 100)
        d_204523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'd')
        # Applying the binary operator '*' (line 100)
        result_mul_204524 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 25), '*', tb_204522, d_204523)
        
        # Applying the binary operator '+' (line 100)
        result_add_204525 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 21), '+', z_204521, result_mul_204524)
        
        # Assigning a type to the variable 'pb' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'pb', result_add_204525)
        
        
        
        # Call to self(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'pa' (line 101)
        pa_204527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'pa', False)
        # Processing the call keyword arguments (line 101)
        kwargs_204528 = {}
        # Getting the type of 'self' (line 101)
        self_204526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'self', False)
        # Calling self(args, kwargs) (line 101)
        self_call_result_204529 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), self_204526, *[pa_204527], **kwargs_204528)
        
        
        # Call to self(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'pb' (line 101)
        pb_204531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'pb', False)
        # Processing the call keyword arguments (line 101)
        kwargs_204532 = {}
        # Getting the type of 'self' (line 101)
        self_204530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'self', False)
        # Calling self(args, kwargs) (line 101)
        self_call_result_204533 = invoke(stypy.reporting.localization.Localization(__file__, 101, 30), self_204530, *[pb_204531], **kwargs_204532)
        
        # Applying the binary operator '<' (line 101)
        result_lt_204534 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 19), '<', self_call_result_204529, self_call_result_204533)
        
        # Testing the type of an if condition (line 101)
        if_condition_204535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 16), result_lt_204534)
        # Assigning a type to the variable 'if_condition_204535' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'if_condition_204535', if_condition_204535)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 102):
        
        # Assigning a Name to a Name (line 102):
        # Getting the type of 'pa' (line 102)
        pa_204536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 33), 'pa')
        # Assigning a type to the variable 'p_boundary' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'p_boundary', pa_204536)
        # SSA branch for the else part of an if statement (line 101)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 104):
        
        # Assigning a Name to a Name (line 104):
        # Getting the type of 'pb' (line 104)
        pb_204537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'pb')
        # Assigning a type to the variable 'p_boundary' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'p_boundary', pb_204537)
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 105):
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'True' (line 105)
        True_204538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 32), 'True')
        # Assigning a type to the variable 'hits_boundary' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'hits_boundary', True_204538)
        
        # Obtaining an instance of the builtin type 'tuple' (line 106)
        tuple_204539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 106)
        # Adding element type (line 106)
        # Getting the type of 'p_boundary' (line 106)
        p_boundary_204540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'p_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 23), tuple_204539, p_boundary_204540)
        # Adding element type (line 106)
        # Getting the type of 'hits_boundary' (line 106)
        hits_boundary_204541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 35), 'hits_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 23), tuple_204539, hits_boundary_204541)
        
        # Assigning a type to the variable 'stypy_return_type' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'stypy_return_type', tuple_204539)
        # SSA join for if statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to dot(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'r' (line 107)
        r_204544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 31), 'r', False)
        # Getting the type of 'r' (line 107)
        r_204545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'r', False)
        # Processing the call keyword arguments (line 107)
        kwargs_204546 = {}
        # Getting the type of 'np' (line 107)
        np_204542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'np', False)
        # Obtaining the member 'dot' of a type (line 107)
        dot_204543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 24), np_204542, 'dot')
        # Calling dot(args, kwargs) (line 107)
        dot_call_result_204547 = invoke(stypy.reporting.localization.Localization(__file__, 107, 24), dot_204543, *[r_204544, r_204545], **kwargs_204546)
        
        # Assigning a type to the variable 'r_squared' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'r_squared', dot_call_result_204547)
        
        # Assigning a BinOp to a Name (line 108):
        
        # Assigning a BinOp to a Name (line 108):
        # Getting the type of 'r_squared' (line 108)
        r_squared_204548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'r_squared')
        # Getting the type of 'dBd' (line 108)
        dBd_204549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'dBd')
        # Applying the binary operator 'div' (line 108)
        result_div_204550 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 20), 'div', r_squared_204548, dBd_204549)
        
        # Assigning a type to the variable 'alpha' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'alpha', result_div_204550)
        
        # Assigning a BinOp to a Name (line 109):
        
        # Assigning a BinOp to a Name (line 109):
        # Getting the type of 'z' (line 109)
        z_204551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'z')
        # Getting the type of 'alpha' (line 109)
        alpha_204552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'alpha')
        # Getting the type of 'd' (line 109)
        d_204553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 33), 'd')
        # Applying the binary operator '*' (line 109)
        result_mul_204554 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 25), '*', alpha_204552, d_204553)
        
        # Applying the binary operator '+' (line 109)
        result_add_204555 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 21), '+', z_204551, result_mul_204554)
        
        # Assigning a type to the variable 'z_next' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'z_next', result_add_204555)
        
        
        
        # Call to norm(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'z_next' (line 110)
        z_next_204559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 33), 'z_next', False)
        # Processing the call keyword arguments (line 110)
        kwargs_204560 = {}
        # Getting the type of 'scipy' (line 110)
        scipy_204556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 110)
        linalg_204557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), scipy_204556, 'linalg')
        # Obtaining the member 'norm' of a type (line 110)
        norm_204558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), linalg_204557, 'norm')
        # Calling norm(args, kwargs) (line 110)
        norm_call_result_204561 = invoke(stypy.reporting.localization.Localization(__file__, 110, 15), norm_204558, *[z_next_204559], **kwargs_204560)
        
        # Getting the type of 'trust_radius' (line 110)
        trust_radius_204562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 44), 'trust_radius')
        # Applying the binary operator '>=' (line 110)
        result_ge_204563 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 15), '>=', norm_call_result_204561, trust_radius_204562)
        
        # Testing the type of an if condition (line 110)
        if_condition_204564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 12), result_ge_204563)
        # Assigning a type to the variable 'if_condition_204564' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'if_condition_204564', if_condition_204564)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 113):
        
        # Assigning a Subscript to a Name (line 113):
        
        # Obtaining the type of the subscript
        int_204565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 16), 'int')
        
        # Call to get_boundaries_intersections(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'z' (line 113)
        z_204568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 59), 'z', False)
        # Getting the type of 'd' (line 113)
        d_204569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 62), 'd', False)
        # Getting the type of 'trust_radius' (line 113)
        trust_radius_204570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 65), 'trust_radius', False)
        # Processing the call keyword arguments (line 113)
        kwargs_204571 = {}
        # Getting the type of 'self' (line 113)
        self_204566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'self', False)
        # Obtaining the member 'get_boundaries_intersections' of a type (line 113)
        get_boundaries_intersections_204567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), self_204566, 'get_boundaries_intersections')
        # Calling get_boundaries_intersections(args, kwargs) (line 113)
        get_boundaries_intersections_call_result_204572 = invoke(stypy.reporting.localization.Localization(__file__, 113, 25), get_boundaries_intersections_204567, *[z_204568, d_204569, trust_radius_204570], **kwargs_204571)
        
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___204573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), get_boundaries_intersections_call_result_204572, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_204574 = invoke(stypy.reporting.localization.Localization(__file__, 113, 16), getitem___204573, int_204565)
        
        # Assigning a type to the variable 'tuple_var_assignment_204390' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'tuple_var_assignment_204390', subscript_call_result_204574)
        
        # Assigning a Subscript to a Name (line 113):
        
        # Obtaining the type of the subscript
        int_204575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 16), 'int')
        
        # Call to get_boundaries_intersections(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'z' (line 113)
        z_204578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 59), 'z', False)
        # Getting the type of 'd' (line 113)
        d_204579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 62), 'd', False)
        # Getting the type of 'trust_radius' (line 113)
        trust_radius_204580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 65), 'trust_radius', False)
        # Processing the call keyword arguments (line 113)
        kwargs_204581 = {}
        # Getting the type of 'self' (line 113)
        self_204576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'self', False)
        # Obtaining the member 'get_boundaries_intersections' of a type (line 113)
        get_boundaries_intersections_204577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), self_204576, 'get_boundaries_intersections')
        # Calling get_boundaries_intersections(args, kwargs) (line 113)
        get_boundaries_intersections_call_result_204582 = invoke(stypy.reporting.localization.Localization(__file__, 113, 25), get_boundaries_intersections_204577, *[z_204578, d_204579, trust_radius_204580], **kwargs_204581)
        
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___204583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), get_boundaries_intersections_call_result_204582, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_204584 = invoke(stypy.reporting.localization.Localization(__file__, 113, 16), getitem___204583, int_204575)
        
        # Assigning a type to the variable 'tuple_var_assignment_204391' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'tuple_var_assignment_204391', subscript_call_result_204584)
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'tuple_var_assignment_204390' (line 113)
        tuple_var_assignment_204390_204585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'tuple_var_assignment_204390')
        # Assigning a type to the variable 'ta' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'ta', tuple_var_assignment_204390_204585)
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'tuple_var_assignment_204391' (line 113)
        tuple_var_assignment_204391_204586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'tuple_var_assignment_204391')
        # Assigning a type to the variable 'tb' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'tb', tuple_var_assignment_204391_204586)
        
        # Assigning a BinOp to a Name (line 114):
        
        # Assigning a BinOp to a Name (line 114):
        # Getting the type of 'z' (line 114)
        z_204587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'z')
        # Getting the type of 'tb' (line 114)
        tb_204588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'tb')
        # Getting the type of 'd' (line 114)
        d_204589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'd')
        # Applying the binary operator '*' (line 114)
        result_mul_204590 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 33), '*', tb_204588, d_204589)
        
        # Applying the binary operator '+' (line 114)
        result_add_204591 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 29), '+', z_204587, result_mul_204590)
        
        # Assigning a type to the variable 'p_boundary' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'p_boundary', result_add_204591)
        
        # Assigning a Name to a Name (line 115):
        
        # Assigning a Name to a Name (line 115):
        # Getting the type of 'True' (line 115)
        True_204592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'True')
        # Assigning a type to the variable 'hits_boundary' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'hits_boundary', True_204592)
        
        # Obtaining an instance of the builtin type 'tuple' (line 116)
        tuple_204593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 116)
        # Adding element type (line 116)
        # Getting the type of 'p_boundary' (line 116)
        p_boundary_204594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'p_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 23), tuple_204593, p_boundary_204594)
        # Adding element type (line 116)
        # Getting the type of 'hits_boundary' (line 116)
        hits_boundary_204595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 35), 'hits_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 23), tuple_204593, hits_boundary_204595)
        
        # Assigning a type to the variable 'stypy_return_type' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'stypy_return_type', tuple_204593)
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 117):
        
        # Assigning a BinOp to a Name (line 117):
        # Getting the type of 'r' (line 117)
        r_204596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'r')
        # Getting the type of 'alpha' (line 117)
        alpha_204597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'alpha')
        # Getting the type of 'Bd' (line 117)
        Bd_204598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'Bd')
        # Applying the binary operator '*' (line 117)
        result_mul_204599 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 25), '*', alpha_204597, Bd_204598)
        
        # Applying the binary operator '+' (line 117)
        result_add_204600 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 21), '+', r_204596, result_mul_204599)
        
        # Assigning a type to the variable 'r_next' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'r_next', result_add_204600)
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to dot(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'r_next' (line 118)
        r_next_204603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 36), 'r_next', False)
        # Getting the type of 'r_next' (line 118)
        r_next_204604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 44), 'r_next', False)
        # Processing the call keyword arguments (line 118)
        kwargs_204605 = {}
        # Getting the type of 'np' (line 118)
        np_204601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 29), 'np', False)
        # Obtaining the member 'dot' of a type (line 118)
        dot_204602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 29), np_204601, 'dot')
        # Calling dot(args, kwargs) (line 118)
        dot_call_result_204606 = invoke(stypy.reporting.localization.Localization(__file__, 118, 29), dot_204602, *[r_next_204603, r_next_204604], **kwargs_204605)
        
        # Assigning a type to the variable 'r_next_squared' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'r_next_squared', dot_call_result_204606)
        
        
        
        # Call to sqrt(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'r_next_squared' (line 119)
        r_next_squared_204609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'r_next_squared', False)
        # Processing the call keyword arguments (line 119)
        kwargs_204610 = {}
        # Getting the type of 'math' (line 119)
        math_204607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 119)
        sqrt_204608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), math_204607, 'sqrt')
        # Calling sqrt(args, kwargs) (line 119)
        sqrt_call_result_204611 = invoke(stypy.reporting.localization.Localization(__file__, 119, 15), sqrt_204608, *[r_next_squared_204609], **kwargs_204610)
        
        # Getting the type of 'tolerance' (line 119)
        tolerance_204612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 43), 'tolerance')
        # Applying the binary operator '<' (line 119)
        result_lt_204613 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 15), '<', sqrt_call_result_204611, tolerance_204612)
        
        # Testing the type of an if condition (line 119)
        if_condition_204614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 12), result_lt_204613)
        # Assigning a type to the variable 'if_condition_204614' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'if_condition_204614', if_condition_204614)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 120):
        
        # Assigning a Name to a Name (line 120):
        # Getting the type of 'False' (line 120)
        False_204615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 32), 'False')
        # Assigning a type to the variable 'hits_boundary' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'hits_boundary', False_204615)
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_204616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        # Getting the type of 'z_next' (line 121)
        z_next_204617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'z_next')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 23), tuple_204616, z_next_204617)
        # Adding element type (line 121)
        # Getting the type of 'hits_boundary' (line 121)
        hits_boundary_204618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 31), 'hits_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 23), tuple_204616, hits_boundary_204618)
        
        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'stypy_return_type', tuple_204616)
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 122):
        
        # Assigning a BinOp to a Name (line 122):
        # Getting the type of 'r_next_squared' (line 122)
        r_next_squared_204619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'r_next_squared')
        # Getting the type of 'r_squared' (line 122)
        r_squared_204620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 41), 'r_squared')
        # Applying the binary operator 'div' (line 122)
        result_div_204621 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 24), 'div', r_next_squared_204619, r_squared_204620)
        
        # Assigning a type to the variable 'beta_next' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'beta_next', result_div_204621)
        
        # Assigning a BinOp to a Name (line 123):
        
        # Assigning a BinOp to a Name (line 123):
        
        # Getting the type of 'r_next' (line 123)
        r_next_204622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'r_next')
        # Applying the 'usub' unary operator (line 123)
        result___neg___204623 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 21), 'usub', r_next_204622)
        
        # Getting the type of 'beta_next' (line 123)
        beta_next_204624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'beta_next')
        # Getting the type of 'd' (line 123)
        d_204625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'd')
        # Applying the binary operator '*' (line 123)
        result_mul_204626 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 31), '*', beta_next_204624, d_204625)
        
        # Applying the binary operator '+' (line 123)
        result_add_204627 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 21), '+', result___neg___204623, result_mul_204626)
        
        # Assigning a type to the variable 'd_next' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'd_next', result_add_204627)
        
        # Assigning a Name to a Name (line 126):
        
        # Assigning a Name to a Name (line 126):
        # Getting the type of 'z_next' (line 126)
        z_next_204628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'z_next')
        # Assigning a type to the variable 'z' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'z', z_next_204628)
        
        # Assigning a Name to a Name (line 127):
        
        # Assigning a Name to a Name (line 127):
        # Getting the type of 'r_next' (line 127)
        r_next_204629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'r_next')
        # Assigning a type to the variable 'r' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'r', r_next_204629)
        
        # Assigning a Name to a Name (line 128):
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'd_next' (line 128)
        d_next_204630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'd_next')
        # Assigning a type to the variable 'd' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'd', d_next_204630)
        # SSA join for while statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solve' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_204631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204631)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solve'
        return stypy_return_type_204631


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 44, 0, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CGSteihaugSubproblem.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CGSteihaugSubproblem' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'CGSteihaugSubproblem', CGSteihaugSubproblem)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
