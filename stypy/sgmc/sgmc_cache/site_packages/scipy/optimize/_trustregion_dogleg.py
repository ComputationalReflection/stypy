
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Dog-leg trust-region optimization.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import numpy as np
5: import scipy.linalg
6: from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
7: 
8: __all__ = []
9: 
10: 
11: def _minimize_dogleg(fun, x0, args=(), jac=None, hess=None,
12:                      **trust_region_options):
13:     '''
14:     Minimization of scalar function of one or more variables using
15:     the dog-leg trust-region algorithm.
16: 
17:     Options
18:     -------
19:     initial_trust_radius : float
20:         Initial trust-region radius.
21:     max_trust_radius : float
22:         Maximum value of the trust-region radius. No steps that are longer
23:         than this value will be proposed.
24:     eta : float
25:         Trust region related acceptance stringency for proposed steps.
26:     gtol : float
27:         Gradient norm must be less than `gtol` before successful
28:         termination.
29: 
30:     '''
31:     if jac is None:
32:         raise ValueError('Jacobian is required for dogleg minimization')
33:     if hess is None:
34:         raise ValueError('Hessian is required for dogleg minimization')
35:     return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
36:                                   subproblem=DoglegSubproblem,
37:                                   **trust_region_options)
38: 
39: 
40: class DoglegSubproblem(BaseQuadraticSubproblem):
41:     '''Quadratic subproblem solved by the dogleg method'''
42: 
43:     def cauchy_point(self):
44:         '''
45:         The Cauchy point is minimal along the direction of steepest descent.
46:         '''
47:         if self._cauchy_point is None:
48:             g = self.jac
49:             Bg = self.hessp(g)
50:             self._cauchy_point = -(np.dot(g, g) / np.dot(g, Bg)) * g
51:         return self._cauchy_point
52: 
53:     def newton_point(self):
54:         '''
55:         The Newton point is a global minimum of the approximate function.
56:         '''
57:         if self._newton_point is None:
58:             g = self.jac
59:             B = self.hess
60:             cho_info = scipy.linalg.cho_factor(B)
61:             self._newton_point = -scipy.linalg.cho_solve(cho_info, g)
62:         return self._newton_point
63: 
64:     def solve(self, trust_radius):
65:         '''
66:         Minimize a function using the dog-leg trust-region algorithm.
67: 
68:         This algorithm requires function values and first and second derivatives.
69:         It also performs a costly Hessian decomposition for most iterations,
70:         and the Hessian is required to be positive definite.
71: 
72:         Parameters
73:         ----------
74:         trust_radius : float
75:             We are allowed to wander only this far away from the origin.
76: 
77:         Returns
78:         -------
79:         p : ndarray
80:             The proposed step.
81:         hits_boundary : bool
82:             True if the proposed step is on the boundary of the trust region.
83: 
84:         Notes
85:         -----
86:         The Hessian is required to be positive definite.
87: 
88:         References
89:         ----------
90:         .. [1] Jorge Nocedal and Stephen Wright,
91:                Numerical Optimization, second edition,
92:                Springer-Verlag, 2006, page 73.
93:         '''
94: 
95:         # Compute the Newton point.
96:         # This is the optimum for the quadratic model function.
97:         # If it is inside the trust radius then return this point.
98:         p_best = self.newton_point()
99:         if scipy.linalg.norm(p_best) < trust_radius:
100:             hits_boundary = False
101:             return p_best, hits_boundary
102: 
103:         # Compute the Cauchy point.
104:         # This is the predicted optimum along the direction of steepest descent.
105:         p_u = self.cauchy_point()
106: 
107:         # If the Cauchy point is outside the trust region,
108:         # then return the point where the path intersects the boundary.
109:         p_u_norm = scipy.linalg.norm(p_u)
110:         if p_u_norm >= trust_radius:
111:             p_boundary = p_u * (trust_radius / p_u_norm)
112:             hits_boundary = True
113:             return p_boundary, hits_boundary
114: 
115:         # Compute the intersection of the trust region boundary
116:         # and the line segment connecting the Cauchy and Newton points.
117:         # This requires solving a quadratic equation.
118:         # ||p_u + t*(p_best - p_u)||**2 == trust_radius**2
119:         # Solve this for positive time t using the quadratic formula.
120:         _, tb = self.get_boundaries_intersections(p_u, p_best - p_u,
121:                                                   trust_radius)
122:         p_boundary = p_u + tb * (p_best - p_u)
123:         hits_boundary = True
124:         return p_boundary, hits_boundary
125: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_203095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Dog-leg trust-region optimization.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_203096 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_203096) is not StypyTypeError):

    if (import_203096 != 'pyd_module'):
        __import__(import_203096)
        sys_modules_203097 = sys.modules[import_203096]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_203097.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_203096)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy.linalg' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_203098 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg')

if (type(import_203098) is not StypyTypeError):

    if (import_203098 != 'pyd_module'):
        __import__(import_203098)
        sys_modules_203099 = sys.modules[import_203098]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', sys_modules_203099.module_type_store, module_type_store)
    else:
        import scipy.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', scipy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', import_203098)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.optimize._trustregion import _minimize_trust_region, BaseQuadraticSubproblem' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_203100 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._trustregion')

if (type(import_203100) is not StypyTypeError):

    if (import_203100 != 'pyd_module'):
        __import__(import_203100)
        sys_modules_203101 = sys.modules[import_203100]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._trustregion', sys_modules_203101.module_type_store, module_type_store, ['_minimize_trust_region', 'BaseQuadraticSubproblem'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_203101, sys_modules_203101.module_type_store, module_type_store)
    else:
        from scipy.optimize._trustregion import _minimize_trust_region, BaseQuadraticSubproblem

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._trustregion', None, module_type_store, ['_minimize_trust_region', 'BaseQuadraticSubproblem'], [_minimize_trust_region, BaseQuadraticSubproblem])

else:
    # Assigning a type to the variable 'scipy.optimize._trustregion' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize._trustregion', import_203100)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a List to a Name (line 8):

# Assigning a List to a Name (line 8):
__all__ = []
module_type_store.set_exportable_members([])

# Obtaining an instance of the builtin type 'list' (line 8)
list_203102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_203102)

@norecursion
def _minimize_dogleg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 11)
    tuple_203103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 11)
    
    # Getting the type of 'None' (line 11)
    None_203104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 43), 'None')
    # Getting the type of 'None' (line 11)
    None_203105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 54), 'None')
    defaults = [tuple_203103, None_203104, None_203105]
    # Create a new context for function '_minimize_dogleg'
    module_type_store = module_type_store.open_function_context('_minimize_dogleg', 11, 0, False)
    
    # Passed parameters checking function
    _minimize_dogleg.stypy_localization = localization
    _minimize_dogleg.stypy_type_of_self = None
    _minimize_dogleg.stypy_type_store = module_type_store
    _minimize_dogleg.stypy_function_name = '_minimize_dogleg'
    _minimize_dogleg.stypy_param_names_list = ['fun', 'x0', 'args', 'jac', 'hess']
    _minimize_dogleg.stypy_varargs_param_name = None
    _minimize_dogleg.stypy_kwargs_param_name = 'trust_region_options'
    _minimize_dogleg.stypy_call_defaults = defaults
    _minimize_dogleg.stypy_call_varargs = varargs
    _minimize_dogleg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_minimize_dogleg', ['fun', 'x0', 'args', 'jac', 'hess'], None, 'trust_region_options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_minimize_dogleg', localization, ['fun', 'x0', 'args', 'jac', 'hess'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_minimize_dogleg(...)' code ##################

    str_203106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '\n    Minimization of scalar function of one or more variables using\n    the dog-leg trust-region algorithm.\n\n    Options\n    -------\n    initial_trust_radius : float\n        Initial trust-region radius.\n    max_trust_radius : float\n        Maximum value of the trust-region radius. No steps that are longer\n        than this value will be proposed.\n    eta : float\n        Trust region related acceptance stringency for proposed steps.\n    gtol : float\n        Gradient norm must be less than `gtol` before successful\n        termination.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 31)
    # Getting the type of 'jac' (line 31)
    jac_203107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'jac')
    # Getting the type of 'None' (line 31)
    None_203108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'None')
    
    (may_be_203109, more_types_in_union_203110) = may_be_none(jac_203107, None_203108)

    if may_be_203109:

        if more_types_in_union_203110:
            # Runtime conditional SSA (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 32)
        # Processing the call arguments (line 32)
        str_203112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'str', 'Jacobian is required for dogleg minimization')
        # Processing the call keyword arguments (line 32)
        kwargs_203113 = {}
        # Getting the type of 'ValueError' (line 32)
        ValueError_203111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 32)
        ValueError_call_result_203114 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), ValueError_203111, *[str_203112], **kwargs_203113)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 32, 8), ValueError_call_result_203114, 'raise parameter', BaseException)

        if more_types_in_union_203110:
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 33)
    # Getting the type of 'hess' (line 33)
    hess_203115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'hess')
    # Getting the type of 'None' (line 33)
    None_203116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'None')
    
    (may_be_203117, more_types_in_union_203118) = may_be_none(hess_203115, None_203116)

    if may_be_203117:

        if more_types_in_union_203118:
            # Runtime conditional SSA (line 33)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 34)
        # Processing the call arguments (line 34)
        str_203120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'str', 'Hessian is required for dogleg minimization')
        # Processing the call keyword arguments (line 34)
        kwargs_203121 = {}
        # Getting the type of 'ValueError' (line 34)
        ValueError_203119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 34)
        ValueError_call_result_203122 = invoke(stypy.reporting.localization.Localization(__file__, 34, 14), ValueError_203119, *[str_203120], **kwargs_203121)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 34, 8), ValueError_call_result_203122, 'raise parameter', BaseException)

        if more_types_in_union_203118:
            # SSA join for if statement (line 33)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to _minimize_trust_region(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'fun' (line 35)
    fun_203124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'fun', False)
    # Getting the type of 'x0' (line 35)
    x0_203125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'x0', False)
    # Processing the call keyword arguments (line 35)
    # Getting the type of 'args' (line 35)
    args_203126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 48), 'args', False)
    keyword_203127 = args_203126
    # Getting the type of 'jac' (line 35)
    jac_203128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 58), 'jac', False)
    keyword_203129 = jac_203128
    # Getting the type of 'hess' (line 35)
    hess_203130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 68), 'hess', False)
    keyword_203131 = hess_203130
    # Getting the type of 'DoglegSubproblem' (line 36)
    DoglegSubproblem_203132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'DoglegSubproblem', False)
    keyword_203133 = DoglegSubproblem_203132
    # Getting the type of 'trust_region_options' (line 37)
    trust_region_options_203134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 36), 'trust_region_options', False)
    kwargs_203135 = {'hess': keyword_203131, 'subproblem': keyword_203133, 'args': keyword_203127, 'jac': keyword_203129, 'trust_region_options_203134': trust_region_options_203134}
    # Getting the type of '_minimize_trust_region' (line 35)
    _minimize_trust_region_203123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), '_minimize_trust_region', False)
    # Calling _minimize_trust_region(args, kwargs) (line 35)
    _minimize_trust_region_call_result_203136 = invoke(stypy.reporting.localization.Localization(__file__, 35, 11), _minimize_trust_region_203123, *[fun_203124, x0_203125], **kwargs_203135)
    
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', _minimize_trust_region_call_result_203136)
    
    # ################# End of '_minimize_dogleg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_minimize_dogleg' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_203137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_203137)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_minimize_dogleg'
    return stypy_return_type_203137

# Assigning a type to the variable '_minimize_dogleg' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '_minimize_dogleg', _minimize_dogleg)
# Declaration of the 'DoglegSubproblem' class
# Getting the type of 'BaseQuadraticSubproblem' (line 40)
BaseQuadraticSubproblem_203138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'BaseQuadraticSubproblem')

class DoglegSubproblem(BaseQuadraticSubproblem_203138, ):
    str_203139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'str', 'Quadratic subproblem solved by the dogleg method')

    @norecursion
    def cauchy_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cauchy_point'
        module_type_store = module_type_store.open_function_context('cauchy_point', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_localization', localization)
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_function_name', 'DoglegSubproblem.cauchy_point')
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_param_names_list', [])
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DoglegSubproblem.cauchy_point.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DoglegSubproblem.cauchy_point', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cauchy_point', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cauchy_point(...)' code ##################

        str_203140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, (-1)), 'str', '\n        The Cauchy point is minimal along the direction of steepest descent.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 47)
        # Getting the type of 'self' (line 47)
        self_203141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'self')
        # Obtaining the member '_cauchy_point' of a type (line 47)
        _cauchy_point_203142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), self_203141, '_cauchy_point')
        # Getting the type of 'None' (line 47)
        None_203143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'None')
        
        (may_be_203144, more_types_in_union_203145) = may_be_none(_cauchy_point_203142, None_203143)

        if may_be_203144:

            if more_types_in_union_203145:
                # Runtime conditional SSA (line 47)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 48):
            
            # Assigning a Attribute to a Name (line 48):
            # Getting the type of 'self' (line 48)
            self_203146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'self')
            # Obtaining the member 'jac' of a type (line 48)
            jac_203147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), self_203146, 'jac')
            # Assigning a type to the variable 'g' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'g', jac_203147)
            
            # Assigning a Call to a Name (line 49):
            
            # Assigning a Call to a Name (line 49):
            
            # Call to hessp(...): (line 49)
            # Processing the call arguments (line 49)
            # Getting the type of 'g' (line 49)
            g_203150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 28), 'g', False)
            # Processing the call keyword arguments (line 49)
            kwargs_203151 = {}
            # Getting the type of 'self' (line 49)
            self_203148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'self', False)
            # Obtaining the member 'hessp' of a type (line 49)
            hessp_203149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 17), self_203148, 'hessp')
            # Calling hessp(args, kwargs) (line 49)
            hessp_call_result_203152 = invoke(stypy.reporting.localization.Localization(__file__, 49, 17), hessp_203149, *[g_203150], **kwargs_203151)
            
            # Assigning a type to the variable 'Bg' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'Bg', hessp_call_result_203152)
            
            # Assigning a BinOp to a Attribute (line 50):
            
            # Assigning a BinOp to a Attribute (line 50):
            
            
            # Call to dot(...): (line 50)
            # Processing the call arguments (line 50)
            # Getting the type of 'g' (line 50)
            g_203155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 42), 'g', False)
            # Getting the type of 'g' (line 50)
            g_203156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'g', False)
            # Processing the call keyword arguments (line 50)
            kwargs_203157 = {}
            # Getting the type of 'np' (line 50)
            np_203153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'np', False)
            # Obtaining the member 'dot' of a type (line 50)
            dot_203154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 35), np_203153, 'dot')
            # Calling dot(args, kwargs) (line 50)
            dot_call_result_203158 = invoke(stypy.reporting.localization.Localization(__file__, 50, 35), dot_203154, *[g_203155, g_203156], **kwargs_203157)
            
            
            # Call to dot(...): (line 50)
            # Processing the call arguments (line 50)
            # Getting the type of 'g' (line 50)
            g_203161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 57), 'g', False)
            # Getting the type of 'Bg' (line 50)
            Bg_203162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 60), 'Bg', False)
            # Processing the call keyword arguments (line 50)
            kwargs_203163 = {}
            # Getting the type of 'np' (line 50)
            np_203159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 50), 'np', False)
            # Obtaining the member 'dot' of a type (line 50)
            dot_203160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 50), np_203159, 'dot')
            # Calling dot(args, kwargs) (line 50)
            dot_call_result_203164 = invoke(stypy.reporting.localization.Localization(__file__, 50, 50), dot_203160, *[g_203161, Bg_203162], **kwargs_203163)
            
            # Applying the binary operator 'div' (line 50)
            result_div_203165 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 35), 'div', dot_call_result_203158, dot_call_result_203164)
            
            # Applying the 'usub' unary operator (line 50)
            result___neg___203166 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 33), 'usub', result_div_203165)
            
            # Getting the type of 'g' (line 50)
            g_203167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 67), 'g')
            # Applying the binary operator '*' (line 50)
            result_mul_203168 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 33), '*', result___neg___203166, g_203167)
            
            # Getting the type of 'self' (line 50)
            self_203169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self')
            # Setting the type of the member '_cauchy_point' of a type (line 50)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_203169, '_cauchy_point', result_mul_203168)

            if more_types_in_union_203145:
                # SSA join for if statement (line 47)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 51)
        self_203170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'self')
        # Obtaining the member '_cauchy_point' of a type (line 51)
        _cauchy_point_203171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), self_203170, '_cauchy_point')
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', _cauchy_point_203171)
        
        # ################# End of 'cauchy_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cauchy_point' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_203172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203172)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cauchy_point'
        return stypy_return_type_203172


    @norecursion
    def newton_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'newton_point'
        module_type_store = module_type_store.open_function_context('newton_point', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_localization', localization)
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_function_name', 'DoglegSubproblem.newton_point')
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_param_names_list', [])
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DoglegSubproblem.newton_point.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DoglegSubproblem.newton_point', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'newton_point', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'newton_point(...)' code ##################

        str_203173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n        The Newton point is a global minimum of the approximate function.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 57)
        # Getting the type of 'self' (line 57)
        self_203174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'self')
        # Obtaining the member '_newton_point' of a type (line 57)
        _newton_point_203175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), self_203174, '_newton_point')
        # Getting the type of 'None' (line 57)
        None_203176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 33), 'None')
        
        (may_be_203177, more_types_in_union_203178) = may_be_none(_newton_point_203175, None_203176)

        if may_be_203177:

            if more_types_in_union_203178:
                # Runtime conditional SSA (line 57)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 58):
            
            # Assigning a Attribute to a Name (line 58):
            # Getting the type of 'self' (line 58)
            self_203179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'self')
            # Obtaining the member 'jac' of a type (line 58)
            jac_203180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), self_203179, 'jac')
            # Assigning a type to the variable 'g' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'g', jac_203180)
            
            # Assigning a Attribute to a Name (line 59):
            
            # Assigning a Attribute to a Name (line 59):
            # Getting the type of 'self' (line 59)
            self_203181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'self')
            # Obtaining the member 'hess' of a type (line 59)
            hess_203182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), self_203181, 'hess')
            # Assigning a type to the variable 'B' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'B', hess_203182)
            
            # Assigning a Call to a Name (line 60):
            
            # Assigning a Call to a Name (line 60):
            
            # Call to cho_factor(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'B' (line 60)
            B_203186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 47), 'B', False)
            # Processing the call keyword arguments (line 60)
            kwargs_203187 = {}
            # Getting the type of 'scipy' (line 60)
            scipy_203183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'scipy', False)
            # Obtaining the member 'linalg' of a type (line 60)
            linalg_203184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), scipy_203183, 'linalg')
            # Obtaining the member 'cho_factor' of a type (line 60)
            cho_factor_203185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), linalg_203184, 'cho_factor')
            # Calling cho_factor(args, kwargs) (line 60)
            cho_factor_call_result_203188 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), cho_factor_203185, *[B_203186], **kwargs_203187)
            
            # Assigning a type to the variable 'cho_info' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'cho_info', cho_factor_call_result_203188)
            
            # Assigning a UnaryOp to a Attribute (line 61):
            
            # Assigning a UnaryOp to a Attribute (line 61):
            
            
            # Call to cho_solve(...): (line 61)
            # Processing the call arguments (line 61)
            # Getting the type of 'cho_info' (line 61)
            cho_info_203192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 57), 'cho_info', False)
            # Getting the type of 'g' (line 61)
            g_203193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 67), 'g', False)
            # Processing the call keyword arguments (line 61)
            kwargs_203194 = {}
            # Getting the type of 'scipy' (line 61)
            scipy_203189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'scipy', False)
            # Obtaining the member 'linalg' of a type (line 61)
            linalg_203190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 34), scipy_203189, 'linalg')
            # Obtaining the member 'cho_solve' of a type (line 61)
            cho_solve_203191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 34), linalg_203190, 'cho_solve')
            # Calling cho_solve(args, kwargs) (line 61)
            cho_solve_call_result_203195 = invoke(stypy.reporting.localization.Localization(__file__, 61, 34), cho_solve_203191, *[cho_info_203192, g_203193], **kwargs_203194)
            
            # Applying the 'usub' unary operator (line 61)
            result___neg___203196 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 33), 'usub', cho_solve_call_result_203195)
            
            # Getting the type of 'self' (line 61)
            self_203197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'self')
            # Setting the type of the member '_newton_point' of a type (line 61)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), self_203197, '_newton_point', result___neg___203196)

            if more_types_in_union_203178:
                # SSA join for if statement (line 57)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 62)
        self_203198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'self')
        # Obtaining the member '_newton_point' of a type (line 62)
        _newton_point_203199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 15), self_203198, '_newton_point')
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', _newton_point_203199)
        
        # ################# End of 'newton_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'newton_point' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_203200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'newton_point'
        return stypy_return_type_203200


    @norecursion
    def solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'solve'
        module_type_store = module_type_store.open_function_context('solve', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_localization', localization)
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_function_name', 'DoglegSubproblem.solve')
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_param_names_list', ['trust_radius'])
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DoglegSubproblem.solve.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DoglegSubproblem.solve', ['trust_radius'], None, None, defaults, varargs, kwargs)

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

        str_203201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'str', '\n        Minimize a function using the dog-leg trust-region algorithm.\n\n        This algorithm requires function values and first and second derivatives.\n        It also performs a costly Hessian decomposition for most iterations,\n        and the Hessian is required to be positive definite.\n\n        Parameters\n        ----------\n        trust_radius : float\n            We are allowed to wander only this far away from the origin.\n\n        Returns\n        -------\n        p : ndarray\n            The proposed step.\n        hits_boundary : bool\n            True if the proposed step is on the boundary of the trust region.\n\n        Notes\n        -----\n        The Hessian is required to be positive definite.\n\n        References\n        ----------\n        .. [1] Jorge Nocedal and Stephen Wright,\n               Numerical Optimization, second edition,\n               Springer-Verlag, 2006, page 73.\n        ')
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to newton_point(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_203204 = {}
        # Getting the type of 'self' (line 98)
        self_203202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'self', False)
        # Obtaining the member 'newton_point' of a type (line 98)
        newton_point_203203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 17), self_203202, 'newton_point')
        # Calling newton_point(args, kwargs) (line 98)
        newton_point_call_result_203205 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), newton_point_203203, *[], **kwargs_203204)
        
        # Assigning a type to the variable 'p_best' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'p_best', newton_point_call_result_203205)
        
        
        
        # Call to norm(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'p_best' (line 99)
        p_best_203209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'p_best', False)
        # Processing the call keyword arguments (line 99)
        kwargs_203210 = {}
        # Getting the type of 'scipy' (line 99)
        scipy_203206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 99)
        linalg_203207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 11), scipy_203206, 'linalg')
        # Obtaining the member 'norm' of a type (line 99)
        norm_203208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 11), linalg_203207, 'norm')
        # Calling norm(args, kwargs) (line 99)
        norm_call_result_203211 = invoke(stypy.reporting.localization.Localization(__file__, 99, 11), norm_203208, *[p_best_203209], **kwargs_203210)
        
        # Getting the type of 'trust_radius' (line 99)
        trust_radius_203212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 39), 'trust_radius')
        # Applying the binary operator '<' (line 99)
        result_lt_203213 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 11), '<', norm_call_result_203211, trust_radius_203212)
        
        # Testing the type of an if condition (line 99)
        if_condition_203214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 8), result_lt_203213)
        # Assigning a type to the variable 'if_condition_203214' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'if_condition_203214', if_condition_203214)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 100):
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'False' (line 100)
        False_203215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 28), 'False')
        # Assigning a type to the variable 'hits_boundary' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'hits_boundary', False_203215)
        
        # Obtaining an instance of the builtin type 'tuple' (line 101)
        tuple_203216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        # Getting the type of 'p_best' (line 101)
        p_best_203217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'p_best')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 19), tuple_203216, p_best_203217)
        # Adding element type (line 101)
        # Getting the type of 'hits_boundary' (line 101)
        hits_boundary_203218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'hits_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 19), tuple_203216, hits_boundary_203218)
        
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type', tuple_203216)
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to cauchy_point(...): (line 105)
        # Processing the call keyword arguments (line 105)
        kwargs_203221 = {}
        # Getting the type of 'self' (line 105)
        self_203219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 14), 'self', False)
        # Obtaining the member 'cauchy_point' of a type (line 105)
        cauchy_point_203220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 14), self_203219, 'cauchy_point')
        # Calling cauchy_point(args, kwargs) (line 105)
        cauchy_point_call_result_203222 = invoke(stypy.reporting.localization.Localization(__file__, 105, 14), cauchy_point_203220, *[], **kwargs_203221)
        
        # Assigning a type to the variable 'p_u' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'p_u', cauchy_point_call_result_203222)
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to norm(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'p_u' (line 109)
        p_u_203226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'p_u', False)
        # Processing the call keyword arguments (line 109)
        kwargs_203227 = {}
        # Getting the type of 'scipy' (line 109)
        scipy_203223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 109)
        linalg_203224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), scipy_203223, 'linalg')
        # Obtaining the member 'norm' of a type (line 109)
        norm_203225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), linalg_203224, 'norm')
        # Calling norm(args, kwargs) (line 109)
        norm_call_result_203228 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), norm_203225, *[p_u_203226], **kwargs_203227)
        
        # Assigning a type to the variable 'p_u_norm' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'p_u_norm', norm_call_result_203228)
        
        
        # Getting the type of 'p_u_norm' (line 110)
        p_u_norm_203229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'p_u_norm')
        # Getting the type of 'trust_radius' (line 110)
        trust_radius_203230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'trust_radius')
        # Applying the binary operator '>=' (line 110)
        result_ge_203231 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 11), '>=', p_u_norm_203229, trust_radius_203230)
        
        # Testing the type of an if condition (line 110)
        if_condition_203232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), result_ge_203231)
        # Assigning a type to the variable 'if_condition_203232' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_203232', if_condition_203232)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 111):
        
        # Assigning a BinOp to a Name (line 111):
        # Getting the type of 'p_u' (line 111)
        p_u_203233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'p_u')
        # Getting the type of 'trust_radius' (line 111)
        trust_radius_203234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 32), 'trust_radius')
        # Getting the type of 'p_u_norm' (line 111)
        p_u_norm_203235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 47), 'p_u_norm')
        # Applying the binary operator 'div' (line 111)
        result_div_203236 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 32), 'div', trust_radius_203234, p_u_norm_203235)
        
        # Applying the binary operator '*' (line 111)
        result_mul_203237 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 25), '*', p_u_203233, result_div_203236)
        
        # Assigning a type to the variable 'p_boundary' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'p_boundary', result_mul_203237)
        
        # Assigning a Name to a Name (line 112):
        
        # Assigning a Name to a Name (line 112):
        # Getting the type of 'True' (line 112)
        True_203238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'True')
        # Assigning a type to the variable 'hits_boundary' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'hits_boundary', True_203238)
        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_203239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        # Getting the type of 'p_boundary' (line 113)
        p_boundary_203240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'p_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), tuple_203239, p_boundary_203240)
        # Adding element type (line 113)
        # Getting the type of 'hits_boundary' (line 113)
        hits_boundary_203241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'hits_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 19), tuple_203239, hits_boundary_203241)
        
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'stypy_return_type', tuple_203239)
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 120):
        
        # Assigning a Subscript to a Name (line 120):
        
        # Obtaining the type of the subscript
        int_203242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'int')
        
        # Call to get_boundaries_intersections(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'p_u' (line 120)
        p_u_203245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 50), 'p_u', False)
        # Getting the type of 'p_best' (line 120)
        p_best_203246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 55), 'p_best', False)
        # Getting the type of 'p_u' (line 120)
        p_u_203247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 64), 'p_u', False)
        # Applying the binary operator '-' (line 120)
        result_sub_203248 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 55), '-', p_best_203246, p_u_203247)
        
        # Getting the type of 'trust_radius' (line 121)
        trust_radius_203249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 50), 'trust_radius', False)
        # Processing the call keyword arguments (line 120)
        kwargs_203250 = {}
        # Getting the type of 'self' (line 120)
        self_203243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'self', False)
        # Obtaining the member 'get_boundaries_intersections' of a type (line 120)
        get_boundaries_intersections_203244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), self_203243, 'get_boundaries_intersections')
        # Calling get_boundaries_intersections(args, kwargs) (line 120)
        get_boundaries_intersections_call_result_203251 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), get_boundaries_intersections_203244, *[p_u_203245, result_sub_203248, trust_radius_203249], **kwargs_203250)
        
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___203252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), get_boundaries_intersections_call_result_203251, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_203253 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), getitem___203252, int_203242)
        
        # Assigning a type to the variable 'tuple_var_assignment_203093' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_203093', subscript_call_result_203253)
        
        # Assigning a Subscript to a Name (line 120):
        
        # Obtaining the type of the subscript
        int_203254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'int')
        
        # Call to get_boundaries_intersections(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'p_u' (line 120)
        p_u_203257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 50), 'p_u', False)
        # Getting the type of 'p_best' (line 120)
        p_best_203258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 55), 'p_best', False)
        # Getting the type of 'p_u' (line 120)
        p_u_203259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 64), 'p_u', False)
        # Applying the binary operator '-' (line 120)
        result_sub_203260 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 55), '-', p_best_203258, p_u_203259)
        
        # Getting the type of 'trust_radius' (line 121)
        trust_radius_203261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 50), 'trust_radius', False)
        # Processing the call keyword arguments (line 120)
        kwargs_203262 = {}
        # Getting the type of 'self' (line 120)
        self_203255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'self', False)
        # Obtaining the member 'get_boundaries_intersections' of a type (line 120)
        get_boundaries_intersections_203256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), self_203255, 'get_boundaries_intersections')
        # Calling get_boundaries_intersections(args, kwargs) (line 120)
        get_boundaries_intersections_call_result_203263 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), get_boundaries_intersections_203256, *[p_u_203257, result_sub_203260, trust_radius_203261], **kwargs_203262)
        
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___203264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), get_boundaries_intersections_call_result_203263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_203265 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), getitem___203264, int_203254)
        
        # Assigning a type to the variable 'tuple_var_assignment_203094' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_203094', subscript_call_result_203265)
        
        # Assigning a Name to a Name (line 120):
        # Getting the type of 'tuple_var_assignment_203093' (line 120)
        tuple_var_assignment_203093_203266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_203093')
        # Assigning a type to the variable '_' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), '_', tuple_var_assignment_203093_203266)
        
        # Assigning a Name to a Name (line 120):
        # Getting the type of 'tuple_var_assignment_203094' (line 120)
        tuple_var_assignment_203094_203267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'tuple_var_assignment_203094')
        # Assigning a type to the variable 'tb' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'tb', tuple_var_assignment_203094_203267)
        
        # Assigning a BinOp to a Name (line 122):
        
        # Assigning a BinOp to a Name (line 122):
        # Getting the type of 'p_u' (line 122)
        p_u_203268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'p_u')
        # Getting the type of 'tb' (line 122)
        tb_203269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'tb')
        # Getting the type of 'p_best' (line 122)
        p_best_203270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 33), 'p_best')
        # Getting the type of 'p_u' (line 122)
        p_u_203271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 42), 'p_u')
        # Applying the binary operator '-' (line 122)
        result_sub_203272 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 33), '-', p_best_203270, p_u_203271)
        
        # Applying the binary operator '*' (line 122)
        result_mul_203273 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 27), '*', tb_203269, result_sub_203272)
        
        # Applying the binary operator '+' (line 122)
        result_add_203274 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 21), '+', p_u_203268, result_mul_203273)
        
        # Assigning a type to the variable 'p_boundary' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'p_boundary', result_add_203274)
        
        # Assigning a Name to a Name (line 123):
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'True' (line 123)
        True_203275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'True')
        # Assigning a type to the variable 'hits_boundary' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'hits_boundary', True_203275)
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_203276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        # Getting the type of 'p_boundary' (line 124)
        p_boundary_203277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'p_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 15), tuple_203276, p_boundary_203277)
        # Adding element type (line 124)
        # Getting the type of 'hits_boundary' (line 124)
        hits_boundary_203278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'hits_boundary')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 15), tuple_203276, hits_boundary_203278)
        
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', tuple_203276)
        
        # ################# End of 'solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solve' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_203279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203279)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solve'
        return stypy_return_type_203279


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 40, 0, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DoglegSubproblem.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DoglegSubproblem' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'DoglegSubproblem', DoglegSubproblem)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
