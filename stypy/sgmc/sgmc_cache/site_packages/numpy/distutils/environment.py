
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: from distutils.dist import Distribution
5: 
6: __metaclass__ = type
7: 
8: class EnvironmentConfig(object):
9:     def __init__(self, distutils_section='ALL', **kw):
10:         self._distutils_section = distutils_section
11:         self._conf_keys = kw
12:         self._conf = None
13:         self._hook_handler = None
14: 
15:     def dump_variable(self, name):
16:         conf_desc = self._conf_keys[name]
17:         hook, envvar, confvar, convert = conf_desc
18:         if not convert:
19:             convert = lambda x : x
20:         print('%s.%s:' % (self._distutils_section, name))
21:         v = self._hook_handler(name, hook)
22:         print('  hook   : %s' % (convert(v),))
23:         if envvar:
24:             v = os.environ.get(envvar, None)
25:             print('  environ: %s' % (convert(v),))
26:         if confvar and self._conf:
27:             v = self._conf.get(confvar, (None, None))[1]
28:             print('  config : %s' % (convert(v),))
29: 
30:     def dump_variables(self):
31:         for name in self._conf_keys:
32:             self.dump_variable(name)
33: 
34:     def __getattr__(self, name):
35:         try:
36:             conf_desc = self._conf_keys[name]
37:         except KeyError:
38:             raise AttributeError(name)
39:         return self._get_var(name, conf_desc)
40: 
41:     def get(self, name, default=None):
42:         try:
43:             conf_desc = self._conf_keys[name]
44:         except KeyError:
45:             return default
46:         var = self._get_var(name, conf_desc)
47:         if var is None:
48:             var = default
49:         return var
50: 
51:     def _get_var(self, name, conf_desc):
52:         hook, envvar, confvar, convert = conf_desc
53:         var = self._hook_handler(name, hook)
54:         if envvar is not None:
55:             var = os.environ.get(envvar, var)
56:         if confvar is not None and self._conf:
57:             var = self._conf.get(confvar, (None, var))[1]
58:         if convert is not None:
59:             var = convert(var)
60:         return var
61: 
62:     def clone(self, hook_handler):
63:         ec = self.__class__(distutils_section=self._distutils_section,
64:                             **self._conf_keys)
65:         ec._hook_handler = hook_handler
66:         return ec
67: 
68:     def use_distribution(self, dist):
69:         if isinstance(dist, Distribution):
70:             self._conf = dist.get_option_dict(self._distutils_section)
71:         else:
72:             self._conf = dist
73: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from distutils.dist import Distribution' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_32250 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.dist')

if (type(import_32250) is not StypyTypeError):

    if (import_32250 != 'pyd_module'):
        __import__(import_32250)
        sys_modules_32251 = sys.modules[import_32250]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.dist', sys_modules_32251.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_32251, sys_modules_32251.module_type_store, module_type_store)
    else:
        from distutils.dist import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.dist', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.dist' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.dist', import_32250)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')


# Assigning a Name to a Name (line 6):

# Assigning a Name to a Name (line 6):
# Getting the type of 'type' (line 6)
type_32252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 16), 'type')
# Assigning a type to the variable '__metaclass__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__metaclass__', type_32252)
# Declaration of the 'EnvironmentConfig' class

class EnvironmentConfig(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_32253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 41), 'str', 'ALL')
        defaults = [str_32253]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironmentConfig.__init__', ['distutils_section'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['distutils_section'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 10):
        
        # Assigning a Name to a Attribute (line 10):
        # Getting the type of 'distutils_section' (line 10)
        distutils_section_32254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 34), 'distutils_section')
        # Getting the type of 'self' (line 10)
        self_32255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self')
        # Setting the type of the member '_distutils_section' of a type (line 10)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), self_32255, '_distutils_section', distutils_section_32254)
        
        # Assigning a Name to a Attribute (line 11):
        
        # Assigning a Name to a Attribute (line 11):
        # Getting the type of 'kw' (line 11)
        kw_32256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 26), 'kw')
        # Getting the type of 'self' (line 11)
        self_32257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'self')
        # Setting the type of the member '_conf_keys' of a type (line 11)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), self_32257, '_conf_keys', kw_32256)
        
        # Assigning a Name to a Attribute (line 12):
        
        # Assigning a Name to a Attribute (line 12):
        # Getting the type of 'None' (line 12)
        None_32258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'None')
        # Getting the type of 'self' (line 12)
        self_32259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self')
        # Setting the type of the member '_conf' of a type (line 12)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), self_32259, '_conf', None_32258)
        
        # Assigning a Name to a Attribute (line 13):
        
        # Assigning a Name to a Attribute (line 13):
        # Getting the type of 'None' (line 13)
        None_32260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'None')
        # Getting the type of 'self' (line 13)
        self_32261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self')
        # Setting the type of the member '_hook_handler' of a type (line 13)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), self_32261, '_hook_handler', None_32260)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def dump_variable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dump_variable'
        module_type_store = module_type_store.open_function_context('dump_variable', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_localization', localization)
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_type_store', module_type_store)
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_function_name', 'EnvironmentConfig.dump_variable')
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_param_names_list', ['name'])
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_varargs_param_name', None)
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_call_defaults', defaults)
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_call_varargs', varargs)
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EnvironmentConfig.dump_variable.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironmentConfig.dump_variable', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dump_variable', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dump_variable(...)' code ##################

        
        # Assigning a Subscript to a Name (line 16):
        
        # Assigning a Subscript to a Name (line 16):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 16)
        name_32262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 36), 'name')
        # Getting the type of 'self' (line 16)
        self_32263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'self')
        # Obtaining the member '_conf_keys' of a type (line 16)
        _conf_keys_32264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 20), self_32263, '_conf_keys')
        # Obtaining the member '__getitem__' of a type (line 16)
        getitem___32265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 20), _conf_keys_32264, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 16)
        subscript_call_result_32266 = invoke(stypy.reporting.localization.Localization(__file__, 16, 20), getitem___32265, name_32262)
        
        # Assigning a type to the variable 'conf_desc' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'conf_desc', subscript_call_result_32266)
        
        # Assigning a Name to a Tuple (line 17):
        
        # Assigning a Subscript to a Name (line 17):
        
        # Obtaining the type of the subscript
        int_32267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'int')
        # Getting the type of 'conf_desc' (line 17)
        conf_desc_32268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 41), 'conf_desc')
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___32269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), conf_desc_32268, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_32270 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), getitem___32269, int_32267)
        
        # Assigning a type to the variable 'tuple_var_assignment_32242' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_32242', subscript_call_result_32270)
        
        # Assigning a Subscript to a Name (line 17):
        
        # Obtaining the type of the subscript
        int_32271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'int')
        # Getting the type of 'conf_desc' (line 17)
        conf_desc_32272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 41), 'conf_desc')
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___32273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), conf_desc_32272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_32274 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), getitem___32273, int_32271)
        
        # Assigning a type to the variable 'tuple_var_assignment_32243' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_32243', subscript_call_result_32274)
        
        # Assigning a Subscript to a Name (line 17):
        
        # Obtaining the type of the subscript
        int_32275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'int')
        # Getting the type of 'conf_desc' (line 17)
        conf_desc_32276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 41), 'conf_desc')
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___32277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), conf_desc_32276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_32278 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), getitem___32277, int_32275)
        
        # Assigning a type to the variable 'tuple_var_assignment_32244' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_32244', subscript_call_result_32278)
        
        # Assigning a Subscript to a Name (line 17):
        
        # Obtaining the type of the subscript
        int_32279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'int')
        # Getting the type of 'conf_desc' (line 17)
        conf_desc_32280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 41), 'conf_desc')
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___32281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), conf_desc_32280, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_32282 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), getitem___32281, int_32279)
        
        # Assigning a type to the variable 'tuple_var_assignment_32245' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_32245', subscript_call_result_32282)
        
        # Assigning a Name to a Name (line 17):
        # Getting the type of 'tuple_var_assignment_32242' (line 17)
        tuple_var_assignment_32242_32283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_32242')
        # Assigning a type to the variable 'hook' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'hook', tuple_var_assignment_32242_32283)
        
        # Assigning a Name to a Name (line 17):
        # Getting the type of 'tuple_var_assignment_32243' (line 17)
        tuple_var_assignment_32243_32284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_32243')
        # Assigning a type to the variable 'envvar' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'envvar', tuple_var_assignment_32243_32284)
        
        # Assigning a Name to a Name (line 17):
        # Getting the type of 'tuple_var_assignment_32244' (line 17)
        tuple_var_assignment_32244_32285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_32244')
        # Assigning a type to the variable 'confvar' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'confvar', tuple_var_assignment_32244_32285)
        
        # Assigning a Name to a Name (line 17):
        # Getting the type of 'tuple_var_assignment_32245' (line 17)
        tuple_var_assignment_32245_32286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple_var_assignment_32245')
        # Assigning a type to the variable 'convert' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 31), 'convert', tuple_var_assignment_32245_32286)
        
        
        # Getting the type of 'convert' (line 18)
        convert_32287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'convert')
        # Applying the 'not' unary operator (line 18)
        result_not__32288 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), 'not', convert_32287)
        
        # Testing the type of an if condition (line 18)
        if_condition_32289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 8), result_not__32288)
        # Assigning a type to the variable 'if_condition_32289' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'if_condition_32289', if_condition_32289)
        # SSA begins for if statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Lambda to a Name (line 19):
        
        # Assigning a Lambda to a Name (line 19):

        @norecursion
        def _stypy_temp_lambda_17(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_17'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_17', 19, 22, True)
            # Passed parameters checking function
            _stypy_temp_lambda_17.stypy_localization = localization
            _stypy_temp_lambda_17.stypy_type_of_self = None
            _stypy_temp_lambda_17.stypy_type_store = module_type_store
            _stypy_temp_lambda_17.stypy_function_name = '_stypy_temp_lambda_17'
            _stypy_temp_lambda_17.stypy_param_names_list = ['x']
            _stypy_temp_lambda_17.stypy_varargs_param_name = None
            _stypy_temp_lambda_17.stypy_kwargs_param_name = None
            _stypy_temp_lambda_17.stypy_call_defaults = defaults
            _stypy_temp_lambda_17.stypy_call_varargs = varargs
            _stypy_temp_lambda_17.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_17', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_17', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 19)
            x_32290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 33), 'x')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'stypy_return_type', x_32290)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_17' in the type store
            # Getting the type of 'stypy_return_type' (line 19)
            stypy_return_type_32291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_32291)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_17'
            return stypy_return_type_32291

        # Assigning a type to the variable '_stypy_temp_lambda_17' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), '_stypy_temp_lambda_17', _stypy_temp_lambda_17)
        # Getting the type of '_stypy_temp_lambda_17' (line 19)
        _stypy_temp_lambda_17_32292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), '_stypy_temp_lambda_17')
        # Assigning a type to the variable 'convert' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'convert', _stypy_temp_lambda_17_32292)
        # SSA join for if statement (line 18)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to print(...): (line 20)
        # Processing the call arguments (line 20)
        str_32294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 14), 'str', '%s.%s:')
        
        # Obtaining an instance of the builtin type 'tuple' (line 20)
        tuple_32295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 20)
        # Adding element type (line 20)
        # Getting the type of 'self' (line 20)
        self_32296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'self', False)
        # Obtaining the member '_distutils_section' of a type (line 20)
        _distutils_section_32297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), self_32296, '_distutils_section')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 26), tuple_32295, _distutils_section_32297)
        # Adding element type (line 20)
        # Getting the type of 'name' (line 20)
        name_32298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 51), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 26), tuple_32295, name_32298)
        
        # Applying the binary operator '%' (line 20)
        result_mod_32299 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 14), '%', str_32294, tuple_32295)
        
        # Processing the call keyword arguments (line 20)
        kwargs_32300 = {}
        # Getting the type of 'print' (line 20)
        print_32293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'print', False)
        # Calling print(args, kwargs) (line 20)
        print_call_result_32301 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), print_32293, *[result_mod_32299], **kwargs_32300)
        
        
        # Assigning a Call to a Name (line 21):
        
        # Assigning a Call to a Name (line 21):
        
        # Call to _hook_handler(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'name' (line 21)
        name_32304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), 'name', False)
        # Getting the type of 'hook' (line 21)
        hook_32305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 37), 'hook', False)
        # Processing the call keyword arguments (line 21)
        kwargs_32306 = {}
        # Getting the type of 'self' (line 21)
        self_32302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'self', False)
        # Obtaining the member '_hook_handler' of a type (line 21)
        _hook_handler_32303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), self_32302, '_hook_handler')
        # Calling _hook_handler(args, kwargs) (line 21)
        _hook_handler_call_result_32307 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), _hook_handler_32303, *[name_32304, hook_32305], **kwargs_32306)
        
        # Assigning a type to the variable 'v' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'v', _hook_handler_call_result_32307)
        
        # Call to print(...): (line 22)
        # Processing the call arguments (line 22)
        str_32309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'str', '  hook   : %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_32310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        
        # Call to convert(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'v' (line 22)
        v_32312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 41), 'v', False)
        # Processing the call keyword arguments (line 22)
        kwargs_32313 = {}
        # Getting the type of 'convert' (line 22)
        convert_32311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'convert', False)
        # Calling convert(args, kwargs) (line 22)
        convert_call_result_32314 = invoke(stypy.reporting.localization.Localization(__file__, 22, 33), convert_32311, *[v_32312], **kwargs_32313)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 33), tuple_32310, convert_call_result_32314)
        
        # Applying the binary operator '%' (line 22)
        result_mod_32315 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 14), '%', str_32309, tuple_32310)
        
        # Processing the call keyword arguments (line 22)
        kwargs_32316 = {}
        # Getting the type of 'print' (line 22)
        print_32308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'print', False)
        # Calling print(args, kwargs) (line 22)
        print_call_result_32317 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), print_32308, *[result_mod_32315], **kwargs_32316)
        
        
        # Getting the type of 'envvar' (line 23)
        envvar_32318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'envvar')
        # Testing the type of an if condition (line 23)
        if_condition_32319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 8), envvar_32318)
        # Assigning a type to the variable 'if_condition_32319' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'if_condition_32319', if_condition_32319)
        # SSA begins for if statement (line 23)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 24):
        
        # Assigning a Call to a Name (line 24):
        
        # Call to get(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'envvar' (line 24)
        envvar_32323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'envvar', False)
        # Getting the type of 'None' (line 24)
        None_32324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 39), 'None', False)
        # Processing the call keyword arguments (line 24)
        kwargs_32325 = {}
        # Getting the type of 'os' (line 24)
        os_32320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'os', False)
        # Obtaining the member 'environ' of a type (line 24)
        environ_32321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), os_32320, 'environ')
        # Obtaining the member 'get' of a type (line 24)
        get_32322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), environ_32321, 'get')
        # Calling get(args, kwargs) (line 24)
        get_call_result_32326 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), get_32322, *[envvar_32323, None_32324], **kwargs_32325)
        
        # Assigning a type to the variable 'v' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'v', get_call_result_32326)
        
        # Call to print(...): (line 25)
        # Processing the call arguments (line 25)
        str_32328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'str', '  environ: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 25)
        tuple_32329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 25)
        # Adding element type (line 25)
        
        # Call to convert(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'v' (line 25)
        v_32331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 45), 'v', False)
        # Processing the call keyword arguments (line 25)
        kwargs_32332 = {}
        # Getting the type of 'convert' (line 25)
        convert_32330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 37), 'convert', False)
        # Calling convert(args, kwargs) (line 25)
        convert_call_result_32333 = invoke(stypy.reporting.localization.Localization(__file__, 25, 37), convert_32330, *[v_32331], **kwargs_32332)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 37), tuple_32329, convert_call_result_32333)
        
        # Applying the binary operator '%' (line 25)
        result_mod_32334 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 18), '%', str_32328, tuple_32329)
        
        # Processing the call keyword arguments (line 25)
        kwargs_32335 = {}
        # Getting the type of 'print' (line 25)
        print_32327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'print', False)
        # Calling print(args, kwargs) (line 25)
        print_call_result_32336 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), print_32327, *[result_mod_32334], **kwargs_32335)
        
        # SSA join for if statement (line 23)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'confvar' (line 26)
        confvar_32337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'confvar')
        # Getting the type of 'self' (line 26)
        self_32338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'self')
        # Obtaining the member '_conf' of a type (line 26)
        _conf_32339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 23), self_32338, '_conf')
        # Applying the binary operator 'and' (line 26)
        result_and_keyword_32340 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 11), 'and', confvar_32337, _conf_32339)
        
        # Testing the type of an if condition (line 26)
        if_condition_32341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 8), result_and_keyword_32340)
        # Assigning a type to the variable 'if_condition_32341' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'if_condition_32341', if_condition_32341)
        # SSA begins for if statement (line 26)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 27):
        
        # Assigning a Subscript to a Name (line 27):
        
        # Obtaining the type of the subscript
        int_32342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 54), 'int')
        
        # Call to get(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'confvar' (line 27)
        confvar_32346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 31), 'confvar', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 27)
        tuple_32347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 27)
        # Adding element type (line 27)
        # Getting the type of 'None' (line 27)
        None_32348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 41), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 41), tuple_32347, None_32348)
        # Adding element type (line 27)
        # Getting the type of 'None' (line 27)
        None_32349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 47), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 41), tuple_32347, None_32349)
        
        # Processing the call keyword arguments (line 27)
        kwargs_32350 = {}
        # Getting the type of 'self' (line 27)
        self_32343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'self', False)
        # Obtaining the member '_conf' of a type (line 27)
        _conf_32344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 16), self_32343, '_conf')
        # Obtaining the member 'get' of a type (line 27)
        get_32345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 16), _conf_32344, 'get')
        # Calling get(args, kwargs) (line 27)
        get_call_result_32351 = invoke(stypy.reporting.localization.Localization(__file__, 27, 16), get_32345, *[confvar_32346, tuple_32347], **kwargs_32350)
        
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___32352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 16), get_call_result_32351, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_32353 = invoke(stypy.reporting.localization.Localization(__file__, 27, 16), getitem___32352, int_32342)
        
        # Assigning a type to the variable 'v' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'v', subscript_call_result_32353)
        
        # Call to print(...): (line 28)
        # Processing the call arguments (line 28)
        str_32355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'str', '  config : %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_32356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        
        # Call to convert(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'v' (line 28)
        v_32358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 45), 'v', False)
        # Processing the call keyword arguments (line 28)
        kwargs_32359 = {}
        # Getting the type of 'convert' (line 28)
        convert_32357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 37), 'convert', False)
        # Calling convert(args, kwargs) (line 28)
        convert_call_result_32360 = invoke(stypy.reporting.localization.Localization(__file__, 28, 37), convert_32357, *[v_32358], **kwargs_32359)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 37), tuple_32356, convert_call_result_32360)
        
        # Applying the binary operator '%' (line 28)
        result_mod_32361 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 18), '%', str_32355, tuple_32356)
        
        # Processing the call keyword arguments (line 28)
        kwargs_32362 = {}
        # Getting the type of 'print' (line 28)
        print_32354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'print', False)
        # Calling print(args, kwargs) (line 28)
        print_call_result_32363 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), print_32354, *[result_mod_32361], **kwargs_32362)
        
        # SSA join for if statement (line 26)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'dump_variable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dump_variable' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_32364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dump_variable'
        return stypy_return_type_32364


    @norecursion
    def dump_variables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dump_variables'
        module_type_store = module_type_store.open_function_context('dump_variables', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_localization', localization)
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_type_store', module_type_store)
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_function_name', 'EnvironmentConfig.dump_variables')
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_param_names_list', [])
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_varargs_param_name', None)
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_call_defaults', defaults)
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_call_varargs', varargs)
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EnvironmentConfig.dump_variables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironmentConfig.dump_variables', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dump_variables', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dump_variables(...)' code ##################

        
        # Getting the type of 'self' (line 31)
        self_32365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'self')
        # Obtaining the member '_conf_keys' of a type (line 31)
        _conf_keys_32366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), self_32365, '_conf_keys')
        # Testing the type of a for loop iterable (line 31)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 8), _conf_keys_32366)
        # Getting the type of the for loop variable (line 31)
        for_loop_var_32367 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 8), _conf_keys_32366)
        # Assigning a type to the variable 'name' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'name', for_loop_var_32367)
        # SSA begins for a for statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to dump_variable(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'name' (line 32)
        name_32370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 31), 'name', False)
        # Processing the call keyword arguments (line 32)
        kwargs_32371 = {}
        # Getting the type of 'self' (line 32)
        self_32368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'self', False)
        # Obtaining the member 'dump_variable' of a type (line 32)
        dump_variable_32369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), self_32368, 'dump_variable')
        # Calling dump_variable(args, kwargs) (line 32)
        dump_variable_call_result_32372 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), dump_variable_32369, *[name_32370], **kwargs_32371)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'dump_variables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dump_variables' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_32373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32373)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dump_variables'
        return stypy_return_type_32373


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_function_name', 'EnvironmentConfig.__getattr__')
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['name'])
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EnvironmentConfig.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironmentConfig.__getattr__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 36):
        
        # Assigning a Subscript to a Name (line 36):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 36)
        name_32374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 40), 'name')
        # Getting the type of 'self' (line 36)
        self_32375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'self')
        # Obtaining the member '_conf_keys' of a type (line 36)
        _conf_keys_32376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 24), self_32375, '_conf_keys')
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___32377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 24), _conf_keys_32376, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_32378 = invoke(stypy.reporting.localization.Localization(__file__, 36, 24), getitem___32377, name_32374)
        
        # Assigning a type to the variable 'conf_desc' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'conf_desc', subscript_call_result_32378)
        # SSA branch for the except part of a try statement (line 35)
        # SSA branch for the except 'KeyError' branch of a try statement (line 35)
        module_type_store.open_ssa_branch('except')
        
        # Call to AttributeError(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'name' (line 38)
        name_32380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'name', False)
        # Processing the call keyword arguments (line 38)
        kwargs_32381 = {}
        # Getting the type of 'AttributeError' (line 38)
        AttributeError_32379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 38)
        AttributeError_call_result_32382 = invoke(stypy.reporting.localization.Localization(__file__, 38, 18), AttributeError_32379, *[name_32380], **kwargs_32381)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 38, 12), AttributeError_call_result_32382, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 35)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _get_var(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'name' (line 39)
        name_32385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'name', False)
        # Getting the type of 'conf_desc' (line 39)
        conf_desc_32386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 35), 'conf_desc', False)
        # Processing the call keyword arguments (line 39)
        kwargs_32387 = {}
        # Getting the type of 'self' (line 39)
        self_32383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'self', False)
        # Obtaining the member '_get_var' of a type (line 39)
        _get_var_32384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), self_32383, '_get_var')
        # Calling _get_var(args, kwargs) (line 39)
        _get_var_call_result_32388 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), _get_var_32384, *[name_32385, conf_desc_32386], **kwargs_32387)
        
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', _get_var_call_result_32388)
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_32389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_32389


    @norecursion
    def get(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 41)
        None_32390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 32), 'None')
        defaults = [None_32390]
        # Create a new context for function 'get'
        module_type_store = module_type_store.open_function_context('get', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EnvironmentConfig.get.__dict__.__setitem__('stypy_localization', localization)
        EnvironmentConfig.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EnvironmentConfig.get.__dict__.__setitem__('stypy_type_store', module_type_store)
        EnvironmentConfig.get.__dict__.__setitem__('stypy_function_name', 'EnvironmentConfig.get')
        EnvironmentConfig.get.__dict__.__setitem__('stypy_param_names_list', ['name', 'default'])
        EnvironmentConfig.get.__dict__.__setitem__('stypy_varargs_param_name', None)
        EnvironmentConfig.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EnvironmentConfig.get.__dict__.__setitem__('stypy_call_defaults', defaults)
        EnvironmentConfig.get.__dict__.__setitem__('stypy_call_varargs', varargs)
        EnvironmentConfig.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EnvironmentConfig.get.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironmentConfig.get', ['name', 'default'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get', localization, ['name', 'default'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get(...)' code ##################

        
        
        # SSA begins for try-except statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 43):
        
        # Assigning a Subscript to a Name (line 43):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 43)
        name_32391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 40), 'name')
        # Getting the type of 'self' (line 43)
        self_32392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'self')
        # Obtaining the member '_conf_keys' of a type (line 43)
        _conf_keys_32393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 24), self_32392, '_conf_keys')
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___32394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 24), _conf_keys_32393, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_32395 = invoke(stypy.reporting.localization.Localization(__file__, 43, 24), getitem___32394, name_32391)
        
        # Assigning a type to the variable 'conf_desc' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'conf_desc', subscript_call_result_32395)
        # SSA branch for the except part of a try statement (line 42)
        # SSA branch for the except 'KeyError' branch of a try statement (line 42)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'default' (line 45)
        default_32396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'default')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type', default_32396)
        # SSA join for try-except statement (line 42)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to _get_var(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'name' (line 46)
        name_32399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 28), 'name', False)
        # Getting the type of 'conf_desc' (line 46)
        conf_desc_32400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'conf_desc', False)
        # Processing the call keyword arguments (line 46)
        kwargs_32401 = {}
        # Getting the type of 'self' (line 46)
        self_32397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'self', False)
        # Obtaining the member '_get_var' of a type (line 46)
        _get_var_32398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 14), self_32397, '_get_var')
        # Calling _get_var(args, kwargs) (line 46)
        _get_var_call_result_32402 = invoke(stypy.reporting.localization.Localization(__file__, 46, 14), _get_var_32398, *[name_32399, conf_desc_32400], **kwargs_32401)
        
        # Assigning a type to the variable 'var' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'var', _get_var_call_result_32402)
        
        # Type idiom detected: calculating its left and rigth part (line 47)
        # Getting the type of 'var' (line 47)
        var_32403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'var')
        # Getting the type of 'None' (line 47)
        None_32404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'None')
        
        (may_be_32405, more_types_in_union_32406) = may_be_none(var_32403, None_32404)

        if may_be_32405:

            if more_types_in_union_32406:
                # Runtime conditional SSA (line 47)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 48):
            
            # Assigning a Name to a Name (line 48):
            # Getting the type of 'default' (line 48)
            default_32407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 'default')
            # Assigning a type to the variable 'var' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'var', default_32407)

            if more_types_in_union_32406:
                # SSA join for if statement (line 47)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'var' (line 49)
        var_32408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'var')
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', var_32408)
        
        # ################# End of 'get(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_32409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32409)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get'
        return stypy_return_type_32409


    @norecursion
    def _get_var(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_var'
        module_type_store = module_type_store.open_function_context('_get_var', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_localization', localization)
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_type_store', module_type_store)
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_function_name', 'EnvironmentConfig._get_var')
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_param_names_list', ['name', 'conf_desc'])
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_varargs_param_name', None)
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_call_defaults', defaults)
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_call_varargs', varargs)
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EnvironmentConfig._get_var.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironmentConfig._get_var', ['name', 'conf_desc'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_var', localization, ['name', 'conf_desc'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_var(...)' code ##################

        
        # Assigning a Name to a Tuple (line 52):
        
        # Assigning a Subscript to a Name (line 52):
        
        # Obtaining the type of the subscript
        int_32410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'int')
        # Getting the type of 'conf_desc' (line 52)
        conf_desc_32411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'conf_desc')
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___32412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), conf_desc_32411, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_32413 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), getitem___32412, int_32410)
        
        # Assigning a type to the variable 'tuple_var_assignment_32246' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_32246', subscript_call_result_32413)
        
        # Assigning a Subscript to a Name (line 52):
        
        # Obtaining the type of the subscript
        int_32414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'int')
        # Getting the type of 'conf_desc' (line 52)
        conf_desc_32415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'conf_desc')
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___32416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), conf_desc_32415, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_32417 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), getitem___32416, int_32414)
        
        # Assigning a type to the variable 'tuple_var_assignment_32247' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_32247', subscript_call_result_32417)
        
        # Assigning a Subscript to a Name (line 52):
        
        # Obtaining the type of the subscript
        int_32418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'int')
        # Getting the type of 'conf_desc' (line 52)
        conf_desc_32419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'conf_desc')
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___32420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), conf_desc_32419, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_32421 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), getitem___32420, int_32418)
        
        # Assigning a type to the variable 'tuple_var_assignment_32248' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_32248', subscript_call_result_32421)
        
        # Assigning a Subscript to a Name (line 52):
        
        # Obtaining the type of the subscript
        int_32422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'int')
        # Getting the type of 'conf_desc' (line 52)
        conf_desc_32423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'conf_desc')
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___32424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), conf_desc_32423, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_32425 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), getitem___32424, int_32422)
        
        # Assigning a type to the variable 'tuple_var_assignment_32249' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_32249', subscript_call_result_32425)
        
        # Assigning a Name to a Name (line 52):
        # Getting the type of 'tuple_var_assignment_32246' (line 52)
        tuple_var_assignment_32246_32426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_32246')
        # Assigning a type to the variable 'hook' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'hook', tuple_var_assignment_32246_32426)
        
        # Assigning a Name to a Name (line 52):
        # Getting the type of 'tuple_var_assignment_32247' (line 52)
        tuple_var_assignment_32247_32427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_32247')
        # Assigning a type to the variable 'envvar' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 14), 'envvar', tuple_var_assignment_32247_32427)
        
        # Assigning a Name to a Name (line 52):
        # Getting the type of 'tuple_var_assignment_32248' (line 52)
        tuple_var_assignment_32248_32428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_32248')
        # Assigning a type to the variable 'confvar' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'confvar', tuple_var_assignment_32248_32428)
        
        # Assigning a Name to a Name (line 52):
        # Getting the type of 'tuple_var_assignment_32249' (line 52)
        tuple_var_assignment_32249_32429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'tuple_var_assignment_32249')
        # Assigning a type to the variable 'convert' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'convert', tuple_var_assignment_32249_32429)
        
        # Assigning a Call to a Name (line 53):
        
        # Assigning a Call to a Name (line 53):
        
        # Call to _hook_handler(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'name' (line 53)
        name_32432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'name', False)
        # Getting the type of 'hook' (line 53)
        hook_32433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 39), 'hook', False)
        # Processing the call keyword arguments (line 53)
        kwargs_32434 = {}
        # Getting the type of 'self' (line 53)
        self_32430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'self', False)
        # Obtaining the member '_hook_handler' of a type (line 53)
        _hook_handler_32431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 14), self_32430, '_hook_handler')
        # Calling _hook_handler(args, kwargs) (line 53)
        _hook_handler_call_result_32435 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), _hook_handler_32431, *[name_32432, hook_32433], **kwargs_32434)
        
        # Assigning a type to the variable 'var' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'var', _hook_handler_call_result_32435)
        
        # Type idiom detected: calculating its left and rigth part (line 54)
        # Getting the type of 'envvar' (line 54)
        envvar_32436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'envvar')
        # Getting the type of 'None' (line 54)
        None_32437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'None')
        
        (may_be_32438, more_types_in_union_32439) = may_not_be_none(envvar_32436, None_32437)

        if may_be_32438:

            if more_types_in_union_32439:
                # Runtime conditional SSA (line 54)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 55):
            
            # Assigning a Call to a Name (line 55):
            
            # Call to get(...): (line 55)
            # Processing the call arguments (line 55)
            # Getting the type of 'envvar' (line 55)
            envvar_32443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'envvar', False)
            # Getting the type of 'var' (line 55)
            var_32444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 41), 'var', False)
            # Processing the call keyword arguments (line 55)
            kwargs_32445 = {}
            # Getting the type of 'os' (line 55)
            os_32440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'os', False)
            # Obtaining the member 'environ' of a type (line 55)
            environ_32441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 18), os_32440, 'environ')
            # Obtaining the member 'get' of a type (line 55)
            get_32442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 18), environ_32441, 'get')
            # Calling get(args, kwargs) (line 55)
            get_call_result_32446 = invoke(stypy.reporting.localization.Localization(__file__, 55, 18), get_32442, *[envvar_32443, var_32444], **kwargs_32445)
            
            # Assigning a type to the variable 'var' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'var', get_call_result_32446)

            if more_types_in_union_32439:
                # SSA join for if statement (line 54)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'confvar' (line 56)
        confvar_32447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'confvar')
        # Getting the type of 'None' (line 56)
        None_32448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'None')
        # Applying the binary operator 'isnot' (line 56)
        result_is_not_32449 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 11), 'isnot', confvar_32447, None_32448)
        
        # Getting the type of 'self' (line 56)
        self_32450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'self')
        # Obtaining the member '_conf' of a type (line 56)
        _conf_32451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 35), self_32450, '_conf')
        # Applying the binary operator 'and' (line 56)
        result_and_keyword_32452 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 11), 'and', result_is_not_32449, _conf_32451)
        
        # Testing the type of an if condition (line 56)
        if_condition_32453 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 8), result_and_keyword_32452)
        # Assigning a type to the variable 'if_condition_32453' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'if_condition_32453', if_condition_32453)
        # SSA begins for if statement (line 56)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 57):
        
        # Assigning a Subscript to a Name (line 57):
        
        # Obtaining the type of the subscript
        int_32454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 55), 'int')
        
        # Call to get(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'confvar' (line 57)
        confvar_32458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 33), 'confvar', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 57)
        tuple_32459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 57)
        # Adding element type (line 57)
        # Getting the type of 'None' (line 57)
        None_32460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 43), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 43), tuple_32459, None_32460)
        # Adding element type (line 57)
        # Getting the type of 'var' (line 57)
        var_32461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 49), 'var', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 43), tuple_32459, var_32461)
        
        # Processing the call keyword arguments (line 57)
        kwargs_32462 = {}
        # Getting the type of 'self' (line 57)
        self_32455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'self', False)
        # Obtaining the member '_conf' of a type (line 57)
        _conf_32456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), self_32455, '_conf')
        # Obtaining the member 'get' of a type (line 57)
        get_32457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), _conf_32456, 'get')
        # Calling get(args, kwargs) (line 57)
        get_call_result_32463 = invoke(stypy.reporting.localization.Localization(__file__, 57, 18), get_32457, *[confvar_32458, tuple_32459], **kwargs_32462)
        
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___32464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), get_call_result_32463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_32465 = invoke(stypy.reporting.localization.Localization(__file__, 57, 18), getitem___32464, int_32454)
        
        # Assigning a type to the variable 'var' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'var', subscript_call_result_32465)
        # SSA join for if statement (line 56)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 58)
        # Getting the type of 'convert' (line 58)
        convert_32466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'convert')
        # Getting the type of 'None' (line 58)
        None_32467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'None')
        
        (may_be_32468, more_types_in_union_32469) = may_not_be_none(convert_32466, None_32467)

        if may_be_32468:

            if more_types_in_union_32469:
                # Runtime conditional SSA (line 58)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 59):
            
            # Assigning a Call to a Name (line 59):
            
            # Call to convert(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'var' (line 59)
            var_32471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'var', False)
            # Processing the call keyword arguments (line 59)
            kwargs_32472 = {}
            # Getting the type of 'convert' (line 59)
            convert_32470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'convert', False)
            # Calling convert(args, kwargs) (line 59)
            convert_call_result_32473 = invoke(stypy.reporting.localization.Localization(__file__, 59, 18), convert_32470, *[var_32471], **kwargs_32472)
            
            # Assigning a type to the variable 'var' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'var', convert_call_result_32473)

            if more_types_in_union_32469:
                # SSA join for if statement (line 58)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'var' (line 60)
        var_32474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'var')
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', var_32474)
        
        # ################# End of '_get_var(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_var' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_32475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32475)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_var'
        return stypy_return_type_32475


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_localization', localization)
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_function_name', 'EnvironmentConfig.clone')
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_param_names_list', ['hook_handler'])
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EnvironmentConfig.clone.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironmentConfig.clone', ['hook_handler'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clone', localization, ['hook_handler'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clone(...)' code ##################

        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to __class__(...): (line 63)
        # Processing the call keyword arguments (line 63)
        # Getting the type of 'self' (line 63)
        self_32478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 46), 'self', False)
        # Obtaining the member '_distutils_section' of a type (line 63)
        _distutils_section_32479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 46), self_32478, '_distutils_section')
        keyword_32480 = _distutils_section_32479
        # Getting the type of 'self' (line 64)
        self_32481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'self', False)
        # Obtaining the member '_conf_keys' of a type (line 64)
        _conf_keys_32482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 30), self_32481, '_conf_keys')
        kwargs_32483 = {'distutils_section': keyword_32480, '_conf_keys_32482': _conf_keys_32482}
        # Getting the type of 'self' (line 63)
        self_32476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'self', False)
        # Obtaining the member '__class__' of a type (line 63)
        class___32477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 13), self_32476, '__class__')
        # Calling __class__(args, kwargs) (line 63)
        class___call_result_32484 = invoke(stypy.reporting.localization.Localization(__file__, 63, 13), class___32477, *[], **kwargs_32483)
        
        # Assigning a type to the variable 'ec' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'ec', class___call_result_32484)
        
        # Assigning a Name to a Attribute (line 65):
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'hook_handler' (line 65)
        hook_handler_32485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'hook_handler')
        # Getting the type of 'ec' (line 65)
        ec_32486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'ec')
        # Setting the type of the member '_hook_handler' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), ec_32486, '_hook_handler', hook_handler_32485)
        # Getting the type of 'ec' (line 66)
        ec_32487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'ec')
        # Assigning a type to the variable 'stypy_return_type' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'stypy_return_type', ec_32487)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_32488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32488)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_32488


    @norecursion
    def use_distribution(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'use_distribution'
        module_type_store = module_type_store.open_function_context('use_distribution', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_localization', localization)
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_type_store', module_type_store)
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_function_name', 'EnvironmentConfig.use_distribution')
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_param_names_list', ['dist'])
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_varargs_param_name', None)
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_call_defaults', defaults)
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_call_varargs', varargs)
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EnvironmentConfig.use_distribution.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironmentConfig.use_distribution', ['dist'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'use_distribution', localization, ['dist'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'use_distribution(...)' code ##################

        
        
        # Call to isinstance(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'dist' (line 69)
        dist_32490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'dist', False)
        # Getting the type of 'Distribution' (line 69)
        Distribution_32491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'Distribution', False)
        # Processing the call keyword arguments (line 69)
        kwargs_32492 = {}
        # Getting the type of 'isinstance' (line 69)
        isinstance_32489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 69)
        isinstance_call_result_32493 = invoke(stypy.reporting.localization.Localization(__file__, 69, 11), isinstance_32489, *[dist_32490, Distribution_32491], **kwargs_32492)
        
        # Testing the type of an if condition (line 69)
        if_condition_32494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), isinstance_call_result_32493)
        # Assigning a type to the variable 'if_condition_32494' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_32494', if_condition_32494)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 70):
        
        # Assigning a Call to a Attribute (line 70):
        
        # Call to get_option_dict(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'self' (line 70)
        self_32497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 46), 'self', False)
        # Obtaining the member '_distutils_section' of a type (line 70)
        _distutils_section_32498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 46), self_32497, '_distutils_section')
        # Processing the call keyword arguments (line 70)
        kwargs_32499 = {}
        # Getting the type of 'dist' (line 70)
        dist_32495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'dist', False)
        # Obtaining the member 'get_option_dict' of a type (line 70)
        get_option_dict_32496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 25), dist_32495, 'get_option_dict')
        # Calling get_option_dict(args, kwargs) (line 70)
        get_option_dict_call_result_32500 = invoke(stypy.reporting.localization.Localization(__file__, 70, 25), get_option_dict_32496, *[_distutils_section_32498], **kwargs_32499)
        
        # Getting the type of 'self' (line 70)
        self_32501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'self')
        # Setting the type of the member '_conf' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), self_32501, '_conf', get_option_dict_call_result_32500)
        # SSA branch for the else part of an if statement (line 69)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 72):
        
        # Assigning a Name to a Attribute (line 72):
        # Getting the type of 'dist' (line 72)
        dist_32502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'dist')
        # Getting the type of 'self' (line 72)
        self_32503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'self')
        # Setting the type of the member '_conf' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), self_32503, '_conf', dist_32502)
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'use_distribution(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'use_distribution' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_32504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32504)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'use_distribution'
        return stypy_return_type_32504


# Assigning a type to the variable 'EnvironmentConfig' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'EnvironmentConfig', EnvironmentConfig)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
