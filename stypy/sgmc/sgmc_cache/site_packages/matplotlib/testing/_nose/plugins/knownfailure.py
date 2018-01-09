
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import os
5: from nose.plugins.errorclass import ErrorClass, ErrorClassPlugin
6: from ..exceptions import KnownFailureTest
7: 
8: 
9: class KnownFailure(ErrorClassPlugin):
10:     '''Plugin that installs a KNOWNFAIL error class for the
11:     KnownFailureClass exception.  When KnownFailureTest is raised,
12:     the exception will be logged in the knownfail attribute of the
13:     result, 'K' or 'KNOWNFAIL' (verbose) will be output, and the
14:     exception will not be counted as an error or failure.
15: 
16:     This is based on numpy.testing.noseclasses.KnownFailure.
17:     '''
18:     enabled = True
19:     knownfail = ErrorClass(KnownFailureTest,
20:                            label='KNOWNFAIL',
21:                            isfailure=False)
22: 
23:     def options(self, parser, env=os.environ):
24:         env_opt = 'NOSE_WITHOUT_KNOWNFAIL'
25:         parser.add_option('--no-knownfail', action='store_true',
26:                           dest='noKnownFail', default=env.get(env_opt, False),
27:                           help='Disable special handling of KnownFailureTest '
28:                                'exceptions')
29: 
30:     def configure(self, options, conf):
31:         if not self.can_configure:
32:             return
33:         self.conf = conf
34:         disable = getattr(options, 'noKnownFail', False)
35:         if disable:
36:             self.enabled = False
37: 
38:     def addError(self, test, err, *zero_nine_capt_args):
39:         # Fixme (Really weird): if I don't leave empty method here,
40:         # nose gets confused and KnownFails become testing errors when
41:         # using the MplNosePlugin and MplTestCase.
42: 
43:         # The *zero_nine_capt_args captures an extra argument. There
44:         # seems to be a bug in
45:         # nose.testing.manager.ZeroNinePlugin.addError() in which a
46:         # 3rd positional argument ("capt") is passed to the plugin's
47:         # addError() method, even if one is not explicitly using the
48:         # ZeroNinePlugin.
49:         pass
50: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from nose.plugins.errorclass import ErrorClass, ErrorClassPlugin' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/plugins/')
import_294255 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'nose.plugins.errorclass')

if (type(import_294255) is not StypyTypeError):

    if (import_294255 != 'pyd_module'):
        __import__(import_294255)
        sys_modules_294256 = sys.modules[import_294255]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'nose.plugins.errorclass', sys_modules_294256.module_type_store, module_type_store, ['ErrorClass', 'ErrorClassPlugin'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_294256, sys_modules_294256.module_type_store, module_type_store)
    else:
        from nose.plugins.errorclass import ErrorClass, ErrorClassPlugin

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'nose.plugins.errorclass', None, module_type_store, ['ErrorClass', 'ErrorClassPlugin'], [ErrorClass, ErrorClassPlugin])

else:
    # Assigning a type to the variable 'nose.plugins.errorclass' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'nose.plugins.errorclass', import_294255)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/plugins/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from matplotlib.testing._nose.exceptions import KnownFailureTest' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/plugins/')
import_294257 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.testing._nose.exceptions')

if (type(import_294257) is not StypyTypeError):

    if (import_294257 != 'pyd_module'):
        __import__(import_294257)
        sys_modules_294258 = sys.modules[import_294257]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.testing._nose.exceptions', sys_modules_294258.module_type_store, module_type_store, ['KnownFailureTest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_294258, sys_modules_294258.module_type_store, module_type_store)
    else:
        from matplotlib.testing._nose.exceptions import KnownFailureTest

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.testing._nose.exceptions', None, module_type_store, ['KnownFailureTest'], [KnownFailureTest])

else:
    # Assigning a type to the variable 'matplotlib.testing._nose.exceptions' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.testing._nose.exceptions', import_294257)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/plugins/')

# Declaration of the 'KnownFailure' class
# Getting the type of 'ErrorClassPlugin' (line 9)
ErrorClassPlugin_294259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'ErrorClassPlugin')

class KnownFailure(ErrorClassPlugin_294259, ):
    unicode_294260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'unicode', u"Plugin that installs a KNOWNFAIL error class for the\n    KnownFailureClass exception.  When KnownFailureTest is raised,\n    the exception will be logged in the knownfail attribute of the\n    result, 'K' or 'KNOWNFAIL' (verbose) will be output, and the\n    exception will not be counted as an error or failure.\n\n    This is based on numpy.testing.noseclasses.KnownFailure.\n    ")

    @norecursion
    def options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'os' (line 23)
        os_294261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'os')
        # Obtaining the member 'environ' of a type (line 23)
        environ_294262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 34), os_294261, 'environ')
        defaults = [environ_294262]
        # Create a new context for function 'options'
        module_type_store = module_type_store.open_function_context('options', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KnownFailure.options.__dict__.__setitem__('stypy_localization', localization)
        KnownFailure.options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KnownFailure.options.__dict__.__setitem__('stypy_type_store', module_type_store)
        KnownFailure.options.__dict__.__setitem__('stypy_function_name', 'KnownFailure.options')
        KnownFailure.options.__dict__.__setitem__('stypy_param_names_list', ['parser', 'env'])
        KnownFailure.options.__dict__.__setitem__('stypy_varargs_param_name', None)
        KnownFailure.options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KnownFailure.options.__dict__.__setitem__('stypy_call_defaults', defaults)
        KnownFailure.options.__dict__.__setitem__('stypy_call_varargs', varargs)
        KnownFailure.options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KnownFailure.options.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KnownFailure.options', ['parser', 'env'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'options', localization, ['parser', 'env'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'options(...)' code ##################

        
        # Assigning a Str to a Name (line 24):
        unicode_294263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 18), 'unicode', u'NOSE_WITHOUT_KNOWNFAIL')
        # Assigning a type to the variable 'env_opt' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'env_opt', unicode_294263)
        
        # Call to add_option(...): (line 25)
        # Processing the call arguments (line 25)
        unicode_294266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'unicode', u'--no-knownfail')
        # Processing the call keyword arguments (line 25)
        unicode_294267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 51), 'unicode', u'store_true')
        keyword_294268 = unicode_294267
        unicode_294269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'unicode', u'noKnownFail')
        keyword_294270 = unicode_294269
        
        # Call to get(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'env_opt' (line 26)
        env_opt_294273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 62), 'env_opt', False)
        # Getting the type of 'False' (line 26)
        False_294274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 71), 'False', False)
        # Processing the call keyword arguments (line 26)
        kwargs_294275 = {}
        # Getting the type of 'env' (line 26)
        env_294271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 54), 'env', False)
        # Obtaining the member 'get' of a type (line 26)
        get_294272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 54), env_294271, 'get')
        # Calling get(args, kwargs) (line 26)
        get_call_result_294276 = invoke(stypy.reporting.localization.Localization(__file__, 26, 54), get_294272, *[env_opt_294273, False_294274], **kwargs_294275)
        
        keyword_294277 = get_call_result_294276
        unicode_294278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'unicode', u'Disable special handling of KnownFailureTest exceptions')
        keyword_294279 = unicode_294278
        kwargs_294280 = {'action': keyword_294268, 'dest': keyword_294270, 'default': keyword_294277, 'help': keyword_294279}
        # Getting the type of 'parser' (line 25)
        parser_294264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'parser', False)
        # Obtaining the member 'add_option' of a type (line 25)
        add_option_294265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), parser_294264, 'add_option')
        # Calling add_option(args, kwargs) (line 25)
        add_option_call_result_294281 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), add_option_294265, *[unicode_294266], **kwargs_294280)
        
        
        # ################# End of 'options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'options' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_294282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294282)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'options'
        return stypy_return_type_294282


    @norecursion
    def configure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'configure'
        module_type_store = module_type_store.open_function_context('configure', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KnownFailure.configure.__dict__.__setitem__('stypy_localization', localization)
        KnownFailure.configure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KnownFailure.configure.__dict__.__setitem__('stypy_type_store', module_type_store)
        KnownFailure.configure.__dict__.__setitem__('stypy_function_name', 'KnownFailure.configure')
        KnownFailure.configure.__dict__.__setitem__('stypy_param_names_list', ['options', 'conf'])
        KnownFailure.configure.__dict__.__setitem__('stypy_varargs_param_name', None)
        KnownFailure.configure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KnownFailure.configure.__dict__.__setitem__('stypy_call_defaults', defaults)
        KnownFailure.configure.__dict__.__setitem__('stypy_call_varargs', varargs)
        KnownFailure.configure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KnownFailure.configure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KnownFailure.configure', ['options', 'conf'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'configure', localization, ['options', 'conf'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'configure(...)' code ##################

        
        
        # Getting the type of 'self' (line 31)
        self_294283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'self')
        # Obtaining the member 'can_configure' of a type (line 31)
        can_configure_294284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), self_294283, 'can_configure')
        # Applying the 'not' unary operator (line 31)
        result_not__294285 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), 'not', can_configure_294284)
        
        # Testing the type of an if condition (line 31)
        if_condition_294286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), result_not__294285)
        # Assigning a type to the variable 'if_condition_294286' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_294286', if_condition_294286)
        # SSA begins for if statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 31)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'conf' (line 33)
        conf_294287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'conf')
        # Getting the type of 'self' (line 33)
        self_294288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'conf' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_294288, 'conf', conf_294287)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to getattr(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'options' (line 34)
        options_294290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'options', False)
        unicode_294291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 35), 'unicode', u'noKnownFail')
        # Getting the type of 'False' (line 34)
        False_294292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 50), 'False', False)
        # Processing the call keyword arguments (line 34)
        kwargs_294293 = {}
        # Getting the type of 'getattr' (line 34)
        getattr_294289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 34)
        getattr_call_result_294294 = invoke(stypy.reporting.localization.Localization(__file__, 34, 18), getattr_294289, *[options_294290, unicode_294291, False_294292], **kwargs_294293)
        
        # Assigning a type to the variable 'disable' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'disable', getattr_call_result_294294)
        
        # Getting the type of 'disable' (line 35)
        disable_294295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'disable')
        # Testing the type of an if condition (line 35)
        if_condition_294296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), disable_294295)
        # Assigning a type to the variable 'if_condition_294296' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'if_condition_294296', if_condition_294296)
        # SSA begins for if statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'False' (line 36)
        False_294297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'False')
        # Getting the type of 'self' (line 36)
        self_294298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self')
        # Setting the type of the member 'enabled' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_294298, 'enabled', False_294297)
        # SSA join for if statement (line 35)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'configure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'configure' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_294299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294299)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'configure'
        return stypy_return_type_294299


    @norecursion
    def addError(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addError'
        module_type_store = module_type_store.open_function_context('addError', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KnownFailure.addError.__dict__.__setitem__('stypy_localization', localization)
        KnownFailure.addError.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KnownFailure.addError.__dict__.__setitem__('stypy_type_store', module_type_store)
        KnownFailure.addError.__dict__.__setitem__('stypy_function_name', 'KnownFailure.addError')
        KnownFailure.addError.__dict__.__setitem__('stypy_param_names_list', ['test', 'err'])
        KnownFailure.addError.__dict__.__setitem__('stypy_varargs_param_name', 'zero_nine_capt_args')
        KnownFailure.addError.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KnownFailure.addError.__dict__.__setitem__('stypy_call_defaults', defaults)
        KnownFailure.addError.__dict__.__setitem__('stypy_call_varargs', varargs)
        KnownFailure.addError.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KnownFailure.addError.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KnownFailure.addError', ['test', 'err'], 'zero_nine_capt_args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addError', localization, ['test', 'err'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addError(...)' code ##################

        pass
        
        # ################# End of 'addError(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addError' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_294300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294300)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addError'
        return stypy_return_type_294300


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 0, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KnownFailure.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'KnownFailure' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'KnownFailure', KnownFailure)

# Assigning a Name to a Name (line 18):
# Getting the type of 'True' (line 18)
True_294301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'True')
# Getting the type of 'KnownFailure'
KnownFailure_294302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'KnownFailure')
# Setting the type of the member 'enabled' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), KnownFailure_294302, 'enabled', True_294301)

# Assigning a Call to a Name (line 19):

# Call to ErrorClass(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'KnownFailureTest' (line 19)
KnownFailureTest_294304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'KnownFailureTest', False)
# Processing the call keyword arguments (line 19)
unicode_294305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'unicode', u'KNOWNFAIL')
keyword_294306 = unicode_294305
# Getting the type of 'False' (line 21)
False_294307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 37), 'False', False)
keyword_294308 = False_294307
kwargs_294309 = {'isfailure': keyword_294308, 'label': keyword_294306}
# Getting the type of 'ErrorClass' (line 19)
ErrorClass_294303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'ErrorClass', False)
# Calling ErrorClass(args, kwargs) (line 19)
ErrorClass_call_result_294310 = invoke(stypy.reporting.localization.Localization(__file__, 19, 16), ErrorClass_294303, *[KnownFailureTest_294304], **kwargs_294309)

# Getting the type of 'KnownFailure'
KnownFailure_294311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'KnownFailure')
# Setting the type of the member 'knownfail' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), KnownFailure_294311, 'knownfail', ErrorClass_call_result_294310)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
