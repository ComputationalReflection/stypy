
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import gc
5: import os
6: from nose.plugins import Plugin
7: 
8: 
9: class PerformGC(Plugin):
10:     '''This plugin adds option to call ``gc.collect`` after each test'''
11:     enabled = False
12: 
13:     def options(self, parser, env=os.environ):
14:         env_opt = 'PERFORM_GC'
15:         parser.add_option('--perform-gc', action='store_true',
16:                           dest='performGC', default=env.get(env_opt, False),
17:                           help='Call gc.collect() after each test')
18: 
19:     def configure(self, options, conf):
20:         if not self.can_configure:
21:             return
22: 
23:         self.enabled = getattr(options, 'performGC', False)
24: 
25:     def afterTest(self, test):
26:         gc.collect()
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import gc' statement (line 4)
import gc

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'gc', gc, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from nose.plugins import Plugin' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/plugins/')
import_294312 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'nose.plugins')

if (type(import_294312) is not StypyTypeError):

    if (import_294312 != 'pyd_module'):
        __import__(import_294312)
        sys_modules_294313 = sys.modules[import_294312]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'nose.plugins', sys_modules_294313.module_type_store, module_type_store, ['Plugin'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_294313, sys_modules_294313.module_type_store, module_type_store)
    else:
        from nose.plugins import Plugin

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'nose.plugins', None, module_type_store, ['Plugin'], [Plugin])

else:
    # Assigning a type to the variable 'nose.plugins' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'nose.plugins', import_294312)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/plugins/')

# Declaration of the 'PerformGC' class
# Getting the type of 'Plugin' (line 9)
Plugin_294314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 16), 'Plugin')

class PerformGC(Plugin_294314, ):
    unicode_294315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'unicode', u'This plugin adds option to call ``gc.collect`` after each test')

    @norecursion
    def options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'os' (line 13)
        os_294316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 34), 'os')
        # Obtaining the member 'environ' of a type (line 13)
        environ_294317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 34), os_294316, 'environ')
        defaults = [environ_294317]
        # Create a new context for function 'options'
        module_type_store = module_type_store.open_function_context('options', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PerformGC.options.__dict__.__setitem__('stypy_localization', localization)
        PerformGC.options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PerformGC.options.__dict__.__setitem__('stypy_type_store', module_type_store)
        PerformGC.options.__dict__.__setitem__('stypy_function_name', 'PerformGC.options')
        PerformGC.options.__dict__.__setitem__('stypy_param_names_list', ['parser', 'env'])
        PerformGC.options.__dict__.__setitem__('stypy_varargs_param_name', None)
        PerformGC.options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PerformGC.options.__dict__.__setitem__('stypy_call_defaults', defaults)
        PerformGC.options.__dict__.__setitem__('stypy_call_varargs', varargs)
        PerformGC.options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PerformGC.options.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PerformGC.options', ['parser', 'env'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 14):
        unicode_294318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'unicode', u'PERFORM_GC')
        # Assigning a type to the variable 'env_opt' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'env_opt', unicode_294318)
        
        # Call to add_option(...): (line 15)
        # Processing the call arguments (line 15)
        unicode_294321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'unicode', u'--perform-gc')
        # Processing the call keyword arguments (line 15)
        unicode_294322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 49), 'unicode', u'store_true')
        keyword_294323 = unicode_294322
        unicode_294324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 31), 'unicode', u'performGC')
        keyword_294325 = unicode_294324
        
        # Call to get(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'env_opt' (line 16)
        env_opt_294328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 60), 'env_opt', False)
        # Getting the type of 'False' (line 16)
        False_294329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 69), 'False', False)
        # Processing the call keyword arguments (line 16)
        kwargs_294330 = {}
        # Getting the type of 'env' (line 16)
        env_294326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 52), 'env', False)
        # Obtaining the member 'get' of a type (line 16)
        get_294327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 52), env_294326, 'get')
        # Calling get(args, kwargs) (line 16)
        get_call_result_294331 = invoke(stypy.reporting.localization.Localization(__file__, 16, 52), get_294327, *[env_opt_294328, False_294329], **kwargs_294330)
        
        keyword_294332 = get_call_result_294331
        unicode_294333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'unicode', u'Call gc.collect() after each test')
        keyword_294334 = unicode_294333
        kwargs_294335 = {'action': keyword_294323, 'dest': keyword_294325, 'default': keyword_294332, 'help': keyword_294334}
        # Getting the type of 'parser' (line 15)
        parser_294319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'parser', False)
        # Obtaining the member 'add_option' of a type (line 15)
        add_option_294320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), parser_294319, 'add_option')
        # Calling add_option(args, kwargs) (line 15)
        add_option_call_result_294336 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), add_option_294320, *[unicode_294321], **kwargs_294335)
        
        
        # ################# End of 'options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'options' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_294337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294337)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'options'
        return stypy_return_type_294337


    @norecursion
    def configure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'configure'
        module_type_store = module_type_store.open_function_context('configure', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PerformGC.configure.__dict__.__setitem__('stypy_localization', localization)
        PerformGC.configure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PerformGC.configure.__dict__.__setitem__('stypy_type_store', module_type_store)
        PerformGC.configure.__dict__.__setitem__('stypy_function_name', 'PerformGC.configure')
        PerformGC.configure.__dict__.__setitem__('stypy_param_names_list', ['options', 'conf'])
        PerformGC.configure.__dict__.__setitem__('stypy_varargs_param_name', None)
        PerformGC.configure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PerformGC.configure.__dict__.__setitem__('stypy_call_defaults', defaults)
        PerformGC.configure.__dict__.__setitem__('stypy_call_varargs', varargs)
        PerformGC.configure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PerformGC.configure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PerformGC.configure', ['options', 'conf'], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 20)
        self_294338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'self')
        # Obtaining the member 'can_configure' of a type (line 20)
        can_configure_294339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 15), self_294338, 'can_configure')
        # Applying the 'not' unary operator (line 20)
        result_not__294340 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 11), 'not', can_configure_294339)
        
        # Testing the type of an if condition (line 20)
        if_condition_294341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 8), result_not__294340)
        # Assigning a type to the variable 'if_condition_294341' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'if_condition_294341', if_condition_294341)
        # SSA begins for if statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 23):
        
        # Call to getattr(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'options' (line 23)
        options_294343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 31), 'options', False)
        unicode_294344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 40), 'unicode', u'performGC')
        # Getting the type of 'False' (line 23)
        False_294345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 53), 'False', False)
        # Processing the call keyword arguments (line 23)
        kwargs_294346 = {}
        # Getting the type of 'getattr' (line 23)
        getattr_294342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'getattr', False)
        # Calling getattr(args, kwargs) (line 23)
        getattr_call_result_294347 = invoke(stypy.reporting.localization.Localization(__file__, 23, 23), getattr_294342, *[options_294343, unicode_294344, False_294345], **kwargs_294346)
        
        # Getting the type of 'self' (line 23)
        self_294348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'enabled' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_294348, 'enabled', getattr_call_result_294347)
        
        # ################# End of 'configure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'configure' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_294349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294349)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'configure'
        return stypy_return_type_294349


    @norecursion
    def afterTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'afterTest'
        module_type_store = module_type_store.open_function_context('afterTest', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PerformGC.afterTest.__dict__.__setitem__('stypy_localization', localization)
        PerformGC.afterTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PerformGC.afterTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        PerformGC.afterTest.__dict__.__setitem__('stypy_function_name', 'PerformGC.afterTest')
        PerformGC.afterTest.__dict__.__setitem__('stypy_param_names_list', ['test'])
        PerformGC.afterTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        PerformGC.afterTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PerformGC.afterTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        PerformGC.afterTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        PerformGC.afterTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PerformGC.afterTest.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PerformGC.afterTest', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'afterTest', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'afterTest(...)' code ##################

        
        # Call to collect(...): (line 26)
        # Processing the call keyword arguments (line 26)
        kwargs_294352 = {}
        # Getting the type of 'gc' (line 26)
        gc_294350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'gc', False)
        # Obtaining the member 'collect' of a type (line 26)
        collect_294351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), gc_294350, 'collect')
        # Calling collect(args, kwargs) (line 26)
        collect_call_result_294353 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), collect_294351, *[], **kwargs_294352)
        
        
        # ################# End of 'afterTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'afterTest' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_294354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294354)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'afterTest'
        return stypy_return_type_294354


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PerformGC.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'PerformGC' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'PerformGC', PerformGC)

# Assigning a Name to a Name (line 11):
# Getting the type of 'False' (line 11)
False_294355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'False')
# Getting the type of 'PerformGC'
PerformGC_294356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PerformGC')
# Setting the type of the member 'enabled' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PerformGC_294356, 'enabled', False_294355)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
