
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ml import minilight
2: import os
3: 
4: '''
5:   Copyright (c) 2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
6:   http://www.hxa7241.org/minilight/
7: '''
8: 
9: 
10: def Relative(path):
11:     return os.path.join(os.path.dirname(__file__), path)
12: 
13: 
14: class MinilightRun:
15:     def main(self):
16:         minilight.main(Relative('ml/cornellbox.txt'))
17: 
18:     def run(self):
19:         self.main()
20:         return True
21: 
22: 
23: def run():
24:     m = MinilightRun()
25:     m.run()
26: 
27: 
28: run()
29: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from ml import minilight' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/benchmark_suite/shedskin/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'ml')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'ml', sys_modules_2.module_type_store, module_type_store, ['minilight'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_2, sys_modules_2.module_type_store, module_type_store)
    else:
        from ml import minilight

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'ml', None, module_type_store, ['minilight'], [minilight])

else:
    # Assigning a type to the variable 'ml' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'ml', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/benchmark_suite/shedskin/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\n  Copyright (c) 2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.\n  http://www.hxa7241.org/minilight/\n')

@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 10, 0, False)
    
    # Passed parameters checking function
    Relative.stypy_localization = localization
    Relative.stypy_type_of_self = None
    Relative.stypy_type_store = module_type_store
    Relative.stypy_function_name = 'Relative'
    Relative.stypy_param_names_list = ['path']
    Relative.stypy_varargs_param_name = None
    Relative.stypy_kwargs_param_name = None
    Relative.stypy_call_defaults = defaults
    Relative.stypy_call_varargs = varargs
    Relative.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Relative', ['path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Relative', localization, ['path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Relative(...)' code ##################

    
    # Call to join(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Call to dirname(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of '__file__' (line 11)
    file___10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 40), '__file__', False)
    # Processing the call keyword arguments (line 11)
    kwargs_11 = {}
    # Getting the type of 'os' (line 11)
    os_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 11)
    path_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 24), os_7, 'path')
    # Obtaining the member 'dirname' of a type (line 11)
    dirname_9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 24), path_8, 'dirname')
    # Calling dirname(args, kwargs) (line 11)
    dirname_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 11, 24), dirname_9, *[file___10], **kwargs_11)
    
    # Getting the type of 'path' (line 11)
    path_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 51), 'path', False)
    # Processing the call keyword arguments (line 11)
    kwargs_14 = {}
    # Getting the type of 'os' (line 11)
    os_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 11)
    path_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), os_4, 'path')
    # Obtaining the member 'join' of a type (line 11)
    join_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), path_5, 'join')
    # Calling join(args, kwargs) (line 11)
    join_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), join_6, *[dirname_call_result_12, path_13], **kwargs_14)
    
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', join_call_result_15)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_16

# Assigning a type to the variable 'Relative' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Relative', Relative)
# Declaration of the 'MinilightRun' class

class MinilightRun:

    @norecursion
    def main(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'main'
        module_type_store = module_type_store.open_function_context('main', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MinilightRun.main.__dict__.__setitem__('stypy_localization', localization)
        MinilightRun.main.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MinilightRun.main.__dict__.__setitem__('stypy_type_store', module_type_store)
        MinilightRun.main.__dict__.__setitem__('stypy_function_name', 'MinilightRun.main')
        MinilightRun.main.__dict__.__setitem__('stypy_param_names_list', [])
        MinilightRun.main.__dict__.__setitem__('stypy_varargs_param_name', None)
        MinilightRun.main.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MinilightRun.main.__dict__.__setitem__('stypy_call_defaults', defaults)
        MinilightRun.main.__dict__.__setitem__('stypy_call_varargs', varargs)
        MinilightRun.main.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MinilightRun.main.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MinilightRun.main', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to main(...): (line 16)
        # Processing the call arguments (line 16)
        
        # Call to Relative(...): (line 16)
        # Processing the call arguments (line 16)
        str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 32), 'str', 'ml/cornellbox.txt')
        # Processing the call keyword arguments (line 16)
        kwargs_21 = {}
        # Getting the type of 'Relative' (line 16)
        Relative_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'Relative', False)
        # Calling Relative(args, kwargs) (line 16)
        Relative_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 16, 23), Relative_19, *[str_20], **kwargs_21)
        
        # Processing the call keyword arguments (line 16)
        kwargs_23 = {}
        # Getting the type of 'minilight' (line 16)
        minilight_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'minilight', False)
        # Obtaining the member 'main' of a type (line 16)
        main_18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), minilight_17, 'main')
        # Calling main(args, kwargs) (line 16)
        main_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), main_18, *[Relative_call_result_22], **kwargs_23)
        
        
        # ################# End of 'main(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'main' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'main'
        return stypy_return_type_25


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MinilightRun.run.__dict__.__setitem__('stypy_localization', localization)
        MinilightRun.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MinilightRun.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        MinilightRun.run.__dict__.__setitem__('stypy_function_name', 'MinilightRun.run')
        MinilightRun.run.__dict__.__setitem__('stypy_param_names_list', [])
        MinilightRun.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        MinilightRun.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MinilightRun.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        MinilightRun.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        MinilightRun.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MinilightRun.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MinilightRun.run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Call to main(...): (line 19)
        # Processing the call keyword arguments (line 19)
        kwargs_28 = {}
        # Getting the type of 'self' (line 19)
        self_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self', False)
        # Obtaining the member 'main' of a type (line 19)
        main_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_26, 'main')
        # Calling main(args, kwargs) (line 19)
        main_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), main_27, *[], **kwargs_28)
        
        # Getting the type of 'True' (line 20)
        True_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', True_30)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_31


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 0, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MinilightRun.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MinilightRun' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'MinilightRun', MinilightRun)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 23, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = []
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    # Assigning a Call to a Name (line 24):
    
    # Call to MinilightRun(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_33 = {}
    # Getting the type of 'MinilightRun' (line 24)
    MinilightRun_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'MinilightRun', False)
    # Calling MinilightRun(args, kwargs) (line 24)
    MinilightRun_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), MinilightRun_32, *[], **kwargs_33)
    
    # Assigning a type to the variable 'm' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'm', MinilightRun_call_result_34)
    
    # Call to run(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_37 = {}
    # Getting the type of 'm' (line 25)
    m_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'm', False)
    # Obtaining the member 'run' of a type (line 25)
    run_36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), m_35, 'run')
    # Calling run(args, kwargs) (line 25)
    run_call_result_38 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), run_36, *[], **kwargs_37)
    
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_39

# Assigning a type to the variable 'run' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'run', run)

# Call to run(...): (line 28)
# Processing the call keyword arguments (line 28)
kwargs_41 = {}
# Getting the type of 'run' (line 28)
run_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'run', False)
# Calling run(args, kwargs) (line 28)
run_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 28, 0), run_40, *[], **kwargs_41)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
