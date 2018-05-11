
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from type_error_copy import TypeError
2: 
3: 
4: class UndefinedTypeError(TypeError):
5:     '''
6:     Child class of TypeError to model an special type of error: A variable has a type that cannot be determined.
7:     '''
8: 
9:     def __init__(self, localization, msg, prints_msg=True):
10:         TypeError.__init__(self, localization, msg, prints_msg)
11:         #super(UndefinedTypeError, self).__init__(localization, msg, prints_msg)

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from type_error_copy import TypeError' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')
import_4165 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_error_copy')

if (type(import_4165) is not StypyTypeError):

    if (import_4165 != 'pyd_module'):
        __import__(import_4165)
        sys_modules_4166 = sys.modules[import_4165]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_error_copy', sys_modules_4166.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_4166, sys_modules_4166.module_type_store, module_type_store)
    else:
        from type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'type_error_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_error_copy', import_4165)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')

# Declaration of the 'UndefinedTypeError' class
# Getting the type of 'TypeError' (line 4)
TypeError_4167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 25), 'TypeError')

class UndefinedTypeError(TypeError_4167, ):
    str_4168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\n    Child class of TypeError to model an special type of error: A variable has a type that cannot be determined.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 9)
        True_4169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 53), 'True')
        defaults = [True_4169]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UndefinedTypeError.__init__', ['localization', 'msg', 'prints_msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['localization', 'msg', 'prints_msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 10)
        # Processing the call arguments (line 10)
        # Getting the type of 'self' (line 10)
        self_4172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 27), 'self', False)
        # Getting the type of 'localization' (line 10)
        localization_4173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 33), 'localization', False)
        # Getting the type of 'msg' (line 10)
        msg_4174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 47), 'msg', False)
        # Getting the type of 'prints_msg' (line 10)
        prints_msg_4175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 52), 'prints_msg', False)
        # Processing the call keyword arguments (line 10)
        kwargs_4176 = {}
        # Getting the type of 'TypeError' (line 10)
        TypeError_4170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'TypeError', False)
        # Obtaining the member '__init__' of a type (line 10)
        init___4171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), TypeError_4170, '__init__')
        # Calling __init__(args, kwargs) (line 10)
        init___call_result_4177 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), init___4171, *[self_4172, localization_4173, msg_4174, prints_msg_4175], **kwargs_4176)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'UndefinedTypeError' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'UndefinedTypeError', UndefinedTypeError)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
