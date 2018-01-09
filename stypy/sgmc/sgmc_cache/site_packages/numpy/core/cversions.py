
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Simple script to compute the api hash of the current API.
2: 
3: The API has is defined by numpy_api_order and ufunc_api_order.
4: 
5: '''
6: from __future__ import division, absolute_import, print_function
7: 
8: from os.path import dirname
9: 
10: from code_generators.genapi import fullapi_hash
11: from code_generators.numpy_api import full_api
12: 
13: if __name__ == '__main__':
14:     curdir = dirname(__file__)
15:     print(fullapi_hash(full_api))
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'Simple script to compute the api hash of the current API.\n\nThe API has is defined by numpy_api_order and ufunc_api_order.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from os.path import dirname' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_1985 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os.path')

if (type(import_1985) is not StypyTypeError):

    if (import_1985 != 'pyd_module'):
        __import__(import_1985)
        sys_modules_1986 = sys.modules[import_1985]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os.path', sys_modules_1986.module_type_store, module_type_store, ['dirname'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_1986, sys_modules_1986.module_type_store, module_type_store)
    else:
        from os.path import dirname

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os.path', None, module_type_store, ['dirname'], [dirname])

else:
    # Assigning a type to the variable 'os.path' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'os.path', import_1985)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from code_generators.genapi import fullapi_hash' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_1987 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generators.genapi')

if (type(import_1987) is not StypyTypeError):

    if (import_1987 != 'pyd_module'):
        __import__(import_1987)
        sys_modules_1988 = sys.modules[import_1987]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generators.genapi', sys_modules_1988.module_type_store, module_type_store, ['fullapi_hash'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_1988, sys_modules_1988.module_type_store, module_type_store)
    else:
        from code_generators.genapi import fullapi_hash

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generators.genapi', None, module_type_store, ['fullapi_hash'], [fullapi_hash])

else:
    # Assigning a type to the variable 'code_generators.genapi' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generators.genapi', import_1987)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from code_generators.numpy_api import full_api' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_1989 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'code_generators.numpy_api')

if (type(import_1989) is not StypyTypeError):

    if (import_1989 != 'pyd_module'):
        __import__(import_1989)
        sys_modules_1990 = sys.modules[import_1989]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'code_generators.numpy_api', sys_modules_1990.module_type_store, module_type_store, ['full_api'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_1990, sys_modules_1990.module_type_store, module_type_store)
    else:
        from code_generators.numpy_api import full_api

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'code_generators.numpy_api', None, module_type_store, ['full_api'], [full_api])

else:
    # Assigning a type to the variable 'code_generators.numpy_api' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'code_generators.numpy_api', import_1989)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to dirname(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of '__file__' (line 14)
    file___1992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 21), '__file__', False)
    # Processing the call keyword arguments (line 14)
    kwargs_1993 = {}
    # Getting the type of 'dirname' (line 14)
    dirname_1991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'dirname', False)
    # Calling dirname(args, kwargs) (line 14)
    dirname_call_result_1994 = invoke(stypy.reporting.localization.Localization(__file__, 14, 13), dirname_1991, *[file___1992], **kwargs_1993)
    
    # Assigning a type to the variable 'curdir' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'curdir', dirname_call_result_1994)
    
    # Call to print(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to fullapi_hash(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'full_api' (line 15)
    full_api_1997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'full_api', False)
    # Processing the call keyword arguments (line 15)
    kwargs_1998 = {}
    # Getting the type of 'fullapi_hash' (line 15)
    fullapi_hash_1996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'fullapi_hash', False)
    # Calling fullapi_hash(args, kwargs) (line 15)
    fullapi_hash_call_result_1999 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), fullapi_hash_1996, *[full_api_1997], **kwargs_1998)
    
    # Processing the call keyword arguments (line 15)
    kwargs_2000 = {}
    # Getting the type of 'print' (line 15)
    print_1995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'print', False)
    # Calling print(args, kwargs) (line 15)
    print_call_result_2001 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), print_1995, *[fullapi_hash_call_result_1999], **kwargs_2000)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
