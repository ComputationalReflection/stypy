
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: The module testing.noseclasses is deprecated as of 2.1
3: '''
4: 
5: from __future__ import (absolute_import, division, print_function,
6:                         unicode_literals)
7: try:
8:     from ._nose.plugins.knownfailure import KnownFailure as _KnownFailure
9:     has_nose = True
10: except ImportError:
11:     has_nose = False
12:     _KnownFailure = object
13: 
14: from .. import cbook
15: 
16: cbook.warn_deprecated(
17:     since="2.1",
18:     message="The noseclass module has been deprecated in 2.1 and will "
19:             "be removed in matplotlib 2.3.")
20: 
21: 
22: @cbook.deprecated("2.1")
23: class KnownFailure(_KnownFailure):
24:     def __init__(self):
25:         if not has_nose:
26:             raise ImportError("Need nose for this plugin.")
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_291954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nThe module testing.noseclasses is deprecated as of 2.1\n')


# SSA begins for try-except statement (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))

# 'from matplotlib.testing._nose.plugins.knownfailure import _KnownFailure' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_291955 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib.testing._nose.plugins.knownfailure')

if (type(import_291955) is not StypyTypeError):

    if (import_291955 != 'pyd_module'):
        __import__(import_291955)
        sys_modules_291956 = sys.modules[import_291955]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib.testing._nose.plugins.knownfailure', sys_modules_291956.module_type_store, module_type_store, ['KnownFailure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_291956, sys_modules_291956.module_type_store, module_type_store)
    else:
        from matplotlib.testing._nose.plugins.knownfailure import KnownFailure as _KnownFailure

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib.testing._nose.plugins.knownfailure', None, module_type_store, ['KnownFailure'], [_KnownFailure])

else:
    # Assigning a type to the variable 'matplotlib.testing._nose.plugins.knownfailure' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib.testing._nose.plugins.knownfailure', import_291955)

# Adding an alias
module_type_store.add_alias('_KnownFailure', 'KnownFailure')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')


# Assigning a Name to a Name (line 9):
# Getting the type of 'True' (line 9)
True_291957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'True')
# Assigning a type to the variable 'has_nose' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'has_nose', True_291957)
# SSA branch for the except part of a try statement (line 7)
# SSA branch for the except 'ImportError' branch of a try statement (line 7)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 11):
# Getting the type of 'False' (line 11)
False_291958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'False')
# Assigning a type to the variable 'has_nose' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'has_nose', False_291958)

# Assigning a Name to a Name (line 12):
# Getting the type of 'object' (line 12)
object_291959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'object')
# Assigning a type to the variable '_KnownFailure' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), '_KnownFailure', object_291959)
# SSA join for try-except statement (line 7)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib import cbook' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_291960 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib')

if (type(import_291960) is not StypyTypeError):

    if (import_291960 != 'pyd_module'):
        __import__(import_291960)
        sys_modules_291961 = sys.modules[import_291960]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', sys_modules_291961.module_type_store, module_type_store, ['cbook'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_291961, sys_modules_291961.module_type_store, module_type_store)
    else:
        from matplotlib import cbook

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', None, module_type_store, ['cbook'], [cbook])

else:
    # Assigning a type to the variable 'matplotlib' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', import_291960)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')


# Call to warn_deprecated(...): (line 16)
# Processing the call keyword arguments (line 16)
unicode_291964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'unicode', u'2.1')
keyword_291965 = unicode_291964
unicode_291966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 12), 'unicode', u'The noseclass module has been deprecated in 2.1 and will be removed in matplotlib 2.3.')
keyword_291967 = unicode_291966
kwargs_291968 = {'message': keyword_291967, 'since': keyword_291965}
# Getting the type of 'cbook' (line 16)
cbook_291962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'cbook', False)
# Obtaining the member 'warn_deprecated' of a type (line 16)
warn_deprecated_291963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 0), cbook_291962, 'warn_deprecated')
# Calling warn_deprecated(args, kwargs) (line 16)
warn_deprecated_call_result_291969 = invoke(stypy.reporting.localization.Localization(__file__, 16, 0), warn_deprecated_291963, *[], **kwargs_291968)

# Declaration of the 'KnownFailure' class
# Getting the type of '_KnownFailure' (line 23)
_KnownFailure_291970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), '_KnownFailure')

class KnownFailure(_KnownFailure_291970, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
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

        
        
        # Getting the type of 'has_nose' (line 25)
        has_nose_291971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'has_nose')
        # Applying the 'not' unary operator (line 25)
        result_not__291972 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 11), 'not', has_nose_291971)
        
        # Testing the type of an if condition (line 25)
        if_condition_291973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 8), result_not__291972)
        # Assigning a type to the variable 'if_condition_291973' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'if_condition_291973', if_condition_291973)
        # SSA begins for if statement (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ImportError(...): (line 26)
        # Processing the call arguments (line 26)
        unicode_291975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'unicode', u'Need nose for this plugin.')
        # Processing the call keyword arguments (line 26)
        kwargs_291976 = {}
        # Getting the type of 'ImportError' (line 26)
        ImportError_291974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'ImportError', False)
        # Calling ImportError(args, kwargs) (line 26)
        ImportError_call_result_291977 = invoke(stypy.reporting.localization.Localization(__file__, 26, 18), ImportError_291974, *[unicode_291975], **kwargs_291976)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 26, 12), ImportError_call_result_291977, 'raise parameter', BaseException)
        # SSA join for if statement (line 25)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'KnownFailure' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'KnownFailure', KnownFailure)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
