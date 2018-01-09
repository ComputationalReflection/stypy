
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from numpy import add_newdoc
2: 
3: add_newdoc('scipy.odr', 'odr',
4:     '''
5:     odr(fcn, beta0, y, x, we=None, wd=None, fjacb=None, fjacd=None, extra_args=None, ifixx=None, ifixb=None, job=0, iprint=0, errfile=None, rptfile=None, ndigit=0, taufac=0.0, sstol=-1.0, partol=-1.0, maxit=-1, stpb=None, stpd=None, sclb=None, scld=None, work=None, iwork=None, full_output=0)
6: 
7:     Low-level function for ODR.
8: 
9:     See Also
10:     --------
11:     ODR
12:     Model
13:     Data
14:     RealData
15: 
16:     Notes
17:     -----
18:     This is a function performing the same operation as the `ODR`,
19:     `Model` and `Data` classes together. The parameters of this
20:     function are explained in the class documentation.
21: 
22:     ''')
23: 
24: add_newdoc('scipy.odr.__odrpack', '_set_exceptions',
25:     '''
26:     _set_exceptions(odr_error, odr_stop)
27: 
28:     Internal function: set exception classes.
29: 
30:     ''')
31: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from numpy import add_newdoc' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
import_162950 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy')

if (type(import_162950) is not StypyTypeError):

    if (import_162950 != 'pyd_module'):
        __import__(import_162950)
        sys_modules_162951 = sys.modules[import_162950]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy', sys_modules_162951.module_type_store, module_type_store, ['add_newdoc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_162951, sys_modules_162951.module_type_store, module_type_store)
    else:
        from numpy import add_newdoc

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy', None, module_type_store, ['add_newdoc'], [add_newdoc])

else:
    # Assigning a type to the variable 'numpy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy', import_162950)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')


# Call to add_newdoc(...): (line 3)
# Processing the call arguments (line 3)
str_162953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'scipy.odr')
str_162954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 24), 'str', 'odr')
str_162955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n    odr(fcn, beta0, y, x, we=None, wd=None, fjacb=None, fjacd=None, extra_args=None, ifixx=None, ifixb=None, job=0, iprint=0, errfile=None, rptfile=None, ndigit=0, taufac=0.0, sstol=-1.0, partol=-1.0, maxit=-1, stpb=None, stpd=None, sclb=None, scld=None, work=None, iwork=None, full_output=0)\n\n    Low-level function for ODR.\n\n    See Also\n    --------\n    ODR\n    Model\n    Data\n    RealData\n\n    Notes\n    -----\n    This is a function performing the same operation as the `ODR`,\n    `Model` and `Data` classes together. The parameters of this\n    function are explained in the class documentation.\n\n    ')
# Processing the call keyword arguments (line 3)
kwargs_162956 = {}
# Getting the type of 'add_newdoc' (line 3)
add_newdoc_162952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3)
add_newdoc_call_result_162957 = invoke(stypy.reporting.localization.Localization(__file__, 3, 0), add_newdoc_162952, *[str_162953, str_162954, str_162955], **kwargs_162956)


# Call to add_newdoc(...): (line 24)
# Processing the call arguments (line 24)
str_162959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'str', 'scipy.odr.__odrpack')
str_162960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 34), 'str', '_set_exceptions')
str_162961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '\n    _set_exceptions(odr_error, odr_stop)\n\n    Internal function: set exception classes.\n\n    ')
# Processing the call keyword arguments (line 24)
kwargs_162962 = {}
# Getting the type of 'add_newdoc' (line 24)
add_newdoc_162958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 24)
add_newdoc_call_result_162963 = invoke(stypy.reporting.localization.Localization(__file__, 24, 0), add_newdoc_162958, *[str_162959, str_162960, str_162961], **kwargs_162962)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
