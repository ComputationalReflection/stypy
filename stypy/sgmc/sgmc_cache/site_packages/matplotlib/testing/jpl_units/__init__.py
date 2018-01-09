
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #=======================================================================
2: 
3: '''
4: This is a sample set of units for use with testing unit conversion
5: of matplotlib routines.  These are used because they use very strict
6: enforcement of unitized data which will test the entire spectrum of how
7: unitized data might be used (it is not always meaningful to convert to
8: a float without specific units given).
9: 
10: UnitDbl is essentially a unitized floating point number.  It has a
11: minimal set of supported units (enough for testing purposes).  All
12: of the mathematical operation are provided to fully test any behaviour
13: that might occur with unitized data.  Remeber that unitized data has
14: rules as to how it can be applied to one another (a value of distance
15: cannot be added to a value of time).  Thus we need to guard against any
16: accidental "default" conversion that will strip away the meaning of the
17: data and render it neutered.
18: 
19: Epoch is different than a UnitDbl of time.  Time is something that can be
20: measured where an Epoch is a specific moment in time.  Epochs are typically
21: referenced as an offset from some predetermined epoch.
22: 
23: A difference of two epochs is a Duration.  The distinction between a
24: Duration and a UnitDbl of time is made because an Epoch can have different
25: frames (or units).  In the case of our test Epoch class the two allowed
26: frames are 'UTC' and 'ET' (Note that these are rough estimates provided for
27: testing purposes and should not be used in production code where accuracy
28: of time frames is desired).  As such a Duration also has a frame of
29: reference and therefore needs to be called out as different that a simple
30: measurement of time since a delta-t in one frame may not be the same in another.
31: '''
32: 
33: #=======================================================================
34: from __future__ import (absolute_import, division, print_function,
35:                         unicode_literals)
36: 
37: import six
38: 
39: from .Duration import Duration
40: from .Epoch import Epoch
41: from .UnitDbl import UnitDbl
42: 
43: from .StrConverter import StrConverter
44: from .EpochConverter import EpochConverter
45: from .UnitDblConverter import UnitDblConverter
46: 
47: from .UnitDblFormatter import UnitDblFormatter
48: 
49: #=======================================================================
50: 
51: __version__ = "1.0"
52: 
53: __all__ = [
54:             'register',
55:             'Duration',
56:             'Epoch',
57:             'UnitDbl',
58:             'UnitDblFormatter',
59:           ]
60: 
61: #=======================================================================
62: def register():
63:    '''Register the unit conversion classes with matplotlib.'''
64:    import matplotlib.units as mplU
65: 
66:    mplU.registry[ str ] = StrConverter()
67:    mplU.registry[ Epoch ] = EpochConverter()
68:    mplU.registry[ UnitDbl ] = UnitDblConverter()
69: 
70: #=======================================================================
71: # Some default unit instances
72: 
73: # Distances
74: m = UnitDbl( 1.0, "m" )
75: km = UnitDbl( 1.0, "km" )
76: mile = UnitDbl( 1.0, "mile" )
77: 
78: # Angles
79: deg = UnitDbl( 1.0, "deg" )
80: rad = UnitDbl( 1.0, "rad" )
81: 
82: # Time
83: sec = UnitDbl( 1.0, "sec" )
84: min = UnitDbl( 1.0, "min" )
85: hr = UnitDbl( 1.0, "hour" )
86: day = UnitDbl( 24.0, "hour" )
87: sec = UnitDbl( 1.0, "sec" )
88: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_293972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'unicode', u'\nThis is a sample set of units for use with testing unit conversion\nof matplotlib routines.  These are used because they use very strict\nenforcement of unitized data which will test the entire spectrum of how\nunitized data might be used (it is not always meaningful to convert to\na float without specific units given).\n\nUnitDbl is essentially a unitized floating point number.  It has a\nminimal set of supported units (enough for testing purposes).  All\nof the mathematical operation are provided to fully test any behaviour\nthat might occur with unitized data.  Remeber that unitized data has\nrules as to how it can be applied to one another (a value of distance\ncannot be added to a value of time).  Thus we need to guard against any\naccidental "default" conversion that will strip away the meaning of the\ndata and render it neutered.\n\nEpoch is different than a UnitDbl of time.  Time is something that can be\nmeasured where an Epoch is a specific moment in time.  Epochs are typically\nreferenced as an offset from some predetermined epoch.\n\nA difference of two epochs is a Duration.  The distinction between a\nDuration and a UnitDbl of time is made because an Epoch can have different\nframes (or units).  In the case of our test Epoch class the two allowed\nframes are \'UTC\' and \'ET\' (Note that these are rough estimates provided for\ntesting purposes and should not be used in production code where accuracy\nof time frames is desired).  As such a Duration also has a frame of\nreference and therefore needs to be called out as different that a simple\nmeasurement of time since a delta-t in one frame may not be the same in another.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'import six' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293973 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'six')

if (type(import_293973) is not StypyTypeError):

    if (import_293973 != 'pyd_module'):
        __import__(import_293973)
        sys_modules_293974 = sys.modules[import_293973]
        import_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'six', sys_modules_293974.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'six', import_293973)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'from matplotlib.testing.jpl_units.Duration import Duration' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293975 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.testing.jpl_units.Duration')

if (type(import_293975) is not StypyTypeError):

    if (import_293975 != 'pyd_module'):
        __import__(import_293975)
        sys_modules_293976 = sys.modules[import_293975]
        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.testing.jpl_units.Duration', sys_modules_293976.module_type_store, module_type_store, ['Duration'])
        nest_module(stypy.reporting.localization.Localization(__file__, 39, 0), __file__, sys_modules_293976, sys_modules_293976.module_type_store, module_type_store)
    else:
        from matplotlib.testing.jpl_units.Duration import Duration

        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.testing.jpl_units.Duration', None, module_type_store, ['Duration'], [Duration])

else:
    # Assigning a type to the variable 'matplotlib.testing.jpl_units.Duration' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.testing.jpl_units.Duration', import_293975)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'from matplotlib.testing.jpl_units.Epoch import Epoch' statement (line 40)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293977 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.testing.jpl_units.Epoch')

if (type(import_293977) is not StypyTypeError):

    if (import_293977 != 'pyd_module'):
        __import__(import_293977)
        sys_modules_293978 = sys.modules[import_293977]
        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.testing.jpl_units.Epoch', sys_modules_293978.module_type_store, module_type_store, ['Epoch'])
        nest_module(stypy.reporting.localization.Localization(__file__, 40, 0), __file__, sys_modules_293978, sys_modules_293978.module_type_store, module_type_store)
    else:
        from matplotlib.testing.jpl_units.Epoch import Epoch

        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.testing.jpl_units.Epoch', None, module_type_store, ['Epoch'], [Epoch])

else:
    # Assigning a type to the variable 'matplotlib.testing.jpl_units.Epoch' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.testing.jpl_units.Epoch', import_293977)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'from matplotlib.testing.jpl_units.UnitDbl import UnitDbl' statement (line 41)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293979 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.testing.jpl_units.UnitDbl')

if (type(import_293979) is not StypyTypeError):

    if (import_293979 != 'pyd_module'):
        __import__(import_293979)
        sys_modules_293980 = sys.modules[import_293979]
        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.testing.jpl_units.UnitDbl', sys_modules_293980.module_type_store, module_type_store, ['UnitDbl'])
        nest_module(stypy.reporting.localization.Localization(__file__, 41, 0), __file__, sys_modules_293980, sys_modules_293980.module_type_store, module_type_store)
    else:
        from matplotlib.testing.jpl_units.UnitDbl import UnitDbl

        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.testing.jpl_units.UnitDbl', None, module_type_store, ['UnitDbl'], [UnitDbl])

else:
    # Assigning a type to the variable 'matplotlib.testing.jpl_units.UnitDbl' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.testing.jpl_units.UnitDbl', import_293979)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 43, 0))

# 'from matplotlib.testing.jpl_units.StrConverter import StrConverter' statement (line 43)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293981 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'matplotlib.testing.jpl_units.StrConverter')

if (type(import_293981) is not StypyTypeError):

    if (import_293981 != 'pyd_module'):
        __import__(import_293981)
        sys_modules_293982 = sys.modules[import_293981]
        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'matplotlib.testing.jpl_units.StrConverter', sys_modules_293982.module_type_store, module_type_store, ['StrConverter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 43, 0), __file__, sys_modules_293982, sys_modules_293982.module_type_store, module_type_store)
    else:
        from matplotlib.testing.jpl_units.StrConverter import StrConverter

        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'matplotlib.testing.jpl_units.StrConverter', None, module_type_store, ['StrConverter'], [StrConverter])

else:
    # Assigning a type to the variable 'matplotlib.testing.jpl_units.StrConverter' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'matplotlib.testing.jpl_units.StrConverter', import_293981)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'from matplotlib.testing.jpl_units.EpochConverter import EpochConverter' statement (line 44)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293983 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.testing.jpl_units.EpochConverter')

if (type(import_293983) is not StypyTypeError):

    if (import_293983 != 'pyd_module'):
        __import__(import_293983)
        sys_modules_293984 = sys.modules[import_293983]
        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.testing.jpl_units.EpochConverter', sys_modules_293984.module_type_store, module_type_store, ['EpochConverter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 44, 0), __file__, sys_modules_293984, sys_modules_293984.module_type_store, module_type_store)
    else:
        from matplotlib.testing.jpl_units.EpochConverter import EpochConverter

        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.testing.jpl_units.EpochConverter', None, module_type_store, ['EpochConverter'], [EpochConverter])

else:
    # Assigning a type to the variable 'matplotlib.testing.jpl_units.EpochConverter' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.testing.jpl_units.EpochConverter', import_293983)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 0))

# 'from matplotlib.testing.jpl_units.UnitDblConverter import UnitDblConverter' statement (line 45)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293985 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib.testing.jpl_units.UnitDblConverter')

if (type(import_293985) is not StypyTypeError):

    if (import_293985 != 'pyd_module'):
        __import__(import_293985)
        sys_modules_293986 = sys.modules[import_293985]
        import_from_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib.testing.jpl_units.UnitDblConverter', sys_modules_293986.module_type_store, module_type_store, ['UnitDblConverter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 45, 0), __file__, sys_modules_293986, sys_modules_293986.module_type_store, module_type_store)
    else:
        from matplotlib.testing.jpl_units.UnitDblConverter import UnitDblConverter

        import_from_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib.testing.jpl_units.UnitDblConverter', None, module_type_store, ['UnitDblConverter'], [UnitDblConverter])

else:
    # Assigning a type to the variable 'matplotlib.testing.jpl_units.UnitDblConverter' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib.testing.jpl_units.UnitDblConverter', import_293985)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 47, 0))

# 'from matplotlib.testing.jpl_units.UnitDblFormatter import UnitDblFormatter' statement (line 47)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293987 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'matplotlib.testing.jpl_units.UnitDblFormatter')

if (type(import_293987) is not StypyTypeError):

    if (import_293987 != 'pyd_module'):
        __import__(import_293987)
        sys_modules_293988 = sys.modules[import_293987]
        import_from_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'matplotlib.testing.jpl_units.UnitDblFormatter', sys_modules_293988.module_type_store, module_type_store, ['UnitDblFormatter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 47, 0), __file__, sys_modules_293988, sys_modules_293988.module_type_store, module_type_store)
    else:
        from matplotlib.testing.jpl_units.UnitDblFormatter import UnitDblFormatter

        import_from_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'matplotlib.testing.jpl_units.UnitDblFormatter', None, module_type_store, ['UnitDblFormatter'], [UnitDblFormatter])

else:
    # Assigning a type to the variable 'matplotlib.testing.jpl_units.UnitDblFormatter' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'matplotlib.testing.jpl_units.UnitDblFormatter', import_293987)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')


# Assigning a Str to a Name (line 51):
unicode_293989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 14), 'unicode', u'1.0')
# Assigning a type to the variable '__version__' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '__version__', unicode_293989)

# Assigning a List to a Name (line 53):
__all__ = [u'register', u'Duration', u'Epoch', u'UnitDbl', u'UnitDblFormatter']
module_type_store.set_exportable_members([u'register', u'Duration', u'Epoch', u'UnitDbl', u'UnitDblFormatter'])

# Obtaining an instance of the builtin type 'list' (line 53)
list_293990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 53)
# Adding element type (line 53)
unicode_293991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'unicode', u'register')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 10), list_293990, unicode_293991)
# Adding element type (line 53)
unicode_293992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 12), 'unicode', u'Duration')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 10), list_293990, unicode_293992)
# Adding element type (line 53)
unicode_293993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'unicode', u'Epoch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 10), list_293990, unicode_293993)
# Adding element type (line 53)
unicode_293994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'unicode', u'UnitDbl')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 10), list_293990, unicode_293994)
# Adding element type (line 53)
unicode_293995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 12), 'unicode', u'UnitDblFormatter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 10), list_293990, unicode_293995)

# Assigning a type to the variable '__all__' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), '__all__', list_293990)

@norecursion
def register(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'register'
    module_type_store = module_type_store.open_function_context('register', 62, 0, False)
    
    # Passed parameters checking function
    register.stypy_localization = localization
    register.stypy_type_of_self = None
    register.stypy_type_store = module_type_store
    register.stypy_function_name = 'register'
    register.stypy_param_names_list = []
    register.stypy_varargs_param_name = None
    register.stypy_kwargs_param_name = None
    register.stypy_call_defaults = defaults
    register.stypy_call_varargs = varargs
    register.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'register', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'register', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'register(...)' code ##################

    unicode_293996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 3), 'unicode', u'Register the unit conversion classes with matplotlib.')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 64, 3))
    
    # 'import matplotlib.units' statement (line 64)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
    import_293997 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 64, 3), 'matplotlib.units')

    if (type(import_293997) is not StypyTypeError):

        if (import_293997 != 'pyd_module'):
            __import__(import_293997)
            sys_modules_293998 = sys.modules[import_293997]
            import_module(stypy.reporting.localization.Localization(__file__, 64, 3), 'mplU', sys_modules_293998.module_type_store, module_type_store)
        else:
            import matplotlib.units as mplU

            import_module(stypy.reporting.localization.Localization(__file__, 64, 3), 'mplU', matplotlib.units, module_type_store)

    else:
        # Assigning a type to the variable 'matplotlib.units' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 3), 'matplotlib.units', import_293997)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
    
    
    # Assigning a Call to a Subscript (line 66):
    
    # Call to StrConverter(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_294000 = {}
    # Getting the type of 'StrConverter' (line 66)
    StrConverter_293999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 26), 'StrConverter', False)
    # Calling StrConverter(args, kwargs) (line 66)
    StrConverter_call_result_294001 = invoke(stypy.reporting.localization.Localization(__file__, 66, 26), StrConverter_293999, *[], **kwargs_294000)
    
    # Getting the type of 'mplU' (line 66)
    mplU_294002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 3), 'mplU')
    # Obtaining the member 'registry' of a type (line 66)
    registry_294003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 3), mplU_294002, 'registry')
    # Getting the type of 'str' (line 66)
    str_294004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), 'str')
    # Storing an element on a container (line 66)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 3), registry_294003, (str_294004, StrConverter_call_result_294001))
    
    # Assigning a Call to a Subscript (line 67):
    
    # Call to EpochConverter(...): (line 67)
    # Processing the call keyword arguments (line 67)
    kwargs_294006 = {}
    # Getting the type of 'EpochConverter' (line 67)
    EpochConverter_294005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 28), 'EpochConverter', False)
    # Calling EpochConverter(args, kwargs) (line 67)
    EpochConverter_call_result_294007 = invoke(stypy.reporting.localization.Localization(__file__, 67, 28), EpochConverter_294005, *[], **kwargs_294006)
    
    # Getting the type of 'mplU' (line 67)
    mplU_294008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 3), 'mplU')
    # Obtaining the member 'registry' of a type (line 67)
    registry_294009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 3), mplU_294008, 'registry')
    # Getting the type of 'Epoch' (line 67)
    Epoch_294010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'Epoch')
    # Storing an element on a container (line 67)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 3), registry_294009, (Epoch_294010, EpochConverter_call_result_294007))
    
    # Assigning a Call to a Subscript (line 68):
    
    # Call to UnitDblConverter(...): (line 68)
    # Processing the call keyword arguments (line 68)
    kwargs_294012 = {}
    # Getting the type of 'UnitDblConverter' (line 68)
    UnitDblConverter_294011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'UnitDblConverter', False)
    # Calling UnitDblConverter(args, kwargs) (line 68)
    UnitDblConverter_call_result_294013 = invoke(stypy.reporting.localization.Localization(__file__, 68, 30), UnitDblConverter_294011, *[], **kwargs_294012)
    
    # Getting the type of 'mplU' (line 68)
    mplU_294014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 3), 'mplU')
    # Obtaining the member 'registry' of a type (line 68)
    registry_294015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 3), mplU_294014, 'registry')
    # Getting the type of 'UnitDbl' (line 68)
    UnitDbl_294016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), 'UnitDbl')
    # Storing an element on a container (line 68)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 3), registry_294015, (UnitDbl_294016, UnitDblConverter_call_result_294013))
    
    # ################# End of 'register(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'register' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_294017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_294017)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'register'
    return stypy_return_type_294017

# Assigning a type to the variable 'register' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'register', register)

# Assigning a Call to a Name (line 74):

# Call to UnitDbl(...): (line 74)
# Processing the call arguments (line 74)
float_294019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 13), 'float')
unicode_294020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 18), 'unicode', u'm')
# Processing the call keyword arguments (line 74)
kwargs_294021 = {}
# Getting the type of 'UnitDbl' (line 74)
UnitDbl_294018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'UnitDbl', False)
# Calling UnitDbl(args, kwargs) (line 74)
UnitDbl_call_result_294022 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), UnitDbl_294018, *[float_294019, unicode_294020], **kwargs_294021)

# Assigning a type to the variable 'm' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'm', UnitDbl_call_result_294022)

# Assigning a Call to a Name (line 75):

# Call to UnitDbl(...): (line 75)
# Processing the call arguments (line 75)
float_294024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 14), 'float')
unicode_294025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'unicode', u'km')
# Processing the call keyword arguments (line 75)
kwargs_294026 = {}
# Getting the type of 'UnitDbl' (line 75)
UnitDbl_294023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 5), 'UnitDbl', False)
# Calling UnitDbl(args, kwargs) (line 75)
UnitDbl_call_result_294027 = invoke(stypy.reporting.localization.Localization(__file__, 75, 5), UnitDbl_294023, *[float_294024, unicode_294025], **kwargs_294026)

# Assigning a type to the variable 'km' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'km', UnitDbl_call_result_294027)

# Assigning a Call to a Name (line 76):

# Call to UnitDbl(...): (line 76)
# Processing the call arguments (line 76)
float_294029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 16), 'float')
unicode_294030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 21), 'unicode', u'mile')
# Processing the call keyword arguments (line 76)
kwargs_294031 = {}
# Getting the type of 'UnitDbl' (line 76)
UnitDbl_294028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'UnitDbl', False)
# Calling UnitDbl(args, kwargs) (line 76)
UnitDbl_call_result_294032 = invoke(stypy.reporting.localization.Localization(__file__, 76, 7), UnitDbl_294028, *[float_294029, unicode_294030], **kwargs_294031)

# Assigning a type to the variable 'mile' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'mile', UnitDbl_call_result_294032)

# Assigning a Call to a Name (line 79):

# Call to UnitDbl(...): (line 79)
# Processing the call arguments (line 79)
float_294034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 15), 'float')
unicode_294035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'unicode', u'deg')
# Processing the call keyword arguments (line 79)
kwargs_294036 = {}
# Getting the type of 'UnitDbl' (line 79)
UnitDbl_294033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 6), 'UnitDbl', False)
# Calling UnitDbl(args, kwargs) (line 79)
UnitDbl_call_result_294037 = invoke(stypy.reporting.localization.Localization(__file__, 79, 6), UnitDbl_294033, *[float_294034, unicode_294035], **kwargs_294036)

# Assigning a type to the variable 'deg' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'deg', UnitDbl_call_result_294037)

# Assigning a Call to a Name (line 80):

# Call to UnitDbl(...): (line 80)
# Processing the call arguments (line 80)
float_294039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 15), 'float')
unicode_294040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'unicode', u'rad')
# Processing the call keyword arguments (line 80)
kwargs_294041 = {}
# Getting the type of 'UnitDbl' (line 80)
UnitDbl_294038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 6), 'UnitDbl', False)
# Calling UnitDbl(args, kwargs) (line 80)
UnitDbl_call_result_294042 = invoke(stypy.reporting.localization.Localization(__file__, 80, 6), UnitDbl_294038, *[float_294039, unicode_294040], **kwargs_294041)

# Assigning a type to the variable 'rad' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'rad', UnitDbl_call_result_294042)

# Assigning a Call to a Name (line 83):

# Call to UnitDbl(...): (line 83)
# Processing the call arguments (line 83)
float_294044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'float')
unicode_294045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 20), 'unicode', u'sec')
# Processing the call keyword arguments (line 83)
kwargs_294046 = {}
# Getting the type of 'UnitDbl' (line 83)
UnitDbl_294043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 6), 'UnitDbl', False)
# Calling UnitDbl(args, kwargs) (line 83)
UnitDbl_call_result_294047 = invoke(stypy.reporting.localization.Localization(__file__, 83, 6), UnitDbl_294043, *[float_294044, unicode_294045], **kwargs_294046)

# Assigning a type to the variable 'sec' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'sec', UnitDbl_call_result_294047)

# Assigning a Call to a Name (line 84):

# Call to UnitDbl(...): (line 84)
# Processing the call arguments (line 84)
float_294049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 15), 'float')
unicode_294050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'unicode', u'min')
# Processing the call keyword arguments (line 84)
kwargs_294051 = {}
# Getting the type of 'UnitDbl' (line 84)
UnitDbl_294048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 6), 'UnitDbl', False)
# Calling UnitDbl(args, kwargs) (line 84)
UnitDbl_call_result_294052 = invoke(stypy.reporting.localization.Localization(__file__, 84, 6), UnitDbl_294048, *[float_294049, unicode_294050], **kwargs_294051)

# Assigning a type to the variable 'min' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'min', UnitDbl_call_result_294052)

# Assigning a Call to a Name (line 85):

# Call to UnitDbl(...): (line 85)
# Processing the call arguments (line 85)
float_294054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 14), 'float')
unicode_294055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 19), 'unicode', u'hour')
# Processing the call keyword arguments (line 85)
kwargs_294056 = {}
# Getting the type of 'UnitDbl' (line 85)
UnitDbl_294053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 5), 'UnitDbl', False)
# Calling UnitDbl(args, kwargs) (line 85)
UnitDbl_call_result_294057 = invoke(stypy.reporting.localization.Localization(__file__, 85, 5), UnitDbl_294053, *[float_294054, unicode_294055], **kwargs_294056)

# Assigning a type to the variable 'hr' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'hr', UnitDbl_call_result_294057)

# Assigning a Call to a Name (line 86):

# Call to UnitDbl(...): (line 86)
# Processing the call arguments (line 86)
float_294059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 15), 'float')
unicode_294060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'unicode', u'hour')
# Processing the call keyword arguments (line 86)
kwargs_294061 = {}
# Getting the type of 'UnitDbl' (line 86)
UnitDbl_294058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 6), 'UnitDbl', False)
# Calling UnitDbl(args, kwargs) (line 86)
UnitDbl_call_result_294062 = invoke(stypy.reporting.localization.Localization(__file__, 86, 6), UnitDbl_294058, *[float_294059, unicode_294060], **kwargs_294061)

# Assigning a type to the variable 'day' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'day', UnitDbl_call_result_294062)

# Assigning a Call to a Name (line 87):

# Call to UnitDbl(...): (line 87)
# Processing the call arguments (line 87)
float_294064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'float')
unicode_294065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'unicode', u'sec')
# Processing the call keyword arguments (line 87)
kwargs_294066 = {}
# Getting the type of 'UnitDbl' (line 87)
UnitDbl_294063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 6), 'UnitDbl', False)
# Calling UnitDbl(args, kwargs) (line 87)
UnitDbl_call_result_294067 = invoke(stypy.reporting.localization.Localization(__file__, 87, 6), UnitDbl_294063, *[float_294064, unicode_294065], **kwargs_294066)

# Assigning a type to the variable 'sec' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'sec', UnitDbl_call_result_294067)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
