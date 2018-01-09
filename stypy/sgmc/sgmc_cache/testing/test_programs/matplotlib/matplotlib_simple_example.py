
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: from matplotlib import pyplot
4: 
5: pyplot.plot([1,2,3,4])
6: pyplot.ylabel('some numbers')
7: pyplot.show()
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from matplotlib import pyplot' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/matplotlib/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'matplotlib')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'matplotlib', sys_modules_2.module_type_store, module_type_store, ['pyplot'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_2, sys_modules_2.module_type_store, module_type_store)
    else:
        from matplotlib import pyplot

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'matplotlib', None, module_type_store, ['pyplot'], [pyplot])

else:
    # Assigning a type to the variable 'matplotlib' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'matplotlib', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/matplotlib/')


# Call to plot(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 12), list_5, int_6)
# Adding element type (line 5)
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 12), list_5, int_7)
# Adding element type (line 5)
int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 12), list_5, int_8)
# Adding element type (line 5)
int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 12), list_5, int_9)

# Processing the call keyword arguments (line 5)
kwargs_10 = {}
# Getting the type of 'pyplot' (line 5)
pyplot_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pyplot', False)
# Obtaining the member 'plot' of a type (line 5)
plot_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 0), pyplot_3, 'plot')
# Calling plot(args, kwargs) (line 5)
plot_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 5, 0), plot_4, *[list_5], **kwargs_10)


# Call to ylabel(...): (line 6)
# Processing the call arguments (line 6)
str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'str', 'some numbers')
# Processing the call keyword arguments (line 6)
kwargs_15 = {}
# Getting the type of 'pyplot' (line 6)
pyplot_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'pyplot', False)
# Obtaining the member 'ylabel' of a type (line 6)
ylabel_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 0), pyplot_12, 'ylabel')
# Calling ylabel(args, kwargs) (line 6)
ylabel_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 6, 0), ylabel_13, *[str_14], **kwargs_15)


# Call to show(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_19 = {}
# Getting the type of 'pyplot' (line 7)
pyplot_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pyplot', False)
# Obtaining the member 'show' of a type (line 7)
show_18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), pyplot_17, 'show')
# Calling show(args, kwargs) (line 7)
show_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), show_18, *[], **kwargs_19)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
