
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: 
6: def generate():
7:     for x in xrange(10):
8:         yield x
9: 
10: 
11: Z = np.fromiter(generate(), dtype=float, count=-1)
12: 
13: # l = globals().copy()
14: # for v in l:
15: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1162 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1162) is not StypyTypeError):

    if (import_1162 != 'pyd_module'):
        __import__(import_1162)
        sys_modules_1163 = sys.modules[import_1162]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1163.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1162)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


@norecursion
def generate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate'
    module_type_store = module_type_store.open_function_context('generate', 6, 0, False)
    
    # Passed parameters checking function
    generate.stypy_localization = localization
    generate.stypy_type_of_self = None
    generate.stypy_type_store = module_type_store
    generate.stypy_function_name = 'generate'
    generate.stypy_param_names_list = []
    generate.stypy_varargs_param_name = None
    generate.stypy_kwargs_param_name = None
    generate.stypy_call_defaults = defaults
    generate.stypy_call_varargs = varargs
    generate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate(...)' code ##################

    
    
    # Call to xrange(...): (line 7)
    # Processing the call arguments (line 7)
    int_1165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'int')
    # Processing the call keyword arguments (line 7)
    kwargs_1166 = {}
    # Getting the type of 'xrange' (line 7)
    xrange_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 7)
    xrange_call_result_1167 = invoke(stypy.reporting.localization.Localization(__file__, 7, 13), xrange_1164, *[int_1165], **kwargs_1166)
    
    # Testing the type of a for loop iterable (line 7)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 7, 4), xrange_call_result_1167)
    # Getting the type of the for loop variable (line 7)
    for_loop_var_1168 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 7, 4), xrange_call_result_1167)
    # Assigning a type to the variable 'x' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'x', for_loop_var_1168)
    # SSA begins for a for statement (line 7)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Creating a generator
    # Getting the type of 'x' (line 8)
    x_1169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'x')
    GeneratorType_1170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 8), GeneratorType_1170, x_1169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', GeneratorType_1170)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'generate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_1171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1171)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate'
    return stypy_return_type_1171

# Assigning a type to the variable 'generate' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'generate', generate)

# Assigning a Call to a Name (line 11):

# Call to fromiter(...): (line 11)
# Processing the call arguments (line 11)

# Call to generate(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_1175 = {}
# Getting the type of 'generate' (line 11)
generate_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'generate', False)
# Calling generate(args, kwargs) (line 11)
generate_call_result_1176 = invoke(stypy.reporting.localization.Localization(__file__, 11, 16), generate_1174, *[], **kwargs_1175)

# Processing the call keyword arguments (line 11)
# Getting the type of 'float' (line 11)
float_1177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 34), 'float', False)
keyword_1178 = float_1177
int_1179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 47), 'int')
keyword_1180 = int_1179
kwargs_1181 = {'count': keyword_1180, 'dtype': keyword_1178}
# Getting the type of 'np' (line 11)
np_1172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
# Obtaining the member 'fromiter' of a type (line 11)
fromiter_1173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_1172, 'fromiter')
# Calling fromiter(args, kwargs) (line 11)
fromiter_call_result_1182 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), fromiter_1173, *[generate_call_result_1176], **kwargs_1181)

# Assigning a type to the variable 'Z' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'Z', fromiter_call_result_1182)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
