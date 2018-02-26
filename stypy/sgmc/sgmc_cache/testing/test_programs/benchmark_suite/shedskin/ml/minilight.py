
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #  MiniLight Python : minimal global illumination renderer
2: #
3: #  Copyright (c) 2007-2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
4: #  http://www.hxa7241.org/
5: 
6: 
7: from camera import Camera
8: from image import Image
9: from scene import Scene
10: 
11: from math import log10
12: from sys import argv, stdout
13: from time import time
14: 
15: BANNER = '''
16:   MiniLight 1.5.2 Python
17:   Copyright (c) 2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
18:   http://www.hxa7241.org/minilight/
19: '''
20: HELP = '''
21: ----------------------------------------------------------------------
22:   MiniLight 1.5.2 Python
23: 
24:   Copyright (c) 2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
25:   http://www.hxa7241.org/minilight/
26: 
27:   2008-02-17
28: ----------------------------------------------------------------------
29: 
30: MiniLight is a minimal global illumination renderer.
31: 
32: usage:
33:   minilight image_file_pathname
34: 
35: The model text file format is:
36:   #MiniLight
37: 
38:   iterations
39: 
40:   imagewidth imageheight
41: 
42:   viewposition viewdirection viewangle
43: 
44:   skyemission groundreflection
45:   vertex0 vertex1 vertex2 reflectivity emitivity
46:   vertex0 vertex1 vertex2 reflectivity emitivity
47:   ...
48: 
49: -- where iterations and image values are ints, viewangle is a float,
50: and all other values are three parenthised floats. The file must end
51: with a newline. Eg.:
52:   #MiniLight
53: 
54:   100
55: 
56:   200 150
57: 
58:   (0 0.75 -2) (0 0 1) 45
59: 
60:   (3626 5572 5802) (0.1 0.09 0.07)
61:   (0 0 0) (0 1 0) (1 1 0)  (0.7 0.7 0.7) (0 0 0)
62: '''
63: MODEL_FORMAT_ID = '#MiniLight'
64: SAVE_PERIOD = 180
65: 
66: def save_image(image_file_pathname, image, frame_no):
67:     image_file = open(image_file_pathname, 'wb')
68:     image.get_formatted(image_file, frame_no - 1)
69:     image_file.close()
70: 
71: def main(arg):
72: ##    print BANNER
73:     model_file_pathname = arg
74:     image_file_pathname = model_file_pathname + '.ppm'
75:     model_file = open(model_file_pathname, 'r')
76:     if model_file.next().strip() != MODEL_FORMAT_ID:
77:         raise 'invalid model file'
78:     for line in model_file:
79:         if not line.isspace():
80:             iterations = int(line)
81:             break
82:     image = Image(model_file)
83:     camera = Camera(model_file)
84:     scene = Scene(model_file, camera.view_position)
85:     model_file.close()
86:     last_time = time() - (SAVE_PERIOD + 1)
87:     try:
88:         for frame_no in range(1, iterations + 1):
89:             camera.get_frame(scene, image)
90:             if SAVE_PERIOD < time() - last_time or frame_no == iterations:
91:                 last_time = time()
92:                 save_image(image_file_pathname, image, frame_no)
93: ##            stdout.write('\b' * ((int(log10(frame_no - 1)) if frame_no > 1 else -1) + 12) + 'iteration: %u' % frame_no)
94: ##            stdout.flush()
95: ##        print '\nfinished'
96:     except KeyboardInterrupt:
97:         save_image(image_file_pathname, image, frame_no)
98: ##        print '\ninterupted'
99: 
100: if __name__ == '__main__':
101:     if len(argv) < 2 or argv[1] == '-?' or argv[1] == '--help':
102:         pass#print HELP
103:     else:
104:         main(argv[1])
105: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from camera import Camera' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_570 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'camera')

if (type(import_570) is not StypyTypeError):

    if (import_570 != 'pyd_module'):
        __import__(import_570)
        sys_modules_571 = sys.modules[import_570]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'camera', sys_modules_571.module_type_store, module_type_store, ['Camera'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_571, sys_modules_571.module_type_store, module_type_store)
    else:
        from camera import Camera

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'camera', None, module_type_store, ['Camera'], [Camera])

else:
    # Assigning a type to the variable 'camera' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'camera', import_570)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from image import Image' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_572 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'image')

if (type(import_572) is not StypyTypeError):

    if (import_572 != 'pyd_module'):
        __import__(import_572)
        sys_modules_573 = sys.modules[import_572]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'image', sys_modules_573.module_type_store, module_type_store, ['Image'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_573, sys_modules_573.module_type_store, module_type_store)
    else:
        from image import Image

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'image', None, module_type_store, ['Image'], [Image])

else:
    # Assigning a type to the variable 'image' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'image', import_572)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scene import Scene' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_574 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scene')

if (type(import_574) is not StypyTypeError):

    if (import_574 != 'pyd_module'):
        __import__(import_574)
        sys_modules_575 = sys.modules[import_574]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scene', sys_modules_575.module_type_store, module_type_store, ['Scene'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_575, sys_modules_575.module_type_store, module_type_store)
    else:
        from scene import Scene

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scene', None, module_type_store, ['Scene'], [Scene])

else:
    # Assigning a type to the variable 'scene' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scene', import_574)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from math import log10' statement (line 11)
try:
    from math import log10

except:
    log10 = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'math', None, module_type_store, ['log10'], [log10])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from sys import argv, stdout' statement (line 12)
try:
    from sys import argv, stdout

except:
    argv = UndefinedType
    stdout = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'sys', None, module_type_store, ['argv', 'stdout'], [argv, stdout])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from time import time' statement (line 13)
try:
    from time import time

except:
    time = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'time', None, module_type_store, ['time'], [time])


# Assigning a Str to a Name (line 15):
str_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\n  MiniLight 1.5.2 Python\n  Copyright (c) 2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.\n  http://www.hxa7241.org/minilight/\n')
# Assigning a type to the variable 'BANNER' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'BANNER', str_576)

# Assigning a Str to a Name (line 20):
str_577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', '\n----------------------------------------------------------------------\n  MiniLight 1.5.2 Python\n\n  Copyright (c) 2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.\n  http://www.hxa7241.org/minilight/\n\n  2008-02-17\n----------------------------------------------------------------------\n\nMiniLight is a minimal global illumination renderer.\n\nusage:\n  minilight image_file_pathname\n\nThe model text file format is:\n  #MiniLight\n\n  iterations\n\n  imagewidth imageheight\n\n  viewposition viewdirection viewangle\n\n  skyemission groundreflection\n  vertex0 vertex1 vertex2 reflectivity emitivity\n  vertex0 vertex1 vertex2 reflectivity emitivity\n  ...\n\n-- where iterations and image values are ints, viewangle is a float,\nand all other values are three parenthised floats. The file must end\nwith a newline. Eg.:\n  #MiniLight\n\n  100\n\n  200 150\n\n  (0 0.75 -2) (0 0 1) 45\n\n  (3626 5572 5802) (0.1 0.09 0.07)\n  (0 0 0) (0 1 0) (1 1 0)  (0.7 0.7 0.7) (0 0 0)\n')
# Assigning a type to the variable 'HELP' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'HELP', str_577)

# Assigning a Str to a Name (line 63):
str_578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'str', '#MiniLight')
# Assigning a type to the variable 'MODEL_FORMAT_ID' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'MODEL_FORMAT_ID', str_578)

# Assigning a Num to a Name (line 64):
int_579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 14), 'int')
# Assigning a type to the variable 'SAVE_PERIOD' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'SAVE_PERIOD', int_579)

@norecursion
def save_image(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'save_image'
    module_type_store = module_type_store.open_function_context('save_image', 66, 0, False)
    
    # Passed parameters checking function
    save_image.stypy_localization = localization
    save_image.stypy_type_of_self = None
    save_image.stypy_type_store = module_type_store
    save_image.stypy_function_name = 'save_image'
    save_image.stypy_param_names_list = ['image_file_pathname', 'image', 'frame_no']
    save_image.stypy_varargs_param_name = None
    save_image.stypy_kwargs_param_name = None
    save_image.stypy_call_defaults = defaults
    save_image.stypy_call_varargs = varargs
    save_image.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'save_image', ['image_file_pathname', 'image', 'frame_no'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'save_image', localization, ['image_file_pathname', 'image', 'frame_no'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'save_image(...)' code ##################

    
    # Assigning a Call to a Name (line 67):
    
    # Call to open(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'image_file_pathname' (line 67)
    image_file_pathname_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'image_file_pathname', False)
    str_582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 43), 'str', 'wb')
    # Processing the call keyword arguments (line 67)
    kwargs_583 = {}
    # Getting the type of 'open' (line 67)
    open_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'open', False)
    # Calling open(args, kwargs) (line 67)
    open_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), open_580, *[image_file_pathname_581, str_582], **kwargs_583)
    
    # Assigning a type to the variable 'image_file' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'image_file', open_call_result_584)
    
    # Call to get_formatted(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'image_file' (line 68)
    image_file_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'image_file', False)
    # Getting the type of 'frame_no' (line 68)
    frame_no_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'frame_no', False)
    int_589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 47), 'int')
    # Applying the binary operator '-' (line 68)
    result_sub_590 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 36), '-', frame_no_588, int_589)
    
    # Processing the call keyword arguments (line 68)
    kwargs_591 = {}
    # Getting the type of 'image' (line 68)
    image_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'image', False)
    # Obtaining the member 'get_formatted' of a type (line 68)
    get_formatted_586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), image_585, 'get_formatted')
    # Calling get_formatted(args, kwargs) (line 68)
    get_formatted_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), get_formatted_586, *[image_file_587, result_sub_590], **kwargs_591)
    
    
    # Call to close(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_595 = {}
    # Getting the type of 'image_file' (line 69)
    image_file_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'image_file', False)
    # Obtaining the member 'close' of a type (line 69)
    close_594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), image_file_593, 'close')
    # Calling close(args, kwargs) (line 69)
    close_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), close_594, *[], **kwargs_595)
    
    
    # ################# End of 'save_image(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'save_image' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_597)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'save_image'
    return stypy_return_type_597

# Assigning a type to the variable 'save_image' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'save_image', save_image)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 71, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = ['arg']
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    # Assigning a Name to a Name (line 73):
    # Getting the type of 'arg' (line 73)
    arg_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'arg')
    # Assigning a type to the variable 'model_file_pathname' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'model_file_pathname', arg_598)
    
    # Assigning a BinOp to a Name (line 74):
    # Getting the type of 'model_file_pathname' (line 74)
    model_file_pathname_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'model_file_pathname')
    str_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 48), 'str', '.ppm')
    # Applying the binary operator '+' (line 74)
    result_add_601 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 26), '+', model_file_pathname_599, str_600)
    
    # Assigning a type to the variable 'image_file_pathname' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'image_file_pathname', result_add_601)
    
    # Assigning a Call to a Name (line 75):
    
    # Call to open(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'model_file_pathname' (line 75)
    model_file_pathname_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'model_file_pathname', False)
    str_604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 43), 'str', 'r')
    # Processing the call keyword arguments (line 75)
    kwargs_605 = {}
    # Getting the type of 'open' (line 75)
    open_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'open', False)
    # Calling open(args, kwargs) (line 75)
    open_call_result_606 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), open_602, *[model_file_pathname_603, str_604], **kwargs_605)
    
    # Assigning a type to the variable 'model_file' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'model_file', open_call_result_606)
    
    
    # Call to strip(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_612 = {}
    
    # Call to next(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_609 = {}
    # Getting the type of 'model_file' (line 76)
    model_file_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'model_file', False)
    # Obtaining the member 'next' of a type (line 76)
    next_608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 7), model_file_607, 'next')
    # Calling next(args, kwargs) (line 76)
    next_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 76, 7), next_608, *[], **kwargs_609)
    
    # Obtaining the member 'strip' of a type (line 76)
    strip_611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 7), next_call_result_610, 'strip')
    # Calling strip(args, kwargs) (line 76)
    strip_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 76, 7), strip_611, *[], **kwargs_612)
    
    # Getting the type of 'MODEL_FORMAT_ID' (line 76)
    MODEL_FORMAT_ID_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'MODEL_FORMAT_ID')
    # Applying the binary operator '!=' (line 76)
    result_ne_615 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 7), '!=', strip_call_result_613, MODEL_FORMAT_ID_614)
    
    # Testing if the type of an if condition is none (line 76)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 4), result_ne_615):
        pass
    else:
        
        # Testing the type of an if condition (line 76)
        if_condition_616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), result_ne_615)
        # Assigning a type to the variable 'if_condition_616' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_616', if_condition_616)
        # SSA begins for if statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 14), 'str', 'invalid model file')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 77, 8), str_617, 'raise parameter', BaseException)
        # SSA join for if statement (line 76)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'model_file' (line 78)
    model_file_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'model_file')
    # Assigning a type to the variable 'model_file_618' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'model_file_618', model_file_618)
    # Testing if the for loop is going to be iterated (line 78)
    # Testing the type of a for loop iterable (line 78)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 4), model_file_618)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 78, 4), model_file_618):
        # Getting the type of the for loop variable (line 78)
        for_loop_var_619 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 4), model_file_618)
        # Assigning a type to the variable 'line' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'line', for_loop_var_619)
        # SSA begins for a for statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to isspace(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_622 = {}
        # Getting the type of 'line' (line 79)
        line_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'line', False)
        # Obtaining the member 'isspace' of a type (line 79)
        isspace_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), line_620, 'isspace')
        # Calling isspace(args, kwargs) (line 79)
        isspace_call_result_623 = invoke(stypy.reporting.localization.Localization(__file__, 79, 15), isspace_621, *[], **kwargs_622)
        
        # Applying the 'not' unary operator (line 79)
        result_not__624 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 11), 'not', isspace_call_result_623)
        
        # Testing if the type of an if condition is none (line 79)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 79, 8), result_not__624):
            pass
        else:
            
            # Testing the type of an if condition (line 79)
            if_condition_625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), result_not__624)
            # Assigning a type to the variable 'if_condition_625' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_625', if_condition_625)
            # SSA begins for if statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 80):
            
            # Call to int(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'line' (line 80)
            line_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'line', False)
            # Processing the call keyword arguments (line 80)
            kwargs_628 = {}
            # Getting the type of 'int' (line 80)
            int_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'int', False)
            # Calling int(args, kwargs) (line 80)
            int_call_result_629 = invoke(stypy.reporting.localization.Localization(__file__, 80, 25), int_626, *[line_627], **kwargs_628)
            
            # Assigning a type to the variable 'iterations' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'iterations', int_call_result_629)
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Call to a Name (line 82):
    
    # Call to Image(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'model_file' (line 82)
    model_file_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'model_file', False)
    # Processing the call keyword arguments (line 82)
    kwargs_632 = {}
    # Getting the type of 'Image' (line 82)
    Image_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'Image', False)
    # Calling Image(args, kwargs) (line 82)
    Image_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), Image_630, *[model_file_631], **kwargs_632)
    
    # Assigning a type to the variable 'image' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'image', Image_call_result_633)
    
    # Assigning a Call to a Name (line 83):
    
    # Call to Camera(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'model_file' (line 83)
    model_file_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'model_file', False)
    # Processing the call keyword arguments (line 83)
    kwargs_636 = {}
    # Getting the type of 'Camera' (line 83)
    Camera_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'Camera', False)
    # Calling Camera(args, kwargs) (line 83)
    Camera_call_result_637 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), Camera_634, *[model_file_635], **kwargs_636)
    
    # Assigning a type to the variable 'camera' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'camera', Camera_call_result_637)
    
    # Assigning a Call to a Name (line 84):
    
    # Call to Scene(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'model_file' (line 84)
    model_file_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'model_file', False)
    # Getting the type of 'camera' (line 84)
    camera_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'camera', False)
    # Obtaining the member 'view_position' of a type (line 84)
    view_position_641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 30), camera_640, 'view_position')
    # Processing the call keyword arguments (line 84)
    kwargs_642 = {}
    # Getting the type of 'Scene' (line 84)
    Scene_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'Scene', False)
    # Calling Scene(args, kwargs) (line 84)
    Scene_call_result_643 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), Scene_638, *[model_file_639, view_position_641], **kwargs_642)
    
    # Assigning a type to the variable 'scene' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'scene', Scene_call_result_643)
    
    # Call to close(...): (line 85)
    # Processing the call keyword arguments (line 85)
    kwargs_646 = {}
    # Getting the type of 'model_file' (line 85)
    model_file_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'model_file', False)
    # Obtaining the member 'close' of a type (line 85)
    close_645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), model_file_644, 'close')
    # Calling close(args, kwargs) (line 85)
    close_call_result_647 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), close_645, *[], **kwargs_646)
    
    
    # Assigning a BinOp to a Name (line 86):
    
    # Call to time(...): (line 86)
    # Processing the call keyword arguments (line 86)
    kwargs_649 = {}
    # Getting the type of 'time' (line 86)
    time_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'time', False)
    # Calling time(args, kwargs) (line 86)
    time_call_result_650 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), time_648, *[], **kwargs_649)
    
    # Getting the type of 'SAVE_PERIOD' (line 86)
    SAVE_PERIOD_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'SAVE_PERIOD')
    int_652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 40), 'int')
    # Applying the binary operator '+' (line 86)
    result_add_653 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 26), '+', SAVE_PERIOD_651, int_652)
    
    # Applying the binary operator '-' (line 86)
    result_sub_654 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 16), '-', time_call_result_650, result_add_653)
    
    # Assigning a type to the variable 'last_time' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'last_time', result_sub_654)
    
    
    # SSA begins for try-except statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Call to range(...): (line 88)
    # Processing the call arguments (line 88)
    int_656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'int')
    # Getting the type of 'iterations' (line 88)
    iterations_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'iterations', False)
    int_658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 46), 'int')
    # Applying the binary operator '+' (line 88)
    result_add_659 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 33), '+', iterations_657, int_658)
    
    # Processing the call keyword arguments (line 88)
    kwargs_660 = {}
    # Getting the type of 'range' (line 88)
    range_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'range', False)
    # Calling range(args, kwargs) (line 88)
    range_call_result_661 = invoke(stypy.reporting.localization.Localization(__file__, 88, 24), range_655, *[int_656, result_add_659], **kwargs_660)
    
    # Assigning a type to the variable 'range_call_result_661' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'range_call_result_661', range_call_result_661)
    # Testing if the for loop is going to be iterated (line 88)
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 8), range_call_result_661)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 88, 8), range_call_result_661):
        # Getting the type of the for loop variable (line 88)
        for_loop_var_662 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 8), range_call_result_661)
        # Assigning a type to the variable 'frame_no' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'frame_no', for_loop_var_662)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to get_frame(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'scene' (line 89)
        scene_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'scene', False)
        # Getting the type of 'image' (line 89)
        image_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 36), 'image', False)
        # Processing the call keyword arguments (line 89)
        kwargs_667 = {}
        # Getting the type of 'camera' (line 89)
        camera_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'camera', False)
        # Obtaining the member 'get_frame' of a type (line 89)
        get_frame_664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), camera_663, 'get_frame')
        # Calling get_frame(args, kwargs) (line 89)
        get_frame_call_result_668 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), get_frame_664, *[scene_665, image_666], **kwargs_667)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'SAVE_PERIOD' (line 90)
        SAVE_PERIOD_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'SAVE_PERIOD')
        
        # Call to time(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_671 = {}
        # Getting the type of 'time' (line 90)
        time_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 29), 'time', False)
        # Calling time(args, kwargs) (line 90)
        time_call_result_672 = invoke(stypy.reporting.localization.Localization(__file__, 90, 29), time_670, *[], **kwargs_671)
        
        # Getting the type of 'last_time' (line 90)
        last_time_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 38), 'last_time')
        # Applying the binary operator '-' (line 90)
        result_sub_674 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 29), '-', time_call_result_672, last_time_673)
        
        # Applying the binary operator '<' (line 90)
        result_lt_675 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), '<', SAVE_PERIOD_669, result_sub_674)
        
        
        # Getting the type of 'frame_no' (line 90)
        frame_no_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 51), 'frame_no')
        # Getting the type of 'iterations' (line 90)
        iterations_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 63), 'iterations')
        # Applying the binary operator '==' (line 90)
        result_eq_678 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 51), '==', frame_no_676, iterations_677)
        
        # Applying the binary operator 'or' (line 90)
        result_or_keyword_679 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), 'or', result_lt_675, result_eq_678)
        
        # Testing if the type of an if condition is none (line 90)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 12), result_or_keyword_679):
            pass
        else:
            
            # Testing the type of an if condition (line 90)
            if_condition_680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 12), result_or_keyword_679)
            # Assigning a type to the variable 'if_condition_680' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'if_condition_680', if_condition_680)
            # SSA begins for if statement (line 90)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 91):
            
            # Call to time(...): (line 91)
            # Processing the call keyword arguments (line 91)
            kwargs_682 = {}
            # Getting the type of 'time' (line 91)
            time_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'time', False)
            # Calling time(args, kwargs) (line 91)
            time_call_result_683 = invoke(stypy.reporting.localization.Localization(__file__, 91, 28), time_681, *[], **kwargs_682)
            
            # Assigning a type to the variable 'last_time' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'last_time', time_call_result_683)
            
            # Call to save_image(...): (line 92)
            # Processing the call arguments (line 92)
            # Getting the type of 'image_file_pathname' (line 92)
            image_file_pathname_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'image_file_pathname', False)
            # Getting the type of 'image' (line 92)
            image_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 48), 'image', False)
            # Getting the type of 'frame_no' (line 92)
            frame_no_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 55), 'frame_no', False)
            # Processing the call keyword arguments (line 92)
            kwargs_688 = {}
            # Getting the type of 'save_image' (line 92)
            save_image_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'save_image', False)
            # Calling save_image(args, kwargs) (line 92)
            save_image_call_result_689 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), save_image_684, *[image_file_pathname_685, image_686, frame_no_687], **kwargs_688)
            
            # SSA join for if statement (line 90)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA branch for the except part of a try statement (line 87)
    # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 87)
    module_type_store.open_ssa_branch('except')
    
    # Call to save_image(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'image_file_pathname' (line 97)
    image_file_pathname_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'image_file_pathname', False)
    # Getting the type of 'image' (line 97)
    image_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'image', False)
    # Getting the type of 'frame_no' (line 97)
    frame_no_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 47), 'frame_no', False)
    # Processing the call keyword arguments (line 97)
    kwargs_694 = {}
    # Getting the type of 'save_image' (line 97)
    save_image_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'save_image', False)
    # Calling save_image(args, kwargs) (line 97)
    save_image_call_result_695 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), save_image_690, *[image_file_pathname_691, image_692, frame_no_693], **kwargs_694)
    
    # SSA join for try-except statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_696)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_696

# Assigning a type to the variable 'main' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'main', main)

if (__name__ == '__main__'):
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'argv' (line 101)
    argv_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'argv', False)
    # Processing the call keyword arguments (line 101)
    kwargs_699 = {}
    # Getting the type of 'len' (line 101)
    len_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 7), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_700 = invoke(stypy.reporting.localization.Localization(__file__, 101, 7), len_697, *[argv_698], **kwargs_699)
    
    int_701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'int')
    # Applying the binary operator '<' (line 101)
    result_lt_702 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), '<', len_call_result_700, int_701)
    
    
    
    # Obtaining the type of the subscript
    int_703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'int')
    # Getting the type of 'argv' (line 101)
    argv_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'argv')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 24), argv_704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_706 = invoke(stypy.reporting.localization.Localization(__file__, 101, 24), getitem___705, int_703)
    
    str_707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 35), 'str', '-?')
    # Applying the binary operator '==' (line 101)
    result_eq_708 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 24), '==', subscript_call_result_706, str_707)
    
    # Applying the binary operator 'or' (line 101)
    result_or_keyword_709 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), 'or', result_lt_702, result_eq_708)
    
    
    # Obtaining the type of the subscript
    int_710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 48), 'int')
    # Getting the type of 'argv' (line 101)
    argv_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 43), 'argv')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 43), argv_711, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_713 = invoke(stypy.reporting.localization.Localization(__file__, 101, 43), getitem___712, int_710)
    
    str_714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 54), 'str', '--help')
    # Applying the binary operator '==' (line 101)
    result_eq_715 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 43), '==', subscript_call_result_713, str_714)
    
    # Applying the binary operator 'or' (line 101)
    result_or_keyword_716 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), 'or', result_or_keyword_709, result_eq_715)
    
    # Testing if the type of an if condition is none (line 101)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 4), result_or_keyword_716):
        
        # Call to main(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Obtaining the type of the subscript
        int_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'int')
        # Getting the type of 'argv' (line 104)
        argv_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 13), argv_720, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), getitem___721, int_719)
        
        # Processing the call keyword arguments (line 104)
        kwargs_723 = {}
        # Getting the type of 'main' (line 104)
        main_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'main', False)
        # Calling main(args, kwargs) (line 104)
        main_call_result_724 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), main_718, *[subscript_call_result_722], **kwargs_723)
        
    else:
        
        # Testing the type of an if condition (line 101)
        if_condition_717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 4), result_or_keyword_716)
        # Assigning a type to the variable 'if_condition_717' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'if_condition_717', if_condition_717)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 101)
        module_type_store.open_ssa_branch('else')
        
        # Call to main(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Obtaining the type of the subscript
        int_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'int')
        # Getting the type of 'argv' (line 104)
        argv_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 13), argv_720, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), getitem___721, int_719)
        
        # Processing the call keyword arguments (line 104)
        kwargs_723 = {}
        # Getting the type of 'main' (line 104)
        main_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'main', False)
        # Calling main(args, kwargs) (line 104)
        main_call_result_724 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), main_718, *[subscript_call_result_722], **kwargs_723)
        
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        



# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
