
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
import_561 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'camera')

if (type(import_561) is not StypyTypeError):

    if (import_561 != 'pyd_module'):
        __import__(import_561)
        sys_modules_562 = sys.modules[import_561]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'camera', sys_modules_562.module_type_store, module_type_store, ['Camera'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_562, sys_modules_562.module_type_store, module_type_store)
    else:
        from camera import Camera

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'camera', None, module_type_store, ['Camera'], [Camera])

else:
    # Assigning a type to the variable 'camera' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'camera', import_561)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from image import Image' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_563 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'image')

if (type(import_563) is not StypyTypeError):

    if (import_563 != 'pyd_module'):
        __import__(import_563)
        sys_modules_564 = sys.modules[import_563]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'image', sys_modules_564.module_type_store, module_type_store, ['Image'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_564, sys_modules_564.module_type_store, module_type_store)
    else:
        from image import Image

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'image', None, module_type_store, ['Image'], [Image])

else:
    # Assigning a type to the variable 'image' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'image', import_563)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scene import Scene' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_565 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scene')

if (type(import_565) is not StypyTypeError):

    if (import_565 != 'pyd_module'):
        __import__(import_565)
        sys_modules_566 = sys.modules[import_565]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scene', sys_modules_566.module_type_store, module_type_store, ['Scene'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_566, sys_modules_566.module_type_store, module_type_store)
    else:
        from scene import Scene

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scene', None, module_type_store, ['Scene'], [Scene])

else:
    # Assigning a type to the variable 'scene' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scene', import_565)

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
str_567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\n  MiniLight 1.5.2 Python\n  Copyright (c) 2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.\n  http://www.hxa7241.org/minilight/\n')
# Assigning a type to the variable 'BANNER' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'BANNER', str_567)

# Assigning a Str to a Name (line 20):
str_568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', '\n----------------------------------------------------------------------\n  MiniLight 1.5.2 Python\n\n  Copyright (c) 2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.\n  http://www.hxa7241.org/minilight/\n\n  2008-02-17\n----------------------------------------------------------------------\n\nMiniLight is a minimal global illumination renderer.\n\nusage:\n  minilight image_file_pathname\n\nThe model text file format is:\n  #MiniLight\n\n  iterations\n\n  imagewidth imageheight\n\n  viewposition viewdirection viewangle\n\n  skyemission groundreflection\n  vertex0 vertex1 vertex2 reflectivity emitivity\n  vertex0 vertex1 vertex2 reflectivity emitivity\n  ...\n\n-- where iterations and image values are ints, viewangle is a float,\nand all other values are three parenthised floats. The file must end\nwith a newline. Eg.:\n  #MiniLight\n\n  100\n\n  200 150\n\n  (0 0.75 -2) (0 0 1) 45\n\n  (3626 5572 5802) (0.1 0.09 0.07)\n  (0 0 0) (0 1 0) (1 1 0)  (0.7 0.7 0.7) (0 0 0)\n')
# Assigning a type to the variable 'HELP' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'HELP', str_568)

# Assigning a Str to a Name (line 63):
str_569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'str', '#MiniLight')
# Assigning a type to the variable 'MODEL_FORMAT_ID' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'MODEL_FORMAT_ID', str_569)

# Assigning a Num to a Name (line 64):
int_570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 14), 'int')
# Assigning a type to the variable 'SAVE_PERIOD' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'SAVE_PERIOD', int_570)

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
    image_file_pathname_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'image_file_pathname', False)
    str_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 43), 'str', 'wb')
    # Processing the call keyword arguments (line 67)
    kwargs_574 = {}
    # Getting the type of 'open' (line 67)
    open_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'open', False)
    # Calling open(args, kwargs) (line 67)
    open_call_result_575 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), open_571, *[image_file_pathname_572, str_573], **kwargs_574)
    
    # Assigning a type to the variable 'image_file' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'image_file', open_call_result_575)
    
    # Call to get_formatted(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'image_file' (line 68)
    image_file_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'image_file', False)
    # Getting the type of 'frame_no' (line 68)
    frame_no_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'frame_no', False)
    int_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 47), 'int')
    # Applying the binary operator '-' (line 68)
    result_sub_581 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 36), '-', frame_no_579, int_580)
    
    # Processing the call keyword arguments (line 68)
    kwargs_582 = {}
    # Getting the type of 'image' (line 68)
    image_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'image', False)
    # Obtaining the member 'get_formatted' of a type (line 68)
    get_formatted_577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), image_576, 'get_formatted')
    # Calling get_formatted(args, kwargs) (line 68)
    get_formatted_call_result_583 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), get_formatted_577, *[image_file_578, result_sub_581], **kwargs_582)
    
    
    # Call to close(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_586 = {}
    # Getting the type of 'image_file' (line 69)
    image_file_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'image_file', False)
    # Obtaining the member 'close' of a type (line 69)
    close_585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), image_file_584, 'close')
    # Calling close(args, kwargs) (line 69)
    close_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), close_585, *[], **kwargs_586)
    
    
    # ################# End of 'save_image(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'save_image' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_588)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'save_image'
    return stypy_return_type_588

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
    arg_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'arg')
    # Assigning a type to the variable 'model_file_pathname' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'model_file_pathname', arg_589)
    
    # Assigning a BinOp to a Name (line 74):
    # Getting the type of 'model_file_pathname' (line 74)
    model_file_pathname_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'model_file_pathname')
    str_591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 48), 'str', '.ppm')
    # Applying the binary operator '+' (line 74)
    result_add_592 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 26), '+', model_file_pathname_590, str_591)
    
    # Assigning a type to the variable 'image_file_pathname' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'image_file_pathname', result_add_592)
    
    # Assigning a Call to a Name (line 75):
    
    # Call to open(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'model_file_pathname' (line 75)
    model_file_pathname_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'model_file_pathname', False)
    str_595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 43), 'str', 'r')
    # Processing the call keyword arguments (line 75)
    kwargs_596 = {}
    # Getting the type of 'open' (line 75)
    open_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'open', False)
    # Calling open(args, kwargs) (line 75)
    open_call_result_597 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), open_593, *[model_file_pathname_594, str_595], **kwargs_596)
    
    # Assigning a type to the variable 'model_file' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'model_file', open_call_result_597)
    
    
    # Call to strip(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_603 = {}
    
    # Call to next(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_600 = {}
    # Getting the type of 'model_file' (line 76)
    model_file_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'model_file', False)
    # Obtaining the member 'next' of a type (line 76)
    next_599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 7), model_file_598, 'next')
    # Calling next(args, kwargs) (line 76)
    next_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 76, 7), next_599, *[], **kwargs_600)
    
    # Obtaining the member 'strip' of a type (line 76)
    strip_602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 7), next_call_result_601, 'strip')
    # Calling strip(args, kwargs) (line 76)
    strip_call_result_604 = invoke(stypy.reporting.localization.Localization(__file__, 76, 7), strip_602, *[], **kwargs_603)
    
    # Getting the type of 'MODEL_FORMAT_ID' (line 76)
    MODEL_FORMAT_ID_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'MODEL_FORMAT_ID')
    # Applying the binary operator '!=' (line 76)
    result_ne_606 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 7), '!=', strip_call_result_604, MODEL_FORMAT_ID_605)
    
    # Testing if the type of an if condition is none (line 76)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 4), result_ne_606):
        pass
    else:
        
        # Testing the type of an if condition (line 76)
        if_condition_607 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), result_ne_606)
        # Assigning a type to the variable 'if_condition_607' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_607', if_condition_607)
        # SSA begins for if statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 14), 'str', 'invalid model file')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 77, 8), str_608, 'raise parameter', BaseException)
        # SSA join for if statement (line 76)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'model_file' (line 78)
    model_file_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'model_file')
    # Assigning a type to the variable 'model_file_609' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'model_file_609', model_file_609)
    # Testing if the for loop is going to be iterated (line 78)
    # Testing the type of a for loop iterable (line 78)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 4), model_file_609)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 78, 4), model_file_609):
        # Getting the type of the for loop variable (line 78)
        for_loop_var_610 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 4), model_file_609)
        # Assigning a type to the variable 'line' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'line', for_loop_var_610)
        # SSA begins for a for statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to isspace(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_613 = {}
        # Getting the type of 'line' (line 79)
        line_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'line', False)
        # Obtaining the member 'isspace' of a type (line 79)
        isspace_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), line_611, 'isspace')
        # Calling isspace(args, kwargs) (line 79)
        isspace_call_result_614 = invoke(stypy.reporting.localization.Localization(__file__, 79, 15), isspace_612, *[], **kwargs_613)
        
        # Applying the 'not' unary operator (line 79)
        result_not__615 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 11), 'not', isspace_call_result_614)
        
        # Testing if the type of an if condition is none (line 79)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 79, 8), result_not__615):
            pass
        else:
            
            # Testing the type of an if condition (line 79)
            if_condition_616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), result_not__615)
            # Assigning a type to the variable 'if_condition_616' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_616', if_condition_616)
            # SSA begins for if statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 80):
            
            # Call to int(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'line' (line 80)
            line_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'line', False)
            # Processing the call keyword arguments (line 80)
            kwargs_619 = {}
            # Getting the type of 'int' (line 80)
            int_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'int', False)
            # Calling int(args, kwargs) (line 80)
            int_call_result_620 = invoke(stypy.reporting.localization.Localization(__file__, 80, 25), int_617, *[line_618], **kwargs_619)
            
            # Assigning a type to the variable 'iterations' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'iterations', int_call_result_620)
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Call to a Name (line 82):
    
    # Call to Image(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'model_file' (line 82)
    model_file_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'model_file', False)
    # Processing the call keyword arguments (line 82)
    kwargs_623 = {}
    # Getting the type of 'Image' (line 82)
    Image_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'Image', False)
    # Calling Image(args, kwargs) (line 82)
    Image_call_result_624 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), Image_621, *[model_file_622], **kwargs_623)
    
    # Assigning a type to the variable 'image' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'image', Image_call_result_624)
    
    # Assigning a Call to a Name (line 83):
    
    # Call to Camera(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'model_file' (line 83)
    model_file_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'model_file', False)
    # Processing the call keyword arguments (line 83)
    kwargs_627 = {}
    # Getting the type of 'Camera' (line 83)
    Camera_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'Camera', False)
    # Calling Camera(args, kwargs) (line 83)
    Camera_call_result_628 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), Camera_625, *[model_file_626], **kwargs_627)
    
    # Assigning a type to the variable 'camera' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'camera', Camera_call_result_628)
    
    # Assigning a Call to a Name (line 84):
    
    # Call to Scene(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'model_file' (line 84)
    model_file_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'model_file', False)
    # Getting the type of 'camera' (line 84)
    camera_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'camera', False)
    # Obtaining the member 'view_position' of a type (line 84)
    view_position_632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 30), camera_631, 'view_position')
    # Processing the call keyword arguments (line 84)
    kwargs_633 = {}
    # Getting the type of 'Scene' (line 84)
    Scene_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'Scene', False)
    # Calling Scene(args, kwargs) (line 84)
    Scene_call_result_634 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), Scene_629, *[model_file_630, view_position_632], **kwargs_633)
    
    # Assigning a type to the variable 'scene' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'scene', Scene_call_result_634)
    
    # Call to close(...): (line 85)
    # Processing the call keyword arguments (line 85)
    kwargs_637 = {}
    # Getting the type of 'model_file' (line 85)
    model_file_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'model_file', False)
    # Obtaining the member 'close' of a type (line 85)
    close_636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), model_file_635, 'close')
    # Calling close(args, kwargs) (line 85)
    close_call_result_638 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), close_636, *[], **kwargs_637)
    
    
    # Assigning a BinOp to a Name (line 86):
    
    # Call to time(...): (line 86)
    # Processing the call keyword arguments (line 86)
    kwargs_640 = {}
    # Getting the type of 'time' (line 86)
    time_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'time', False)
    # Calling time(args, kwargs) (line 86)
    time_call_result_641 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), time_639, *[], **kwargs_640)
    
    # Getting the type of 'SAVE_PERIOD' (line 86)
    SAVE_PERIOD_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'SAVE_PERIOD')
    int_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 40), 'int')
    # Applying the binary operator '+' (line 86)
    result_add_644 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 26), '+', SAVE_PERIOD_642, int_643)
    
    # Applying the binary operator '-' (line 86)
    result_sub_645 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 16), '-', time_call_result_641, result_add_644)
    
    # Assigning a type to the variable 'last_time' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'last_time', result_sub_645)
    
    
    # SSA begins for try-except statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Call to range(...): (line 88)
    # Processing the call arguments (line 88)
    int_647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'int')
    # Getting the type of 'iterations' (line 88)
    iterations_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'iterations', False)
    int_649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 46), 'int')
    # Applying the binary operator '+' (line 88)
    result_add_650 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 33), '+', iterations_648, int_649)
    
    # Processing the call keyword arguments (line 88)
    kwargs_651 = {}
    # Getting the type of 'range' (line 88)
    range_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'range', False)
    # Calling range(args, kwargs) (line 88)
    range_call_result_652 = invoke(stypy.reporting.localization.Localization(__file__, 88, 24), range_646, *[int_647, result_add_650], **kwargs_651)
    
    # Assigning a type to the variable 'range_call_result_652' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'range_call_result_652', range_call_result_652)
    # Testing if the for loop is going to be iterated (line 88)
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 8), range_call_result_652)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 88, 8), range_call_result_652):
        # Getting the type of the for loop variable (line 88)
        for_loop_var_653 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 8), range_call_result_652)
        # Assigning a type to the variable 'frame_no' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'frame_no', for_loop_var_653)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to get_frame(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'scene' (line 89)
        scene_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'scene', False)
        # Getting the type of 'image' (line 89)
        image_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 36), 'image', False)
        # Processing the call keyword arguments (line 89)
        kwargs_658 = {}
        # Getting the type of 'camera' (line 89)
        camera_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'camera', False)
        # Obtaining the member 'get_frame' of a type (line 89)
        get_frame_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), camera_654, 'get_frame')
        # Calling get_frame(args, kwargs) (line 89)
        get_frame_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), get_frame_655, *[scene_656, image_657], **kwargs_658)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'SAVE_PERIOD' (line 90)
        SAVE_PERIOD_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'SAVE_PERIOD')
        
        # Call to time(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_662 = {}
        # Getting the type of 'time' (line 90)
        time_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 29), 'time', False)
        # Calling time(args, kwargs) (line 90)
        time_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 90, 29), time_661, *[], **kwargs_662)
        
        # Getting the type of 'last_time' (line 90)
        last_time_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 38), 'last_time')
        # Applying the binary operator '-' (line 90)
        result_sub_665 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 29), '-', time_call_result_663, last_time_664)
        
        # Applying the binary operator '<' (line 90)
        result_lt_666 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), '<', SAVE_PERIOD_660, result_sub_665)
        
        
        # Getting the type of 'frame_no' (line 90)
        frame_no_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 51), 'frame_no')
        # Getting the type of 'iterations' (line 90)
        iterations_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 63), 'iterations')
        # Applying the binary operator '==' (line 90)
        result_eq_669 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 51), '==', frame_no_667, iterations_668)
        
        # Applying the binary operator 'or' (line 90)
        result_or_keyword_670 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), 'or', result_lt_666, result_eq_669)
        
        # Testing if the type of an if condition is none (line 90)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 12), result_or_keyword_670):
            pass
        else:
            
            # Testing the type of an if condition (line 90)
            if_condition_671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 12), result_or_keyword_670)
            # Assigning a type to the variable 'if_condition_671' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'if_condition_671', if_condition_671)
            # SSA begins for if statement (line 90)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 91):
            
            # Call to time(...): (line 91)
            # Processing the call keyword arguments (line 91)
            kwargs_673 = {}
            # Getting the type of 'time' (line 91)
            time_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'time', False)
            # Calling time(args, kwargs) (line 91)
            time_call_result_674 = invoke(stypy.reporting.localization.Localization(__file__, 91, 28), time_672, *[], **kwargs_673)
            
            # Assigning a type to the variable 'last_time' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'last_time', time_call_result_674)
            
            # Call to save_image(...): (line 92)
            # Processing the call arguments (line 92)
            # Getting the type of 'image_file_pathname' (line 92)
            image_file_pathname_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'image_file_pathname', False)
            # Getting the type of 'image' (line 92)
            image_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 48), 'image', False)
            # Getting the type of 'frame_no' (line 92)
            frame_no_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 55), 'frame_no', False)
            # Processing the call keyword arguments (line 92)
            kwargs_679 = {}
            # Getting the type of 'save_image' (line 92)
            save_image_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'save_image', False)
            # Calling save_image(args, kwargs) (line 92)
            save_image_call_result_680 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), save_image_675, *[image_file_pathname_676, image_677, frame_no_678], **kwargs_679)
            
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
    image_file_pathname_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'image_file_pathname', False)
    # Getting the type of 'image' (line 97)
    image_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'image', False)
    # Getting the type of 'frame_no' (line 97)
    frame_no_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 47), 'frame_no', False)
    # Processing the call keyword arguments (line 97)
    kwargs_685 = {}
    # Getting the type of 'save_image' (line 97)
    save_image_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'save_image', False)
    # Calling save_image(args, kwargs) (line 97)
    save_image_call_result_686 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), save_image_681, *[image_file_pathname_682, image_683, frame_no_684], **kwargs_685)
    
    # SSA join for try-except statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_687)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_687

# Assigning a type to the variable 'main' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'main', main)

if (__name__ == '__main__'):
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'argv' (line 101)
    argv_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'argv', False)
    # Processing the call keyword arguments (line 101)
    kwargs_690 = {}
    # Getting the type of 'len' (line 101)
    len_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 7), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_691 = invoke(stypy.reporting.localization.Localization(__file__, 101, 7), len_688, *[argv_689], **kwargs_690)
    
    int_692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'int')
    # Applying the binary operator '<' (line 101)
    result_lt_693 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), '<', len_call_result_691, int_692)
    
    
    
    # Obtaining the type of the subscript
    int_694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'int')
    # Getting the type of 'argv' (line 101)
    argv_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'argv')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 24), argv_695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_697 = invoke(stypy.reporting.localization.Localization(__file__, 101, 24), getitem___696, int_694)
    
    str_698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 35), 'str', '-?')
    # Applying the binary operator '==' (line 101)
    result_eq_699 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 24), '==', subscript_call_result_697, str_698)
    
    # Applying the binary operator 'or' (line 101)
    result_or_keyword_700 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), 'or', result_lt_693, result_eq_699)
    
    
    # Obtaining the type of the subscript
    int_701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 48), 'int')
    # Getting the type of 'argv' (line 101)
    argv_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 43), 'argv')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 43), argv_702, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_704 = invoke(stypy.reporting.localization.Localization(__file__, 101, 43), getitem___703, int_701)
    
    str_705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 54), 'str', '--help')
    # Applying the binary operator '==' (line 101)
    result_eq_706 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 43), '==', subscript_call_result_704, str_705)
    
    # Applying the binary operator 'or' (line 101)
    result_or_keyword_707 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), 'or', result_or_keyword_700, result_eq_706)
    
    # Testing if the type of an if condition is none (line 101)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 4), result_or_keyword_707):
        
        # Call to main(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Obtaining the type of the subscript
        int_710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'int')
        # Getting the type of 'argv' (line 104)
        argv_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 13), argv_711, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_713 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), getitem___712, int_710)
        
        # Processing the call keyword arguments (line 104)
        kwargs_714 = {}
        # Getting the type of 'main' (line 104)
        main_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'main', False)
        # Calling main(args, kwargs) (line 104)
        main_call_result_715 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), main_709, *[subscript_call_result_713], **kwargs_714)
        
    else:
        
        # Testing the type of an if condition (line 101)
        if_condition_708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 4), result_or_keyword_707)
        # Assigning a type to the variable 'if_condition_708' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'if_condition_708', if_condition_708)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 101)
        module_type_store.open_ssa_branch('else')
        
        # Call to main(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Obtaining the type of the subscript
        int_710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'int')
        # Getting the type of 'argv' (line 104)
        argv_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'argv', False)
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 13), argv_711, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_713 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), getitem___712, int_710)
        
        # Processing the call keyword arguments (line 104)
        kwargs_714 = {}
        # Getting the type of 'main' (line 104)
        main_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'main', False)
        # Calling main(args, kwargs) (line 104)
        main_call_result_715 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), main_709, *[subscript_call_result_713], **kwargs_714)
        
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        



# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
