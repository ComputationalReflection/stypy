
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #  MiniLight Python : minimal global illumination renderer
2: #
3: #  Copyright (c) 2007-2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
4: #  http://www.hxa7241.org/
5: 
6: 
7: from math import log10
8: from vector3f import Vector3f, Vector3f_seq
9: 
10: PPM_ID = 'P6'
11: MINILIGHT_URI = 'http://www.hxa7241.org/minilight/'
12: DISPLAY_LUMINANCE_MAX = 200.0
13: RGB_LUMINANCE = Vector3f(0.2126, 0.7152, 0.0722)
14: GAMMA_ENCODE = 0.45
15: 
16: 
17: class Image(object):
18: 
19:     def __init__(self, in_stream):
20:         for line in in_stream:
21:             if not line.isspace():
22:                 self.width, self.height = self.dim(line.split()[0]), self.dim(line.split()[1])
23:                 self.pixels = [0.0] * self.width * self.height * 3
24:                 break
25: 
26:     def dim(self, dimension):
27:         return min(max(1, int(dimension)), 10000)
28: 
29:     def add_to_pixel(self, x, y, radiance):
30:         if x >= 0 and x < self.width and y >= 0 and y < self.height:
31:             index = (x + ((self.height - 1 - y) * self.width)) * 3
32:             self.pixels[index] += radiance.x
33:             self.pixels[index+1] += radiance.y
34:             self.pixels[index+2] += radiance.z
35: 
36:     def get_formatted(self, out, iteration):
37:         divider = 1.0 / ((iteration if iteration > 0 else 0) + 1)
38:         tonemap_scaling = self.calculate_tone_mapping(self.pixels, divider)
39:         out.write('%s\n# %s\n\n%u %u\n255\n' % (PPM_ID, MINILIGHT_URI, self.width, self.height))
40:         for channel in self.pixels:
41:             mapped = channel * divider * tonemap_scaling
42:             gammaed = (mapped if mapped > 0.0 else 0.0) ** GAMMA_ENCODE
43:             out.write(chr(min(int((gammaed * 255.0) + 0.5), 255)))
44: 
45:     def calculate_tone_mapping(self, pixels, divider):
46:         sum_of_logs = 0.0
47:         for i in range(len(pixels) / 3):
48:             y = Vector3f_seq(pixels[i * 3: i * 3 + 3]).dot(RGB_LUMINANCE) * divider
49:             sum_of_logs += log10(y if y > 1e-4 else 1e-4)
50:         log_mean_luminance = 10.0 ** (sum_of_logs / (len(pixels) / 3))
51:         a = 1.219 + (DISPLAY_LUMINANCE_MAX * 0.25) ** 0.4
52:         b = 1.219 + log_mean_luminance ** 0.4
53:         return ((a / b) ** 2.5) / DISPLAY_LUMINANCE_MAX
54: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from math import log10' statement (line 7)
try:
    from math import log10

except:
    log10 = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'math', None, module_type_store, ['log10'], [log10])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from vector3f import Vector3f, Vector3f_seq' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_273 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f')

if (type(import_273) is not StypyTypeError):

    if (import_273 != 'pyd_module'):
        __import__(import_273)
        sys_modules_274 = sys.modules[import_273]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', sys_modules_274.module_type_store, module_type_store, ['Vector3f', 'Vector3f_seq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_274, sys_modules_274.module_type_store, module_type_store)
    else:
        from vector3f import Vector3f, Vector3f_seq

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', None, module_type_store, ['Vector3f', 'Vector3f_seq'], [Vector3f, Vector3f_seq])

else:
    # Assigning a type to the variable 'vector3f' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', import_273)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')


# Assigning a Str to a Name (line 10):

# Assigning a Str to a Name (line 10):
str_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'str', 'P6')
# Assigning a type to the variable 'PPM_ID' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'PPM_ID', str_275)

# Assigning a Str to a Name (line 11):

# Assigning a Str to a Name (line 11):
str_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'str', 'http://www.hxa7241.org/minilight/')
# Assigning a type to the variable 'MINILIGHT_URI' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'MINILIGHT_URI', str_276)

# Assigning a Num to a Name (line 12):

# Assigning a Num to a Name (line 12):
float_277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 24), 'float')
# Assigning a type to the variable 'DISPLAY_LUMINANCE_MAX' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'DISPLAY_LUMINANCE_MAX', float_277)

# Assigning a Call to a Name (line 13):

# Assigning a Call to a Name (line 13):

# Call to Vector3f(...): (line 13)
# Processing the call arguments (line 13)
float_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'float')
float_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 33), 'float')
float_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 41), 'float')
# Processing the call keyword arguments (line 13)
kwargs_282 = {}
# Getting the type of 'Vector3f' (line 13)
Vector3f_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'Vector3f', False)
# Calling Vector3f(args, kwargs) (line 13)
Vector3f_call_result_283 = invoke(stypy.reporting.localization.Localization(__file__, 13, 16), Vector3f_278, *[float_279, float_280, float_281], **kwargs_282)

# Assigning a type to the variable 'RGB_LUMINANCE' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'RGB_LUMINANCE', Vector3f_call_result_283)

# Assigning a Num to a Name (line 14):

# Assigning a Num to a Name (line 14):
float_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'float')
# Assigning a type to the variable 'GAMMA_ENCODE' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'GAMMA_ENCODE', float_284)
# Declaration of the 'Image' class

class Image(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Image.__init__', ['in_stream'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['in_stream'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Getting the type of 'in_stream' (line 20)
        in_stream_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'in_stream')
        # Assigning a type to the variable 'in_stream_285' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'in_stream_285', in_stream_285)
        # Testing if the for loop is going to be iterated (line 20)
        # Testing the type of a for loop iterable (line 20)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 8), in_stream_285)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 20, 8), in_stream_285):
            # Getting the type of the for loop variable (line 20)
            for_loop_var_286 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 8), in_stream_285)
            # Assigning a type to the variable 'line' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'line', for_loop_var_286)
            # SSA begins for a for statement (line 20)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to isspace(...): (line 21)
            # Processing the call keyword arguments (line 21)
            kwargs_289 = {}
            # Getting the type of 'line' (line 21)
            line_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'line', False)
            # Obtaining the member 'isspace' of a type (line 21)
            isspace_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 19), line_287, 'isspace')
            # Calling isspace(args, kwargs) (line 21)
            isspace_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), isspace_288, *[], **kwargs_289)
            
            # Applying the 'not' unary operator (line 21)
            result_not__291 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 15), 'not', isspace_call_result_290)
            
            # Testing if the type of an if condition is none (line 21)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 21, 12), result_not__291):
                pass
            else:
                
                # Testing the type of an if condition (line 21)
                if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 12), result_not__291)
                # Assigning a type to the variable 'if_condition_292' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'if_condition_292', if_condition_292)
                # SSA begins for if statement (line 21)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Tuple to a Tuple (line 22):
                
                # Assigning a Call to a Name (line 22):
                
                # Call to dim(...): (line 22)
                # Processing the call arguments (line 22)
                
                # Obtaining the type of the subscript
                int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 64), 'int')
                
                # Call to split(...): (line 22)
                # Processing the call keyword arguments (line 22)
                kwargs_298 = {}
                # Getting the type of 'line' (line 22)
                line_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 51), 'line', False)
                # Obtaining the member 'split' of a type (line 22)
                split_297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 51), line_296, 'split')
                # Calling split(args, kwargs) (line 22)
                split_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 22, 51), split_297, *[], **kwargs_298)
                
                # Obtaining the member '__getitem__' of a type (line 22)
                getitem___300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 51), split_call_result_299, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 22)
                subscript_call_result_301 = invoke(stypy.reporting.localization.Localization(__file__, 22, 51), getitem___300, int_295)
                
                # Processing the call keyword arguments (line 22)
                kwargs_302 = {}
                # Getting the type of 'self' (line 22)
                self_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 42), 'self', False)
                # Obtaining the member 'dim' of a type (line 22)
                dim_294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 42), self_293, 'dim')
                # Calling dim(args, kwargs) (line 22)
                dim_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 22, 42), dim_294, *[subscript_call_result_301], **kwargs_302)
                
                # Assigning a type to the variable 'tuple_assignment_271' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'tuple_assignment_271', dim_call_result_303)
                
                # Assigning a Call to a Name (line 22):
                
                # Call to dim(...): (line 22)
                # Processing the call arguments (line 22)
                
                # Obtaining the type of the subscript
                int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 91), 'int')
                
                # Call to split(...): (line 22)
                # Processing the call keyword arguments (line 22)
                kwargs_309 = {}
                # Getting the type of 'line' (line 22)
                line_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 78), 'line', False)
                # Obtaining the member 'split' of a type (line 22)
                split_308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 78), line_307, 'split')
                # Calling split(args, kwargs) (line 22)
                split_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 22, 78), split_308, *[], **kwargs_309)
                
                # Obtaining the member '__getitem__' of a type (line 22)
                getitem___311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 78), split_call_result_310, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 22)
                subscript_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 22, 78), getitem___311, int_306)
                
                # Processing the call keyword arguments (line 22)
                kwargs_313 = {}
                # Getting the type of 'self' (line 22)
                self_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 69), 'self', False)
                # Obtaining the member 'dim' of a type (line 22)
                dim_305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 69), self_304, 'dim')
                # Calling dim(args, kwargs) (line 22)
                dim_call_result_314 = invoke(stypy.reporting.localization.Localization(__file__, 22, 69), dim_305, *[subscript_call_result_312], **kwargs_313)
                
                # Assigning a type to the variable 'tuple_assignment_272' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'tuple_assignment_272', dim_call_result_314)
                
                # Assigning a Name to a Attribute (line 22):
                # Getting the type of 'tuple_assignment_271' (line 22)
                tuple_assignment_271_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'tuple_assignment_271')
                # Getting the type of 'self' (line 22)
                self_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'self')
                # Setting the type of the member 'width' of a type (line 22)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), self_316, 'width', tuple_assignment_271_315)
                
                # Assigning a Name to a Attribute (line 22):
                # Getting the type of 'tuple_assignment_272' (line 22)
                tuple_assignment_272_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'tuple_assignment_272')
                # Getting the type of 'self' (line 22)
                self_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 28), 'self')
                # Setting the type of the member 'height' of a type (line 22)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 28), self_318, 'height', tuple_assignment_272_317)
                
                # Assigning a BinOp to a Attribute (line 23):
                
                # Assigning a BinOp to a Attribute (line 23):
                
                # Obtaining an instance of the builtin type 'list' (line 23)
                list_319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'list')
                # Adding type elements to the builtin type 'list' instance (line 23)
                # Adding element type (line 23)
                float_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 31), 'float')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 30), list_319, float_320)
                
                # Getting the type of 'self' (line 23)
                self_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 38), 'self')
                # Obtaining the member 'width' of a type (line 23)
                width_322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 38), self_321, 'width')
                # Applying the binary operator '*' (line 23)
                result_mul_323 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 30), '*', list_319, width_322)
                
                # Getting the type of 'self' (line 23)
                self_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 51), 'self')
                # Obtaining the member 'height' of a type (line 23)
                height_325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 51), self_324, 'height')
                # Applying the binary operator '*' (line 23)
                result_mul_326 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 49), '*', result_mul_323, height_325)
                
                int_327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 65), 'int')
                # Applying the binary operator '*' (line 23)
                result_mul_328 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 63), '*', result_mul_326, int_327)
                
                # Getting the type of 'self' (line 23)
                self_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'self')
                # Setting the type of the member 'pixels' of a type (line 23)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 16), self_329, 'pixels', result_mul_328)
                # SSA join for if statement (line 21)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def dim(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dim'
        module_type_store = module_type_store.open_function_context('dim', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Image.dim.__dict__.__setitem__('stypy_localization', localization)
        Image.dim.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Image.dim.__dict__.__setitem__('stypy_type_store', module_type_store)
        Image.dim.__dict__.__setitem__('stypy_function_name', 'Image.dim')
        Image.dim.__dict__.__setitem__('stypy_param_names_list', ['dimension'])
        Image.dim.__dict__.__setitem__('stypy_varargs_param_name', None)
        Image.dim.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Image.dim.__dict__.__setitem__('stypy_call_defaults', defaults)
        Image.dim.__dict__.__setitem__('stypy_call_varargs', varargs)
        Image.dim.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Image.dim.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Image.dim', ['dimension'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dim', localization, ['dimension'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dim(...)' code ##################

        
        # Call to min(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to max(...): (line 27)
        # Processing the call arguments (line 27)
        int_332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'int')
        
        # Call to int(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'dimension' (line 27)
        dimension_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'dimension', False)
        # Processing the call keyword arguments (line 27)
        kwargs_335 = {}
        # Getting the type of 'int' (line 27)
        int_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'int', False)
        # Calling int(args, kwargs) (line 27)
        int_call_result_336 = invoke(stypy.reporting.localization.Localization(__file__, 27, 26), int_333, *[dimension_334], **kwargs_335)
        
        # Processing the call keyword arguments (line 27)
        kwargs_337 = {}
        # Getting the type of 'max' (line 27)
        max_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'max', False)
        # Calling max(args, kwargs) (line 27)
        max_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 27, 19), max_331, *[int_332, int_call_result_336], **kwargs_337)
        
        int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 43), 'int')
        # Processing the call keyword arguments (line 27)
        kwargs_340 = {}
        # Getting the type of 'min' (line 27)
        min_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'min', False)
        # Calling min(args, kwargs) (line 27)
        min_call_result_341 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), min_330, *[max_call_result_338, int_339], **kwargs_340)
        
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', min_call_result_341)
        
        # ################# End of 'dim(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dim' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_342)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dim'
        return stypy_return_type_342


    @norecursion
    def add_to_pixel(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_to_pixel'
        module_type_store = module_type_store.open_function_context('add_to_pixel', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Image.add_to_pixel.__dict__.__setitem__('stypy_localization', localization)
        Image.add_to_pixel.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Image.add_to_pixel.__dict__.__setitem__('stypy_type_store', module_type_store)
        Image.add_to_pixel.__dict__.__setitem__('stypy_function_name', 'Image.add_to_pixel')
        Image.add_to_pixel.__dict__.__setitem__('stypy_param_names_list', ['x', 'y', 'radiance'])
        Image.add_to_pixel.__dict__.__setitem__('stypy_varargs_param_name', None)
        Image.add_to_pixel.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Image.add_to_pixel.__dict__.__setitem__('stypy_call_defaults', defaults)
        Image.add_to_pixel.__dict__.__setitem__('stypy_call_varargs', varargs)
        Image.add_to_pixel.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Image.add_to_pixel.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Image.add_to_pixel', ['x', 'y', 'radiance'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_to_pixel', localization, ['x', 'y', 'radiance'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_to_pixel(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 30)
        x_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'x')
        int_344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'int')
        # Applying the binary operator '>=' (line 30)
        result_ge_345 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), '>=', x_343, int_344)
        
        
        # Getting the type of 'x' (line 30)
        x_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'x')
        # Getting the type of 'self' (line 30)
        self_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'self')
        # Obtaining the member 'width' of a type (line 30)
        width_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 26), self_347, 'width')
        # Applying the binary operator '<' (line 30)
        result_lt_349 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 22), '<', x_346, width_348)
        
        # Applying the binary operator 'and' (line 30)
        result_and_keyword_350 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), 'and', result_ge_345, result_lt_349)
        
        # Getting the type of 'y' (line 30)
        y_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 41), 'y')
        int_352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 46), 'int')
        # Applying the binary operator '>=' (line 30)
        result_ge_353 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 41), '>=', y_351, int_352)
        
        # Applying the binary operator 'and' (line 30)
        result_and_keyword_354 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), 'and', result_and_keyword_350, result_ge_353)
        
        # Getting the type of 'y' (line 30)
        y_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 52), 'y')
        # Getting the type of 'self' (line 30)
        self_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 56), 'self')
        # Obtaining the member 'height' of a type (line 30)
        height_357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 56), self_356, 'height')
        # Applying the binary operator '<' (line 30)
        result_lt_358 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 52), '<', y_355, height_357)
        
        # Applying the binary operator 'and' (line 30)
        result_and_keyword_359 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), 'and', result_and_keyword_354, result_lt_358)
        
        # Testing if the type of an if condition is none (line 30)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 30, 8), result_and_keyword_359):
            pass
        else:
            
            # Testing the type of an if condition (line 30)
            if_condition_360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 8), result_and_keyword_359)
            # Assigning a type to the variable 'if_condition_360' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'if_condition_360', if_condition_360)
            # SSA begins for if statement (line 30)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 31):
            
            # Assigning a BinOp to a Name (line 31):
            # Getting the type of 'x' (line 31)
            x_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'x')
            # Getting the type of 'self' (line 31)
            self_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'self')
            # Obtaining the member 'height' of a type (line 31)
            height_363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 27), self_362, 'height')
            int_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 41), 'int')
            # Applying the binary operator '-' (line 31)
            result_sub_365 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 27), '-', height_363, int_364)
            
            # Getting the type of 'y' (line 31)
            y_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 45), 'y')
            # Applying the binary operator '-' (line 31)
            result_sub_367 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 43), '-', result_sub_365, y_366)
            
            # Getting the type of 'self' (line 31)
            self_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 50), 'self')
            # Obtaining the member 'width' of a type (line 31)
            width_369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 50), self_368, 'width')
            # Applying the binary operator '*' (line 31)
            result_mul_370 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 26), '*', result_sub_367, width_369)
            
            # Applying the binary operator '+' (line 31)
            result_add_371 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 21), '+', x_361, result_mul_370)
            
            int_372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 65), 'int')
            # Applying the binary operator '*' (line 31)
            result_mul_373 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 20), '*', result_add_371, int_372)
            
            # Assigning a type to the variable 'index' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'index', result_mul_373)
            
            # Getting the type of 'self' (line 32)
            self_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'self')
            # Obtaining the member 'pixels' of a type (line 32)
            pixels_375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), self_374, 'pixels')
            
            # Obtaining the type of the subscript
            # Getting the type of 'index' (line 32)
            index_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'index')
            # Getting the type of 'self' (line 32)
            self_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'self')
            # Obtaining the member 'pixels' of a type (line 32)
            pixels_378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), self_377, 'pixels')
            # Obtaining the member '__getitem__' of a type (line 32)
            getitem___379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), pixels_378, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 32)
            subscript_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), getitem___379, index_376)
            
            # Getting the type of 'radiance' (line 32)
            radiance_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 34), 'radiance')
            # Obtaining the member 'x' of a type (line 32)
            x_382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 34), radiance_381, 'x')
            # Applying the binary operator '+=' (line 32)
            result_iadd_383 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 12), '+=', subscript_call_result_380, x_382)
            # Getting the type of 'self' (line 32)
            self_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'self')
            # Obtaining the member 'pixels' of a type (line 32)
            pixels_385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), self_384, 'pixels')
            # Getting the type of 'index' (line 32)
            index_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'index')
            # Storing an element on a container (line 32)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 12), pixels_385, (index_386, result_iadd_383))
            
            
            # Getting the type of 'self' (line 33)
            self_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'self')
            # Obtaining the member 'pixels' of a type (line 33)
            pixels_388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), self_387, 'pixels')
            
            # Obtaining the type of the subscript
            # Getting the type of 'index' (line 33)
            index_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'index')
            int_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'int')
            # Applying the binary operator '+' (line 33)
            result_add_391 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 24), '+', index_389, int_390)
            
            # Getting the type of 'self' (line 33)
            self_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'self')
            # Obtaining the member 'pixels' of a type (line 33)
            pixels_393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), self_392, 'pixels')
            # Obtaining the member '__getitem__' of a type (line 33)
            getitem___394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), pixels_393, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 33)
            subscript_call_result_395 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), getitem___394, result_add_391)
            
            # Getting the type of 'radiance' (line 33)
            radiance_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 36), 'radiance')
            # Obtaining the member 'y' of a type (line 33)
            y_397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 36), radiance_396, 'y')
            # Applying the binary operator '+=' (line 33)
            result_iadd_398 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 12), '+=', subscript_call_result_395, y_397)
            # Getting the type of 'self' (line 33)
            self_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'self')
            # Obtaining the member 'pixels' of a type (line 33)
            pixels_400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), self_399, 'pixels')
            # Getting the type of 'index' (line 33)
            index_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'index')
            int_402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'int')
            # Applying the binary operator '+' (line 33)
            result_add_403 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 24), '+', index_401, int_402)
            
            # Storing an element on a container (line 33)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 12), pixels_400, (result_add_403, result_iadd_398))
            
            
            # Getting the type of 'self' (line 34)
            self_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'self')
            # Obtaining the member 'pixels' of a type (line 34)
            pixels_405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), self_404, 'pixels')
            
            # Obtaining the type of the subscript
            # Getting the type of 'index' (line 34)
            index_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'index')
            int_407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'int')
            # Applying the binary operator '+' (line 34)
            result_add_408 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 24), '+', index_406, int_407)
            
            # Getting the type of 'self' (line 34)
            self_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'self')
            # Obtaining the member 'pixels' of a type (line 34)
            pixels_410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), self_409, 'pixels')
            # Obtaining the member '__getitem__' of a type (line 34)
            getitem___411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), pixels_410, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 34)
            subscript_call_result_412 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), getitem___411, result_add_408)
            
            # Getting the type of 'radiance' (line 34)
            radiance_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 36), 'radiance')
            # Obtaining the member 'z' of a type (line 34)
            z_414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 36), radiance_413, 'z')
            # Applying the binary operator '+=' (line 34)
            result_iadd_415 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), '+=', subscript_call_result_412, z_414)
            # Getting the type of 'self' (line 34)
            self_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'self')
            # Obtaining the member 'pixels' of a type (line 34)
            pixels_417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), self_416, 'pixels')
            # Getting the type of 'index' (line 34)
            index_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'index')
            int_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'int')
            # Applying the binary operator '+' (line 34)
            result_add_420 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 24), '+', index_418, int_419)
            
            # Storing an element on a container (line 34)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 12), pixels_417, (result_add_420, result_iadd_415))
            
            # SSA join for if statement (line 30)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'add_to_pixel(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_to_pixel' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_421)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_to_pixel'
        return stypy_return_type_421


    @norecursion
    def get_formatted(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_formatted'
        module_type_store = module_type_store.open_function_context('get_formatted', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Image.get_formatted.__dict__.__setitem__('stypy_localization', localization)
        Image.get_formatted.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Image.get_formatted.__dict__.__setitem__('stypy_type_store', module_type_store)
        Image.get_formatted.__dict__.__setitem__('stypy_function_name', 'Image.get_formatted')
        Image.get_formatted.__dict__.__setitem__('stypy_param_names_list', ['out', 'iteration'])
        Image.get_formatted.__dict__.__setitem__('stypy_varargs_param_name', None)
        Image.get_formatted.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Image.get_formatted.__dict__.__setitem__('stypy_call_defaults', defaults)
        Image.get_formatted.__dict__.__setitem__('stypy_call_varargs', varargs)
        Image.get_formatted.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Image.get_formatted.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Image.get_formatted', ['out', 'iteration'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_formatted', localization, ['out', 'iteration'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_formatted(...)' code ##################

        
        # Assigning a BinOp to a Name (line 37):
        
        # Assigning a BinOp to a Name (line 37):
        float_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'float')
        
        
        # Getting the type of 'iteration' (line 37)
        iteration_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 39), 'iteration')
        int_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 51), 'int')
        # Applying the binary operator '>' (line 37)
        result_gt_425 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 39), '>', iteration_423, int_424)
        
        # Testing the type of an if expression (line 37)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 26), result_gt_425)
        # SSA begins for if expression (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'iteration' (line 37)
        iteration_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'iteration')
        # SSA branch for the else part of an if expression (line 37)
        module_type_store.open_ssa_branch('if expression else')
        int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 58), 'int')
        # SSA join for if expression (line 37)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_428 = union_type.UnionType.add(iteration_426, int_427)
        
        int_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 63), 'int')
        # Applying the binary operator '+' (line 37)
        result_add_430 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 25), '+', if_exp_428, int_429)
        
        # Applying the binary operator 'div' (line 37)
        result_div_431 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 18), 'div', float_422, result_add_430)
        
        # Assigning a type to the variable 'divider' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'divider', result_div_431)
        
        # Assigning a Call to a Name (line 38):
        
        # Assigning a Call to a Name (line 38):
        
        # Call to calculate_tone_mapping(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'self' (line 38)
        self_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 54), 'self', False)
        # Obtaining the member 'pixels' of a type (line 38)
        pixels_435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 54), self_434, 'pixels')
        # Getting the type of 'divider' (line 38)
        divider_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 67), 'divider', False)
        # Processing the call keyword arguments (line 38)
        kwargs_437 = {}
        # Getting the type of 'self' (line 38)
        self_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 26), 'self', False)
        # Obtaining the member 'calculate_tone_mapping' of a type (line 38)
        calculate_tone_mapping_433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 26), self_432, 'calculate_tone_mapping')
        # Calling calculate_tone_mapping(args, kwargs) (line 38)
        calculate_tone_mapping_call_result_438 = invoke(stypy.reporting.localization.Localization(__file__, 38, 26), calculate_tone_mapping_433, *[pixels_435, divider_436], **kwargs_437)
        
        # Assigning a type to the variable 'tonemap_scaling' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'tonemap_scaling', calculate_tone_mapping_call_result_438)
        
        # Call to write(...): (line 39)
        # Processing the call arguments (line 39)
        str_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'str', '%s\n# %s\n\n%u %u\n255\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        # Getting the type of 'PPM_ID' (line 39)
        PPM_ID_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 48), 'PPM_ID', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 48), tuple_442, PPM_ID_443)
        # Adding element type (line 39)
        # Getting the type of 'MINILIGHT_URI' (line 39)
        MINILIGHT_URI_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 56), 'MINILIGHT_URI', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 48), tuple_442, MINILIGHT_URI_444)
        # Adding element type (line 39)
        # Getting the type of 'self' (line 39)
        self_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 71), 'self', False)
        # Obtaining the member 'width' of a type (line 39)
        width_446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 71), self_445, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 48), tuple_442, width_446)
        # Adding element type (line 39)
        # Getting the type of 'self' (line 39)
        self_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 83), 'self', False)
        # Obtaining the member 'height' of a type (line 39)
        height_448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 83), self_447, 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 48), tuple_442, height_448)
        
        # Applying the binary operator '%' (line 39)
        result_mod_449 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 18), '%', str_441, tuple_442)
        
        # Processing the call keyword arguments (line 39)
        kwargs_450 = {}
        # Getting the type of 'out' (line 39)
        out_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'out', False)
        # Obtaining the member 'write' of a type (line 39)
        write_440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), out_439, 'write')
        # Calling write(args, kwargs) (line 39)
        write_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), write_440, *[result_mod_449], **kwargs_450)
        
        
        # Getting the type of 'self' (line 40)
        self_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'self')
        # Obtaining the member 'pixels' of a type (line 40)
        pixels_453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 23), self_452, 'pixels')
        # Assigning a type to the variable 'pixels_453' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'pixels_453', pixels_453)
        # Testing if the for loop is going to be iterated (line 40)
        # Testing the type of a for loop iterable (line 40)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 8), pixels_453)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 40, 8), pixels_453):
            # Getting the type of the for loop variable (line 40)
            for_loop_var_454 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 8), pixels_453)
            # Assigning a type to the variable 'channel' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'channel', for_loop_var_454)
            # SSA begins for a for statement (line 40)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 41):
            
            # Assigning a BinOp to a Name (line 41):
            # Getting the type of 'channel' (line 41)
            channel_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'channel')
            # Getting the type of 'divider' (line 41)
            divider_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'divider')
            # Applying the binary operator '*' (line 41)
            result_mul_457 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 21), '*', channel_455, divider_456)
            
            # Getting the type of 'tonemap_scaling' (line 41)
            tonemap_scaling_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 41), 'tonemap_scaling')
            # Applying the binary operator '*' (line 41)
            result_mul_459 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 39), '*', result_mul_457, tonemap_scaling_458)
            
            # Assigning a type to the variable 'mapped' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'mapped', result_mul_459)
            
            # Assigning a BinOp to a Name (line 42):
            
            # Assigning a BinOp to a Name (line 42):
            
            
            # Getting the type of 'mapped' (line 42)
            mapped_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 33), 'mapped')
            float_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 42), 'float')
            # Applying the binary operator '>' (line 42)
            result_gt_462 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 33), '>', mapped_460, float_461)
            
            # Testing the type of an if expression (line 42)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 23), result_gt_462)
            # SSA begins for if expression (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            # Getting the type of 'mapped' (line 42)
            mapped_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'mapped')
            # SSA branch for the else part of an if expression (line 42)
            module_type_store.open_ssa_branch('if expression else')
            float_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 51), 'float')
            # SSA join for if expression (line 42)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_465 = union_type.UnionType.add(mapped_463, float_464)
            
            # Getting the type of 'GAMMA_ENCODE' (line 42)
            GAMMA_ENCODE_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 59), 'GAMMA_ENCODE')
            # Applying the binary operator '**' (line 42)
            result_pow_467 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 22), '**', if_exp_465, GAMMA_ENCODE_466)
            
            # Assigning a type to the variable 'gammaed' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'gammaed', result_pow_467)
            
            # Call to write(...): (line 43)
            # Processing the call arguments (line 43)
            
            # Call to chr(...): (line 43)
            # Processing the call arguments (line 43)
            
            # Call to min(...): (line 43)
            # Processing the call arguments (line 43)
            
            # Call to int(...): (line 43)
            # Processing the call arguments (line 43)
            # Getting the type of 'gammaed' (line 43)
            gammaed_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 35), 'gammaed', False)
            float_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 45), 'float')
            # Applying the binary operator '*' (line 43)
            result_mul_475 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 35), '*', gammaed_473, float_474)
            
            float_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 54), 'float')
            # Applying the binary operator '+' (line 43)
            result_add_477 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 34), '+', result_mul_475, float_476)
            
            # Processing the call keyword arguments (line 43)
            kwargs_478 = {}
            # Getting the type of 'int' (line 43)
            int_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 30), 'int', False)
            # Calling int(args, kwargs) (line 43)
            int_call_result_479 = invoke(stypy.reporting.localization.Localization(__file__, 43, 30), int_472, *[result_add_477], **kwargs_478)
            
            int_480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 60), 'int')
            # Processing the call keyword arguments (line 43)
            kwargs_481 = {}
            # Getting the type of 'min' (line 43)
            min_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 26), 'min', False)
            # Calling min(args, kwargs) (line 43)
            min_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 43, 26), min_471, *[int_call_result_479, int_480], **kwargs_481)
            
            # Processing the call keyword arguments (line 43)
            kwargs_483 = {}
            # Getting the type of 'chr' (line 43)
            chr_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'chr', False)
            # Calling chr(args, kwargs) (line 43)
            chr_call_result_484 = invoke(stypy.reporting.localization.Localization(__file__, 43, 22), chr_470, *[min_call_result_482], **kwargs_483)
            
            # Processing the call keyword arguments (line 43)
            kwargs_485 = {}
            # Getting the type of 'out' (line 43)
            out_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'out', False)
            # Obtaining the member 'write' of a type (line 43)
            write_469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), out_468, 'write')
            # Calling write(args, kwargs) (line 43)
            write_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), write_469, *[chr_call_result_484], **kwargs_485)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'get_formatted(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_formatted' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_487)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_formatted'
        return stypy_return_type_487


    @norecursion
    def calculate_tone_mapping(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'calculate_tone_mapping'
        module_type_store = module_type_store.open_function_context('calculate_tone_mapping', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_localization', localization)
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_type_store', module_type_store)
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_function_name', 'Image.calculate_tone_mapping')
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_param_names_list', ['pixels', 'divider'])
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_varargs_param_name', None)
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_call_defaults', defaults)
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_call_varargs', varargs)
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Image.calculate_tone_mapping.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Image.calculate_tone_mapping', ['pixels', 'divider'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'calculate_tone_mapping', localization, ['pixels', 'divider'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'calculate_tone_mapping(...)' code ##################

        
        # Assigning a Num to a Name (line 46):
        
        # Assigning a Num to a Name (line 46):
        float_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'float')
        # Assigning a type to the variable 'sum_of_logs' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'sum_of_logs', float_488)
        
        
        # Call to range(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to len(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'pixels' (line 47)
        pixels_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'pixels', False)
        # Processing the call keyword arguments (line 47)
        kwargs_492 = {}
        # Getting the type of 'len' (line 47)
        len_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'len', False)
        # Calling len(args, kwargs) (line 47)
        len_call_result_493 = invoke(stypy.reporting.localization.Localization(__file__, 47, 23), len_490, *[pixels_491], **kwargs_492)
        
        int_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'int')
        # Applying the binary operator 'div' (line 47)
        result_div_495 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 23), 'div', len_call_result_493, int_494)
        
        # Processing the call keyword arguments (line 47)
        kwargs_496 = {}
        # Getting the type of 'range' (line 47)
        range_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'range', False)
        # Calling range(args, kwargs) (line 47)
        range_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), range_489, *[result_div_495], **kwargs_496)
        
        # Assigning a type to the variable 'range_call_result_497' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'range_call_result_497', range_call_result_497)
        # Testing if the for loop is going to be iterated (line 47)
        # Testing the type of a for loop iterable (line 47)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 8), range_call_result_497)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 47, 8), range_call_result_497):
            # Getting the type of the for loop variable (line 47)
            for_loop_var_498 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 8), range_call_result_497)
            # Assigning a type to the variable 'i' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'i', for_loop_var_498)
            # SSA begins for a for statement (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 48):
            
            # Assigning a BinOp to a Name (line 48):
            
            # Call to dot(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'RGB_LUMINANCE' (line 48)
            RGB_LUMINANCE_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 59), 'RGB_LUMINANCE', False)
            # Processing the call keyword arguments (line 48)
            kwargs_516 = {}
            
            # Call to Vector3f_seq(...): (line 48)
            # Processing the call arguments (line 48)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 48)
            i_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 36), 'i', False)
            int_501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 40), 'int')
            # Applying the binary operator '*' (line 48)
            result_mul_502 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 36), '*', i_500, int_501)
            
            # Getting the type of 'i' (line 48)
            i_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 43), 'i', False)
            int_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 47), 'int')
            # Applying the binary operator '*' (line 48)
            result_mul_505 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 43), '*', i_503, int_504)
            
            int_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 51), 'int')
            # Applying the binary operator '+' (line 48)
            result_add_507 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 43), '+', result_mul_505, int_506)
            
            slice_508 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 48, 29), result_mul_502, result_add_507, None)
            # Getting the type of 'pixels' (line 48)
            pixels_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'pixels', False)
            # Obtaining the member '__getitem__' of a type (line 48)
            getitem___510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 29), pixels_509, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 48)
            subscript_call_result_511 = invoke(stypy.reporting.localization.Localization(__file__, 48, 29), getitem___510, slice_508)
            
            # Processing the call keyword arguments (line 48)
            kwargs_512 = {}
            # Getting the type of 'Vector3f_seq' (line 48)
            Vector3f_seq_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'Vector3f_seq', False)
            # Calling Vector3f_seq(args, kwargs) (line 48)
            Vector3f_seq_call_result_513 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), Vector3f_seq_499, *[subscript_call_result_511], **kwargs_512)
            
            # Obtaining the member 'dot' of a type (line 48)
            dot_514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), Vector3f_seq_call_result_513, 'dot')
            # Calling dot(args, kwargs) (line 48)
            dot_call_result_517 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), dot_514, *[RGB_LUMINANCE_515], **kwargs_516)
            
            # Getting the type of 'divider' (line 48)
            divider_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 76), 'divider')
            # Applying the binary operator '*' (line 48)
            result_mul_519 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 16), '*', dot_call_result_517, divider_518)
            
            # Assigning a type to the variable 'y' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'y', result_mul_519)
            
            # Getting the type of 'sum_of_logs' (line 49)
            sum_of_logs_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'sum_of_logs')
            
            # Call to log10(...): (line 49)
            # Processing the call arguments (line 49)
            
            
            # Getting the type of 'y' (line 49)
            y_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'y', False)
            float_523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 42), 'float')
            # Applying the binary operator '>' (line 49)
            result_gt_524 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 38), '>', y_522, float_523)
            
            # Testing the type of an if expression (line 49)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 33), result_gt_524)
            # SSA begins for if expression (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            # Getting the type of 'y' (line 49)
            y_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'y', False)
            # SSA branch for the else part of an if expression (line 49)
            module_type_store.open_ssa_branch('if expression else')
            float_526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 52), 'float')
            # SSA join for if expression (line 49)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_527 = union_type.UnionType.add(y_525, float_526)
            
            # Processing the call keyword arguments (line 49)
            kwargs_528 = {}
            # Getting the type of 'log10' (line 49)
            log10_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 27), 'log10', False)
            # Calling log10(args, kwargs) (line 49)
            log10_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 49, 27), log10_521, *[if_exp_527], **kwargs_528)
            
            # Applying the binary operator '+=' (line 49)
            result_iadd_530 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 12), '+=', sum_of_logs_520, log10_call_result_529)
            # Assigning a type to the variable 'sum_of_logs' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'sum_of_logs', result_iadd_530)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a BinOp to a Name (line 50):
        
        # Assigning a BinOp to a Name (line 50):
        float_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'float')
        # Getting the type of 'sum_of_logs' (line 50)
        sum_of_logs_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 38), 'sum_of_logs')
        
        # Call to len(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'pixels' (line 50)
        pixels_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 57), 'pixels', False)
        # Processing the call keyword arguments (line 50)
        kwargs_535 = {}
        # Getting the type of 'len' (line 50)
        len_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 53), 'len', False)
        # Calling len(args, kwargs) (line 50)
        len_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 50, 53), len_533, *[pixels_534], **kwargs_535)
        
        int_537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 67), 'int')
        # Applying the binary operator 'div' (line 50)
        result_div_538 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 53), 'div', len_call_result_536, int_537)
        
        # Applying the binary operator 'div' (line 50)
        result_div_539 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 38), 'div', sum_of_logs_532, result_div_538)
        
        # Applying the binary operator '**' (line 50)
        result_pow_540 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 29), '**', float_531, result_div_539)
        
        # Assigning a type to the variable 'log_mean_luminance' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'log_mean_luminance', result_pow_540)
        
        # Assigning a BinOp to a Name (line 51):
        
        # Assigning a BinOp to a Name (line 51):
        float_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'float')
        # Getting the type of 'DISPLAY_LUMINANCE_MAX' (line 51)
        DISPLAY_LUMINANCE_MAX_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'DISPLAY_LUMINANCE_MAX')
        float_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 45), 'float')
        # Applying the binary operator '*' (line 51)
        result_mul_544 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 21), '*', DISPLAY_LUMINANCE_MAX_542, float_543)
        
        float_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 54), 'float')
        # Applying the binary operator '**' (line 51)
        result_pow_546 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 20), '**', result_mul_544, float_545)
        
        # Applying the binary operator '+' (line 51)
        result_add_547 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 12), '+', float_541, result_pow_546)
        
        # Assigning a type to the variable 'a' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'a', result_add_547)
        
        # Assigning a BinOp to a Name (line 52):
        
        # Assigning a BinOp to a Name (line 52):
        float_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'float')
        # Getting the type of 'log_mean_luminance' (line 52)
        log_mean_luminance_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'log_mean_luminance')
        float_550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 42), 'float')
        # Applying the binary operator '**' (line 52)
        result_pow_551 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 20), '**', log_mean_luminance_549, float_550)
        
        # Applying the binary operator '+' (line 52)
        result_add_552 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 12), '+', float_548, result_pow_551)
        
        # Assigning a type to the variable 'b' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'b', result_add_552)
        # Getting the type of 'a' (line 53)
        a_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'a')
        # Getting the type of 'b' (line 53)
        b_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'b')
        # Applying the binary operator 'div' (line 53)
        result_div_555 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 17), 'div', a_553, b_554)
        
        float_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 27), 'float')
        # Applying the binary operator '**' (line 53)
        result_pow_557 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 16), '**', result_div_555, float_556)
        
        # Getting the type of 'DISPLAY_LUMINANCE_MAX' (line 53)
        DISPLAY_LUMINANCE_MAX_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'DISPLAY_LUMINANCE_MAX')
        # Applying the binary operator 'div' (line 53)
        result_div_559 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 15), 'div', result_pow_557, DISPLAY_LUMINANCE_MAX_558)
        
        # Assigning a type to the variable 'stypy_return_type' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', result_div_559)
        
        # ################# End of 'calculate_tone_mapping(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'calculate_tone_mapping' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_560)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'calculate_tone_mapping'
        return stypy_return_type_560


# Assigning a type to the variable 'Image' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'Image', Image)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
