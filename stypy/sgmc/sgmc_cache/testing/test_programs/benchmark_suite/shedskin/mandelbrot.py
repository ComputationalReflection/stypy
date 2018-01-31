
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # By Daniel Rosengren, modified
2: #   http://www.timestretch.com/FractalBenchmark.html
3: # See also vectorized Python+Numeric+Pygame version:
4: #   http://www.pygame.org/pcr/mandelbrot/index.php
5: 
6: def mandelbrot(max_iterations=1000):
7:     bailout = 16
8:     for y in xrange(-39, 39):
9:         line = []
10:         for x in xrange(-39, 39):
11:             cr = y / 40.0 - 0.5
12:             ci = x / 40.0
13:             zi = 0.0
14:             zr = 0.0
15:             i = 0
16:             while True:
17:                 i += 1
18:                 temp = zr * zi
19:                 zr2 = zr * zr
20:                 zi2 = zi * zi
21:                 zr = zr2 - zi2 + cr
22:                 zi = temp + temp + ci
23:                 if zi2 + zr2 > bailout:
24:                     line.append(" ")
25:                     break
26:                 if i > max_iterations:
27:                     line.append("#")
28:                     break
29: 
30: 
31: ##        print "".join(line)
32: 
33: def run():
34:     for x in range(10):
35:         mandelbrot()
36: 
37:     return True
38: 
39: 
40: run()
41: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def mandelbrot(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 30), 'int')
    defaults = [int_1]
    # Create a new context for function 'mandelbrot'
    module_type_store = module_type_store.open_function_context('mandelbrot', 6, 0, False)
    
    # Passed parameters checking function
    mandelbrot.stypy_localization = localization
    mandelbrot.stypy_type_of_self = None
    mandelbrot.stypy_type_store = module_type_store
    mandelbrot.stypy_function_name = 'mandelbrot'
    mandelbrot.stypy_param_names_list = ['max_iterations']
    mandelbrot.stypy_varargs_param_name = None
    mandelbrot.stypy_kwargs_param_name = None
    mandelbrot.stypy_call_defaults = defaults
    mandelbrot.stypy_call_varargs = varargs
    mandelbrot.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mandelbrot', ['max_iterations'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mandelbrot', localization, ['max_iterations'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mandelbrot(...)' code ##################

    
    # Assigning a Num to a Name (line 7):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'int')
    # Assigning a type to the variable 'bailout' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'bailout', int_2)
    
    
    # Call to xrange(...): (line 8)
    # Processing the call arguments (line 8)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 25), 'int')
    # Processing the call keyword arguments (line 8)
    kwargs_6 = {}
    # Getting the type of 'xrange' (line 8)
    xrange_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 8)
    xrange_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 8, 13), xrange_3, *[int_4, int_5], **kwargs_6)
    
    # Testing if the for loop is going to be iterated (line 8)
    # Testing the type of a for loop iterable (line 8)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 8, 4), xrange_call_result_7)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 8, 4), xrange_call_result_7):
        # Getting the type of the for loop variable (line 8)
        for_loop_var_8 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 8, 4), xrange_call_result_7)
        # Assigning a type to the variable 'y' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'y', for_loop_var_8)
        # SSA begins for a for statement (line 8)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 9):
        
        # Obtaining an instance of the builtin type 'list' (line 9)
        list_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 9)
        
        # Assigning a type to the variable 'line' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'line', list_9)
        
        
        # Call to xrange(...): (line 10)
        # Processing the call arguments (line 10)
        int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'int')
        int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 29), 'int')
        # Processing the call keyword arguments (line 10)
        kwargs_13 = {}
        # Getting the type of 'xrange' (line 10)
        xrange_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 10)
        xrange_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 10, 17), xrange_10, *[int_11, int_12], **kwargs_13)
        
        # Testing if the for loop is going to be iterated (line 10)
        # Testing the type of a for loop iterable (line 10)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 10, 8), xrange_call_result_14)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 10, 8), xrange_call_result_14):
            # Getting the type of the for loop variable (line 10)
            for_loop_var_15 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 10, 8), xrange_call_result_14)
            # Assigning a type to the variable 'x' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'x', for_loop_var_15)
            # SSA begins for a for statement (line 10)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 11):
            # Getting the type of 'y' (line 11)
            y_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'y')
            float_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'float')
            # Applying the binary operator 'div' (line 11)
            result_div_18 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 17), 'div', y_16, float_17)
            
            float_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 28), 'float')
            # Applying the binary operator '-' (line 11)
            result_sub_20 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 17), '-', result_div_18, float_19)
            
            # Assigning a type to the variable 'cr' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'cr', result_sub_20)
            
            # Assigning a BinOp to a Name (line 12):
            # Getting the type of 'x' (line 12)
            x_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'x')
            float_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'float')
            # Applying the binary operator 'div' (line 12)
            result_div_23 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 17), 'div', x_21, float_22)
            
            # Assigning a type to the variable 'ci' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'ci', result_div_23)
            
            # Assigning a Num to a Name (line 13):
            float_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'float')
            # Assigning a type to the variable 'zi' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'zi', float_24)
            
            # Assigning a Num to a Name (line 14):
            float_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 17), 'float')
            # Assigning a type to the variable 'zr' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'zr', float_25)
            
            # Assigning a Num to a Name (line 15):
            int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'int')
            # Assigning a type to the variable 'i' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'i', int_26)
            
            # Getting the type of 'True' (line 16)
            True_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'True')
            # Testing if the while is going to be iterated (line 16)
            # Testing the type of an if condition (line 16)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 16, 12), True_27)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 16, 12), True_27):
                # SSA begins for while statement (line 16)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Getting the type of 'i' (line 17)
                i_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'i')
                int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'int')
                # Applying the binary operator '+=' (line 17)
                result_iadd_30 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 16), '+=', i_28, int_29)
                # Assigning a type to the variable 'i' (line 17)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'i', result_iadd_30)
                
                
                # Assigning a BinOp to a Name (line 18):
                # Getting the type of 'zr' (line 18)
                zr_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'zr')
                # Getting the type of 'zi' (line 18)
                zi_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 28), 'zi')
                # Applying the binary operator '*' (line 18)
                result_mul_33 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 23), '*', zr_31, zi_32)
                
                # Assigning a type to the variable 'temp' (line 18)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'temp', result_mul_33)
                
                # Assigning a BinOp to a Name (line 19):
                # Getting the type of 'zr' (line 19)
                zr_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'zr')
                # Getting the type of 'zr' (line 19)
                zr_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'zr')
                # Applying the binary operator '*' (line 19)
                result_mul_36 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 22), '*', zr_34, zr_35)
                
                # Assigning a type to the variable 'zr2' (line 19)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'zr2', result_mul_36)
                
                # Assigning a BinOp to a Name (line 20):
                # Getting the type of 'zi' (line 20)
                zi_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'zi')
                # Getting the type of 'zi' (line 20)
                zi_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 27), 'zi')
                # Applying the binary operator '*' (line 20)
                result_mul_39 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 22), '*', zi_37, zi_38)
                
                # Assigning a type to the variable 'zi2' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'zi2', result_mul_39)
                
                # Assigning a BinOp to a Name (line 21):
                # Getting the type of 'zr2' (line 21)
                zr2_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'zr2')
                # Getting the type of 'zi2' (line 21)
                zi2_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'zi2')
                # Applying the binary operator '-' (line 21)
                result_sub_42 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 21), '-', zr2_40, zi2_41)
                
                # Getting the type of 'cr' (line 21)
                cr_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 33), 'cr')
                # Applying the binary operator '+' (line 21)
                result_add_44 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 31), '+', result_sub_42, cr_43)
                
                # Assigning a type to the variable 'zr' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'zr', result_add_44)
                
                # Assigning a BinOp to a Name (line 22):
                # Getting the type of 'temp' (line 22)
                temp_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'temp')
                # Getting the type of 'temp' (line 22)
                temp_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 28), 'temp')
                # Applying the binary operator '+' (line 22)
                result_add_47 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 21), '+', temp_45, temp_46)
                
                # Getting the type of 'ci' (line 22)
                ci_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 35), 'ci')
                # Applying the binary operator '+' (line 22)
                result_add_49 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 33), '+', result_add_47, ci_48)
                
                # Assigning a type to the variable 'zi' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'zi', result_add_49)
                
                # Getting the type of 'zi2' (line 23)
                zi2_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'zi2')
                # Getting the type of 'zr2' (line 23)
                zr2_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'zr2')
                # Applying the binary operator '+' (line 23)
                result_add_52 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 19), '+', zi2_50, zr2_51)
                
                # Getting the type of 'bailout' (line 23)
                bailout_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 31), 'bailout')
                # Applying the binary operator '>' (line 23)
                result_gt_54 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 19), '>', result_add_52, bailout_53)
                
                # Testing if the type of an if condition is none (line 23)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 23, 16), result_gt_54):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 23)
                    if_condition_55 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 16), result_gt_54)
                    # Assigning a type to the variable 'if_condition_55' (line 23)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'if_condition_55', if_condition_55)
                    # SSA begins for if statement (line 23)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to append(...): (line 24)
                    # Processing the call arguments (line 24)
                    str_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 32), 'str', ' ')
                    # Processing the call keyword arguments (line 24)
                    kwargs_59 = {}
                    # Getting the type of 'line' (line 24)
                    line_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'line', False)
                    # Obtaining the member 'append' of a type (line 24)
                    append_57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), line_56, 'append')
                    # Calling append(args, kwargs) (line 24)
                    append_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 24, 20), append_57, *[str_58], **kwargs_59)
                    
                    # SSA join for if statement (line 23)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'i' (line 26)
                i_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 19), 'i')
                # Getting the type of 'max_iterations' (line 26)
                max_iterations_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'max_iterations')
                # Applying the binary operator '>' (line 26)
                result_gt_63 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 19), '>', i_61, max_iterations_62)
                
                # Testing if the type of an if condition is none (line 26)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 26, 16), result_gt_63):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 26)
                    if_condition_64 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 16), result_gt_63)
                    # Assigning a type to the variable 'if_condition_64' (line 26)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'if_condition_64', if_condition_64)
                    # SSA begins for if statement (line 26)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to append(...): (line 27)
                    # Processing the call arguments (line 27)
                    str_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 32), 'str', '#')
                    # Processing the call keyword arguments (line 27)
                    kwargs_68 = {}
                    # Getting the type of 'line' (line 27)
                    line_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'line', False)
                    # Obtaining the member 'append' of a type (line 27)
                    append_66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 20), line_65, 'append')
                    # Calling append(args, kwargs) (line 27)
                    append_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 27, 20), append_66, *[str_67], **kwargs_68)
                    
                    # SSA join for if statement (line 26)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for while statement (line 16)
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'mandelbrot(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mandelbrot' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mandelbrot'
    return stypy_return_type_70

# Assigning a type to the variable 'mandelbrot' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'mandelbrot', mandelbrot)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 33, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = []
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    
    # Call to range(...): (line 34)
    # Processing the call arguments (line 34)
    int_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'int')
    # Processing the call keyword arguments (line 34)
    kwargs_73 = {}
    # Getting the type of 'range' (line 34)
    range_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'range', False)
    # Calling range(args, kwargs) (line 34)
    range_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 34, 13), range_71, *[int_72], **kwargs_73)
    
    # Testing if the for loop is going to be iterated (line 34)
    # Testing the type of a for loop iterable (line 34)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 4), range_call_result_74)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 34, 4), range_call_result_74):
        # Getting the type of the for loop variable (line 34)
        for_loop_var_75 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 4), range_call_result_74)
        # Assigning a type to the variable 'x' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'x', for_loop_var_75)
        # SSA begins for a for statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to mandelbrot(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_77 = {}
        # Getting the type of 'mandelbrot' (line 35)
        mandelbrot_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'mandelbrot', False)
        # Calling mandelbrot(args, kwargs) (line 35)
        mandelbrot_call_result_78 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), mandelbrot_76, *[], **kwargs_77)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 37)
    True_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', True_79)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_80)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_80

# Assigning a type to the variable 'run' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'run', run)

# Call to run(...): (line 40)
# Processing the call keyword arguments (line 40)
kwargs_82 = {}
# Getting the type of 'run' (line 40)
run_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'run', False)
# Calling run(args, kwargs) (line 40)
run_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 40, 0), run_81, *[], **kwargs_82)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
