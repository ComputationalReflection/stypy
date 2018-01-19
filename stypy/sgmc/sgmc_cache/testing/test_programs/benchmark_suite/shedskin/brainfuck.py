
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # bfi - BrainFuck GNU interpreter, (C) 2002 Philippe Biondi, biondi@cartel-securite.fr
2: # Modified for ShedSkin, but can be used with Python/Psyco too
3: 
4: from sys import stdin, stdout, argv
5: import os
6: 
7: def Relative(path):
8:     return os.path.join(os.path.dirname(__file__), path)
9: 
10: def BF_interpreter(prog):
11:     CELL = 255 # Or 65535  default 255
12:     # clear program to speed up execution
13:     prog = "".join([c for c in prog if c in "><+-.,[]"])
14:     len_prog = len(prog)
15: 
16:     tape = [0] # This can be initialized to 30000 cells
17:     ip = 0
18:     p = 0
19:     level = 0
20: 
21:     while ip < len_prog:
22:         x = prog[ip]
23:         ip += 1
24: 
25:         if x == '+':
26:             tape[p] = (tape[p]+1) & CELL
27:         elif x == '-':
28:             tape[p] = (tape[p]-1) & CELL
29:         elif x == '>':
30:             p += 1
31:             if len(tape) <= p:
32:                 tape.append(0)
33:         elif x == '<':
34:             if p:
35:                 p -= 1
36:             else:
37:                 #print "Warning: inserting one element at the begining"
38:                 tape.insert(0, 0)
39:         elif x == '.':
40:             pass#stdout.write( chr(tape[p]) )
41:             chr(tape[p])
42:         elif x == ',':
43:             tape[p] = ord(stdin.read(1))
44:         elif x == '[':
45:             if not tape[p]:
46:                 while True:
47:                     if prog[ip] == '[':
48:                         level += 1
49:                     if prog[ip] == ']':
50:                         if level:
51:                             level -= 1
52:                         else:
53:                             break
54:                     ip += 1
55:                 ip += 1
56:         elif x == ']':
57:             ip -= 2
58:             while True:
59:                 if prog[ip] == ']':
60:                     level += 1
61:                 if prog[ip] == '[':
62:                     if level:
63:                         level -= 1
64:                     else:
65:                         break
66:                 ip -= 1
67: 
68: def run():
69:     program = file(Relative('testdata/99bottles.bf')).read()
70:     BF_interpreter(program)
71:     return True
72: 
73: run()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from sys import stdin, stdout, argv' statement (line 4)
try:
    from sys import stdin, stdout, argv

except:
    stdin = UndefinedType
    stdout = UndefinedType
    argv = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', None, module_type_store, ['stdin', 'stdout', 'argv'], [stdin, stdout, argv])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 7, 0, False)
    
    # Passed parameters checking function
    Relative.stypy_localization = localization
    Relative.stypy_type_of_self = None
    Relative.stypy_type_store = module_type_store
    Relative.stypy_function_name = 'Relative'
    Relative.stypy_param_names_list = ['path']
    Relative.stypy_varargs_param_name = None
    Relative.stypy_kwargs_param_name = None
    Relative.stypy_call_defaults = defaults
    Relative.stypy_call_varargs = varargs
    Relative.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Relative', ['path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Relative', localization, ['path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Relative(...)' code ##################

    
    # Call to join(...): (line 8)
    # Processing the call arguments (line 8)
    
    # Call to dirname(...): (line 8)
    # Processing the call arguments (line 8)
    # Getting the type of '__file__' (line 8)
    file___7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 40), '__file__', False)
    # Processing the call keyword arguments (line 8)
    kwargs_8 = {}
    # Getting the type of 'os' (line 8)
    os_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 8)
    path_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 24), os_4, 'path')
    # Obtaining the member 'dirname' of a type (line 8)
    dirname_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 24), path_5, 'dirname')
    # Calling dirname(args, kwargs) (line 8)
    dirname_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 8, 24), dirname_6, *[file___7], **kwargs_8)
    
    # Getting the type of 'path' (line 8)
    path_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 51), 'path', False)
    # Processing the call keyword arguments (line 8)
    kwargs_11 = {}
    # Getting the type of 'os' (line 8)
    os_1 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 8)
    path_2 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 11), os_1, 'path')
    # Obtaining the member 'join' of a type (line 8)
    join_3 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 11), path_2, 'join')
    # Calling join(args, kwargs) (line 8)
    join_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 8, 11), join_3, *[dirname_call_result_9, path_10], **kwargs_11)
    
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type', join_call_result_12)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_13

# Assigning a type to the variable 'Relative' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'Relative', Relative)

@norecursion
def BF_interpreter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'BF_interpreter'
    module_type_store = module_type_store.open_function_context('BF_interpreter', 10, 0, False)
    
    # Passed parameters checking function
    BF_interpreter.stypy_localization = localization
    BF_interpreter.stypy_type_of_self = None
    BF_interpreter.stypy_type_store = module_type_store
    BF_interpreter.stypy_function_name = 'BF_interpreter'
    BF_interpreter.stypy_param_names_list = ['prog']
    BF_interpreter.stypy_varargs_param_name = None
    BF_interpreter.stypy_kwargs_param_name = None
    BF_interpreter.stypy_call_defaults = defaults
    BF_interpreter.stypy_call_varargs = varargs
    BF_interpreter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'BF_interpreter', ['prog'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'BF_interpreter', localization, ['prog'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'BF_interpreter(...)' code ##################

    
    # Assigning a Num to a Name (line 11):
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'int')
    # Assigning a type to the variable 'CELL' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'CELL', int_14)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to join(...): (line 13)
    # Processing the call arguments (line 13)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'prog' (line 13)
    prog_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 31), 'prog', False)
    comprehension_22 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 20), prog_21)
    # Assigning a type to the variable 'c' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'c', comprehension_22)
    
    # Getting the type of 'c' (line 13)
    c_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 39), 'c', False)
    str_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 44), 'str', '><+-.,[]')
    # Applying the binary operator 'in' (line 13)
    result_contains_20 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 39), 'in', c_18, str_19)
    
    # Getting the type of 'c' (line 13)
    c_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'c', False)
    list_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 20), list_23, c_17)
    # Processing the call keyword arguments (line 13)
    kwargs_24 = {}
    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', '')
    # Obtaining the member 'join' of a type (line 13)
    join_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 11), str_15, 'join')
    # Calling join(args, kwargs) (line 13)
    join_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 13, 11), join_16, *[list_23], **kwargs_24)
    
    # Assigning a type to the variable 'prog' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'prog', join_call_result_25)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to len(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'prog' (line 14)
    prog_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'prog', False)
    # Processing the call keyword arguments (line 14)
    kwargs_28 = {}
    # Getting the type of 'len' (line 14)
    len_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'len', False)
    # Calling len(args, kwargs) (line 14)
    len_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 14, 15), len_26, *[prog_27], **kwargs_28)
    
    # Assigning a type to the variable 'len_prog' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'len_prog', len_call_result_29)
    
    # Assigning a List to a Name (line 16):
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 11), list_30, int_31)
    
    # Assigning a type to the variable 'tape' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tape', list_30)
    
    # Assigning a Num to a Name (line 17):
    int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 9), 'int')
    # Assigning a type to the variable 'ip' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ip', int_32)
    
    # Assigning a Num to a Name (line 18):
    int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
    # Assigning a type to the variable 'p' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'p', int_33)
    
    # Assigning a Num to a Name (line 19):
    int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 12), 'int')
    # Assigning a type to the variable 'level' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'level', int_34)
    
    
    # Getting the type of 'ip' (line 21)
    ip_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'ip')
    # Getting the type of 'len_prog' (line 21)
    len_prog_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'len_prog')
    # Applying the binary operator '<' (line 21)
    result_lt_37 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 10), '<', ip_35, len_prog_36)
    
    # Testing the type of an if condition (line 21)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 4), result_lt_37)
    # SSA begins for while statement (line 21)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Subscript to a Name (line 22):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ip' (line 22)
    ip_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'ip')
    # Getting the type of 'prog' (line 22)
    prog_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'prog')
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), prog_39, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), getitem___40, ip_38)
    
    # Assigning a type to the variable 'x' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'x', subscript_call_result_41)
    
    # Getting the type of 'ip' (line 23)
    ip_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'ip')
    int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'int')
    # Applying the binary operator '+=' (line 23)
    result_iadd_44 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 8), '+=', ip_42, int_43)
    # Assigning a type to the variable 'ip' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'ip', result_iadd_44)
    
    
    
    # Getting the type of 'x' (line 25)
    x_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'x')
    str_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'str', '+')
    # Applying the binary operator '==' (line 25)
    result_eq_47 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 11), '==', x_45, str_46)
    
    # Testing the type of an if condition (line 25)
    if_condition_48 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 8), result_eq_47)
    # Assigning a type to the variable 'if_condition_48' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'if_condition_48', if_condition_48)
    # SSA begins for if statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 26):
    
    # Obtaining the type of the subscript
    # Getting the type of 'p' (line 26)
    p_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'p')
    # Getting the type of 'tape' (line 26)
    tape_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'tape')
    # Obtaining the member '__getitem__' of a type (line 26)
    getitem___51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 23), tape_50, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 26)
    subscript_call_result_52 = invoke(stypy.reporting.localization.Localization(__file__, 26, 23), getitem___51, p_49)
    
    int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'int')
    # Applying the binary operator '+' (line 26)
    result_add_54 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 23), '+', subscript_call_result_52, int_53)
    
    # Getting the type of 'CELL' (line 26)
    CELL_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'CELL')
    # Applying the binary operator '&' (line 26)
    result_and__56 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 22), '&', result_add_54, CELL_55)
    
    # Getting the type of 'tape' (line 26)
    tape_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'tape')
    # Getting the type of 'p' (line 26)
    p_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'p')
    # Storing an element on a container (line 26)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), tape_57, (p_58, result_and__56))
    # SSA branch for the else part of an if statement (line 25)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'x' (line 27)
    x_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'x')
    str_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 18), 'str', '-')
    # Applying the binary operator '==' (line 27)
    result_eq_61 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 13), '==', x_59, str_60)
    
    # Testing the type of an if condition (line 27)
    if_condition_62 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 13), result_eq_61)
    # Assigning a type to the variable 'if_condition_62' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'if_condition_62', if_condition_62)
    # SSA begins for if statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 28):
    
    # Obtaining the type of the subscript
    # Getting the type of 'p' (line 28)
    p_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'p')
    # Getting the type of 'tape' (line 28)
    tape_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'tape')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___65 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 23), tape_64, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 28, 23), getitem___65, p_63)
    
    int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'int')
    # Applying the binary operator '-' (line 28)
    result_sub_68 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 23), '-', subscript_call_result_66, int_67)
    
    # Getting the type of 'CELL' (line 28)
    CELL_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 36), 'CELL')
    # Applying the binary operator '&' (line 28)
    result_and__70 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 22), '&', result_sub_68, CELL_69)
    
    # Getting the type of 'tape' (line 28)
    tape_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'tape')
    # Getting the type of 'p' (line 28)
    p_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'p')
    # Storing an element on a container (line 28)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 12), tape_71, (p_72, result_and__70))
    # SSA branch for the else part of an if statement (line 27)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'x' (line 29)
    x_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'x')
    str_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'str', '>')
    # Applying the binary operator '==' (line 29)
    result_eq_75 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 13), '==', x_73, str_74)
    
    # Testing the type of an if condition (line 29)
    if_condition_76 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 13), result_eq_75)
    # Assigning a type to the variable 'if_condition_76' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'if_condition_76', if_condition_76)
    # SSA begins for if statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'p' (line 30)
    p_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'p')
    int_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'int')
    # Applying the binary operator '+=' (line 30)
    result_iadd_79 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 12), '+=', p_77, int_78)
    # Assigning a type to the variable 'p' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'p', result_iadd_79)
    
    
    
    
    # Call to len(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'tape' (line 31)
    tape_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'tape', False)
    # Processing the call keyword arguments (line 31)
    kwargs_82 = {}
    # Getting the type of 'len' (line 31)
    len_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'len', False)
    # Calling len(args, kwargs) (line 31)
    len_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), len_80, *[tape_81], **kwargs_82)
    
    # Getting the type of 'p' (line 31)
    p_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'p')
    # Applying the binary operator '<=' (line 31)
    result_le_85 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), '<=', len_call_result_83, p_84)
    
    # Testing the type of an if condition (line 31)
    if_condition_86 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 12), result_le_85)
    # Assigning a type to the variable 'if_condition_86' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'if_condition_86', if_condition_86)
    # SSA begins for if statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 32)
    # Processing the call arguments (line 32)
    int_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 28), 'int')
    # Processing the call keyword arguments (line 32)
    kwargs_90 = {}
    # Getting the type of 'tape' (line 32)
    tape_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'tape', False)
    # Obtaining the member 'append' of a type (line 32)
    append_88 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), tape_87, 'append')
    # Calling append(args, kwargs) (line 32)
    append_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 32, 16), append_88, *[int_89], **kwargs_90)
    
    # SSA join for if statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 29)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'x' (line 33)
    x_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'x')
    str_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'str', '<')
    # Applying the binary operator '==' (line 33)
    result_eq_94 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 13), '==', x_92, str_93)
    
    # Testing the type of an if condition (line 33)
    if_condition_95 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 13), result_eq_94)
    # Assigning a type to the variable 'if_condition_95' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'if_condition_95', if_condition_95)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'p' (line 34)
    p_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'p')
    # Testing the type of an if condition (line 34)
    if_condition_97 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 12), p_96)
    # Assigning a type to the variable 'if_condition_97' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'if_condition_97', if_condition_97)
    # SSA begins for if statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'p' (line 35)
    p_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'p')
    int_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'int')
    # Applying the binary operator '-=' (line 35)
    result_isub_100 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 16), '-=', p_98, int_99)
    # Assigning a type to the variable 'p' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'p', result_isub_100)
    
    # SSA branch for the else part of an if statement (line 34)
    module_type_store.open_ssa_branch('else')
    
    # Call to insert(...): (line 38)
    # Processing the call arguments (line 38)
    int_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 28), 'int')
    int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 31), 'int')
    # Processing the call keyword arguments (line 38)
    kwargs_105 = {}
    # Getting the type of 'tape' (line 38)
    tape_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'tape', False)
    # Obtaining the member 'insert' of a type (line 38)
    insert_102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), tape_101, 'insert')
    # Calling insert(args, kwargs) (line 38)
    insert_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), insert_102, *[int_103, int_104], **kwargs_105)
    
    # SSA join for if statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 33)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'x' (line 39)
    x_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'x')
    str_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'str', '.')
    # Applying the binary operator '==' (line 39)
    result_eq_109 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 13), '==', x_107, str_108)
    
    # Testing the type of an if condition (line 39)
    if_condition_110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 13), result_eq_109)
    # Assigning a type to the variable 'if_condition_110' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'if_condition_110', if_condition_110)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    
    # Call to chr(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining the type of the subscript
    # Getting the type of 'p' (line 41)
    p_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'p', False)
    # Getting the type of 'tape' (line 41)
    tape_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'tape', False)
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), tape_113, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 41, 16), getitem___114, p_112)
    
    # Processing the call keyword arguments (line 41)
    kwargs_116 = {}
    # Getting the type of 'chr' (line 41)
    chr_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'chr', False)
    # Calling chr(args, kwargs) (line 41)
    chr_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), chr_111, *[subscript_call_result_115], **kwargs_116)
    
    # SSA branch for the else part of an if statement (line 39)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'x' (line 42)
    x_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'x')
    str_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'str', ',')
    # Applying the binary operator '==' (line 42)
    result_eq_120 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 13), '==', x_118, str_119)
    
    # Testing the type of an if condition (line 42)
    if_condition_121 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 13), result_eq_120)
    # Assigning a type to the variable 'if_condition_121' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'if_condition_121', if_condition_121)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 43):
    
    # Call to ord(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Call to read(...): (line 43)
    # Processing the call arguments (line 43)
    int_125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 37), 'int')
    # Processing the call keyword arguments (line 43)
    kwargs_126 = {}
    # Getting the type of 'stdin' (line 43)
    stdin_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 26), 'stdin', False)
    # Obtaining the member 'read' of a type (line 43)
    read_124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 26), stdin_123, 'read')
    # Calling read(args, kwargs) (line 43)
    read_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 43, 26), read_124, *[int_125], **kwargs_126)
    
    # Processing the call keyword arguments (line 43)
    kwargs_128 = {}
    # Getting the type of 'ord' (line 43)
    ord_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'ord', False)
    # Calling ord(args, kwargs) (line 43)
    ord_call_result_129 = invoke(stypy.reporting.localization.Localization(__file__, 43, 22), ord_122, *[read_call_result_127], **kwargs_128)
    
    # Getting the type of 'tape' (line 43)
    tape_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'tape')
    # Getting the type of 'p' (line 43)
    p_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'p')
    # Storing an element on a container (line 43)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), tape_130, (p_131, ord_call_result_129))
    # SSA branch for the else part of an if statement (line 42)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'x' (line 44)
    x_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'x')
    str_133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 18), 'str', '[')
    # Applying the binary operator '==' (line 44)
    result_eq_134 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 13), '==', x_132, str_133)
    
    # Testing the type of an if condition (line 44)
    if_condition_135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 13), result_eq_134)
    # Assigning a type to the variable 'if_condition_135' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'if_condition_135', if_condition_135)
    # SSA begins for if statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'p' (line 45)
    p_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'p')
    # Getting the type of 'tape' (line 45)
    tape_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'tape')
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 19), tape_137, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), getitem___138, p_136)
    
    # Applying the 'not' unary operator (line 45)
    result_not__140 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 15), 'not', subscript_call_result_139)
    
    # Testing the type of an if condition (line 45)
    if_condition_141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 12), result_not__140)
    # Assigning a type to the variable 'if_condition_141' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'if_condition_141', if_condition_141)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'True' (line 46)
    True_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'True')
    # Testing the type of an if condition (line 46)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 16), True_142)
    # SSA begins for while statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ip' (line 47)
    ip_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 28), 'ip')
    # Getting the type of 'prog' (line 47)
    prog_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'prog')
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 23), prog_144, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_146 = invoke(stypy.reporting.localization.Localization(__file__, 47, 23), getitem___145, ip_143)
    
    str_147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 35), 'str', '[')
    # Applying the binary operator '==' (line 47)
    result_eq_148 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 23), '==', subscript_call_result_146, str_147)
    
    # Testing the type of an if condition (line 47)
    if_condition_149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 20), result_eq_148)
    # Assigning a type to the variable 'if_condition_149' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'if_condition_149', if_condition_149)
    # SSA begins for if statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'level' (line 48)
    level_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'level')
    int_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'int')
    # Applying the binary operator '+=' (line 48)
    result_iadd_152 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 24), '+=', level_150, int_151)
    # Assigning a type to the variable 'level' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'level', result_iadd_152)
    
    # SSA join for if statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ip' (line 49)
    ip_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 28), 'ip')
    # Getting the type of 'prog' (line 49)
    prog_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'prog')
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 23), prog_154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 49, 23), getitem___155, ip_153)
    
    str_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 35), 'str', ']')
    # Applying the binary operator '==' (line 49)
    result_eq_158 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 23), '==', subscript_call_result_156, str_157)
    
    # Testing the type of an if condition (line 49)
    if_condition_159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 20), result_eq_158)
    # Assigning a type to the variable 'if_condition_159' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'if_condition_159', if_condition_159)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'level' (line 50)
    level_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'level')
    # Testing the type of an if condition (line 50)
    if_condition_161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 24), level_160)
    # Assigning a type to the variable 'if_condition_161' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'if_condition_161', if_condition_161)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'level' (line 51)
    level_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'level')
    int_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 37), 'int')
    # Applying the binary operator '-=' (line 51)
    result_isub_164 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 28), '-=', level_162, int_163)
    # Assigning a type to the variable 'level' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'level', result_isub_164)
    
    # SSA branch for the else part of an if statement (line 50)
    module_type_store.open_ssa_branch('else')
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ip' (line 54)
    ip_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'ip')
    int_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'int')
    # Applying the binary operator '+=' (line 54)
    result_iadd_167 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 20), '+=', ip_165, int_166)
    # Assigning a type to the variable 'ip' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'ip', result_iadd_167)
    
    # SSA join for while statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ip' (line 55)
    ip_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'ip')
    int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'int')
    # Applying the binary operator '+=' (line 55)
    result_iadd_170 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 16), '+=', ip_168, int_169)
    # Assigning a type to the variable 'ip' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'ip', result_iadd_170)
    
    # SSA join for if statement (line 45)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 44)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'x' (line 56)
    x_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 13), 'x')
    str_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 18), 'str', ']')
    # Applying the binary operator '==' (line 56)
    result_eq_173 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 13), '==', x_171, str_172)
    
    # Testing the type of an if condition (line 56)
    if_condition_174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 13), result_eq_173)
    # Assigning a type to the variable 'if_condition_174' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 13), 'if_condition_174', if_condition_174)
    # SSA begins for if statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ip' (line 57)
    ip_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'ip')
    int_176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 18), 'int')
    # Applying the binary operator '-=' (line 57)
    result_isub_177 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 12), '-=', ip_175, int_176)
    # Assigning a type to the variable 'ip' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'ip', result_isub_177)
    
    
    # Getting the type of 'True' (line 58)
    True_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'True')
    # Testing the type of an if condition (line 58)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 12), True_178)
    # SSA begins for while statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ip' (line 59)
    ip_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'ip')
    # Getting the type of 'prog' (line 59)
    prog_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'prog')
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 19), prog_180, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 59, 19), getitem___181, ip_179)
    
    str_183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'str', ']')
    # Applying the binary operator '==' (line 59)
    result_eq_184 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 19), '==', subscript_call_result_182, str_183)
    
    # Testing the type of an if condition (line 59)
    if_condition_185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 16), result_eq_184)
    # Assigning a type to the variable 'if_condition_185' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'if_condition_185', if_condition_185)
    # SSA begins for if statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'level' (line 60)
    level_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'level')
    int_187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'int')
    # Applying the binary operator '+=' (line 60)
    result_iadd_188 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), '+=', level_186, int_187)
    # Assigning a type to the variable 'level' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'level', result_iadd_188)
    
    # SSA join for if statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ip' (line 61)
    ip_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'ip')
    # Getting the type of 'prog' (line 61)
    prog_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'prog')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 19), prog_190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_192 = invoke(stypy.reporting.localization.Localization(__file__, 61, 19), getitem___191, ip_189)
    
    str_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 31), 'str', '[')
    # Applying the binary operator '==' (line 61)
    result_eq_194 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 19), '==', subscript_call_result_192, str_193)
    
    # Testing the type of an if condition (line 61)
    if_condition_195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 16), result_eq_194)
    # Assigning a type to the variable 'if_condition_195' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'if_condition_195', if_condition_195)
    # SSA begins for if statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'level' (line 62)
    level_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'level')
    # Testing the type of an if condition (line 62)
    if_condition_197 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 20), level_196)
    # Assigning a type to the variable 'if_condition_197' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'if_condition_197', if_condition_197)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'level' (line 63)
    level_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'level')
    int_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 33), 'int')
    # Applying the binary operator '-=' (line 63)
    result_isub_200 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 24), '-=', level_198, int_199)
    # Assigning a type to the variable 'level' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'level', result_isub_200)
    
    # SSA branch for the else part of an if statement (line 62)
    module_type_store.open_ssa_branch('else')
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ip' (line 66)
    ip_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'ip')
    int_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 22), 'int')
    # Applying the binary operator '-=' (line 66)
    result_isub_203 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 16), '-=', ip_201, int_202)
    # Assigning a type to the variable 'ip' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'ip', result_isub_203)
    
    # SSA join for while statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 56)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 29)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 27)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 21)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'BF_interpreter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'BF_interpreter' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'BF_interpreter'
    return stypy_return_type_204

# Assigning a type to the variable 'BF_interpreter' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'BF_interpreter', BF_interpreter)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 68, 0, False)
    
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

    
    # Assigning a Call to a Name (line 69):
    
    # Call to read(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_213 = {}
    
    # Call to file(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Call to Relative(...): (line 69)
    # Processing the call arguments (line 69)
    str_207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'str', 'testdata/99bottles.bf')
    # Processing the call keyword arguments (line 69)
    kwargs_208 = {}
    # Getting the type of 'Relative' (line 69)
    Relative_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'Relative', False)
    # Calling Relative(args, kwargs) (line 69)
    Relative_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 69, 19), Relative_206, *[str_207], **kwargs_208)
    
    # Processing the call keyword arguments (line 69)
    kwargs_210 = {}
    # Getting the type of 'file' (line 69)
    file_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'file', False)
    # Calling file(args, kwargs) (line 69)
    file_call_result_211 = invoke(stypy.reporting.localization.Localization(__file__, 69, 14), file_205, *[Relative_call_result_209], **kwargs_210)
    
    # Obtaining the member 'read' of a type (line 69)
    read_212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 14), file_call_result_211, 'read')
    # Calling read(args, kwargs) (line 69)
    read_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 69, 14), read_212, *[], **kwargs_213)
    
    # Assigning a type to the variable 'program' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'program', read_call_result_214)
    
    # Call to BF_interpreter(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'program' (line 70)
    program_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'program', False)
    # Processing the call keyword arguments (line 70)
    kwargs_217 = {}
    # Getting the type of 'BF_interpreter' (line 70)
    BF_interpreter_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'BF_interpreter', False)
    # Calling BF_interpreter(args, kwargs) (line 70)
    BF_interpreter_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), BF_interpreter_215, *[program_216], **kwargs_217)
    
    # Getting the type of 'True' (line 71)
    True_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type', True_219)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_220)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_220

# Assigning a type to the variable 'run' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'run', run)

# Call to run(...): (line 73)
# Processing the call keyword arguments (line 73)
kwargs_222 = {}
# Getting the type of 'run' (line 73)
run_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'run', False)
# Calling run(args, kwargs) (line 73)
run_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 73, 0), run_221, *[], **kwargs_222)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
