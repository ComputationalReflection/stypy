
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import math
2: 
3: 
4: class Foo:
5:     def __init__(self):
6:         pass
7: 
8: 
9: r = sum(x * x for x in range(10))
10: 
11: err = r.nothing()  # Unreported
12: 
13: r2 = abs(3)
14: r3 = abs("3")  # Unreported: Parameter types are not checked
15: r4 = abs(2, 3)  # Reported: Arities are checked
16: 
17: r4.nothing()  # Unreported
18: 
19: r5 = abs(3)
20: err2 = r5.nothing()  # Unreported
21: 
22: r6 = all(3)  # Reported
23: r7 = all([3])
24: err3 = r7.nothing()  # Reported
25: 
26: r8 = bytearray(Foo())  # Unreported
27: err4 = r8.nothing()  # Reported
28: 
29: words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
30: S = [x ** 2 for x in range(10)]
31: 
32: err5 = math.fsum(words)  # Reported
33: r9 = math.fsum(S)
34: err6 = r9.nothing()  # Reported
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import math' statement (line 1)
import math

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', math, module_type_store)

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 5, 4, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Foo' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'Foo', Foo)

# Assigning a Call to a Name (line 9):

# Call to sum(...): (line 9)
# Processing the call arguments (line 9)
# Calculating generator expression
module_type_store = module_type_store.open_function_context('list comprehension expression', 9, 8, True)
# Calculating comprehension expression

# Call to range(...): (line 9)
# Processing the call arguments (line 9)
int_7004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 29), 'int')
# Processing the call keyword arguments (line 9)
kwargs_7005 = {}
# Getting the type of 'range' (line 9)
range_7003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'range', False)
# Calling range(args, kwargs) (line 9)
range_call_result_7006 = invoke(stypy.reporting.localization.Localization(__file__, 9, 23), range_7003, *[int_7004], **kwargs_7005)

comprehension_7007 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 8), range_call_result_7006)
# Assigning a type to the variable 'x' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'x', comprehension_7007)
# Getting the type of 'x' (line 9)
x_7000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'x', False)
# Getting the type of 'x' (line 9)
x_7001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'x', False)
# Applying the binary operator '*' (line 9)
result_mul_7002 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 8), '*', x_7000, x_7001)

list_7008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'list')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 8), list_7008, result_mul_7002)
# Processing the call keyword arguments (line 9)
kwargs_7009 = {}
# Getting the type of 'sum' (line 9)
sum_6999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'sum', False)
# Calling sum(args, kwargs) (line 9)
sum_call_result_7010 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), sum_6999, *[list_7008], **kwargs_7009)

# Assigning a type to the variable 'r' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r', sum_call_result_7010)

# Assigning a Call to a Name (line 11):

# Call to nothing(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_7013 = {}
# Getting the type of 'r' (line 11)
r_7011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 6), 'r', False)
# Obtaining the member 'nothing' of a type (line 11)
nothing_7012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 6), r_7011, 'nothing')
# Calling nothing(args, kwargs) (line 11)
nothing_call_result_7014 = invoke(stypy.reporting.localization.Localization(__file__, 11, 6), nothing_7012, *[], **kwargs_7013)

# Assigning a type to the variable 'err' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'err', nothing_call_result_7014)

# Assigning a Call to a Name (line 13):

# Call to abs(...): (line 13)
# Processing the call arguments (line 13)
int_7016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'int')
# Processing the call keyword arguments (line 13)
kwargs_7017 = {}
# Getting the type of 'abs' (line 13)
abs_7015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'abs', False)
# Calling abs(args, kwargs) (line 13)
abs_call_result_7018 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), abs_7015, *[int_7016], **kwargs_7017)

# Assigning a type to the variable 'r2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r2', abs_call_result_7018)

# Assigning a Call to a Name (line 14):

# Call to abs(...): (line 14)
# Processing the call arguments (line 14)
str_7020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 9), 'str', '3')
# Processing the call keyword arguments (line 14)
kwargs_7021 = {}
# Getting the type of 'abs' (line 14)
abs_7019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'abs', False)
# Calling abs(args, kwargs) (line 14)
abs_call_result_7022 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), abs_7019, *[str_7020], **kwargs_7021)

# Assigning a type to the variable 'r3' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r3', abs_call_result_7022)

# Assigning a Call to a Name (line 15):

# Call to abs(...): (line 15)
# Processing the call arguments (line 15)
int_7024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 9), 'int')
int_7025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'int')
# Processing the call keyword arguments (line 15)
kwargs_7026 = {}
# Getting the type of 'abs' (line 15)
abs_7023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'abs', False)
# Calling abs(args, kwargs) (line 15)
abs_call_result_7027 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), abs_7023, *[int_7024, int_7025], **kwargs_7026)

# Assigning a type to the variable 'r4' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r4', abs_call_result_7027)

# Call to nothing(...): (line 17)
# Processing the call keyword arguments (line 17)
kwargs_7030 = {}
# Getting the type of 'r4' (line 17)
r4_7028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r4', False)
# Obtaining the member 'nothing' of a type (line 17)
nothing_7029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 0), r4_7028, 'nothing')
# Calling nothing(args, kwargs) (line 17)
nothing_call_result_7031 = invoke(stypy.reporting.localization.Localization(__file__, 17, 0), nothing_7029, *[], **kwargs_7030)


# Assigning a Call to a Name (line 19):

# Call to abs(...): (line 19)
# Processing the call arguments (line 19)
int_7033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'int')
# Processing the call keyword arguments (line 19)
kwargs_7034 = {}
# Getting the type of 'abs' (line 19)
abs_7032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'abs', False)
# Calling abs(args, kwargs) (line 19)
abs_call_result_7035 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), abs_7032, *[int_7033], **kwargs_7034)

# Assigning a type to the variable 'r5' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r5', abs_call_result_7035)

# Assigning a Call to a Name (line 20):

# Call to nothing(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_7038 = {}
# Getting the type of 'r5' (line 20)
r5_7036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'r5', False)
# Obtaining the member 'nothing' of a type (line 20)
nothing_7037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 7), r5_7036, 'nothing')
# Calling nothing(args, kwargs) (line 20)
nothing_call_result_7039 = invoke(stypy.reporting.localization.Localization(__file__, 20, 7), nothing_7037, *[], **kwargs_7038)

# Assigning a type to the variable 'err2' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'err2', nothing_call_result_7039)

# Assigning a Call to a Name (line 22):

# Call to all(...): (line 22)
# Processing the call arguments (line 22)
int_7041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'int')
# Processing the call keyword arguments (line 22)
kwargs_7042 = {}
# Getting the type of 'all' (line 22)
all_7040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'all', False)
# Calling all(args, kwargs) (line 22)
all_call_result_7043 = invoke(stypy.reporting.localization.Localization(__file__, 22, 5), all_7040, *[int_7041], **kwargs_7042)

# Assigning a type to the variable 'r6' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r6', all_call_result_7043)

# Assigning a Call to a Name (line 23):

# Call to all(...): (line 23)
# Processing the call arguments (line 23)

# Obtaining an instance of the builtin type 'list' (line 23)
list_7045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_7046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), list_7045, int_7046)

# Processing the call keyword arguments (line 23)
kwargs_7047 = {}
# Getting the type of 'all' (line 23)
all_7044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'all', False)
# Calling all(args, kwargs) (line 23)
all_call_result_7048 = invoke(stypy.reporting.localization.Localization(__file__, 23, 5), all_7044, *[list_7045], **kwargs_7047)

# Assigning a type to the variable 'r7' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r7', all_call_result_7048)

# Assigning a Call to a Name (line 24):

# Call to nothing(...): (line 24)
# Processing the call keyword arguments (line 24)
kwargs_7051 = {}
# Getting the type of 'r7' (line 24)
r7_7049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 7), 'r7', False)
# Obtaining the member 'nothing' of a type (line 24)
nothing_7050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 7), r7_7049, 'nothing')
# Calling nothing(args, kwargs) (line 24)
nothing_call_result_7052 = invoke(stypy.reporting.localization.Localization(__file__, 24, 7), nothing_7050, *[], **kwargs_7051)

# Assigning a type to the variable 'err3' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'err3', nothing_call_result_7052)

# Assigning a Call to a Name (line 26):

# Call to bytearray(...): (line 26)
# Processing the call arguments (line 26)

# Call to Foo(...): (line 26)
# Processing the call keyword arguments (line 26)
kwargs_7055 = {}
# Getting the type of 'Foo' (line 26)
Foo_7054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'Foo', False)
# Calling Foo(args, kwargs) (line 26)
Foo_call_result_7056 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), Foo_7054, *[], **kwargs_7055)

# Processing the call keyword arguments (line 26)
kwargs_7057 = {}
# Getting the type of 'bytearray' (line 26)
bytearray_7053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 26)
bytearray_call_result_7058 = invoke(stypy.reporting.localization.Localization(__file__, 26, 5), bytearray_7053, *[Foo_call_result_7056], **kwargs_7057)

# Assigning a type to the variable 'r8' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'r8', bytearray_call_result_7058)

# Assigning a Call to a Name (line 27):

# Call to nothing(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_7061 = {}
# Getting the type of 'r8' (line 27)
r8_7059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'r8', False)
# Obtaining the member 'nothing' of a type (line 27)
nothing_7060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 7), r8_7059, 'nothing')
# Calling nothing(args, kwargs) (line 27)
nothing_call_result_7062 = invoke(stypy.reporting.localization.Localization(__file__, 27, 7), nothing_7060, *[], **kwargs_7061)

# Assigning a type to the variable 'err4' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'err4', nothing_call_result_7062)

# Assigning a List to a Name (line 29):

# Obtaining an instance of the builtin type 'list' (line 29)
list_7063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
str_7064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'str', 'The')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_7063, str_7064)
# Adding element type (line 29)
str_7065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'str', 'quick')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_7063, str_7065)
# Adding element type (line 29)
str_7066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'str', 'brown')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_7063, str_7066)
# Adding element type (line 29)
str_7067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'str', 'fox')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_7063, str_7067)
# Adding element type (line 29)
str_7068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 41), 'str', 'jumps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_7063, str_7068)
# Adding element type (line 29)
str_7069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 50), 'str', 'over')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_7063, str_7069)
# Adding element type (line 29)
str_7070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 58), 'str', 'the')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_7063, str_7070)
# Adding element type (line 29)
str_7071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 65), 'str', 'lazy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_7063, str_7071)
# Adding element type (line 29)
str_7072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 73), 'str', 'dog')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), list_7063, str_7072)

# Assigning a type to the variable 'words' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'words', list_7063)

# Assigning a ListComp to a Name (line 30):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 30)
# Processing the call arguments (line 30)
int_7077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'int')
# Processing the call keyword arguments (line 30)
kwargs_7078 = {}
# Getting the type of 'range' (line 30)
range_7076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'range', False)
# Calling range(args, kwargs) (line 30)
range_call_result_7079 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), range_7076, *[int_7077], **kwargs_7078)

comprehension_7080 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 5), range_call_result_7079)
# Assigning a type to the variable 'x' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 5), 'x', comprehension_7080)
# Getting the type of 'x' (line 30)
x_7073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 5), 'x')
int_7074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'int')
# Applying the binary operator '**' (line 30)
result_pow_7075 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 5), '**', x_7073, int_7074)

list_7081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 5), list_7081, result_pow_7075)
# Assigning a type to the variable 'S' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'S', list_7081)

# Assigning a Call to a Name (line 32):

# Call to fsum(...): (line 32)
# Processing the call arguments (line 32)
# Getting the type of 'words' (line 32)
words_7084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'words', False)
# Processing the call keyword arguments (line 32)
kwargs_7085 = {}
# Getting the type of 'math' (line 32)
math_7082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 7), 'math', False)
# Obtaining the member 'fsum' of a type (line 32)
fsum_7083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 7), math_7082, 'fsum')
# Calling fsum(args, kwargs) (line 32)
fsum_call_result_7086 = invoke(stypy.reporting.localization.Localization(__file__, 32, 7), fsum_7083, *[words_7084], **kwargs_7085)

# Assigning a type to the variable 'err5' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'err5', fsum_call_result_7086)

# Assigning a Call to a Name (line 33):

# Call to fsum(...): (line 33)
# Processing the call arguments (line 33)
# Getting the type of 'S' (line 33)
S_7089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'S', False)
# Processing the call keyword arguments (line 33)
kwargs_7090 = {}
# Getting the type of 'math' (line 33)
math_7087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 5), 'math', False)
# Obtaining the member 'fsum' of a type (line 33)
fsum_7088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 5), math_7087, 'fsum')
# Calling fsum(args, kwargs) (line 33)
fsum_call_result_7091 = invoke(stypy.reporting.localization.Localization(__file__, 33, 5), fsum_7088, *[S_7089], **kwargs_7090)

# Assigning a type to the variable 'r9' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'r9', fsum_call_result_7091)

# Assigning a Call to a Name (line 34):

# Call to nothing(...): (line 34)
# Processing the call keyword arguments (line 34)
kwargs_7094 = {}
# Getting the type of 'r9' (line 34)
r9_7092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 7), 'r9', False)
# Obtaining the member 'nothing' of a type (line 34)
nothing_7093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 7), r9_7092, 'nothing')
# Calling nothing(args, kwargs) (line 34)
nothing_call_result_7095 = invoke(stypy.reporting.localization.Localization(__file__, 34, 7), nothing_7093, *[], **kwargs_7094)

# Assigning a type to the variable 'err6' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'err6', nothing_call_result_7095)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
