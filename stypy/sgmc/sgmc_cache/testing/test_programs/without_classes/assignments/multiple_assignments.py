
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: r = "hi"
2: a, b = 5, 3
3: c = 4, 5
4: (m, n, o) = (4, 5, 6)
5: (m, n, o) = [4, 5, 6]
6: [x, y, z, r] = [1, 2, 3, 4]
7: (x, y, z, r) = [1, 2, 3, 4]
8: 
9: x1 = x2 = x3 = 5
10: 
11: (r1, r2) = (r3, r4) = (8, 9)
12: [lr1, lr2] = [lr3, lr4] = (13, 14)
13: 
14: (lr1, lr2) = [lr3, lr4] = (113, 114)
15: 
16: 
17: def func():
18:     r = "hi"
19:     a, b = 5, 3
20:     c = 4, 5
21:     (m, n, o) = (4, 5, 6)
22:     (m, n, o) = [4, 5, 6]
23:     [x, y, z, r] = [1, 2, 3, 4]
24:     (x, y, z, r) = [1, 2, 3, 4]
25: 
26:     x1 = x2 = x3 = 5
27: 
28:     (r1, r2) = (r3, r4) = (8, 9)
29:     [lr1, lr2] = [lr3, lr4] = (13, 14)
30: 
31:     (lr1, lr2) = [lr3, lr4] = (113, 114)
32: 
33: func()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 1):

# Assigning a Str to a Name (line 1):
str_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 4), 'str', 'hi')
# Assigning a type to the variable 'r' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'r', str_424)

# Assigning a Tuple to a Tuple (line 2):

# Assigning a Num to a Name (line 2):
int_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 7), 'int')
# Assigning a type to the variable 'tuple_assignment_368' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'tuple_assignment_368', int_425)

# Assigning a Num to a Name (line 2):
int_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'int')
# Assigning a type to the variable 'tuple_assignment_369' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'tuple_assignment_369', int_426)

# Assigning a Name to a Name (line 2):
# Getting the type of 'tuple_assignment_368' (line 2)
tuple_assignment_368_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'tuple_assignment_368')
# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'a', tuple_assignment_368_427)

# Assigning a Name to a Name (line 2):
# Getting the type of 'tuple_assignment_369' (line 2)
tuple_assignment_369_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'tuple_assignment_369')
# Assigning a type to the variable 'b' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 3), 'b', tuple_assignment_369_428)

# Assigning a Tuple to a Name (line 3):

# Assigning a Tuple to a Name (line 3):

# Obtaining an instance of the builtin type 'tuple' (line 3)
tuple_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3)
# Adding element type (line 3)
int_430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 4), tuple_429, int_430)
# Adding element type (line 3)
int_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 4), tuple_429, int_431)

# Assigning a type to the variable 'c' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'c', tuple_429)

# Assigning a Tuple to a Tuple (line 4):

# Assigning a Num to a Name (line 4):
int_432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 13), 'int')
# Assigning a type to the variable 'tuple_assignment_370' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'tuple_assignment_370', int_432)

# Assigning a Num to a Name (line 4):
int_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 16), 'int')
# Assigning a type to the variable 'tuple_assignment_371' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'tuple_assignment_371', int_433)

# Assigning a Num to a Name (line 4):
int_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 19), 'int')
# Assigning a type to the variable 'tuple_assignment_372' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'tuple_assignment_372', int_434)

# Assigning a Name to a Name (line 4):
# Getting the type of 'tuple_assignment_370' (line 4)
tuple_assignment_370_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'tuple_assignment_370')
# Assigning a type to the variable 'm' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 1), 'm', tuple_assignment_370_435)

# Assigning a Name to a Name (line 4):
# Getting the type of 'tuple_assignment_371' (line 4)
tuple_assignment_371_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'tuple_assignment_371')
# Assigning a type to the variable 'n' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'n', tuple_assignment_371_436)

# Assigning a Name to a Name (line 4):
# Getting the type of 'tuple_assignment_372' (line 4)
tuple_assignment_372_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'tuple_assignment_372')
# Assigning a type to the variable 'o' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 7), 'o', tuple_assignment_372_437)

# Assigning a List to a Tuple (line 5):

# Assigning a Num to a Name (line 5):
int_438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'int')
# Assigning a type to the variable 'tuple_assignment_373' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_373', int_438)

# Assigning a Num to a Name (line 5):
int_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'int')
# Assigning a type to the variable 'tuple_assignment_374' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_374', int_439)

# Assigning a Num to a Name (line 5):
int_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'int')
# Assigning a type to the variable 'tuple_assignment_375' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_375', int_440)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_assignment_373' (line 5)
tuple_assignment_373_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_373')
# Assigning a type to the variable 'm' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 1), 'm', tuple_assignment_373_441)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_assignment_374' (line 5)
tuple_assignment_374_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_374')
# Assigning a type to the variable 'n' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'n', tuple_assignment_374_442)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_assignment_375' (line 5)
tuple_assignment_375_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_assignment_375')
# Assigning a type to the variable 'o' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 7), 'o', tuple_assignment_375_443)

# Assigning a List to a List (line 6):

# Assigning a Num to a Name (line 6):
int_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 16), 'int')
# Assigning a type to the variable 'list_assignment_376' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'list_assignment_376', int_444)

# Assigning a Num to a Name (line 6):
int_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 19), 'int')
# Assigning a type to the variable 'list_assignment_377' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'list_assignment_377', int_445)

# Assigning a Num to a Name (line 6):
int_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'int')
# Assigning a type to the variable 'list_assignment_378' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'list_assignment_378', int_446)

# Assigning a Num to a Name (line 6):
int_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'int')
# Assigning a type to the variable 'list_assignment_379' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'list_assignment_379', int_447)

# Assigning a Name to a Name (line 6):
# Getting the type of 'list_assignment_376' (line 6)
list_assignment_376_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'list_assignment_376')
# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 1), 'x', list_assignment_376_448)

# Assigning a Name to a Name (line 6):
# Getting the type of 'list_assignment_377' (line 6)
list_assignment_377_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'list_assignment_377')
# Assigning a type to the variable 'y' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'y', list_assignment_377_449)

# Assigning a Name to a Name (line 6):
# Getting the type of 'list_assignment_378' (line 6)
list_assignment_378_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'list_assignment_378')
# Assigning a type to the variable 'z' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'z', list_assignment_378_450)

# Assigning a Name to a Name (line 6):
# Getting the type of 'list_assignment_379' (line 6)
list_assignment_379_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'list_assignment_379')
# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 10), 'r', list_assignment_379_451)

# Assigning a List to a Tuple (line 7):

# Assigning a Num to a Name (line 7):
int_452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 16), 'int')
# Assigning a type to the variable 'tuple_assignment_380' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_380', int_452)

# Assigning a Num to a Name (line 7):
int_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'int')
# Assigning a type to the variable 'tuple_assignment_381' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_381', int_453)

# Assigning a Num to a Name (line 7):
int_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 22), 'int')
# Assigning a type to the variable 'tuple_assignment_382' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_382', int_454)

# Assigning a Num to a Name (line 7):
int_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'int')
# Assigning a type to the variable 'tuple_assignment_383' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_383', int_455)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_assignment_380' (line 7)
tuple_assignment_380_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_380')
# Assigning a type to the variable 'x' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 1), 'x', tuple_assignment_380_456)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_assignment_381' (line 7)
tuple_assignment_381_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_381')
# Assigning a type to the variable 'y' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'y', tuple_assignment_381_457)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_assignment_382' (line 7)
tuple_assignment_382_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_382')
# Assigning a type to the variable 'z' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 7), 'z', tuple_assignment_382_458)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_assignment_383' (line 7)
tuple_assignment_383_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_383')
# Assigning a type to the variable 'r' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'r', tuple_assignment_383_459)

# Multiple assignment of 3 elements.

# Assigning a Num to a Name (line 9):
int_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'int')
# Assigning a type to the variable 'x3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'x3', int_460)

# Assigning a Name to a Name (line 9):
# Getting the type of 'x3' (line 9)
x3_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'x3')
# Assigning a type to the variable 'x2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'x2', x3_461)

# Assigning a Name to a Name (line 9):
# Getting the type of 'x2' (line 9)
x2_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'x2')
# Assigning a type to the variable 'x1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'x1', x2_462)

# Multiple assignment of 2 elements.

# Assigning a Num to a Name (line 11):
int_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'int')
# Assigning a type to the variable 'tuple_assignment_384' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_assignment_384', int_463)

# Assigning a Num to a Name (line 11):
int_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'int')
# Assigning a type to the variable 'tuple_assignment_385' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_assignment_385', int_464)

# Assigning a Name to a Name (line 11):
# Getting the type of 'tuple_assignment_384' (line 11)
tuple_assignment_384_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_assignment_384')
# Assigning a type to the variable 'r3' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'r3', tuple_assignment_384_465)

# Assigning a Name to a Name (line 11):
# Getting the type of 'tuple_assignment_385' (line 11)
tuple_assignment_385_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_assignment_385')
# Assigning a type to the variable 'r4' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'r4', tuple_assignment_385_466)

# Assigning a Name to a Name (line 11):
# Getting the type of 'r3' (line 11)
r3_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'r3')
# Assigning a type to the variable 'tuple_assignment_386' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_assignment_386', r3_467)

# Assigning a Name to a Name (line 11):
# Getting the type of 'r4' (line 11)
r4_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'r4')
# Assigning a type to the variable 'tuple_assignment_387' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_assignment_387', r4_468)

# Assigning a Name to a Name (line 11):
# Getting the type of 'tuple_assignment_386' (line 11)
tuple_assignment_386_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_assignment_386')
# Assigning a type to the variable 'r1' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 1), 'r1', tuple_assignment_386_469)

# Assigning a Name to a Name (line 11):
# Getting the type of 'tuple_assignment_387' (line 11)
tuple_assignment_387_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_assignment_387')
# Assigning a type to the variable 'r2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'r2', tuple_assignment_387_470)

# Multiple assignment of 2 elements.

# Assigning a Num to a Name (line 12):
int_471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 27), 'int')
# Assigning a type to the variable 'list_assignment_388' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'list_assignment_388', int_471)

# Assigning a Num to a Name (line 12):
int_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'int')
# Assigning a type to the variable 'list_assignment_389' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'list_assignment_389', int_472)

# Assigning a Name to a Name (line 12):
# Getting the type of 'list_assignment_388' (line 12)
list_assignment_388_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'list_assignment_388')
# Assigning a type to the variable 'lr3' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'lr3', list_assignment_388_473)

# Assigning a Name to a Name (line 12):
# Getting the type of 'list_assignment_389' (line 12)
list_assignment_389_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'list_assignment_389')
# Assigning a type to the variable 'lr4' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'lr4', list_assignment_389_474)

# Assigning a Name to a Name (line 12):
# Getting the type of 'lr3' (line 12)
lr3_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'lr3')
# Assigning a type to the variable 'list_assignment_390' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'list_assignment_390', lr3_475)

# Assigning a Name to a Name (line 12):
# Getting the type of 'lr4' (line 12)
lr4_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'lr4')
# Assigning a type to the variable 'list_assignment_391' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'list_assignment_391', lr4_476)

# Assigning a Name to a Name (line 12):
# Getting the type of 'list_assignment_390' (line 12)
list_assignment_390_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'list_assignment_390')
# Assigning a type to the variable 'lr1' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 1), 'lr1', list_assignment_390_477)

# Assigning a Name to a Name (line 12):
# Getting the type of 'list_assignment_391' (line 12)
list_assignment_391_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'list_assignment_391')
# Assigning a type to the variable 'lr2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 6), 'lr2', list_assignment_391_478)

# Multiple assignment of 2 elements.

# Assigning a Num to a Name (line 14):
int_479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'int')
# Assigning a type to the variable 'list_assignment_392' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'list_assignment_392', int_479)

# Assigning a Num to a Name (line 14):
int_480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 32), 'int')
# Assigning a type to the variable 'list_assignment_393' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'list_assignment_393', int_480)

# Assigning a Name to a Name (line 14):
# Getting the type of 'list_assignment_392' (line 14)
list_assignment_392_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'list_assignment_392')
# Assigning a type to the variable 'lr3' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'lr3', list_assignment_392_481)

# Assigning a Name to a Name (line 14):
# Getting the type of 'list_assignment_393' (line 14)
list_assignment_393_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'list_assignment_393')
# Assigning a type to the variable 'lr4' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'lr4', list_assignment_393_482)

# Assigning a Name to a Name (line 14):
# Getting the type of 'lr3' (line 14)
lr3_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'lr3')
# Assigning a type to the variable 'tuple_assignment_394' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'tuple_assignment_394', lr3_483)

# Assigning a Name to a Name (line 14):
# Getting the type of 'lr4' (line 14)
lr4_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'lr4')
# Assigning a type to the variable 'tuple_assignment_395' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'tuple_assignment_395', lr4_484)

# Assigning a Name to a Name (line 14):
# Getting the type of 'tuple_assignment_394' (line 14)
tuple_assignment_394_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'tuple_assignment_394')
# Assigning a type to the variable 'lr1' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 1), 'lr1', tuple_assignment_394_485)

# Assigning a Name to a Name (line 14):
# Getting the type of 'tuple_assignment_395' (line 14)
tuple_assignment_395_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'tuple_assignment_395')
# Assigning a type to the variable 'lr2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 6), 'lr2', tuple_assignment_395_486)

@norecursion
def func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'func'
    module_type_store = module_type_store.open_function_context('func', 17, 0, False)
    
    # Passed parameters checking function
    func.stypy_localization = localization
    func.stypy_type_of_self = None
    func.stypy_type_store = module_type_store
    func.stypy_function_name = 'func'
    func.stypy_param_names_list = []
    func.stypy_varargs_param_name = None
    func.stypy_kwargs_param_name = None
    func.stypy_call_defaults = defaults
    func.stypy_call_varargs = varargs
    func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'func', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'func', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'func(...)' code ##################

    
    # Assigning a Str to a Name (line 18):
    
    # Assigning a Str to a Name (line 18):
    str_487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'str', 'hi')
    # Assigning a type to the variable 'r' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'r', str_487)
    
    # Assigning a Tuple to a Tuple (line 19):
    
    # Assigning a Num to a Name (line 19):
    int_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'int')
    # Assigning a type to the variable 'tuple_assignment_396' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'tuple_assignment_396', int_488)
    
    # Assigning a Num to a Name (line 19):
    int_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 14), 'int')
    # Assigning a type to the variable 'tuple_assignment_397' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'tuple_assignment_397', int_489)
    
    # Assigning a Name to a Name (line 19):
    # Getting the type of 'tuple_assignment_396' (line 19)
    tuple_assignment_396_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'tuple_assignment_396')
    # Assigning a type to the variable 'a' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'a', tuple_assignment_396_490)
    
    # Assigning a Name to a Name (line 19):
    # Getting the type of 'tuple_assignment_397' (line 19)
    tuple_assignment_397_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'tuple_assignment_397')
    # Assigning a type to the variable 'b' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 7), 'b', tuple_assignment_397_491)
    
    # Assigning a Tuple to a Name (line 20):
    
    # Assigning a Tuple to a Name (line 20):
    
    # Obtaining an instance of the builtin type 'tuple' (line 20)
    tuple_492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 20)
    # Adding element type (line 20)
    int_493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), tuple_492, int_493)
    # Adding element type (line 20)
    int_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), tuple_492, int_494)
    
    # Assigning a type to the variable 'c' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'c', tuple_492)
    
    # Assigning a Tuple to a Tuple (line 21):
    
    # Assigning a Num to a Name (line 21):
    int_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 17), 'int')
    # Assigning a type to the variable 'tuple_assignment_398' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_assignment_398', int_495)
    
    # Assigning a Num to a Name (line 21):
    int_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'int')
    # Assigning a type to the variable 'tuple_assignment_399' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_assignment_399', int_496)
    
    # Assigning a Num to a Name (line 21):
    int_497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'int')
    # Assigning a type to the variable 'tuple_assignment_400' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_assignment_400', int_497)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'tuple_assignment_398' (line 21)
    tuple_assignment_398_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_assignment_398')
    # Assigning a type to the variable 'm' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'm', tuple_assignment_398_498)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'tuple_assignment_399' (line 21)
    tuple_assignment_399_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_assignment_399')
    # Assigning a type to the variable 'n' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'n', tuple_assignment_399_499)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'tuple_assignment_400' (line 21)
    tuple_assignment_400_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_assignment_400')
    # Assigning a type to the variable 'o' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'o', tuple_assignment_400_500)
    
    # Assigning a List to a Tuple (line 22):
    
    # Assigning a Num to a Name (line 22):
    int_501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
    # Assigning a type to the variable 'tuple_assignment_401' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'tuple_assignment_401', int_501)
    
    # Assigning a Num to a Name (line 22):
    int_502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'int')
    # Assigning a type to the variable 'tuple_assignment_402' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'tuple_assignment_402', int_502)
    
    # Assigning a Num to a Name (line 22):
    int_503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'int')
    # Assigning a type to the variable 'tuple_assignment_403' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'tuple_assignment_403', int_503)
    
    # Assigning a Name to a Name (line 22):
    # Getting the type of 'tuple_assignment_401' (line 22)
    tuple_assignment_401_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'tuple_assignment_401')
    # Assigning a type to the variable 'm' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'm', tuple_assignment_401_504)
    
    # Assigning a Name to a Name (line 22):
    # Getting the type of 'tuple_assignment_402' (line 22)
    tuple_assignment_402_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'tuple_assignment_402')
    # Assigning a type to the variable 'n' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'n', tuple_assignment_402_505)
    
    # Assigning a Name to a Name (line 22):
    # Getting the type of 'tuple_assignment_403' (line 22)
    tuple_assignment_403_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'tuple_assignment_403')
    # Assigning a type to the variable 'o' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'o', tuple_assignment_403_506)
    
    # Assigning a List to a List (line 23):
    
    # Assigning a Num to a Name (line 23):
    int_507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'int')
    # Assigning a type to the variable 'list_assignment_404' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'list_assignment_404', int_507)
    
    # Assigning a Num to a Name (line 23):
    int_508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'int')
    # Assigning a type to the variable 'list_assignment_405' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'list_assignment_405', int_508)
    
    # Assigning a Num to a Name (line 23):
    int_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'int')
    # Assigning a type to the variable 'list_assignment_406' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'list_assignment_406', int_509)
    
    # Assigning a Num to a Name (line 23):
    int_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'int')
    # Assigning a type to the variable 'list_assignment_407' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'list_assignment_407', int_510)
    
    # Assigning a Name to a Name (line 23):
    # Getting the type of 'list_assignment_404' (line 23)
    list_assignment_404_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'list_assignment_404')
    # Assigning a type to the variable 'x' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'x', list_assignment_404_511)
    
    # Assigning a Name to a Name (line 23):
    # Getting the type of 'list_assignment_405' (line 23)
    list_assignment_405_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'list_assignment_405')
    # Assigning a type to the variable 'y' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'y', list_assignment_405_512)
    
    # Assigning a Name to a Name (line 23):
    # Getting the type of 'list_assignment_406' (line 23)
    list_assignment_406_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'list_assignment_406')
    # Assigning a type to the variable 'z' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'z', list_assignment_406_513)
    
    # Assigning a Name to a Name (line 23):
    # Getting the type of 'list_assignment_407' (line 23)
    list_assignment_407_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'list_assignment_407')
    # Assigning a type to the variable 'r' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'r', list_assignment_407_514)
    
    # Assigning a List to a Tuple (line 24):
    
    # Assigning a Num to a Name (line 24):
    int_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'int')
    # Assigning a type to the variable 'tuple_assignment_408' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_assignment_408', int_515)
    
    # Assigning a Num to a Name (line 24):
    int_516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'int')
    # Assigning a type to the variable 'tuple_assignment_409' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_assignment_409', int_516)
    
    # Assigning a Num to a Name (line 24):
    int_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'int')
    # Assigning a type to the variable 'tuple_assignment_410' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_assignment_410', int_517)
    
    # Assigning a Num to a Name (line 24):
    int_518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'int')
    # Assigning a type to the variable 'tuple_assignment_411' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_assignment_411', int_518)
    
    # Assigning a Name to a Name (line 24):
    # Getting the type of 'tuple_assignment_408' (line 24)
    tuple_assignment_408_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_assignment_408')
    # Assigning a type to the variable 'x' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 5), 'x', tuple_assignment_408_519)
    
    # Assigning a Name to a Name (line 24):
    # Getting the type of 'tuple_assignment_409' (line 24)
    tuple_assignment_409_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_assignment_409')
    # Assigning a type to the variable 'y' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'y', tuple_assignment_409_520)
    
    # Assigning a Name to a Name (line 24):
    # Getting the type of 'tuple_assignment_410' (line 24)
    tuple_assignment_410_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_assignment_410')
    # Assigning a type to the variable 'z' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'z', tuple_assignment_410_521)
    
    # Assigning a Name to a Name (line 24):
    # Getting the type of 'tuple_assignment_411' (line 24)
    tuple_assignment_411_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_assignment_411')
    # Assigning a type to the variable 'r' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'r', tuple_assignment_411_522)
    
    # Multiple assignment of 3 elements.
    
    # Assigning a Num to a Name (line 26):
    int_523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'int')
    # Assigning a type to the variable 'x3' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'x3', int_523)
    
    # Assigning a Name to a Name (line 26):
    # Getting the type of 'x3' (line 26)
    x3_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'x3')
    # Assigning a type to the variable 'x2' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'x2', x3_524)
    
    # Assigning a Name to a Name (line 26):
    # Getting the type of 'x2' (line 26)
    x2_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'x2')
    # Assigning a type to the variable 'x1' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'x1', x2_525)
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Num to a Name (line 28):
    int_526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 27), 'int')
    # Assigning a type to the variable 'tuple_assignment_412' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_assignment_412', int_526)
    
    # Assigning a Num to a Name (line 28):
    int_527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 30), 'int')
    # Assigning a type to the variable 'tuple_assignment_413' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_assignment_413', int_527)
    
    # Assigning a Name to a Name (line 28):
    # Getting the type of 'tuple_assignment_412' (line 28)
    tuple_assignment_412_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_assignment_412')
    # Assigning a type to the variable 'r3' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'r3', tuple_assignment_412_528)
    
    # Assigning a Name to a Name (line 28):
    # Getting the type of 'tuple_assignment_413' (line 28)
    tuple_assignment_413_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_assignment_413')
    # Assigning a type to the variable 'r4' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'r4', tuple_assignment_413_529)
    
    # Assigning a Name to a Name (line 28):
    # Getting the type of 'r3' (line 28)
    r3_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'r3')
    # Assigning a type to the variable 'tuple_assignment_414' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_assignment_414', r3_530)
    
    # Assigning a Name to a Name (line 28):
    # Getting the type of 'r4' (line 28)
    r4_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'r4')
    # Assigning a type to the variable 'tuple_assignment_415' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_assignment_415', r4_531)
    
    # Assigning a Name to a Name (line 28):
    # Getting the type of 'tuple_assignment_414' (line 28)
    tuple_assignment_414_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_assignment_414')
    # Assigning a type to the variable 'r1' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 5), 'r1', tuple_assignment_414_532)
    
    # Assigning a Name to a Name (line 28):
    # Getting the type of 'tuple_assignment_415' (line 28)
    tuple_assignment_415_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_assignment_415')
    # Assigning a type to the variable 'r2' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 9), 'r2', tuple_assignment_415_533)
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Num to a Name (line 29):
    int_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 31), 'int')
    # Assigning a type to the variable 'list_assignment_416' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'list_assignment_416', int_534)
    
    # Assigning a Num to a Name (line 29):
    int_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 35), 'int')
    # Assigning a type to the variable 'list_assignment_417' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'list_assignment_417', int_535)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'list_assignment_416' (line 29)
    list_assignment_416_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'list_assignment_416')
    # Assigning a type to the variable 'lr3' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'lr3', list_assignment_416_536)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'list_assignment_417' (line 29)
    list_assignment_417_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'list_assignment_417')
    # Assigning a type to the variable 'lr4' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'lr4', list_assignment_417_537)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'lr3' (line 29)
    lr3_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'lr3')
    # Assigning a type to the variable 'list_assignment_418' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'list_assignment_418', lr3_538)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'lr4' (line 29)
    lr4_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'lr4')
    # Assigning a type to the variable 'list_assignment_419' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'list_assignment_419', lr4_539)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'list_assignment_418' (line 29)
    list_assignment_418_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'list_assignment_418')
    # Assigning a type to the variable 'lr1' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 5), 'lr1', list_assignment_418_540)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'list_assignment_419' (line 29)
    list_assignment_419_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'list_assignment_419')
    # Assigning a type to the variable 'lr2' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'lr2', list_assignment_419_541)
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Num to a Name (line 31):
    int_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'int')
    # Assigning a type to the variable 'list_assignment_420' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'list_assignment_420', int_542)
    
    # Assigning a Num to a Name (line 31):
    int_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'int')
    # Assigning a type to the variable 'list_assignment_421' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'list_assignment_421', int_543)
    
    # Assigning a Name to a Name (line 31):
    # Getting the type of 'list_assignment_420' (line 31)
    list_assignment_420_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'list_assignment_420')
    # Assigning a type to the variable 'lr3' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'lr3', list_assignment_420_544)
    
    # Assigning a Name to a Name (line 31):
    # Getting the type of 'list_assignment_421' (line 31)
    list_assignment_421_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'list_assignment_421')
    # Assigning a type to the variable 'lr4' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'lr4', list_assignment_421_545)
    
    # Assigning a Name to a Name (line 31):
    # Getting the type of 'lr3' (line 31)
    lr3_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'lr3')
    # Assigning a type to the variable 'tuple_assignment_422' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'tuple_assignment_422', lr3_546)
    
    # Assigning a Name to a Name (line 31):
    # Getting the type of 'lr4' (line 31)
    lr4_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'lr4')
    # Assigning a type to the variable 'tuple_assignment_423' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'tuple_assignment_423', lr4_547)
    
    # Assigning a Name to a Name (line 31):
    # Getting the type of 'tuple_assignment_422' (line 31)
    tuple_assignment_422_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'tuple_assignment_422')
    # Assigning a type to the variable 'lr1' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 5), 'lr1', tuple_assignment_422_548)
    
    # Assigning a Name to a Name (line 31):
    # Getting the type of 'tuple_assignment_423' (line 31)
    tuple_assignment_423_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'tuple_assignment_423')
    # Assigning a type to the variable 'lr2' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'lr2', tuple_assignment_423_549)
    
    # ################# End of 'func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'func' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_550)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'func'
    return stypy_return_type_550

# Assigning a type to the variable 'func' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'func', func)

# Call to func(...): (line 33)
# Processing the call keyword arguments (line 33)
kwargs_552 = {}
# Getting the type of 'func' (line 33)
func_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'func', False)
# Calling func(args, kwargs) (line 33)
func_call_result_553 = invoke(stypy.reporting.localization.Localization(__file__, 33, 0), func_551, *[], **kwargs_552)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
