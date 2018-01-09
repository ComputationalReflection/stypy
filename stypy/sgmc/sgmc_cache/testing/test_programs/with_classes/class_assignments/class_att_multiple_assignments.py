
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class C:
2:     r = "hi"
3:     a, b = 5, 3
4:     c = 4, 5
5:     (m, n, o) = (4, 5, 6)
6:     (m, n, o) = [4, 5, 6]
7:     [x, y, z, r] = [1, 2, 3, 4]
8:     (x, y, z, r) = [1, 2, 3, 4]
9: 
10:     x1=x2=x3=5
11: 
12:     (r1,r2)=(r3,r4)=(8,9)
13:     [lr1,lr2]=[lr3,lr4]=(13,14)
14: 
15:     (lr1,lr2)=[lr3,lr4]=(113,114)
16: 
17:     def method(self):
18:         r = "hi"
19:         a, b = 5, 3
20:         c = 4, 5
21:         (m, n, o) = (4, 5, 6)
22:         (m, n, o) = [4, 5, 6]
23:         [x, y, z, r] = [1, 2, 3, 4]
24:         (x, y, z, r) = [1, 2, 3, 4]
25: 
26:         x1=x2=x3=5
27: 
28:         (r1,r2)=(r3,r4)=(8,9)
29:         [lr1,lr2]=[lr3,lr4]=(13,14)
30: 
31:         (lr1,lr2)=[lr3,lr4]=(113,114)
32: 
33: ca = C.a
34: cb = C.b
35: cc = C.c
36: cr = C.r
37: cm = C.m
38: cn = C.n
39: co = C.o
40: cx = C.x
41: cy = C.y
42: cz = C.z
43: cx1 = C.x1
44: cx2 = C.x2
45: cx3 = C.x3
46: cr1 = C.r1
47: cr2 = C.r2
48: cr3 = C.r3
49: cr4 = C.r4
50: clr1 = C.lr1
51: clr2 = C.lr2
52: clr3 = C.lr3
53: clr4 = C.lr4
54: 
55: c = C()
56: c.method()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'C' class

class C:
    
    # Assigning a Str to a Name (line 2):
    
    # Assigning a Tuple to a Tuple (line 3):
    
    # Assigning a Tuple to a Name (line 4):
    
    # Assigning a Tuple to a Tuple (line 5):
    
    # Assigning a List to a Tuple (line 6):
    
    # Assigning a List to a List (line 7):
    
    # Assigning a List to a Tuple (line 8):
    
    # Multiple assignment of 3 elements.
    
    # Multiple assignment of 2 elements.
    
    # Multiple assignment of 2 elements.
    
    # Multiple assignment of 2 elements.

    @norecursion
    def method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'method'
        module_type_store = module_type_store.open_function_context('method', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        C.method.__dict__.__setitem__('stypy_localization', localization)
        C.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        C.method.__dict__.__setitem__('stypy_type_store', module_type_store)
        C.method.__dict__.__setitem__('stypy_function_name', 'C.method')
        C.method.__dict__.__setitem__('stypy_param_names_list', [])
        C.method.__dict__.__setitem__('stypy_varargs_param_name', None)
        C.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        C.method.__dict__.__setitem__('stypy_call_defaults', defaults)
        C.method.__dict__.__setitem__('stypy_call_varargs', varargs)
        C.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        C.method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'method(...)' code ##################

        
        # Assigning a Str to a Name (line 18):
        
        # Assigning a Str to a Name (line 18):
        str_1389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 12), 'str', 'hi')
        # Assigning a type to the variable 'r' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'r', str_1389)
        
        # Assigning a Tuple to a Tuple (line 19):
        
        # Assigning a Num to a Name (line 19):
        int_1390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'int')
        # Assigning a type to the variable 'tuple_assignment_1361' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_1361', int_1390)
        
        # Assigning a Num to a Name (line 19):
        int_1391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'int')
        # Assigning a type to the variable 'tuple_assignment_1362' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_1362', int_1391)
        
        # Assigning a Name to a Name (line 19):
        # Getting the type of 'tuple_assignment_1361' (line 19)
        tuple_assignment_1361_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_1361')
        # Assigning a type to the variable 'a' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'a', tuple_assignment_1361_1392)
        
        # Assigning a Name to a Name (line 19):
        # Getting the type of 'tuple_assignment_1362' (line 19)
        tuple_assignment_1362_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_1362')
        # Assigning a type to the variable 'b' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'b', tuple_assignment_1362_1393)
        
        # Assigning a Tuple to a Name (line 20):
        
        # Assigning a Tuple to a Name (line 20):
        
        # Obtaining an instance of the builtin type 'tuple' (line 20)
        tuple_1394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 20)
        # Adding element type (line 20)
        int_1395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), tuple_1394, int_1395)
        # Adding element type (line 20)
        int_1396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), tuple_1394, int_1396)
        
        # Assigning a type to the variable 'c' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'c', tuple_1394)
        
        # Assigning a Tuple to a Tuple (line 21):
        
        # Assigning a Num to a Name (line 21):
        int_1397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 21), 'int')
        # Assigning a type to the variable 'tuple_assignment_1363' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'tuple_assignment_1363', int_1397)
        
        # Assigning a Num to a Name (line 21):
        int_1398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'int')
        # Assigning a type to the variable 'tuple_assignment_1364' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'tuple_assignment_1364', int_1398)
        
        # Assigning a Num to a Name (line 21):
        int_1399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 27), 'int')
        # Assigning a type to the variable 'tuple_assignment_1365' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'tuple_assignment_1365', int_1399)
        
        # Assigning a Name to a Name (line 21):
        # Getting the type of 'tuple_assignment_1363' (line 21)
        tuple_assignment_1363_1400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'tuple_assignment_1363')
        # Assigning a type to the variable 'm' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 9), 'm', tuple_assignment_1363_1400)
        
        # Assigning a Name to a Name (line 21):
        # Getting the type of 'tuple_assignment_1364' (line 21)
        tuple_assignment_1364_1401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'tuple_assignment_1364')
        # Assigning a type to the variable 'n' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'n', tuple_assignment_1364_1401)
        
        # Assigning a Name to a Name (line 21):
        # Getting the type of 'tuple_assignment_1365' (line 21)
        tuple_assignment_1365_1402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'tuple_assignment_1365')
        # Assigning a type to the variable 'o' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'o', tuple_assignment_1365_1402)
        
        # Assigning a List to a Tuple (line 22):
        
        # Assigning a Num to a Name (line 22):
        int_1403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
        # Assigning a type to the variable 'tuple_assignment_1366' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_1366', int_1403)
        
        # Assigning a Num to a Name (line 22):
        int_1404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'int')
        # Assigning a type to the variable 'tuple_assignment_1367' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_1367', int_1404)
        
        # Assigning a Num to a Name (line 22):
        int_1405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'int')
        # Assigning a type to the variable 'tuple_assignment_1368' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_1368', int_1405)
        
        # Assigning a Name to a Name (line 22):
        # Getting the type of 'tuple_assignment_1366' (line 22)
        tuple_assignment_1366_1406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_1366')
        # Assigning a type to the variable 'm' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'm', tuple_assignment_1366_1406)
        
        # Assigning a Name to a Name (line 22):
        # Getting the type of 'tuple_assignment_1367' (line 22)
        tuple_assignment_1367_1407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_1367')
        # Assigning a type to the variable 'n' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'n', tuple_assignment_1367_1407)
        
        # Assigning a Name to a Name (line 22):
        # Getting the type of 'tuple_assignment_1368' (line 22)
        tuple_assignment_1368_1408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_1368')
        # Assigning a type to the variable 'o' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'o', tuple_assignment_1368_1408)
        
        # Assigning a List to a List (line 23):
        
        # Assigning a Num to a Name (line 23):
        int_1409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
        # Assigning a type to the variable 'list_assignment_1369' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'list_assignment_1369', int_1409)
        
        # Assigning a Num to a Name (line 23):
        int_1410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'int')
        # Assigning a type to the variable 'list_assignment_1370' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'list_assignment_1370', int_1410)
        
        # Assigning a Num to a Name (line 23):
        int_1411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'int')
        # Assigning a type to the variable 'list_assignment_1371' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'list_assignment_1371', int_1411)
        
        # Assigning a Num to a Name (line 23):
        int_1412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 33), 'int')
        # Assigning a type to the variable 'list_assignment_1372' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'list_assignment_1372', int_1412)
        
        # Assigning a Name to a Name (line 23):
        # Getting the type of 'list_assignment_1369' (line 23)
        list_assignment_1369_1413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'list_assignment_1369')
        # Assigning a type to the variable 'x' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 9), 'x', list_assignment_1369_1413)
        
        # Assigning a Name to a Name (line 23):
        # Getting the type of 'list_assignment_1370' (line 23)
        list_assignment_1370_1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'list_assignment_1370')
        # Assigning a type to the variable 'y' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'y', list_assignment_1370_1414)
        
        # Assigning a Name to a Name (line 23):
        # Getting the type of 'list_assignment_1371' (line 23)
        list_assignment_1371_1415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'list_assignment_1371')
        # Assigning a type to the variable 'z' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'z', list_assignment_1371_1415)
        
        # Assigning a Name to a Name (line 23):
        # Getting the type of 'list_assignment_1372' (line 23)
        list_assignment_1372_1416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'list_assignment_1372')
        # Assigning a type to the variable 'r' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 'r', list_assignment_1372_1416)
        
        # Assigning a List to a Tuple (line 24):
        
        # Assigning a Num to a Name (line 24):
        int_1417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'int')
        # Assigning a type to the variable 'tuple_assignment_1373' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_assignment_1373', int_1417)
        
        # Assigning a Num to a Name (line 24):
        int_1418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 27), 'int')
        # Assigning a type to the variable 'tuple_assignment_1374' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_assignment_1374', int_1418)
        
        # Assigning a Num to a Name (line 24):
        int_1419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'int')
        # Assigning a type to the variable 'tuple_assignment_1375' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_assignment_1375', int_1419)
        
        # Assigning a Num to a Name (line 24):
        int_1420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'int')
        # Assigning a type to the variable 'tuple_assignment_1376' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_assignment_1376', int_1420)
        
        # Assigning a Name to a Name (line 24):
        # Getting the type of 'tuple_assignment_1373' (line 24)
        tuple_assignment_1373_1421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_assignment_1373')
        # Assigning a type to the variable 'x' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 9), 'x', tuple_assignment_1373_1421)
        
        # Assigning a Name to a Name (line 24):
        # Getting the type of 'tuple_assignment_1374' (line 24)
        tuple_assignment_1374_1422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_assignment_1374')
        # Assigning a type to the variable 'y' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'y', tuple_assignment_1374_1422)
        
        # Assigning a Name to a Name (line 24):
        # Getting the type of 'tuple_assignment_1375' (line 24)
        tuple_assignment_1375_1423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_assignment_1375')
        # Assigning a type to the variable 'z' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), 'z', tuple_assignment_1375_1423)
        
        # Assigning a Name to a Name (line 24):
        # Getting the type of 'tuple_assignment_1376' (line 24)
        tuple_assignment_1376_1424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_assignment_1376')
        # Assigning a type to the variable 'r' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'r', tuple_assignment_1376_1424)
        
        # Multiple assignment of 3 elements.
        
        # Assigning a Num to a Name (line 26):
        int_1425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'int')
        # Assigning a type to the variable 'x3' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'x3', int_1425)
        
        # Assigning a Name to a Name (line 26):
        # Getting the type of 'x3' (line 26)
        x3_1426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'x3')
        # Assigning a type to the variable 'x2' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'x2', x3_1426)
        
        # Assigning a Name to a Name (line 26):
        # Getting the type of 'x2' (line 26)
        x2_1427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'x2')
        # Assigning a type to the variable 'x1' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'x1', x2_1427)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Num to a Name (line 28):
        int_1428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'int')
        # Assigning a type to the variable 'tuple_assignment_1377' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_assignment_1377', int_1428)
        
        # Assigning a Num to a Name (line 28):
        int_1429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 27), 'int')
        # Assigning a type to the variable 'tuple_assignment_1378' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_assignment_1378', int_1429)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_assignment_1377' (line 28)
        tuple_assignment_1377_1430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_assignment_1377')
        # Assigning a type to the variable 'r3' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'r3', tuple_assignment_1377_1430)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_assignment_1378' (line 28)
        tuple_assignment_1378_1431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_assignment_1378')
        # Assigning a type to the variable 'r4' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'r4', tuple_assignment_1378_1431)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'r3' (line 28)
        r3_1432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'r3')
        # Assigning a type to the variable 'tuple_assignment_1379' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_assignment_1379', r3_1432)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'r4' (line 28)
        r4_1433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'r4')
        # Assigning a type to the variable 'tuple_assignment_1380' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_assignment_1380', r4_1433)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_assignment_1379' (line 28)
        tuple_assignment_1379_1434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_assignment_1379')
        # Assigning a type to the variable 'r1' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 9), 'r1', tuple_assignment_1379_1434)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_assignment_1380' (line 28)
        tuple_assignment_1380_1435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_assignment_1380')
        # Assigning a type to the variable 'r2' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'r2', tuple_assignment_1380_1435)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Num to a Name (line 29):
        int_1436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'int')
        # Assigning a type to the variable 'list_assignment_1381' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'list_assignment_1381', int_1436)
        
        # Assigning a Num to a Name (line 29):
        int_1437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 32), 'int')
        # Assigning a type to the variable 'list_assignment_1382' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'list_assignment_1382', int_1437)
        
        # Assigning a Name to a Name (line 29):
        # Getting the type of 'list_assignment_1381' (line 29)
        list_assignment_1381_1438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'list_assignment_1381')
        # Assigning a type to the variable 'lr3' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'lr3', list_assignment_1381_1438)
        
        # Assigning a Name to a Name (line 29):
        # Getting the type of 'list_assignment_1382' (line 29)
        list_assignment_1382_1439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'list_assignment_1382')
        # Assigning a type to the variable 'lr4' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'lr4', list_assignment_1382_1439)
        
        # Assigning a Name to a Name (line 29):
        # Getting the type of 'lr3' (line 29)
        lr3_1440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'lr3')
        # Assigning a type to the variable 'list_assignment_1383' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'list_assignment_1383', lr3_1440)
        
        # Assigning a Name to a Name (line 29):
        # Getting the type of 'lr4' (line 29)
        lr4_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'lr4')
        # Assigning a type to the variable 'list_assignment_1384' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'list_assignment_1384', lr4_1441)
        
        # Assigning a Name to a Name (line 29):
        # Getting the type of 'list_assignment_1383' (line 29)
        list_assignment_1383_1442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'list_assignment_1383')
        # Assigning a type to the variable 'lr1' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 9), 'lr1', list_assignment_1383_1442)
        
        # Assigning a Name to a Name (line 29):
        # Getting the type of 'list_assignment_1384' (line 29)
        list_assignment_1384_1443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'list_assignment_1384')
        # Assigning a type to the variable 'lr2' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'lr2', list_assignment_1384_1443)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Num to a Name (line 31):
        int_1444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'int')
        # Assigning a type to the variable 'list_assignment_1385' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'list_assignment_1385', int_1444)
        
        # Assigning a Num to a Name (line 31):
        int_1445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 33), 'int')
        # Assigning a type to the variable 'list_assignment_1386' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'list_assignment_1386', int_1445)
        
        # Assigning a Name to a Name (line 31):
        # Getting the type of 'list_assignment_1385' (line 31)
        list_assignment_1385_1446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'list_assignment_1385')
        # Assigning a type to the variable 'lr3' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'lr3', list_assignment_1385_1446)
        
        # Assigning a Name to a Name (line 31):
        # Getting the type of 'list_assignment_1386' (line 31)
        list_assignment_1386_1447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'list_assignment_1386')
        # Assigning a type to the variable 'lr4' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'lr4', list_assignment_1386_1447)
        
        # Assigning a Name to a Name (line 31):
        # Getting the type of 'lr3' (line 31)
        lr3_1448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'lr3')
        # Assigning a type to the variable 'tuple_assignment_1387' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'tuple_assignment_1387', lr3_1448)
        
        # Assigning a Name to a Name (line 31):
        # Getting the type of 'lr4' (line 31)
        lr4_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'lr4')
        # Assigning a type to the variable 'tuple_assignment_1388' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'tuple_assignment_1388', lr4_1449)
        
        # Assigning a Name to a Name (line 31):
        # Getting the type of 'tuple_assignment_1387' (line 31)
        tuple_assignment_1387_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'tuple_assignment_1387')
        # Assigning a type to the variable 'lr1' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 9), 'lr1', tuple_assignment_1387_1450)
        
        # Assigning a Name to a Name (line 31):
        # Getting the type of 'tuple_assignment_1388' (line 31)
        tuple_assignment_1388_1451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'tuple_assignment_1388')
        # Assigning a type to the variable 'lr2' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'lr2', tuple_assignment_1388_1451)
        
        # ################# End of 'method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'method' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1452)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'method'
        return stypy_return_type_1452


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1, 0, False)
        # Assigning a type to the variable 'self' (line 2)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'C' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'C', C)

# Assigning a Str to a Name (line 2):
str_1453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 8), 'str', 'hi')
# Getting the type of 'C'
C_1454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'r' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1454, 'r', str_1453)

# Assigning a Num to a Name (line 3):
int_1455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'int')
# Getting the type of 'C'
C_1456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1333' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1456, 'tuple_assignment_1333', int_1455)

# Assigning a Num to a Name (line 3):
int_1457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 14), 'int')
# Getting the type of 'C'
C_1458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1334' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1458, 'tuple_assignment_1334', int_1457)

# Assigning a Name to a Name (line 3):
# Getting the type of 'C'
C_1459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1333' of a type
tuple_assignment_1333_1460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1459, 'tuple_assignment_1333')
# Getting the type of 'C'
C_1461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'a' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1461, 'a', tuple_assignment_1333_1460)

# Assigning a Name to a Name (line 3):
# Getting the type of 'C'
C_1462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1334' of a type
tuple_assignment_1334_1463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1462, 'tuple_assignment_1334')
# Getting the type of 'C'
C_1464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'b' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1464, 'b', tuple_assignment_1334_1463)

# Assigning a Tuple to a Name (line 4):

# Obtaining an instance of the builtin type 'tuple' (line 4)
tuple_1465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4)
# Adding element type (line 4)
int_1466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 8), tuple_1465, int_1466)
# Adding element type (line 4)
int_1467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 8), tuple_1465, int_1467)

# Getting the type of 'C'
C_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'c' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1468, 'c', tuple_1465)

# Assigning a Num to a Name (line 5):
int_1469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
# Getting the type of 'C'
C_1470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1335' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1470, 'tuple_assignment_1335', int_1469)

# Assigning a Num to a Name (line 5):
int_1471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
# Getting the type of 'C'
C_1472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1336' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1472, 'tuple_assignment_1336', int_1471)

# Assigning a Num to a Name (line 5):
int_1473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
# Getting the type of 'C'
C_1474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1337' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1474, 'tuple_assignment_1337', int_1473)

# Assigning a Name to a Name (line 5):
# Getting the type of 'C'
C_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1335' of a type
tuple_assignment_1335_1476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1475, 'tuple_assignment_1335')
# Getting the type of 'C'
C_1477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'm' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1477, 'm', tuple_assignment_1335_1476)

# Assigning a Name to a Name (line 5):
# Getting the type of 'C'
C_1478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1336' of a type
tuple_assignment_1336_1479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1478, 'tuple_assignment_1336')
# Getting the type of 'C'
C_1480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'n' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1480, 'n', tuple_assignment_1336_1479)

# Assigning a Name to a Name (line 5):
# Getting the type of 'C'
C_1481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1337' of a type
tuple_assignment_1337_1482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1481, 'tuple_assignment_1337')
# Getting the type of 'C'
C_1483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'o' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1483, 'o', tuple_assignment_1337_1482)

# Assigning a Num to a Name (line 6):
int_1484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 17), 'int')
# Getting the type of 'C'
C_1485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1338' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1485, 'tuple_assignment_1338', int_1484)

# Assigning a Num to a Name (line 6):
int_1486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'int')
# Getting the type of 'C'
C_1487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1339' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1487, 'tuple_assignment_1339', int_1486)

# Assigning a Num to a Name (line 6):
int_1488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 23), 'int')
# Getting the type of 'C'
C_1489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1340' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1489, 'tuple_assignment_1340', int_1488)

# Assigning a Name to a Name (line 6):
# Getting the type of 'C'
C_1490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1338' of a type
tuple_assignment_1338_1491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1490, 'tuple_assignment_1338')
# Getting the type of 'C'
C_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'm' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1492, 'm', tuple_assignment_1338_1491)

# Assigning a Name to a Name (line 6):
# Getting the type of 'C'
C_1493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1339' of a type
tuple_assignment_1339_1494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1493, 'tuple_assignment_1339')
# Getting the type of 'C'
C_1495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'n' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1495, 'n', tuple_assignment_1339_1494)

# Assigning a Name to a Name (line 6):
# Getting the type of 'C'
C_1496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1340' of a type
tuple_assignment_1340_1497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1496, 'tuple_assignment_1340')
# Getting the type of 'C'
C_1498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'o' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1498, 'o', tuple_assignment_1340_1497)

# Assigning a Num to a Name (line 7):
int_1499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'int')
# Getting the type of 'C'
C_1500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'list_assignment_1341' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1500, 'list_assignment_1341', int_1499)

# Assigning a Num to a Name (line 7):
int_1501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 23), 'int')
# Getting the type of 'C'
C_1502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'list_assignment_1342' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1502, 'list_assignment_1342', int_1501)

# Assigning a Num to a Name (line 7):
int_1503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'int')
# Getting the type of 'C'
C_1504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'list_assignment_1343' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1504, 'list_assignment_1343', int_1503)

# Assigning a Num to a Name (line 7):
int_1505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 29), 'int')
# Getting the type of 'C'
C_1506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'list_assignment_1344' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1506, 'list_assignment_1344', int_1505)

# Assigning a Name to a Name (line 7):
# Getting the type of 'C'
C_1507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'list_assignment_1341' of a type
list_assignment_1341_1508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1507, 'list_assignment_1341')
# Getting the type of 'C'
C_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'x' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1509, 'x', list_assignment_1341_1508)

# Assigning a Name to a Name (line 7):
# Getting the type of 'C'
C_1510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'list_assignment_1342' of a type
list_assignment_1342_1511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1510, 'list_assignment_1342')
# Getting the type of 'C'
C_1512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'y' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1512, 'y', list_assignment_1342_1511)

# Assigning a Name to a Name (line 7):
# Getting the type of 'C'
C_1513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'list_assignment_1343' of a type
list_assignment_1343_1514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1513, 'list_assignment_1343')
# Getting the type of 'C'
C_1515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'z' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1515, 'z', list_assignment_1343_1514)

# Assigning a Name to a Name (line 7):
# Getting the type of 'C'
C_1516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'list_assignment_1344' of a type
list_assignment_1344_1517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1516, 'list_assignment_1344')
# Getting the type of 'C'
C_1518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'r' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1518, 'r', list_assignment_1344_1517)

# Assigning a Num to a Name (line 8):
int_1519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
# Getting the type of 'C'
C_1520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1345' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1520, 'tuple_assignment_1345', int_1519)

# Assigning a Num to a Name (line 8):
int_1521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 23), 'int')
# Getting the type of 'C'
C_1522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1346' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1522, 'tuple_assignment_1346', int_1521)

# Assigning a Num to a Name (line 8):
int_1523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 26), 'int')
# Getting the type of 'C'
C_1524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1347' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1524, 'tuple_assignment_1347', int_1523)

# Assigning a Num to a Name (line 8):
int_1525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 29), 'int')
# Getting the type of 'C'
C_1526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1348' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1526, 'tuple_assignment_1348', int_1525)

# Assigning a Name to a Name (line 8):
# Getting the type of 'C'
C_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1345' of a type
tuple_assignment_1345_1528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1527, 'tuple_assignment_1345')
# Getting the type of 'C'
C_1529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'x' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1529, 'x', tuple_assignment_1345_1528)

# Assigning a Name to a Name (line 8):
# Getting the type of 'C'
C_1530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1346' of a type
tuple_assignment_1346_1531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1530, 'tuple_assignment_1346')
# Getting the type of 'C'
C_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'y' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1532, 'y', tuple_assignment_1346_1531)

# Assigning a Name to a Name (line 8):
# Getting the type of 'C'
C_1533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1347' of a type
tuple_assignment_1347_1534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1533, 'tuple_assignment_1347')
# Getting the type of 'C'
C_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'z' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1535, 'z', tuple_assignment_1347_1534)

# Assigning a Name to a Name (line 8):
# Getting the type of 'C'
C_1536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1348' of a type
tuple_assignment_1348_1537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1536, 'tuple_assignment_1348')
# Getting the type of 'C'
C_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'r' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1538, 'r', tuple_assignment_1348_1537)

# Assigning a Num to a Name (line 10):
int_1539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'int')
# Getting the type of 'C'
C_1540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'x3' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1540, 'x3', int_1539)

# Assigning a Name to a Name (line 10):
# Getting the type of 'C'
C_1541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'x3' of a type
x3_1542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1541, 'x3')
# Getting the type of 'C'
C_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'x2' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1543, 'x2', x3_1542)

# Assigning a Name to a Name (line 10):
# Getting the type of 'C'
C_1544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'x2' of a type
x2_1545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1544, 'x2')
# Getting the type of 'C'
C_1546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'x1' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1546, 'x1', x2_1545)

# Assigning a Num to a Name (line 12):
int_1547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'int')
# Getting the type of 'C'
C_1548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1349' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1548, 'tuple_assignment_1349', int_1547)

# Assigning a Num to a Name (line 12):
int_1549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'int')
# Getting the type of 'C'
C_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1350' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1550, 'tuple_assignment_1350', int_1549)

# Assigning a Name to a Name (line 12):
# Getting the type of 'C'
C_1551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1349' of a type
tuple_assignment_1349_1552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1551, 'tuple_assignment_1349')
# Getting the type of 'C'
C_1553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'r3' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1553, 'r3', tuple_assignment_1349_1552)

# Assigning a Name to a Name (line 12):
# Getting the type of 'C'
C_1554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1350' of a type
tuple_assignment_1350_1555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1554, 'tuple_assignment_1350')
# Getting the type of 'C'
C_1556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'r4' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1556, 'r4', tuple_assignment_1350_1555)

# Assigning a Name to a Name (line 12):
# Getting the type of 'C'
C_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'r3' of a type
r3_1558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1557, 'r3')
# Getting the type of 'C'
C_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1351' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1559, 'tuple_assignment_1351', r3_1558)

# Assigning a Name to a Name (line 12):
# Getting the type of 'C'
C_1560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'r4' of a type
r4_1561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1560, 'r4')
# Getting the type of 'C'
C_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1352' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1562, 'tuple_assignment_1352', r4_1561)

# Assigning a Name to a Name (line 12):
# Getting the type of 'C'
C_1563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1351' of a type
tuple_assignment_1351_1564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1563, 'tuple_assignment_1351')
# Getting the type of 'C'
C_1565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'r1' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1565, 'r1', tuple_assignment_1351_1564)

# Assigning a Name to a Name (line 12):
# Getting the type of 'C'
C_1566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1352' of a type
tuple_assignment_1352_1567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1566, 'tuple_assignment_1352')
# Getting the type of 'C'
C_1568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'r2' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1568, 'r2', tuple_assignment_1352_1567)

# Assigning a Num to a Name (line 13):
int_1569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'int')
# Getting the type of 'C'
C_1570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'list_assignment_1353' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1570, 'list_assignment_1353', int_1569)

# Assigning a Num to a Name (line 13):
int_1571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 28), 'int')
# Getting the type of 'C'
C_1572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'list_assignment_1354' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1572, 'list_assignment_1354', int_1571)

# Assigning a Name to a Name (line 13):
# Getting the type of 'C'
C_1573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'list_assignment_1353' of a type
list_assignment_1353_1574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1573, 'list_assignment_1353')
# Getting the type of 'C'
C_1575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'lr3' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1575, 'lr3', list_assignment_1353_1574)

# Assigning a Name to a Name (line 13):
# Getting the type of 'C'
C_1576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'list_assignment_1354' of a type
list_assignment_1354_1577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1576, 'list_assignment_1354')
# Getting the type of 'C'
C_1578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'lr4' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1578, 'lr4', list_assignment_1354_1577)

# Assigning a Name to a Name (line 13):
# Getting the type of 'C'
C_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'lr3' of a type
lr3_1580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1579, 'lr3')
# Getting the type of 'C'
C_1581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'list_assignment_1355' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1581, 'list_assignment_1355', lr3_1580)

# Assigning a Name to a Name (line 13):
# Getting the type of 'C'
C_1582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'lr4' of a type
lr4_1583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1582, 'lr4')
# Getting the type of 'C'
C_1584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'list_assignment_1356' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1584, 'list_assignment_1356', lr4_1583)

# Assigning a Name to a Name (line 13):
# Getting the type of 'C'
C_1585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'list_assignment_1355' of a type
list_assignment_1355_1586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1585, 'list_assignment_1355')
# Getting the type of 'C'
C_1587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'lr1' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1587, 'lr1', list_assignment_1355_1586)

# Assigning a Name to a Name (line 13):
# Getting the type of 'C'
C_1588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'list_assignment_1356' of a type
list_assignment_1356_1589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1588, 'list_assignment_1356')
# Getting the type of 'C'
C_1590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'lr2' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1590, 'lr2', list_assignment_1356_1589)

# Assigning a Num to a Name (line 15):
int_1591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'int')
# Getting the type of 'C'
C_1592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'list_assignment_1357' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1592, 'list_assignment_1357', int_1591)

# Assigning a Num to a Name (line 15):
int_1593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 29), 'int')
# Getting the type of 'C'
C_1594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'list_assignment_1358' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1594, 'list_assignment_1358', int_1593)

# Assigning a Name to a Name (line 15):
# Getting the type of 'C'
C_1595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'list_assignment_1357' of a type
list_assignment_1357_1596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1595, 'list_assignment_1357')
# Getting the type of 'C'
C_1597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'lr3' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1597, 'lr3', list_assignment_1357_1596)

# Assigning a Name to a Name (line 15):
# Getting the type of 'C'
C_1598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'list_assignment_1358' of a type
list_assignment_1358_1599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1598, 'list_assignment_1358')
# Getting the type of 'C'
C_1600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'lr4' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1600, 'lr4', list_assignment_1358_1599)

# Assigning a Name to a Name (line 15):
# Getting the type of 'C'
C_1601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'lr3' of a type
lr3_1602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1601, 'lr3')
# Getting the type of 'C'
C_1603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1359' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1603, 'tuple_assignment_1359', lr3_1602)

# Assigning a Name to a Name (line 15):
# Getting the type of 'C'
C_1604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'lr4' of a type
lr4_1605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1604, 'lr4')
# Getting the type of 'C'
C_1606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_assignment_1360' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1606, 'tuple_assignment_1360', lr4_1605)

# Assigning a Name to a Name (line 15):
# Getting the type of 'C'
C_1607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1359' of a type
tuple_assignment_1359_1608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1607, 'tuple_assignment_1359')
# Getting the type of 'C'
C_1609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'lr1' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1609, 'lr1', tuple_assignment_1359_1608)

# Assigning a Name to a Name (line 15):
# Getting the type of 'C'
C_1610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_assignment_1360' of a type
tuple_assignment_1360_1611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1610, 'tuple_assignment_1360')
# Getting the type of 'C'
C_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'lr2' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1612, 'lr2', tuple_assignment_1360_1611)

# Assigning a Attribute to a Name (line 33):

# Assigning a Attribute to a Name (line 33):
# Getting the type of 'C' (line 33)
C_1613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 5), 'C')
# Obtaining the member 'a' of a type (line 33)
a_1614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 5), C_1613, 'a')
# Assigning a type to the variable 'ca' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'ca', a_1614)

# Assigning a Attribute to a Name (line 34):

# Assigning a Attribute to a Name (line 34):
# Getting the type of 'C' (line 34)
C_1615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 5), 'C')
# Obtaining the member 'b' of a type (line 34)
b_1616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 5), C_1615, 'b')
# Assigning a type to the variable 'cb' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'cb', b_1616)

# Assigning a Attribute to a Name (line 35):

# Assigning a Attribute to a Name (line 35):
# Getting the type of 'C' (line 35)
C_1617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 5), 'C')
# Obtaining the member 'c' of a type (line 35)
c_1618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 5), C_1617, 'c')
# Assigning a type to the variable 'cc' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'cc', c_1618)

# Assigning a Attribute to a Name (line 36):

# Assigning a Attribute to a Name (line 36):
# Getting the type of 'C' (line 36)
C_1619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 5), 'C')
# Obtaining the member 'r' of a type (line 36)
r_1620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 5), C_1619, 'r')
# Assigning a type to the variable 'cr' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'cr', r_1620)

# Assigning a Attribute to a Name (line 37):

# Assigning a Attribute to a Name (line 37):
# Getting the type of 'C' (line 37)
C_1621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 5), 'C')
# Obtaining the member 'm' of a type (line 37)
m_1622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 5), C_1621, 'm')
# Assigning a type to the variable 'cm' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'cm', m_1622)

# Assigning a Attribute to a Name (line 38):

# Assigning a Attribute to a Name (line 38):
# Getting the type of 'C' (line 38)
C_1623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 5), 'C')
# Obtaining the member 'n' of a type (line 38)
n_1624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 5), C_1623, 'n')
# Assigning a type to the variable 'cn' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'cn', n_1624)

# Assigning a Attribute to a Name (line 39):

# Assigning a Attribute to a Name (line 39):
# Getting the type of 'C' (line 39)
C_1625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 5), 'C')
# Obtaining the member 'o' of a type (line 39)
o_1626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 5), C_1625, 'o')
# Assigning a type to the variable 'co' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'co', o_1626)

# Assigning a Attribute to a Name (line 40):

# Assigning a Attribute to a Name (line 40):
# Getting the type of 'C' (line 40)
C_1627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 5), 'C')
# Obtaining the member 'x' of a type (line 40)
x_1628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 5), C_1627, 'x')
# Assigning a type to the variable 'cx' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'cx', x_1628)

# Assigning a Attribute to a Name (line 41):

# Assigning a Attribute to a Name (line 41):
# Getting the type of 'C' (line 41)
C_1629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 5), 'C')
# Obtaining the member 'y' of a type (line 41)
y_1630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 5), C_1629, 'y')
# Assigning a type to the variable 'cy' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'cy', y_1630)

# Assigning a Attribute to a Name (line 42):

# Assigning a Attribute to a Name (line 42):
# Getting the type of 'C' (line 42)
C_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 5), 'C')
# Obtaining the member 'z' of a type (line 42)
z_1632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 5), C_1631, 'z')
# Assigning a type to the variable 'cz' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'cz', z_1632)

# Assigning a Attribute to a Name (line 43):

# Assigning a Attribute to a Name (line 43):
# Getting the type of 'C' (line 43)
C_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 6), 'C')
# Obtaining the member 'x1' of a type (line 43)
x1_1634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 6), C_1633, 'x1')
# Assigning a type to the variable 'cx1' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'cx1', x1_1634)

# Assigning a Attribute to a Name (line 44):

# Assigning a Attribute to a Name (line 44):
# Getting the type of 'C' (line 44)
C_1635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 6), 'C')
# Obtaining the member 'x2' of a type (line 44)
x2_1636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 6), C_1635, 'x2')
# Assigning a type to the variable 'cx2' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'cx2', x2_1636)

# Assigning a Attribute to a Name (line 45):

# Assigning a Attribute to a Name (line 45):
# Getting the type of 'C' (line 45)
C_1637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 6), 'C')
# Obtaining the member 'x3' of a type (line 45)
x3_1638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 6), C_1637, 'x3')
# Assigning a type to the variable 'cx3' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'cx3', x3_1638)

# Assigning a Attribute to a Name (line 46):

# Assigning a Attribute to a Name (line 46):
# Getting the type of 'C' (line 46)
C_1639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 6), 'C')
# Obtaining the member 'r1' of a type (line 46)
r1_1640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 6), C_1639, 'r1')
# Assigning a type to the variable 'cr1' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'cr1', r1_1640)

# Assigning a Attribute to a Name (line 47):

# Assigning a Attribute to a Name (line 47):
# Getting the type of 'C' (line 47)
C_1641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 6), 'C')
# Obtaining the member 'r2' of a type (line 47)
r2_1642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 6), C_1641, 'r2')
# Assigning a type to the variable 'cr2' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'cr2', r2_1642)

# Assigning a Attribute to a Name (line 48):

# Assigning a Attribute to a Name (line 48):
# Getting the type of 'C' (line 48)
C_1643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 6), 'C')
# Obtaining the member 'r3' of a type (line 48)
r3_1644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 6), C_1643, 'r3')
# Assigning a type to the variable 'cr3' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'cr3', r3_1644)

# Assigning a Attribute to a Name (line 49):

# Assigning a Attribute to a Name (line 49):
# Getting the type of 'C' (line 49)
C_1645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 6), 'C')
# Obtaining the member 'r4' of a type (line 49)
r4_1646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 6), C_1645, 'r4')
# Assigning a type to the variable 'cr4' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'cr4', r4_1646)

# Assigning a Attribute to a Name (line 50):

# Assigning a Attribute to a Name (line 50):
# Getting the type of 'C' (line 50)
C_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'C')
# Obtaining the member 'lr1' of a type (line 50)
lr1_1648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 7), C_1647, 'lr1')
# Assigning a type to the variable 'clr1' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'clr1', lr1_1648)

# Assigning a Attribute to a Name (line 51):

# Assigning a Attribute to a Name (line 51):
# Getting the type of 'C' (line 51)
C_1649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 7), 'C')
# Obtaining the member 'lr2' of a type (line 51)
lr2_1650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 7), C_1649, 'lr2')
# Assigning a type to the variable 'clr2' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'clr2', lr2_1650)

# Assigning a Attribute to a Name (line 52):

# Assigning a Attribute to a Name (line 52):
# Getting the type of 'C' (line 52)
C_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 7), 'C')
# Obtaining the member 'lr3' of a type (line 52)
lr3_1652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 7), C_1651, 'lr3')
# Assigning a type to the variable 'clr3' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'clr3', lr3_1652)

# Assigning a Attribute to a Name (line 53):

# Assigning a Attribute to a Name (line 53):
# Getting the type of 'C' (line 53)
C_1653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'C')
# Obtaining the member 'lr4' of a type (line 53)
lr4_1654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 7), C_1653, 'lr4')
# Assigning a type to the variable 'clr4' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'clr4', lr4_1654)

# Assigning a Call to a Name (line 55):

# Assigning a Call to a Name (line 55):

# Call to C(...): (line 55)
# Processing the call keyword arguments (line 55)
kwargs_1656 = {}
# Getting the type of 'C' (line 55)
C_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'C', False)
# Calling C(args, kwargs) (line 55)
C_call_result_1657 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), C_1655, *[], **kwargs_1656)

# Assigning a type to the variable 'c' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'c', C_call_result_1657)

# Call to method(...): (line 56)
# Processing the call keyword arguments (line 56)
kwargs_1660 = {}
# Getting the type of 'c' (line 56)
c_1658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'c', False)
# Obtaining the member 'method' of a type (line 56)
method_1659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 0), c_1658, 'method')
# Calling method(args, kwargs) (line 56)
method_call_result_1661 = invoke(stypy.reporting.localization.Localization(__file__, 56, 0), method_1659, *[], **kwargs_1660)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
