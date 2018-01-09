
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.randint(0, 10, (10, 10))
6: shape = (5, 5)
7: fill = 0
8: position = (1, 1)
9: 
10: R = np.ones(shape, dtype=Z.dtype) * fill
11: P = np.array(list(position)).astype(int)
12: Rs = np.array(list(R.shape)).astype(int)
13: Zs = np.array(list(Z.shape)).astype(int)
14: 
15: R_start = np.zeros((len(shape),)).astype(int)
16: R_stop = np.array(list(shape)).astype(int)
17: Z_start = (P - Rs // 2)
18: Z_stop = (P + Rs // 2) + Rs % 2
19: 
20: R_start = (R_start - np.minimum(Z_start, 0)).tolist()
21: Z_start = (np.maximum(Z_start, 0)).tolist()
22: R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop - Zs, 0))).tolist()
23: Z_stop = (np.minimum(Z_stop, Zs)).tolist()
24: 
25: r = [slice(start, stop) for start, stop in zip(R_start, R_stop)]
26: z = [slice(start, stop) for start, stop in zip(Z_start, Z_stop)]
27: R[r] = Z[z]
28: 
29: # l = globals().copy()
30: # for v in l:
31: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
32: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1617 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1617) is not StypyTypeError):

    if (import_1617 != 'pyd_module'):
        __import__(import_1617)
        sys_modules_1618 = sys.modules[import_1617]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1618.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1617)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to randint(...): (line 5)
# Processing the call arguments (line 5)
int_1622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
int_1623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_1624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_1625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 30), tuple_1624, int_1625)
# Adding element type (line 5)
int_1626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 30), tuple_1624, int_1626)

# Processing the call keyword arguments (line 5)
kwargs_1627 = {}
# Getting the type of 'np' (line 5)
np_1619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_1620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1619, 'random')
# Obtaining the member 'randint' of a type (line 5)
randint_1621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_1620, 'randint')
# Calling randint(args, kwargs) (line 5)
randint_call_result_1628 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), randint_1621, *[int_1622, int_1623, tuple_1624], **kwargs_1627)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', randint_call_result_1628)

# Assigning a Tuple to a Name (line 6):

# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_1629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
int_1630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 9), tuple_1629, int_1630)
# Adding element type (line 6)
int_1631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 9), tuple_1629, int_1631)

# Assigning a type to the variable 'shape' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'shape', tuple_1629)

# Assigning a Num to a Name (line 7):
int_1632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 7), 'int')
# Assigning a type to the variable 'fill' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'fill', int_1632)

# Assigning a Tuple to a Name (line 8):

# Obtaining an instance of the builtin type 'tuple' (line 8)
tuple_1633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 8)
# Adding element type (line 8)
int_1634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 12), tuple_1633, int_1634)
# Adding element type (line 8)
int_1635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 12), tuple_1633, int_1635)

# Assigning a type to the variable 'position' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'position', tuple_1633)

# Assigning a BinOp to a Name (line 10):

# Call to ones(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'shape' (line 10)
shape_1638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'shape', False)
# Processing the call keyword arguments (line 10)
# Getting the type of 'Z' (line 10)
Z_1639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 25), 'Z', False)
# Obtaining the member 'dtype' of a type (line 10)
dtype_1640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 25), Z_1639, 'dtype')
keyword_1641 = dtype_1640
kwargs_1642 = {'dtype': keyword_1641}
# Getting the type of 'np' (line 10)
np_1636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'np', False)
# Obtaining the member 'ones' of a type (line 10)
ones_1637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), np_1636, 'ones')
# Calling ones(args, kwargs) (line 10)
ones_call_result_1643 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), ones_1637, *[shape_1638], **kwargs_1642)

# Getting the type of 'fill' (line 10)
fill_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 36), 'fill')
# Applying the binary operator '*' (line 10)
result_mul_1645 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 4), '*', ones_call_result_1643, fill_1644)

# Assigning a type to the variable 'R' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'R', result_mul_1645)

# Assigning a Call to a Name (line 11):

# Call to astype(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'int' (line 11)
int_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 36), 'int', False)
# Processing the call keyword arguments (line 11)
kwargs_1656 = {}

# Call to array(...): (line 11)
# Processing the call arguments (line 11)

# Call to list(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'position' (line 11)
position_1649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 18), 'position', False)
# Processing the call keyword arguments (line 11)
kwargs_1650 = {}
# Getting the type of 'list' (line 11)
list_1648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'list', False)
# Calling list(args, kwargs) (line 11)
list_call_result_1651 = invoke(stypy.reporting.localization.Localization(__file__, 11, 13), list_1648, *[position_1649], **kwargs_1650)

# Processing the call keyword arguments (line 11)
kwargs_1652 = {}
# Getting the type of 'np' (line 11)
np_1646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
# Obtaining the member 'array' of a type (line 11)
array_1647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_1646, 'array')
# Calling array(args, kwargs) (line 11)
array_call_result_1653 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), array_1647, *[list_call_result_1651], **kwargs_1652)

# Obtaining the member 'astype' of a type (line 11)
astype_1654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), array_call_result_1653, 'astype')
# Calling astype(args, kwargs) (line 11)
astype_call_result_1657 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), astype_1654, *[int_1655], **kwargs_1656)

# Assigning a type to the variable 'P' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'P', astype_call_result_1657)

# Assigning a Call to a Name (line 12):

# Call to astype(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'int' (line 12)
int_1668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 36), 'int', False)
# Processing the call keyword arguments (line 12)
kwargs_1669 = {}

# Call to array(...): (line 12)
# Processing the call arguments (line 12)

# Call to list(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'R' (line 12)
R_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'R', False)
# Obtaining the member 'shape' of a type (line 12)
shape_1662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 19), R_1661, 'shape')
# Processing the call keyword arguments (line 12)
kwargs_1663 = {}
# Getting the type of 'list' (line 12)
list_1660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'list', False)
# Calling list(args, kwargs) (line 12)
list_call_result_1664 = invoke(stypy.reporting.localization.Localization(__file__, 12, 14), list_1660, *[shape_1662], **kwargs_1663)

# Processing the call keyword arguments (line 12)
kwargs_1665 = {}
# Getting the type of 'np' (line 12)
np_1658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'np', False)
# Obtaining the member 'array' of a type (line 12)
array_1659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), np_1658, 'array')
# Calling array(args, kwargs) (line 12)
array_call_result_1666 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), array_1659, *[list_call_result_1664], **kwargs_1665)

# Obtaining the member 'astype' of a type (line 12)
astype_1667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), array_call_result_1666, 'astype')
# Calling astype(args, kwargs) (line 12)
astype_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), astype_1667, *[int_1668], **kwargs_1669)

# Assigning a type to the variable 'Rs' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'Rs', astype_call_result_1670)

# Assigning a Call to a Name (line 13):

# Call to astype(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'int' (line 13)
int_1681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 36), 'int', False)
# Processing the call keyword arguments (line 13)
kwargs_1682 = {}

# Call to array(...): (line 13)
# Processing the call arguments (line 13)

# Call to list(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'Z' (line 13)
Z_1674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'Z', False)
# Obtaining the member 'shape' of a type (line 13)
shape_1675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 19), Z_1674, 'shape')
# Processing the call keyword arguments (line 13)
kwargs_1676 = {}
# Getting the type of 'list' (line 13)
list_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'list', False)
# Calling list(args, kwargs) (line 13)
list_call_result_1677 = invoke(stypy.reporting.localization.Localization(__file__, 13, 14), list_1673, *[shape_1675], **kwargs_1676)

# Processing the call keyword arguments (line 13)
kwargs_1678 = {}
# Getting the type of 'np' (line 13)
np_1671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'np', False)
# Obtaining the member 'array' of a type (line 13)
array_1672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), np_1671, 'array')
# Calling array(args, kwargs) (line 13)
array_call_result_1679 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), array_1672, *[list_call_result_1677], **kwargs_1678)

# Obtaining the member 'astype' of a type (line 13)
astype_1680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), array_call_result_1679, 'astype')
# Calling astype(args, kwargs) (line 13)
astype_call_result_1683 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), astype_1680, *[int_1681], **kwargs_1682)

# Assigning a type to the variable 'Zs' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'Zs', astype_call_result_1683)

# Assigning a Call to a Name (line 15):

# Call to astype(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'int' (line 15)
int_1694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 41), 'int', False)
# Processing the call keyword arguments (line 15)
kwargs_1695 = {}

# Call to zeros(...): (line 15)
# Processing the call arguments (line 15)

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_1686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)
# Adding element type (line 15)

# Call to len(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'shape' (line 15)
shape_1688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'shape', False)
# Processing the call keyword arguments (line 15)
kwargs_1689 = {}
# Getting the type of 'len' (line 15)
len_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'len', False)
# Calling len(args, kwargs) (line 15)
len_call_result_1690 = invoke(stypy.reporting.localization.Localization(__file__, 15, 20), len_1687, *[shape_1688], **kwargs_1689)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), tuple_1686, len_call_result_1690)

# Processing the call keyword arguments (line 15)
kwargs_1691 = {}
# Getting the type of 'np' (line 15)
np_1684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'np', False)
# Obtaining the member 'zeros' of a type (line 15)
zeros_1685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 10), np_1684, 'zeros')
# Calling zeros(args, kwargs) (line 15)
zeros_call_result_1692 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), zeros_1685, *[tuple_1686], **kwargs_1691)

# Obtaining the member 'astype' of a type (line 15)
astype_1693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 10), zeros_call_result_1692, 'astype')
# Calling astype(args, kwargs) (line 15)
astype_call_result_1696 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), astype_1693, *[int_1694], **kwargs_1695)

# Assigning a type to the variable 'R_start' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'R_start', astype_call_result_1696)

# Assigning a Call to a Name (line 16):

# Call to astype(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'int' (line 16)
int_1706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 38), 'int', False)
# Processing the call keyword arguments (line 16)
kwargs_1707 = {}

# Call to array(...): (line 16)
# Processing the call arguments (line 16)

# Call to list(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'shape' (line 16)
shape_1700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'shape', False)
# Processing the call keyword arguments (line 16)
kwargs_1701 = {}
# Getting the type of 'list' (line 16)
list_1699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'list', False)
# Calling list(args, kwargs) (line 16)
list_call_result_1702 = invoke(stypy.reporting.localization.Localization(__file__, 16, 18), list_1699, *[shape_1700], **kwargs_1701)

# Processing the call keyword arguments (line 16)
kwargs_1703 = {}
# Getting the type of 'np' (line 16)
np_1697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'np', False)
# Obtaining the member 'array' of a type (line 16)
array_1698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 9), np_1697, 'array')
# Calling array(args, kwargs) (line 16)
array_call_result_1704 = invoke(stypy.reporting.localization.Localization(__file__, 16, 9), array_1698, *[list_call_result_1702], **kwargs_1703)

# Obtaining the member 'astype' of a type (line 16)
astype_1705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 9), array_call_result_1704, 'astype')
# Calling astype(args, kwargs) (line 16)
astype_call_result_1708 = invoke(stypy.reporting.localization.Localization(__file__, 16, 9), astype_1705, *[int_1706], **kwargs_1707)

# Assigning a type to the variable 'R_stop' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'R_stop', astype_call_result_1708)

# Assigning a BinOp to a Name (line 17):
# Getting the type of 'P' (line 17)
P_1709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'P')
# Getting the type of 'Rs' (line 17)
Rs_1710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'Rs')
int_1711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'int')
# Applying the binary operator '//' (line 17)
result_floordiv_1712 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 15), '//', Rs_1710, int_1711)

# Applying the binary operator '-' (line 17)
result_sub_1713 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 11), '-', P_1709, result_floordiv_1712)

# Assigning a type to the variable 'Z_start' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'Z_start', result_sub_1713)

# Assigning a BinOp to a Name (line 18):
# Getting the type of 'P' (line 18)
P_1714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'P')
# Getting the type of 'Rs' (line 18)
Rs_1715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'Rs')
int_1716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'int')
# Applying the binary operator '//' (line 18)
result_floordiv_1717 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 14), '//', Rs_1715, int_1716)

# Applying the binary operator '+' (line 18)
result_add_1718 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 10), '+', P_1714, result_floordiv_1717)

# Getting the type of 'Rs' (line 18)
Rs_1719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'Rs')
int_1720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'int')
# Applying the binary operator '%' (line 18)
result_mod_1721 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 25), '%', Rs_1719, int_1720)

# Applying the binary operator '+' (line 18)
result_add_1722 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 9), '+', result_add_1718, result_mod_1721)

# Assigning a type to the variable 'Z_stop' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'Z_stop', result_add_1722)

# Assigning a Call to a Name (line 20):

# Call to tolist(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_1732 = {}
# Getting the type of 'R_start' (line 20)
R_start_1723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'R_start', False)

# Call to minimum(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'Z_start' (line 20)
Z_start_1726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 32), 'Z_start', False)
int_1727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 41), 'int')
# Processing the call keyword arguments (line 20)
kwargs_1728 = {}
# Getting the type of 'np' (line 20)
np_1724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'np', False)
# Obtaining the member 'minimum' of a type (line 20)
minimum_1725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 21), np_1724, 'minimum')
# Calling minimum(args, kwargs) (line 20)
minimum_call_result_1729 = invoke(stypy.reporting.localization.Localization(__file__, 20, 21), minimum_1725, *[Z_start_1726, int_1727], **kwargs_1728)

# Applying the binary operator '-' (line 20)
result_sub_1730 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 11), '-', R_start_1723, minimum_call_result_1729)

# Obtaining the member 'tolist' of a type (line 20)
tolist_1731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 11), result_sub_1730, 'tolist')
# Calling tolist(args, kwargs) (line 20)
tolist_call_result_1733 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), tolist_1731, *[], **kwargs_1732)

# Assigning a type to the variable 'R_start' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'R_start', tolist_call_result_1733)

# Assigning a Call to a Name (line 21):

# Call to tolist(...): (line 21)
# Processing the call keyword arguments (line 21)
kwargs_1741 = {}

# Call to maximum(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'Z_start' (line 21)
Z_start_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'Z_start', False)
int_1737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 31), 'int')
# Processing the call keyword arguments (line 21)
kwargs_1738 = {}
# Getting the type of 'np' (line 21)
np_1734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'np', False)
# Obtaining the member 'maximum' of a type (line 21)
maximum_1735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 11), np_1734, 'maximum')
# Calling maximum(args, kwargs) (line 21)
maximum_call_result_1739 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), maximum_1735, *[Z_start_1736, int_1737], **kwargs_1738)

# Obtaining the member 'tolist' of a type (line 21)
tolist_1740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 11), maximum_call_result_1739, 'tolist')
# Calling tolist(args, kwargs) (line 21)
tolist_call_result_1742 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), tolist_1740, *[], **kwargs_1741)

# Assigning a type to the variable 'Z_start' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'Z_start', tolist_call_result_1742)

# Assigning a Call to a Name (line 22):

# Call to tolist(...): (line 22)
# Processing the call keyword arguments (line 22)
kwargs_1759 = {}

# Call to maximum(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'R_start' (line 22)
R_start_1745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'R_start', False)
# Getting the type of 'R_stop' (line 22)
R_stop_1746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'R_stop', False)

# Call to maximum(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'Z_stop' (line 22)
Z_stop_1749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 50), 'Z_stop', False)
# Getting the type of 'Zs' (line 22)
Zs_1750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 59), 'Zs', False)
# Applying the binary operator '-' (line 22)
result_sub_1751 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 50), '-', Z_stop_1749, Zs_1750)

int_1752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 63), 'int')
# Processing the call keyword arguments (line 22)
kwargs_1753 = {}
# Getting the type of 'np' (line 22)
np_1747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'np', False)
# Obtaining the member 'maximum' of a type (line 22)
maximum_1748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 39), np_1747, 'maximum')
# Calling maximum(args, kwargs) (line 22)
maximum_call_result_1754 = invoke(stypy.reporting.localization.Localization(__file__, 22, 39), maximum_1748, *[result_sub_1751, int_1752], **kwargs_1753)

# Applying the binary operator '-' (line 22)
result_sub_1755 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 30), '-', R_stop_1746, maximum_call_result_1754)

# Processing the call keyword arguments (line 22)
kwargs_1756 = {}
# Getting the type of 'np' (line 22)
np_1743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'np', False)
# Obtaining the member 'maximum' of a type (line 22)
maximum_1744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 9), np_1743, 'maximum')
# Calling maximum(args, kwargs) (line 22)
maximum_call_result_1757 = invoke(stypy.reporting.localization.Localization(__file__, 22, 9), maximum_1744, *[R_start_1745, result_sub_1755], **kwargs_1756)

# Obtaining the member 'tolist' of a type (line 22)
tolist_1758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 9), maximum_call_result_1757, 'tolist')
# Calling tolist(args, kwargs) (line 22)
tolist_call_result_1760 = invoke(stypy.reporting.localization.Localization(__file__, 22, 9), tolist_1758, *[], **kwargs_1759)

# Assigning a type to the variable 'R_stop' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'R_stop', tolist_call_result_1760)

# Assigning a Call to a Name (line 23):

# Call to tolist(...): (line 23)
# Processing the call keyword arguments (line 23)
kwargs_1768 = {}

# Call to minimum(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'Z_stop' (line 23)
Z_stop_1763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'Z_stop', False)
# Getting the type of 'Zs' (line 23)
Zs_1764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 29), 'Zs', False)
# Processing the call keyword arguments (line 23)
kwargs_1765 = {}
# Getting the type of 'np' (line 23)
np_1761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'np', False)
# Obtaining the member 'minimum' of a type (line 23)
minimum_1762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 10), np_1761, 'minimum')
# Calling minimum(args, kwargs) (line 23)
minimum_call_result_1766 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), minimum_1762, *[Z_stop_1763, Zs_1764], **kwargs_1765)

# Obtaining the member 'tolist' of a type (line 23)
tolist_1767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 10), minimum_call_result_1766, 'tolist')
# Calling tolist(args, kwargs) (line 23)
tolist_call_result_1769 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), tolist_1767, *[], **kwargs_1768)

# Assigning a type to the variable 'Z_stop' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'Z_stop', tolist_call_result_1769)

# Assigning a ListComp to a Name (line 25):
# Calculating list comprehension
# Calculating comprehension expression

# Call to zip(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'R_start' (line 25)
R_start_1776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 47), 'R_start', False)
# Getting the type of 'R_stop' (line 25)
R_stop_1777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 56), 'R_stop', False)
# Processing the call keyword arguments (line 25)
kwargs_1778 = {}
# Getting the type of 'zip' (line 25)
zip_1775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 43), 'zip', False)
# Calling zip(args, kwargs) (line 25)
zip_call_result_1779 = invoke(stypy.reporting.localization.Localization(__file__, 25, 43), zip_1775, *[R_start_1776, R_stop_1777], **kwargs_1778)

comprehension_1780 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), zip_call_result_1779)
# Assigning a type to the variable 'start' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 5), 'start', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), comprehension_1780))
# Assigning a type to the variable 'stop' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 5), 'stop', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), comprehension_1780))

# Call to slice(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'start' (line 25)
start_1771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'start', False)
# Getting the type of 'stop' (line 25)
stop_1772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'stop', False)
# Processing the call keyword arguments (line 25)
kwargs_1773 = {}
# Getting the type of 'slice' (line 25)
slice_1770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 5), 'slice', False)
# Calling slice(args, kwargs) (line 25)
slice_call_result_1774 = invoke(stypy.reporting.localization.Localization(__file__, 25, 5), slice_1770, *[start_1771, stop_1772], **kwargs_1773)

list_1781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), list_1781, slice_call_result_1774)
# Assigning a type to the variable 'r' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'r', list_1781)

# Assigning a ListComp to a Name (line 26):
# Calculating list comprehension
# Calculating comprehension expression

# Call to zip(...): (line 26)
# Processing the call arguments (line 26)
# Getting the type of 'Z_start' (line 26)
Z_start_1788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 47), 'Z_start', False)
# Getting the type of 'Z_stop' (line 26)
Z_stop_1789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 56), 'Z_stop', False)
# Processing the call keyword arguments (line 26)
kwargs_1790 = {}
# Getting the type of 'zip' (line 26)
zip_1787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 43), 'zip', False)
# Calling zip(args, kwargs) (line 26)
zip_call_result_1791 = invoke(stypy.reporting.localization.Localization(__file__, 26, 43), zip_1787, *[Z_start_1788, Z_stop_1789], **kwargs_1790)

comprehension_1792 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), zip_call_result_1791)
# Assigning a type to the variable 'start' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'start', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), comprehension_1792))
# Assigning a type to the variable 'stop' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'stop', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), comprehension_1792))

# Call to slice(...): (line 26)
# Processing the call arguments (line 26)
# Getting the type of 'start' (line 26)
start_1783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'start', False)
# Getting the type of 'stop' (line 26)
stop_1784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'stop', False)
# Processing the call keyword arguments (line 26)
kwargs_1785 = {}
# Getting the type of 'slice' (line 26)
slice_1782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'slice', False)
# Calling slice(args, kwargs) (line 26)
slice_call_result_1786 = invoke(stypy.reporting.localization.Localization(__file__, 26, 5), slice_1782, *[start_1783, stop_1784], **kwargs_1785)

list_1793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), list_1793, slice_call_result_1786)
# Assigning a type to the variable 'z' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'z', list_1793)

# Assigning a Subscript to a Subscript (line 27):

# Obtaining the type of the subscript
# Getting the type of 'z' (line 27)
z_1794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 9), 'z')
# Getting the type of 'Z' (line 27)
Z_1795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'Z')
# Obtaining the member '__getitem__' of a type (line 27)
getitem___1796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 7), Z_1795, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 27)
subscript_call_result_1797 = invoke(stypy.reporting.localization.Localization(__file__, 27, 7), getitem___1796, z_1794)

# Getting the type of 'R' (line 27)
R_1798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'R')
# Getting the type of 'r' (line 27)
r_1799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 2), 'r')
# Storing an element on a container (line 27)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 0), R_1798, (r_1799, subscript_call_result_1797))

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
