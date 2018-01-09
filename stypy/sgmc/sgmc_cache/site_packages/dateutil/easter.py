
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 -*-
2: '''
3: This module offers a generic easter computing method for any given year, using
4: Western, Orthodox or Julian algorithms.
5: '''
6: 
7: import datetime
8: 
9: __all__ = ["easter", "EASTER_JULIAN", "EASTER_ORTHODOX", "EASTER_WESTERN"]
10: 
11: EASTER_JULIAN = 1
12: EASTER_ORTHODOX = 2
13: EASTER_WESTERN = 3
14: 
15: 
16: def easter(year, method=EASTER_WESTERN):
17:     '''
18:     This method was ported from the work done by GM Arts,
19:     on top of the algorithm by Claus Tondering, which was
20:     based in part on the algorithm of Ouding (1940), as
21:     quoted in "Explanatory Supplement to the Astronomical
22:     Almanac", P.  Kenneth Seidelmann, editor.
23: 
24:     This algorithm implements three different easter
25:     calculation methods:
26: 
27:     1 - Original calculation in Julian calendar, valid in
28:         dates after 326 AD
29:     2 - Original method, with date converted to Gregorian
30:         calendar, valid in years 1583 to 4099
31:     3 - Revised method, in Gregorian calendar, valid in
32:         years 1583 to 4099 as well
33: 
34:     These methods are represented by the constants:
35: 
36:     * ``EASTER_JULIAN   = 1``
37:     * ``EASTER_ORTHODOX = 2``
38:     * ``EASTER_WESTERN  = 3``
39: 
40:     The default method is method 3.
41: 
42:     More about the algorithm may be found at:
43: 
44:     http://users.chariot.net.au/~gmarts/eastalg.htm
45: 
46:     and
47: 
48:     http://www.tondering.dk/claus/calendar.html
49: 
50:     '''
51: 
52:     if not (1 <= method <= 3):
53:         raise ValueError("invalid method")
54: 
55:     # g - Golden year - 1
56:     # c - Century
57:     # h - (23 - Epact) mod 30
58:     # i - Number of days from March 21 to Paschal Full Moon
59:     # j - Weekday for PFM (0=Sunday, etc)
60:     # p - Number of days from March 21 to Sunday on or before PFM
61:     #     (-6 to 28 methods 1 & 3, to 56 for method 2)
62:     # e - Extra days to add for method 2 (converting Julian
63:     #     date to Gregorian date)
64: 
65:     y = year
66:     g = y % 19
67:     e = 0
68:     if method < 3:
69:         # Old method
70:         i = (19*g + 15) % 30
71:         j = (y + y//4 + i) % 7
72:         if method == 2:
73:             # Extra dates to convert Julian to Gregorian date
74:             e = 10
75:             if y > 1600:
76:                 e = e + y//100 - 16 - (y//100 - 16)//4
77:     else:
78:         # New method
79:         c = y//100
80:         h = (c - c//4 - (8*c + 13)//25 + 19*g + 15) % 30
81:         i = h - (h//28)*(1 - (h//28)*(29//(h + 1))*((21 - g)//11))
82:         j = (y + y//4 + i + 2 - c + c//4) % 7
83: 
84:     # p can be from -6 to 56 corresponding to dates 22 March to 23 May
85:     # (later dates apply to method 2, although 23 May never actually occurs)
86:     p = i - j + e
87:     d = 1 + (p + 27 + (p + 6)//40) % 31
88:     m = 3 + (p + 26)//30
89:     return datetime.date(int(y), int(m), int(d))
90: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_308680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', '\nThis module offers a generic easter computing method for any given year, using\nWestern, Orthodox or Julian algorithms.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import datetime' statement (line 7)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'datetime', datetime, module_type_store)


# Assigning a List to a Name (line 9):
__all__ = ['easter', 'EASTER_JULIAN', 'EASTER_ORTHODOX', 'EASTER_WESTERN']
module_type_store.set_exportable_members(['easter', 'EASTER_JULIAN', 'EASTER_ORTHODOX', 'EASTER_WESTERN'])

# Obtaining an instance of the builtin type 'list' (line 9)
list_308681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_308682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'easter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_308681, str_308682)
# Adding element type (line 9)
str_308683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 21), 'str', 'EASTER_JULIAN')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_308681, str_308683)
# Adding element type (line 9)
str_308684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 38), 'str', 'EASTER_ORTHODOX')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_308681, str_308684)
# Adding element type (line 9)
str_308685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 57), 'str', 'EASTER_WESTERN')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_308681, str_308685)

# Assigning a type to the variable '__all__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__all__', list_308681)

# Assigning a Num to a Name (line 11):
int_308686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'int')
# Assigning a type to the variable 'EASTER_JULIAN' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'EASTER_JULIAN', int_308686)

# Assigning a Num to a Name (line 12):
int_308687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
# Assigning a type to the variable 'EASTER_ORTHODOX' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'EASTER_ORTHODOX', int_308687)

# Assigning a Num to a Name (line 13):
int_308688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'int')
# Assigning a type to the variable 'EASTER_WESTERN' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'EASTER_WESTERN', int_308688)

@norecursion
def easter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'EASTER_WESTERN' (line 16)
    EASTER_WESTERN_308689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'EASTER_WESTERN')
    defaults = [EASTER_WESTERN_308689]
    # Create a new context for function 'easter'
    module_type_store = module_type_store.open_function_context('easter', 16, 0, False)
    
    # Passed parameters checking function
    easter.stypy_localization = localization
    easter.stypy_type_of_self = None
    easter.stypy_type_store = module_type_store
    easter.stypy_function_name = 'easter'
    easter.stypy_param_names_list = ['year', 'method']
    easter.stypy_varargs_param_name = None
    easter.stypy_kwargs_param_name = None
    easter.stypy_call_defaults = defaults
    easter.stypy_call_varargs = varargs
    easter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'easter', ['year', 'method'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'easter', localization, ['year', 'method'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'easter(...)' code ##################

    str_308690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', '\n    This method was ported from the work done by GM Arts,\n    on top of the algorithm by Claus Tondering, which was\n    based in part on the algorithm of Ouding (1940), as\n    quoted in "Explanatory Supplement to the Astronomical\n    Almanac", P.  Kenneth Seidelmann, editor.\n\n    This algorithm implements three different easter\n    calculation methods:\n\n    1 - Original calculation in Julian calendar, valid in\n        dates after 326 AD\n    2 - Original method, with date converted to Gregorian\n        calendar, valid in years 1583 to 4099\n    3 - Revised method, in Gregorian calendar, valid in\n        years 1583 to 4099 as well\n\n    These methods are represented by the constants:\n\n    * ``EASTER_JULIAN   = 1``\n    * ``EASTER_ORTHODOX = 2``\n    * ``EASTER_WESTERN  = 3``\n\n    The default method is method 3.\n\n    More about the algorithm may be found at:\n\n    http://users.chariot.net.au/~gmarts/eastalg.htm\n\n    and\n\n    http://www.tondering.dk/claus/calendar.html\n\n    ')
    
    
    
    int_308691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'int')
    # Getting the type of 'method' (line 52)
    method_308692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'method')
    # Applying the binary operator '<=' (line 52)
    result_le_308693 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 12), '<=', int_308691, method_308692)
    int_308694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'int')
    # Applying the binary operator '<=' (line 52)
    result_le_308695 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 12), '<=', method_308692, int_308694)
    # Applying the binary operator '&' (line 52)
    result_and__308696 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 12), '&', result_le_308693, result_le_308695)
    
    # Applying the 'not' unary operator (line 52)
    result_not__308697 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 7), 'not', result_and__308696)
    
    # Testing the type of an if condition (line 52)
    if_condition_308698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 4), result_not__308697)
    # Assigning a type to the variable 'if_condition_308698' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'if_condition_308698', if_condition_308698)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 53)
    # Processing the call arguments (line 53)
    str_308700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'str', 'invalid method')
    # Processing the call keyword arguments (line 53)
    kwargs_308701 = {}
    # Getting the type of 'ValueError' (line 53)
    ValueError_308699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 53)
    ValueError_call_result_308702 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), ValueError_308699, *[str_308700], **kwargs_308701)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 53, 8), ValueError_call_result_308702, 'raise parameter', BaseException)
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'year' (line 65)
    year_308703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'year')
    # Assigning a type to the variable 'y' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'y', year_308703)
    
    # Assigning a BinOp to a Name (line 66):
    # Getting the type of 'y' (line 66)
    y_308704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'y')
    int_308705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'int')
    # Applying the binary operator '%' (line 66)
    result_mod_308706 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 8), '%', y_308704, int_308705)
    
    # Assigning a type to the variable 'g' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'g', result_mod_308706)
    
    # Assigning a Num to a Name (line 67):
    int_308707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 8), 'int')
    # Assigning a type to the variable 'e' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'e', int_308707)
    
    
    # Getting the type of 'method' (line 68)
    method_308708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 7), 'method')
    int_308709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'int')
    # Applying the binary operator '<' (line 68)
    result_lt_308710 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 7), '<', method_308708, int_308709)
    
    # Testing the type of an if condition (line 68)
    if_condition_308711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), result_lt_308710)
    # Assigning a type to the variable 'if_condition_308711' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'if_condition_308711', if_condition_308711)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 70):
    int_308712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'int')
    # Getting the type of 'g' (line 70)
    g_308713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'g')
    # Applying the binary operator '*' (line 70)
    result_mul_308714 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 13), '*', int_308712, g_308713)
    
    int_308715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'int')
    # Applying the binary operator '+' (line 70)
    result_add_308716 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 13), '+', result_mul_308714, int_308715)
    
    int_308717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 26), 'int')
    # Applying the binary operator '%' (line 70)
    result_mod_308718 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 12), '%', result_add_308716, int_308717)
    
    # Assigning a type to the variable 'i' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'i', result_mod_308718)
    
    # Assigning a BinOp to a Name (line 71):
    # Getting the type of 'y' (line 71)
    y_308719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), 'y')
    # Getting the type of 'y' (line 71)
    y_308720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'y')
    int_308721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'int')
    # Applying the binary operator '//' (line 71)
    result_floordiv_308722 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 17), '//', y_308720, int_308721)
    
    # Applying the binary operator '+' (line 71)
    result_add_308723 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 13), '+', y_308719, result_floordiv_308722)
    
    # Getting the type of 'i' (line 71)
    i_308724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'i')
    # Applying the binary operator '+' (line 71)
    result_add_308725 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 22), '+', result_add_308723, i_308724)
    
    int_308726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'int')
    # Applying the binary operator '%' (line 71)
    result_mod_308727 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 12), '%', result_add_308725, int_308726)
    
    # Assigning a type to the variable 'j' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'j', result_mod_308727)
    
    
    # Getting the type of 'method' (line 72)
    method_308728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'method')
    int_308729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 21), 'int')
    # Applying the binary operator '==' (line 72)
    result_eq_308730 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 11), '==', method_308728, int_308729)
    
    # Testing the type of an if condition (line 72)
    if_condition_308731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 8), result_eq_308730)
    # Assigning a type to the variable 'if_condition_308731' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'if_condition_308731', if_condition_308731)
    # SSA begins for if statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 74):
    int_308732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'int')
    # Assigning a type to the variable 'e' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'e', int_308732)
    
    
    # Getting the type of 'y' (line 75)
    y_308733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'y')
    int_308734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'int')
    # Applying the binary operator '>' (line 75)
    result_gt_308735 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 15), '>', y_308733, int_308734)
    
    # Testing the type of an if condition (line 75)
    if_condition_308736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), result_gt_308735)
    # Assigning a type to the variable 'if_condition_308736' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_308736', if_condition_308736)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 76):
    # Getting the type of 'e' (line 76)
    e_308737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'e')
    # Getting the type of 'y' (line 76)
    y_308738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'y')
    int_308739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 27), 'int')
    # Applying the binary operator '//' (line 76)
    result_floordiv_308740 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 24), '//', y_308738, int_308739)
    
    # Applying the binary operator '+' (line 76)
    result_add_308741 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 20), '+', e_308737, result_floordiv_308740)
    
    int_308742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
    # Applying the binary operator '-' (line 76)
    result_sub_308743 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 31), '-', result_add_308741, int_308742)
    
    # Getting the type of 'y' (line 76)
    y_308744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 39), 'y')
    int_308745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 42), 'int')
    # Applying the binary operator '//' (line 76)
    result_floordiv_308746 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 39), '//', y_308744, int_308745)
    
    int_308747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 48), 'int')
    # Applying the binary operator '-' (line 76)
    result_sub_308748 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 39), '-', result_floordiv_308746, int_308747)
    
    int_308749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 53), 'int')
    # Applying the binary operator '//' (line 76)
    result_floordiv_308750 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 38), '//', result_sub_308748, int_308749)
    
    # Applying the binary operator '-' (line 76)
    result_sub_308751 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 36), '-', result_sub_308743, result_floordiv_308750)
    
    # Assigning a type to the variable 'e' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'e', result_sub_308751)
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 68)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 79):
    # Getting the type of 'y' (line 79)
    y_308752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'y')
    int_308753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 15), 'int')
    # Applying the binary operator '//' (line 79)
    result_floordiv_308754 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 12), '//', y_308752, int_308753)
    
    # Assigning a type to the variable 'c' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'c', result_floordiv_308754)
    
    # Assigning a BinOp to a Name (line 80):
    # Getting the type of 'c' (line 80)
    c_308755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'c')
    # Getting the type of 'c' (line 80)
    c_308756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'c')
    int_308757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'int')
    # Applying the binary operator '//' (line 80)
    result_floordiv_308758 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 17), '//', c_308756, int_308757)
    
    # Applying the binary operator '-' (line 80)
    result_sub_308759 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '-', c_308755, result_floordiv_308758)
    
    int_308760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 25), 'int')
    # Getting the type of 'c' (line 80)
    c_308761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'c')
    # Applying the binary operator '*' (line 80)
    result_mul_308762 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 25), '*', int_308760, c_308761)
    
    int_308763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 31), 'int')
    # Applying the binary operator '+' (line 80)
    result_add_308764 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 25), '+', result_mul_308762, int_308763)
    
    int_308765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 36), 'int')
    # Applying the binary operator '//' (line 80)
    result_floordiv_308766 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 24), '//', result_add_308764, int_308765)
    
    # Applying the binary operator '-' (line 80)
    result_sub_308767 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 22), '-', result_sub_308759, result_floordiv_308766)
    
    int_308768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 41), 'int')
    # Getting the type of 'g' (line 80)
    g_308769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'g')
    # Applying the binary operator '*' (line 80)
    result_mul_308770 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 41), '*', int_308768, g_308769)
    
    # Applying the binary operator '+' (line 80)
    result_add_308771 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 39), '+', result_sub_308767, result_mul_308770)
    
    int_308772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 48), 'int')
    # Applying the binary operator '+' (line 80)
    result_add_308773 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 46), '+', result_add_308771, int_308772)
    
    int_308774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 54), 'int')
    # Applying the binary operator '%' (line 80)
    result_mod_308775 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 12), '%', result_add_308773, int_308774)
    
    # Assigning a type to the variable 'h' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'h', result_mod_308775)
    
    # Assigning a BinOp to a Name (line 81):
    # Getting the type of 'h' (line 81)
    h_308776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'h')
    # Getting the type of 'h' (line 81)
    h_308777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'h')
    int_308778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'int')
    # Applying the binary operator '//' (line 81)
    result_floordiv_308779 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 17), '//', h_308777, int_308778)
    
    int_308780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'int')
    # Getting the type of 'h' (line 81)
    h_308781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'h')
    int_308782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 33), 'int')
    # Applying the binary operator '//' (line 81)
    result_floordiv_308783 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 30), '//', h_308781, int_308782)
    
    int_308784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 38), 'int')
    # Getting the type of 'h' (line 81)
    h_308785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 43), 'h')
    int_308786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 47), 'int')
    # Applying the binary operator '+' (line 81)
    result_add_308787 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 43), '+', h_308785, int_308786)
    
    # Applying the binary operator '//' (line 81)
    result_floordiv_308788 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 38), '//', int_308784, result_add_308787)
    
    # Applying the binary operator '*' (line 81)
    result_mul_308789 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 29), '*', result_floordiv_308783, result_floordiv_308788)
    
    int_308790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 53), 'int')
    # Getting the type of 'g' (line 81)
    g_308791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 58), 'g')
    # Applying the binary operator '-' (line 81)
    result_sub_308792 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 53), '-', int_308790, g_308791)
    
    int_308793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 62), 'int')
    # Applying the binary operator '//' (line 81)
    result_floordiv_308794 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 52), '//', result_sub_308792, int_308793)
    
    # Applying the binary operator '*' (line 81)
    result_mul_308795 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 50), '*', result_mul_308789, result_floordiv_308794)
    
    # Applying the binary operator '-' (line 81)
    result_sub_308796 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 25), '-', int_308780, result_mul_308795)
    
    # Applying the binary operator '*' (line 81)
    result_mul_308797 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 16), '*', result_floordiv_308779, result_sub_308796)
    
    # Applying the binary operator '-' (line 81)
    result_sub_308798 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 12), '-', h_308776, result_mul_308797)
    
    # Assigning a type to the variable 'i' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'i', result_sub_308798)
    
    # Assigning a BinOp to a Name (line 82):
    # Getting the type of 'y' (line 82)
    y_308799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'y')
    # Getting the type of 'y' (line 82)
    y_308800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'y')
    int_308801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 20), 'int')
    # Applying the binary operator '//' (line 82)
    result_floordiv_308802 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 17), '//', y_308800, int_308801)
    
    # Applying the binary operator '+' (line 82)
    result_add_308803 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 13), '+', y_308799, result_floordiv_308802)
    
    # Getting the type of 'i' (line 82)
    i_308804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'i')
    # Applying the binary operator '+' (line 82)
    result_add_308805 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 22), '+', result_add_308803, i_308804)
    
    int_308806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 28), 'int')
    # Applying the binary operator '+' (line 82)
    result_add_308807 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 26), '+', result_add_308805, int_308806)
    
    # Getting the type of 'c' (line 82)
    c_308808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 32), 'c')
    # Applying the binary operator '-' (line 82)
    result_sub_308809 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 30), '-', result_add_308807, c_308808)
    
    # Getting the type of 'c' (line 82)
    c_308810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 36), 'c')
    int_308811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 39), 'int')
    # Applying the binary operator '//' (line 82)
    result_floordiv_308812 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 36), '//', c_308810, int_308811)
    
    # Applying the binary operator '+' (line 82)
    result_add_308813 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 34), '+', result_sub_308809, result_floordiv_308812)
    
    int_308814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 44), 'int')
    # Applying the binary operator '%' (line 82)
    result_mod_308815 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 12), '%', result_add_308813, int_308814)
    
    # Assigning a type to the variable 'j' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'j', result_mod_308815)
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 86):
    # Getting the type of 'i' (line 86)
    i_308816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'i')
    # Getting the type of 'j' (line 86)
    j_308817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'j')
    # Applying the binary operator '-' (line 86)
    result_sub_308818 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 8), '-', i_308816, j_308817)
    
    # Getting the type of 'e' (line 86)
    e_308819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'e')
    # Applying the binary operator '+' (line 86)
    result_add_308820 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 14), '+', result_sub_308818, e_308819)
    
    # Assigning a type to the variable 'p' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'p', result_add_308820)
    
    # Assigning a BinOp to a Name (line 87):
    int_308821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'int')
    # Getting the type of 'p' (line 87)
    p_308822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'p')
    int_308823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 17), 'int')
    # Applying the binary operator '+' (line 87)
    result_add_308824 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), '+', p_308822, int_308823)
    
    # Getting the type of 'p' (line 87)
    p_308825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'p')
    int_308826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'int')
    # Applying the binary operator '+' (line 87)
    result_add_308827 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 23), '+', p_308825, int_308826)
    
    int_308828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 31), 'int')
    # Applying the binary operator '//' (line 87)
    result_floordiv_308829 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 22), '//', result_add_308827, int_308828)
    
    # Applying the binary operator '+' (line 87)
    result_add_308830 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 20), '+', result_add_308824, result_floordiv_308829)
    
    int_308831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 37), 'int')
    # Applying the binary operator '%' (line 87)
    result_mod_308832 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 12), '%', result_add_308830, int_308831)
    
    # Applying the binary operator '+' (line 87)
    result_add_308833 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 8), '+', int_308821, result_mod_308832)
    
    # Assigning a type to the variable 'd' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'd', result_add_308833)
    
    # Assigning a BinOp to a Name (line 88):
    int_308834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 8), 'int')
    # Getting the type of 'p' (line 88)
    p_308835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'p')
    int_308836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 17), 'int')
    # Applying the binary operator '+' (line 88)
    result_add_308837 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '+', p_308835, int_308836)
    
    int_308838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 22), 'int')
    # Applying the binary operator '//' (line 88)
    result_floordiv_308839 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 12), '//', result_add_308837, int_308838)
    
    # Applying the binary operator '+' (line 88)
    result_add_308840 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 8), '+', int_308834, result_floordiv_308839)
    
    # Assigning a type to the variable 'm' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'm', result_add_308840)
    
    # Call to date(...): (line 89)
    # Processing the call arguments (line 89)
    
    # Call to int(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'y' (line 89)
    y_308844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'y', False)
    # Processing the call keyword arguments (line 89)
    kwargs_308845 = {}
    # Getting the type of 'int' (line 89)
    int_308843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'int', False)
    # Calling int(args, kwargs) (line 89)
    int_call_result_308846 = invoke(stypy.reporting.localization.Localization(__file__, 89, 25), int_308843, *[y_308844], **kwargs_308845)
    
    
    # Call to int(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'm' (line 89)
    m_308848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 37), 'm', False)
    # Processing the call keyword arguments (line 89)
    kwargs_308849 = {}
    # Getting the type of 'int' (line 89)
    int_308847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'int', False)
    # Calling int(args, kwargs) (line 89)
    int_call_result_308850 = invoke(stypy.reporting.localization.Localization(__file__, 89, 33), int_308847, *[m_308848], **kwargs_308849)
    
    
    # Call to int(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'd' (line 89)
    d_308852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 45), 'd', False)
    # Processing the call keyword arguments (line 89)
    kwargs_308853 = {}
    # Getting the type of 'int' (line 89)
    int_308851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'int', False)
    # Calling int(args, kwargs) (line 89)
    int_call_result_308854 = invoke(stypy.reporting.localization.Localization(__file__, 89, 41), int_308851, *[d_308852], **kwargs_308853)
    
    # Processing the call keyword arguments (line 89)
    kwargs_308855 = {}
    # Getting the type of 'datetime' (line 89)
    datetime_308841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'datetime', False)
    # Obtaining the member 'date' of a type (line 89)
    date_308842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 11), datetime_308841, 'date')
    # Calling date(args, kwargs) (line 89)
    date_call_result_308856 = invoke(stypy.reporting.localization.Localization(__file__, 89, 11), date_308842, *[int_call_result_308846, int_call_result_308850, int_call_result_308854], **kwargs_308855)
    
    # Assigning a type to the variable 'stypy_return_type' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type', date_call_result_308856)
    
    # ################# End of 'easter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'easter' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_308857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_308857)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'easter'
    return stypy_return_type_308857

# Assigning a type to the variable 'easter' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'easter', easter)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
